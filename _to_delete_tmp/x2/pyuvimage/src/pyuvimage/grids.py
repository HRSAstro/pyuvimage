"""Image-grid geometry: Nyquist sampling, field of view, mesh shape, mask.

Adapted from the pyuvimage_dev prototype's ``src/utils/grids.py`` (returns an
immutable geometry object instead of mutating a settings dict).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0


def nyquist_pixel_scale_arcsec(max_baseline_wavelengths: float) -> float:
    """Nyquist pixel scale = half the finest fringe: 0.5 / b_max [rad]."""
    if max_baseline_wavelengths <= 0:
        raise ValueError("max baseline must be positive")
    return 0.5 / max_baseline_wavelengths * RAD_TO_ARCSEC


@dataclass(frozen=True)
class ImageGeometry:
    """Resolved geometry of the reconstruction."""

    fov_arcsec: float          # full width of the (square) field
    pixel_scale: float         # image / dirty-image grid pixel scale [arcsec]
    shape_native: tuple[int, int]  # dirty-image grid shape
    mesh_shape: tuple[int, int]    # source-reconstruction mesh shape
    mask_radius: float         # circular mask radius [arcsec]
    nyquist_pixel_scale: float

    @property
    def mesh_pixel_scale(self) -> float:
        return self.fov_arcsec / self.mesh_shape[0]


def resolve_geometry(
    fov_arcsec: float,
    max_baseline_wavelengths: float,
    pixel_scale: float | str = "auto",
    mesh_shape: tuple[int, int] | None = None,
    oversample: float = 2.0,
) -> ImageGeometry:
    """Derive all grid quantities from the field of view and the uv coverage.

    Parameters
    ----------
    fov_arcsec
        Full width of the reconstructed field (the one required user input).
    pixel_scale
        "auto"/"nyquist" -> Nyquist (0.5 / b_max); or an explicit arcsec value
        for the *source mesh*.
    mesh_shape
        Explicit override of the mesh shape.
    oversample
        Integer factor by which the product grid is finer than the model mesh.
        Every product is written on that one product grid, so they all share a
        pixel scale; the model is block-replicated onto it and therefore looks
        piecewise-constant at the mesh scale, which is honest -- that is its
        real resolution.

        This must be > 1 for the residual map to be diagnostic. With mesh ==
        product grid the normal equations give A^T W r = H s, so the residual
        dirty image collapses to the prior's pull rather than the data misfit,
        and goes to zero exactly where the prior is weak.
    """
    nyq = nyquist_pixel_scale_arcsec(max_baseline_wavelengths)

    if isinstance(pixel_scale, str):
        if pixel_scale in ("auto", "nyquist"):
            # Nyquist sampling of the longest baseline: the information limit
            # of the data. Products are written on a grid `oversample` times
            # finer, which keeps the residual map diagnostic (see above) and
            # gives ~4 pixels across the beam for viewing and beam fitting.
            mesh_scale = nyq
        elif pixel_scale in ("fine", "nyquist/2"):
            mesh_scale = nyq / 2.0
        else:
            raise ValueError(f"unknown pixel_scale option {pixel_scale!r}")
    else:
        mesh_scale = float(pixel_scale)
        if mesh_scale > nyq * 1.0001:
            import warnings

            warnings.warn(
                f"requested pixel scale {mesh_scale:.4g}\" is coarser than the "
                f"Nyquist scale {nyq:.4g}\"; the reconstruction cannot use the "
                "longest baselines fully.",
                stacklevel=2,
            )

    if mesh_shape is None:
        n = int(np.ceil(fov_arcsec / mesh_scale))
        n = max(8, n + (n % 2))  # even, at least 8
        mesh_shape = (n, n)
    else:
        mesh_shape = (int(mesh_shape[0]), int(mesh_shape[1]))
    # Recompute the actual mesh scale so mesh exactly tiles the FOV.
    mesh_scale = fov_arcsec / mesh_shape[0]

    k = max(1, int(round(oversample)))
    n_img = mesh_shape[0] * k  # exact integer multiple: lossless block resampling
    img_scale = fov_arcsec / n_img

    return ImageGeometry(
        fov_arcsec=float(fov_arcsec),
        pixel_scale=img_scale,
        shape_native=(n_img, n_img),
        mesh_shape=mesh_shape,
        mask_radius=float(fov_arcsec) / 2.0,
        nyquist_pixel_scale=nyq,
    )

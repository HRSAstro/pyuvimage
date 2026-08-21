"""Grid helpers: Nyquist sampling and FOV-driven mesh / mask construction."""

from __future__ import annotations

import numpy as np

# Radians → arcsec (IAU exact: 180/π * 3600).
_RADIANS_TO_ARCSEC = 180.0 / np.pi * 3600.0


def max_uv_distance_wavelengths(uv_wavelengths):
    """Longest projected baseline in wavelengths (``√(u²+v²)``)."""
    uv = np.asarray(uv_wavelengths, dtype=float)
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError(
            f"Expected uv_wavelengths with last axis length 2; got shape {uv.shape}"
        )
    return float(np.nanmax(np.hypot(uv[..., 0], uv[..., 1])))


def nyquist_pixel_scale_arcsec_from_uv(uv_wavelengths):
    """
    Pixel scale that Nyquist-samples the longest baseline.

    ``Δθ = 0.5 * λ / b_max`` (radians), with ``b_max / λ = max√(u²+v²)``.
    Returned in arcsec.
    """
    uv_max = max_uv_distance_wavelengths(uv_wavelengths)
    if not np.isfinite(uv_max) or uv_max <= 0.0:
        raise ValueError(
            f"Cannot derive Nyquist pixel scale: max UV distance is {uv_max}"
        )
    return (0.5 / uv_max) * _RADIANS_TO_ARCSEC


def resolve_source_pixel_scale(settings, uv_wavelengths):
    """
    Resolve ``source_pixel_scale`` to a float (arcsec).

    Default / ``\"nyquist\"`` / ``\"auto\"`` → ``0.5 λ/b_max`` from UV.
    """
    explicit = settings.get("source_pixel_scale", "nyquist")
    if isinstance(explicit, (int, float)):
        scale = float(explicit)
        if scale <= 0.0:
            raise ValueError(f"source_pixel_scale must be positive; got {scale}")
        return scale
    if isinstance(explicit, str) and explicit.lower() in {"nyquist", "auto"}:
        return nyquist_pixel_scale_arcsec_from_uv(uv_wavelengths)
    raise ValueError(
        f"Unsupported source_pixel_scale: {explicit!r}. "
        "Use a positive float (arcsec) or 'nyquist' / 'auto'."
    )


def derive_mesh_shape(fov, source_pixel_scale, mesh_shape=None):
    """
    Return ``(ny, nx)`` for a rectangular source mesh.

    If ``mesh_shape`` is given, use it. Otherwise
    ``n ≈ round(fov / source_pixel_scale)``, at least 2.
    """
    if mesh_shape is not None:
        shape = tuple(int(v) for v in mesh_shape)
        if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
            raise ValueError(f"mesh_shape must be [ny, nx] with ny,nx >= 2; got {mesh_shape}")
        return shape
    n = int(round(float(fov) / float(source_pixel_scale)))
    n = max(2, n)
    return (n, n)


def resolve_grids(settings, uv_wavelengths):
    """
    Resolve FOV, source pixel scale, mesh shape, and mask geometry.

    Requires ``settings['fov']`` (full field width in arcsec).
    Mutates ``settings`` with concrete numeric values used by the fit.
    Returns a dict of resolved quantities.
    """
    if "fov" not in settings:
        raise KeyError(
            "settings['fov'] is required (full field of view in arcsec)."
        )
    fov = float(settings["fov"])
    if not np.isfinite(fov) or fov <= 0.0:
        raise ValueError(f"fov must be a positive finite number; got {fov!r}")

    source_pixel_scale = resolve_source_pixel_scale(settings, uv_wavelengths)
    nyquist_pixel_scale = nyquist_pixel_scale_arcsec_from_uv(uv_wavelengths)
    mesh_shape = derive_mesh_shape(
        fov=fov,
        source_pixel_scale=source_pixel_scale,
        mesh_shape=settings.get("mesh_shape"),
    )

    # ``mask_n_pixels`` is the pixel count across the science FOV. Optional
    # ``mask_pad_pixels`` extends the circular mask beyond that FOV so edge
    # artefacts can sit outside the science region (Autolens has no dedicated
    # interferometer mask-pad; this is the simple geometric equivalent).
    science_n_pixels = int(settings.get("mask_n_pixels", mesh_shape[0]))
    if science_n_pixels < 2:
        raise ValueError(f"mask_n_pixels must be >= 2; got {science_n_pixels}")
    mask_pad_pixels = int(settings.get("mask_pad_pixels", 0))
    if mask_pad_pixels < 0:
        raise ValueError(f"mask_pad_pixels must be >= 0; got {mask_pad_pixels}")

    # Imaging grid must be at least Nyquist: Δθ <= 0.5 λ/b_max.
    min_science_n = max(2, int(np.ceil(fov / nyquist_pixel_scale)))
    if science_n_pixels < min_science_n:
        import logging

        logging.getLogger(__name__).warning(
            "mask_n_pixels=%d gives pixel scale %.6g\" > Nyquist %.6g\"; "
            "raising science grid to %d pixels across the FOV.",
            science_n_pixels,
            fov / science_n_pixels,
            nyquist_pixel_scale,
            min_science_n,
        )
        science_n_pixels = min_science_n

    mask_pixel_scale = fov / science_n_pixels
    if mask_pixel_scale > nyquist_pixel_scale * (1.0 + 1e-9):
        raise ValueError(
            f"mask_pixel_scale={mask_pixel_scale}\" exceeds Nyquist "
            f"{nyquist_pixel_scale}\" (0.5 λ/b_max). Increase mask_n_pixels."
        )
    mask_n_pixels = science_n_pixels + 2 * mask_pad_pixels
    mask_radius = 0.5 * mask_n_pixels * mask_pixel_scale
    # Full on-sky width of the (padded) mask grid used for dirty/recon maps.
    mask_fov = mask_n_pixels * mask_pixel_scale

    settings["fov"] = fov
    settings["source_pixel_scale"] = float(source_pixel_scale)
    settings["nyquist_pixel_scale"] = float(nyquist_pixel_scale)
    settings["mesh_shape"] = list(mesh_shape)
    settings["mask_n_pixels"] = mask_n_pixels
    settings["science_n_pixels"] = science_n_pixels
    settings["mask_pad_pixels"] = mask_pad_pixels
    settings["mask_pixel_scale"] = float(mask_pixel_scale)
    settings["mask_radius"] = float(mask_radius)
    settings["mask_fov"] = float(mask_fov)

    return {
        "fov": fov,
        "source_pixel_scale": float(source_pixel_scale),
        "nyquist_pixel_scale": float(nyquist_pixel_scale),
        "mesh_shape": mesh_shape,
        "mask_n_pixels": mask_n_pixels,
        "science_n_pixels": science_n_pixels,
        "mask_pad_pixels": mask_pad_pixels,
        "mask_pixel_scale": float(mask_pixel_scale),
        "mask_radius": float(mask_radius),
        "mask_fov": float(mask_fov),
    }


def transformer_class_from_settings(settings):
    """Return ``TransformerDFT`` or ``TransformerNUFFT`` from settings."""
    import autolens as al

    name = str(settings.get("transformer", "dft")).lower()
    if name in {"dft", "transformer_dft"}:
        return al.TransformerDFT
    if name in {"nufft", "transformer_nufft", "auto"}:
        return al.TransformerNUFFT
    raise ValueError(
        f"Unsupported transformer: {name!r}. Choose 'dft' or 'nufft'."
    )

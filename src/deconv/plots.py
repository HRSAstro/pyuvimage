"""Plot dirty image, reconstruction, and residual for a deconv run."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def robust_rms(values):
    """Gaussian-equivalent RMS from the median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad) if mad > 0 else float(np.std(values))


def sky_extent_arcsec(shape, pixel_scale):
    """
    ``imshow`` extent ``[x0, x1, y0, y1]`` for a centred sky grid.

    ``shape`` is ``(ny, nx)``; ``pixel_scale`` is arcsec per pixel.
    """
    ny, nx = (int(shape[0]), int(shape[1]))
    pixel_scale = float(pixel_scale)
    if ny < 1 or nx < 1:
        raise ValueError(f"shape must be positive; got {shape}")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise ValueError(f"pixel_scale must be positive; got {pixel_scale}")
    half_y = 0.5 * ny * pixel_scale
    half_x = 0.5 * nx * pixel_scale
    return [-half_x, half_x, -half_y, half_y]


def require_common_sky_grid(named_images, *, pixel_scale, expected_shape=None):
    """
    Require every image to share one pixel grid.

    Parameters
    ----------
    named_images
        Mapping ``{name: array}`` or sequence of ``(name, array)``.
    pixel_scale
        Arcsec per pixel (required).
    expected_shape
        Optional ``(ny, nx)`` that every array must match (e.g. mask native).

    Returns
    -------
    shape : tuple[int, int]
    extent : list[float]
        Shared ``imshow`` extent in arcsec.
    """
    if isinstance(named_images, dict):
        items = list(named_images.items())
    else:
        items = list(named_images)
    if not items:
        raise ValueError("named_images must contain at least one image")

    shapes = {}
    for name, image in items:
        arr = np.asarray(image)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2-D; got shape {arr.shape}")
        shapes[name] = tuple(int(v) for v in arr.shape)

    unique = set(shapes.values())
    if len(unique) != 1:
        detail = ", ".join(f"{k}={v}" for k, v in shapes.items())
        raise ValueError(
            "Truth / dirty / reconstruction / residual must share one image "
            f"size and pixel scale; got {detail}"
        )
    shape = unique.pop()
    if expected_shape is not None:
        expected_shape = (int(expected_shape[0]), int(expected_shape[1]))
        if shape != expected_shape:
            raise ValueError(
                f"Image grid shape {shape} does not match expected mask grid "
                f"{expected_shape}"
            )
    extent = sky_extent_arcsec(shape, pixel_scale)
    return shape, extent


def load_truth_matching_grid(path, *, shape, pixel_scale, atol=1e-9):
    """
    Load a truth FITS image and require it match ``shape`` / ``pixel_scale``.

    Checks array shape and, when present, ``PIXSCALE`` / ``NPIX`` header cards.
    """
    from astropy.io import fits

    path = Path(path)
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    expected = (int(shape[0]), int(shape[1]))
    got = tuple(int(v) for v in data.shape)
    if got != expected:
        raise ValueError(
            f"Truth image {path} has shape {got}, expected mask grid {expected}. "
            "Regenerate the mock with the current fov / mask_n_pixels / "
            "mask_pad_pixels (omit --skip-mock)."
        )

    pix = header.get("PIXSCALE")
    if pix is not None and abs(float(pix) - float(pixel_scale)) > float(atol):
        raise ValueError(
            f"Truth image {path} PIXSCALE={float(pix)} does not match mask "
            f"pixel scale {float(pixel_scale)}. Regenerate the mock."
        )
    npix = header.get("NPIX")
    if npix is not None and int(npix) != expected[0]:
        raise ValueError(
            f"Truth image {path} NPIX={int(npix)} does not match mask "
            f"n_pixels={expected[0]}. Regenerate the mock."
        )
    return data


def dirty_noise_rms_from_fit(fit, n_realizations=8, seed=0):
    """
    Estimate dirty-image noise RMS by transforming visibility noise draws.

    Falls back to a robust RMS of the dirty residual if the transformer path
    is unavailable.
    """
    try:
        import autolens as al

        noise = np.asarray(fit.dataset.noise_map)
        if np.iscomplexobj(noise):
            sigma = np.real(noise).astype(float)
        elif noise.ndim == 2 and noise.shape[-1] == 2:
            sigma = noise[..., 0].astype(float)
        else:
            sigma = np.asarray(noise, dtype=float).reshape(-1)

        transformer = fit.dataset.transformer
        rng = np.random.default_rng(seed)
        rms_list = []
        for _ in range(int(n_realizations)):
            noise_vis = (
                sigma
                / np.sqrt(2.0)
                * (rng.standard_normal(sigma.shape) + 1j * rng.standard_normal(sigma.shape))
            )
            dirty = transformer.image_from(
                visibilities=al.Visibilities(visibilities=noise_vis)
            )
            arr = np.asarray(dirty.native if hasattr(dirty, "native") else dirty)
            rms_list.append(robust_rms(arr))
        estimate = float(np.median(rms_list))
        if np.isfinite(estimate) and estimate > 0:
            return estimate
    except Exception as exc:
        logger.warning("Dirty noise RMS from visibility draws failed (%s)", exc)

    residual = np.asarray(fit.dirty_residual_map.native)
    return robust_rms(residual)


def plot_fit_summary(
    *,
    dirty_image,
    reconstruction,
    truth_image=None,
    residual_dirty=None,
    residual_sigma=None,
    noise_rms=None,
    output_path,
    title="pyuvimage deconv",
    coefficient=None,
    pixel_scale,
    nyquist_pixel_scale=None,
    expected_shape=None,
):
    """
    Save a multi-panel PNG summarizing the fit.

    **All panels are forced onto one sky grid**: identical ``(ny, nx)`` and the
    same ``pixel_scale`` (arcsec). Pass ``expected_shape`` (mask native) to
    also reject arrays that do not match the reconstruction mask.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Dirty", dirty_image), ("Reconstruction", reconstruction)]
    if residual_sigma is not None:
        label = "Dirty residual / σ"
        if noise_rms is not None and np.isfinite(noise_rms):
            label = f"Dirty residual / σ\n(σ={noise_rms:.3g})"
        panels.append((label, residual_sigma))
    elif residual_dirty is not None:
        panels.append(("Dirty residual", residual_dirty))
    if truth_image is not None:
        panels.insert(0, ("Truth", truth_image))

    shape, extent = require_common_sky_grid(
        panels, pixel_scale=pixel_scale, expected_shape=expected_shape
    )
    ny, nx = shape

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]

    for ax, (label, image) in zip(axes, panels):
        data = np.asarray(image, dtype=float)
        is_residual = "residual" in label.lower()
        if is_residual:
            vmax = np.nanpercentile(np.abs(data), 99.5)
            vmax = max(float(vmax), 1.0)
            im = ax.imshow(
                data,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=extent,
                aspect="equal",
                interpolation="nearest",
            )
        else:
            vmax = np.nanpercentile(np.abs(data), 99.5)
            im = ax.imshow(
                data,
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=max(vmax, 1e-12),
                extent=extent,
                aspect="equal",
                interpolation="nearest",
            )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("ΔRA [arcsec]", fontsize=8)
        ax.set_ylabel("ΔDec [arcsec]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    subtitle = title
    if coefficient is not None:
        subtitle = f"{title}  (λ={coefficient:.4g})"
    scale_note = f"pixel scale={float(pixel_scale):.4g}\"  grid={ny}×{nx}"
    if nyquist_pixel_scale is not None:
        scale_note += f"  (Nyquist ½λ/b_max={float(nyquist_pixel_scale):.4g}\")"
    subtitle = f"{subtitle}\n{scale_note}"
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def dirty_images_from_fit(fit):
    """
    Return dirty data/model/residual arrays and a noise-normalized residual.

    Returns
    -------
    dirty_data, dirty_model, dirty_residual, residual_sigma, noise_rms
    """
    dirty_data = np.asarray(fit.dirty_image.native)
    dirty_model = np.asarray(fit.dirty_model_image.native)
    dirty_residual = dirty_data - dirty_model
    noise_rms = dirty_noise_rms_from_fit(fit)
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        noise_rms = robust_rms(dirty_residual)
    residual_sigma = dirty_residual / noise_rms
    return dirty_data, dirty_model, dirty_residual, residual_sigma, noise_rms

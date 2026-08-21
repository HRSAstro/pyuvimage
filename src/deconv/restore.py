"""CLEAN-style restore: model ⊗ clean beam + dirty residual, in Jy/beam."""

from __future__ import annotations

import numpy as np

# Solid-angle conversions for a Gaussian beam.
SIGMA_TO_FWHM = np.sqrt(8.0 * np.log(2.0))
FWHM_TO_AREA = 2.0 * np.pi / (8.0 * np.log(2.0))


def gaussian_beam_area_pixels(sigma_x, sigma_y):
    """
    Gaussian beam solid angle in pixels².

    ``Ω = FWHM_TO_AREA * FWHM_x * FWHM_y`` with ``FWHM = SIGMA_TO_FWHM * σ``.
    """
    fwhm_x = SIGMA_TO_FWHM * float(sigma_x)
    fwhm_y = SIGMA_TO_FWHM * float(sigma_y)
    return FWHM_TO_AREA * fwhm_x * fwhm_y


def dirty_beam_from_transformer(transformer, mask):
    """
    Dirty beam (PSF) from a unit impulse at the mask centre.

    Returns a native 2-D array on the mask grid.
    """
    import autolens as al

    kept = ~np.asarray(mask, dtype=bool)
    # Place impulse on the unmasked pixel closest to the array centre.
    yy, xx = np.indices(mask.shape_native)
    cy = 0.5 * (mask.shape_native[0] - 1)
    cx = 0.5 * (mask.shape_native[1] - 1)
    dist = np.where(kept, (yy - cy) ** 2 + (xx - cx) ** 2, np.inf)
    y0, x0 = np.unravel_index(np.argmin(dist), dist.shape)
    native = np.zeros(mask.shape_native, dtype=float)
    native[y0, x0] = 1.0
    image = al.Array2D(values=native[kept], mask=mask)
    vis = transformer.visibilities_from(image=image)
    beam = transformer.image_from(visibilities=vis)
    return np.asarray(beam.native if hasattr(beam, "native") else beam, dtype=float)


def fit_clean_beam_gaussian(dirty_beam, *, window_frac=0.25, pixel_scale=None):
    """
    Fit an elliptical Gaussian to the main lobe of the dirty beam.

    Returns a peak-normalized kernel (same shape as ``dirty_beam``) and a
    dict of fit parameters. Beam area uses ``FWHM_TO_AREA`` / ``SIGMA_TO_FWHM``.
    """
    from astropy.modeling import fitting, models

    beam = np.asarray(dirty_beam, dtype=float)
    ny, nx = beam.shape
    peak = float(np.nanmax(beam))
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError("Dirty beam peak is zero; cannot fit a clean beam.")
    y0, x0 = np.unravel_index(np.nanargmax(beam), beam.shape)

    # Fit only a window around the peak (main lobe).
    wy = max(3, int(round(window_frac * ny)))
    wx = max(3, int(round(window_frac * nx)))
    y1, y2 = max(0, y0 - wy // 2), min(ny, y0 + wy // 2 + 1)
    x1, x2 = max(0, x0 - wx // 2), min(nx, x0 + wx // 2 + 1)
    patch = beam[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]

    # Initial FWHM guess from second moments of the positive core.
    pos = np.maximum(patch, 0.0)
    w = pos / max(float(pos.sum()), 1e-12)
    y_mean = float((w * yy).sum())
    x_mean = float((w * xx).sum())
    y_var = float((w * (yy - y_mean) ** 2).sum())
    x_var = float((w * (xx - x_mean) ** 2).sum())
    sy = max(np.sqrt(max(y_var, 1e-6)), 0.5)
    sx = max(np.sqrt(max(x_var, 1e-6)), 0.5)

    g_init = models.Gaussian2D(
        amplitude=peak,
        x_mean=x_mean,
        y_mean=y_mean,
        x_stddev=sx,
        y_stddev=sy,
        theta=0.0,
    )
    fitter = fitting.LevMarLSQFitter()
    g_fit = fitter(g_init, xx, yy, patch)

    yy_full, xx_full = np.mgrid[0:ny, 0:nx]
    kernel = np.asarray(g_fit(xx_full, yy_full), dtype=float)
    kpeak = float(np.nanmax(kernel))
    if kpeak != 0.0:
        kernel = kernel / kpeak

    sigma_x = float(g_fit.x_stddev.value)
    sigma_y = float(g_fit.y_stddev.value)
    fwhm_x = SIGMA_TO_FWHM * sigma_x
    fwhm_y = SIGMA_TO_FWHM * sigma_y
    beam_area_pix = gaussian_beam_area_pixels(sigma_x, sigma_y)
    theta = float(g_fit.theta.value)

    # Major = larger FWHM; BPA = position angle of major axis, degrees East of North.
    # Image axes: +x ≈ East (ΔRA increasing right), +y ≈ North (ΔDec up).
    if fwhm_x >= fwhm_y:
        bmaj_pix, bmin_pix = fwhm_x, fwhm_y
        phi = theta  # major along Gaussian x-axis
    else:
        bmaj_pix, bmin_pix = fwhm_y, fwhm_x
        phi = theta + 0.5 * np.pi  # major along Gaussian y-axis
    # Direction (cos φ, sin φ) in (E, N); PA from North toward East.
    bpa_deg = float(np.degrees(np.arctan2(np.cos(phi), np.sin(phi))))
    bpa_deg = ((bpa_deg + 90.0) % 180.0) - 90.0

    params = {
        "amplitude": float(g_fit.amplitude.value),
        "x_mean": float(g_fit.x_mean.value),
        "y_mean": float(g_fit.y_mean.value),
        "x_stddev": sigma_x,
        "y_stddev": sigma_y,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "bmaj_pix": float(bmaj_pix),
        "bmin_pix": float(bmin_pix),
        "bpa_deg": bpa_deg,
        "theta": theta,
        "dirty_beam_peak": peak,
        "beam_area_pixels": float(beam_area_pix),
    }
    if pixel_scale is not None:
        ps = float(pixel_scale)
        params["pixel_scale"] = ps
        params["fwhm_x_arcsec"] = fwhm_x * ps
        params["fwhm_y_arcsec"] = fwhm_y * ps
        params["bmaj_arcsec"] = float(bmaj_pix) * ps
        params["bmin_arcsec"] = float(bmin_pix) * ps
        params["beam_area_arcsec2"] = float(beam_area_pix) * ps * ps
    return kernel, params


def restore_clean_image(
    model,
    residual,
    clean_beam,
    *,
    beam_area_pixels,
    dirty_beam_peak,
):
    """
    CLEAN-style restore in **Jy/beam** (clean-beam solid angle).

    Assumes ``model`` is in Jy/pixel. With a peak-normalized clean beam:

        restored_Jy/beam = (model ⊗ B_clean) × (Ω_beam / Σ B_clean)

    where ``Ω_beam = FWHM_TO_AREA * FWHM_x * FWHM_y``. The dirty residual is
    converted to Jy/beam via the dirty-beam peak (unit-flux impulse response):

        residual_Jy/beam = residual / peak(B_dirty)

        clean = restored_Jy/beam + residual_Jy/beam
    """
    from scipy.signal import fftconvolve

    model = np.asarray(model, dtype=float)
    residual = np.asarray(residual, dtype=float)
    clean_beam = np.asarray(clean_beam, dtype=float)
    if model.shape != residual.shape:
        raise ValueError(
            f"model shape {model.shape} != residual shape {residual.shape}"
        )
    if clean_beam.shape != model.shape:
        raise ValueError(
            f"clean_beam shape {clean_beam.shape} != model shape {model.shape}"
        )

    beam_area = float(beam_area_pixels)
    dirty_peak = float(dirty_beam_peak)
    if beam_area <= 0.0:
        raise ValueError(f"beam_area_pixels must be positive; got {beam_area}")
    if dirty_peak == 0.0 or not np.isfinite(dirty_peak):
        raise ValueError(f"dirty_beam_peak must be finite and non-zero; got {dirty_peak}")

    kernel_sum = float(np.sum(clean_beam))
    if kernel_sum == 0.0:
        raise ValueError("clean_beam sum is zero")

    restored = fftconvolve(model, clean_beam, mode="same")
    restored_jy_beam = restored * (beam_area / kernel_sum)
    residual_jy_beam = residual / dirty_peak
    return restored_jy_beam + residual_jy_beam


def clean_image_from_model_and_dirty(
    *,
    transformer,
    mask,
    model,
    dirty_data,
    dirty_model,
    clean_beam=None,
    beam_params=None,
    pixel_scale=None,
):
    """
    Build a Jy/beam CLEAN restore from a sky model and dirty residual.
    """
    residual = np.asarray(dirty_data, dtype=float) - np.asarray(dirty_model, dtype=float)
    if clean_beam is None or beam_params is None:
        dirty_beam = dirty_beam_from_transformer(transformer, mask)
        clean_beam, beam_params = fit_clean_beam_gaussian(
            dirty_beam, pixel_scale=pixel_scale
        )

    clean = restore_clean_image(
        model,
        residual,
        clean_beam,
        beam_area_pixels=beam_params["beam_area_pixels"],
        dirty_beam_peak=beam_params["dirty_beam_peak"],
    )
    return clean, residual, clean_beam, beam_params

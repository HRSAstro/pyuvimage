"""Standard fit product plots and FITS outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.deconv.plots import require_common_sky_grid, sky_extent_arcsec
from src.deconv.restore import (
    clean_image_from_model_and_dirty,
    dirty_beam_from_transformer,
    fit_clean_beam_gaussian,
)
from src.utils.io import write_image_fits

logger = logging.getLogger(__name__)


def _base_header(pixel_scale, *, bunit=None, extra=None):
    cards = {}
    if extra:
        cards.update(extra)
    # Always set last so callers cannot overwrite with a bare float.
    cards["PIXSCALE"] = (float(pixel_scale), "pixel scale [arcsec]")
    if bunit is not None:
        cards["BUNIT"] = (str(bunit), "brightness unit")
    return cards


def _beam_header_cards(beam_params):
    """BMAJ/BMIN [arcsec], BPA [deg East of North]."""
    return {
        "BMAJ": (
            float(beam_params["bmaj_arcsec"]),
            "clean beam major FWHM [arcsec]",
        ),
        "BMIN": (
            float(beam_params["bmin_arcsec"]),
            "clean beam minor FWHM [arcsec]",
        ),
        "BPA": (
            float(beam_params["bpa_deg"]),
            "clean beam PA [deg East of North]",
        ),
    }


def plot_fit_diagnostic(
    *,
    dirty_image,
    dirty_model,
    reconstruction,
    residual_sigma,
    output_path,
    pixel_scale,
    noise_rms=None,
    title="pyuvimage fit",
    expected_shape=None,
):
    """
    Four-panel PNG: dirty, dirty model, reconstruction, noise-normalized residual.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resid_label = "Residual / σ"
    if noise_rms is not None and np.isfinite(noise_rms):
        resid_label = f"Residual / σ\n(σ={float(noise_rms):.3g})"

    panels = [
        ("Dirty", dirty_image),
        ("Dirty model", dirty_model),
        ("Reconstruction", reconstruction),
        (resid_label, residual_sigma),
    ]
    shape, extent = require_common_sky_grid(
        panels, pixel_scale=pixel_scale, expected_shape=expected_shape
    )
    ny, nx = shape

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2))
    for ax, (label, image) in zip(axes, panels):
        data = np.asarray(image, dtype=float)
        is_residual = "residual" in label.lower()
        if is_residual:
            vmax = max(float(np.nanpercentile(np.abs(data), 99.5)), 1.0)
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
            vmax = float(np.nanpercentile(np.abs(data), 99.5))
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

    fig.suptitle(
        f'{title}\npixel scale={float(pixel_scale):.4g}"  grid={ny}×{nx}',
        fontsize=11,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def plot_clean_image(
    *,
    clean_image,
    output_path,
    pixel_scale,
    title="Clean image [Jy/beam]",
    expected_shape=None,
    beam_params=None,
):
    """Single-panel PNG of the CLEAN restore in Jy/beam."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Clean [Jy/beam]", clean_image)]
    shape, extent = require_common_sky_grid(
        panels, pixel_scale=pixel_scale, expected_shape=expected_shape
    )
    data = np.asarray(clean_image, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))
    im = ax.imshow(
        data,
        origin="lower",
        cmap="magma",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("ΔRA [arcsec]")
    ax.set_ylabel("ΔDec [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    note = f'pixel scale={float(pixel_scale):.4g}"  grid={shape[0]}×{shape[1]}'
    if beam_params is not None and "bmaj_arcsec" in beam_params:
        note += (
            f'  beam={beam_params["bmaj_arcsec"]:.3g}"×'
            f'{beam_params["bmin_arcsec"]:.3g}" '
            f'PA={beam_params["bpa_deg"]:.1f}°'
        )
    fig.suptitle(note, fontsize=9)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def write_fit_products(
    *,
    output_dir,
    dirty_data,
    dirty_model,
    reconstruction,
    residual_sigma,
    noise_rms,
    dataset,
    pixel_scale,
    title="pyuvimage fit",
    expected_shape=None,
    extra_header=None,
):
    """
    Write standard fit plots and FITS products.

    Plots
    -----
    - ``fit_summary.png``: dirty / dirty model / reconstruction / residual/σ
    - ``clean.png``: CLEAN restore [Jy/beam]

    FITS
    ----
    - ``dirty_image.fits``
    - ``dirty_model.fits``
    - ``residual_sigma.fits`` — (data−model)/σ; header ``NOISE`` = σ [Jy/beam]
    - ``clean_image.fits`` — Jy/beam
    - ``reconstruction.fits`` — sky model on the mask grid

    Clean and residual FITS include ``BMAJ``/``BMIN`` [arcsec] and ``BPA``
    [deg East of North]. All products include ``PIXSCALE`` [arcsec].
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ps = float(pixel_scale)
    extra = dict(extra_header or {})

    dirty_data = np.asarray(dirty_data, dtype=float)
    dirty_model = np.asarray(dirty_model, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    residual_sigma = np.asarray(residual_sigma, dtype=float)
    noise_rms = float(noise_rms)

    mask = dataset.mask
    transformer = dataset.transformer
    dirty_beam = dirty_beam_from_transformer(transformer, mask)
    clean_beam, beam_params = fit_clean_beam_gaussian(
        dirty_beam, pixel_scale=ps
    )
    clean, residual_dirty, _, beam_params = clean_image_from_model_and_dirty(
        transformer=transformer,
        mask=mask,
        model=reconstruction,
        dirty_data=dirty_data,
        dirty_model=dirty_model,
        clean_beam=clean_beam,
        beam_params=beam_params,
        pixel_scale=ps,
    )

    dirty_peak = float(beam_params["dirty_beam_peak"])
    noise_rms_jybeam = noise_rms / dirty_peak if dirty_peak != 0.0 else noise_rms

    summary_path = plot_fit_diagnostic(
        dirty_image=dirty_data,
        dirty_model=dirty_model,
        reconstruction=reconstruction,
        residual_sigma=residual_sigma,
        noise_rms=noise_rms,
        output_path=output_dir / "fit_summary.png",
        pixel_scale=ps,
        title=title,
        expected_shape=expected_shape,
    )
    clean_plot_path = plot_clean_image(
        clean_image=clean,
        output_path=output_dir / "clean.png",
        pixel_scale=ps,
        expected_shape=expected_shape,
        beam_params=beam_params,
    )

    beam_cards = _beam_header_cards(beam_params)
    paths = {
        "fit_summary": summary_path,
        "clean_plot": clean_plot_path,
        "dirty_image": write_image_fits(
            output_dir / "dirty_image.fits",
            dirty_data,
            header_cards=_base_header(ps, bunit="JY/PIXEL", extra=extra),
        ),
        "dirty_model": write_image_fits(
            output_dir / "dirty_model.fits",
            dirty_model,
            header_cards=_base_header(ps, bunit="JY/PIXEL", extra=extra),
        ),
        "reconstruction": write_image_fits(
            output_dir / "reconstruction.fits",
            reconstruction,
            header_cards=_base_header(ps, bunit="JY/PIXEL", extra=extra),
        ),
        "residual_sigma": write_image_fits(
            output_dir / "residual_sigma.fits",
            residual_sigma,
            header_cards=_base_header(
                ps,
                bunit="SIGMA",
                extra={
                    **extra,
                    **beam_cards,
                    "NOISE": (
                        float(noise_rms_jybeam),
                        "rms noise used for normalisation [Jy/beam]",
                    ),
                    "NOISEDRT": (
                        float(noise_rms),
                        "rms noise in dirty-image units",
                    ),
                },
            ),
        ),
        "clean_image": write_image_fits(
            output_dir / "clean_image.fits",
            clean,
            header_cards=_base_header(
                ps,
                bunit="JY/BEAM",
                extra={**extra, **beam_cards},
            ),
        ),
    }
    logger.info(
        "Fit products: BMAJ=%.4g\" BMIN=%.4g\" BPA=%.2f°  NOISE=%.4g Jy/beam",
        beam_params["bmaj_arcsec"],
        beam_params["bmin_arcsec"],
        beam_params["bpa_deg"],
        noise_rms_jybeam,
    )
    return {
        "paths": paths,
        "clean_image": clean,
        "beam_params": beam_params,
        "noise_rms": noise_rms,
        "noise_rms_jybeam": noise_rms_jybeam,
        "residual_dirty": residual_dirty,
    }

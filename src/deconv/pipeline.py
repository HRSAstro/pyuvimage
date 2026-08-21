"""Top-level deconvolution pipeline: MFS and per-channel cube modes."""

from __future__ import annotations

import logging
from pathlib import Path

import autofit as af
import numpy as np

from src.deconv.data import load_cube_data, load_cube_data_weights
from src.deconv.invert import (
    channel_dataset_from,
    mfs_dataset_from,
    run_inversion,
)
from src.deconv.model import coefficient_from_instance, reconstruction_mask_from_settings
from src.deconv.settings import validate_settings
from src.utils.grids import resolve_grids, transformer_class_from_settings
from src.utils.io import write_image_fits

logger = logging.getLogger(__name__)


def run_deconv(settings):
    """
    Run UV-plane image deconvolution from validated runner settings.

    ``mode=mfs``: channel-mean collapse → one inversion.
    ``mode=cube``: fit λ on MFS (default), then invert each channel with λ frozen;
    stack reconstructions into an image cube.
    """
    validate_settings(settings)

    af.conf.instance.push(
        new_path=settings.get("config_path", "./config"),
        output_path=settings["output_path"],
    )

    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    weights = load_cube_data_weights(settings)
    grids = resolve_grids(settings, uv_wavelengths)
    logger.info(
        "Grids: fov=%.4f\", source_pixel_scale=%.6g\", mesh_shape=%s, "
        "mask=%s² @ %.6g\" (science=%s², pad=%s pix, radius=%.4f\")",
        grids["fov"],
        grids["source_pixel_scale"],
        grids["mesh_shape"],
        grids["mask_n_pixels"],
        grids["mask_pixel_scale"],
        grids["science_n_pixels"],
        grids["mask_pad_pixels"],
        grids["mask_radius"],
    )

    mask_2d = reconstruction_mask_from_settings(settings)
    transformer_class = transformer_class_from_settings(settings)
    mode = settings["mode"]

    if mode == "mfs":
        return _run_mfs(
            settings=settings,
            uv_wavelengths=uv_wavelengths,
            visibilities=visibilities,
            sigma=sigma,
            weights=weights,
            mask_2d=mask_2d,
            transformer_class=transformer_class,
            frequencies=frequencies,
        )

    reconstructor = settings.get("reconstructor", "pixelization")
    if reconstructor in {"log_sky", "linear_sky", "auto"}:
        raise ValueError(
            f"reconstructor={reconstructor!r} currently supports mode='mfs' only."
        )

    return _run_cube(
        settings=settings,
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        weights=weights,
        mask_2d=mask_2d,
        transformer_class=transformer_class,
        frequencies=frequencies,
    )


def _run_mfs(
    settings,
    uv_wavelengths,
    visibilities,
    sigma,
    weights,
    mask_2d,
    transformer_class,
    frequencies,
):
    dataset = mfs_dataset_from(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        mask_2d=mask_2d,
        settings=settings,
        weights=weights,
        transformer_class=transformer_class,
    )

    reconstructor = settings.get("reconstructor", "pixelization")
    snr_info = None
    if reconstructor == "auto":
        from src.deconv.sky_reg import choose_sky_reconstructor, estimate_dirty_snr

        snr_info = estimate_dirty_snr(dataset)
        threshold = float(settings.get("sky_auto", {}).get("snr_threshold", 100.0))
        chosen = choose_sky_reconstructor(snr_info.snr, threshold)
        logger.info(
            "Auto reconstructor: dirty_peak=%.4g noise_rms=%.4g SNR=%.4g "
            "threshold=%.4g → %s",
            snr_info.peak,
            snr_info.noise_rms,
            snr_info.snr,
            threshold,
            chosen,
        )
        reconstructor = chosen
        settings["reconstructor_resolved"] = chosen
        settings["dirty_snr"] = {
            "peak": snr_info.peak,
            "noise_rms": snr_info.noise_rms,
            "snr": snr_info.snr,
            "threshold": threshold,
        }

    if reconstructor in {"log_sky", "linear_sky"}:
        from src.deconv.sky_reg import run_sky_fit_with_reg_search

        if settings["mode"] != "mfs":
            raise ValueError(
                f"reconstructor={reconstructor!r} currently supports mode='mfs' only."
            )
        sky_result = run_sky_fit_with_reg_search(settings, dataset, reconstructor)
        source_image = sky_result.image
        label = "log-sky" if reconstructor == "log_sky" else "linear-sky"
        _assert_mask_grid_image(source_image, settings, name=f"{label} reconstruction")
        _assert_mask_grid_image(sky_result.dirty_data, settings, name="dirty data")
        _assert_mask_grid_image(
            sky_result.residual_sigma, settings, name="dirty residual"
        )
        out_path = _write_mfs_product(settings, source_image, frequencies)
        logger.info("Wrote MFS %s reconstruction: %s", label, out_path)
        if getattr(sky_result, "optimize_smooth", False):
            logger.info(
                "%s smooth search: init=%.4g best=%.4g LLWR=%.6g n_trials=%d",
                label,
                sky_result.smooth_init,
                sky_result.smooth_best,
                sky_result.llwr if sky_result.llwr is not None else float("nan"),
                len(sky_result.smooth_trials or []),
            )
            if getattr(sky_result, "log_evidence_approx", None) is not None:
                logger.info(
                    "%s evidence approx FOM=%.6g",
                    label,
                    sky_result.log_evidence_approx,
                )
        payload = {
            "mode": "mfs",
            "reconstructor": reconstructor,
            "result": sky_result,
            "source_image": source_image,
            "dataset": dataset,
            "output_path": out_path,
            "dirty_snr": settings.get("dirty_snr"),
        }
        return _attach_mfs_fit_products(
            settings, dataset, payload, frequencies, title=f"{label} fit"
        )

    result, source_image = run_inversion(
        settings, dataset, search_name=settings["search"].get("name", "mfs")
    )
    _assert_mask_grid_image(source_image, settings, name="reconstruction")
    out_path = _write_mfs_product(settings, source_image, frequencies)
    logger.info("Wrote MFS reconstruction: %s", out_path)
    payload = {
        "mode": "mfs",
        "reconstructor": "pixelization",
        "result": result,
        "source_image": source_image,
        "dataset": dataset,
        "output_path": out_path,
    }
    return _attach_mfs_fit_products(
        settings, dataset, payload, frequencies, title="MFS deconv fit"
    )


def _attach_mfs_fit_products(settings, dataset, payload, frequencies, *, title):
    """Write standard diagnostic plots + FITS products for an MFS fit."""
    from src.deconv.plots import dirty_images_from_fit
    from src.deconv.products import write_fit_products

    reconstructor = payload["reconstructor"]
    source_image = payload["source_image"]
    if reconstructor in {"log_sky", "linear_sky"}:
        sky = payload["result"]
        dirty_data = sky.dirty_data
        dirty_model = sky.dirty_model
        residual_sigma = sky.residual_sigma
        noise_rms = sky.noise_rms
        if getattr(sky, "smooth_best", None) is not None:
            title = f"{title} (smooth={float(sky.smooth_best):g})"
        elif getattr(sky, "smooth", None) is not None:
            title = f"{title} (smooth={float(sky.smooth):g})"
    else:
        fit = payload["result"].max_log_likelihood_fit
        dirty_data, dirty_model, _, residual_sigma, noise_rms = dirty_images_from_fit(
            fit
        )

    pixel_scale = float(settings["mask_pixel_scale"])
    expected_shape = (int(settings["mask_n_pixels"]), int(settings["mask_n_pixels"]))
    products = write_fit_products(
        output_dir=settings["output_path"],
        dirty_data=dirty_data,
        dirty_model=dirty_model,
        reconstruction=source_image,
        residual_sigma=residual_sigma,
        noise_rms=noise_rms,
        dataset=dataset,
        pixel_scale=pixel_scale,
        title=title,
        expected_shape=expected_shape,
        extra_header=_header_cards(settings, frequencies),
    )
    payload["products"] = products
    payload["dirty_data"] = dirty_data
    payload["dirty_model"] = dirty_model
    payload["residual_sigma"] = residual_sigma
    payload["noise_rms"] = noise_rms
    return payload


def _run_cube(
    settings,
    uv_wavelengths,
    visibilities,
    sigma,
    weights,
    mask_2d,
    transformer_class,
    frequencies,
):
    cube_reg = settings.get("cube", {}).get("regularization", "from_mfs")
    fixed_coefficient = None
    mfs_result = None

    if cube_reg == "from_mfs":
        logger.info("Cube mode: fitting regularization on MFS collapse first.")
        mfs_dataset = mfs_dataset_from(
            uv_wavelengths=uv_wavelengths,
            visibilities=visibilities,
            sigma=sigma,
            mask_2d=mask_2d,
            settings=settings,
            weights=weights,
            transformer_class=transformer_class,
        )
        mfs_result, mfs_image = run_inversion(
            settings, mfs_dataset, search_name="mfs_for_cube"
        )
        try:
            fixed_coefficient = coefficient_from_instance(
                mfs_result.max_log_likelihood_instance
            )
            logger.info("Freezing regularization coefficient=%.6g for channels", fixed_coefficient)
        except AttributeError:
            logger.warning(
                "Could not extract a single coefficient (adapt reg?); "
                "continuing with per-channel free regularization."
            )
            fixed_coefficient = None
        _write_mfs_product(settings, mfs_image, frequencies, filename="reconstruction_mfs.fits")

    n_chan = int(np.asarray(visibilities).shape[0])
    channel_images = []
    channel_results = []
    for i in range(n_chan):
        logger.info("Cube channel %s / %s", i + 1, n_chan)
        dataset = channel_dataset_from(
            uv_wavelengths=uv_wavelengths,
            visibilities=visibilities,
            sigma=sigma,
            mask_2d=mask_2d,
            channel_index=i,
            transformer_class=transformer_class,
        )
        # Per-channel free λ only when requested and Constant was not frozen.
        coeff = fixed_coefficient if cube_reg == "from_mfs" else None
        if cube_reg == "per_channel":
            coeff = None
        result, image = run_inversion(
            settings,
            dataset,
            fixed_coefficient=coeff,
            search_name=f"channel_{i:04d}",
        )
        channel_images.append(image)
        channel_results.append(result)

    cube = np.stack(channel_images, axis=0)
    out_path = _write_cube_product(settings, cube, frequencies)
    logger.info("Wrote cube reconstruction: %s", out_path)
    return {
        "mode": "cube",
        "result": mfs_result,
        "channel_results": channel_results,
        "source_cube": cube,
        "output_path": out_path,
        "fixed_coefficient": fixed_coefficient,
    }


def _assert_mask_grid_image(image, settings, *, name):
    """Require ``image`` to match the resolved mask native grid."""
    expected = (int(settings["mask_n_pixels"]), int(settings["mask_n_pixels"]))
    got = tuple(int(v) for v in np.asarray(image).shape)
    if got != expected:
        raise ValueError(
            f"{name} shape {got} does not match mask grid {expected} "
            f"(pixel scale {settings['mask_pixel_scale']}\")"
        )


def _header_cards(settings, frequencies):
    mask_fov = float(settings.get("mask_fov", settings["fov"]))
    cards = {
        "FOV": mask_fov,
        "SCIFOV": float(settings["fov"]),
        "PIXSCALE": float(settings["mask_pixel_scale"]),
        "NPIX": int(settings["mask_n_pixels"]),
        "SRCPIX": float(settings["source_pixel_scale"]),
        "NYQPIX": float(settings.get("nyquist_pixel_scale", settings["source_pixel_scale"])),
        "MESH0": int(settings["mesh_shape"][0]),
        "MESH1": int(settings["mesh_shape"][1]),
        "MODE": str(settings["mode"]),
    }
    freq = np.squeeze(np.asarray(frequencies, dtype=float))
    if freq.ndim == 0:
        cards["FREQ"] = float(freq)
    elif freq.size:
        cards["FREQ0"] = float(freq.flat[0])
        cards["FREQN"] = float(freq.flat[-1])
        cards["NCHAN"] = int(freq.size)
    return cards


def _write_mfs_product(settings, source_image, frequencies, filename="reconstruction_mfs.fits"):
    out_dir = Path(settings["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    return write_image_fits(
        path,
        source_image,
        header_cards=_header_cards(settings, frequencies),
    )


def _write_cube_product(settings, cube, frequencies):
    out_dir = Path(settings["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reconstruction_cube.fits"
    return write_image_fits(
        path,
        cube,
        header_cards=_header_cards(settings, frequencies),
    )

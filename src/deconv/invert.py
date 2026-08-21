"""Build Interferometer datasets and run pixelized inversions."""

from __future__ import annotations

import logging

import autofit as af
import autolens as al
import numpy as np

from src.deconv.analysis import DeconvAnalysisInterferometer
from src.deconv.model import (
    adapt_images_for_dataset,
    build_deconv_model,
    inversion_settings_from_settings,
    reconstruction_mask_from_settings,
)
from src.deconv.moment0 import mfs_arrays_from, mfs_settings_from_settings
from src.deconv.search import build_optimizer_from_settings
from src.utils.grids import transformer_class_from_settings
from src.utils.jax_compat import ensure_numpy_jax_stub, jax_is_usable

logger = logging.getLogger(__name__)


def _never_visualize(paths, during_analysis=True):
    return False


class _FixedResult:
    def __init__(self, instance, fit):
        self.max_log_likelihood_instance = instance
        self.max_log_likelihood_fit = fit


def interferometer_from_visibilities(
    uv_wavelengths,
    visibilities,
    sigma,
    mask_2d,
    transformer_class=None,
):
    """Build ``al.Interferometer`` from collapsed or single-channel arrays."""
    if transformer_class is None:
        transformer_class = al.TransformerDFT

    data = al.Visibilities(visibilities=visibilities)
    noise_map = al.VisibilitiesNoiseMap(visibilities=sigma)
    kwargs = {}
    if transformer_class is al.TransformerDFT:
        kwargs["raise_error_dft_visibilities_limit"] = False

    return al.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask_2d,
        transformer_class=transformer_class,
        **kwargs,
    )


def mfs_dataset_from(
    uv_wavelengths,
    visibilities,
    sigma,
    mask_2d,
    settings,
    weights=None,
    transformer_class=None,
):
    """Channel-mean Interferometer dataset."""
    if transformer_class is None:
        transformer_class = transformer_class_from_settings(settings)
    mfs_settings = mfs_settings_from_settings(settings)
    vis_mfs, sigma_mfs, uv_mfs = mfs_arrays_from(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        mfs_settings=mfs_settings,
        weights=weights,
    )
    return interferometer_from_visibilities(
        uv_wavelengths=uv_mfs,
        visibilities=vis_mfs,
        sigma=sigma_mfs,
        mask_2d=mask_2d,
        transformer_class=transformer_class,
    )


def channel_dataset_from(
    uv_wavelengths,
    visibilities,
    sigma,
    mask_2d,
    channel_index,
    transformer_class=None,
):
    """Single spectral channel Interferometer dataset."""
    return interferometer_from_visibilities(
        uv_wavelengths=uv_wavelengths[channel_index],
        visibilities=visibilities[channel_index],
        sigma=sigma[channel_index],
        mask_2d=mask_2d,
        transformer_class=transformer_class,
    )


def _prepare_dataset(dataset, settings):
    use_jax = bool(settings.get("use_jax", True))
    ensure_numpy_jax_stub()
    if use_jax and not jax_is_usable():
        logger.warning("jax/jaxlib unavailable; forcing use_jax=False.")
        use_jax = False
    if use_jax:
        dataset = dataset.apply_sparse_operator(use_jax=True)
    return dataset, use_jax


def run_inversion(settings, dataset, *, fixed_coefficient=None, search_name=None):
    """
    Run a pixelized inversion on ``dataset``.

    Returns ``(result, source_image_2d)`` where ``source_image_2d`` is the
    reconstruction interpolated onto a regular FOV grid.
    """
    mask_2d = dataset.mask
    model = build_deconv_model(
        settings, fixed_coefficient=fixed_coefficient, mask_2d=mask_2d
    )
    search_cfg = dict(settings.get("search", {}))
    if search_name is not None:
        search_cfg["name"] = search_name
    search = build_optimizer_from_settings(search_cfg)

    dataset, use_jax = _prepare_dataset(dataset, settings)
    adapt_images = adapt_images_for_dataset(
        settings, dataset, mask_2d=mask_2d
    )

    figure_of_merit = search_cfg.get(
        "figure_of_merit", "log_likelihood_with_regularization"
    )
    analysis = DeconvAnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=inversion_settings_from_settings(settings),
        raise_inversion_positions_likelihood_exception=False,
        use_jax=use_jax,
        figure_of_merit=figure_of_merit,
    )
    if not settings.get("visualize", False):
        analysis.should_visualize = _never_visualize

    if model.prior_count == 0:
        instance = model.instance_from_prior_medians()
        fit = analysis.fit_from(instance=instance)
        result = _FixedResult(instance, fit)
    else:
        result = search.fit(model=model, analysis=analysis)

    # Map reconstruction onto the real-space mask grid (same as dirty/residual).
    source_image = source_image_from_fit(result.max_log_likelihood_fit)
    return result, source_image


def source_image_from_fit(fit, fov=None, pixel_scale=None):
    """
    Return the reconstruction on the real-space mask grid.

    Prefer ``fit.galaxy_image_dict`` (already matches dirty / residual shape and
    pixel scale). Fall back to interpolating mesh values onto that same grid,
    or onto ``[-fov/2, +fov/2]`` at ``pixel_scale`` if no dirty image is present.
    """
    from scipy.interpolate import griddata

    galaxy_images = getattr(fit, "galaxy_image_dict", None) or {}
    source_candidates = []
    for galaxy, image in galaxy_images.items():
        arr = np.asarray(image.native if hasattr(image, "native") else image, dtype=float)
        if np.nanmax(np.abs(arr)) <= 0.0:
            continue
        has_pix = getattr(galaxy, "pixelization", None) is not None
        source_candidates.append((has_pix, float(getattr(galaxy, "redshift", 0.0)), arr))
    if source_candidates:
        source_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return source_candidates[0][2]

    dirty = getattr(fit, "dirty_image", None)
    if dirty is not None:
        shape = tuple(dirty.shape_native)
        pixel_scales = dirty.pixel_scales
        if isinstance(pixel_scales, (list, tuple)):
            pixel_scale = float(pixel_scales[0])
        else:
            pixel_scale = float(pixel_scales)
        half_y = 0.5 * shape[0] * pixel_scale
        half_x = 0.5 * shape[1] * pixel_scale
        y = np.linspace(half_y - 0.5 * pixel_scale, -half_y + 0.5 * pixel_scale, shape[0])
        x = np.linspace(-half_x + 0.5 * pixel_scale, half_x - 0.5 * pixel_scale, shape[1])
    else:
        if fov is None or pixel_scale is None:
            raise ValueError(
                "source_image_from_fit requires fit.dirty_image or both fov and pixel_scale."
            )
        pixel_scale = float(pixel_scale)
        n = max(2, int(round(float(fov) / pixel_scale)))
        half = float(fov) / 2.0
        y = np.linspace(half - 0.5 * pixel_scale, -half + 0.5 * pixel_scale, n)
        x = np.linspace(-half + 0.5 * pixel_scale, half - 0.5 * pixel_scale, n)

    inversion = fit.inversion
    mapper = inversion.cls_list_from(cls=al.Mapper)[0]
    mesh = np.asarray(mapper.source_plane_mesh_grid)
    recon = np.asarray(inversion.reconstruction).astype(float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    points = np.column_stack([mesh[:, 0], mesh[:, 1]])
    image = griddata(points, recon, (yy, xx), method="linear", fill_value=0.0)
    return np.asarray(image, dtype=float)

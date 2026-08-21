"""Collapse multi-channel visibilities for MFS (channel-mean) datasets."""

import numpy as np


def collapse_visibilities_to_mfs(
    visibilities,
    sigma,
    *,
    sigma_mode="independent_mean",
    sigma_scale=1.0,
    weights=None,
):
    """
    Collapse channel axis to a spectrally-averaged visibility vector and noise.

    Parameters
    ----------
    visibilities
        Shape ``(n_channels, n_vis, 2)`` with real/imag on the last axis.
    sigma
        Same shape as ``visibilities``.
    sigma_mode
        ``independent_mean`` (default), ``single_channel``, or ``weights``.
    """
    visibilities = np.asarray(visibilities)
    sigma = np.asarray(sigma)

    vis_complex = visibilities[..., 0] + 1j * visibilities[..., 1]
    n_channels = vis_complex.shape[0]
    vis_mfs = vis_complex.mean(axis=0)

    if sigma_mode == "weights":
        if weights is None:
            raise ValueError("sigma_mode='weights' requires a weights array.")
        weights = np.asarray(weights)
        weight_arr = weights[..., 0] if weights.ndim == 3 else weights
        sigma_re = 1.0 / np.sqrt(weight_arr)
        sigma_im = sigma_re
    else:
        sigma_re = sigma[..., 0]
        sigma_im = sigma[..., 1]

    if sigma_mode in {"independent_mean", "weights"}:
        sigma_mfs_re = np.sqrt(np.sum(sigma_re**2, axis=0)) / n_channels
        sigma_mfs_im = np.sqrt(np.sum(sigma_im**2, axis=0)) / n_channels
    elif sigma_mode == "single_channel":
        ref = n_channels // 2
        sigma_mfs_re = sigma_re[ref] / np.sqrt(n_channels)
        sigma_mfs_im = sigma_im[ref] / np.sqrt(n_channels)
    else:
        raise ValueError(
            f"Unsupported sigma_mode: {sigma_mode!r}. "
            "Choose 'independent_mean', 'single_channel', or 'weights'."
        )

    sigma_mfs = np.stack(
        (sigma_mfs_re * sigma_scale, sigma_mfs_im * sigma_scale),
        axis=-1,
    )
    return vis_mfs, sigma_mfs


def collapse_uv_to_mfs(uv_wavelengths, uv_mode="average"):
    """Collapse per-channel UV coordinates for an MFS dataset."""
    uv_wavelengths = np.asarray(uv_wavelengths)
    if uv_mode == "reference_channel":
        ref = uv_wavelengths.shape[0] // 2
        return uv_wavelengths[ref]
    if uv_mode == "average":
        return uv_wavelengths.mean(axis=0)
    raise ValueError(
        f"Unsupported uv_mode: {uv_mode!r}. Choose 'average' or 'reference_channel'."
    )


def mfs_arrays_from(
    uv_wavelengths,
    visibilities,
    sigma,
    *,
    mfs_settings=None,
    weights=None,
):
    """Return ``(visibilities, sigma, uv_wavelengths)`` for an MFS dataset."""
    mfs_settings = mfs_settings or {}
    vis_out, sigma_out = collapse_visibilities_to_mfs(
        visibilities=visibilities,
        sigma=sigma,
        sigma_mode=mfs_settings.get("sigma_mode", "independent_mean"),
        sigma_scale=float(mfs_settings.get("sigma_scale", 1.0)),
        weights=weights,
    )
    uv_out = collapse_uv_to_mfs(
        uv_wavelengths=uv_wavelengths,
        uv_mode=mfs_settings.get("uv_mode", "average"),
    )
    return vis_out, sigma_out, uv_out


def mfs_settings_from_settings(settings):
    mfs_cfg = settings.get("mfs", {})
    return {
        "sigma_mode": mfs_cfg.get("sigma_mode", "independent_mean"),
        "sigma_scale": float(mfs_cfg.get("sigma_scale", 1.0)),
        "uv_mode": mfs_cfg.get("uv_mode", "average"),
    }

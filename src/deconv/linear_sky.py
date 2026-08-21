"""Non-negative linear sky reconstruction (brightness I, not log-brightness).

Fits ``I >= 0`` on the reconstruction grid against interferometer visibilities
via the dataset transformer, plus a discrete gradient smoothness prior on ``I``.
Intended for lower-SNR data where ``I = I0 exp(m)`` collapses toward empty sky.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.deconv.log_sky import (
    _array2d_from_slim,
    _as_complex_1d,
    _dirty_maps_from_vis_residual,
    _downsample_adjoint,
    _neighbor_edges,
    _radial_edge_mask,
    _rectangular_border_mask,
    _resolve_log_grid,
    _sigma_real,
    _smooth_energy_and_grad,
    _upsample_to_fine,
    _visibility_scale,
)

logger = logging.getLogger(__name__)


@dataclass
class LinearSkyResult:
    """Products of a linear-sky fit."""

    image: np.ndarray
    smooth: float
    chi2: float
    smooth_term: float
    hamiltonian: float
    n_iter: int
    success: bool
    message: str
    dirty_data: np.ndarray
    dirty_model: np.ndarray
    dirty_residual: np.ndarray
    residual_sigma: np.ndarray
    noise_rms: float
    edge_term: float = 0.0
    params: np.ndarray | None = None
    n_free: int = 0
    llwr: float | None = None
    log_evidence_approx: float | None = None
    noise_normalization: float | None = None
    optimize_smooth: bool = False
    smooth_init: float | None = None
    smooth_best: float | None = None
    smooth_trials: list | None = None


def _edge_zeroth_energy_and_grad_I(i_coarse, edge_mask):
    """Zeroth-order prior 0.5 * sum I^2 on edge pixels; gradient wrt I."""
    if not np.any(edge_mask):
        return 0.0, np.zeros(i_coarse.size, dtype=float)
    i_edge = i_coarse[edge_mask]
    energy = 0.5 * float(np.dot(i_edge, i_edge))
    grad_2d = np.zeros_like(i_coarse)
    grad_2d[edge_mask] = i_edge
    return energy, grad_2d.ravel()


def _resolve_linear_grid(settings, mask):
    """Reuse log-sky grid resolver with ``linear_sky`` config keys."""
    tmp = dict(settings)
    tmp["log_sky"] = dict(settings.get("linear_sky") or {})
    return _resolve_log_grid(tmp, mask)


def run_linear_sky_fit(
    settings,
    dataset,
    *,
    smooth=None,
    edge_prior=None,
    maxiter=None,
    x0=None,
):
    """
    Fit non-negative brightness ``I`` with NUFFT/DFT likelihood.

    Free parameters default to the dirty/mask grid (``linear_sky.pixel_scale=
    "mask"``). Same upsample path as log-sky when a coarser free grid is used.

    Optional ``smooth`` / ``edge_prior`` / ``maxiter`` / ``x0`` override the
    settings block (used by the outer LLWR regularization search).
    """
    import autolens as al

    cfg = dict(settings.get("linear_sky") or {})
    if smooth is None:
        smooth_cfg = cfg.get("smooth", 1.0)
        if isinstance(smooth_cfg, str):
            raise ValueError(
                "linear_sky.smooth is 'auto'; call run_sky_fit_with_reg_search "
                "or pass an explicit smooth= override."
            )
        smooth = float(smooth_cfg)
    else:
        smooth = float(smooth)
    maxiter = int(cfg.get("maxiter", 200) if maxiter is None else maxiter)
    edge_frac = float(cfg.get("edge_frac", 0.0))
    if edge_prior is None:
        if "edge_prior" in cfg:
            edge_prior = float(cfg["edge_prior"])
        elif edge_frac > 0:
            edge_prior = float(cfg.get("edge_prior_ratio", 100.0)) * smooth
        else:
            edge_prior = 0.0
    else:
        edge_prior = float(edge_prior)

    mask = dataset.mask
    kept = ~np.asarray(mask, dtype=bool)
    transformer = dataset.transformer
    data = _as_complex_1d(dataset.data)
    sigma = _sigma_real(dataset.noise_map)
    inv_var = 1.0 / np.maximum(sigma, 1e-12) ** 2

    grid = _resolve_linear_grid(settings, mask)
    log_shape = grid["log_shape"]
    fine_shape = grid["fine_shape"]
    upsample_order = grid["upsample_order"]
    dirty_native = np.asarray(dataset.dirty_image.native, dtype=float)
    dirty_coarse = _downsample_adjoint(
        np.maximum(dirty_native, 0.0), log_shape, order=upsample_order
    )
    area = (fine_shape[0] / log_shape[0]) * (fine_shape[1] / log_shape[1])
    dirty_coarse = dirty_coarse / max(area, 1.0)

    pos = np.maximum(dirty_coarse, 0.0)
    peak = float(np.max(pos)) if pos.size else 0.0
    template_coarse = np.ones(log_shape, dtype=float) if peak <= 0 else pos / peak
    template_native = _upsample_to_fine(
        template_coarse, fine_shape, order=upsample_order
    )
    template_slim = template_native[kept]

    scale = _visibility_scale(transformer, template_slim, mask, data, sigma)
    scale = abs(scale) if np.isfinite(scale) and scale != 0 else 1.0

    eps = 1e-12
    i0 = np.maximum(scale * template_coarse, 0.0)

    kept_counts = _downsample_adjoint(kept.astype(float), log_shape, order=0)
    active = kept_counts > 0

    use_edge_zeroed = bool(settings.get("use_edge_zeroed_pixels", False))
    zeroed = np.zeros(log_shape, dtype=bool)
    if use_edge_zeroed:
        zeroed = _rectangular_border_mask(log_shape) & active
        active = active & ~zeroed

    edge_mask = _radial_edge_mask(log_shape, edge_frac) & active
    n_edge = int(edge_mask.sum())
    n_zeroed = int(zeroed.sum())

    i0 = i0.copy()
    i0[kept_counts <= 0] = 0.0
    if n_zeroed > 0:
        i0[zeroed] = 0.0
    if n_edge > 0:
        i0[edge_mask] = 0.0
    i0_flat = i0.ravel()

    edge_i, edge_j = _neighbor_edges(active)
    if n_edge > 0 and edge_i.size > 0:
        edge_flat = edge_mask.ravel()
        touches_edge = edge_flat[edge_i] | edge_flat[edge_j]
        smooth_weights = np.where(touches_edge, 10.0, 1.0)
    else:
        smooth_weights = None

    # Non-negative brightness; freeze inactive / border pixels at 0.
    lower = np.zeros(i0_flat.size, dtype=float)
    upper = np.full(i0_flat.size, np.inf)
    frozen = ~active.ravel()
    lower[frozen] = i0_flat[frozen]
    upper[frozen] = i0_flat[frozen]
    bounds = list(zip(lower, upper))

    if x0 is not None:
        i0_flat = np.asarray(x0, dtype=float).ravel()
        if i0_flat.size != lower.size:
            raise ValueError(
                f"linear-sky x0 size {i0_flat.size} != free-parameter size {lower.size}"
            )
        i0_flat = np.maximum(i0_flat, 0.0)
        i0_flat[frozen] = lower[frozen]

    def brightness_from_i(i_flat):
        i_coarse = np.asarray(i_flat, dtype=float).reshape(log_shape)
        i_coarse = np.maximum(i_coarse, 0.0)
        i_native = _upsample_to_fine(i_coarse, fine_shape, order=upsample_order)
        i_native = np.where(kept, i_native, 0.0)
        return i_coarse, i_native, i_native[kept]

    def objective(i_flat):
        i_coarse, _, image_slim = brightness_from_i(i_flat)
        image = _array2d_from_slim(image_slim, mask)
        model = _as_complex_1d(transformer.visibilities_from(image=image))
        resid = model - data
        chi2 = float(np.real(np.vdot(resid * inv_var, resid)))
        smooth_term, _ = _smooth_energy_and_grad(
            i_flat, edge_i, edge_j, weights=smooth_weights
        )
        edge_term, _ = _edge_zeroth_energy_and_grad_I(i_coarse, edge_mask)
        return 0.5 * chi2 + smooth * smooth_term + edge_prior * edge_term

    def gradient(i_flat):
        i_coarse, _, image_slim = brightness_from_i(i_flat)
        image = _array2d_from_slim(image_slim, mask)
        model = _as_complex_1d(transformer.visibilities_from(image=image))
        resid = model - data
        dirty_grad = transformer.image_from(
            visibilities=al.Visibilities(visibilities=resid * inv_var)
        )
        grad_I_native = np.asarray(dirty_grad.native, dtype=float)
        grad_block = _downsample_adjoint(
            grad_I_native * kept.astype(float),
            log_shape,
            order=upsample_order,
        )
        # Linear in I: χ² gradient is the dirty residual adjoint (no I factor).
        grad_chi = grad_block.ravel()
        _, grad_s = _smooth_energy_and_grad(
            i_flat, edge_i, edge_j, weights=smooth_weights
        )
        _, grad_e = _edge_zeroth_energy_and_grad_I(i_coarse, edge_mask)
        return grad_chi + smooth * grad_s + edge_prior * grad_e

    logger.info(
        "Linear-sky fit: shape=%s (%.5g\"), dirty=%s (%.5g\"), "
        "upsample=%s, n_active=%d, smooth=%.4g, "
        "edge_zeroed=%s n_zeroed=%d, edge_frac=%.3g n_edge=%d edge_prior=%.4g, "
        "maxiter=%d",
        log_shape,
        grid["recon_pixel_scale"],
        fine_shape,
        grid["mask_pixel_scale"],
        "bilinear"
        if upsample_order == 1
        else ("none" if log_shape == fine_shape else "nearest"),
        int(active.sum()),
        smooth,
        use_edge_zeroed,
        n_zeroed,
        edge_frac,
        n_edge,
        edge_prior,
        maxiter,
    )

    opt = minimize(
        objective,
        i0_flat,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": 1e-10,
            "gtol": 1e-7,
            "maxfun": maxiter * 20,
        },
    )

    i_best = np.asarray(opt.x, dtype=float)
    i_coarse_best, image_native, image_slim = brightness_from_i(i_best)

    image = _array2d_from_slim(image_slim, mask)
    model_vis = _as_complex_1d(transformer.visibilities_from(image=image))
    resid = model_vis - data
    chi2 = float(np.real(np.vdot(resid * inv_var, resid)))
    smooth_term, _ = _smooth_energy_and_grad(
        i_best, edge_i, edge_j, weights=smooth_weights
    )
    edge_term, _ = _edge_zeroth_energy_and_grad_I(i_coarse_best, edge_mask)
    H = 0.5 * chi2 + smooth * smooth_term + edge_prior * edge_term

    dirty_data, dirty_model, dirty_residual, residual_sigma, noise_rms = (
        _dirty_maps_from_vis_residual(transformer, data, model_vis, sigma)
    )

    result = LinearSkyResult(
        image=image_native,
        smooth=smooth,
        chi2=chi2,
        smooth_term=smooth_term,
        hamiltonian=H,
        n_iter=int(opt.nit),
        success=bool(opt.success),
        message=str(opt.message),
        dirty_data=dirty_data,
        dirty_model=dirty_model,
        dirty_residual=dirty_residual,
        residual_sigma=residual_sigma,
        noise_rms=noise_rms,
        edge_term=float(edge_term),
        params=i_best,
        n_free=int(active.sum()),
        smooth_best=float(smooth),
    )
    logger.info(
        "Linear-sky done: success=%s nit=%d H=%.6g chi2=%.6g smooth=%.6g "
        "edge=%.6g peak=%.4g",
        result.success,
        result.n_iter,
        result.hamiltonian,
        result.chi2,
        result.smooth_term,
        edge_term,
        float(np.max(image_native)) if image_native.size else 0.0,
    )
    return result

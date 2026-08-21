"""Simplified log-brightness sky reconstruction (RESOLVE-inspired).

Optimizes a real field ``m`` on the real-space mask with

    I = I0 * exp(m)

against interferometer visibilities via the dataset transformer, plus a
discrete gradient smoothness prior on ``m``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class LogSkyResult:
    """Products of a log-sky fit."""

    image: np.ndarray
    m: np.ndarray
    i0: float
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


def _as_complex_1d(values):
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128).ravel()
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex128).ravel()
    return arr.astype(np.complex128).ravel()


def _sigma_real(noise_map):
    """Per-visibility real/imag noise std (Autolens stores complex noise maps)."""
    arr = np.asarray(noise_map)
    if np.iscomplexobj(arr):
        return np.maximum(np.real(arr).astype(float).ravel(), 1e-12)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return np.maximum(arr[..., 0].ravel(), 1e-12)
    return np.maximum(arr.ravel(), 1e-12)


def _array2d_from_slim(slim, mask):
    import autolens as al

    return al.Array2D(values=np.asarray(slim, dtype=float), mask=mask)


def _neighbor_edges(mask_bool_kept):
    """
    Horizontal and vertical neighbor pairs among kept (unmasked) pixels.

    ``mask_bool_kept`` is True where the pixel is free (Autolens unmasked).
    Returns ``(i_flat, j_flat)`` index pairs into the **native C-order ravel**
    of the full 2-D grid (not a packed slim vector).
    """
    ny, nx = mask_bool_kept.shape
    i_list = []
    j_list = []
    for y in range(ny):
        for x in range(nx - 1):
            if mask_bool_kept[y, x] and mask_bool_kept[y, x + 1]:
                i_list.append(y * nx + x)
                j_list.append(y * nx + (x + 1))
    for y in range(ny - 1):
        for x in range(nx):
            if mask_bool_kept[y, x] and mask_bool_kept[y + 1, x]:
                i_list.append(y * nx + x)
                j_list.append((y + 1) * nx + x)
    return np.asarray(i_list, dtype=int), np.asarray(j_list, dtype=int)


def _smooth_energy_and_grad(m_slim, edge_i, edge_j, weights=None):
    """Sum of squared neighbor differences and its gradient."""
    if edge_i.size == 0:
        return 0.0, np.zeros_like(m_slim)
    diff = m_slim[edge_i] - m_slim[edge_j]
    if weights is None:
        w = 1.0
        energy = 0.5 * float(np.dot(diff, diff))
        grad = np.zeros_like(m_slim)
        np.add.at(grad, edge_i, diff)
        np.add.at(grad, edge_j, -diff)
        return energy, grad
    w = np.asarray(weights, dtype=float)
    wdiff = w * diff
    energy = 0.5 * float(np.dot(diff, wdiff))
    grad = np.zeros_like(m_slim)
    np.add.at(grad, edge_i, wdiff)
    np.add.at(grad, edge_j, -wdiff)
    return energy, grad


def _radial_edge_mask(shape, edge_frac):
    """
    Boolean mask of pixels in the outer ``edge_frac`` of radius.

    Radius is normalized by the inscribed-circle half-width of the grid
    (circular FOV), so ``edge_frac=0.1`` selects the annulus
    ``0.9 R_disk <= r <= R_disk``. Corners outside the disk are excluded
    so the prior does not clamp a huge fraction of a square grid.
    """
    edge_frac = float(edge_frac)
    if edge_frac <= 0.0:
        return np.zeros(shape, dtype=bool)
    if edge_frac >= 1.0:
        return np.ones(shape, dtype=bool)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = 0.5 * (ny - 1)
    cx = 0.5 * (nx - 1)
    r = np.hypot(yy - cy, xx - cx)
    r_disk = 0.5 * float(min(ny, nx) - 1)
    if r_disk <= 0:
        return np.zeros(shape, dtype=bool)
    return (r <= r_disk + 1e-9) & (r >= (1.0 - edge_frac) * r_disk)


def _rectangular_border_mask(shape):
    """
    Boolean mask of the rectangular mesh border (Autolens edge-zeroed pixels).

    Matches ``autoarray`` ``rectangular_edge_pixel_list_from``: top/bottom
    rows and left/right columns of the coarse log grid.
    """
    ny, nx = (int(shape[0]), int(shape[1]))
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _edge_zeroth_energy_and_grad(i_coarse, edge_mask):
    """
    Zeroth-order prior 0.5 * sum I^2 on edge pixels (pull brightness to 0).

    Gradient wrt m uses dI/dm = I, so d(0.5 I^2)/dm = I^2 on edge pixels.
    """
    if not np.any(edge_mask):
        return 0.0, np.zeros(i_coarse.size, dtype=float)
    i_edge = i_coarse[edge_mask]
    energy = 0.5 * float(np.dot(i_edge, i_edge))
    grad_2d = np.zeros_like(i_coarse)
    grad_2d[edge_mask] = i_edge * i_edge
    return energy, grad_2d.ravel()


def _visibility_scale(transformer, image_slim, mask, data, sigma):
    """Least-squares scalar aligning R(image) to the data."""
    image = _array2d_from_slim(image_slim, mask)
    model = _as_complex_1d(transformer.visibilities_from(image=image))
    d = _as_complex_1d(data)
    w = 1.0 / np.maximum(sigma, 1e-12) ** 2
    num = np.real(np.vdot(model * w, d))
    den = np.real(np.vdot(model * w, model))
    if not np.isfinite(den) or den <= 0:
        return 1.0
    return float(num / den)


def _dirty_maps_from_vis_residual(transformer, data, model_vis, sigma):
    """Dirty data/model/residual and noise RMS for plotting."""
    import autolens as al
    from src.deconv.plots import robust_rms

    dirty_data = np.asarray(
        transformer.image_from(visibilities=al.Visibilities(visibilities=data)).native,
        dtype=float,
    )
    dirty_model = np.asarray(
        transformer.image_from(
            visibilities=al.Visibilities(visibilities=model_vis)
        ).native,
        dtype=float,
    )
    dirty_residual = dirty_data - dirty_model

    rng = np.random.default_rng(0)
    rms_list = []
    for _ in range(8):
        noise_vis = (
            sigma
            / np.sqrt(2.0)
            * (rng.standard_normal(sigma.shape) + 1j * rng.standard_normal(sigma.shape))
        )
        dirty = transformer.image_from(
            visibilities=al.Visibilities(visibilities=noise_vis)
        )
        arr = np.asarray(dirty.native if hasattr(dirty, "native") else dirty, dtype=float)
        rms_list.append(robust_rms(arr))
    noise_rms = float(np.median(rms_list))
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        noise_rms = robust_rms(dirty_residual)
    residual_sigma = dirty_residual / noise_rms
    return dirty_data, dirty_model, dirty_residual, residual_sigma, noise_rms


def _resolve_log_grid(settings, mask):
    """
    Resolve free log-field shape, pixel scale, and upsample order.

    ``log_sky.pixel_scale``:
      - ``\"mask\"`` / ``\"auto\"`` (default): same grid as the dirty image
      - ``\"nyquist\"``: ``0.5 λ/b_max`` across the mask FOV
      - float (arcsec): explicit free-grid pixel scale

    If the free-grid scale is coarser than Nyquist, brightness is bilinearly
    upsampled onto the dirty/mask grid. Same-shape grids need no upsample.
    Optional ``log_sky.mesh_shape`` still overrides the free shape directly
    (upsample order follows the implied pixel scale vs Nyquist).
    """
    log_cfg = settings.get("log_sky") or {}
    fine_shape = tuple(int(v) for v in mask.shape_native)
    mask_pixel_scale = float(settings["mask_pixel_scale"])
    mask_fov = float(settings.get("mask_fov", fine_shape[0] * mask_pixel_scale))
    nyquist = float(settings.get("nyquist_pixel_scale", mask_pixel_scale))

    mesh_shape = log_cfg.get("mesh_shape", None)
    if mesh_shape is not None:
        log_shape = tuple(int(v) for v in mesh_shape)
        if len(log_shape) != 2 or log_shape[0] < 2 or log_shape[1] < 2:
            raise ValueError(
                f"log_sky.mesh_shape must be [ny, nx] with ny,nx >= 2; got {mesh_shape!r}"
            )
        recon_pixel_scale = mask_fov / log_shape[0]
    else:
        ps_cfg = log_cfg.get("pixel_scale", "mask")
        if ps_cfg is None or (
            isinstance(ps_cfg, str) and ps_cfg.lower() in {"mask", "auto"}
        ):
            log_shape = fine_shape
            recon_pixel_scale = mask_pixel_scale
        elif isinstance(ps_cfg, str) and ps_cfg.lower() == "nyquist":
            recon_pixel_scale = nyquist
            n = max(2, int(round(mask_fov / recon_pixel_scale)))
            log_shape = (n, n)
            recon_pixel_scale = mask_fov / n
        else:
            recon_pixel_scale = float(ps_cfg)
            if not np.isfinite(recon_pixel_scale) or recon_pixel_scale <= 0.0:
                raise ValueError(
                    f"log_sky.pixel_scale must be positive; got {ps_cfg!r}"
                )
            if recon_pixel_scale < mask_pixel_scale * (1.0 - 1e-9):
                raise ValueError(
                    f"log_sky.pixel_scale={recon_pixel_scale}\" is finer than the "
                    f"dirty/mask scale {mask_pixel_scale}\". Use 'mask' or a "
                    "coarser scale."
                )
            n = max(2, int(round(mask_fov / recon_pixel_scale)))
            log_shape = (n, n)
            recon_pixel_scale = mask_fov / n

    if log_shape == fine_shape:
        upsample_order = 0
    elif recon_pixel_scale > nyquist * (1.0 + 1e-9):
        # Coarser than Nyquist → bilinear onto the dirty grid.
        upsample_order = 1
    else:
        upsample_order = 0

    return {
        "log_shape": log_shape,
        "fine_shape": fine_shape,
        "recon_pixel_scale": float(recon_pixel_scale),
        "mask_pixel_scale": mask_pixel_scale,
        "nyquist_pixel_scale": nyquist,
        "upsample_order": int(upsample_order),
    }


def _block_factors(fine_shape, coarse_shape):
    fy, fx = fine_shape
    cy, cx = coarse_shape
    if fy % cy != 0 or fx % cx != 0:
        return None
    return fy // cy, fx // cx


def _upsample_to_fine(coarse, fine_shape, *, order=0):
    """Upsample a free-grid image onto the dirty/mask grid."""
    coarse = np.asarray(coarse, dtype=float)
    fine_shape = (int(fine_shape[0]), int(fine_shape[1]))
    if tuple(coarse.shape) == fine_shape:
        return coarse.copy()
    order = int(order)
    if order == 0:
        factors = _block_factors(fine_shape, coarse.shape)
        if factors is not None:
            by, bx = factors
            return np.repeat(np.repeat(coarse, by, axis=0), bx, axis=1)
        from scipy.ndimage import zoom

        return zoom(
            coarse,
            (fine_shape[0] / coarse.shape[0], fine_shape[1] / coarse.shape[1]),
            order=0,
            mode="nearest",
        )
    if order == 1:
        from scipy.ndimage import zoom

        return zoom(
            coarse,
            (fine_shape[0] / coarse.shape[0], fine_shape[1] / coarse.shape[1]),
            order=1,
            mode="nearest",
        )
    raise ValueError(f"Unsupported upsample order {order}; use 0 or 1")


def _downsample_adjoint(fine, coarse_shape, *, order=0):
    """
    Adjoint of ``_upsample_to_fine`` (exact for NN block-sum; approximate
    area-weighted bilinear zoom otherwise).
    """
    fine = np.asarray(fine, dtype=float)
    coarse_shape = (int(coarse_shape[0]), int(coarse_shape[1]))
    if tuple(fine.shape) == coarse_shape:
        return fine.copy()
    order = int(order)
    if order == 0:
        factors = _block_factors(fine.shape, coarse_shape)
        if factors is not None:
            by, bx = factors
            cy, cx = coarse_shape
            return fine.reshape(cy, by, cx, bx).sum(axis=(1, 3))
        from scipy.ndimage import zoom

        area = (fine.shape[0] / coarse_shape[0]) * (fine.shape[1] / coarse_shape[1])
        return (
            zoom(
                fine,
                (coarse_shape[0] / fine.shape[0], coarse_shape[1] / fine.shape[1]),
                order=0,
                mode="nearest",
            )
            * area
        )
    if order == 1:
        from scipy.ndimage import zoom

        area = (fine.shape[0] / coarse_shape[0]) * (fine.shape[1] / coarse_shape[1])
        return (
            zoom(
                fine,
                (coarse_shape[0] / fine.shape[0], coarse_shape[1] / fine.shape[1]),
                order=1,
                mode="nearest",
            )
            * area
        )
    raise ValueError(f"Unsupported downsample order {order}; use 0 or 1")


def _downsample_sum(fine, coarse_shape):
    """Backward-compatible NN / area downsample (order 0 adjoint). """
    return _downsample_adjoint(fine, coarse_shape, order=0)


def run_log_sky_fit(
    settings,
    dataset,
    *,
    smooth=None,
    edge_prior=None,
    maxiter=None,
    x0=None,
):
    """
    Fit ``I = I0 * exp(m)`` with NUFFT/DFT likelihood.

    Free parameters default to the dirty/mask grid (``log_sky.pixel_scale=
    "mask"``). A coarser ``log_sky.pixel_scale`` (arcsec) is allowed; if it is
    larger than Nyquist, brightness is bilinearly upsampled onto the dirty
    grid for the visibility forward model.

    Optional ``smooth`` / ``edge_prior`` / ``maxiter`` / ``x0`` override the
    settings block (used by the outer LLWR regularization search).
    """
    import autolens as al

    cfg = dict(settings.get("log_sky") or {})
    if smooth is None:
        smooth_cfg = cfg.get("smooth", 1.0)
        if isinstance(smooth_cfg, str):
            raise ValueError(
                "log_sky.smooth is 'auto'; call run_sky_fit_with_reg_search "
                "or pass an explicit smooth= override."
            )
        smooth = float(smooth_cfg)
    else:
        smooth = float(smooth)
    maxiter = int(cfg.get("maxiter", 200) if maxiter is None else maxiter)
    i0_cfg = cfg.get("i0", "auto")
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

    grid = _resolve_log_grid(settings, mask)
    log_shape = grid["log_shape"]
    fine_shape = grid["fine_shape"]
    upsample_order = grid["upsample_order"]
    dirty_native = np.asarray(dataset.dirty_image.native, dtype=float)
    dirty_coarse = _downsample_adjoint(
        np.maximum(dirty_native, 0.0), log_shape, order=upsample_order
    )
    # Convert block / area sums back to mean-like template.
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

    if i0_cfg == "auto" or i0_cfg is None:
        i0 = float(scale)
    else:
        i0 = float(i0_cfg)

    eps = 1e-6
    target_coarse = np.maximum(scale * template_coarse, eps * abs(i0))
    m0 = np.log(target_coarse / abs(i0))

    # Coarse cells that cover no unmasked fine pixels cannot affect χ² —
    # freeze them faint so they do not pollute smoothness / edge priors.
    kept_counts = _downsample_adjoint(
        kept.astype(float), log_shape, order=0
    )
    active = kept_counts > 0

    # Autolens-style edge-zeroed pixels: hard-freeze the rectangular border.
    use_edge_zeroed = bool(settings.get("use_edge_zeroed_pixels", False))
    zeroed = np.zeros(log_shape, dtype=bool)
    if use_edge_zeroed:
        zeroed = _rectangular_border_mask(log_shape) & active
        active = active & ~zeroed

    edge_mask = _radial_edge_mask(log_shape, edge_frac) & active
    n_edge = int(edge_mask.sum())
    n_zeroed = int(zeroed.sum())
    m0 = m0.copy()
    m0[kept_counts <= 0] = np.log(eps)
    if n_zeroed > 0:
        m0[zeroed] = np.log(eps)
    if n_edge > 0:
        m0[edge_mask] = np.log(eps)
    m0 = m0.ravel()

    edge_i, edge_j = _neighbor_edges(active)
    # Mild boost of gradient smoothness on bonds that touch an edge-prior pixel.
    if n_edge > 0 and edge_i.size > 0:
        edge_flat = edge_mask.ravel()
        touches_edge = edge_flat[edge_i] | edge_flat[edge_j]
        smooth_weights = np.where(touches_edge, 10.0, 1.0)
    else:
        smooth_weights = None

    # Freeze inactive + edge-zeroed coarse pixels via equal lower/upper bounds.
    lower = np.full(m0.size, -20.0)
    upper = np.full(m0.size, 20.0)
    frozen = ~active.ravel()
    lower[frozen] = m0[frozen]
    upper[frozen] = m0[frozen]
    bounds = list(zip(lower, upper))

    def brightness_from_m(m_flat):
        m_2d = np.asarray(m_flat, dtype=float).reshape(log_shape)
        i_coarse = abs(i0) * np.exp(np.clip(m_2d, -20.0, 20.0))
        i_native = _upsample_to_fine(i_coarse, fine_shape, order=upsample_order)
        i_native = np.where(kept, i_native, 0.0)
        return i_coarse, i_native, i_native[kept]

    def objective(m_flat):
        i_coarse, _, image_slim = brightness_from_m(m_flat)
        image = _array2d_from_slim(image_slim, mask)
        model = _as_complex_1d(transformer.visibilities_from(image=image))
        resid = model - data
        chi2 = float(np.real(np.vdot(resid * inv_var, resid)))
        smooth_term, _ = _smooth_energy_and_grad(
            m_flat, edge_i, edge_j, weights=smooth_weights
        )
        edge_term, _ = _edge_zeroth_energy_and_grad(i_coarse, edge_mask)
        return 0.5 * chi2 + smooth * smooth_term + edge_prior * edge_term

    def gradient(m_flat):
        i_coarse, i_native, image_slim = brightness_from_m(m_flat)
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
        grad_chi = (i_coarse * grad_block).ravel()
        _, grad_s = _smooth_energy_and_grad(
            m_flat, edge_i, edge_j, weights=smooth_weights
        )
        _, grad_e = _edge_zeroth_energy_and_grad(i_coarse, edge_mask)
        return grad_chi + smooth * grad_s + edge_prior * grad_e

    if x0 is not None:
        m0 = np.asarray(x0, dtype=float).ravel()
        if m0.size != lower.size:
            raise ValueError(
                f"log-sky x0 size {m0.size} != free-parameter size {lower.size}"
            )

    logger.info(
        "Log-sky fit: log_shape=%s (%.5g\"), dirty=%s (%.5g\"), "
        "upsample=%s, n_active=%d, I0=%.4g, smooth=%.4g, "
        "edge_zeroed=%s n_zeroed=%d, edge_frac=%.3g n_edge=%d edge_prior=%.4g, "
        "maxiter=%d",
        log_shape,
        grid["recon_pixel_scale"],
        fine_shape,
        grid["mask_pixel_scale"],
        "bilinear" if upsample_order == 1 else ("none" if log_shape == fine_shape else "nearest"),
        int(active.sum()),
        i0,
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
        m0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxfun": maxiter * 20},
    )

    m_best = np.asarray(opt.x, dtype=float)
    i_coarse_best, image_native, image_slim = brightness_from_m(m_best)

    image = _array2d_from_slim(image_slim, mask)
    model_vis = _as_complex_1d(transformer.visibilities_from(image=image))
    resid = model_vis - data
    chi2 = float(np.real(np.vdot(resid * inv_var, resid)))
    smooth_term, _ = _smooth_energy_and_grad(
        m_best, edge_i, edge_j, weights=smooth_weights
    )
    edge_term, _ = _edge_zeroth_energy_and_grad(i_coarse_best, edge_mask)
    H = 0.5 * chi2 + smooth * smooth_term + edge_prior * edge_term

    dirty_data, dirty_model, dirty_residual, residual_sigma, noise_rms = (
        _dirty_maps_from_vis_residual(transformer, data, model_vis, sigma)
    )

    m_native = _upsample_to_fine(
        m_best.reshape(log_shape), fine_shape, order=upsample_order
    )
    m_native = np.where(kept, m_native, 0.0)

    result = LogSkyResult(
        image=image_native,
        m=m_native,
        i0=float(abs(i0)),
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
        params=m_best,
        n_free=int(active.sum()),
        smooth_best=float(smooth),
    )
    logger.info(
        "Log-sky done: success=%s nit=%d H=%.6g chi2=%.6g smooth=%.6g "
        "edge=%.6g peak=%.4g",
        result.success,
        result.n_iter,
        result.hamiltonian,
        result.chi2,
        result.smooth_term,
        edge_term,
        float(np.nanmax(result.image)),
    )
    return result

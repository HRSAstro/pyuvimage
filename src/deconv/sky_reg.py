"""SNR routing and Autolens-style outer regularization search for sky fits.

Outer ``smooth`` search starts high and relaxes until the fit is noise-consistent
(``chi2 ≈ n_data``), recording Autolens ``log_likelihood_with_regularization``
(LLWR) and an evidence proxy for diagnostics:

    LLWR = -0.5 * (chi2 + 2*smooth*smooth_term + 2*edge_prior*edge_term
                   + noise_normalization)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DirtySNR:
    """Dirty-image peak / noise RMS estimate."""

    peak: float
    noise_rms: float
    snr: float


@dataclass
class SmoothSearchResult:
    """Outer ``smooth`` search diagnostics."""

    smooth_init: float
    smooth_best: float
    llwr_best: float
    trials: list = field(default_factory=list)  # dicts: smooth, llwr, chi2, ...
    n_evals: int = 0


def noise_normalization_from_sigma(sigma):
    """Autolens-style Σ ln(2π σ²) for real and imag parts (complex visibilities)."""
    sigma = np.asarray(sigma, dtype=float).ravel()
    sigma = np.maximum(sigma, 1e-12)
    # Real + imag each contribute ln(2π σ²).
    return float(2.0 * np.sum(np.log(2.0 * np.pi * sigma * sigma)))


def llwr_from_terms(chi2, smooth_term, edge_term, smooth, edge_prior, noise_norm=0.0):
    """
    Autolens log_likelihood_with_regularization for sky fits.

    With sky energy terms defined so Hamiltonian
    ``H = 0.5*chi2 + smooth*smooth_term + edge_prior*edge_term``,
    this is ``-H - 0.5*noise_norm``.
    """
    chi2 = float(chi2)
    smooth_term = float(smooth_term)
    edge_term = float(edge_term)
    smooth = float(smooth)
    edge_prior = float(edge_prior)
    noise_norm = float(noise_norm)
    return -0.5 * (
        chi2
        + 2.0 * smooth * smooth_term
        + 2.0 * edge_prior * edge_term
        + noise_norm
    )


def llwr_from_hamiltonian(hamiltonian, noise_norm=0.0):
    """LLWR from sky Hamiltonian (noise term optional)."""
    return -float(hamiltonian) - 0.5 * float(noise_norm)


def log_evidence_approx_from_terms(
    chi2,
    smooth_term,
    edge_term,
    smooth,
    edge_prior,
    n_free,
    noise_norm=0.0,
    n_data=None,
):
    """
    Autolens-like evidence proxy for nonlinear sky maps.

    Starts from ``log_likelihood_with_regularization`` and adds a leading
    Occam piece ``+0.5 n_eff log(smooth)`` standing in for ``+0.5 ln det(H)``
    with ``H ∝ smooth``. ``n_eff`` is capped by the visibility degrees of
    freedom (``n_data``, typically ``2 * n_vis``) so a fine sky grid does not
    force ``smooth → ∞``.
    """
    llwr = llwr_from_terms(
        chi2, smooth_term, edge_term, smooth, edge_prior, noise_norm
    )
    n_eff = max(int(n_free), 1)
    if n_data is not None:
        n_eff = min(n_eff, max(int(n_data), 10))
    smooth = max(float(smooth), 1e-30)
    return llwr + 0.5 * n_eff * float(np.log(smooth))


def estimate_dirty_snr(dataset, *, n_noise=8, seed=0):
    """
    Dirty-image peak / noise RMS.

    Noise RMS is the median robust RMS of dirty images from noise-only
    visibility realizations (same recipe as sky residual plots).
    """
    import autolens as al

    from src.deconv.log_sky import _as_complex_1d, _sigma_real
    from src.deconv.plots import robust_rms

    dirty = np.asarray(dataset.dirty_image.native, dtype=float)
    peak = float(np.nanmax(np.abs(dirty))) if dirty.size else 0.0

    transformer = dataset.transformer
    sigma = _sigma_real(dataset.noise_map)
    rng = np.random.default_rng(seed)
    rms_list = []
    for _ in range(int(n_noise)):
        noise_vis = (
            sigma
            / np.sqrt(2.0)
            * (rng.standard_normal(sigma.shape) + 1j * rng.standard_normal(sigma.shape))
        )
        dirty_n = transformer.image_from(
            visibilities=al.Visibilities(visibilities=_as_complex_1d(noise_vis))
        )
        arr = np.asarray(
            dirty_n.native if hasattr(dirty_n, "native") else dirty_n, dtype=float
        )
        rms_list.append(robust_rms(arr))
    noise_rms = float(np.median(rms_list)) if rms_list else 0.0
    if not np.isfinite(noise_rms) or noise_rms <= 0.0:
        noise_rms = float(robust_rms(dirty)) if dirty.size else 1.0
        if not np.isfinite(noise_rms) or noise_rms <= 0.0:
            noise_rms = 1.0
    snr = peak / noise_rms if noise_rms > 0 else 0.0
    return DirtySNR(peak=peak, noise_rms=noise_rms, snr=float(snr))


def choose_sky_reconstructor(snr, threshold):
    """High SNR → log_sky; low SNR → linear_sky."""
    if float(snr) >= float(threshold):
        return "log_sky"
    return "linear_sky"


def _sky_cfg_key(kind):
    if kind not in {"log_sky", "linear_sky"}:
        raise ValueError(f"Unsupported sky kind: {kind!r}")
    return kind


def resolve_sky_reg_options(settings, kind):
    """
    Resolve whether to optimize ``smooth`` and the search hyperparameters.

    Returns dict with keys:
    optimize, smooth (fixed float or None), smooth_init, smooth_bounds,
    edge_prior_ratio, maxiter, maxiter_trial, edge_frac.
    """
    key = _sky_cfg_key(kind)
    cfg = dict(settings.get(key) or {})
    smooth_cfg = cfg.get("smooth", 1.0)
    optimize = bool(cfg.get("optimize_smooth", False))
    if isinstance(smooth_cfg, str) and str(smooth_cfg).lower() == "auto":
        optimize = True
        smooth_fixed = None
    else:
        smooth_fixed = float(smooth_cfg)

    defaults_init = {"log_sky": 1.0e4, "linear_sky": 1.0e6}
    smooth_init = float(cfg.get("smooth_init", defaults_init[kind]))
    bounds = cfg.get("smooth_bounds", [1.0e-2, 1.0e10])
    if len(bounds) != 2:
        raise ValueError(f"{key}.smooth_bounds must be [min, max]; got {bounds!r}")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (lo > 0 and hi > lo):
        raise ValueError(f"{key}.smooth_bounds must satisfy 0 < min < max; got {bounds!r}")

    edge_prior_ratio = float(cfg.get("edge_prior_ratio", 100.0))
    if edge_prior_ratio < 0.0:
        raise ValueError(f"{key}.edge_prior_ratio must be >= 0")

    maxiter = int(cfg.get("maxiter", 200))
    maxiter_trial = int(cfg.get("maxiter_trial", max(100, min(maxiter, maxiter // 5 or 100))))
    edge_frac = float(cfg.get("edge_frac", 0.0))

    return {
        "optimize": optimize,
        "smooth": smooth_fixed,
        "smooth_init": smooth_init,
        "smooth_bounds": (lo, hi),
        "edge_prior_ratio": edge_prior_ratio,
        "maxiter": maxiter,
        "maxiter_trial": maxiter_trial,
        "edge_frac": edge_frac,
    }


def _fom_from_result(result, smooth, edge_prior, noise_norm, n_data=None):
    """Outer figure of merit (evidence approximation)."""
    edge_term = float(getattr(result, "edge_term", 0.0))
    n_free = int(getattr(result, "n_free", 0) or 0)
    if n_free <= 0 and getattr(result, "params", None) is not None:
        n_free = int(np.asarray(result.params).size)
    if n_free <= 0:
        n_free = 1
    return log_evidence_approx_from_terms(
        result.chi2,
        result.smooth_term,
        edge_term,
        smooth,
        edge_prior,
        n_free,
        noise_norm,
        n_data=n_data,
    )


def _attach_search_meta(result, search: SmoothSearchResult, edge_term, llwr, noise_norm, fom):
    """Copy result with regularization-search diagnostics attached."""
    updates = {
        "edge_term": float(edge_term),
        "llwr": float(llwr),
        "log_evidence_approx": float(fom),
        "noise_normalization": float(noise_norm),
        "smooth_init": float(search.smooth_init),
        "smooth_best": float(search.smooth_best),
        "smooth_trials": list(search.trials),
        "optimize_smooth": True,
    }
    try:
        return replace(result, **updates)
    except TypeError:
        for k, v in updates.items():
            object.__setattr__(result, k, v)
        return result


def optimize_smooth(
    run_inner,
    *,
    smooth_init,
    smooth_bounds,
    edge_prior_ratio,
    edge_frac,
    maxiter,
    maxiter_trial,
    noise_norm=0.0,
    n_data=None,
):
    """
    Two-step outer search for ``smooth``:

    1. Fit at high ``smooth_init`` (warm start).
    2. Scan log-spaced ``smooth`` from high → low; pick the largest value
       whose ``chi2`` is noise-consistent (``chi2 <= chi2_target``), else the
       trial with ``chi2`` closest to ``n_data``. Evidence/LLWR are recorded
       for diagnostics.

    This matches the intended “start high, then relax regularization” workflow
    and avoids bare-LLWR collapse without a full ``ln det(F+H)`` term.
    """
    lo, hi = smooth_bounds
    smooth0 = float(np.clip(smooth_init, lo, hi))
    edge0 = float(edge_prior_ratio * smooth0) if edge_frac > 0 else 0.0
    n_data = int(n_data) if n_data is not None else None
    chi2_target = float(n_data) if n_data is not None else None

    logger.info(
        "Smooth search step 1: high-smooth init=%.4g edge_prior=%.4g maxiter=%d",
        smooth0,
        edge0,
        maxiter_trial,
    )
    warm = run_inner(smooth0, edge0, maxiter_trial, None)
    edge_term0 = float(getattr(warm, "edge_term", 0.0))
    llwr0 = llwr_from_terms(
        warm.chi2, warm.smooth_term, edge_term0, smooth0, edge0, noise_norm
    )
    fom0 = _fom_from_result(warm, smooth0, edge0, noise_norm, n_data=n_data)
    trials = [
        {
            "smooth": smooth0,
            "fom": fom0,
            "llwr": llwr0,
            "chi2": float(warm.chi2),
            "smooth_term": float(warm.smooth_term),
            "edge_term": edge_term0,
            "peak": float(np.nanmax(warm.image)) if getattr(warm, "image", None) is not None else 0.0,
        }
    ]
    x_warm = np.asarray(warm.params, dtype=float)

    # Log-spaced scan from high → low (include init and bounds).
    n_grid = 12
    grid = np.unique(
        np.concatenate(
            [
                [smooth0],
                np.logspace(np.log10(hi), np.log10(lo), n_grid),
            ]
        )
    )
    grid = grid[(grid >= lo) & (grid <= hi)]
    grid = np.sort(grid)[::-1]  # high → low

    logger.info(
        "Smooth search step 2: scan %d smooth values high→low "
        "(chi2_target=%s, n_data=%s)",
        len(grid),
        f"{chi2_target:.4g}" if chi2_target is not None else "None",
        n_data,
    )

    best = {
        "smooth": smooth0,
        "chi2": float(warm.chi2),
        "fom": fom0,
        "llwr": llwr0,
        "result": warm,
        "x": x_warm,
        "noise_ok": False,
    }
    x_curr = x_warm
    evaluated = []
    for smooth in grid:
        smooth = float(smooth)
        edge_prior = float(edge_prior_ratio * smooth) if edge_frac > 0 else 0.0
        res = run_inner(smooth, edge_prior, maxiter_trial, x_curr)
        edge_term = float(getattr(res, "edge_term", 0.0))
        llwr = llwr_from_terms(
            res.chi2, res.smooth_term, edge_term, smooth, edge_prior, noise_norm
        )
        fom = _fom_from_result(res, smooth, edge_prior, noise_norm, n_data=n_data)
        chi2 = float(res.chi2)
        entry = {
            "smooth": smooth,
            "fom": fom,
            "llwr": llwr,
            "chi2": chi2,
            "smooth_term": float(res.smooth_term),
            "edge_term": edge_term,
            "peak": float(np.nanmax(res.image)) if getattr(res, "image", None) is not None else 0.0,
            "result": res,
            "x": np.asarray(res.params, dtype=float),
        }
        trials.append({k: v for k, v in entry.items() if k not in {"result", "x"}})
        evaluated.append(entry)
        x_curr = entry["x"]

        noise_ok = chi2_target is not None and chi2 <= 1.2 * chi2_target
        if noise_ok and not best["noise_ok"]:
            # First (largest, since we scan high→low) smooth that meets the target.
            best.update(
                smooth=smooth,
                chi2=chi2,
                fom=fom,
                llwr=llwr,
                result=res,
                x=entry["x"],
                noise_ok=True,
            )

    if not best["noise_ok"] and evaluated:
        # Never reached the noise target (often under-iterated trials).
        # Prefer the most-regularized solution among those near the best chi2.
        chi2_floor = min(e["chi2"] for e in evaluated)
        near = [e for e in evaluated if e["chi2"] <= 1.25 * chi2_floor]
        pick = max(near, key=lambda e: e["smooth"])
        best.update(
            smooth=pick["smooth"],
            chi2=pick["chi2"],
            fom=pick["fom"],
            llwr=pick["llwr"],
            result=pick["result"],
            x=pick["x"],
            noise_ok=False,
        )
        logger.info(
            "Smooth search: no trial met chi2_target; chose most-regularized "
            "near-best chi2 (smooth=%.4g chi2=%.4g; best_chi2=%.4g)",
            pick["smooth"],
            pick["chi2"],
            chi2_floor,
        )

    smooth_best = float(best["smooth"])
    edge_final = float(edge_prior_ratio * smooth_best) if edge_frac > 0 else 0.0
    logger.info(
        "Smooth search step 3: final fit smooth=%.4g edge_prior=%.4g maxiter=%d "
        "chi2=%.4g target=%s",
        smooth_best,
        edge_final,
        maxiter,
        best["chi2"],
        f"{chi2_target:.4g}" if chi2_target is not None else "None",
    )
    final = run_inner(smooth_best, edge_final, maxiter, best["x"])
    edge_term = float(getattr(final, "edge_term", 0.0))
    llwr_final = llwr_from_terms(
        final.chi2, final.smooth_term, edge_term, smooth_best, edge_final, noise_norm
    )
    fom_final = _fom_from_result(
        final, smooth_best, edge_final, noise_norm, n_data=n_data
    )
    trials.append(
        {
            "smooth": smooth_best,
            "fom": fom_final,
            "llwr": llwr_final,
            "chi2": float(final.chi2),
            "smooth_term": float(final.smooth_term),
            "edge_term": edge_term,
            "peak": float(np.nanmax(final.image)) if getattr(final, "image", None) is not None else 0.0,
            "final": True,
        }
    )

    search = SmoothSearchResult(
        smooth_init=smooth0,
        smooth_best=smooth_best,
        llwr_best=llwr_final,
        trials=trials,
        n_evals=len(trials),
    )
    return _attach_search_meta(
        final, search, edge_term, llwr_final, noise_norm, fom_final
    )


def run_sky_fit_with_reg_search(settings, dataset, kind):
    """
    Run log-sky or linear-sky, optionally with outer ``smooth`` LLWR search.
    """
    from src.deconv.log_sky import _sigma_real, run_log_sky_fit
    from src.deconv.linear_sky import run_linear_sky_fit

    opts = resolve_sky_reg_options(settings, kind)
    sigma = _sigma_real(dataset.noise_map)
    noise_norm = noise_normalization_from_sigma(sigma)
    n_data = 2 * int(np.asarray(sigma).size)  # real + imag

    if kind == "log_sky":
        runner = run_log_sky_fit
    else:
        runner = run_linear_sky_fit

    if not opts["optimize"]:
        result = runner(settings, dataset)
        edge_term = float(getattr(result, "edge_term", 0.0))
        llwr = llwr_from_hamiltonian(result.hamiltonian, noise_norm)
        cfg = dict(settings.get(kind) or {})
        edge_frac = float(cfg.get("edge_frac", 0.0))
        if edge_frac > 0.0:
            if "edge_prior" in cfg:
                edge_prior = float(cfg["edge_prior"])
            else:
                edge_prior = float(cfg.get("edge_prior_ratio", 100.0)) * float(
                    result.smooth
                )
        else:
            edge_prior = 0.0
        fom = log_evidence_approx_from_terms(
            result.chi2,
            result.smooth_term,
            edge_term,
            result.smooth,
            edge_prior,
            int(getattr(result, "n_free", 0) or 1),
            noise_norm,
            n_data=n_data,
        )
        try:
            return replace(
                result,
                edge_term=edge_term,
                llwr=llwr,
                log_evidence_approx=fom,
                noise_normalization=noise_norm,
                optimize_smooth=False,
                smooth_init=None,
                smooth_best=float(result.smooth),
                smooth_trials=[],
            )
        except TypeError:
            object.__setattr__(result, "llwr", llwr)
            object.__setattr__(result, "optimize_smooth", False)
            return result

    def run_inner(smooth, edge_prior, maxiter, x0):
        return runner(
            settings,
            dataset,
            smooth=float(smooth),
            edge_prior=float(edge_prior),
            maxiter=int(maxiter),
            x0=x0,
        )

    return optimize_smooth(
        run_inner,
        smooth_init=opts["smooth_init"],
        smooth_bounds=opts["smooth_bounds"],
        edge_prior_ratio=opts["edge_prior_ratio"],
        edge_frac=opts["edge_frac"],
        maxiter=opts["maxiter"],
        maxiter_trial=opts["maxiter_trial"],
        noise_norm=noise_norm,
        n_data=n_data,
    )

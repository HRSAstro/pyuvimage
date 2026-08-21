"""Core forward-modelling machinery built on PyAutoGalaxy.

The model is a freeform source on a uniform rectangular mesh, fitted to the
visibilities by a regularised linear inversion (normal equations
``(F + lam*H) s = D``).  The regularisation strength ``lam`` is chosen by the
Morozov discrepancy principle by default -- the strongest smoothing that
still fits the data to the noise level (chi^2 ~= N) -- so the model can
neither overfit the noise nor be needlessly smooth.  Bayesian-evidence
maximisation is available as an alternative criterion.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

import autogalaxy as ag

from .grids import ImageGeometry

logger = logging.getLogger("pyuvimage")

# The DFT is exact but O(n_vis * n_pix); beyond this many visibilities the
# NUFFT (JAX-based) is required for sane runtimes.
DFT_MAX_VIS = 20_000

REGULARIZATIONS = ("gibbs", "adaptive", "matern", "gaussian", "exponential", "constant")

# Kernel (Gaussian-process) source priors carry a correlation length; these
# are the schemes for which `scale` is a free hyperparameter.
KERNEL_REGULARIZATIONS = ("matern", "exponential", "gaussian", "adaptive", "gibbs")

# Priors that additionally carry a spatial envelope (see envelope.py).
ENVELOPE_REGULARIZATIONS = ("gaussian",)

# Priors that need a first-pass model to adapt to (two-stage fits).
ADAPTIVE_REGULARIZATIONS = ("adaptive", "gibbs")

# Matern smoothness. nu=0.5 is the (rough) exponential kernel, nu=1.5 is once
# differentiable, nu->inf is a Gaussian. PyAutoLabs fit this with a
# Uniform(0.5, 5.5) prior; we fix it by default and expose it as a knob.
DEFAULT_NU = 1.5

# Hyperparameter search bounds, in log10. PyAutoLabs' shipped priors are
# LogUniform(1e-6, 1e6) on both coefficient and scale; the scale is bounded
# below by the pixel scale and above by the field of view at fit time, since
# values outside that range are meaningless for the image.
LOG_COEFFICIENT_BOUNDS = (-6.0, 6.0)
# ...but the coefficient's meaningful magnitude depends on the data units, so
# the bracket is extended up to here when the shipped range is too narrow.
MAX_LOG_COEFFICIENT = 18.0

# Brightness-adaptive regularisation: ratio of the smoothing strength in
# bright (inner) vs faint (outer) regions, and the brightness contrast scale.
ADAPT_FLOOR = 1e-2   # prior width in the faintest regions, relative to the peak
# How strongly the adaptive prior's width tracks brightness, w = b^power.
# 2.0 is the default: measured against p=1 on the extended+compact mock it
# gives the better extended model without overfitting, which is why `adaptive`
# is the default prior.
ADAPT_POWER = 2.0
GIBBS_ELL_FLOOR = 0.25  # shortest correlation length, as a fraction of the beam


def jax_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import jax  # noqa: F401
        import nufftax  # noqa: F401

        return True
    except Exception:
        return False


def resolve_transformer(n_vis: int, transformer: str = "auto"):
    """Pick the Fourier transform implementation."""
    if transformer == "dft":
        return ag.TransformerDFT
    if transformer == "nufft":
        return ag.TransformerNUFFT
    if transformer != "auto":
        raise ValueError(f"unknown transformer {transformer!r}")
    if n_vis <= DFT_MAX_VIS:
        return ag.TransformerDFT
    if jax_available():
        return ag.TransformerNUFFT
    warnings.warn(
        f"{n_vis} visibilities with no JAX/nufftax installed: falling back to "
        "the direct DFT, which will be slow. Install pyuvimage[jax] for the "
        "fast NUFFT.",
        stacklevel=2,
    )
    return ag.TransformerDFT


def make_dataset(
    uv_wavelengths: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    geometry: ImageGeometry,
    transformer: str = "auto",
    mask_shape: str = "square",
) -> ag.Interferometer:
    """Assemble the PyAutoGalaxy Interferometer dataset.

    ``mask_shape="square"`` (default) makes the reconstruction region the full
    square field. A circular mask leaves the mesh's corner pixels covering no
    image pixels at all, so no data constrains them and the prior alone sets
    their value -- which showed up as bright spurious blobs in the corners of
    the restored image, carrying ~29% of the source flux on one test mock.
    """
    if mask_shape == "square":
        mask = ag.Mask2D.all_false(
            shape_native=geometry.shape_native,
            pixel_scales=geometry.pixel_scale,
        )
    elif mask_shape == "circular":
        mask = ag.Mask2D.circular(
            shape_native=geometry.shape_native,
            pixel_scales=geometry.pixel_scale,
            radius=geometry.mask_radius * 1.001,  # keep boundary pixels
        )
    else:
        raise ValueError("mask_shape must be 'square' or 'circular'")
    cls = resolve_transformer(n_vis=len(data), transformer=transformer)
    return ag.Interferometer(
        data=ag.Visibilities(np.asarray(data, dtype=complex)),
        noise_map=ag.VisibilitiesNoiseMap(np.asarray(noise, dtype=complex)),
        uv_wavelengths=np.asarray(uv_wavelengths, dtype=float),
        real_space_mask=mask,
        transformer_class=cls,
        raise_error_dft_visibilities_limit=False,
    )


def make_regularization(
    kind: str,
    coefficient: float,
    scale: float | None = None,
    nu: float = DEFAULT_NU,
    envelope: dict | None = None,
):
    """Build the source prior (regularisation scheme).

    ``scale`` is a correlation length in **arcsec** -- the kernel schemes build
    their covariance from the mesh coordinates themselves, which are in the
    image plane's angular units.
    """
    kind = kind.lower()
    if kind == "constant":
        return ag.reg.Constant(coefficient=coefficient)
    if kind == "matern":
        from .envelope import CachedMaternKernel

        return CachedMaternKernel(
            coefficient=coefficient, scale=scale or 1.0, nu=nu
        )
    if kind == "gaussian":
        # Matern GP modulated by a Gaussian envelope on the prior width:
        # supplies the spatial information a stationary prior lacks, which
        # is what keeps dirty-beam sidelobes out of the model when the
        # visibilities are sparse.
        from .envelope import GaussianEnvelopeMatern

        env = envelope or {}
        return GaussianEnvelopeMatern(
            coefficient=coefficient,
            scale=scale or 1.0,
            nu=nu,
            envelope_fwhm=env.get("fwhm", 1.0),
            envelope_floor=env.get("floor", 1e-2),
            centre=env.get("centre", (0.0, 0.0)),
        )
    if kind == "exponential":
        return ag.reg.ExponentialKernel(
            coefficient=coefficient, scale=scale or 1.0
        )
    if kind == "gibbs":
        # Non-stationary: the correlation length shortens where the source is
        # bright, so an unresolved component is not asked to be beam-smooth.
        from .envelope import GibbsMatern

        env = envelope or {}
        return GibbsMatern(
            coefficient=coefficient,
            scale=scale or 1.0,
            brightness=env.get("brightness"),
            ell_floor=env.get("ell_floor", GIBBS_ELL_FLOOR),
            power=env.get("power", ADAPT_POWER),
        )
    if kind == "adaptive":
        # Brightness-adaptive: smooth less where a first-pass model says the
        # source is bright, more where it is faint.
        from .envelope import AdaptiveMatern

        env = envelope or {}
        return AdaptiveMatern(
            coefficient=coefficient,
            scale=scale or 1.0,
            nu=nu,
            brightness=env.get("brightness"),
            floor=env.get("floor", ADAPT_FLOOR),
            power=env.get("power", ADAPT_POWER),
        )
    raise ValueError(
        f"unknown regularization {kind!r}; options: {REGULARIZATIONS}"
    )


def galaxy_for(
    mesh_shape: tuple[int, int],
    reg_kind: str,
    coefficient: float,
    reg_scale: float | None = None,
    nu: float = DEFAULT_NU,
    envelope: dict | None = None,
) -> ag.Galaxy:
    pix = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=mesh_shape),
        regularization=make_regularization(
            reg_kind, coefficient, reg_scale, nu, envelope
        ),
    )
    # The redshift is inert without lensing; any value works.
    return ag.Galaxy(redshift=1.0, pixelization=pix)


def fit_at(
    dataset: ag.Interferometer,
    mesh_shape: tuple[int, int],
    reg_kind: str,
    coefficient: float,
    positive_only: bool = True,
    reg_scale: float | None = None,
    nu: float = DEFAULT_NU,
    envelope: dict | None = None,
    adapt_image=None,
) -> ag.FitInterferometer:
    galaxy = galaxy_for(mesh_shape, reg_kind, coefficient, reg_scale, nu, envelope)
    settings = ag.Settings(use_positive_only_solver=positive_only)
    kwargs = {}
    if adapt_image is not None:
        kwargs["adapt_images"] = ag.AdaptImages(
            galaxy_image_dict={galaxy: adapt_image}
        )
    return ag.FitInterferometer(
        dataset=dataset, galaxies=[galaxy], settings=settings, **kwargs
    )


def _safe_evidence(fit: ag.FitInterferometer) -> float:
    try:
        return float(fit.figure_of_merit)
    except Exception as e:  # LinAlgError etc.
        logger.info("    (evidence evaluation failed: %s: %s)", type(e).__name__, e)
        return -np.inf


@dataclass
class PriorScan:
    """Record of the source-prior hyperparameter optimisation."""

    criterion: str = ""
    reg_kind: str = ""
    n_data: int = 0
    free_parameters: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    best: dict = field(default_factory=dict)

    def record(self, params: dict, log_evidence: float, chi_squared: float) -> None:
        self.trials.append(
            {
                **{k: float(v) for k, v in params.items()},
                "log_evidence": float(log_evidence),
                "chi_squared": float(chi_squared),
                "chi_squared_per_datum": (
                    float(chi_squared) / self.n_data if self.n_data else float("nan")
                ),
            }
        )

    def as_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "regularization": self.reg_kind,
            "free_parameters": list(self.free_parameters),
            "n_data": int(self.n_data),
            "best": {k: float(v) for k, v in self.best.items()},
            "n_evaluations": len(self.trials),
            "trials": self.trials,
        }


def _chi_squared(fit: ag.FitInterferometer) -> float:
    try:
        return float(fit.inversion.fast_chi_squared)
    except Exception:
        return float("nan")


def optimise_prior(
    dataset: ag.Interferometer,
    geometry: ImageGeometry,
    reg_kind: str = "matern",
    criterion: str = "discrepancy",
    nu: float = DEFAULT_NU,
    fixed_scale: float | None = None,
    envelope: dict | None = None,
    optimise_envelope: bool = False,
    adapt_image=None,
    chi2_target: float = 1.0,
    max_evaluations: int = 60,
) -> tuple[dict, PriorScan]:
    """Optimise the source-prior hyperparameters.

    Follows the PyAutoLabs convention: the pixelized source's prior is a
    regularisation scheme whose hyperparameters -- the ``coefficient``, plus a
    correlation ``scale`` in arcsec for the kernel (Gaussian-process) schemes
    -- are free parameters chosen by maximising the Bayesian log evidence.
    The evidence balances goodness of fit against model complexity, so a
    single global smoothing strength is no longer imposed by hand.

    The search is a coarse log-spaced grid followed by Nelder-Mead
    refinement, always with the fast unconstrained solver (the evidence is
    defined for the linear solution); positivity is applied to the final fit.

    ``criterion="discrepancy"`` instead drives chi^2 to ``chi2_target * N``,
    which is more robust when the noise map is trustworthy but the evidence
    is poorly behaved (e.g. far fewer visibilities than model pixels).
    """
    mesh_shape = geometry.mesh_shape
    n_data = 2 * len(np.asarray(dataset.data))
    # a kernel prior whose correlation length is pinned has only one free
    # hyperparameter, exactly like the non-kernel schemes
    is_kernel_scheme = reg_kind in KERNEL_REGULARIZATIONS
    # The second free hyperparameter is either the kernel correlation length
    # or, when the correlation length is pinned to the beam and the user asks
    # for it, the Gaussian envelope's width.
    kernel = is_kernel_scheme and fixed_scale is None
    second = "scale" if kernel else ("envelope_fwhm" if optimise_envelope else None)
    two_d = second is not None
    free = ["coefficient"] + ([second] if two_d else [])
    scan = PriorScan(
        criterion=criterion, reg_kind=reg_kind, n_data=n_data,
        free_parameters=free,
    )

    # scale bounds: below the mesh pixel scale the kernel is meaningless,
    # above ~half the field it cannot represent structure.
    log_scale_bounds = (
        np.log10(geometry.mesh_pixel_scale),
        np.log10(max(geometry.fov_arcsec / 2.0, geometry.mesh_pixel_scale * 4)),
    )
    if second == "envelope_fwhm":
        # an envelope narrower than the beam is meaningless; wider than the
        # field cannot constrain anything
        beam_like = fixed_scale or geometry.mesh_pixel_scale
        log_scale_bounds = (
            np.log10(beam_like), np.log10(geometry.fov_arcsec / 2.0)
        )

    def evaluate(log_params: np.ndarray) -> tuple[float, float]:
        coefficient = 10.0 ** float(log_params[0])
        scale = 10.0 ** float(log_params[1]) if kernel else fixed_scale
        env = envelope
        if second == "envelope_fwhm":
            env = {**(envelope or {}), "fwhm": 10.0 ** float(log_params[1])}
        try:
            fit = fit_at(
                dataset, mesh_shape, reg_kind, coefficient,
                positive_only=False, reg_scale=scale, nu=nu,
                envelope=env, adapt_image=adapt_image,
            )
            ev = _safe_evidence(fit)
            chi2 = _chi_squared(fit)
        except Exception as e:
            logger.debug("prior evaluation failed: %s", e)
            return -np.inf, float("nan")
        params = {"coefficient": coefficient}
        if scale is not None:
            params["scale"] = scale
        if second == "envelope_fwhm":
            params["envelope_fwhm"] = env["fwhm"]
        scan.record(params, ev, chi2)
        logger.info(
            "  coefficient=%.4g%s  log_evidence=%.6g  chi2/N=%.4g",
            coefficient,
            f"  scale={scale:.4g}\"" if kernel else "",
            ev, chi2 / n_data if n_data else np.nan,
        )
        return ev, chi2

    def score(log_params: np.ndarray) -> float:
        """Higher is better."""
        for i, (lo, hi) in enumerate(bounds):
            if not (lo <= log_params[i] <= hi):
                return -np.inf
        ev, chi2 = evaluate(log_params)
        if criterion == "evidence":
            return ev
        if criterion == "discrepancy":
            if not np.isfinite(chi2) or chi2 <= 0:
                return -np.inf
            # drive chi^2 to the target; break ties (kernel schemes) by evidence
            miss = np.log10(chi2 / (chi2_target * n_data)) ** 2
            tie = ev / (1.0 + abs(ev)) if np.isfinite(ev) else 0.0
            return -miss + 1e-3 * tie
        raise ValueError("criterion must be 'evidence' or 'discrepancy'")

    bounds = [LOG_COEFFICIENT_BOUNDS] + ([log_scale_bounds] if two_d else [])

    if criterion == "discrepancy":
        # chi^2 rises monotonically with the prior strength, so the
        # coefficient that fits exactly to the noise level can be bisected.
        # For kernel priors the correlation scale is the remaining freedom:
        # among the priors that all fit to the noise level, take the one the
        # evidence prefers.
        target = chi2_target * n_data

        # If even the weakest prior cannot reach the noise level, the
        # discrepancy criterion has no solution and bisection would drive the
        # coefficient to its floor -- silently switching regularisation off
        # and returning a noisy, unregularised model.  Detect that first.
        # the probe vector must match the number of free hyperparameters
        probe = [LOG_COEFFICIENT_BOUNDS[0]]
        if kernel:
            probe.append(float(np.mean(log_scale_bounds)))
        _, chi2_weakest = evaluate(np.array(probe))
        if np.isfinite(chi2_weakest) and chi2_weakest > target:
            logger.warning(
                "chi^2/N = %.3g with essentially no regularisation, above the "
                "target of %.3g: the model cannot reproduce this data however "
                "little it is smoothed, so fitting to the noise level is not "
                "possible. Common causes: the source has real structure finer "
                "than the model pixel scale, the noise map underestimates the "
                "noise, or the field of view is too small. Falling back to "
                "maximum-evidence selection so the prior is not switched off.",
                chi2_weakest / n_data, chi2_target,
            )
            best, ev_scan = optimise_prior(
                dataset, geometry, reg_kind=reg_kind, criterion="evidence",
                nu=nu, fixed_scale=fixed_scale, envelope=envelope,
                adapt_image=adapt_image, max_evaluations=max_evaluations,
            )
            scan.trials.extend(ev_scan.trials)
            scan.criterion = "discrepancy->evidence (unreachable target)"
            scan.best = best
            return best, scan

        def coefficient_for_target(log_scale: float | None) -> tuple[float, float]:
            lo, hi = LOG_COEFFICIENT_BOUNDS
            p = (lambda c: np.array([c, log_scale])) if two_d else (
                lambda c: np.array([c])
            )
            # The coefficient's natural scale depends on the data's units and
            # signal-to-noise, so the shipped LogUniform(1e-6, 1e6) range is
            # not always wide enough: extend the bracket until chi^2 brackets
            # the target.
            _, chi2_hi = evaluate(p(hi))
            while (
                np.isfinite(chi2_hi) and chi2_hi < target
                and hi < MAX_LOG_COEFFICIENT
            ):
                hi = min(hi + 3.0, MAX_LOG_COEFFICIENT)
                _, chi2_hi = evaluate(p(hi))
            if np.isfinite(chi2_hi) and chi2_hi < target:
                logger.warning(
                    "even the strongest prior tried (coefficient 1e%g) fits "
                    "the data better than the noise level (chi2/N = %.3g): "
                    "the model has far more freedom than the data constrain, "
                    "so its faint structure is set by the prior.",
                    hi, chi2_hi / n_data,
                )
                return hi, chi2_hi
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                _, chi2 = evaluate(p(mid))
                if not np.isfinite(chi2) or chi2 < target:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.02:
                    break
            best_c = 0.5 * (lo + hi)
            ev, chi2 = evaluate(p(best_c))
            return best_c, ev

        if not two_d:
            c, _ = coefficient_for_target(None)
            best_log = np.array([c])
        else:
            scales = np.linspace(log_scale_bounds[0], log_scale_bounds[1], 4)
            results = []
            for ls in scales:
                c, ev = coefficient_for_target(ls)
                results.append((ev if np.isfinite(ev) else -np.inf, c, ls))
                logger.info(
                    "  scale=%.4g\" -> coefficient=%.4g (chi2 target), "
                    "log_evidence=%.6g", 10**ls, 10**c, ev,
                )
            best_ev, best_c, best_ls = max(results, key=lambda t: t[0])
            # one refinement pass around the best scale
            step = (log_scale_bounds[1] - log_scale_bounds[0]) / 6.0
            for ls in (best_ls - step, best_ls + step):
                if not (log_scale_bounds[0] <= ls <= log_scale_bounds[1]):
                    continue
                c, ev = coefficient_for_target(ls)
                if np.isfinite(ev) and ev > best_ev:
                    best_ev, best_c, best_ls = ev, c, ls
            best_log = np.array([best_c, best_ls])
    else:
        # ---- coarse grid then local refinement, maximising the evidence
        coarse_coeff = np.linspace(
            LOG_COEFFICIENT_BOUNDS[0], LOG_COEFFICIENT_BOUNDS[1], 7
        )
        if two_d:
            coarse_scale = np.linspace(log_scale_bounds[0], log_scale_bounds[1], 3)
            grid = [np.array([c, sc]) for sc in coarse_scale for c in coarse_coeff]
        else:
            grid = [np.array([c]) for c in coarse_coeff]
        scores = [score(p) for p in grid]
        if not np.any(np.isfinite(scores)):
            raise RuntimeError(
                "the inversion failed for every source prior tried; check the "
                "noise map, the uv coordinates and the field of view"
            )
        x0 = grid[int(np.nanargmax(scores))]

        from scipy.optimize import minimize

        remaining = max(10, max_evaluations - len(scan.trials))
        res = minimize(
            lambda p: -score(p), x0, method="Nelder-Mead",
            options={
                "maxfev": remaining, "xatol": 0.01, "fatol": 1e-3,
                "initial_simplex": (
                    np.vstack([x0, x0 + np.eye(len(x0))[0] * 0.5]
                              + ([x0 + np.eye(len(x0))[1] * 0.3] if two_d else []))
                ),
            },
        )
        best_log = res.x if np.isfinite(res.fun) else x0

    best = {"coefficient": 10.0 ** float(best_log[0])}
    if kernel:
        best["scale"] = 10.0 ** float(best_log[1])
    else:
        if fixed_scale is not None:
            best["scale"] = float(fixed_scale)
        if second == "envelope_fwhm":
            best["envelope_fwhm"] = 10.0 ** float(best_log[1])
    if is_kernel_scheme:
        best["nu"] = float(nu)
    scan.best = best
    logger.info(
        "source prior optimised (%s, %d evaluations): %s",
        criterion, len(scan.trials),
        ", ".join(f"{k}={v:.4g}" for k, v in best.items()),
    )
    return best, scan


def _block_contrast(image: np.ndarray, oversample: int) -> float:
    """Peak-to-peak spread *within* an oversample x oversample block, relative
    to the block mean -- i.e. how strong the checkerboard is."""
    if oversample < 2:
        return 0.0
    a = np.asarray(image, dtype=float)
    n = (a.shape[0] // oversample) * oversample
    blocks = a[:n, :n].reshape(
        n // oversample, oversample, n // oversample, oversample)
    hi = blocks.max(axis=(1, 3))
    lo = blocks.min(axis=(1, 3))
    mean = blocks.mean(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmedian(np.where(mean > 0, (hi - lo) / mean, np.nan)))


def _deblock(image: np.ndarray, oversample: int) -> np.ndarray:
    """Remove the mesh/image checkerboard, keeping the upper envelope.

    A grey dilation over one block replaces every pixel with the largest
    uncertainty in its neighbourhood (killing the low interpolated pixels),
    then a box mean of the same size smooths the result back to a continuous
    field.  Deliberately conservative: the map may slightly over-state the
    error on interpolated pixels, which is the safe direction for a
    significance map.
    """
    if oversample < 2:
        return image
    from scipy.ndimage import maximum_filter, uniform_filter

    a = np.asarray(image, dtype=float)
    finite = np.isfinite(a)
    filled = np.where(finite, a, 0.0)
    envelope = maximum_filter(filled, size=oversample, mode="nearest")
    smoothed = uniform_filter(envelope, size=oversample, mode="nearest")
    return np.where(finite, smoothed, np.nan)


@dataclass
class SingleFit:
    """Everything downstream products need from one fit."""

    fit: ag.FitInterferometer
    geometry: ImageGeometry
    prior: dict
    scan: PriorScan | None = None

    @property
    def coefficient(self) -> float:
        return float(self.prior["coefficient"])

    @property
    def model_mesh_image(self) -> np.ndarray:
        """Reconstruction on the source mesh, 2D [Jy / mesh pixel].

        The inversion solves for surface brightness per *image* pixel (each
        mesh cell's value is replicated over oversample^2 image pixels by the
        mapper), so the per-mesh-pixel flux carries that factor.
        """
        recon = np.asarray(self.fit.inversion.reconstruction)
        k2 = (
            self.geometry.shape_native[0] // self.geometry.mesh_shape[0]
        ) ** 2
        return recon.reshape(self.geometry.mesh_shape) * k2

    @property
    def model_image(self) -> np.ndarray:
        """The model image on the product grid, exactly as the fit formed it.

        Built from each linear object's mapping matrix, which is what the
        inversion actually used.  The rectangular mesh mapper *interpolates*
        between mesh pixel centres rather than block-replicating them, so
        reshaping the reconstruction and repeating each value over its block
        is not the model: it differs by up to ~45% per pixel.
        """
        inv = self.fit.inversion
        slim = None
        for obj, values in inv.reconstruction_dict.items():
            contrib = np.asarray(obj.mapping_matrix) @ np.asarray(values)
            slim = contrib if slim is None else slim + contrib
        return np.asarray(
            ag.Array2D(
                values=slim, mask=self.fit.dataset.real_space_mask
            ).native
        )

    @property
    def posterior_covariance(self) -> np.ndarray:
        """Posterior covariance of the reconstruction, (F + H)^-1.

        For a linear inversion with Gaussian noise and a Gaussian (GP) prior
        the posterior is Gaussian with this covariance, where F is the
        curvature A^T C_d^-1 A and H the regularisation matrix.

        NOTE this is conditional on the prior *and its fitted
        hyperparameters*, and on the noise map being correct.  It does not
        include the systematic error from the prior itself being wrong, which
        on a poorly-sampled field is usually the larger effect.
        """
        return np.linalg.inv(np.asarray(self.fit.inversion.curvature_reg_matrix))

    @property
    def sampling_covariance(self) -> np.ndarray:
        """Covariance of the *estimator*: (F+H)^-1 F (F+H)^-1.

        The MAP solution is s = (F+H)^-1 D with D = A^T C_d^-1 d, so over
        noise realisations Var(D) = F and the estimator scatters with this
        covariance -- smaller than the posterior, because regularisation
        shrinks the solution.  Verified against a 30-realisation Monte Carlo
        to 0.4%.
        """
        cov = self.posterior_covariance
        return cov @ np.asarray(self.fit.inversion.curvature_matrix) @ cov

    def _propagate(self, cov: np.ndarray) -> np.ndarray:
        """sqrt(diag(M C M^T)) on the image grid, for a parameter covariance."""
        var = None
        for obj, _ in self.fit.inversion.reconstruction_dict.items():
            M = np.asarray(obj.mapping_matrix)
            sl = self._parameter_slice(obj)
            block = cov[sl, :][:, sl]
            contrib = np.einsum("ij,jk,ik->i", M, block, M, optimize=True)
            var = contrib if var is None else var + contrib
        return np.asarray(
            ag.Array2D(
                values=np.sqrt(np.clip(var, 0.0, None)),
                mask=self.fit.dataset.real_space_mask,
            ).native
        )

    @property
    def model_uncertainty_sampling(self) -> np.ndarray:
        """1-sigma scatter of the model image over noise realisations.

        How much each pixel would jitter if the source were re-observed with
        the same array and the same prior.  It does NOT include the smoothing
        bias, which is typically the larger term (~3x this on our test mock).
        """
        return self._propagate(self.sampling_covariance)

    @property
    def model_uncertainty(self) -> np.ndarray:
        """Per-pixel 1-sigma uncertainty of the model image [Jy/pixel].

        The mesh mapper interpolates, so an image pixel is a weighted sum of
        several mesh pixels whose errors are strongly correlated (over roughly
        the prior's correlation length).  The uncertainty is therefore
        propagated properly as sqrt(diag(M C M^T)) rather than by copying
        per-mesh-pixel errors onto the image grid, which would misstate it.

        This is the Bayesian posterior width.  Measured against a Monte Carlo
        on our test mock it is 1.25x the *total* rms error from the truth, so
        it is a reasonable (slightly conservative) error bar -- but see
        `model_uncertainty_sampling` for the purely random part, and note that
        the smoothing bias, not the noise, dominates the difference.
        """
        return self._propagate(self.posterior_covariance)

    def model_image_at_scale(self, factor: float) -> np.ndarray:
        """The model image with the regularisation strength scaled by `factor`.

        H = coefficient x C^-1 scales linearly with the coefficient, so this
        needs no refit: re-solve (F + factor*H) s = D and re-map.  Returns
        None if the rescaled system is not positive definite (weakening the
        prior far enough makes F alone singular wherever the mesh has pixels
        the uv coverage does not constrain).
        """
        inv = self.fit.inversion
        F = np.asarray(inv.curvature_matrix)
        H = np.asarray(inv.regularization_matrix)
        D = np.asarray(inv.data_vector)
        try:
            values = np.linalg.solve(F + float(factor) * H, D)
        except np.linalg.LinAlgError:
            return None
        slim = None
        for obj, _ in inv.reconstruction_dict.items():
            M = np.asarray(obj.mapping_matrix)
            sl = self._parameter_slice(obj)
            contrib = M @ values[sl]
            slim = contrib if slim is None else slim + contrib
        return np.asarray(
            ag.Array2D(
                values=slim, mask=self.fit.dataset.real_space_mask
            ).native
        )

    def prior_systematic(self, spread_dex: float = 0.5) -> np.ndarray:
        """Per-pixel systematic from the choice of regularisation strength.

        The statistical error `model_uncertainty` is conditional on one prior
        *and* one strength for it.  How far the model moves when that strength
        is varied over the range these data allow is a measurable systematic,
        and on compact features it is the larger term.  Same construction as
        the point-source flux systematic, which turned pulls of up to 24 sigma
        into pulls under 3.

        This does not capture the prior *family* being wrong -- nothing
        cheap does.
        """
        base = self.model_image
        worst = np.zeros_like(base)
        for factor in (10.0**spread_dex, 10.0**-spread_dex):
            alt = self.model_image_at_scale(factor)
            if alt is None:
                continue
            worst = np.maximum(worst, np.abs(alt - base))
        return worst

    def model_uncertainty_total(
        self, spread_dex: float = 0.5, deblock: bool = True
    ) -> tuple[np.ndarray, dict]:
        """Single total 1-sigma map on the model image [Jy/pixel].

        sqrt(statistical^2 + systematic^2), with the mesh/image checkerboard
        removed.  Returns (map, terms) where `terms` holds the two components
        and the median of each, for the header and the parameter record.

        **The checkerboard.** Products live on a grid `oversample` times finer
        than the model mesh, and the mapper interpolates: a pixel sitting on a
        mesh node inherits one mesh pixel's variance, while a pixel between
        nodes is a weighted average of several and has a genuinely smaller
        one.  Both numbers are correct, but the alternating pattern is an
        artefact of the two grids, not of the sky, and it lands straight in
        any significance map the user makes.  `deblock` replaces it with its
        upper envelope (see `_deblock`), which is the conservative choice:
        an over-stated error never manufactures a detection.
        """
        stat = self.model_uncertainty
        sys_ = self.prior_systematic(spread_dex)
        total = np.hypot(stat, sys_)
        raw_total = total
        if deblock:
            total = _deblock(total, self.geometry.oversample)
        terms = {
            "statistical_median": float(np.nanmedian(stat)),
            "systematic_median": float(np.nanmedian(sys_)),
            "total_median": float(np.nanmedian(total)),
            "checkerboard_amplitude": float(_block_contrast(
                raw_total, self.geometry.oversample)),
            "systematic_spread_dex": float(spread_dex),
            "deblocked": bool(deblock),
        }
        return total, terms

    def _parameter_slice(self, obj):
        """Index range of one linear object within the stacked parameter vector."""
        start = 0
        for other, values in self.fit.inversion.reconstruction_dict.items():
            n = np.asarray(values).size
            if other is obj:
                return slice(start, start + n)
            start += n
        raise KeyError("linear object not found in the inversion")

    def aperture_uncertainty(self, region: np.ndarray) -> float:
        """1-sigma uncertainty on the summed flux inside a region [Jy].

        Per-pixel errors must NOT be added in quadrature: the posterior
        covariance is strongly correlated over roughly the prior's correlation
        length.  Quadrature is then wrong in either direction depending on the
        correlation structure -- on our test mock it *overstates* a compact
        aperture's error by ~1.4x, because interpolated pixels are
        anticorrelated with their neighbours.  This computes w^T (M C M^T) w
        properly, with w the region's indicator on the image grid.

        The result covers the random error only.  The smoothing bias is
        typically larger: on the same mock a knot's aperture flux came out
        0.0115 +/- 0.0001 against a true 0.0129, an 11% offset at ~12 sigma.

        Parameters
        ----------
        region
            Boolean array on the product grid selecting the aperture.
        """
        region = np.asarray(region, dtype=bool)
        if region.shape != self.geometry.shape_native:
            raise ValueError(
                f"region shape {region.shape} != image grid "
                f"{self.geometry.shape_native}"
            )
        mask = self.fit.dataset.real_space_mask
        w_native = region.astype(float)
        w = np.asarray(ag.Array2D(values=w_native, mask=mask).slim)
        cov = self.posterior_covariance
        var = 0.0
        for obj, _ in self.fit.inversion.reconstruction_dict.items():
            M = np.asarray(obj.mapping_matrix)
            sl = self._parameter_slice(obj)
            v = M.T @ w                      # aperture weights in mesh space
            var += float(v @ cov[sl, :][:, sl] @ v)
        return float(np.sqrt(max(var, 0.0)))

    @property
    def model_visibilities(self) -> np.ndarray:
        return np.asarray(self.fit.model_data)

    @property
    def residual_visibilities(self) -> np.ndarray:
        return np.asarray(self.fit.dataset.data) - self.model_visibilities

    @property
    def log_evidence(self) -> float:
        return _safe_evidence(self.fit)

    @property
    def chi_squared(self) -> float:
        return _chi_squared(self.fit)


def fit_dataset(
    dataset: ag.Interferometer,
    geometry: ImageGeometry,
    reg_kind: str = "matern",
    prior: dict | None = None,
    positive_only: bool = True,
    criterion: str = "discrepancy",
    nu: float = DEFAULT_NU,
    fixed_scale: float | None = None,
    envelope: dict | None = None,
    optimise_envelope: bool = False,
    chi2_target: float = 1.0,
    adapt_image=None,
) -> SingleFit:
    """Fit one Interferometer dataset (one channel, or the MFS stack).

    ``prior`` fixes the source-prior hyperparameters (keys ``coefficient``
    and, for kernel schemes, ``scale`` in arcsec); if ``None`` they are
    optimised.  ``reg_kind="adapt"`` runs two stages: a first pass with
    constant regularisation whose model becomes the brightness map that the
    second pass's adaptive regularisation follows.
    """
    mesh_shape = geometry.mesh_shape

    envelope = dict(envelope or {})
    if reg_kind in ADAPTIVE_REGULARIZATIONS and envelope.get("brightness") is None:
        logger.info("%s prior: first pass (plain Matern)...", reg_kind)
        first = fit_dataset(
            dataset, geometry, reg_kind="matern", prior=None,
            positive_only=positive_only, criterion=criterion, nu=nu,
            fixed_scale=fixed_scale, chi2_target=chi2_target,
        )
        envelope["brightness"] = np.clip(first.model_mesh_image.ravel(), 0.0, None)
        logger.info(
            "%s prior: second pass (tracks the first-pass model)", reg_kind,
        )

    # ---- positivity sanity check -------------------------------------
    # The non-negative solver is not always reliable: on some datasets it
    # returns a solution that ignores the prior entirely (identical chi^2 for
    # coefficients spanning eight orders of magnitude) and fits far worse than
    # the unconstrained solve. Silently shipping that would make every prior
    # look identical, so check once and fall back if it happens.
    if positive_only and prior is None:
        probe_scale = fixed_scale if reg_kind in KERNEL_REGULARIZATIONS else None
        probe_kwargs = dict(
            reg_scale=probe_scale, nu=nu, envelope=envelope,
            adapt_image=adapt_image,
        )

        def _probe(coefficient, positive):
            try:
                return _chi_squared(fit_at(
                    dataset, mesh_shape, reg_kind, coefficient,
                    positive_only=positive, **probe_kwargs))
            except Exception:
                return np.nan

        free = _probe(1.0, False)
        constrained = _probe(1.0, True)
        n_vis = len(np.asarray(dataset.data))
        reason = None
        if (
            np.isfinite(free) and np.isfinite(constrained)
            and constrained > max(2.0 * free, free + 2 * n_vis)
        ):
            reason = (
                f"it fits far worse than the unconstrained solve "
                f"(chi^2 {constrained:.4g} vs {free:.4g})"
            )
        else:
            # Second, independent symptom: the solver *ignores* the prior.
            # Compare two strengths twelve decades apart -- no real prior can
            # give the same chi^2 at both.  This is what the single-point
            # comparison above misses, and it is the common failure: on a
            # coarse-beam mock every coefficient from 1e-6 to 5.9 returned
            # chi^2/N = 1.159 to four figures.
            weak, strong = _probe(1e-3, True), _probe(1e9, True)
            if (
                np.isfinite(weak) and np.isfinite(strong) and weak > 0
                and abs(strong - weak) / weak < 0.01
            ):
                reason = (
                    f"it returns the same chi^2 ({weak:.4g}) for "
                    f"regularisation strengths twelve decades apart, so it is "
                    f"ignoring the prior entirely"
                )
        if reason is not None:
            logger.warning(
                "the non-negative solver is unreliable on this data: %s. "
                "Disabling positivity for this fit; the model may contain "
                "small negative values.", reason,
            )
            positive_only = False

    scan = None
    if prior is None:
        prior, scan = optimise_prior(
            dataset, geometry, reg_kind=reg_kind, criterion=criterion,
            nu=nu, fixed_scale=fixed_scale, envelope=envelope,
            optimise_envelope=optimise_envelope, adapt_image=adapt_image,
            chi2_target=chi2_target,
        )
    if "envelope_fwhm" in prior:
        envelope = {**envelope, "fwhm": float(prior["envelope_fwhm"])}
    prior = dict(prior)
    if reg_kind in KERNEL_REGULARIZATIONS:
        prior.setdefault("nu", nu)

    def _fit(coefficient: float) -> ag.FitInterferometer:
        return fit_at(
            dataset, mesh_shape, reg_kind, coefficient,
            positive_only=positive_only, reg_scale=prior.get("scale"),
            nu=prior.get("nu", nu), envelope=envelope, adapt_image=adapt_image,
        )

    fit = _fit(prior["coefficient"])

    # The hyperparameter search uses the fast unconstrained solver, but the
    # final fit may impose positivity, which raises chi^2. When that shifts
    # the fit off the noise level, re-bisect the coefficient with the solver
    # actually in use so the delivered model really does fit to the noise.
    if scan is not None and criterion == "discrepancy" and positive_only:
        n_data = 2 * len(np.asarray(dataset.data))
        target = chi2_target * n_data
        chi2 = _chi_squared(fit)
        if not np.isfinite(chi2) or abs(
            np.log10(max(chi2, 1e-30) / target)
        ) > np.log10(1.5):
            logger.info(
                "positivity moved chi2/N to %.4g; re-optimising the "
                "coefficient with the constrained solver...",
                chi2 / n_data if np.isfinite(chi2) else np.nan,
            )
            # Bisect using only constrained evaluations, tracked separately
            # from the (unconstrained) hyperparameter search trials.
            tried: list[tuple[float, float]] = [(prior["coefficient"], chi2)]
            lo, hi = LOG_COEFFICIENT_BOUNDS[0], np.log10(prior["coefficient"])
            while chi2 < target and hi < MAX_LOG_COEFFICIENT:
                hi = min(hi + 3.0, MAX_LOG_COEFFICIENT)
                trial = _fit(10.0**hi)
                chi2 = _chi_squared(trial)
                tried.append((10.0**hi, chi2))
                logger.info(
                    "  coefficient=%.4g (positive)  chi2/N=%.4g",
                    10.0**hi, chi2 / n_data if np.isfinite(chi2) else np.nan,
                )
            for _ in range(9):
                mid = 0.5 * (lo + hi)
                trial = _fit(10.0**mid)
                c = _chi_squared(trial)
                tried.append((10.0**mid, c))
                scan.record(
                    {**prior, "coefficient": 10.0**mid},
                    _safe_evidence(trial), c,
                )
                logger.info(
                    "  coefficient=%.4g (positive)  chi2/N=%.4g",
                    10.0**mid, c / n_data if np.isfinite(c) else np.nan,
                )
                if not np.isfinite(c) or c < target:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.02:
                    break
            usable = [
                (co, c) for co, c in tried if np.isfinite(c) and c > 0
            ]
            if usable:
                prior["coefficient"] = float(
                    min(usable, key=lambda t: abs(np.log10(t[1] / target)))[0]
                )
            else:
                prior["coefficient"] = 10.0 ** (0.5 * (lo + hi))
            fit = _fit(prior["coefficient"])
            logger.info(
                "  chosen coefficient=%.4g (chi2/N=%.4g)",
                prior["coefficient"], _chi_squared(fit) / n_data,
            )
            scan.best = dict(prior)

            # Second, stronger check on the non-negative solver.  The probe
            # above compares it with the unconstrained solve at a single
            # coefficient, and that misses the case where it merely *ignores*
            # the prior: here every trial spanning many decades of coefficient
            # returned chi^2/N = 1.448 to four figures, which no real prior
            # can do.  Detect a flat response and fall back.
            # tried[0] is the coefficient the *unconstrained* search chose,
            # so it sits apart from the rest; judge the flatness on the
            # constrained bisection trials alone.
            finite = [
                (co, c) for co, c in tried[1:] if np.isfinite(c) and c > 0
            ]
            if len(finite) >= 3:
                spread = max(c for _, c in finite) / min(c for _, c in finite)
                decades = np.ptp(np.log10([co for co, _ in finite]))
                if spread < 1.01 and decades > 3.0:
                    logger.warning(
                        "the non-negative solver returned the same chi^2 "
                        "(%.4g) across %.0f decades of regularisation "
                        "strength: it is ignoring the prior. Disabling "
                        "positivity for this fit; the model may contain "
                        "small negative values.",
                        finite[0][1], decades,
                    )
                    positive_only = False
                    prior, scan = optimise_prior(
                        dataset, geometry, reg_kind=reg_kind,
                        criterion=criterion, nu=nu, fixed_scale=fixed_scale,
                        envelope=envelope, optimise_envelope=optimise_envelope,
                        adapt_image=adapt_image, chi2_target=chi2_target,
                    )
                    if reg_kind in KERNEL_REGULARIZATIONS:
                        prior.setdefault("nu", nu)
                    fit = _fit(prior["coefficient"])

    chi2_final = _chi_squared(fit)
    n_data_final = 2 * len(np.asarray(dataset.data))
    if np.isfinite(chi2_final):
        ratio = chi2_final / (chi2_target * n_data_final)
        if ratio > 1.3:
            logger.warning(
                "chi^2/N = %.3g against a target of %.3g: the model does not "
                "reproduce the data and its products should not be trusted. "
                "Usual causes are emission outside --fov, a mesh too coarse "
                "for the S/N, or an underestimated noise map.",
                chi2_final / n_data_final, chi2_target,
            )

    return SingleFit(fit=fit, geometry=geometry, prior=prior, scan=scan)

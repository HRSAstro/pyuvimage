"""Point components alone: no mesh, no prior (`--points-only`).

The model is a sum of unresolved sources,

    V(u, v) = sum_k a_k exp(-2 pi i (x_k u + y_k v)),

linear in the fluxes a_k and non-linear in the positions (y_k, x_k). With the
Gram matrix C_ij = p_i^T W p_j and data term Dp_i = p_i^T W d of the point
columns p_i (`pointsource.point_visibilities`),

    chi^2(a, positions) = d^T W d - 2 a^T Dp + a^T C a,   a_hat = C^-1 Dp,

so the fluxes are profiled out exactly and only the positions are searched.

Two ways of getting C and Dp, chosen by how the data are held:

* `VisibilityPoints` -- in memory, from the visibilities themselves (exact,
  and exact for unequal sigma_re and sigma_im);
* `GridPoints` -- streamed, read off the oversampled kernel and dirty image
  of the streaming pass (`streamed_points.PointGrids`; C_ij = K(p_i - p_j),
  Dp_i = Dd(p_i), to 2e-9 of K(0)).

Positions are either the user's (`positions`, refined unless ``refine`` is
off) or detected: the most significant positive peak of the profile-
likelihood significance map on the image grid, refined by
`pointsource.refine_position` (a sub-pixel scan, then Nelder-Mead, with the
other points' fluxes re-fitted at every trial), kept if its flux clears
``significance`` sigma, and repeated up to ``max_points``.

**Uncertainties** are statistical: the inverse Fisher matrix of the full
chi^2 in (a, y, x), cov = 2 H^-1 with H the Hessian at the best fit (central
differences; exact in the fluxes, which enter quadratically). Positions fixed
(`refine=False`) leaves the fluxes' C^-1. Both are scaled by max(1, chi^2/N)
when the points do not describe the data to the noise -- the same inflation
the mesh fits use (`docs/uncertainty.md`); `chi2_inflation` records it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .pointsource import (
    ARCSEC_TO_RAD,
    DEFAULT_MAX_POINTS,
    DEFAULT_SIGNIFICANCE,
    detection_lattice,
    grid_to_sky,
    refine_position,
    sky_to_grid,
)

logger = logging.getLogger("pyuvimage")

#: minimum separation between detections, in beam FWHM (as `fit_point_sources`)
MIN_SEPARATION_BEAMS = 0.75
#: refinement passes over all points once the set is complete
REFINE_PASSES = 2
#: a ridge on C, relative to its trace, so two coincident points stay solvable
GRAM_RIDGE = 1e-10
#: smallest Gaussian sigma the fit may reach, in image pixels
SIZE_FLOOR_PIXELS = 1e-3
#: Delta chi^2 (after the chi^2/N inflation) a Gaussian must gain over a point
#: -- and over a line (zero minor axis) -- to count as resolved; as the mesh
#: path's unresolved test (`pointsource.DEFAULT_RESOLVED_DELTA_CHI2`)
RESOLVED_DELTA_CHI2 = 9.0
#: Delta chi^2 that sets a one-sided 95% upper limit on an unresolved size
UPPER_LIMIT_DELTA_CHI2 = 2.706


# ---------------------------------------------------------------------------
# the two sources of C and Dp
# ---------------------------------------------------------------------------

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian_uv_terms(uv, shape):
    """(taper, u_maj, u_min) of an elliptical Gaussian ``shape`` = (sigma_maj,
    sigma_min [arcsec], theta [rad, the major axis's PA east of north]).

    A Gaussian of covariance S has the taper exp(-2 pi^2 u^T S u). The major
    axis points along (y, x) = (cos theta, -sin theta) on the grid (north up,
    east = -x), and grid y pairs with v, x with u, so u_maj = v cos - u sin,
    u_min = u cos + v sin.
    """
    s_a, s_b, th = shape
    c, sn = np.cos(th), np.sin(th)
    u_maj = uv[:, 1] * c - uv[:, 0] * sn
    u_min = uv[:, 0] * c + uv[:, 1] * sn
    a, b = s_a * ARCSEC_TO_RAD, s_b * ARCSEC_TO_RAD
    taper = np.exp(-2.0 * np.pi ** 2 * (a * a * u_maj ** 2 + b * b * u_min ** 2))
    return taper, u_maj, u_min


def _stacked_columns(uv, ys, xs, shapes=None) -> np.ndarray:
    """``[Re; Im]`` of unit components at grid (ys, xs) arcsec, (2 n_vis, k):
    points, or Gaussians where ``shapes[j]`` is not None; the convention of
    `pointsource.point_visibilities`."""
    ys = np.asarray(ys, dtype=float).ravel()
    xs = np.asarray(xs, dtype=float).ravel()
    n = uv.shape[0]
    phase = np.multiply.outer(uv[:, 0], xs * ARCSEC_TO_RAD)
    phase += np.multiply.outer(uv[:, 1], ys * ARCSEC_TO_RAD)
    phase *= -2.0 * np.pi
    out = np.empty((2 * n, ys.size))
    np.cos(phase, out=out[:n])
    np.sin(phase, out=out[n:])
    for j, shape in enumerate(shapes or []):
        if shape is not None:
            t = gaussian_uv_terms(uv, shape)[0]
            out[:n, j] *= t
            out[n:, j] *= t
    return out


class VisibilityPoints:
    """C and Dp from the visibilities (in-memory data; exact)."""

    def __init__(self, dataset, imager):
        from .fitting import adjoint_image

        self._adjoint = adjoint_image
        self.dataset = dataset
        self.imager = imager
        self.transformer = dataset.transformer
        self.uv = np.asarray(dataset.uv_wavelengths, dtype=float)
        noise = np.asarray(dataset.noise_map)
        data = np.asarray(dataset.data)
        self.w_re, self.w_im = noise.real ** -2.0, noise.imag ** -2.0
        self.w_stack = np.concatenate([self.w_re, self.w_im])
        self.wd_stack = self.w_stack * np.concatenate([data.real, data.imag])
        self.data = data
        self.chi2_const = float(np.sum(self.wd_stack * np.concatenate([data.real, data.imag])))
        self.n_data = 2 * data.size
        #: C_jj of a point anywhere: exact for equal sigmas, their mean otherwise
        self.k0 = float(0.5 * np.sum(self.w_re + self.w_im))

    def terms(self, positions, shapes=None) -> tuple[np.ndarray, np.ndarray]:
        P = _stacked_columns(self.uv, [p[0] for p in positions], [p[1] for p in positions],
                             shapes)
        return P.T @ (self.w_stack[:, None] * P), P.T @ self.wd_stack

    def _image(self, c) -> np.ndarray:
        import autoarray as aa

        return np.asarray(self._adjoint(self.transformer, aa.Visibilities(c)).native).ravel()

    def lattice_dirty(self, ys, xs) -> np.ndarray:
        """Dp at every image pixel: the adjoint of the weighted data."""
        return self._image(self.w_re * self.data.real + 1j * self.w_im * self.data.imag)

    def lattice_cross(self, ys, xs, position, shape=None) -> np.ndarray:
        """C between every image pixel and a component at `position`."""
        P = _stacked_columns(self.uv, [position[0]], [position[1]], [shape])[:, 0]
        n = self.uv.shape[0]
        return self._image(self.w_re * P[:n] + 1j * self.w_im * P[n:])

    def residual_image(self, positions, amplitudes, shapes=None) -> np.ndarray:
        """Dirty image of the residual visibilities [Jy/beam]."""
        model = np.zeros(self.uv.shape[0], dtype=complex)
        if len(positions):
            P = _stacked_columns(self.uv, [p[0] for p in positions], [p[1] for p in positions],
                                 shapes)
            n = self.uv.shape[0]
            model = (P[:n] + 1j * P[n:]) @ np.asarray(amplitudes, dtype=float)
        return np.asarray(self.imager.dirty_image(self.data - model))


class GridPoints:
    """C and Dp from the streamed terms' oversampled grids."""

    def __init__(self, terms, imager):
        from .streamed_points import PointGrids

        self.sparse_terms = terms
        self.imager = imager
        self.grids = PointGrids(terms)
        self.chi2_const = float(terms.data_term)
        self.n_data = 2 * int(terms.n_vis)
        self.k0 = float(self.grids.kernel(0.0, 0.0))

    def terms(self, positions, shapes=None) -> tuple[np.ndarray, np.ndarray]:
        if shapes is not None and any(sh is not None for sh in shapes):
            raise NotImplementedError("Gaussian components are fitted in memory only")
        ys = np.array([p[0] for p in positions], dtype=float)
        xs = np.array([p[1] for p in positions], dtype=float)
        C = self.grids.kernel(ys[:, None] - ys[None, :], xs[:, None] - xs[None, :])
        return 0.5 * (C + C.T), np.asarray(self.grids.dirty(ys, xs), dtype=float)

    def lattice_dirty(self, ys, xs) -> np.ndarray:
        return np.asarray(self.grids.dirty(ys, xs), dtype=float)

    def lattice_cross(self, ys, xs, position) -> np.ndarray:
        return np.asarray(self.grids.kernel(ys - position[0], xs - position[1]), dtype=float)

    def residual_image(self, positions, amplitudes, shapes=None) -> np.ndarray:
        """dirty(data) - sum a_k K(x - p_k) / sum w [Jy/beam]."""
        ny, nx = self.sparse_terms.shape_native
        pix = float(self.sparse_terms.pixel_scale)
        iy, ix = np.mgrid[0:ny, 0:nx]
        gy = ((ny - 1) / 2.0 - iy) * pix
        gx = (ix - (nx - 1) / 2.0) * pix
        out = np.array(self.imager.dirty_image_of_data(), dtype=float)
        for (py, px), a in zip(positions, amplitudes):
            out -= float(a) * self.grids.kernel(gy - py, gx - px) / self.imager._norm
        return out


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

class PointOnlySystem:
    """chi^2 of point components, fluxes profiled out (see the module notes).

    `chi_squared(positions)` is what `pointsource.refine_position` asks for.
    """

    def __init__(self, backend, fixed=None):
        self.backend = backend
        self.n_data = backend.n_data
        self.chi2_const = backend.chi2_const
        #: components whose geometry is held while points are searched
        #: [((y, x), shape or None)], fluxes re-fitted with the points'
        self.fixed = list(fixed or [])
        self._cross: dict = {}

    def _terms(self, positions):
        pos = list(positions) + [c[0] for c in self.fixed]
        shapes = [None] * len(positions) + [c[1] for c in self.fixed]
        if not self.fixed:
            return self.backend.terms(pos)
        return self.backend.terms(pos, shapes)

    def solve(self, positions):
        """(fluxes, their covariance C^-1, chi^2) at fixed positions; with
        `fixed` components their fluxes are fitted too and marginalised."""
        positions = list(positions)
        k = len(positions)
        if not positions and not self.fixed:
            return np.zeros(0), np.zeros((0, 0)), float(self.chi2_const)
        C, Dp = self._terms(positions)
        C = C + np.eye(C.shape[0]) * GRAM_RIDGE * max(np.trace(C), 1e-30)
        cov = np.linalg.inv(C)
        a = cov @ Dp
        return a[:k], cov[:k, :k], float(self.chi2_const - a @ Dp)

    def chi_squared(self, positions) -> float:
        return self.solve(positions)[2]

    def chi2_full(self, amplitudes, positions) -> float:
        """chi^2 at given fluxes *and* positions (no profiling): the Hessian's
        function."""
        C, Dp = self.backend.terms(list(positions))
        a = np.asarray(amplitudes, dtype=float)
        return float(self.chi2_const - 2.0 * a @ Dp + a @ C @ a)

    def significance_map(self, accepted, ys, xs):
        """Per lattice position: the flux a new point there would take, and
        its significance, with the accepted points' fluxes re-fitted.

        r_j = Dp_j - c_j C_S^-1 Dp_S,  s_j = k0 - c_j C_S^-1 c_j^T,
        flux r_j / s_j,  significance r_j / sqrt(s_j).
        """
        r = self.backend.lattice_dirty(ys, xs).copy()
        s = np.full(r.shape, self.backend.k0)
        comps = [(p, None) for p in accepted] + self.fixed
        if comps:
            Kc = np.stack([self._lattice_cross(ys, xs, p, sh) for p, sh in comps], axis=1)
            C, Dp = self._terms(accepted)
            C = C + np.eye(C.shape[0]) * GRAM_RIDGE * max(np.trace(C), 1e-30)
            Ci = np.linalg.inv(C)
            r -= Kc @ (Ci @ Dp)
            s -= np.einsum("ij,jk,ik->i", Kc, Ci, Kc)
        s = np.maximum(s, 1e-30 * self.backend.k0)
        return r / s, r / np.sqrt(s)

    def _lattice_cross(self, ys, xs, position, shape=None):
        key = (float(position[0]), float(position[1]),
               None if shape is None else tuple(float(v) for v in shape))
        hit = self._cross.get(key)
        if hit is None:
            hit = self._cross[key] = (
                self.backend.lattice_cross(ys, xs, position) if shape is None
                else self.backend.lattice_cross(ys, xs, position, shape))
            if len(self._cross) > 64:
                self._cross.pop(next(iter(self._cross)))
        return hit

    def fisher_covariance(self, positions, amplitudes, beam_fwhm: float, pixel: float):
        """2 H^-1 of chi^2(a, y, x) at the best fit, ordered [a..., y..., x...].

        Steps: the flux error in a (chi^2 is quadratic there, so any step is
        exact); in position a fraction of the expected error, beam sigma over
        S/N, floored well below a pixel. Returns None when H is not positive
        definite (the positions are not at a minimum).
        """
        k = len(positions)
        a0 = np.asarray(amplitudes, dtype=float)
        _, cov_a, _ = self.solve(positions)
        sig_a = np.sqrt(np.maximum(np.diag(cov_a), 1e-300))
        snr = np.abs(a0) / sig_a
        beam_sigma = max(float(beam_fwhm), 2.0 * pixel) / 2.3548
        h_pos = np.maximum(0.3 * beam_sigma / np.maximum(snr, 1.0), 1e-3 * pixel)
        theta0 = np.concatenate([a0, [p[0] for p in positions], [p[1] for p in positions]])
        steps = np.concatenate([sig_a, h_pos, h_pos])
        n = 3 * k

        def f(theta):
            pos = list(zip(theta[k:2 * k], theta[2 * k:]))
            return self.chi2_full(theta[:k], pos)

        f0 = f(theta0)
        H = np.empty((n, n))
        e = np.eye(n)
        fp = [f(theta0 + steps[i] * e[i]) for i in range(n)]
        fm = [f(theta0 - steps[i] * e[i]) for i in range(n)]
        for i in range(n):
            H[i, i] = (fp[i] - 2.0 * f0 + fm[i]) / steps[i] ** 2
            for j in range(i + 1, n):
                di, dj = steps[i] * e[i], steps[j] * e[j]
                H[i, j] = H[j, i] = (
                    f(theta0 + di + dj) - f(theta0 + di - dj)
                    - f(theta0 - di + dj) + f(theta0 - di - dj)
                ) / (4.0 * steps[i] * steps[j])
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return None
        return 2.0 * np.linalg.inv(H)


@dataclass
class FittedPoint:
    """One point component of a `--points-only` fit."""

    d_ra: float                 # offset from the image centre [arcsec], +ve East
    d_dec: float                # [arcsec], +ve North
    flux: float                 # Jy, apparent (as observed)
    flux_error: float           # 1 sigma, Jy (statistical, chi^2/N-inflated)
    d_ra_error: float | None    # 1 sigma, arcsec; None for fixed positions
    d_dec_error: float | None
    ra_dec_correlation: float | None
    user_supplied: bool = False
    #: offset from the phase centre (the pointing) [arcsec]
    d_ra_phase_centre: float | None = None
    d_dec_phase_centre: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    primary_beam: float | None = None
    flux_pbcor: float | None = None
    flux_error_pbcor: float | None = None

    @property
    def significance(self) -> float:
        return abs(self.flux) / self.flux_error if self.flux_error > 0 else np.inf

    def as_dict(self) -> dict:
        out = {k: (None if v is None else (bool(v) if isinstance(v, bool) else float(v)))
               for k, v in asdict(self).items()}
        out["position"] = "user" if self.user_supplied else "auto-detected"
        out.pop("user_supplied")
        out["significance"] = float(self.significance)
        return out


@dataclass
class PointOnlyFit:
    points: list
    grid_positions: list
    amplitudes: np.ndarray
    covariance: np.ndarray            # [a..., y..., x...] (or [a...] for fixed positions)
    chi_squared: float
    n_data: int
    chi2_inflation: float
    positions_fitted: bool
    log: list = field(default_factory=list)
    #: `fit_with_gaussians` only
    gaussians: list = field(default_factory=list)
    gaussian_components: list = field(default_factory=list)   # [((y, x), shape)]
    gaussian_amplitudes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    covariance_labels: list | None = None

    @property
    def chi2_per_n(self) -> float:
        return self.chi_squared / self.n_data


def fit_points_only(
    system: PointOnlySystem,
    geometry,
    positions=None,
    significance: float = DEFAULT_SIGNIFICANCE,
    max_points: int = DEFAULT_MAX_POINTS,
    refine: bool = True,
    beam_fwhm: float | None = None,
) -> PointOnlyFit:
    """Fit point components to the data with nothing else in the model.

    ``positions`` are (dRA, dDec) offsets [arcsec] from the image centre:
    kept whatever their significance (refined unless ``refine`` is False).
    Without them, peaks are detected and kept above ``significance``.
    """
    pix = float(geometry.pixel_scale)
    beam = float(beam_fwhm) if beam_fwhm else 2.0 * pix
    step = pix / 2.0
    notes: list = []

    def refine_all(pos, passes):
        pos = list(pos)
        for _ in range(passes):
            for i in range(len(pos)):
                pos[i] = refine_position(system, pos, i, step=step)
        return pos

    if positions:
        accepted = [sky_to_grid(*p) for p in positions]
        user = [True] * len(accepted)
        if refine:
            accepted = refine_all(accepted, REFINE_PASSES)
    else:
        lat_y, lat_x = detection_lattice(geometry)
        min_sep = MIN_SEPARATION_BEAMS * beam
        accepted, excluded = [], []
        for _ in range(max_points + 5):
            if len(accepted) >= max_points:
                break
            _, sig = system.significance_map(accepted, lat_y, lat_x)
            score = sig.copy()
            for (ey, ex) in accepted + excluded:
                score[np.hypot(lat_y - ey, lat_x - ex) <= min_sep] = -np.inf
            j = int(np.argmax(score))
            if not np.isfinite(score[j]) or score[j] < significance:
                break
            cand = (float(lat_y[j]), float(lat_x[j]))
            trial = accepted + [cand]
            if refine:
                trial[-1] = refine_position(system, trial, len(trial) - 1, step=step)
            a, cov, _ = system.solve(trial)
            snr = a[-1] / np.sqrt(cov[-1, -1])
            if snr < significance:
                excluded.extend([cand, trial[-1]])
                notes.append(f"candidate at {grid_to_sky(*trial[-1])} rejected: "
                             f"{snr:.1f} sigma after refinement")
                continue
            accepted = trial
            logger.info("  point %d at (dRA, dDec) = (%.4f, %.4f)\": %.4g Jy, %.1f sigma",
                        len(accepted), *grid_to_sky(*accepted[-1]), a[-1], snr)
        if refine and len(accepted) > 1:
            accepted = refine_all(accepted, REFINE_PASSES)
        # a later point can take flux from an earlier one: prune below the cut
        while accepted:
            a, cov, _ = system.solve(accepted)
            snr = a / np.sqrt(np.diag(cov))
            weakest = int(np.argmin(snr))
            if snr[weakest] >= significance:
                break
            notes.append(f"point at {grid_to_sky(*accepted[weakest])} dropped: "
                         f"{snr[weakest]:.1f} sigma once the others were fitted")
            accepted.pop(weakest)
            if refine and accepted:
                accepted = refine_all(accepted, 1)
        user = [False] * len(accepted)

    a, cov_a, chi2 = system.solve(accepted)
    inflation = max(1.0, chi2 / system.n_data)
    cov = None
    if refine and accepted and not system.fixed:
        cov = system.fisher_covariance(accepted, a, beam, pix)
        if cov is None:
            notes.append("the chi^2 Hessian is not positive definite at the fitted "
                         "positions; position errors omitted, flux errors at "
                         "fixed positions")
    fitted = cov is not None
    if not fitted:
        cov = cov_a
    cov = cov * inflation
    k = len(accepted)
    points = []
    for i, (y, x) in enumerate(accepted):
        d_ra, d_dec = grid_to_sky(y, x)
        if fitted:
            sy, sx = np.sqrt(cov[k + i, k + i]), np.sqrt(cov[2 * k + i, 2 * k + i])
            # dRA = -x: the correlation with dDec = y changes sign
            rho = -cov[k + i, 2 * k + i] / (sy * sx) if sy > 0 and sx > 0 else 0.0
        else:
            sy = sx = rho = None
        points.append(FittedPoint(
            d_ra=d_ra, d_dec=d_dec, flux=float(a[i]), flux_error=float(np.sqrt(cov[i, i])),
            d_ra_error=None if sx is None else float(sx),
            d_dec_error=None if sy is None else float(sy),
            ra_dec_correlation=None if rho is None else float(rho),
            user_supplied=user[i],
        ))
    return PointOnlyFit(points=points, grid_positions=accepted, amplitudes=a,
                        covariance=cov, chi_squared=chi2, n_data=system.n_data,
                        chi2_inflation=inflation, positions_fitted=fitted, log=notes)


# ---------------------------------------------------------------------------
# Gaussian components (user-given; in memory)
# ---------------------------------------------------------------------------

@dataclass
class FittedGaussian:
    """One elliptical Gaussian component: its intrinsic (deconvolved) shape,
    since the fit is to the visibilities."""

    d_ra: float
    d_dec: float
    flux: float                  # Jy, total, apparent
    flux_error: float
    d_ra_error: float
    d_dec_error: float
    ra_dec_correlation: float
    fwhm_major: float            # arcsec
    fwhm_major_error: float
    fwhm_minor: float
    fwhm_minor_error: float
    pa_deg: float                # major axis, east of north, [0, 180)
    pa_error_deg: float
    #: "yes", "major axis only" or "no" (consistent with a point); see
    #: `resolution_test`. Sizes and their errors mean little when "no".
    resolved: str | None = None
    delta_chi2_vs_point: float | None = None     # Gaussian over point
    delta_chi2_vs_line: float | None = None      # over a zero minor axis
    fwhm_upper_limit: float | None = None        # 95%, circular; when unresolved
    fwhm_minor_upper_limit: float | None = None  # 95%; when resolved along the major axis only
    d_ra_phase_centre: float | None = None
    d_dec_phase_centre: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    primary_beam: float | None = None
    flux_pbcor: float | None = None
    flux_error_pbcor: float | None = None

    @property
    def significance(self) -> float:
        return abs(self.flux) / self.flux_error if self.flux_error > 0 else np.inf

    @property
    def shape(self):
        """(sigma_maj, sigma_min [arcsec], theta [rad]) for the uv taper."""
        return (self.fwhm_major / FWHM_PER_SIGMA, self.fwhm_minor / FWHM_PER_SIGMA,
                np.radians(self.pa_deg))

    def as_dict(self) -> dict:
        out = {k: (v if v is None or isinstance(v, str) else float(v))
               for k, v in asdict(self).items()}
        out["significance"] = float(self.significance)
        return out


class JointModel:
    """All components at once, for the joint least-squares fit.

    Parameters, per component in order: flux, y, x [arcsec] (the position
    omitted for a fixed point), then for a Gaussian its free shape
    parameters among ln sigma_a, ln sigma_b [arcsec] and theta [rad];
    ``holds[i]`` = {index: value} fixes any of (sigma_a, sigma_b, theta) at a
    value (sigma in arcsec) -- how the resolution tests ask for a Gaussian of
    given size. The residual is the noise-weighted ``[Re; Im](d - model)``,
    so chi^2 = |r|^2, and the Jacobian is analytic.
    """

    def __init__(self, backend, kinds, fixed_position, holds=None):
        self.uv = backend.uv
        self.sw = np.sqrt(backend.w_stack)
        self.d = np.concatenate([backend.data.real, backend.data.imag])
        self.kinds = list(kinds)                     # "point" / "gaussian"
        self.fixed_position = list(fixed_position)
        self.holds = [dict(h or {}) for h in (holds or [None] * len(self.kinds))]
        self.free_shape = [[k for k in range(3) if k not in h] if kind == "gaussian" else []
                           for kind, h in zip(self.kinds, self.holds)]
        self.slices, i = [], 0
        for fixed, free in zip(self.fixed_position, self.free_shape):
            n = 1 + (0 if fixed else 2) + len(free)
            self.slices.append(slice(i, i + n))
            i += n
        self.n_params = i
        self._held_pos = [None] * len(self.kinds)

    def pack(self, comps) -> np.ndarray:
        """comps: [(flux, (y, x), shape or None)] -> parameter vector."""
        out = []
        for (a, (y, x), shape), fixed, free in zip(comps, self.fixed_position,
                                                   self.free_shape):
            out.append(a)
            if not fixed:
                out += [y, x]
            for k in free:
                out.append(shape[k] if k == 2 else np.log(shape[k]))
        self._held_pos = [c[1] for c in comps]
        return np.array(out, dtype=float)

    def unpack(self, theta):
        comps = []
        for sl, fixed, kind, held, free, hold in zip(
                self.slices, self.fixed_position, self.kinds, self._held_pos,
                self.free_shape, self.holds):
            q = theta[sl]
            a, j = q[0], 1
            if fixed:
                pos = held
            else:
                pos, j = (q[1], q[2]), 3
            shape = None
            if kind == "gaussian":
                vals = dict(hold)
                for k in free:
                    vals[k] = float(q[j]) if k == 2 else float(np.exp(q[j]))
                    j += 1
                shape = (float(vals[0]), float(vals[1]), float(vals[2]))
            comps.append((float(a), (float(pos[0]), float(pos[1])), shape))
        return comps

    def _parts(self, theta):
        n = self.uv.shape[0]
        out = []
        for a, (y, x), shape in self.unpack(theta):
            ph = -2.0 * np.pi * ARCSEC_TO_RAD * (self.uv[:, 0] * x + self.uv[:, 1] * y)
            c, sn = np.cos(ph), np.sin(ph)
            t = um = un = None
            if shape is not None:
                t, um, un = gaussian_uv_terms(self.uv, shape)
            out.append((a, ph, c, sn, t, um, un, shape))
        return n, out

    def residual(self, theta) -> np.ndarray:
        n, parts = self._parts(theta)
        m = np.zeros(2 * n)
        for a, _, c, sn, t, *_ in parts:
            tt = 1.0 if t is None else t
            m[:n] += a * tt * c
            m[n:] += a * tt * sn
        return self.sw * (self.d - m)

    def jacobian(self, theta) -> np.ndarray:
        n, parts = self._parts(theta)
        J = np.empty((2 * n, self.n_params))
        two_pi_arc = 2.0 * np.pi * ARCSEC_TO_RAD
        for (a, _, c, sn, t, um, un, shape), sl, fixed, free in zip(
                parts, self.slices, self.fixed_position, self.free_shape):
            tt = np.ones(n) if t is None else t
            col = np.concatenate([tt * c, tt * sn])           # d model / d flux
            k = sl.start
            J[:, k] = col
            k += 1
            if not fixed:
                dphase = np.concatenate([-tt * sn, tt * c])  # d[cos; sin]/d phase
                J[:, k] = a * dphase * np.tile(-two_pi_arc * self.uv[:, 1], 2)   # y
                J[:, k + 1] = a * dphase * np.tile(-two_pi_arc * self.uv[:, 0], 2)  # x
                k += 2
            if shape is not None:
                sa, sb = shape[0] * ARCSEC_TO_RAD, shape[1] * ARCSEC_TO_RAD
                f2 = -4.0 * np.pi ** 2
                d = {0: f2 * sa * sa * um ** 2, 1: f2 * sb * sb * un ** 2,
                     2: f2 * (sb * sb - sa * sa) * um * un}
                for q in free:
                    J[:, k] = a * col * np.tile(d[q], 2)
                    k += 1
        return -self.sw[:, None] * J


def _joint_fit(backend, geometry, comps, kinds, fixed, holds=None, notes=None):
    """Trust-region least squares of all components from ``comps`` =
    [(flux or None, (y, x), shape or None)] (a missing flux starts at the
    linear solution). Returns (model, result)."""
    from scipy.optimize import least_squares

    model = JointModel(backend, kinds, fixed, holds)
    pos = [c[1] for c in comps]
    shapes = []
    for c, kind, h in zip(comps, kinds, model.holds):
        if kind != "gaussian":
            shapes.append(None)
            continue
        sh = list(c[2])
        for k, v in h.items():
            sh[k] = v
        shapes.append(tuple(sh))
    C, Dp = backend.terms(pos, shapes)
    a_lin = np.linalg.solve(C + np.eye(len(pos)) * GRAM_RIDGE * np.trace(C), Dp)
    a0 = [a_lin[i] if c[0] is None else c[0] for i, c in enumerate(comps)]
    theta0 = model.pack(list(zip(a0, pos, shapes)))
    lo = np.full(model.n_params, -np.inf)
    floor = np.log(SIZE_FLOOR_PIXELS * float(geometry.pixel_scale))
    for sl, fx, free in zip(model.slices, model.fixed_position, model.free_shape):
        k = sl.start + 1 + (0 if fx else 2)
        for q in free:
            if q < 2:
                lo[k] = floor
            k += 1
    theta0 = np.maximum(theta0, lo + 1e-9)
    res = least_squares(model.residual, theta0, jac=model.jacobian, method="trf",
                        bounds=(lo, np.inf), x_scale="jac", xtol=1e-12, ftol=1e-12,
                        gtol=1e-12, max_nfev=500)
    if not res.success and notes is not None:
        notes.append(f"joint fit: {res.message}")
    return model, res


def resolution_test(backend, geometry, model, res, index, inflation, beam, notes=None):
    """Is Gaussian `index` resolved, and if not, how large could it be?

    CASA's imfit answers this in the image plane: it deconvolves the beam
    from the fitted size and from the eight combinations of size and PA
    shifted by their errors, flags a component that cannot be deconvolved
    as consistent with a point (`ispoint`), and gives the largest of the
    deconvolved trial sizes as an upper limit. Here the size is already
    intrinsic (the fit is to the visibilities), but its Fisher errors are
    wrong exactly where this question is asked -- at a size of zero, a
    boundary -- so the test is a likelihood ratio instead. With every other
    parameter of every component re-fitted each time:

    * Delta chi^2 (point -> Gaussian) below `RESOLVED_DELTA_CHI2`: **not
      resolved**, and the 95% upper limit on a circular FWHM is where chi^2
      rises `UPPER_LIMIT_DELTA_CHI2` above the point fit's;
    * else Delta chi^2 (line -> Gaussian), the minor axis held at zero, below
      it: **resolved along the major axis only**, with a 95% upper limit on
      the minor axis the same way;
    * else **resolved**.

    Delta chi^2 are divided by the chi^2/N inflation, as the errors are
    multiplied by it. Returns (resolved, d_point, d_line, fwhm_limit,
    fwhm_minor_limit).
    """
    comps = model.unpack(res.x)
    kinds, fixed = list(model.kinds), list(model.fixed_position)
    base = float(res.fun @ res.fun)
    a, pos, shape = comps[index]
    major = 0 if shape[0] >= shape[1] else 1
    tiny = SIZE_FLOOR_PIXELS * float(geometry.pixel_scale)

    def chi2_with(kind, hold=None):
        k2 = list(kinds)
        k2[index] = kind
        holds = [None] * len(kinds)
        holds[index] = hold
        c2 = list(comps)
        c2[index] = (a, pos, None if kind == "point" else shape)
        _, r = _joint_fit(backend, geometry, c2, k2, fixed, holds)
        return float(r.fun @ r.fun)

    chi_point = chi2_with("point")
    chi_line = chi2_with("gaussian", {1 - major: tiny})
    d_point = (chi_point - base) / inflation
    d_line = (chi_line - base) / inflation

    def upper_limit(chi2_of, ref, start):
        """Smallest FWHM at which chi^2 is `UPPER_LIMIT_DELTA_CHI2` above
        ``ref``, by doubling then bisection on sigma."""
        delta = lambda sig: (chi2_of(sig) - ref) / inflation - UPPER_LIMIT_DELTA_CHI2
        lo, hi = 0.0, max(start, tiny * 10)
        for _ in range(12):
            if delta(hi) > 0:
                break
            lo, hi = hi, 2.0 * hi
        else:
            return None
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if delta(mid) < 0 else (lo, mid)
        return float(FWHM_PER_SIGMA * hi)

    beam_sigma = float(beam) / FWHM_PER_SIGMA
    limit = minor_limit = None
    if d_point < RESOLVED_DELTA_CHI2:
        resolved = "no"
        limit = upper_limit(
            lambda sig: chi2_with("gaussian", {0: sig, 1: sig, 2: 0.0}), chi_point,
            max(shape[major], 0.25 * beam_sigma))
    elif d_line < RESOLVED_DELTA_CHI2:
        resolved = "major axis only"
        minor_limit = upper_limit(
            lambda sig: chi2_with("gaussian", {1 - major: sig}), base,
            max(shape[1 - major], 0.25 * beam_sigma))
    else:
        resolved = "yes"
    return resolved, float(d_point), float(d_line), limit, minor_limit


def fit_with_gaussians(
    backend,
    geometry,
    gaussians,
    positions=None,
    detect: bool = False,
    significance: float = DEFAULT_SIGNIFICANCE,
    max_points: int = DEFAULT_MAX_POINTS,
    refine: bool = True,
    beam_fwhm: float | None = None,
) -> PointOnlyFit:
    """Gaussian components (and any points), all parameters fitted jointly.

    ``gaussians``: [(dRA, dDec, fwhm_major or None, fwhm_minor or None, pa_deg
    or None)] -- starting values; a missing size starts at half the beam.
    ``positions``: points as in `fit_points_only` (held fixed when ``refine``
    is False). ``detect``: also look for points, with the Gaussians' (and
    given points') geometry held while searching, then everything refitted.

    Joint Levenberg-Marquardt (`scipy.optimize.least_squares`, trust region)
    on the noise-weighted visibility residual with an analytic Jacobian; the
    covariance is (J^T J)^-1 at the solution (the Gauss-Newton inverse
    Fisher matrix), times max(1, chi^2/N).
    """
    from scipy.optimize import least_squares

    beam = float(beam_fwhm) if beam_fwhm else 2.0 * float(geometry.pixel_scale)
    notes: list = []
    pts = [sky_to_grid(*p) for p in (positions or [])]
    gauss0 = []
    for g in gaussians:
        d_ra, d_dec = g[0], g[1]
        maj = g[2] if len(g) > 2 and g[2] else 0.5 * beam
        mnr = g[3] if len(g) > 3 and g[3] else maj
        maj, mnr = max(maj, mnr), min(maj, mnr)
        mnr = min(mnr, 0.9 * maj)              # off round, so theta is not degenerate
        pa = g[4] if len(g) > 4 and g[4] is not None else 0.0
        gauss0.append((sky_to_grid(d_ra, d_dec),
                       (maj / FWHM_PER_SIGMA, mnr / FWHM_PER_SIGMA, np.radians(pa))))

    def joint(points, point_user, gauss):
        kinds = ["point"] * len(points) + ["gaussian"] * len(gauss)
        fixed = [not refine and u for u in point_user] + [False] * len(gauss)
        comps = ([(None, p, None) for p in points] + [(None, g[0], g[1]) for g in gauss])
        return _joint_fit(backend, geometry, comps, kinds, fixed, notes=notes)

    user = [True] * len(pts)
    model, res = joint(pts, user, gauss0)
    comps = model.unpack(res.x)
    if detect:
        k0 = len(pts)
        held = [(c[1], c[2]) for c in comps]
        system = PointOnlySystem(backend, fixed=held)
        found = fit_points_only(system, geometry, positions=None, significance=significance,
                                max_points=max_points, refine=refine, beam_fwhm=beam)
        notes += found.log
        new = list(found.grid_positions)
        pts = [c[1] for c in comps[:k0]] + new
        user = user + [False] * len(new)
        gauss0 = [(c[1], c[2]) for c in comps[k0:]]
        model, res = joint(pts, user, gauss0)
        comps = model.unpack(res.x)
        # with every geometry free, a detected point can lose its flux to a
        # Gaussian it was compensating for: drop the weakest below the cut
        while True:
            J = res.jac
            cov = np.linalg.pinv(J.T @ J) * max(1.0, float(res.fun @ res.fun) / backend.n_data)
            snr = [(comps[i][0] / np.sqrt(cov[model.slices[i].start, model.slices[i].start]), i)
                   for i in range(len(pts)) if not user[i]]
            if not snr or min(snr)[0] >= significance:
                break
            weak, i = min(snr)
            notes.append(f"point at {grid_to_sky(*comps[i][1])} dropped: {weak:.1f} sigma "
                         "once the Gaussians were refitted")
            pts = [c[1] for j, c in enumerate(comps[:len(pts)]) if j != i]
            user = [u for j, u in enumerate(user) if j != i]
            gauss0 = [(c[1], c[2]) for c in comps[len(pts) + 1:]]
            model, res = joint(pts, user, gauss0)
            comps = model.unpack(res.x)

    r = res.fun
    chi2 = float(r @ r)
    inflation = max(1.0, chi2 / backend.n_data)
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J) * inflation
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J) * inflation
        notes.append("J^T J is singular: errors from its pseudo-inverse")

    resolution = {}
    for i, kind in enumerate(model.kinds):
        if kind == "gaussian":
            resolution[i] = resolution_test(backend, geometry, model, res, i, inflation,
                                            beam, notes)
    points, gaussians_out = [], []
    labels = []
    for i, ((a, (y, x), shape), sl, kind, fixed) in enumerate(
            zip(comps, model.slices, model.kinds, model.fixed_position)):
        c = cov[sl, sl]
        d_ra, d_dec = grid_to_sky(y, x)
        if fixed:
            sy = sx = rho = None
        else:
            sy, sx = np.sqrt(c[1, 1]), np.sqrt(c[2, 2])
            rho = -c[1, 2] / (sy * sx) if sy > 0 and sx > 0 else 0.0
        if kind == "point":
            labels += [f"point{i}:flux"] + ([] if fixed else [f"point{i}:y", f"point{i}:x"])
            points.append(FittedPoint(
                d_ra=d_ra, d_dec=d_dec, flux=a, flux_error=float(np.sqrt(c[0, 0])),
                d_ra_error=None if sx is None else float(sx),
                d_dec_error=None if sy is None else float(sy),
                ra_dec_correlation=None if rho is None else float(rho),
                user_supplied=user[i]))
            continue
        labels += [f"gaussian{i}:{q}" for q in ("flux", "y", "x", "ln_sigma_a",
                                                 "ln_sigma_b", "theta")]
        sa, sb, th = shape
        e_a, e_b, e_t = np.sqrt(np.diag(c)[3:6])
        if sb > sa:                                   # b is the major axis
            sa, sb, e_a, e_b, th = sb, sa, e_b, e_a, th + np.pi / 2.0
        gaussians_out.append(FittedGaussian(
            d_ra=d_ra, d_dec=d_dec, flux=a, flux_error=float(np.sqrt(c[0, 0])),
            d_ra_error=float(sx), d_dec_error=float(sy), ra_dec_correlation=float(rho),
            fwhm_major=float(FWHM_PER_SIGMA * sa),
            fwhm_major_error=float(FWHM_PER_SIGMA * sa * e_a),
            fwhm_minor=float(FWHM_PER_SIGMA * sb),
            fwhm_minor_error=float(FWHM_PER_SIGMA * sb * e_b),
            pa_deg=float(np.degrees(th) % 180.0), pa_error_deg=float(np.degrees(e_t)),
            resolved=resolution[i][0], delta_chi2_vs_point=resolution[i][1],
            delta_chi2_vs_line=resolution[i][2], fwhm_upper_limit=resolution[i][3],
            fwhm_minor_upper_limit=resolution[i][4]))
    point_comps = [c for c, kind in zip(comps, model.kinds) if kind == "point"]
    gauss_comps = [c for c, kind in zip(comps, model.kinds) if kind == "gaussian"]
    fit = PointOnlyFit(
        points=points, grid_positions=[c[1] for c in point_comps],
        amplitudes=np.array([c[0] for c in point_comps]), covariance=cov,
        chi_squared=chi2, n_data=backend.n_data, chi2_inflation=inflation,
        positions_fitted=True, log=notes, gaussians=gaussians_out,
        gaussian_components=[(c[1], c[2]) for c in gauss_comps],
        gaussian_amplitudes=np.array([c[0] for c in gauss_comps]),
        covariance_labels=labels)
    return fit


def restore_gaussians(image_shape, pixel_scale, gaussians, beam, existing=None):
    """Add Gaussians, convolved with the restoring beam, to an image [Jy/beam]:
    F (2 pi s_b1 s_b2) N(r; mu, S_g + S_b) -- a point (S_g = 0) peaks at F."""
    ny, nx = image_shape
    out = np.zeros(image_shape) if existing is None else np.array(existing, dtype=float)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    gy = ((ny - 1) / 2.0 - yy) * pixel_scale
    gx = (xx - (nx - 1) / 2.0) * pixel_scale
    sb = (beam.bmaj_arcsec / FWHM_PER_SIGMA, beam.bmin_arcsec / FWHM_PER_SIGMA,
          np.radians(beam.bpa_deg))
    area = 2.0 * np.pi * sb[0] * sb[1]
    S_b = _cov_yx(sb)
    for g in gaussians:
        S = S_b + _cov_yx(g.shape)
        out += g.flux * area * _normal(gy + 0.0, gx, (g.d_dec, -g.d_ra), S)
    return out


def gaussian_model_image(image_shape, pixel_scale, gaussians, supersample: int = 5):
    """Each Gaussian's flux on the pixel grid [Jy/pixel], averaged over
    ``supersample``^2 points per pixel."""
    ny, nx = image_shape
    out = np.zeros(image_shape)
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    for g in gaussians:
        S = _cov_yx(g.shape)
        acc = np.zeros(image_shape)
        for dy in off:
            for dx in off:
                gy = ((ny - 1) / 2.0 - (yy + dy)) * pixel_scale
                gx = ((xx + dx) - (nx - 1) / 2.0) * pixel_scale
                acc += _normal(gy, gx, (g.d_dec, -g.d_ra), S)
        out += g.flux * acc / supersample ** 2 * pixel_scale ** 2
    return out


def _cov_yx(shape) -> np.ndarray:
    """Covariance in grid (y, x) arcsec^2 of (sigma_maj, sigma_min, theta)."""
    s_a, s_b, th = shape
    e_a = np.array([np.cos(th), -np.sin(th)])
    e_b = np.array([np.sin(th), np.cos(th)])
    return s_a ** 2 * np.outer(e_a, e_a) + s_b ** 2 * np.outer(e_b, e_b)


def _normal(gy, gx, mu, S) -> np.ndarray:
    Si = np.linalg.inv(S)
    dy, dx = gy - mu[0], gx - mu[1]
    q = Si[0, 0] * dy * dy + 2 * Si[0, 1] * dy * dx + Si[1, 1] * dx * dx
    return np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(np.linalg.det(S)))


# ---------------------------------------------------------------------------
# positions on the sky, the primary beam
# ---------------------------------------------------------------------------

def _locate(points, geometry, meta, frequency_hz, dish, pb_factor):
    """Phase-centre offsets, RA/Dec through the products' WCS, and the
    primary-beam-corrected flux (the PB is centred on the pointing)."""
    from astropy.wcs import WCS

    from . import primary_beam
    from .products import build_header

    y0, x0 = (float(v) for v in (meta.get("image_centre_offset_arcsec") or (0.0, 0.0)))
    n = int(geometry.shape_native[0])
    pix = float(geometry.pixel_scale)
    wcs = None
    if meta.get("phase_centre_ra_deg") is not None:
        wcs = WCS(build_header(n, pix, meta, "Jy/beam"))
    fwhm = primary_beam.pb_fwhm_arcsec(frequency_hz, dish, pb_factor) if dish else None
    for p in points:
        x, y = -p.d_ra, p.d_dec                       # grid (y, x) about the image centre
        p.d_ra_phase_centre = -(x + x0)
        p.d_dec_phase_centre = y + y0
        if wcs is not None:
            col = (n - 1) / 2.0 + x / pix             # FITS: row 0 at the bottom
            row = (n - 1) / 2.0 + y / pix
            sky = wcs.pixel_to_world(col, row)
            p.ra_deg, p.dec_deg = float(sky.ra.deg), float(sky.dec.deg)
        if fwhm:
            r2 = p.d_ra_phase_centre ** 2 + p.d_dec_phase_centre ** 2
            pb = float(np.exp(-0.5 * r2 / (fwhm / 2.3548) ** 2))
            p.primary_beam = pb
            p.flux_pbcor = p.flux / pb
            p.flux_error_pbcor = p.flux_error / pb
    return fwhm


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@dataclass
class PointRunResult:
    geometry: object
    fit: PointOnlyFit
    beam: object
    rms: float
    dirty_image: np.ndarray       # native, Jy/beam
    residual: np.ndarray          # native, Jy/beam
    restored: np.ndarray          # points convolved with the restoring beam, Jy/beam
    parameters: dict
    written: dict

    @property
    def points(self) -> list:
        return self.fit.points


def run_points(
    dataset,
    fov: float,
    out="pyuvimage_out",
    pixel_scale="auto",
    oversample: int = 2,
    image_centre="0,0",
    positions=None,
    gaussians=None,
    detect: bool | None = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    max_points: int = DEFAULT_MAX_POINTS,
    refine: bool = True,
    pb_correction: bool = True,
    dish_diameter: float | None = None,
    pb_factor: float | None = None,
    streaming="auto",
    chunk_k: int | None = None,
    kernel_cache: str | None = None,
    reload: bool = False,
    transformer: str = "auto",
    mask_shape: str = "square",
    write: bool = True,
) -> PointRunResult:
    """`run(..., points_only=True)`: see the module notes.

    ``positions`` are image (x, y) offsets [arcsec] (the `--point`
    convention), or None to detect. ``gaussians``: [(x, y[, fwhm_major[,
    fwhm_minor, pa_deg]])] in the same axes, starting values for elliptical
    Gaussian components (`fit_with_gaussians`; in memory). ``detect``: look
    for points as well; by default only when neither points nor Gaussians
    are given.
    """
    import autogalaxy as ag

    from . import api, fitting, primary_beam
    from . import streaming as stm
    from .beam import fit_beam
    from .grids import resolve_geometry
    from .pointsource import image_to_sky
    from .uvdata import MultiSpwUVData, UVData, read_dataset

    t0 = time.time()
    pb_factor = primary_beam.DEFAULT_PB_FACTOR if pb_factor is None else float(pb_factor)
    on_disk = not isinstance(dataset, (UVData, MultiSpwUVData))
    if streaming is True and not stm.nufftax_available():
        raise RuntimeError("streamed point components need nufftax; --no-streaming "
                           "fits them from the visibilities in memory")
    if gaussians and streaming is True:
        raise NotImplementedError("Gaussian components are fitted in memory only: "
                                  "drop --streaming")
    stream = (streaming is True or (streaming == "auto" and on_disk and not gaussians
                                    and stm.nufftax_available()))
    if streaming == "auto" and on_disk and not stream:
        logger.info("streaming auto -> in memory: %s",
                    "Gaussian components need the visibilities" if gaussians
                    else "nufftax is not installed")
    if detect is None:
        detect = not positions and not gaussians

    if stream:
        from .uvdata import describe_pooling, reim_asymmetry

        chunk_k = int(chunk_k or stm.STREAM_CHUNK_K)
        header = stm.scan_header(dataset, chunk_k)
        geometry = resolve_geometry(
            fov_arcsec=fov, max_baseline_wavelengths=header.max_baseline_wavelengths,
            pixel_scale=pixel_scale, mesh_shape=None, oversample=oversample,
            effective_baseline_wavelengths=header.baseline_percentile_wavelengths(
                api.BASELINE_PERCENTILE))
        level, message = describe_pooling(reim_asymmetry(header.noise))
        pool = level is not None
        if pool:
            logger.log(level, message)
        mask = fitting.make_mask(geometry, mask_shape)
        cache_dir = Path(kernel_cache) if kernel_cache else (Path(out).parent if out else None)
        shift = api._streamed_centre(dataset, header, image_centre, fov, dish_diameter,
                                     chunk_k, cache_dir, reuse_cache=not reload)
        if shift is not None:
            header = header.with_centre(*shift)
            pool = True
        terms = stm.sparse_terms_for(
            dataset, geometry, mask, ag.TransformerDFT, cache_dir=cache_dir,
            chunk_k=chunk_k, mask_shape=mask_shape, pool_noise=pool,
            reuse_cache=not reload, centre=shift, point_oversample=api.POINT_OVERSAMPLE)
        stub = stm.stub_dataset_from_terms(terms, geometry, mask)
        imager = fitting.imager_for(stub)
        backend = GridPoints(terms, imager)
        meta, freq = header.meta, float(header.central_frequency)
        n_vis = int(terms.n_vis)
        dirty = np.asarray(imager.dirty_image_of_data(), dtype=float)
    else:
        uvd = dataset if not on_disk else read_dataset(dataset)
        uvd = api._recentre(uvd, image_centre, fov, dish_diameter, transformer)
        geometry = resolve_geometry(
            fov_arcsec=fov, max_baseline_wavelengths=uvd.max_baseline_wavelengths,
            pixel_scale=pixel_scale, mesh_shape=None, oversample=oversample,
            effective_baseline_wavelengths=uvd.baseline_percentile_wavelengths(
                api.BASELINE_PERCENTILE))
        uv, d, nz = uvd.flattened()
        ds = fitting.make_dataset(uv, d, nz, geometry, transformer=transformer,
                                  mask_shape=mask_shape)
        imager = fitting.imager_for(ds)
        backend = VisibilityPoints(ds, imager)
        meta, freq = uvd.meta, float(uvd.central_frequency)
        n_vis = int(d.size)
        dirty = np.asarray(imager.dirty_image(d), dtype=float)
    logger.info("points only: %d visibilities, image grid %dx%d at %.4g\"/pix (%s)",
                n_vis, *geometry.shape_native, geometry.pixel_scale,
                "streamed" if stream else "in memory")

    beam = fit_beam(imager.dirty_beam, geometry.pixel_scale)
    beam_fwhm = float(np.sqrt(beam.bmaj_arcsec * beam.bmin_arcsec))
    rms = float(imager.rms)
    sky_positions = [image_to_sky(*q) for q in positions] if positions else None
    sky_gaussians = [(*image_to_sky(g[0], g[1]), *g[2:]) for g in (gaussians or [])]
    what = []
    if sky_positions:
        what.append(f"{len(sky_positions)} given point(s)")
    if sky_gaussians:
        what.append(f"{len(sky_gaussians)} Gaussian(s)")
    if detect:
        what.append(f"points detected above {significance:.1f} sigma, at most {max_points}")
    logger.info("fitting components only: %s...", ", ".join(what))
    if sky_gaussians:
        fit = fit_with_gaussians(backend, geometry, sky_gaussians, positions=sky_positions,
                                 detect=detect, significance=significance,
                                 max_points=max_points, refine=refine, beam_fwhm=beam_fwhm)
    else:
        system = PointOnlySystem(backend)
        fit = fit_points_only(system, geometry,
                              positions=None if detect else sky_positions,
                              significance=significance, max_points=max_points,
                              refine=refine, beam_fwhm=beam_fwhm)
    for note in fit.log:
        logger.info("  %s", note)
    dish = dish_diameter or meta.get("dish_diameter_m")
    if pb_correction and not dish:
        logger.warning("no dish diameter known: point fluxes are not primary-beam "
                       "corrected (pass dish_diameter=... to enable)")
    pb_fwhm = _locate(fit.points + fit.gaussians, geometry, meta, freq,
                      dish if pb_correction else None, pb_factor)

    from .pointsource import restore_points

    if fit.gaussians:
        residual = backend.residual_image(
            list(fit.grid_positions) + [c[0] for c in fit.gaussian_components],
            np.concatenate([fit.amplitudes, fit.gaussian_amplitudes]),
            [None] * len(fit.grid_positions) + [c[1] for c in fit.gaussian_components])
    else:
        residual = backend.residual_image(fit.grid_positions, fit.amplitudes)
    restored = restore_points(geometry.shape_native, geometry.pixel_scale, fit.points, beam)
    restored = restore_gaussians(geometry.shape_native, geometry.pixel_scale, fit.gaussians,
                                 beam, existing=restored)
    logger.info("components only: %d point(s), %d Gaussian(s), chi2/N %.4f%s, residual "
                "peak %.1f sigma, %.0f s",
                len(fit.points), len(fit.gaussians), fit.chi2_per_n,
                f" (errors inflated x{np.sqrt(fit.chi2_inflation):.3f})"
                if fit.chi2_inflation > 1 else "",
                float(np.nanmax(np.abs(residual))) / rms, time.time() - t0)
    for i, p in enumerate(fit.points):
        pos_err = ("" if p.d_ra_error is None else
                   f" +- ({1e3 * p.d_ra_error:.2f}, {1e3 * p.d_dec_error:.2f}) mas")
        pb = "" if p.flux_pbcor is None else f"; PB-corrected {p.flux_pbcor:.5g} +- {p.flux_error_pbcor:.2g} Jy"
        logger.info("  point %d: (dRA, dDec) = (%.4f, %.4f)\"%s, %.5g +- %.2g Jy (%.1f sigma)%s",
                    i + 1, p.d_ra, p.d_dec, pos_err, p.flux, p.flux_error, p.significance, pb)
    for i, g in enumerate(fit.gaussians):
        pb = "" if g.flux_pbcor is None else f"; PB-corrected {g.flux_pbcor:.5g} Jy"
        logger.info("  Gaussian %d: (dRA, dDec) = (%.4f, %.4f)\" +- (%.2f, %.2f) mas, %.5g "
                    "+- %.2g Jy, FWHM %.4f +- %.4f x %.4f +- %.4f\", PA %.1f +- %.1f deg%s",
                    i + 1, g.d_ra, g.d_dec, 1e3 * g.d_ra_error, 1e3 * g.d_dec_error, g.flux,
                    g.flux_error, g.fwhm_major, g.fwhm_major_error, g.fwhm_minor,
                    g.fwhm_minor_error, g.pa_deg, g.pa_error_deg, pb)
        if g.resolved == "no":
            logger.info("    not resolved: consistent with a point (Delta chi^2 %.1f over a "
                        "point); FWHM < %.4g\" (95%%)", g.delta_chi2_vs_point,
                        g.fwhm_upper_limit if g.fwhm_upper_limit is not None else float("nan"))
        elif g.resolved == "major axis only":
            logger.info("    resolved along the major axis only (Delta chi^2 %.1f over a "
                        "line); minor FWHM < %.4g\" (95%%)", g.delta_chi2_vs_line,
                        g.fwhm_minor_upper_limit if g.fwhm_minor_upper_limit is not None
                        else float("nan"))
        else:
            logger.info("    resolved (Delta chi^2 %.1f over a point, %.1f over a line)",
                        g.delta_chi2_vs_point, g.delta_chi2_vs_line)

    parameters = {
        "model": "points and Gaussians" if fit.gaussians else "points only",
        "data": {"n_samples": n_vis, "n_data": fit.n_data, "streamed": bool(stream),
                 "central_frequency_hz": freq},
        "geometry": {"fov_arcsec": float(geometry.fov_arcsec),
                     "pixel_scale_arcsec": float(geometry.pixel_scale),
                     "shape": list(geometry.shape_native)},
        "beam": {"bmaj_arcsec": beam.bmaj_arcsec, "bmin_arcsec": beam.bmin_arcsec,
                 "bpa_deg": beam.bpa_deg},
        "rms_jy_beam": rms,
        "fit_quality": {"chi_squared": fit.chi_squared, "n_data": fit.n_data,
                        "chi2_per_n": fit.chi2_per_n,
                        "residual_peak_sigma": float(np.nanmax(np.abs(residual))) / rms},
        "uncertainty": {
            "type": "statistical",
            "method": ("(J^T J)^-1 of the joint least-squares fit (Gauss-Newton inverse "
                       "Fisher matrix)" if fit.gaussians else
                       "inverse Fisher matrix of chi^2 in flux and position"
                       if fit.positions_fitted else "flux covariance at fixed positions"),
            "chi2_inflation": fit.chi2_inflation,
        },
        "primary_beam": {"corrected": bool(pb_fwhm), "fwhm_arcsec": pb_fwhm,
                         "pb_factor": pb_factor if pb_fwhm else None},
        "detection": {"significance": float(significance), "max_points": int(max_points),
                      "refine": bool(refine), "notes": fit.log},
        "point_sources": {
            "n_points": len(fit.points),
            "total_flux_jy": float(sum(p.flux for p in fit.points)),
            "points": [p.as_dict() for p in fit.points],
            "covariance_order": fit.covariance_labels or (
                (["flux"] * len(fit.points) + ["y"] * len(fit.points)
                 + ["x"] * len(fit.points)) if fit.positions_fitted
                else ["flux"] * len(fit.points)),
            "covariance": np.asarray(fit.covariance).tolist(),
        },
    }
    if fit.gaussians:
        parameters["point_sources"]["gaussians"] = [g.as_dict() for g in fit.gaussians]
        parameters["point_sources"]["n_gaussians"] = len(fit.gaussians)
        parameters["point_sources"]["total_flux_jy"] += float(sum(g.flux for g in fit.gaussians))
    written = {}
    if write:
        written = _write(out, geometry, meta, freq, beam, rms, dirty, residual, restored,
                         fit, parameters)
    return PointRunResult(geometry=geometry, fit=fit, beam=beam, rms=rms,
                          dirty_image=dirty, residual=residual, restored=restored,
                          parameters=parameters, written=written)


def _write(out, geometry, meta, freq, beam, rms, dirty, residual, restored, fit,
           parameters) -> dict:
    from astropy.io import fits

    from .products import add_points, build_header, to_fits_orientation

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(geometry.shape_native[0])
    pix = float(geometry.pixel_scale)
    freqs = np.array([freq])

    def hdr(bunit, **extra):
        return build_header(n, pix, meta, bunit, beam=beam, frequencies_hz=freqs,
                            extra=extra or None)

    written = {}

    def w(name, data, header):
        fits.writeto(out / name, np.asarray(to_fits_orientation(data), dtype=np.float32),
                     header, overwrite=True)
        written[name] = out / name

    model = add_points(gaussian_model_image(geometry.shape_native, pix, fit.gaussians),
                       fit.points, pix)
    w("model.fits", model, hdr("Jy/pixel", PTFLUX=(
        float(np.sum(fit.amplitudes) + np.sum(fit.gaussian_amplitudes)),
        "total component flux [Jy]")))
    w("dirty_image.fits", dirty, hdr("Jy/beam"))
    w("model_reconvolved.fits", restored, hdr("Jy/beam", RMS=(rms, "image-plane rms [Jy/beam]")))
    w("residual.fits", residual / rms, hdr("sigma", RMS=(rms, "image-plane rms [Jy/beam]")))
    (out / "fit_parameters.json").write_text(json.dumps(parameters, indent=2))
    written["fit_parameters.json"] = out / "fit_parameters.json"
    # the same list where a mesh fit puts its points
    (out / "point_sources.json").write_text(
        json.dumps(parameters["point_sources"], indent=2))
    written["point_sources.json"] = out / "point_sources.json"
    _summary(out / "summary.png", geometry, dirty, restored, residual / rms, fit, beam, rms)
    written["summary.png"] = out / "summary.png"
    logger.info("products written to %s", out)
    return written


def _summary(path, geometry, dirty, restored, resid, fit, beam, rms) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = geometry.fov_arcsec / 2.0
    ext = [half, -half, -half, half]
    rmax = max(float(np.nanmax(np.abs(resid))), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    panels = [(dirty, "dirty image", "Jy/beam", "inferno", None),
              (restored, f"{len(fit.points)} point(s)"
               + (f", {len(fit.gaussians)} Gaussian(s)" if fit.gaussians else "")
               + ", restored", "Jy/beam", "inferno", None),
              (resid, f"residual (peak {rmax:.1f}$\\sigma$, rms {np.nanstd(resid):.2f}$\\sigma$)",
               r"$\sigma$", "RdBu_r", (-rmax, rmax))]
    for ax, (img, title, unit, cmap, clim) in zip(axes, panels):
        im = ax.imshow(np.asarray(img), origin="upper", extent=ext, cmap=cmap)
        if clim is not None:
            im.set_clim(*clim)
        for p in fit.points:
            ax.plot(p.d_ra, p.d_dec, "o", mfc="none", mec="cyan", ms=11, mew=1.3)
        for g in fit.gaussians:
            from matplotlib.patches import Ellipse

            # FWHM ellipse in (dRA, dDec) data coordinates: the height (major)
            # axis turned from north toward +dRA (east) by PA is a clockwise
            # rotation in data space, hence -PA
            ax.add_patch(Ellipse((g.d_ra, g.d_dec), g.fwhm_minor, g.fwhm_major,
                                 angle=-g.pa_deg, fill=False, ec="lime", lw=1.2,
                                 ls="-" if g.resolved == "yes" else "--"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dRA [arcsec]")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(unit, fontsize=9)
    axes[0].set_ylabel("dDec [arcsec]")
    fig.suptitle(f"{'components' if fit.gaussians else 'points'} only | beam {beam.bmaj_arcsec:.3g}\" x {beam.bmin_arcsec:.3g}\" "
                 f"pa {beam.bpa_deg:.0f} deg | rms {rms:.3g} Jy/beam | "
                 f"chi2/N {fit.chi2_per_n:.4f}", fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)

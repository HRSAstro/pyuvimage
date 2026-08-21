"""Analytic point-source components, solved inside the same linear system.

A true point source has an exact, closed-form visibility::

    V(u, v) = A * exp(-2i pi (x u + y v))

so it needs no image grid at all.  That matters: representing a point by a
narrow profile on the pixel grid is only accurate when it happens to sit on a
pixel centre.  Measured against the analytic truth on our test data, a
nearest-pixel delta half a pixel off-centre gives a representation error of
chi^2/N = 31.5, and the best gridded Gaussian still gives ~1.9 -- i.e. an
error at or above the noise, for a source the fit is supposed to describe
perfectly.

The amplitude ``A`` enters linearly, so it is solved *simultaneously* with the
pixelized model rather than fitted afterwards; only the position is
non-linear, and that is refined by a small search.  autoarray exposes an
`operated_mapping_matrix_override` hook for exactly this, but its
interferometer inversion ignores it (it always calls the transformer), so the
augmented normal equations are assembled here.  The mesh curvature and data
vector are taken from the framework's own inversion and reproduce it to ~1e-16.

This is opt-in.  Point components are never added unless asked for, and an
auto-detected one is kept only if its amplitude clears a significance cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

logger = logging.getLogger("pyuvimage")

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

DEFAULT_SIGNIFICANCE = 5.0
DEFAULT_MAX_POINTS = 5


@dataclass
class PointSource:
    """One fitted point component."""

    d_ra: float          # offset from the phase centre [arcsec], +ve East
    d_dec: float         # offset from the phase centre [arcsec], +ve North
    flux: float          # Jy
    flux_error: float    # 1 sigma, Jy: statistical and systematic combined
    user_supplied: bool = False
    flux_error_stat: float = 0.0   # from the amplitude covariance alone
    flux_error_sys: float = 0.0    # from the choice of prior strength

    @property
    def significance(self) -> float:
        """Detection significance, on the *statistical* error only.

        The systematic below is a scale uncertainty on an already-detected
        source; folding it into the detection significance would make a real
        point source look marginal because we are unsure of its brightness.
        """
        err = self.flux_error_stat or self.flux_error
        return abs(self.flux) / err if err > 0 else np.inf

    def as_dict(self) -> dict:
        return {
            "d_ra_arcsec": float(self.d_ra),
            "d_dec_arcsec": float(self.d_dec),
            "flux_jy": float(self.flux),
            "flux_error_jy": float(self.flux_error),
            "flux_error_stat_jy": float(self.flux_error_stat),
            "flux_error_sys_jy": float(self.flux_error_sys),
            "significance": float(self.significance),
            "position": "user" if self.user_supplied else "auto-detected",
        }


def sky_to_grid(d_ra: float, d_dec: float) -> tuple[float, float]:
    """(dRA, dDec) offsets -> the (y, x) grid coordinates the transformer uses.

    RA increases eastward, which is towards *decreasing* image column, so the
    grid's x axis runs opposite to dRA.  Getting this backwards puts a source
    on the wrong side of the field, so it has its own round-trip test.
    """
    return float(d_dec), float(-d_ra)


def grid_to_sky(y: float, x: float) -> tuple[float, float]:
    return float(-x), float(y)


def gaussian_visibilities(
    uv_wavelengths: np.ndarray, y: float, x: float, sigma_arcsec: float
) -> np.ndarray:
    """Visibilities of a circular Gaussian -- also analytic.

    Used to ask whether a candidate is genuinely *unresolved*: a point and a
    small Gaussian fit an unresolved source equally well, but a resolved one
    (for instance the central cusp of an exponential disc, which the smoothed
    mesh cannot render) is fitted markedly better by the Gaussian.
    """
    s_rad = sigma_arcsec * ARCSEC_TO_RAD
    taper = np.exp(
        -2.0 * np.pi**2 * s_rad**2
        * (uv_wavelengths[:, 0] ** 2 + uv_wavelengths[:, 1] ** 2)
    )
    return taper * point_visibilities(uv_wavelengths, y, x)


def point_visibilities(uv_wavelengths: np.ndarray, y: float, x: float) -> np.ndarray:
    """Exact unit-amplitude visibilities of a point at grid position (y, x).

    Matches autoarray's DFT convention, phase = -2 pi (x u + y v) with x, y in
    radians (verified to machine precision against the transformer).
    """
    phase = -2.0 * np.pi * (
        x * ARCSEC_TO_RAD * uv_wavelengths[:, 0]
        + y * ARCSEC_TO_RAD * uv_wavelengths[:, 1]
    )
    return np.cos(phase) + 1j * np.sin(phase)


class AugmentedSystem:
    """The mesh's linear system, extensible with analytic point components.

    Holds the mesh curvature `F`, regularisation `H` and data vector `D` from
    the framework's inversion, and solves

        [[F + H,  B],      [s]     [D ]
         [B^T,    C]]  @   [a]  =  [Dp]

    with `B`, `C`, `Dp` the point columns' cross-, self- and data terms.  The
    mesh block is factorised once, so trying a new position costs only a small
    Schur-complement solve.
    """

    def __init__(self, inversion, dataset):
        self.inversion = inversion
        self.uv = np.asarray(dataset.uv_wavelengths)
        noise = np.asarray(dataset.noise_map)
        data = np.asarray(dataset.data)
        self.w_re, self.w_im = 1.0 / noise.real**2, 1.0 / noise.imag**2
        self.d_re, self.d_im = data.real, data.imag
        self.A = np.asarray(inversion.operated_mapping_matrix)
        self.F = np.asarray(inversion.curvature_matrix)
        self.D = np.asarray(inversion.data_vector)
        self.H = np.asarray(inversion.regularization_matrix)
        self.n_mesh = self.F.shape[0]
        self.n_data = 2 * len(self.d_re)
        self.h_scale = 1.0
        self._cho = cho_factor(self.F + self.H, lower=True, check_finite=False)
        self.chi2_const = float(
            np.sum(self.d_re**2 * self.w_re + self.d_im**2 * self.w_im)
        )

    def set_regularization_scale(self, factor: float) -> bool:
        """Rescale the regularisation, refactorising the mesh block.

        The prior enters as H = coefficient x C^-1, so the whole matrix scales
        with the coefficient and a retune costs one Cholesky.

        Returns False (leaving the previous factorisation in place) when the
        rescaled matrix is not positive definite.  That is a real regime, not
        a defect: with the prior weakened far enough, F alone is singular
        wherever the mesh has pixels the uv coverage does not constrain, which
        happens as soon as the mesh is comparable in size to the data.
        """
        try:
            cho = cho_factor(
                self.F + float(factor) * self.H, lower=True, check_finite=False
            )
        except (np.linalg.LinAlgError, ValueError):
            return False
        self.h_scale = float(factor)
        self._cho = cho
        return True

    # ---- the framework's own quadratic forms, rebuilt for arbitrary columns
    def _curv(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        return P.real.T @ (self.w_re[:, None] * Q.real) + P.imag.T @ (
            self.w_im[:, None] * Q.imag
        )

    def _dvec(self, P: np.ndarray) -> np.ndarray:
        return P.real.T @ (self.w_re * self.d_re) + P.imag.T @ (
            self.w_im * self.d_im
        )

    def columns_for(self, positions, sigma_arcsec: float = 0.0) -> np.ndarray:
        if sigma_arcsec > 0:
            return np.column_stack(
                [gaussian_visibilities(self.uv, y, x, sigma_arcsec)
                 for (y, x) in positions]
            )
        return np.column_stack(
            [point_visibilities(self.uv, y, x) for (y, x) in positions]
        )

    def scan(self, positions, ys, xs, chunk: int = 1024):
        """Exact amplitude and significance of adding *one* point, everywhere.

        This is the detector.  The obvious alternative -- take the peak of the
        residual dirty image -- fails badly, because by construction the mesh
        fit has already been driven to chi^2 = N and has absorbed much of the
        compact source into itself; what is left is sidelobe structure, and
        detection wanders off to it.  Here the question asked at every trial
        position is the right one: *how much would chi^2 drop if a point were
        added here*, with the mesh (and any points already accepted) free to
        re-adjust.

        With the mesh block eliminated, one extra column j gives exactly

            a_j = r_j / s_j,   Var(a_j) = 1 / s_j,   a_j / sd(a_j) = r_j/sqrt(s_j)

        where s_j = C_jj - b_j^T M^-1 b_j, r_j = Dp_j - b_j^T M^-1 D and
        M = F + H.  This is a matched filter that already accounts for the
        mesh's ability to mimic a point, and it costs one BLAS call per chunk.
        (r_j^2/s_j is the drop in the *penalised* objective chi^2 + s^T H s,
        not in chi^2 itself -- the regularisation term moves too.)

        Returns (amplitude, significance), both flat arrays over the lattice.
        """
        if positions:
            P0 = self.columns_for(positions)
            A_eff = np.hstack([self.A, P0])
            B0 = self._curv(self.A, P0)
            C0 = self._curv(P0, P0)
            M = np.block([
                [self.F + self.h_scale * self.H, B0],
                [B0.T, C0],
            ])
            D_eff = np.concatenate([self.D, self._dvec(P0)])
        else:
            A_eff = self.A
            M = self.F + self.h_scale * self.H
            D_eff = self.D
        M = M + np.eye(M.shape[0]) * 1e-12 * max(np.trace(M), 1e-30)
        cho = cho_factor(M, lower=True, check_finite=False)
        MinvD = cho_solve(cho, D_eff, check_finite=False)

        ys = np.asarray(ys, dtype=float)
        xs = np.asarray(xs, dtype=float)
        amp = np.empty(ys.size)
        sig = np.empty(ys.size)
        for lo in range(0, ys.size, chunk):
            hi = min(lo + chunk, ys.size)
            P = np.column_stack([
                point_visibilities(self.uv, y, x)
                for y, x in zip(ys[lo:hi], xs[lo:hi])
            ])
            B = self._curv(A_eff, P)
            c = (self.w_re @ P.real**2) + (self.w_im @ P.imag**2)
            r = self._dvec(P) - B.T @ MinvD
            MinvB = cho_solve(cho, B, check_finite=False)
            sch = c - np.einsum("ij,ij->j", B, MinvB)
            sch = np.maximum(sch, 1e-30)
            amp[lo:hi] = r / sch
            sig[lo:hi] = np.abs(r) / np.sqrt(sch)
        return amp, sig

    def chi_squared_with_width(self, positions, index, sigma_arcsec) -> float:
        """chi^2 with one component widened to a Gaussian of this sigma."""
        cols = [
            gaussian_visibilities(self.uv, y, x, sigma_arcsec) if i == index
            else point_visibilities(self.uv, y, x)
            for i, (y, x) in enumerate(positions)
        ]
        P = np.column_stack(cols)
        B = self._curv(self.A, P); C = self._curv(P, P); Dp = self._dvec(P)
        MinvB = cho_solve(self._cho, B, check_finite=False)
        MinvD = cho_solve(self._cho, self.D, check_finite=False)
        S = C - B.T @ MinvB
        S = S + np.eye(S.shape[0]) * 1e-10 * max(np.trace(S), 1e-30)
        a = np.linalg.solve(S, Dp - B.T @ MinvD)
        s = MinvD - MinvB @ a
        theta = np.concatenate([s, a])
        F_aug = np.block([[self.F, B], [B.T, C]])
        return float(
            self.chi2_const + theta @ F_aug @ theta
            - 2.0 * theta @ np.concatenate([self.D, Dp])
        )

    def solve(self, positions) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Solve for (mesh values, point amplitudes, chi^2, amplitude covariance)."""
        if not positions:
            s = cho_solve(self._cho, self.D, check_finite=False)
            chi2 = self.chi2_const + s @ self.F @ s - 2.0 * s @ self.D
            return s, np.array([]), float(chi2), np.zeros((0, 0))

        P = self.columns_for(positions)
        B = self._curv(self.A, P)
        C = self._curv(P, P)
        Dp = self._dvec(P)

        # Schur complement: eliminate the mesh block, which is already factorised
        MinvB = cho_solve(self._cho, B, check_finite=False)
        MinvD = cho_solve(self._cho, self.D, check_finite=False)
        S = C - B.T @ MinvB
        rhs = Dp - B.T @ MinvD
        # a tiny ridge keeps two coincident points from making S singular
        S_reg = S + np.eye(S.shape[0]) * 1e-10 * max(np.trace(S), 1e-30)
        a = np.linalg.solve(S_reg, rhs)
        s = MinvD - MinvB @ a
        theta = np.concatenate([s, a])
        F_aug = np.block([[self.F, B], [B.T, C]])
        D_aug = np.concatenate([self.D, Dp])
        chi2 = self.chi2_const + theta @ F_aug @ theta - 2.0 * theta @ D_aug
        # S^-1 is the amplitudes' covariance, mesh uncertainty marginalised out
        return s, a, float(chi2), np.linalg.inv(S_reg)

    def chi_squared(self, positions) -> float:
        return self.solve(positions)[2]

    def model_visibilities(self, mesh_values, positions, amplitudes) -> np.ndarray:
        vis = self.A @ mesh_values
        if len(positions):
            vis = vis + self.columns_for(positions) @ amplitudes
        return vis


def refine_position(
    system: AugmentedSystem,
    positions: list,
    index: int,
    step: float,
    n_steps: int = 2,
) -> tuple[float, float]:
    """Optimise one point's position, amplitudes profiled out analytically.

    A coarse scan on a sub-pixel lattice, then Nelder-Mead.  The scan matters:
    chi^2(position) oscillates on the scale of the synthesised beam, so a
    local method started at the wrong fringe converges to the wrong peak.
    """
    y0, x0 = positions[index]

    def chi2_at(y, x):
        trial = list(positions)
        trial[index] = (y, x)
        return system.chi_squared(trial)

    offsets = np.arange(-n_steps, n_steps + 1) * step
    best = (chi2_at(y0, x0), y0, x0)
    for dy in offsets:
        for dx in offsets:
            if dy == 0 and dx == 0:
                continue
            c = chi2_at(y0 + dy, x0 + dx)
            if c < best[0]:
                best = (c, y0 + dy, x0 + dx)
    _, ys, xs = best
    res = minimize(
        lambda p: chi2_at(p[0], p[1]), np.array([ys, xs]),
        method="Nelder-Mead",
        options={"xatol": step / 20.0, "fatol": 1e-3, "maxfev": 120},
    )
    return (float(res.x[0]), float(res.x[1])) if res.success else (ys, xs)


DEFAULT_RESOLVED_DELTA_CHI2 = 9.0

# Gaussian widths tried when asking "is this candidate actually a point?",
# as fractions of the synthesised beam's sigma.
_WIDTH_FRACTIONS = (0.15, 0.3, 0.5, 0.75, 1.0, 1.5)


def unresolved_test(
    system: AugmentedSystem,
    positions: list,
    index: int,
    beam_fwhm: float,
    delta_chi2: float = DEFAULT_RESOLVED_DELTA_CHI2,
) -> tuple[bool, float, float]:
    """Is component `index` consistent with being a true point source?

    A delta and a small Gaussian describe an unresolved source equally well,
    so letting the width float should buy nothing.  A *resolved* feature --
    for instance the central cusp of an exponential disc, which the smoothed
    mesh cannot render and which a delta would otherwise be recruited to
    absorb -- is fitted markedly better once the width is free.

    Returns (unresolved, best_sigma_arcsec, delta_chi2_gained).
    """
    chi2_point = system.chi_squared(positions)
    sigma_beam = beam_fwhm / 2.3548
    best, best_sigma = chi2_point, 0.0
    for frac in _WIDTH_FRACTIONS:
        s = frac * sigma_beam
        c = system.chi_squared_with_width(positions, index, s)
        if c < best:
            best, best_sigma = c, s
    gain = chi2_point - best
    return gain < delta_chi2, best_sigma, gain


def detection_lattice(geometry) -> tuple[np.ndarray, np.ndarray]:
    """Trial positions for the detector: the product grid, in grid (y, x)."""
    n = geometry.shape_native[0]
    pix = geometry.pixel_scale
    iy, ix = np.mgrid[0:n, 0:n]
    return (((n - 1) / 2 - iy) * pix).ravel(), ((ix - (n - 1) / 2) * pix).ravel()


def _best_candidate(system, accepted, ys, xs, excluded, radius):
    """Most significant *positive* trial position, avoiding earlier tries."""
    amp, sig = system.scan(accepted, ys, xs)
    score = np.where(amp > 0, sig, -np.inf)
    for (ey, ex) in excluded:
        if radius > 0:
            score[np.hypot(ys - ey, xs - ex) < radius] = -np.inf
    j = int(np.argmax(score))
    if not np.isfinite(score[j]):
        return None
    return (float(ys[j]), float(xs[j]))



def retune_regularization(
    system: AugmentedSystem, positions: list, chi2_target: float = 1.0,
    max_iter: int = 40, min_factor: float = 1e-8, max_factor: float = 1e12,
) -> float:
    """Re-hit chi^2 = target*N now that point components carry some of the flux.

    The regularisation strength was chosen for the mesh alone.  A point
    absorbs signal the mesh had been straining to reproduce, so the same
    strength leaves the combined fit *below* the target -- the extended model
    keeps freedom it no longer needs and spends it on noise.  chi^2 rises
    monotonically with the strength, so a bracket-and-bisect on the scale
    factor restores the criterion.  Returns the factor applied to the prior
    coefficient; 1.0 means the retune was abandoned and nothing changed.

    Every step goes through `set_regularization_scale`, which refuses a
    factorisation that is not positive definite: weakening the prior too far
    makes F singular whenever the mesh has pixels the uv coverage does not
    constrain.  The bracket stops there rather than crashing.
    """
    target = chi2_target * system.n_data

    def chi2_at(factor) -> float | None:
        if not system.set_regularization_scale(factor):
            return None
        try:
            return system.chi_squared(positions)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def give_up(reason: str) -> float:
        system.set_regularization_scale(1.0)
        logger.warning(
            "  could not restore chi^2 = %.2f N with point components "
            "present (%s); keeping the mesh-only regularisation",
            chi2_target, reason,
        )
        return 1.0

    start = chi2_at(1.0)
    if start is None:
        return give_up("the mesh-only system does not factorise")

    if start > target:
        # too stiff already: loosen, but only as far as F stays invertible
        lo, hi = 1.0, 1.0
        while lo > min_factor:
            trial = lo * 0.1
            c = chi2_at(trial)
            if c is None:
                return give_up(
                    f"the prior cannot be weakened below {lo:.1e}x without "
                    "the curvature matrix becoming singular"
                )
            lo = trial
            if c <= target:
                break
        else:
            return give_up("the floor on the scale factor was reached")
    else:
        lo, hi = 1.0, 1.0
        while hi < max_factor:
            hi *= 10.0
            c = chi2_at(hi)
            if c is None:
                return give_up(f"the system stops factorising above {hi:.1e}x")
            if c >= target:
                break
        else:
            return give_up("the ceiling on the scale factor was reached")

    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        c = chi2_at(mid)
        if c is None:
            break
        if c < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.01:
            break
    factor = float(np.sqrt(lo * hi))
    if not system.set_regularization_scale(factor):
        return give_up("the final factor does not factorise")
    return factor


def fit_point_sources(
    inversion,
    dataset,
    geometry,
    positions=None,
    significance: float = DEFAULT_SIGNIFICANCE,
    max_points: int = DEFAULT_MAX_POINTS,
    refine: bool = True,
    dirty_imager=None,   # unused; kept so older call sites keep working
    beam_fwhm: float | None = None,
    resolved_delta_chi2: float = DEFAULT_RESOLVED_DELTA_CHI2,
    retune: bool = True,
    chi2_target: float = 1.0,
):
    """Fit analytic point components alongside the pixelized model.

    Parameters
    ----------
    positions
        Optional list of (dRA, dDec) offsets in arcsec.  Supplied positions are
        kept (and refined, if ``refine``) regardless of significance -- the
        user asked for them.  If ``None``, candidates are detected from the
        residual map and each is kept only if it clears ``significance``
        *and* passes the unresolved test.
    beam_fwhm
        Synthesised beam FWHM [arcsec].  Sets the minimum separation between
        detections and the width scale of the unresolved test.  Auto-detection
        without it is far more trigger-happy, so it is strongly recommended.
    """
    system = AugmentedSystem(inversion, dataset)
    pixel = geometry.pixel_scale
    accepted: list = []
    user_flags: list = []

    if positions:
        grid_positions = [sky_to_grid(*p) for p in positions]
        for i in range(len(grid_positions)):
            if refine:
                grid_positions[i] = refine_position(
                    system, grid_positions, i, step=pixel / 2.0
                )
        accepted = grid_positions
        user_flags = [True] * len(accepted)
    else:
        # Detect: take the strongest residual peak, fit it, and keep it only if
        # it is both significant and genuinely unresolved.  Two guards matter
        # here.  Without a minimum separation, several deltas stack inside one
        # beam and split a single feature between them; without the unresolved
        # test, deltas are recruited to absorb the central cusp of a smooth
        # source that the mesh cannot render.  Both were seen in practice.
        beam = float(beam_fwhm) if beam_fwhm else 0.0
        min_separation = 0.75 * beam
        excluded: list = []
        lat_y, lat_x = detection_lattice(geometry)
        for _ in range(max_points + 3):
            if len(accepted) >= max_points:
                break
            peak = _best_candidate(
                system, accepted, lat_y, lat_x, excluded, min_separation
            )
            if peak is None:
                break
            trial = accepted + [peak]
            if refine:
                trial[-1] = refine_position(system, trial, len(trial) - 1,
                                            step=pixel / 2.0)
            # exclude both the lattice pick and where refinement took it:
            # otherwise a rejected candidate is found again from the next
            # lattice point just outside the exclusion radius, and refinement
            # walks straight back to the same minimum
            excluded.extend([peak, trial[-1]])
            sky = grid_to_sky(*trial[-1])
            _, amps, _, cov = system.solve(trial)
            sig = abs(amps[-1]) / np.sqrt(max(cov[-1, -1], 1e-300))
            if amps[-1] <= 0:
                # a negative "source" is the fit patching a trough, not sky
                logger.info(
                    "  candidate at dRA %.3f\", dDec %.3f\" rejected: "
                    "negative amplitude (%.3g Jy)", *sky, amps[-1],
                )
                continue
            if sig < significance:
                logger.info(
                    "  candidate at dRA %.3f\", dDec %.3f\" rejected: "
                    "%.1f sigma < %.1f", *sky, sig, significance,
                )
                break
            gain = 0.0
            if beam > 0:
                ok, best_sigma, gain = unresolved_test(
                    system, trial, len(trial) - 1, beam, resolved_delta_chi2
                )
                if not ok:
                    logger.info(
                        "  candidate at dRA %.3f\", dDec %.3f\" rejected: "
                        "resolved (a %.3f\" sigma Gaussian fits better by "
                        "delta chi2 = %.1f)", *sky, best_sigma, gain,
                    )
                    continue
            accepted = trial
            user_flags.append(False)
            logger.info(
                "  point source accepted at dRA %.3f\", dDec %.3f\": "
                "%.4g Jy (%.1f sigma, unresolved: widening gains only "
                "delta chi2 = %.1f)", *sky, amps[-1], sig, gain,
            )

    # With points present, three things have to settle together: the
    # regularisation strength (the points took flux the mesh was straining to
    # reproduce, so chi^2 falls below target), the positions (refined under
    # the pre-retune prior, and a stiffer mesh moves where the best position
    # is), and which candidates still clear the significance cut (the retune
    # changes the errors).  Each depends on the others, so iterate.
    #
    # Skipping the re-polish was a real bug: at high S/N it left amplitudes
    # read off stale positions, 20% flux errors with 30 mas offsets.
    factor = 1.0
    if accepted:
        chi2_before = system.chi_squared(accepted) / system.n_data
        for _ in range(3):
            if retune:
                factor = retune_regularization(system, accepted, chi2_target)
            if not refine or factor == 1.0:
                break
            moved = 0.0
            for i in range(len(accepted)):
                before = accepted[i]
                accepted[i] = refine_position(
                    system, accepted, i, step=pixel / 2.0
                )
                moved = max(moved, float(np.hypot(
                    accepted[i][0] - before[0], accepted[i][1] - before[1])))
            if moved < 0.01 * pixel:
                break
        # the loop can exit on a position move, which leaves chi^2 below the
        # target again -- so always finish on a retune, never on a refine
        if retune:
            factor = retune_regularization(system, accepted, chi2_target)
        if retune:
            logger.info(
                "  regularisation rescaled by %.3g with point components "
                "present (chi2/N %.3f -> %.3f)", factor, chi2_before,
                system.chi_squared(accepted) / system.n_data,
            )
        elif chi2_before < 0.9 * chi2_target:
            logger.info(
                "  chi2/N is %.3f with the point components and no retune: "
                "the mesh keeps freedom it no longer needs. Drop "
                "point_retune=False to re-impose chi^2 = %.2f N.",
                chi2_before, chi2_target,
            )

    mesh, amps, chi2, cov = system.solve(accepted)

    # Candidates were accepted on their significance under the pre-retune
    # prior.  The retune changes the errors, so apply the cut again and drop
    # anything that no longer clears it -- at high S/N this is what removes
    # the last spurious detection, which fell from 20 sigma to 3.
    if accepted and not positions:
        while len(accepted) > 0:
            errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            sigs = np.where(
                errs > 0, np.abs(amps) / np.maximum(errs, 1e-300), np.inf)
            # a sign flip is as disqualifying as a low significance: the
            # amplitude is re-solved at the retuned prior, and a component
            # that has gone negative is patching a trough, not a source
            sigs = np.where(np.asarray(amps) > 0, sigs, -1.0)
            weakest = int(np.argmin(sigs))
            if sigs[weakest] >= significance:
                break
            logger.info(
                "  point at dRA %.3f\", dDec %.3f\" dropped after the "
                "regularisation retune: %s",
                *grid_to_sky(*accepted[weakest]),
                f"amplitude went negative ({amps[weakest]:.3g} Jy)"
                if amps[weakest] <= 0 else
                f"{sigs[weakest]:.1f} sigma < {significance:.1f}",
            )
            del accepted[weakest]
            if weakest < len(user_flags):
                del user_flags[weakest]
            if accepted and retune:
                factor = retune_regularization(system, accepted, chi2_target)
            mesh, amps, chi2, cov = system.solve(accepted)

    # Systematic on the amplitude: how far it moves when the regularisation
    # strength is varied over the range that is defensible for these data.
    # The split of flux between a point and the mesh underneath it is broken
    # only by the prior, so the amplitude inherits the prior's arbitrariness.
    # Without this the quoted error is purely statistical and conditional on
    # one strength, and it is badly optimistic -- across the generalisation
    # mocks it gave pulls of up to 24 sigma on fluxes only 20% off.
    sys_err = np.zeros(len(accepted))
    if accepted:
        held = system.h_scale
        alternatives = [held * 3.0, held / 3.0]
        if factor != 1.0:
            alternatives.append(held / factor)   # the un-retuned strength
        deviations = []
        for alt_scale in alternatives:
            if not system.set_regularization_scale(alt_scale):
                continue
            try:
                alt = system.solve(accepted)[1]
            except (np.linalg.LinAlgError, ValueError):
                continue
            if len(alt) == len(amps):
                deviations.append(np.abs(np.asarray(alt) - np.asarray(amps)))
        system.set_regularization_scale(held)
        if deviations:
            sys_err = np.max(np.vstack(deviations), axis=0)

    points = [
        PointSource(
            *grid_to_sky(y, x), flux=float(a),
            flux_error=float(np.hypot(np.sqrt(max(cov[i, i], 0.0)), sys_err[i])),
            user_supplied=user_flags[i] if i < len(user_flags) else False,
            flux_error_stat=float(np.sqrt(max(cov[i, i], 0.0))),
            flux_error_sys=float(sys_err[i]),
        )
        for i, ((y, x), a) in enumerate(zip(accepted, amps))
    ]
    return PointSolution(
        system=system, mesh_values=mesh, points=points, chi_squared=chi2,
        grid_positions=accepted, amplitudes=amps, amplitude_covariance=cov,
        regularization_factor=factor,
    )


@dataclass
class PointSolution:
    """The combined pixelized + point-source solution."""

    system: AugmentedSystem
    mesh_values: np.ndarray
    points: list
    chi_squared: float
    grid_positions: list
    amplitudes: np.ndarray
    amplitude_covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    regularization_factor: float = 1.0

    @property
    def model_visibilities(self) -> np.ndarray:
        return self.system.model_visibilities(
            self.mesh_values, self.grid_positions, self.amplitudes
        )

    @property
    def total_point_flux(self) -> float:
        return float(np.sum([p.flux for p in self.points]))

    def as_dict(self) -> dict:
        return {
            "n_points": len(self.points),
            "total_point_flux_jy": self.total_point_flux,
            "chi_squared": float(self.chi_squared),
            "regularization_rescaled_by": float(self.regularization_factor),
            "points": [p.as_dict() for p in self.points],
        }


def restore_points(
    image_shape, pixel_scale, points, beam, existing=None
) -> np.ndarray:
    """Add point sources to a restored image as beam-shaped Gaussians.

    Evaluated analytically at the fitted sub-pixel position -- the restoring
    beam is smooth, so unlike the model grid this placement is exact.
    """
    from .beam import SIGMA_TO_FWHM

    ny, nx = image_shape
    out = np.zeros(image_shape) if existing is None else np.array(existing)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    smaj = beam.bmaj_arcsec / SIGMA_TO_FWHM
    smin = beam.bmin_arcsec / SIGMA_TO_FWHM
    theta = np.radians(beam.bpa_deg)
    ct, st = np.cos(theta), np.sin(theta)
    for p in points:
        dy = (cy - yy) * pixel_scale - p.d_dec
        dx = (xx - cx) * pixel_scale - (-p.d_ra)
        xr = dx * ct + dy * st
        yr = -dx * st + dy * ct
        out += p.flux * np.exp(-0.5 * ((xr / smin) ** 2 + (yr / smaj) ** 2))
    return out


class PointAugmentedFit:
    """Adapter presenting a point-augmented solution like a plain `SingleFit`.

    Delegates everything to the underlying mesh fit except the quantities the
    point components change: the mesh values, the model visibilities, chi^2,
    and the mesh uncertainty (which must now be marginalised over the point
    amplitudes).
    """

    def __init__(self, single_fit, solution: PointSolution):
        self._sf = single_fit
        self.solution = solution

    def __getattr__(self, item):          # prior, geometry, coefficient, ...
        return getattr(self._sf, item)

    @property
    def points(self) -> list:
        return self.solution.points

    @property
    def chi_squared(self) -> float:
        return self.solution.chi_squared

    @property
    def model_visibilities(self) -> np.ndarray:
        return self.solution.model_visibilities

    @property
    def model_mesh_image(self) -> np.ndarray:
        geom = self._sf.geometry
        k2 = (geom.shape_native[0] // geom.mesh_shape[0]) ** 2
        return self.solution.mesh_values.reshape(geom.mesh_shape) * k2

    @property
    def model_image(self) -> np.ndarray:
        """The *extended* model on the product grid; points are separate."""
        import autogalaxy as ag

        for obj, _ in self._sf.fit.inversion.reconstruction_dict.items():
            if type(obj).__name__ == "Mapper":
                slim = np.asarray(obj.mapping_matrix) @ self.solution.mesh_values
                return np.asarray(
                    ag.Array2D(
                        values=slim, mask=self._sf.fit.dataset.real_space_mask
                    ).native
                )
        raise RuntimeError("no mesh in the inversion")

    @property
    def sampling_covariance(self) -> np.ndarray:
        """Estimator covariance under the (possibly rescaled) regularisation."""
        sysm = self.solution.system
        if not self.solution.grid_positions:
            return self._sf.sampling_covariance
        M_inv = cho_solve(sysm._cho, np.eye(sysm.n_mesh), check_finite=False)
        return M_inv @ sysm.F @ M_inv

    @property
    def posterior_covariance(self) -> np.ndarray:
        """Mesh covariance marginalised over the point amplitudes.

        Cov = M^-1 + (M^-1 B) S^-1 (M^-1 B)^T -- ignoring the second term would
        understate the mesh error wherever a point competes with it for flux.
        """
        sysm = self.solution.system
        if not self.solution.grid_positions:
            return self._sf.posterior_covariance
        # rebuilt from the system, not from the mesh-only fit: the point solve
        # may have rescaled the regularisation, so (F+H)^-1 has changed
        M_inv = cho_solve(
            sysm._cho, np.eye(sysm.n_mesh), check_finite=False
        )
        P = sysm.columns_for(self.solution.grid_positions)
        B = sysm._curv(sysm.A, P)
        MinvB = cho_solve(sysm._cho, B, check_finite=False)
        return M_inv + MinvB @ self.solution.amplitude_covariance @ MinvB.T

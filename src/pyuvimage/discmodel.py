"""Surface-brightness perturbations on a uniform disc, fitted in the uv-plane.

The target is a resolved stellar disc (an AGB photosphere, say): to first
order a **uniform disc**, whose visibilities are the closed-form Airy pattern

    V(u, v) = F * 2 J1(2 pi R q) / (2 pi R q) * exp(-2 pi i (x0 u + y0 v))

with R the angular radius in radians and q = sqrt(u^2 + v^2).  What is
actually of interest is the *departure* from uniformity -- hot spots,
convective cells, limb asymmetries -- so the model is

    V_model = V_disc(F, R, x0, y0)  +  A s

where `A` maps a **polar grid of surface-brightness perturbation cells**
confined to the disc onto visibilities, and `s` (one amplitude per cell, in
Jy/arcsec^2) is solved *linearly* under a Matern GP prior, exactly as the
mesh amplitudes are in `fitting.LinearSystem`.  Only the disc parameters
(radius, centre; the flux is itself linear and profiled out analytically)
are non-linear, and they are fitted with Nelder-Mead, as `pointsource`
refines its positions.

Why a polar grid rather than the existing Cartesian mesh: the perturbations
live *on the disc*, so the natural coordinates are (r, theta); a polar grid
puts no freedom outside the stellar limb (where a Cartesian mesh would spend
most of its pixels absorbing sidelobe structure), and its cells line up with
the radial/azimuthal decompositions (limb darkening = radial trend,
hot spots = azimuthal structure) one wants to read off the result.

Degeneracies, stated up front: a constant perturbation is exactly degenerate
with the disc flux, and the lowest azimuthal modes lean on the disc centre.
They are broken (softly) by the GP prior, and (procedurally) by fitting the
disc alone first and only then letting the perturbations mop up what a
uniform disc cannot represent.  The perturbation map should therefore be
read as "structure beyond the best uniform disc", which is what it is.

Conventions are the package's own throughout: grid (y, x) = (dDec, -dRA) in
arcsec, phase = -2 pi (x u + y v) (see `pointsource.point_visibilities`,
verified against the transformer to machine precision).  A polar grid filled
with the disc's uniform surface brightness reproduces the analytic Airy
visibilities -- that round trip is a unit test, not a hope.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import j1

from .envelope import cached_inverse_covariance
from .pointsource import ARCSEC_TO_RAD, grid_to_sky, image_to_sky, sky_to_grid

# First zero of J1: the Airy disc's first visibility null is at
# 2 pi R q = j1_1, so R = j1_1 / (2 pi q).
_J1_FIRST_ZERO = 3.8317059702075125

# Stationary priors that need only the polar cell centres (no brightness map).
DISC_REGULARIZATIONS = ("matern", "exponential", "gaussian", "constant")

logger = logging.getLogger("pyuvimage")


# ---------------------------------------------------------------------------
# The uniform disc
# ---------------------------------------------------------------------------

@dataclass
class UniformDisc:
    """A uniform circular disc, in the user-facing sky frame."""

    flux_jy: float           # total flux [Jy]
    radius_arcsec: float     # angular radius [arcsec]
    d_ra: float = 0.0        # centre offset from phase centre [arcsec], +E
    d_dec: float = 0.0       # centre offset [arcsec], +N

    @property
    def surface_brightness(self) -> float:
        """Uniform surface brightness [Jy / arcsec^2]."""
        return self.flux_jy / (np.pi * self.radius_arcsec**2)

    @property
    def centre_grid(self) -> tuple[float, float]:
        """Centre in grid (y, x) coordinates."""
        return sky_to_grid(self.d_ra, self.d_dec)

    def visibilities(self, uv_wavelengths: np.ndarray) -> np.ndarray:
        y, x = self.centre_grid
        return self.flux_jy * disc_shape_visibilities(
            uv_wavelengths, self.radius_arcsec, y, x
        )

    def as_dict(self) -> dict:
        return {
            "flux_jy": float(self.flux_jy),
            "radius_arcsec": float(self.radius_arcsec),
            "d_ra_arcsec": float(self.d_ra),
            "d_dec_arcsec": float(self.d_dec),
            "surface_brightness_jy_arcsec2": float(self.surface_brightness),
        }


def disc_shape_visibilities(
    uv_wavelengths: np.ndarray, radius_arcsec: float, y: float, x: float
) -> np.ndarray:
    """Unit-total-flux uniform disc at grid position (y, x): the Airy pattern.

    The flux multiplies this linearly, so the disc fit can profile it out.
    """
    uv = np.asarray(uv_wavelengths)
    q = np.hypot(uv[:, 0], uv[:, 1])
    arg = 2.0 * np.pi * radius_arcsec * ARCSEC_TO_RAD * q
    # 2 J1(z)/z -> 1 as z -> 0; the series is used below z = 1e-8 where the
    # ratio is 1 to double precision anyway
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = np.where(arg > 1e-8, 2.0 * j1(arg) / np.where(arg > 0, arg, 1.0), 1.0)
    phase = -2.0 * np.pi * (
        x * ARCSEC_TO_RAD * uv[:, 0] + y * ARCSEC_TO_RAD * uv[:, 1]
    )
    return shape * (np.cos(phase) + 1j * np.sin(phase))


def guess_radius_arcsec(uv_wavelengths: np.ndarray, data: np.ndarray) -> float:
    """Angular radius from the first null of the azimuthally-averaged |V|.

    A uniform disc has its first zero at ``2 pi R q = 3.8317``.  If the
    amplitude has no clear minimum (an unresolved source, or a noisy one)
    the guess falls back to a resolution-element scale so Nelder-Mead still
    has somewhere to start.
    """
    uv = np.asarray(uv_wavelengths, dtype=float)
    q = np.hypot(uv[:, 0], uv[:, 1])
    amp = np.abs(np.asarray(data))
    ok = q > 0
    q, amp = q[ok], amp[ok]
    if q.size < 16:
        raise ValueError("not enough visibilities to guess a disc radius")
    n_bins = int(np.clip(q.size // 40, 12, 40))
    edges = np.linspace(float(q.min()), float(q.max()), n_bins + 1)
    idx = np.clip(np.digitize(q, edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = idx == i
        if np.any(sel):
            means[i] = float(np.mean(amp[sel]))
    centres = 0.5 * (edges[:-1] + edges[1:])
    valid = np.isfinite(means)
    means, centres = means[valid], centres[valid]
    for i in range(1, len(means) - 1):
        if means[i] <= means[i - 1] and means[i] <= means[i + 1]:
            return float(
                _J1_FIRST_ZERO / (2.0 * np.pi * centres[i]) / ARCSEC_TO_RAD
            )
    return float(0.6 / (2.0 * q.max()) / ARCSEC_TO_RAD)


# ---------------------------------------------------------------------------
# The polar grid
# ---------------------------------------------------------------------------

@dataclass
class PolarGrid:
    """Equal-ish-area polar cells tiling a disc of `radius` at grid `centre`.

    Rings are uniform in radius; ring k holds ``round(az_oversample * 2 pi
    (k + 1/2))`` azimuthal cells, so cells are roughly square (dr x r dtheta
    ~ dr x dr) everywhere.  Together the cells tile the disc exactly: summing
    all columns at the disc's uniform surface brightness reproduces the
    analytic Airy visibilities (tested).

    theta is measured from North (+y) through East (-x on the image, +dRA on
    the sky), i.e. the astronomical position angle.
    """

    radius: float                       # arcsec
    n_rings: int
    centre: tuple[float, float] = (0.0, 0.0)   # grid (y, x), arcsec
    az_oversample: float = 1.0

    # filled in __post_init__
    r_inner: np.ndarray = field(init=False)     # (n_cells,)
    r_outer: np.ndarray = field(init=False)
    theta_lo: np.ndarray = field(init=False)    # (n_cells,) radians
    theta_hi: np.ndarray = field(init=False)
    ring_index: np.ndarray = field(init=False)  # (n_cells,) int
    cells_per_ring: list = field(init=False)

    def __post_init__(self):
        if self.n_rings < 1:
            raise ValueError("n_rings must be >= 1")
        dr = self.radius / self.n_rings
        r_in, r_out, t_lo, t_hi, ring = [], [], [], [], []
        self.cells_per_ring = []
        for k in range(self.n_rings):
            n_az = max(1, int(round(self.az_oversample * 2.0 * np.pi * (k + 0.5))))
            self.cells_per_ring.append(n_az)
            edges = np.linspace(0.0, 2.0 * np.pi, n_az + 1)
            r_in.extend([k * dr] * n_az)
            r_out.extend([(k + 1) * dr] * n_az)
            t_lo.extend(edges[:-1])
            t_hi.extend(edges[1:])
            ring.extend([k] * n_az)
        self.r_inner = np.asarray(r_in)
        self.r_outer = np.asarray(r_out)
        self.theta_lo = np.asarray(t_lo)
        self.theta_hi = np.asarray(t_hi)
        self.ring_index = np.asarray(ring, dtype=int)

    # ------------------------------------------------------------- geometry
    @property
    def n_cells(self) -> int:
        return self.r_inner.size

    @property
    def areas(self) -> np.ndarray:
        """Cell areas [arcsec^2]."""
        return 0.5 * (self.r_outer**2 - self.r_inner**2) * (
            self.theta_hi - self.theta_lo
        )

    @property
    def r_mid(self) -> np.ndarray:
        """Area-weighted radial centroid of each cell [arcsec]."""
        # centroid radius of an annular sector: (2/3)(r2^3 - r1^3)/(r2^2 - r1^2)
        return (2.0 / 3.0) * (self.r_outer**3 - self.r_inner**3) / (
            self.r_outer**2 - self.r_inner**2
        )

    @property
    def theta_mid(self) -> np.ndarray:
        return 0.5 * (self.theta_lo + self.theta_hi)

    def _polar_to_grid(self, r, theta) -> tuple[np.ndarray, np.ndarray]:
        """(r, theta from North towards East) -> grid (y, x) [arcsec]."""
        y0, x0 = self.centre
        # position angle: 0 = North (+y), 90 deg = East = +dRA = -x (grid)
        return y0 + r * np.cos(theta), x0 - r * np.sin(theta)

    @property
    def cell_points(self) -> np.ndarray:
        """(n_cells, 2) cell centroids in grid (y, x), for the GP kernel."""
        y, x = self._polar_to_grid(self.r_mid, self.theta_mid)
        return np.column_stack([y, x])

    # -------------------------------------------------------------- columns
    def visibility_columns(
        self,
        uv_wavelengths: np.ndarray,
        max_phase_step: float = 0.4,
        max_sub: int = 12,
    ) -> np.ndarray:
        """Complex (n_vis, n_cells) columns: unit surface brightness on each cell.

        Each cell is integrated by midpoint subsampling in (r, theta); a cell
        of unit surface brightness [Jy/arcsec^2] contributes ``sum_p w_p
        exp(-2 pi i (x_p u + y_p v))`` with the subcell areas as weights, so
        a column carries units of arcsec^2 and the solved amplitudes are
        Jy/arcsec^2.

        The subdivision is sized from the data: the phase across a length L
        at the longest baseline q_max is ``2 pi q_max L``, and the midpoint
        rule is accurate once the per-subcell phase span is below
        ``max_phase_step`` radians (0.4 rad keeps the quadrature error ~1e-3
        of the column, well under thermal noise; the round-trip test against
        the analytic Airy pattern is the check).  Capped at `max_sub`^2
        points per cell to bound the cost.
        """
        uv = np.asarray(uv_wavelengths)
        q_max = float(np.max(np.hypot(uv[:, 0], uv[:, 1])))
        dr = self.radius / self.n_rings
        n_vis = uv.shape[0]
        cols = np.empty((n_vis, self.n_cells), dtype=complex)
        ux = uv[:, 0] * ARCSEC_TO_RAD
        vy = uv[:, 1] * ARCSEC_TO_RAD

        def n_needed(length_arcsec: float) -> int:
            span = 2.0 * np.pi * q_max * length_arcsec * ARCSEC_TO_RAD
            return int(np.clip(np.ceil(span / max_phase_step), 1, max_sub))

        for i in range(self.n_cells):
            r1, r2 = self.r_inner[i], self.r_outer[i]
            t1, t2 = self.theta_lo[i], self.theta_hi[i]
            n_r = n_needed(r2 - r1)
            n_t = n_needed(r2 * (t2 - t1))
            r_edges = np.linspace(r1, r2, n_r + 1)
            t_edges = np.linspace(t1, t2, n_t + 1)
            # midpoints and exact subcell areas: 0.5 (r_hi^2 - r_lo^2) dtheta
            r_m = 0.5 * (r_edges[:-1] + r_edges[1:])
            w_r = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
            t_m = 0.5 * (t_edges[:-1] + t_edges[1:])
            w_t = np.diff(t_edges)
            rr, tt = np.meshgrid(r_m, t_m, indexing="ij")
            ww = np.outer(w_r, w_t).ravel()
            yy, xx = self._polar_to_grid(rr.ravel(), tt.ravel())
            phase = -2.0 * np.pi * (np.outer(ux, xx) + np.outer(vy, yy))
            cols[:, i] = (np.cos(phase) + 1j * np.sin(phase)) @ ww
        return cols

    # ------------------------------------------------------------ rendering
    def render(
        self,
        values: np.ndarray,
        shape: tuple[int, int],
        pixel_scale: float,
        outside=np.nan,
    ) -> np.ndarray:
        """Paint per-cell `values` onto a native (row 0 = +y) Cartesian image."""
        ny, nx = shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        y = (cy - yy) * pixel_scale - self.centre[0]
        x = (xx - cx) * pixel_scale - self.centre[1]
        r = np.hypot(y, x)
        theta = np.arctan2(-x, y) % (2.0 * np.pi)   # PA from North towards East
        dr = self.radius / self.n_rings
        img = np.full(shape, outside, dtype=float)
        inside = r < self.radius
        ring = np.minimum((r[inside] / dr).astype(int), self.n_rings - 1)
        offsets = np.concatenate([[0], np.cumsum(self.cells_per_ring)])
        n_az = np.asarray(self.cells_per_ring)[ring]
        az = np.minimum(
            (theta[inside] / (2.0 * np.pi) * n_az).astype(int), n_az - 1
        )
        img[inside] = np.asarray(values)[offsets[ring] + az]
        return img


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

@dataclass
class DiscFitResult:
    """The fitted disc plus its surface-brightness perturbation map."""

    disc: UniformDisc
    grid: PolarGrid | None
    perturbations: np.ndarray            # (n_cells,) Jy/arcsec^2
    perturbation_errors: np.ndarray      # 1 sigma, Jy/arcsec^2
    chi_squared: float
    n_data: int                          # real + imaginary counted separately
    coefficient: float                   # delivered GP prior strength
    disc_chi_squared: float              # chi^2 of the uniform disc alone
    regularization: str = "matern"
    scale: float | None = None
    nu: float | None = None
    model_visibilities: np.ndarray = field(repr=False, default=None)
    uv_wavelengths: np.ndarray | None = field(repr=False, default=None)
    data: np.ndarray | None = field(repr=False, default=None)
    noise: np.ndarray | None = field(repr=False, default=None)
    meta: dict = field(default_factory=dict)

    @property
    def chi_squared_reduced(self) -> float:
        return self.chi_squared / self.n_data

    @property
    def fractional(self) -> np.ndarray:
        """Perturbations as a fraction of the disc's surface brightness."""
        return self.perturbations / self.disc.surface_brightness

    @property
    def significance(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.perturbation_errors > 0,
                self.perturbations / self.perturbation_errors, 0.0,
            )

    def perturbation_image(
        self, shape: tuple[int, int], pixel_scale: float, fractional: bool = True
    ) -> np.ndarray:
        vals = self.fractional if fractional else self.perturbations
        return self.grid.render(vals, shape, pixel_scale)

    def model_image(self, shape: tuple[int, int], pixel_scale: float) -> np.ndarray:
        """Disc + perturbations, in Jy/arcsec^2, NaN-free."""
        img = self.grid.render(
            self.perturbations, shape, pixel_scale, outside=0.0
        )
        base = self.grid.render(
            np.full(self.grid.n_cells, self.disc.surface_brightness),
            shape, pixel_scale, outside=0.0,
        )
        return base + img

    def as_dict(self) -> dict:
        return {
            "disc": self.disc.as_dict(),
            "n_cells": 0 if self.grid is None else int(self.grid.n_cells),
            "chi_squared": float(self.chi_squared),
            "chi_squared_reduced": float(self.chi_squared_reduced),
            "disc_only_chi_squared_reduced": float(
                self.disc_chi_squared / self.n_data),
            "regularization": self.regularization,
            "coefficient": float(self.coefficient),
            "scale": None if self.scale is None else float(self.scale),
            "nu": None if self.nu is None else float(self.nu),
            "max_fractional_perturbation": float(
                np.max(np.abs(self.fractional))) if self.grid else 0.0,
        }


class DiscPerturbationFit:
    """Fit a uniform disc plus polar-grid perturbations to visibilities.

    Parameters
    ----------
    uv_wavelengths, data, noise
        Flattened samples, as `UVData.flattened` returns them.
    disc
        Initial disc.  ``radius_arcsec`` matters most; a decent guess is the
        first null of the azimuthally averaged amplitude (``q_null ~ 0.61 /
        R_rad``).  Flux and centre are refined from any starting point.

    Usage::

        fit = DiscPerturbationFit(uv, data, noise, disc=UniformDisc(1.0, 0.03))
        result = fit.fit(n_rings=6)
    """

    def __init__(self, uv_wavelengths, data, noise, disc: UniformDisc):
        self.uv = np.asarray(uv_wavelengths, dtype=float)
        data = np.asarray(data)
        noise = np.asarray(noise)
        self.d_stack = np.concatenate([data.real, data.imag])
        self.w_stack = np.concatenate([1.0 / noise.real**2, 1.0 / noise.imag**2])
        self.wd_stack = self.w_stack * self.d_stack
        self.n_vis = self.uv.shape[0]
        self.n_data = 2 * self.n_vis
        self.chi2_const = float(np.sum(self.w_stack * self.d_stack**2))
        self.disc0 = disc
        # filled by _build_grid
        self.grid: PolarGrid | None = None
        self.A_stack: np.ndarray | None = None      # (2 n_vis, n_cells)
        self.F: np.ndarray | None = None
        self.AtWd: np.ndarray | None = None
        self.Cinv: np.ndarray | None = None

    # ------------------------------------------------------------- helpers
    def _stack(self, vis: np.ndarray) -> np.ndarray:
        return np.concatenate([vis.real, vis.imag])

    def _disc_chi2_profiled(
        self, radius: float, y: float, x: float
    ) -> tuple[float, float]:
        """(chi^2, best flux) with the flux minimised analytically."""
        g = self._stack(disc_shape_visibilities(self.uv, radius, y, x))
        gw = self.w_stack * g
        denom = float(g @ gw)
        flux = float(self.d_stack @ gw) / denom if denom > 0 else 0.0
        chi2 = self.chi2_const - flux**2 * denom
        return chi2, flux

    def fit_disc(self, disc: UniformDisc | None = None) -> UniformDisc:
        """Nelder-Mead over (radius, centre), flux profiled out analytically."""
        d0 = disc or self.disc0
        y0, x0 = d0.centre_grid

        def objective(p):
            radius, y, x = p
            if radius <= 0:
                return np.inf
            return self._disc_chi2_profiled(radius, y, x)[0]

        res = minimize(
            objective, np.array([d0.radius_arcsec, y0, x0]),
            method="Nelder-Mead",
            options={"xatol": 1e-6 * d0.radius_arcsec, "fatol": 1e-3,
                     "maxfev": 2000},
        )
        radius, y, x = res.x
        _, flux = self._disc_chi2_profiled(radius, y, x)
        d_ra, d_dec = grid_to_sky(y, x)
        fitted = UniformDisc(flux, float(radius), d_ra, d_dec)
        logger.info(
            "  uniform disc: F = %.4g Jy, R = %.4g\", centre (dRA, dDec) = "
            "(%.4g\", %.4g\"), chi2/N = %.3f",
            flux, radius, d_ra, d_dec, res.fun / self.n_data,
        )
        return fitted

    # --------------------------------------------------------- linear system
    def _build_grid(
        self,
        disc: UniformDisc,
        n_rings: int,
        az_oversample: float,
        scale: float | None,
        nu: float,
        reg: str = "matern",
        envelope_fwhm: float | None = None,
        envelope_floor: float = 1e-2,
    ) -> None:
        reg = reg.lower()
        if reg not in DISC_REGULARIZATIONS:
            raise ValueError(
                f"unknown regularization {reg!r}; options: {DISC_REGULARIZATIONS}"
            )
        self.grid = PolarGrid(
            radius=disc.radius_arcsec, n_rings=n_rings,
            centre=disc.centre_grid, az_oversample=az_oversample,
        )
        n_bytes = 16.0 * self.n_vis * self.grid.n_cells
        if n_bytes > 1e9:
            logger.warning(
                "  polar mapping matrix is %.1f GB (%d vis x %d cells); "
                "consider fewer rings or averaging the data",
                n_bytes / 1e9, self.n_vis, self.grid.n_cells,
            )
        cols = self.grid.visibility_columns(self.uv)
        self.A_stack = np.concatenate([cols.real, cols.imag])
        del cols
        Aw = self.w_stack[:, None] * self.A_stack
        self.F = self.A_stack.T @ Aw
        self.AtWd = None  # depends on the disc; set in _solve
        if scale is None:
            # default GP correlation length: one ring width
            scale = disc.radius_arcsec / n_rings
        self._scale = float(scale)
        self._reg = reg
        if reg == "exponential":
            nu = 0.5
        self._nu = float(nu)

        if reg == "constant":
            # Tikhonov: H = lam I.  No spatial correlation.
            self.Cinv = np.eye(self.grid.n_cells)
        else:
            weights = None
            weights_key: tuple = ()
            jitter_relative = False
            if reg == "gaussian":
                from .envelope import SIGMA_TO_FWHM

                # Envelope FWHM defaults to the disc diameter so the prior
                # is permissive across the star and tight outside — though
                # the polar grid already has no cells outside the limb.
                fwhm = (
                    float(envelope_fwhm)
                    if envelope_fwhm is not None
                    else 2.0 * disc.radius_arcsec
                )
                pts = self.grid.cell_points
                y0, x0 = disc.centre_grid
                sigma = fwhm / SIGMA_TO_FWHM
                g = np.exp(
                    -0.5 * ((pts[:, 0] - y0) ** 2 + (pts[:, 1] - x0) ** 2)
                    / sigma**2
                )
                weights = envelope_floor + (1.0 - envelope_floor) * g
                weights_key = ("gauss", fwhm, envelope_floor, y0, x0)
                jitter_relative = True
            self.Cinv = cached_inverse_covariance(
                pixel_points=self.grid.cell_points,
                scale=self._scale,
                nu=self._nu,
                weights=weights,
                jitter_relative=jitter_relative,
                weights_key=weights_key,
            )

    def _solve(
        self, disc: UniformDisc, coefficient: float
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Solve (F + lam C^-1) s = A^T W (d - V_disc); return (s, chi2, cho)."""
        resid = self.d_stack - self._stack(disc.visibilities(self.uv))
        AtWr = self.A_stack.T @ (self.w_stack * resid)
        M = self.F + coefficient * self.Cinv
        cho = cho_factor(M, lower=True, check_finite=False)
        s = cho_solve(cho, AtWr, check_finite=False)
        chi2 = float(
            np.sum(self.w_stack * resid**2) - 2.0 * s @ AtWr + s @ self.F @ s
        )
        return s, chi2, cho

    def _log_evidence(self, s, chi2, coefficient, cho) -> float:
        """The same evidence `fitting.LinearSystem` maximises, up to constants
        that do not depend on the coefficient."""
        H = coefficient * self.Cinv
        log_det_M = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        sign, log_det_H = np.linalg.slogdet(H)
        if sign <= 0:
            return -np.inf
        return -0.5 * (chi2 + float(s @ H @ s) + log_det_M - log_det_H)

    def _choose_coefficient(
        self, disc: UniformDisc, criterion, chi2_target: float
    ) -> float:
        """Pick the GP prior strength: a number, 'evidence', or chi^2 target.

        'evidence': golden-section on log10(lam) after a coarse scan --
        the evidence in lam is smooth and single-peaked here (one linear
        system, one hyperparameter).
        'chi2': bisection to chi^2 = target * N, as `retune_regularization`
        does; chi^2 rises monotonically with lam.
        """
        if isinstance(criterion, (int, float)):
            return float(criterion)

        logs = np.linspace(-6.0, 8.0, 15)
        if criterion == "evidence":
            def score(lg):
                s, chi2, cho = self._solve(disc, 10.0**lg)
                return self._log_evidence(s, chi2, 10.0**lg, cho)

            vals = [score(lg) for lg in logs]
            j = int(np.nanargmax(vals))
            lo, hi = logs[max(0, j - 1)], logs[min(len(logs) - 1, j + 1)]
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            c, d = b - gr * (b - a), a + gr * (b - a)
            fc, fd = score(c), score(d)
            for _ in range(40):
                if b - a < 0.01:
                    break
                if fc > fd:
                    b, d, fd = d, c, fc
                    c = b - gr * (b - a)
                    fc = score(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + gr * (b - a)
                    fd = score(d)
            return 10.0 ** (0.5 * (a + b))

        if criterion == "chi2":
            target = chi2_target * self.n_data
            chi2s = []
            for lg in logs:
                chi2s.append(self._solve(disc, 10.0**lg)[1])
            chi2s = np.asarray(chi2s)
            if chi2s[-1] < target:      # even the stiffest prior over-fits
                return float(10.0 ** logs[-1])
            if chi2s[0] > target:       # even the loosest cannot reach it
                return float(10.0 ** logs[0])
            j = int(np.argmax(chi2s > target))
            lo, hi = logs[j - 1], logs[j]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if self._solve(disc, 10.0**mid)[1] > target:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 0.005:
                    break
            return float(10.0 ** (0.5 * (lo + hi)))

        raise ValueError(f"unknown coefficient criterion: {criterion!r}")

    # ---------------------------------------------------------------- driver
    def fit(
        self,
        n_rings: int = 6,
        az_oversample: float = 1.0,
        coefficient="evidence",
        chi2_target: float = 1.0,
        scale: float | None = None,
        nu: float = 1.5,
        reg: str = "matern",
        envelope_fwhm: float | None = None,
        envelope_floor: float = 1e-2,
        n_iter: int = 2,
        refit_disc: bool = True,
    ) -> DiscFitResult:
        """Fit disc, then perturbations, alternating `n_iter` times.

        Parameters
        ----------
        n_rings
            Radial resolution of the polar grid.  Cells are ~R/n_rings on a
            side; there is no point pushing far beyond the resolution
            lambda / (2 b_max) of the data.
        coefficient
            Prior strength: ``"evidence"`` (default, maximise the linear
            system's evidence), ``"chi2"`` (bisect to ``chi2_target * N``),
            or an explicit number.
        reg
            Prior type: ``matern`` (default), ``exponential`` (Matérn ν=0.5),
            ``gaussian`` (Matérn × Gaussian envelope on the disc), or
            ``constant`` (Tikhonov, no spatial correlation).
        scale
            Matern correlation length [arcsec].  Default: one ring width.
            Ignored for ``constant``.
        n_iter
            Alternations of (disc refit against perturbation-subtracted
            data) and (perturbation re-solve).  The disc parameters and the
            low-order perturbation modes are degenerate, so the disc is
            always fitted *first* and the perturbations mop up the rest;
            with ``refit_disc=False`` the disc from the first pass is held.
        """
        disc = self.fit_disc()
        disc_chi2 = self._disc_chi2_profiled(
            disc.radius_arcsec, *disc.centre_grid)[0]

        grid_kw = dict(
            n_rings=n_rings, az_oversample=az_oversample, scale=scale, nu=nu,
            reg=reg, envelope_fwhm=envelope_fwhm, envelope_floor=envelope_floor,
        )
        self._build_grid(disc, **grid_kw)
        lam = self._choose_coefficient(disc, coefficient, chi2_target)
        s, chi2, cho = self._solve(disc, lam)
        logger.info(
            "  polar grid: %d cells in %d rings; reg=%s, coefficient %.3g, "
            "chi2/N %.3f -> %.3f",
            self.grid.n_cells, n_rings, self._reg, lam,
            disc_chi2 / self.n_data, chi2 / self.n_data,
        )

        for _ in range(max(0, n_iter - 1)):
            if not refit_disc:
                break
            # refit the disc against the data minus the current perturbations
            pert_vis = self._model_perturbation_vis(s)
            saved = (self.d_stack, self.wd_stack, self.chi2_const)
            self.d_stack = saved[0] - self._stack(pert_vis)
            self.wd_stack = self.w_stack * self.d_stack
            self.chi2_const = float(np.sum(self.w_stack * self.d_stack**2))
            new_disc = self.fit_disc(disc)
            self.d_stack, self.wd_stack, self.chi2_const = saved

            moved = np.hypot(
                new_disc.d_ra - disc.d_ra, new_disc.d_dec - disc.d_dec
            )
            dR = abs(new_disc.radius_arcsec - disc.radius_arcsec)
            disc = new_disc
            # the grid follows the disc; rebuild only if it moved appreciably
            if moved > 0.02 * disc.radius_arcsec or dR > 0.02 * disc.radius_arcsec:
                self._build_grid(disc, **grid_kw)
            lam = self._choose_coefficient(disc, coefficient, chi2_target)
            s, chi2, cho = self._solve(disc, lam)

        # sampling covariance of s: M^-1 F M^-1 (as `SingleFit` quotes)
        Minv = cho_solve(cho, np.eye(self.grid.n_cells), check_finite=False)
        cov = Minv @ self.F @ Minv
        errors = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        model_vis = (
            disc.visibilities(self.uv) + self._model_perturbation_vis(s)
        )
        return DiscFitResult(
            disc=disc, grid=self.grid, perturbations=s,
            perturbation_errors=errors, chi_squared=chi2, n_data=self.n_data,
            coefficient=lam, disc_chi_squared=disc_chi2,
            regularization=self._reg, scale=self._scale, nu=self._nu,
            model_visibilities=model_vis,
            uv_wavelengths=self.uv,
        )

    def _model_perturbation_vis(self, s: np.ndarray) -> np.ndarray:
        stacked = self.A_stack @ s
        return stacked[: self.n_vis] + 1j * stacked[self.n_vis:]


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def gaussian_spot_visibilities(
    uv_wavelengths: np.ndarray,
    flux_jy: float,
    d_ra: float,
    d_dec: float,
    sigma_arcsec: float,
) -> np.ndarray:
    """A compact Gaussian 'hot spot', analytic (for mocks and comparisons)."""
    from .pointsource import gaussian_visibilities

    y, x = sky_to_grid(d_ra, d_dec)
    return flux_jy * gaussian_visibilities(uv_wavelengths, y, x, sigma_arcsec)


def make_disc_dataset(
    n_vis: int = 5000,
    frequency_hz: float = 330e9,
    disc: UniformDisc | None = None,
    spots: list | None = None,
    sigma_jy: float = 5e-4,
    resolution_elements: float = 4.0,
    seed: int = 0,
):
    """A mock uniform disc plus Gaussian hot spots, all analytic.

    `spots` is a list of ``(flux_jy, d_ra, d_dec, sigma_arcsec)``; positive
    flux is a hot spot, negative a dark one.  The maximum baseline is sized
    so the disc diameter spans `resolution_elements` resolution elements.

    Returns ``(uvdata, disc, spots)``.
    """
    from .mock import random_uv_coverage, uv_of
    from .uvdata import C_M_S, UVData

    if disc is None:
        disc = UniformDisc(flux_jy=0.35, radius_arcsec=0.030)
    if spots is None:
        spots = []
    # b_max such that lambda / b_max = diameter / resolution_elements
    diameter_rad = 2.0 * disc.radius_arcsec * ARCSEC_TO_RAD
    b_max = resolution_elements * (C_M_S / frequency_hz) / diameter_rad
    uvw = random_uv_coverage(n_vis, b_max, frequency_hz, seed=seed)
    uv = uv_of(uvw, frequency_hz)

    vis = disc.visibilities(uv)
    for flux, d_ra, d_dec, sig in spots:
        vis = vis + gaussian_spot_visibilities(uv, flux, d_ra, d_dec, sig)

    rng = np.random.default_rng(seed + 1)
    noisy = vis + rng.normal(0, sigma_jy, n_vis) + 1j * rng.normal(
        0, sigma_jy, n_vis)
    uvd = UVData(
        uvw=uvw,
        frequencies=np.array([frequency_hz]),
        data=noisy[None, :],
        noise=np.full((1, n_vis), sigma_jy + 1j * sigma_jy),
        meta={"telescope": "mock", "dish_diameter_m": 12.0,
              "phase_centre_ra_deg": 68.0, "phase_centre_dec_deg": -62.1},
    )
    return uvd, disc, spots


def make_rdor_mock(
    n_vis: int = 4000,
    sigma_jy: float = 5e-4,
    seed: int = 0,
):
    """An R Dor-like photosphere: ~35 mas radius disc plus two spots."""
    disc = UniformDisc(flux_jy=0.40, radius_arcsec=0.035)
    spots = [
        (0.025, 0.015, 0.010, 0.006),    # hot spot, NE
        (-0.012, -0.012, 0.008, 0.007),  # dark patch, SW
    ]
    return make_disc_dataset(
        n_vis=n_vis, frequency_hz=330e9, disc=disc, spots=spots,
        sigma_jy=sigma_jy, resolution_elements=5.0, seed=seed,
    )


def gaussian_spot_image(
    shape: tuple[int, int],
    pixel_scale: float,
    spots: list,
    disc: UniformDisc,
    fractional: bool = True,
    outside=np.nan,
) -> np.ndarray:
    """Render analytic Gaussian spots as a native (row 0 = +y) image.

    Each spot is ``(flux_jy, d_ra, d_dec, sigma_arcsec)``.  Outside the
    disc the map is ``outside`` so the comparison with the polar-grid
    reconstruction (which has no freedom beyond the limb) is fair.
    """
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    y = (cy - yy) * pixel_scale          # grid y = dDec
    x = (xx - cx) * pixel_scale          # grid x = -dRA
    d_ra, d_dec = -x, y
    img = np.zeros(shape, dtype=float)
    for flux, s_ra, s_dec, sig in spots:
        r2 = (d_ra - s_ra) ** 2 + (d_dec - s_dec) ** 2
        img += flux / (2.0 * np.pi * sig**2) * np.exp(-0.5 * r2 / sig**2)
    r = np.hypot(d_ra - disc.d_ra, d_dec - disc.d_dec)
    img = np.where(r <= disc.radius_arcsec, img, outside)
    if fractional:
        img = img / disc.surface_brightness
    return img


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def run_disc(
    dataset,
    disc: UniformDisc | None = None,
    radius_arcsec: float | None = None,
    flux_jy: float | None = None,
    centre=None,
    n_rings: int = 6,
    az_oversample: float = 1.0,
    coefficient="evidence",
    chi2_target: float = 1.0,
    scale: float | None = None,
    nu: float = 1.5,
    reg: str = "matern",
    envelope_fwhm: float | None = None,
    envelope_floor: float = 1e-2,
    n_iter: int = 2,
    refit_disc: bool = True,
    out: str | Path | None = "pyuvimage_disc_out",
    fov: float | None = None,
    pixel_scale: float | None = None,
    write: bool = True,
    truth_disc: UniformDisc | None = None,
    truth_spots: list | None = None,
) -> DiscFitResult:
    """Fit a uniform disc plus polar-grid perturbations and write products.

    Parameters
    ----------
    dataset
        A `UVData` (or multi-spw dataset), or a path from ``pyuvimage import``.
    disc
        Starting disc.  If omitted, ``radius_arcsec`` / ``flux_jy`` /
        ``centre`` are used; a missing radius is guessed from the first
        visibility null.
    centre
        Image-plane ``(x, y)`` arcsec from the phase centre (+x right, +y
        up on the summary figure), the same convention as ``--point``.
        Converted to (dRA, dDec) internally.
    coefficient
        Prior strength: ``"evidence"``, ``"chi2"``, or a number.
    reg
        Prior type: ``matern`` (default), ``exponential``, ``gaussian``,
        or ``constant``.  Strength is still set by ``coefficient``.
    out
        Directory for FITS / JSON / PNG products.  Ignored if ``write`` is
        False.
    truth_disc, truth_spots
        Optional ground truth for mocks: if given, the summary figure and
        ``truth_perturbation.fits`` show the input spot map beside the
        reconstruction.
    """
    from .uvdata import read_dataset

    if isinstance(dataset, (str, Path)):
        dataset = read_dataset(dataset)
    if not hasattr(dataset, "flattened"):
        raise TypeError(
            f"dataset must be UVData-like with flattened(), got {type(dataset)}"
        )
    uv, data, noise = dataset.flattened()

    if disc is None:
        if radius_arcsec is None:
            radius_arcsec = guess_radius_arcsec(uv, data)
            logger.info("  guessed disc radius %.4g\" from the first |V| null",
                        radius_arcsec)
        if centre is None:
            d_ra, d_dec = 0.0, 0.0
        else:
            d_ra, d_dec = image_to_sky(float(centre[0]), float(centre[1]))
        if flux_jy is None:
            # zero-spacing flux: mean of the shortest 2% of baselines
            q = np.hypot(uv[:, 0], uv[:, 1])
            n_short = max(8, q.size // 50)
            flux_jy = float(np.mean(np.real(data[np.argsort(q)[:n_short]])))
            flux_jy = max(flux_jy, 1e-6)
        disc = UniformDisc(flux_jy, float(radius_arcsec), d_ra, d_dec)

    logger.info(
        "fitting disc + polar perturbations (%d visibilities, starting "
        "R = %.4g\", reg=%s)",
        uv.shape[0], disc.radius_arcsec, reg,
    )
    fitter = DiscPerturbationFit(uv, data, noise, disc)
    result = fitter.fit(
        n_rings=n_rings, az_oversample=az_oversample,
        coefficient=coefficient, chi2_target=chi2_target,
        scale=scale, nu=nu, reg=reg,
        envelope_fwhm=envelope_fwhm, envelope_floor=envelope_floor,
        n_iter=n_iter, refit_disc=refit_disc,
    )
    result.uv_wavelengths = uv
    result.data = data
    result.noise = noise
    result.meta = dict(getattr(dataset, "meta", {}) or {})

    logger.info(
        "  done: chi2/N %.3f (disc alone %.3f); max |dI/I| = %.3f",
        result.chi_squared_reduced,
        result.disc_chi_squared / result.n_data,
        float(np.max(np.abs(result.fractional))) if result.grid else 0.0,
    )
    if write and out is not None:
        write_disc_products(
            result, out, dataset=dataset, fov=fov, pixel_scale=pixel_scale,
            truth_disc=truth_disc, truth_spots=truth_spots,
        )
    return result


def write_disc_products(
    result: DiscFitResult,
    out_dir: str | Path,
    dataset=None,
    fov: float | None = None,
    pixel_scale: float | None = None,
    overwrite: bool = True,
    truth_disc: UniformDisc | None = None,
    truth_spots: list | None = None,
) -> dict:
    """Write FITS maps, a cell table, fit_parameters.json and summary.png."""
    from astropy.io import fits

    from .products import build_header, to_fits_orientation

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    disc = result.disc
    R = disc.radius_arcsec
    if fov is None:
        fov = 3.0 * 2.0 * R
    if pixel_scale is None:
        n_rings = result.grid.n_rings if result.grid else 6
        pixel_scale = R / max(n_rings, 1) / 8.0
    n = int(np.ceil(fov / pixel_scale))
    if n % 2 == 0:
        n += 1
    shape = (n, n)
    meta = {}
    if dataset is not None:
        meta = dict(getattr(dataset, "meta", {}) or {})
    meta = meta or getattr(result, "meta", {}) or {}

    model = result.model_image(shape, pixel_scale)
    frac = result.perturbation_image(shape, pixel_scale, fractional=True)
    sig = result.grid.render(result.significance, shape, pixel_scale)
    truth_frac = None
    if truth_spots:
        tdisc = truth_disc or result.disc
        truth_frac = gaussian_spot_image(
            shape, pixel_scale, truth_spots, tdisc, fractional=True,
        )

    extra = {
        "DISCFLUX": (disc.flux_jy, "uniform-disc flux [Jy]"),
        "DISCRAD": (disc.radius_arcsec, "uniform-disc radius [arcsec]"),
        "DISCDRA": (disc.d_ra, "disc centre dRA [arcsec]"),
        "DISCDEC": (disc.d_dec, "disc centre dDec [arcsec]"),
        "NCELLS": (result.grid.n_cells, "polar perturbation cells"),
        "NRINGS": (result.grid.n_rings, "polar rings"),
        "REGKIND": (result.regularization, "source prior type"),
        "REGCOEF": (result.coefficient, "GP prior coefficient"),
        "CHI2N": (result.chi_squared_reduced, "chi^2 / N"),
        "CHI2DISC": (
            result.disc_chi_squared / result.n_data,
            "uniform-disc-only chi^2 / N",
        ),
    }
    freq = None
    if dataset is not None and getattr(dataset, "frequencies", None) is not None:
        freq = np.atleast_1d(np.mean(np.asarray(dataset.frequencies)))

    written = {}

    def _w(name, native, bunit):
        hdr = build_header(n, pixel_scale, meta, bunit, extra=extra)
        if freq is not None:
            hdr["RESTFRQ"] = (float(freq[0]), "representative frequency [Hz]")
        path = out / name
        fits.writeto(
            path, np.asarray(to_fits_orientation(native), dtype=np.float32),
            hdr, overwrite=overwrite,
        )
        written[name] = path

    _w("model.fits", model, "Jy/arcsec2")
    _w("perturbation.fits", frac, "fraction")
    _w("significance.fits", sig, "sigma")
    if truth_frac is not None:
        _w("truth_perturbation.fits", truth_frac, "fraction")

    cells = {
        "r_mid_arcsec": result.grid.r_mid,
        "theta_mid_rad": result.grid.theta_mid,
        "area_arcsec2": result.grid.areas,
        "perturbation_jy_arcsec2": result.perturbations,
        "perturbation_error_jy_arcsec2": result.perturbation_errors,
        "fractional": result.fractional,
        "significance": result.significance,
    }
    np.savez(out / "perturbations.npz", **cells)
    written["perturbations.npz"] = out / "perturbations.npz"

    params = result.as_dict()
    params["grid"] = {
        "n_rings": int(result.grid.n_rings),
        "n_cells": int(result.grid.n_cells),
        "cells_per_ring": [int(c) for c in result.grid.cells_per_ring],
        "az_oversample": float(result.grid.az_oversample),
        "radius_arcsec": float(result.grid.radius),
    }
    params["render"] = {"fov_arcsec": float(fov), "pixel_scale_arcsec": float(pixel_scale)}
    if truth_spots:
        params["truth_spots"] = [
            {"flux_jy": float(f), "d_ra_arcsec": float(ra),
             "d_dec_arcsec": float(dec), "sigma_arcsec": float(sig)}
            for f, ra, dec, sig in truth_spots
        ]
    (out / "fit_parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    written["fit_parameters.json"] = out / "fit_parameters.json"

    uv = getattr(result, "uv_wavelengths", None)
    data = getattr(result, "data", None)
    fig_path = out / "summary.png"
    _write_summary_figure(
        result, model, frac, sig, pixel_scale, fig_path,
        uv_wavelengths=uv, data=data, truth_frac=truth_frac,
    )
    written["summary.png"] = fig_path
    logger.info("  disc products written to %s", out)
    return written


def _write_summary_figure(
    result: DiscFitResult,
    model: np.ndarray,
    frac: np.ndarray,
    sig: np.ndarray,
    pixel_scale: float,
    path: Path,
    uv_wavelengths=None,
    data=None,
    truth_frac: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = model.shape[0]
    half = 0.5 * (n - 1) * pixel_scale
    # dRA increases left (radio convention); dDec increases up.
    extent = [half, -half, -half, half]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0))

    def _circle(ax, disc=None):
        d = disc or result.disc
        ax.add_patch(plt.Circle(
            (d.d_ra, d.d_dec), d.radius_arcsec,
            fill=False, ec="0.4", lw=0.8, ls="--",
        ))
        ax.set_aspect("equal")
        ax.set_xlabel("dRA [arcsec]")
        ax.set_ylabel("dDec [arcsec]")

    # Shared colour scale for truth vs recovered fractional maps
    amax = np.nanmax(np.abs(frac))
    if truth_frac is not None:
        amax = max(amax, float(np.nanmax(np.abs(truth_frac))))
    amax = 0.05 if not np.isfinite(amax) or amax == 0 else amax

    if truth_frac is not None:
        ax = axes[0, 0]
        im = ax.imshow(
            np.flipud(truth_frac), origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-amax, vmax=amax, interpolation="nearest",
        )
        ax.set_title("input $\\delta I / I_0$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _circle(ax)

        ax = axes[0, 1]
        im = ax.imshow(
            np.flipud(frac), origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-amax, vmax=amax, interpolation="nearest",
        )
        ax.set_title("recovered $\\delta I / I_0$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _circle(ax)
    else:
        ax = axes[0, 0]
        vmax = np.nanpercentile(np.abs(model), 99.5)
        im = ax.imshow(
            np.flipud(model), origin="lower", extent=extent,
            cmap="magma", vmin=0, vmax=vmax, interpolation="nearest",
        )
        ax.set_title("model (disc + $\\delta I$)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jy / arcsec$^2$")
        _circle(ax)

        ax = axes[0, 1]
        im = ax.imshow(
            np.flipud(frac), origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-amax, vmax=amax, interpolation="nearest",
        )
        ax.set_title("fractional perturbation $\\delta I / I_0$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _circle(ax)

    ax = axes[1, 0]
    if truth_frac is not None:
        vmax = np.nanpercentile(np.abs(model), 99.5)
        im = ax.imshow(
            np.flipud(model), origin="lower", extent=extent,
            cmap="magma", vmin=0, vmax=vmax, interpolation="nearest",
        )
        ax.set_title("model (disc + $\\delta I$)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jy / arcsec$^2$")
        _circle(ax)
    else:
        smax = np.nanmax(np.abs(sig))
        smax = 3.0 if not np.isfinite(smax) or smax == 0 else max(smax, 3.0)
        im = ax.imshow(
            np.flipud(sig), origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-smax, vmax=smax, interpolation="nearest",
        )
        ax.set_title("significance")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\sigma$")
        _circle(ax)

    ax = axes[1, 1]
    if uv_wavelengths is not None and data is not None:
        q = np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1])
        amp = np.abs(data)
        model_amp = np.abs(result.model_visibilities)
        disc_amp = np.abs(result.disc.visibilities(uv_wavelengths))
        n_bins = 25
        edges = np.linspace(0.0, float(q.max()), n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(q, edges) - 1, 0, n_bins - 1)

        def _bin(y):
            out = np.full(n_bins, np.nan)
            for i in range(n_bins):
                sel = idx == i
                if np.any(sel):
                    out[i] = float(np.mean(y[sel]))
            return out

        ax.plot(centres / 1e6, _bin(amp), "k.", ms=4, label="data")
        ax.plot(centres / 1e6, _bin(disc_amp), color="0.55", lw=1.2, label="disc")
        ax.plot(centres / 1e6, _bin(model_amp), color="C1", lw=1.4, label="disc+$\\delta I$")
        ax.set_xlabel("$q$ [M$\\lambda$]")
        ax.set_ylabel("$|V|$ [Jy]")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("azimuthally averaged amplitude")
    else:
        ax.axis("off")

    d = result.as_dict()
    fig.suptitle(
        f"F = {d['disc']['flux_jy']:.3g} Jy,  "
        f"R = {d['disc']['radius_arcsec']*1e3:.2f} mas,  "
        f"$\\chi^2/N$ = {d['chi_squared_reduced']:.3f}  "
        f"(disc {d['disc_only_chi_squared_reduced']:.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

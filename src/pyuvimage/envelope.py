"""A Gaussian *envelope* source prior for sparse visibility data.

The default Matern prior is stationary: it says the sky is smooth on beam
scales, but nothing about *where* the flux sits.  With few visibilities that
is not enough -- a dirty-beam sidelobe far from the source is as acceptable
to the prior as the source itself, so sidelobe structure leaks into the
model.

This module adds the missing spatial information in the simplest possible
form: a zero-mean Gaussian process whose prior *standard deviation* follows a
2D Gaussian centred on the image (by default the size of the synthesised
beam).  Near the centre the prior is permissive; far from it the prior
standard deviation falls to a small floor, so those pixels are pulled towards
zero unless the data insist otherwise.

Concretely the kernel covariance becomes

    C_ij = Matern(|r_i - r_j|; scale, nu) * w_i * w_j
    w_i  = floor + (1 - floor) * exp(-0.5 * |r_i - centre|^2 / sigma_env^2)

and the regularization matrix is H = coefficient * C^-1, exactly as for the
plain Matern prior (which is recovered as envelope_fwhm -> infinity).

Caveat: this prior assumes the emission is near the phase centre.  That is
usually true for a targeted observation, but it will bias against genuinely
offset or extended emission -- widen `envelope_fwhm` (or use `--reg matern`)
when that is a concern.
"""

from __future__ import annotations

import numpy as np

from autoarray.inversion.regularization.matern_kernel import (
    MaternKernel,
    apply_jitter,
    inv_via_cholesky,
    matern_cov_matrix_from,
)

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# ---------------------------------------------------------------------------
# Covariance cache.
#
# Building the Matern covariance (a dense N x N Bessel evaluation) and
# inverting it costs ~11 s at 4096 mesh pixels and dominates a fit.  But it
# depends only on the mesh, the correlation scale/smoothness and the envelope
# -- *not* on the regularisation coefficient, which is what the hyperparameter
# search actually varies.  Caching it turns a ~20-trial search from ~20
# inversions into one.
# ---------------------------------------------------------------------------
_COV_CACHE: dict = {}
_COV_CACHE_MAX = 3


def _cache_key(pixel_points: np.ndarray, *parts) -> tuple:
    pts = np.ascontiguousarray(np.asarray(pixel_points, dtype=float))
    return (pts.shape, hash(pts.tobytes()), *parts)


def cached_inverse_covariance(
    pixel_points: np.ndarray,
    scale: float,
    nu: float,
    weights: np.ndarray | None = None,
    jitter: float = 1e-8,
    jitter_relative: bool = False,
    weights_key: tuple = (),
) -> np.ndarray:
    """C^-1 for a (weighted) Matern kernel, cached across coefficient trials."""
    key = _cache_key(
        pixel_points, float(scale), float(nu), float(jitter),
        bool(jitter_relative), weights_key,
    )
    hit = _COV_CACHE.get(key)
    if hit is not None:
        return hit
    cov = matern_cov_matrix_from(
        scale=scale, nu=nu, pixel_points=pixel_points, weights=weights,
        jitter=jitter, jitter_relative=jitter_relative,
    )
    inv = inv_via_cholesky(cov)
    if len(_COV_CACHE) >= _COV_CACHE_MAX:
        _COV_CACHE.pop(next(iter(_COV_CACHE)))
    _COV_CACHE[key] = inv
    return inv


def clear_covariance_cache() -> None:
    _COV_CACHE.clear()


class _OwnCovarianceShortcuts:
    """The evidence shortcuts autoarray offers kernel priors, from *this*
    kernel's covariance.

    `MaternKernel` ships two opt-in short cuts that the inversion consults
    only under non-default settings -- `log_det_regularization_matrix_term_from`
    under ``Settings.log_det_method == "slogdet"`` and
    `regularization_term_from` under ``regularization_term_method ==
    "cho_solve"`` -- and both rebuild a *plain* Matern covariance from
    ``scale`` and ``nu`` to do it. Every prior in this module is a Matern
    kernel with something else multiplied in (an envelope, a brightness
    weighting, a varying correlation length), so inheriting those would report
    the log-determinant and the quadratic form of a covariance the prior does
    not use. Under today's default settings neither is consulted, and
    `fitting.LinearSystem` deliberately reproduces the default formed-matrix
    Cholesky rather than these; they exist so that the day either setting is
    switched on, the evidence stays the evidence of the prior that was fitted.

    Subclasses provide `_covariance(linear_obj, xp)`: the jittered covariance
    whose inverse `regularization_matrix_from` returns times the coefficient.
    """

    def _covariance(self, linear_obj, xp=np):  # pragma: no cover - abstract
        raise NotImplementedError

    def log_det_regularization_matrix_term_from(self, linear_obj, xp=np) -> float:
        """log det H = pixels * log(coefficient) - log det C."""
        c = self._covariance(linear_obj, xp=xp)
        log_det_cov = 2.0 * xp.sum(xp.log(xp.diag(xp.linalg.cholesky(c))))
        return float(c.shape[0] * np.log(self.coefficient) - log_det_cov)

    def regularization_term_from(self, linear_obj, reconstruction, xp=np) -> float:
        """s^T H s = coefficient * s^T C^-1 s, through a Cholesky solve of C."""
        from autoarray.inversion.regularization.matern_kernel import (
            quadratic_form_via_cholesky,
        )

        return self.coefficient * quadratic_form_via_cholesky(
            self._covariance(linear_obj, xp=xp), reconstruction, xp=xp
        )


class CachedMaternKernel(MaternKernel):
    """`MaternKernel` whose covariance inverse is cached (see above).

    A plain Matern kernel, so the shortcuts it inherits from upstream describe
    its own covariance and are left alone.
    """

    def regularization_matrix_from(self, linear_obj, xp=np) -> np.ndarray:
        if xp is not np:  # JAX path: leave upstream behaviour untouched
            return super().regularization_matrix_from(linear_obj, xp=xp)
        inv = cached_inverse_covariance(
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            scale=self.scale, nu=self.nu, jitter=self.jitter_value,
            jitter_relative=self.jitter_relative,
        )
        return self.coefficient * inv


class GaussianEnvelopeMatern(_OwnCovarianceShortcuts, MaternKernel):
    """Matern GP prior modulated by a Gaussian envelope on the prior width."""

    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        nu: float = 1.5,
        envelope_fwhm: float = 1.0,
        envelope_floor: float = 1e-2,
        centre: tuple[float, float] = (0.0, 0.0),
    ):
        """
        Parameters
        ----------
        coefficient
            Overall prior strength (as for any regularization scheme).
        scale, nu
            Matern correlation length [arcsec] and smoothness.
        envelope_fwhm
            FWHM [arcsec] of the Gaussian envelope on the prior standard
            deviation.  Defaults to the synthesised beam size upstream.
        envelope_floor
            Prior standard deviation far from the centre, relative to the
            centre.  Small values suppress distant structure strongly; 0 is
            not allowed (it would make the covariance singular).
        centre
            Envelope centre (y, x) in arcsec relative to the phase centre.
        """
        if not 0.0 < envelope_floor < 1.0:
            raise ValueError("envelope_floor must lie in (0, 1)")
        if envelope_fwhm <= 0:
            raise ValueError("envelope_fwhm must be positive")
        super().__init__(
            coefficient=coefficient, scale=scale, nu=nu,
            # the weighted covariance has a hugely varying diagonal, so the
            # stabilising jitter must be relative (see apply_jitter upstream)
            jitter_relative=True,
        )
        self.envelope_fwhm = float(envelope_fwhm)
        self.envelope_floor = float(envelope_floor)
        self.centre = (float(centre[0]), float(centre[1]))

    # ------------------------------------------------------------------
    def envelope_weights(self, pixel_points) -> np.ndarray:
        """Prior standard deviation at each mesh pixel (peak 1 at the centre)."""
        pts = np.asarray(pixel_points, dtype=float)
        dy = pts[:, 0] - self.centre[0]
        dx = pts[:, 1] - self.centre[1]
        sigma = self.envelope_fwhm / SIGMA_TO_FWHM
        g = np.exp(-0.5 * (dy**2 + dx**2) / sigma**2)
        return self.envelope_floor + (1.0 - self.envelope_floor) * g

    def _covariance(self, linear_obj, xp=np) -> np.ndarray:
        pts = linear_obj.source_plane_mesh_grid.array
        return matern_cov_matrix_from(
            scale=self.scale,
            nu=self.nu,
            pixel_points=pts,
            weights=self.envelope_weights(pts),
            jitter=self.jitter_value,
            jitter_relative=True,
            xp=xp,
        )

    def regularization_matrix_from(self, linear_obj, xp=np) -> np.ndarray:
        if xp is not np:
            return self.coefficient * inv_via_cholesky(
                self._covariance(linear_obj, xp=xp), xp=xp
            )
        pts = linear_obj.source_plane_mesh_grid.array
        inv = cached_inverse_covariance(
            pixel_points=pts, scale=self.scale, nu=self.nu,
            weights=self.envelope_weights(pts), jitter=self.jitter_value,
            jitter_relative=True,
            weights_key=(
                self.envelope_fwhm, self.envelope_floor, self.centre,
            ),
        )
        return self.coefficient * inv


def peak_offset_arcsec(
    dirty_image: np.ndarray, pixel_scale: float
) -> tuple[float, float]:
    """(y, x) offset in arcsec of the dirty image's peak from the grid centre.

    Native arrays have row 0 at +y (North) and column 0 at -x, so
    y = (cy - row) * pixel_scale and x = (col - cx) * pixel_scale.
    """
    img = np.nan_to_num(np.asarray(dirty_image, dtype=float))
    ny, nx = img.shape
    row, col = np.unravel_index(int(np.argmax(img)), img.shape)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    return ((cy - row) * pixel_scale, (col - cx) * pixel_scale)


def estimate_envelope(
    dirty_image: np.ndarray,
    pixel_scale: float,
    rms: float,
    beam_fwhm: float,
    n_sigma: float = 5.0,
    min_beams: float = 3.0,
    max_fwhm: float | None = None,
) -> tuple[tuple[float, float], float]:
    """Locate and size the Gaussian envelope from the dirty image.

    The envelope is centred on the dirty peak -- not the phase centre, since
    the source need not sit there -- and sized from the second moment of the
    significant emission around it.  The dirty image is the true sky convolved
    with the dirty beam, so this over-estimates the source size, which is what
    we want from a permissive envelope.

    Returns ((y, x) centre in arcsec, FWHM in arcsec).
    """
    img = np.nan_to_num(np.asarray(dirty_image, dtype=float))
    centre = peak_offset_arcsec(img, pixel_scale)
    floor_fwhm = min_beams * beam_fwhm

    peak = float(np.nanmax(img))
    threshold = max(n_sigma * rms, 0.2 * peak)
    sig = img > threshold
    if sig.sum() < 4:
        fwhm = floor_fwhm
    else:
        ny, nx = img.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        y = (cy - yy) * pixel_scale - centre[0]
        x = (xx - cx) * pixel_scale - centre[1]
        w = np.clip(img, 0.0, None) * sig
        total = w.sum()
        var = (w * (y**2 + x**2)).sum() / total / 2.0  # per axis
        fwhm = SIGMA_TO_FWHM * np.sqrt(max(var, 0.0))
        fwhm = max(fwhm, floor_fwhm)
    if max_fwhm is not None:
        fwhm = min(fwhm, max_fwhm)
    return centre, float(fwhm)


class AdaptiveMatern(_OwnCovarianceShortcuts, MaternKernel):
    """Matern GP prior whose width follows a first-pass brightness map.

    The analogue, for a pixelized source, of the adaptive treatment PyAutoLens
    uses for foreground lens light: rather than sub-sampling the grid (a no-op
    here, since the model mesh and image grid are aligned), the *prior* is
    allowed to vary across the image.  A single global correlation length has
    to compromise between a bright, compact core and faint extended emission,
    and the core loses -- which shows up as a strong residual at the peak.

    Here the prior standard deviation is raised where a first-pass model says
    the source is bright, so the core is smoothed less and the faint outskirts
    more:

        w_i = floor + (1 - floor) * (b_i / max(b))^power

    with `b` the first-pass model.  `power = 0` recovers the plain Matern
    prior.  Optionally multiplied by a Gaussian envelope (see
    `GaussianEnvelopeMatern`) so the two can be combined.
    """

    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        nu: float = 1.5,
        brightness: np.ndarray | None = None,
        floor: float = 1e-2,
        power: float = 1.0,
    ):
        if brightness is None:
            raise ValueError("AdaptiveMatern requires a first-pass brightness map")
        if not 0.0 < floor < 1.0:
            raise ValueError("floor must lie in (0, 1)")
        super().__init__(
            coefficient=coefficient, scale=scale, nu=nu, jitter_relative=True
        )
        b = np.clip(np.asarray(brightness, dtype=float).ravel(), 0.0, None)
        peak = float(b.max()) if b.size and b.max() > 0 else 1.0
        self.brightness = b / peak
        self.floor = float(floor)
        self.power = float(power)

    def adaptive_weights(self) -> np.ndarray:
        w = self.brightness ** self.power
        return self.floor + (1.0 - self.floor) * w

    def _weights_for(self, pts) -> np.ndarray:
        w = self.adaptive_weights()
        if w.size != pts.shape[0]:
            raise ValueError(
                f"brightness map has {w.size} pixels but the mesh has "
                f"{pts.shape[0]}"
            )
        return w

    def _covariance(self, linear_obj, xp=np) -> np.ndarray:
        pts = linear_obj.source_plane_mesh_grid.array
        return matern_cov_matrix_from(
            scale=self.scale, nu=self.nu, pixel_points=pts,
            weights=self._weights_for(pts), jitter=self.jitter_value,
            jitter_relative=True, xp=xp,
        )

    def regularization_matrix_from(self, linear_obj, xp=np) -> np.ndarray:
        pts = linear_obj.source_plane_mesh_grid.array
        w = self._weights_for(pts)
        inv = cached_inverse_covariance(
            pixel_points=pts, scale=self.scale, nu=self.nu, weights=w,
            jitter=self.jitter_value, jitter_relative=True,
            weights_key=("adapt", hash(w.tobytes()), self.floor, self.power),
        )
        return self.coefficient * inv


class GibbsMatern(_OwnCovarianceShortcuts, MaternKernel):
    """Non-stationary GP prior: the *correlation length* varies with brightness.

    `AdaptiveMatern` varies the prior's amplitude but keeps one global
    correlation length, so at an unresolved source the prior still asks for
    beam-scale smoothness -- exactly the wrong request.  This kernel shortens
    the correlation length where a first-pass model says the source is bright:

        ell_i = ell_min + (ell_max - ell_min) * (1 - (b_i / max b)^power)

    with ell_max the beam and ell_min a fraction of it.  The covariance is the
    Gibbs (non-stationary squared-exponential) kernel, which stays positive
    definite for any smoothly varying ell:

        C_ij = (2 ell_i ell_j / (ell_i^2 + ell_j^2)) * exp(-d_ij^2 / (ell_i^2 + ell_j^2))

    optionally modulated by amplitude weights w_i w_j as in `AdaptiveMatern`.
    Note this is a squared-exponential family, so `nu` does not apply.
    """

    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        nu: float = 1.5,          # accepted for interface symmetry; unused
        brightness: np.ndarray | None = None,
        ell_floor: float = 0.25,
        power: float = 1.0,
        weights: np.ndarray | None = None,
    ):
        if brightness is None:
            raise ValueError("GibbsMatern requires a first-pass brightness map")
        if not 0.0 < ell_floor <= 1.0:
            raise ValueError("ell_floor must lie in (0, 1]")
        super().__init__(
            coefficient=coefficient, scale=scale, nu=nu, jitter_relative=True
        )
        b = np.clip(np.asarray(brightness, dtype=float).ravel(), 0.0, None)
        peak = float(b.max()) if b.size and b.max() > 0 else 1.0
        self.brightness = b / peak
        self.ell_floor = float(ell_floor)
        self.power = float(power)
        self.weights = None if weights is None else np.asarray(weights, float)

    def length_scales(self) -> np.ndarray:
        ell_min = self.ell_floor * self.scale
        return ell_min + (self.scale - ell_min) * (
            1.0 - self.brightness**self.power
        )

    def _length_scales_for(self, pts) -> np.ndarray:
        ell = self.length_scales()
        if ell.size != pts.shape[0]:
            raise ValueError(
                f"brightness map has {ell.size} pixels but the mesh has "
                f"{pts.shape[0]}"
            )
        return ell

    def _covariance(self, linear_obj, xp=np) -> np.ndarray:
        pts = np.asarray(linear_obj.source_plane_mesh_grid.array)
        return _gibbs_covariance(
            pts, self._length_scales_for(pts), self.weights, self.jitter_value
        )

    def regularization_matrix_from(self, linear_obj, xp=np) -> np.ndarray:
        pts = np.asarray(linear_obj.source_plane_mesh_grid.array)
        ell = self._length_scales_for(pts)
        inv = _cached_gibbs_inverse(
            pts, ell, self.weights, self.jitter_value,
            key=(hash(ell.tobytes()),
                 None if self.weights is None else hash(self.weights.tobytes())),
        )
        return self.coefficient * inv


def _gibbs_covariance(pts, ell, weights, jitter) -> np.ndarray:
    """The jittered Gibbs covariance `GibbsMatern` inverts."""
    l2 = ell**2
    s = l2[:, None] + l2[None, :]
    sq = np.sum(pts * pts, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T), 0.0)
    cov = (2.0 * ell[:, None] * ell[None, :] / s) * np.exp(-d2 / s)
    if weights is not None:
        cov = cov * (weights[:, None] * weights[None, :])
    return apply_jitter(cov, jitter=jitter, jitter_relative=True)


def _cached_gibbs_inverse(pts, ell, weights, jitter, key):
    ckey = _cache_key(pts, "gibbs", *key, float(jitter))
    hit = _COV_CACHE.get(ckey)
    if hit is not None:
        return hit
    inv = inv_via_cholesky(_gibbs_covariance(pts, ell, weights, jitter))
    if len(_COV_CACHE) >= _COV_CACHE_MAX:
        _COV_CACHE.pop(next(iter(_COV_CACHE)))
    _COV_CACHE[ckey] = inv
    return inv

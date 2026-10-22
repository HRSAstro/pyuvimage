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


# ---------------------------------------------------------------------------
# the two sources of C and Dp
# ---------------------------------------------------------------------------

def _stacked_columns(uv, ys, xs) -> np.ndarray:
    """``[Re; Im]`` of unit points at grid (ys, xs) arcsec, (2 n_vis, k);
    the convention of `pointsource.point_visibilities`."""
    ys = np.asarray(ys, dtype=float).ravel()
    xs = np.asarray(xs, dtype=float).ravel()
    n = uv.shape[0]
    phase = np.multiply.outer(uv[:, 0], xs * ARCSEC_TO_RAD)
    phase += np.multiply.outer(uv[:, 1], ys * ARCSEC_TO_RAD)
    phase *= -2.0 * np.pi
    out = np.empty((2 * n, ys.size))
    np.cos(phase, out=out[:n])
    np.sin(phase, out=out[n:])
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

    def terms(self, positions) -> tuple[np.ndarray, np.ndarray]:
        P = _stacked_columns(self.uv, [p[0] for p in positions], [p[1] for p in positions])
        return P.T @ (self.w_stack[:, None] * P), P.T @ self.wd_stack

    def _image(self, c) -> np.ndarray:
        import autoarray as aa

        return np.asarray(self._adjoint(self.transformer, aa.Visibilities(c)).native).ravel()

    def lattice_dirty(self, ys, xs) -> np.ndarray:
        """Dp at every image pixel: the adjoint of the weighted data."""
        return self._image(self.w_re * self.data.real + 1j * self.w_im * self.data.imag)

    def lattice_cross(self, ys, xs, position) -> np.ndarray:
        """C between every image pixel and a point at `position`."""
        P = _stacked_columns(self.uv, [position[0]], [position[1]])[:, 0]
        n = self.uv.shape[0]
        return self._image(self.w_re * P[:n] + 1j * self.w_im * P[n:])

    def residual_image(self, positions, amplitudes) -> np.ndarray:
        """Dirty image of the residual visibilities [Jy/beam]."""
        model = np.zeros(self.uv.shape[0], dtype=complex)
        if len(positions):
            P = _stacked_columns(self.uv, [p[0] for p in positions], [p[1] for p in positions])
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

    def terms(self, positions) -> tuple[np.ndarray, np.ndarray]:
        ys = np.array([p[0] for p in positions], dtype=float)
        xs = np.array([p[1] for p in positions], dtype=float)
        C = self.grids.kernel(ys[:, None] - ys[None, :], xs[:, None] - xs[None, :])
        return 0.5 * (C + C.T), np.asarray(self.grids.dirty(ys, xs), dtype=float)

    def lattice_dirty(self, ys, xs) -> np.ndarray:
        return np.asarray(self.grids.dirty(ys, xs), dtype=float)

    def lattice_cross(self, ys, xs, position) -> np.ndarray:
        return np.asarray(self.grids.kernel(ys - position[0], xs - position[1]), dtype=float)

    def residual_image(self, positions, amplitudes) -> np.ndarray:
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

    def __init__(self, backend):
        self.backend = backend
        self.n_data = backend.n_data
        self.chi2_const = backend.chi2_const
        self._cross: dict = {}

    def solve(self, positions):
        """(fluxes, their covariance C^-1, chi^2) at fixed positions."""
        positions = list(positions)
        if not positions:
            return np.zeros(0), np.zeros((0, 0)), float(self.chi2_const)
        C, Dp = self.backend.terms(positions)
        C = C + np.eye(len(positions)) * GRAM_RIDGE * max(np.trace(C), 1e-30)
        cov = np.linalg.inv(C)
        a = cov @ Dp
        return a, cov, float(self.chi2_const - a @ Dp)

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
        if accepted:
            Kc = np.stack([self._lattice_cross(ys, xs, p) for p in accepted], axis=1)
            C, Dp = self.backend.terms(accepted)
            C = C + np.eye(len(accepted)) * GRAM_RIDGE * max(np.trace(C), 1e-30)
            Ci = np.linalg.inv(C)
            r -= Kc @ (Ci @ Dp)
            s -= np.einsum("ij,jk,ik->i", Kc, Ci, Kc)
        s = np.maximum(s, 1e-30 * self.backend.k0)
        return r / s, r / np.sqrt(s)

    def _lattice_cross(self, ys, xs, position):
        key = (float(position[0]), float(position[1]))
        hit = self._cross.get(key)
        if hit is None:
            hit = self._cross[key] = self.backend.lattice_cross(ys, xs, position)
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
    if refine and accepted:
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
    convention), or None to detect.
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
    stream = (streaming is True or (streaming == "auto" and on_disk
                                    and stm.nufftax_available()))
    if streaming == "auto" and on_disk and not stream:
        logger.info("streaming auto -> in memory: nufftax is not installed")

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
    logger.info("fitting point components only (%s)...",
                "user positions" if sky_positions else
                f"auto-detect above {significance:.1f} sigma, at most {max_points}")
    system = PointOnlySystem(backend)
    fit = fit_points_only(system, geometry, positions=sky_positions,
                          significance=significance, max_points=max_points,
                          refine=refine, beam_fwhm=beam_fwhm)
    for note in fit.log:
        logger.info("  %s", note)
    dish = dish_diameter or meta.get("dish_diameter_m")
    if pb_correction and not dish:
        logger.warning("no dish diameter known: point fluxes are not primary-beam "
                       "corrected (pass dish_diameter=... to enable)")
    pb_fwhm = _locate(fit.points, geometry, meta, freq, dish if pb_correction else None,
                      pb_factor)

    from .pointsource import restore_points

    residual = backend.residual_image(fit.grid_positions, fit.amplitudes)
    restored = restore_points(geometry.shape_native, geometry.pixel_scale, fit.points, beam)
    logger.info("points only: %d point(s), chi2/N %.4f%s, residual peak %.1f sigma, %.0f s",
                len(fit.points), fit.chi2_per_n,
                f" (errors inflated x{np.sqrt(fit.chi2_inflation):.3f})"
                if fit.chi2_inflation > 1 else "",
                float(np.nanmax(np.abs(residual))) / rms, time.time() - t0)
    for i, p in enumerate(fit.points):
        pos_err = ("" if p.d_ra_error is None else
                   f" +- ({1e3 * p.d_ra_error:.2f}, {1e3 * p.d_dec_error:.2f}) mas")
        pb = "" if p.flux_pbcor is None else f"; PB-corrected {p.flux_pbcor:.5g} +- {p.flux_error_pbcor:.2g} Jy"
        logger.info("  point %d: (dRA, dDec) = (%.4f, %.4f)\"%s, %.5g +- %.2g Jy (%.1f sigma)%s",
                    i + 1, p.d_ra, p.d_dec, pos_err, p.flux, p.flux_error, p.significance, pb)

    parameters = {
        "model": "points only",
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
            "method": ("inverse Fisher matrix of chi^2 in flux and position"
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
            "covariance_order": (["flux"] * len(fit.points) + ["y"] * len(fit.points)
                                 + ["x"] * len(fit.points)) if fit.positions_fitted
            else ["flux"] * len(fit.points),
            "covariance": np.asarray(fit.covariance).tolist(),
        },
    }
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

    model = add_points(np.zeros(geometry.shape_native), fit.points, pix)
    w("model.fits", model, hdr("Jy/pixel", PTFLUX=(float(np.sum(fit.amplitudes)),
                                                   "total point flux [Jy]")))
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
              (restored, f"{len(fit.points)} point(s), restored", "Jy/beam", "inferno", None),
              (resid, f"residual (peak {rmax:.1f}$\\sigma$, rms {np.nanstd(resid):.2f}$\\sigma$)",
               r"$\sigma$", "RdBu_r", (-rmax, rmax))]
    for ax, (img, title, unit, cmap, clim) in zip(axes, panels):
        im = ax.imshow(np.asarray(img), origin="upper", extent=ext, cmap=cmap)
        if clim is not None:
            im.set_clim(*clim)
        for p in fit.points:
            ax.plot(p.d_ra, p.d_dec, "o", mfc="none", mec="cyan", ms=11, mew=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dRA [arcsec]")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(unit, fontsize=9)
    axes[0].set_ylabel("dDec [arcsec]")
    fig.suptitle(f"points only | beam {beam.bmaj_arcsec:.3g}\" x {beam.bmin_arcsec:.3g}\" "
                 f"pa {beam.bpa_deg:.0f} deg | rms {rms:.3g} Jy/beam | "
                 f"chi2/N {fit.chi2_per_n:.4f}", fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)

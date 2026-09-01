"""The stacked-real / memoised `AugmentedSystem` reproduces the original code.

`pointsource.AugmentedSystem` was rewritten on 1 Sep 2026 for speed (one real
GEMM on ``[A.real; A.imag]`` instead of strided complex slices, per-position
column terms cached across `solve`, the lattice terms cached across `scan`).
Each optimisation is checked here against `_Reference`, a transcription of the
code it replaced, operating on the same complex arrays with the same ridges.
The tolerances are those of a different BLAS summation order (~1e-12
relative), not of any algorithmic change.  Measured on the crowded mock,
`fit_point_sources` went from 6.9 s to 1.1 s (auto-detection) and 13.4 s to
0.7 s (two user positions) with positions and fluxes agreeing to 1e-8
relative and chi^2 to 1e-10; at 1e5 visibilities and 576 mesh pixels one
`solve` went from 1.4 s to 0.12 s cold (one new column) and 8 ms with every
column cached, and a 64x64 `scan` from ~110 s (chunk 256; the default chunk of
1024 was killed at 7 GB) to 41 s the first time and 0.2 s on repeat.
"""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from pyuvimage import fitting, mock
from pyuvimage.pointsource import (
    AugmentedSystem,
    detection_lattice,
    fit_point_sources,
    gaussian_visibilities,
    point_visibilities,
    sky_to_grid,
)
from pyuvimage import pointsource

PRIOR = {"coefficient": 1e7, "scale": 0.25, "nu": 1.5}


class _Reference:
    """The pre-optimisation `AugmentedSystem`, kept verbatim as the oracle."""

    def __init__(self, inversion, dataset):
        self.uv = np.asarray(dataset.uv_wavelengths)
        noise = np.asarray(dataset.noise_map)
        data = np.asarray(dataset.data)
        self.w_re, self.w_im = 1.0 / noise.real**2, 1.0 / noise.imag**2
        self.d_re, self.d_im = data.real, data.imag
        self.A = np.asarray(inversion.operated_mapping_matrix)
        self.F = np.asarray(inversion.curvature_matrix)
        self.D = np.asarray(inversion.data_vector)
        self.H = np.asarray(inversion.regularization_matrix)
        self.h_scale = 1.0
        self._cho = cho_factor(self.F + self.H, lower=True, check_finite=False)
        self.chi2_const = float(
            np.sum(self.d_re**2 * self.w_re + self.d_im**2 * self.w_im))

    def set_regularization_scale(self, factor):
        self._cho = cho_factor(self.F + factor * self.H, lower=True,
                               check_finite=False)
        self.h_scale = float(factor)

    def _curv(self, P, Q):
        return P.real.T @ (self.w_re[:, None] * Q.real) + P.imag.T @ (
            self.w_im[:, None] * Q.imag)

    def _dvec(self, P):
        return P.real.T @ (self.w_re * self.d_re) + P.imag.T @ (
            self.w_im * self.d_im)

    def columns_for(self, positions, sigmas=None):
        sigmas = sigmas or [0.0] * len(positions)
        return np.column_stack([
            gaussian_visibilities(self.uv, y, x, s) if s > 0
            else point_visibilities(self.uv, y, x)
            for (y, x), s in zip(positions, sigmas)])

    def scan(self, positions, ys, xs, chunk=1024):
        if positions:
            P0 = self.columns_for(positions)
            A_eff = np.hstack([self.A, P0])
            B0 = self._curv(self.A, P0)
            M = np.block([[self.F + self.h_scale * self.H, B0],
                          [B0.T, self._curv(P0, P0)]])
            D_eff = np.concatenate([self.D, self._dvec(P0)])
        else:
            A_eff, M, D_eff = self.A, self.F + self.h_scale * self.H, self.D
        M = M + np.eye(M.shape[0]) * 1e-12 * max(np.trace(M), 1e-30)
        cho = cho_factor(M, lower=True, check_finite=False)
        MinvD = cho_solve(cho, D_eff, check_finite=False)
        amp, sig = np.empty(ys.size), np.empty(ys.size)
        for lo in range(0, ys.size, chunk):
            hi = min(lo + chunk, ys.size)
            P = np.column_stack([point_visibilities(self.uv, y, x)
                                 for y, x in zip(ys[lo:hi], xs[lo:hi])])
            B = self._curv(A_eff, P)
            c = (self.w_re @ P.real**2) + (self.w_im @ P.imag**2)
            r = self._dvec(P) - B.T @ MinvD
            MinvB = cho_solve(cho, B, check_finite=False)
            sch = np.maximum(c - np.einsum("ij,ij->j", B, MinvB), 1e-30)
            amp[lo:hi] = r / sch
            sig[lo:hi] = np.abs(r) / np.sqrt(sch)
        return amp, sig

    def solve(self, positions, sigmas=None):
        if not positions:
            s = cho_solve(self._cho, self.D, check_finite=False)
            return s, np.array([]), float(
                self.chi2_const + s @ self.F @ s - 2.0 * s @ self.D), np.zeros((0, 0))
        P = self.columns_for(positions, sigmas)
        B, C, Dp = self._curv(self.A, P), self._curv(P, P), self._dvec(P)
        MinvB = cho_solve(self._cho, B, check_finite=False)
        MinvD = cho_solve(self._cho, self.D, check_finite=False)
        S = C - B.T @ MinvB
        S_reg = S + np.eye(S.shape[0]) * 1e-10 * max(np.trace(S), 1e-30)
        a = np.linalg.solve(S_reg, Dp - B.T @ MinvD)
        s = MinvD - MinvB @ a
        theta = np.concatenate([s, a])
        F_aug = np.block([[self.F, B], [B.T, C]])
        chi2 = self.chi2_const + theta @ F_aug @ theta - 2.0 * theta @ np.concatenate([self.D, Dp])
        return s, a, float(chi2), np.linalg.inv(S_reg)


@pytest.fixture(scope="module")
def systems():
    uvd, _, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=400, mesh_n=24, compact_flux=0.012, compact_centre=(0.8, -0.7))
    uv, d, nz = uvd.flattened()
    ds = fitting.make_dataset(uv, d, nz, geom)
    fit = fitting.fit_dataset(ds, geom, reg_kind="matern", prior=PRIOR,
                              positive_only=False)
    inv = fit.fit.inversion
    dec, ra = comps["compact"]["centre"]
    positions = [sky_to_grid(-ra, dec), (0.31, -0.62), (-0.55, 0.23)]
    return AugmentedSystem(inv, ds), _Reference(inv, ds), ds, geom, positions


def _close(a, b, rtol=1e-11):
    """Equal to rounding: 1e-11 relative, or 1e-10 of the array's largest
    entry for elements near zero.  Mesh values and scan amplitudes come out
    of a subtraction (``MinvD - MinvB a``, ``Dp - B^T MinvD``), so an element
    1e4 times smaller than its neighbours legitimately carries their rounding:
    measured differences are 1e-16 absolute on a 4e-5 scale."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    if b.size:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-10 * np.max(np.abs(b)))


def test_stacked_columns_are_bitwise_the_analytic_columns(systems):
    """The broadcast build keeps `point_visibilities`' operation order, so
    the values are identical bit for bit, not just close."""
    system, ref, *_ , positions = systems
    P = system.columns_for(positions)
    assert np.array_equal(P, ref.columns_for(positions))
    G = system.columns_for(positions, [0.0, 0.08, 0.0])
    _close(G, ref.columns_for(positions, [0.0, 0.08, 0.0]), rtol=1e-15)
    assert np.array_equal(system.A, ref.A)


def test_solve_matches_the_reference_for_every_call_pattern(systems):
    """Cold, warm (cached columns), after a rescale, one column moved, and
    with a Gaussian width -- the paths `refine_position`,
    `retune_regularization` and `unresolved_test` exercise."""
    system, ref, *_, positions = systems
    for pos in ([], positions[:1], positions, positions):        # cold, warm
        for got, want in zip(system.solve(pos), ref.solve(pos)):
            _close(got, want)
    moved = [positions[0], (0.3123, -0.6201), positions[2]]
    for got, want in zip(system.solve(moved), ref.solve(moved)):
        _close(got, want)
    for factor in (37.0, 1.0):
        assert system.set_regularization_scale(factor)
        ref.set_regularization_scale(factor)
        for got, want in zip(system.solve(positions), ref.solve(positions)):
            _close(got, want)
    sig = [0.0, 0.0, 0.05]
    # chi^2 is chi2_const + theta F theta - 2 theta D, a cancellation of
    # O(N) terms, so 1e-11 relative is rounding (measured: 1.6e-12)
    assert system.chi_squared_with_width(positions, 2, 0.05) == pytest.approx(
        ref.solve(positions, sig)[2], rel=1e-11)
    # a widened column and its point version must not share a cache entry
    assert system.chi_squared(positions) == pytest.approx(
        ref.solve(positions)[2], rel=1e-12)


def test_scan_matches_the_reference_with_and_without_accepted_points(systems):
    system, ref, ds, geom, positions = systems
    ys, xs = detection_lattice(geom)
    for accepted in ([], positions[:1], positions[:2], []):
        # the []-lattice terms are memoised on the first call and reused by
        # every later one, so the repeat is a genuine test of the memo
        amp, sig = system.scan(accepted, ys, xs)
        amp_ref, sig_ref = ref.scan(accepted, ys, xs)
        _close(amp, amp_ref, rtol=1e-9)
        _close(sig, sig_ref, rtol=1e-9)
    assert system._lattice is not None and system._lattice["keep_Pw"]
    # an explicit small chunk gives the same answer as one big one
    amp, sig = system.scan(positions[:1], ys, xs, chunk=100)
    _close(amp, ref.scan(positions[:1], ys, xs)[0], rtol=1e-9)
    # ... and a rescaled prior invalidates nothing it should not
    assert system.set_regularization_scale(5.0)
    ref.set_regularization_scale(5.0)
    amp, sig = system.scan(positions[:1], ys, xs)
    _close(sig, ref.scan(positions[:1], ys, xs)[1], rtol=1e-9)
    system.set_regularization_scale(1.0)
    ref.set_regularization_scale(1.0)


def test_lattice_memo_is_dropped_when_it_would_not_fit(systems, monkeypatch):
    """Above the budget the scan streams like the original, same numbers."""
    system, ref, ds, geom, positions = systems
    ys, xs = detection_lattice(geom)
    # 48x48 lattice, 576 mesh pixels, 400 visibilities: the terms are 10.6 MB
    # and the weighted columns another 14.7 MB, so 16 MB keeps only the terms
    monkeypatch.setattr(pointsource, "SCAN_CHUNK_BYTES", 16 * 2**20)
    system._lattice = None
    amp, sig = system.scan(positions[:1], ys, xs)
    assert system._lattice is not None and not system._lattice["keep_Pw"]
    _close(amp, ref.scan(positions[:1], ys, xs)[0], rtol=1e-9)
    monkeypatch.setattr(pointsource, "SCAN_CHUNK_BYTES", 1024)
    system._lattice = None
    amp, sig = system.scan(positions[:1], ys, xs)
    assert system._lattice is None
    _close(amp, ref.scan(positions[:1], ys, xs)[0], rtol=1e-9)


def test_column_cache_is_bounded_and_evicts_the_least_recent(systems):
    system, *_ = systems
    system._columns.clear()
    cap = system._column_cache_max
    hot = (0.1, 0.2)
    for i in range(cap + 20):
        system.chi_squared([hot, (0.5 + 1e-4 * i, -0.3)])
    assert len(system._columns) == cap
    assert (0.1, 0.2, 0.0) in system._columns          # touched every call
    assert (0.5, -0.3, 0.0) not in system._columns     # the first transient


def test_model_visibilities_match_the_complex_product(systems):
    system, ref, *_, positions = systems
    s, a, _, _ = system.solve(positions[:2])
    want = ref.A @ s + ref.columns_for(positions[:2]) @ a
    _close(system.model_visibilities(s, positions[:2], a), want, rtol=1e-12)


def test_rejected_candidate_is_not_repicked_without_a_beam():
    """With radius 0 the strict `<` excluded nothing; `<=` excludes the point."""
    class Scan:
        def scan(self, accepted, ys, xs):
            return np.array([1.0, 1.0, -1.0]), np.array([9.0, 7.0, 20.0])

    ys, xs = np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])
    assert pointsource._best_candidate(Scan(), [], ys, xs, [], 0.0) == (0.0, 0.0)
    assert pointsource._best_candidate(
        Scan(), [], ys, xs, [(0.0, 0.0)], 0.0) == (1.0, 0.0)
    assert pointsource._best_candidate(
        Scan(), [], ys, xs, [(0.0, 0.0), (1.0, 0.0)], 0.0) is None
    # a refined position off the lattice excludes nothing at radius 0, as before
    assert pointsource._best_candidate(
        Scan(), [], ys, xs, [(0.01, 0.0)], 0.0) == (0.0, 0.0)


def test_dirty_imager_supplies_the_beam_when_no_fwhm_is_given(systems):
    """`dirty_imager=` used to be accepted and ignored."""
    from pyuvimage import beam as beam_mod

    system, ref, ds, geom, _ = systems
    imager = beam_mod.DirtyImager(ds)
    b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
    fwhm = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))
    inv = system.inversion
    with_fwhm = fit_point_sources(inv, ds, geom, beam_fwhm=fwhm, retune=False)
    with_imager = fit_point_sources(inv, ds, geom, dirty_imager=imager,
                                    retune=False)
    assert len(with_fwhm.points) == len(with_imager.points) == 1
    assert with_imager.points[0].flux == pytest.approx(
        with_fwhm.points[0].flux, rel=1e-10)
    assert with_imager.chi_squared == pytest.approx(with_fwhm.chi_squared,
                                                    rel=1e-12)

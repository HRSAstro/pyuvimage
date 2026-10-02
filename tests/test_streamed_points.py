"""Point components on the streamed path (`streamed_points`).

The streamed pass keeps no visibilities, so a point column's three terms --
A^T W P, P^T W P, P^T W d -- are read off the w-tilde kernel K and the data's
dirty image Dd, accumulated on grids `oversample` times finer than the image.
The in-memory `AugmentedSystem` evaluates the same sums visibility by
visibility, and is the definition of correct: everything here pins the
streamed system against it, and the grids against direct sums.
"""
import numpy as np
import pytest

import autogalaxy as ag

from pyuvimage import fitting, mock
from pyuvimage import streaming as stm
from pyuvimage.pointsource import (
    ARCSEC_TO_RAD,
    AugmentedSystem,
    SparseMesh,
    augmented_structure_ratio,
    detection_lattice,
    fit_point_sources,
)

needs_sparse = pytest.mark.skipif(
    fitting.sparse_inversion_diagnosis() is not None
    or not stm.nufftax_available(),
    reason="the sparse operator needs JAX, the fine grids nufftax",
)
pytestmark = needs_sparse

PRIOR = {"coefficient": 1e7, "scale": 0.25, "nu": 1.5}
Q = 8


@pytest.fixture(scope="module")
def case():
    """One mock (equal sigma_re and sigma_im, as the w-tilde path needs),
    fitted in memory and from its streamed terms with the same prior."""
    from pyuvimage.streamed_points import StreamedAugmentedSystem

    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=400, mesh_n=24, compact_flux=0.012, compact_centre=(0.8, -0.7))
    uv, d, nz = uvd.flattened()
    assert np.allclose(nz.real, nz.imag)
    ds = fitting.with_sparse_operator(
        fitting.make_dataset(uv, d, nz, geom), geometry=geom)
    fit = fitting.fit_dataset(
        ds, geom, reg_kind="matern", prior=PRIOR, positive_only=False)

    terms = stm.accumulate_sparse_terms(
        stm.iter_uvdata_chunks(uvd, 4096), geom, ds.real_space_mask,
        ag.TransformerDFT, point_oversample=Q)
    stub = stm.stub_dataset_from_terms(terms, geom, ds.real_space_mask)
    imager = fitting.imager_for(stub)
    sfit = fitting.fit_dataset(
        stub, geom, reg_kind="matern", prior=PRIOR, positive_only=False)
    return dict(
        uv=uv, d=d, w=1.0 / nz.real**2, geom=geom, terms=terms,
        mem=(fit.fit.inversion, ds), stub=stub, imager=imager,
        streamed=(sfit.fit.inversion, stub, terms, imager),
        system=StreamedAugmentedSystem(sfit.fit.inversion, stub, terms, imager),
        StreamedAugmentedSystem=StreamedAugmentedSystem,
    )


# positions off the grid, one within 0.27" of the 3" field's edge (where the
# circular Gaussian smoothing would wrap without the grids' margins)
POSITIONS = [(0.813, 0.721), (-0.31, 0.47), (0.123, -1.234)]
SIGMAS = [0.0, 0.07, 0.2]


def _direct(case, y, x, s, kind):
    uv, w, d = case["uv"], case["w"], case["d"]
    taper = np.exp(-2 * np.pi**2 * (s * ARCSEC_TO_RAD) ** 2 * (uv**2).sum(1))
    phase = 2 * np.pi * (uv[:, 0] * x + uv[:, 1] * y) * ARCSEC_TO_RAD
    if kind == "K":
        return np.sum(w * taper * np.cos(phase))
    return np.real(np.sum(w * d * taper * np.exp(1j * phase)))


def test_grids_reproduce_the_direct_sums(case):
    """K(lag) and Dd(position), widened or not, to ~1e-9 of their peak --
    including next to the field's edge, where an unpadded grid wrapped
    (3.5% on Dd there before the margins)."""
    from pyuvimage.streamed_points import PointGrids

    g = PointGrids(case["terms"])
    k0 = case["w"].sum()
    d_peak = max(abs(_direct(case, y, x, 0.0, "D")) for y, x in POSITIONS)
    for (y, x) in POSITIONS + [(1.4, -1.45)]:
        for s in (0.0, 0.07, 0.2, 0.3):
            assert g.kernel(y, x, s) == pytest.approx(
                _direct(case, y, x, s, "K"), abs=1e-8 * k0)
            # 0.3" in the corner is 6 sigma from the dirty grid's edge, and
            # its wrap shows at 7e-8; the unresolved test's widest here is 0.22"
            assert g.dirty(y, x, s) == pytest.approx(
                _direct(case, y, x, s, "D"), abs=(1e-7 if s > 0.25 else 1e-8) * d_peak)
    # a lag of most of a field, widened: needs the kernel's lag margin
    # (`streaming.POINT_KERNEL_PAD`, 0.75" here: 5.7 sigma beyond this lag)
    assert g.kernel(2.6, -2.5, 0.2) == pytest.approx(
        _direct(case, 2.6, -2.5, 0.2, "K"), abs=1e-8 * k0)


def test_no_grids_no_system(case):
    from pyuvimage.streamed_points import PointGrids

    plain = stm.accumulate_sparse_terms(
        stm.iter_uvdata_chunks(mock.make_extended_plus_compact_dataset(
            n_vis=50, mesh_n=6)[0], 4096),
        case["geom"], case["stub"].real_space_mask, ag.TransformerDFT)
    with pytest.raises(ValueError, match="point_oversample"):
        PointGrids(plain)


def test_column_terms_match_the_in_memory_system(case):
    mem = AugmentedSystem(*case["mem"])
    st = case["system"]
    assert st.chi2_const == pytest.approx(mem.chi2_const, rel=1e-12)
    np.testing.assert_allclose(st.F, mem.F, rtol=0, atol=1e-12 * np.abs(mem.F).max())
    np.testing.assert_allclose(st.D, mem.D, rtol=0, atol=1e-12 * np.abs(mem.D).max())
    P, Bm, Dm = mem._column_terms(POSITIONS, SIGMAS)
    Cm = mem._point_gram(POSITIONS, SIGMAS, P)
    none, Bs, Ds = st._column_terms(POSITIONS, SIGMAS)
    assert none is None
    np.testing.assert_allclose(Bs, Bm, rtol=0, atol=1e-8 * np.abs(Bm).max())
    np.testing.assert_allclose(Ds, Dm, rtol=0, atol=1e-8 * np.abs(Dm).max())
    np.testing.assert_allclose(st._point_gram(POSITIONS, SIGMAS), Cm,
                               rtol=0, atol=1e-8 * case["w"].sum())


def test_solve_scan_and_structure_match_the_in_memory_system(case):
    mem = AugmentedSystem(*case["mem"])
    st = case["system"]
    for positions in ([], POSITIONS[:1], POSITIONS[:2]):
        sm, am, cm, covm = mem.solve(positions)
        ss, as_, cs, covs = st.solve(positions)
        assert cs == pytest.approx(cm, abs=1e-6)        # chi^2 ~ 470: delta 1e-6
        np.testing.assert_allclose(as_, am, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(ss, sm, rtol=0, atol=1e-6 * np.abs(sm).max())
    ys, xs = detection_lattice(case["geom"])
    for accepted in ([], POSITIONS[:1]):
        am, sm = mem.scan(accepted, ys, xs)
        as_, ss = st.scan(accepted, ys, xs)
        np.testing.assert_allclose(as_, am, rtol=0, atol=1e-7 * np.abs(am).max())
        np.testing.assert_allclose(ss, sm, rtol=0, atol=1e-7 * sm.max())
    rm = augmented_structure_ratio(mem, POSITIONS[:1], fitting.imager_for(case["mem"][1]))
    rs = augmented_structure_ratio(st, POSITIONS[:1], case["imager"])
    assert rs == pytest.approx(rm, rel=1e-9)


def test_visibility_space_is_refused_not_faked(case):
    """The stub's eight visibilities are not the data; nothing may read them."""
    st = case["system"]
    with pytest.raises(NotImplementedError):
        st.model_visibilities(np.zeros(st.n_mesh), [], [])
    with pytest.raises(NotImplementedError):
        st.columns_for(POSITIONS)
    with pytest.raises(AttributeError):
        st.d_re


@pytest.mark.parametrize("positions", [None, [(0.7, 0.8)]])   # (dRA, dDec)
def test_fit_point_sources_is_the_same_fit(case, positions):
    """Detection, refinement, the unresolved test (widened columns), the
    structure retune and the systematic error: the same answer either way."""
    beam = 0.35
    mem = fit_point_sources(
        case["mem"][0], case["mem"][1], case["geom"], positions=positions,
        beam_fwhm=beam, retune_criterion="structure",
        dirty_imager=fitting.imager_for(case["mem"][1]))
    inv, stub, terms, imager = case["streamed"]
    st = fit_point_sources(
        inv, stub, case["geom"], positions=positions, beam_fwhm=beam,
        retune_criterion="structure", dirty_imager=imager,
        system=case["StreamedAugmentedSystem"](inv, stub, terms, imager))
    # The two systems' chi^2 surfaces agree to ~1e-7, and Nelder-Mead stops
    # within its own tolerance (step/20 = 1.6 mas here) of the minimum, so
    # positions agree to well inside that, not to rounding; likewise the
    # retune, which bisects the structure ratio to a 1% bracket
    assert len(st.points) == len(mem.points) >= 1
    for a, b in zip(st.points, mem.points):
        assert a.d_ra == pytest.approx(b.d_ra, abs=1e-4)
        assert a.d_dec == pytest.approx(b.d_dec, abs=1e-4)
        assert a.flux == pytest.approx(b.flux, rel=1e-4)
        assert a.flux_error == pytest.approx(b.flux_error, rel=1e-3)
    assert st.regularization_factor == pytest.approx(mem.regularization_factor, rel=1e-3)
    assert st.chi_squared == pytest.approx(mem.chi_squared, rel=1e-5)


def test_cross_on_grid_batches_without_changing_the_answer(case, monkeypatch):
    """`(W~ M)^T` is built a batch of columns at a time (all at once was 5 GB
    of FFT transients on Teresa's grid); the batching changes nothing."""
    from pyuvimage import pointsource

    inv, ds = case["mem"]
    ys, xs = detection_lattice(case["geom"])
    whole = SparseMesh(inv, ds).cross_on_grid(ys, xs)
    monkeypatch.setattr(pointsource, "SCAN_CHUNK_BYTES", 1)   # one column per batch
    batched = SparseMesh(inv, ds).cross_on_grid(ys, xs)
    # bit-identical with pocketfft on Linux, but not guaranteed: an FFT
    # library may vectorise a different batch size differently (it does on
    # Hannah's macOS build), so rounding is the honest bound
    tol = 1e-13 * np.abs(whole).max()
    np.testing.assert_allclose(batched, whole, rtol=0, atol=tol)
    # and a strict subset of the grid is the matching columns
    sub = SparseMesh(inv, ds).cross_on_grid(ys[::7], xs[::7])
    np.testing.assert_allclose(sub, whole[:, ::7], rtol=0, atol=tol)


def test_streamed_run_with_points_matches_the_in_memory_run(tmp_path):
    """`api.run` end to end: MFS points now stream rather than holding the
    visibilities, and every product agrees with the in-memory sparse run."""
    from pyuvimage import api

    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=400, mesh_n=24, compact_flux=0.012, compact_centre=(0.8, -0.7))
    kw = dict(fov=3.0, reg="matern", criterion="structure", inversion="sparse",
              pb_correction=False, write=False, uncertainty_map=False,
              point_sources=True)
    ref = api.run(uvd, out=tmp_path / "mem", **kw)
    st = api.run(uvd, out=tmp_path / "str", streaming=True, chunk_k=4096,
                 kernel_cache=str(tmp_path), **kw)
    r, s = ref.products[0], st.products[0]
    assert len(s.points) == len(r.points) == 1
    assert s.points[0].flux == pytest.approx(r.points[0].flux, rel=1e-6)
    assert s.points[0].d_ra == pytest.approx(r.points[0].d_ra, abs=1e-6)
    assert s.chi_squared == pytest.approx(r.chi_squared, rel=1e-6)
    np.testing.assert_allclose(s.model_image, r.model_image, rtol=0,
                               atol=1e-6 * np.abs(r.model_image).max())
    # the residual map includes the point's own dirty image, off the grids
    np.testing.assert_allclose(s.residual_sigma, r.residual_sigma, rtol=0, atol=1e-5)
    np.testing.assert_allclose(s.reconvolved, r.reconvolved, rtol=0,
                               atol=1e-6 * np.abs(r.reconvolved).max())
    assert st.parameters["point_sources"]["n_points"] == 1
    assert st.parameters["streaming"]["n_visibilities_streamed"] == 400


def test_auto_streams_mfs_points_and_holds_cube_points(monkeypatch, caplog):
    from pyuvimage import api

    monkeypatch.setattr(stm, "scan_header", lambda *a, **k: type(
        "H", (), {"n_samples": 10**6})())
    stream, _ = api.resolve_streaming("auto", "some/file.npz", point_sources=True)
    assert stream
    stream, _ = api.resolve_streaming("auto", "some/file.npz", mode="cube",
                                      point_sources=True)
    assert not stream
    monkeypatch.setattr(stm, "nufftax_available", lambda: False)
    stream, _ = api.resolve_streaming("auto", "some/file.npz", point_sources=True)
    assert not stream

"""Point components on the sparse (w-tilde) inversion.

Until this landed, `--point-sources` forced `--inversion dense`: the bordered
system's first act was `inversion.operated_mapping_matrix`, the dense
`n_vis x n_mesh` build the w-tilde path exists to avoid (21.6 GB on Ruby
CO(7-6) against a 0.10 MB kernel), so asking for both gave up the whole
benefit.

`pointsource.SparseMesh` removes the need for it. `A = F M`, so the two
operations the bordered system actually wants are

    A^T Pw = M^T Re(F^H (w_re Re(P) + i w_im Im(P)))      (the cross-terms)
    A s    = F (M s)                                       (model visibilities)

-- one adjoint transform per point column and one forward transform per model,
never the matrix. Everything here pins that against the dense backend, which
is the definition of correct: the two must agree to rounding, not merely
closely.
"""
import numpy as np
import pytest

from pyuvimage import fitting, mock
from pyuvimage.pointsource import (
    AugmentedSystem,
    DenseMesh,
    SparseMesh,
    detection_lattice,
    fit_point_sources,
    mesh_operator,
)

# The same prior `tests/test_pointsource.py` detects this knot under: weak
# regularisation lets the mesh absorb a compact source, and then there is
# nothing left for the detector to find on either path.
PRIOR = {"coefficient": 1e7, "scale": 0.25, "nu": 1.5}

sparse_only = pytest.mark.skipif(
    fitting.sparse_inversion_diagnosis() is not None,
    reason=fitting.sparse_inversion_diagnosis() or "",
)


@pytest.fixture(scope="module")
def case():
    """One mock, fitted twice: dense and sparse, same everything else."""
    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=400, mesh_n=24, compact_flux=0.012, compact_centre=(0.8, -0.7))
    uv, d, nz = uvd.flattened()

    dense_ds = fitting.make_dataset(uv, d, nz, geom)
    dense_fit = fitting.fit_dataset(
        dense_ds, geom, reg_kind="matern", prior=PRIOR, positive_only=False)

    sparse_ds = fitting.with_sparse_operator(
        fitting.make_dataset(uv, d, nz, geom), geometry=geom)
    sparse_fit = fitting.fit_dataset(
        sparse_ds, geom, reg_kind="matern", prior=PRIOR, positive_only=False)

    return dict(
        geom=geom, comps=comps,
        dense=(dense_fit.fit.inversion, dense_ds),
        sparse=(sparse_fit.fit.inversion, sparse_ds),
    )


@pytest.fixture(scope="module")
def systems(case):
    """The dense system, and the same system with only the backend swapped.

    Comparing two separately-built inversions would confound two things: the
    backend, and the sparse inversion's own (tiny) disagreement with the dense
    one in `F` and `D`. Swapping just the mesh operator holds `F`, `H` and `D`
    fixed, so any difference is the backend's alone.
    """
    dense = AugmentedSystem(*case["dense"])
    hybrid = AugmentedSystem(*case["dense"])
    hybrid.mesh = SparseMesh(*case["sparse"])
    return dense, hybrid


def _beam(dataset, geometry) -> float:
    from pyuvimage.beam import DirtyImager, fit_beam

    b = fit_beam(DirtyImager(dataset).dirty_beam, geometry.pixel_scale)
    return float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))


# --- the backend is chosen by the dataset, not by a flag --------------------

def test_the_dataset_picks_the_backend(case):
    assert isinstance(mesh_operator(*case["dense"]), DenseMesh)
    assert isinstance(mesh_operator(*case["sparse"]), SparseMesh)


@sparse_only
def test_the_sparse_backend_refuses_to_form_the_matrix(case):
    """The refusal is the feature: silently building it would allocate exactly
    what --inversion sparse was chosen to avoid."""
    sparse = AugmentedSystem(*case["sparse"])
    with pytest.raises(NotImplementedError, match="never forms"):
        sparse.A
    with pytest.raises(NotImplementedError):
        sparse.A_stack


# --- the two operations, against the dense reference ------------------------

@sparse_only
def test_the_cross_terms_match_the_dense_matrix(systems):
    """A^T W P, the term that needed the dense build."""
    dense, sparse = systems
    positions = [(0.8, -0.7), (-0.35, 0.22), (0.013, 1.07)]  # all sub-pixel
    P = dense._stacked_columns(
        [p[0] for p in positions], [p[1] for p in positions])
    Pw = dense.w_stack[:, None] * P

    B_dense = dense.mesh.cross(Pw)
    B_sparse = sparse.mesh.cross(Pw)
    assert np.allclose(B_sparse, B_dense, rtol=0, atol=1e-15 * np.max(np.abs(B_dense)))


@sparse_only
def test_the_forward_direction_matches(systems, case):
    """A s, which `model_visibilities` needs to write model visibilities."""
    dense, sparse = systems
    rng = np.random.default_rng(3)
    s = rng.normal(size=dense.n_mesh)
    vis_dense = dense.mesh.forward(s)
    vis_sparse = sparse.mesh.forward(s)
    assert np.allclose(
        vis_sparse, vis_dense, rtol=0,
        atol=1e-12 * np.max(np.abs(vis_dense)))


@sparse_only
def test_a_gaussian_column_matches_too(systems):
    """The widened column the unresolved test uses goes down the same path."""
    dense, sparse = systems
    P = dense._stacked_columns([0.8], [-0.7], [0.05])
    Pw = dense.w_stack[:, None] * P
    B_dense, B_sparse = dense.mesh.cross(Pw), sparse.mesh.cross(Pw)
    assert np.allclose(
        B_sparse, B_dense, rtol=0, atol=1e-15 * np.max(np.abs(B_dense)))


# --- and the whole solve ----------------------------------------------------

@sparse_only
def test_the_augmented_solve_agrees(systems):
    """Mesh values, amplitudes, chi^2 and the amplitude covariance."""
    dense, sparse = systems
    positions = [(0.8, -0.7)]
    s_d, a_d, chi2_d, cov_d = dense.solve(positions)
    s_s, a_s, chi2_s, cov_s = sparse.solve(positions)
    # `F + H` is ill-conditioned by construction (2.3e10 here: the mesh has
    # pixels the uv coverage barely constrains), so a 1e-16 difference in the
    # cross-terms reaches the amplitudes at ~1e-6. That is the arithmetic, not
    # the backend -- `test_the_difference_is_conditioning_not_error` pins it.
    assert np.allclose(s_s, s_d, rtol=1e-4, atol=1e-6 * np.max(np.abs(s_d)))
    assert np.allclose(a_s, a_d, rtol=1e-4)
    assert chi2_s == pytest.approx(chi2_d, rel=1e-8)
    assert np.allclose(cov_s, cov_d, rtol=1e-4)


@sparse_only
def test_two_points_agree(systems):
    dense, sparse = systems
    positions = [(0.8, -0.7), (-0.4, 0.3)]
    for i, (d, s) in enumerate(zip(dense.solve(positions)[:3],
                                   sparse.solve(positions)[:3])):
        d, s = np.atleast_1d(d), np.atleast_1d(s)
        assert np.allclose(
            s, d, rtol=1e-4, atol=1e-6 * np.max(np.abs(d))), f"element {i}"


@sparse_only
def test_the_detection_scan_agrees(systems, case):
    """The matched filter over the whole lattice -- the detector itself.

    This is the expensive one: on the dense path it is a single GEMM against
    the whole of `A`; here it is one adjoint transform per trial position.
    The answer must not depend on which.
    """
    dense, sparse = systems
    ys, xs = detection_lattice(case["geom"])
    amp_d, sig_d = dense.scan([], ys, xs)
    amp_s, sig_s = sparse.scan([], ys, xs)
    assert np.allclose(amp_s, amp_d, rtol=1e-4, atol=1e-6 * np.max(np.abs(amp_d)))
    assert np.allclose(sig_s, sig_d, rtol=1e-4, atol=1e-6 * np.max(np.abs(sig_d)))
    # and, the thing that actually matters, they pick the same pixel
    assert np.argmax(sig_s) == np.argmax(sig_d)


@sparse_only
def test_the_scan_with_a_point_already_accepted_agrees(systems, case):
    dense, sparse = systems
    ys, xs = detection_lattice(case["geom"])
    amp_d, sig_d = dense.scan([(0.8, -0.7)], ys, xs)
    amp_s, sig_s = sparse.scan([(0.8, -0.7)], ys, xs)
    assert np.allclose(sig_s, sig_d, rtol=1e-4, atol=1e-6 * np.max(np.abs(sig_d)))
    assert np.argmax(sig_s) == np.argmax(sig_d)


@sparse_only
def test_model_visibilities_agree(systems):
    dense, sparse = systems
    positions = [(0.8, -0.7)]
    s_d, a_d, _, _ = dense.solve(positions)
    v_d = dense.model_visibilities(s_d, positions, a_d)
    v_s = sparse.model_visibilities(s_d, positions, a_d)
    assert np.allclose(v_s, v_d, rtol=0, atol=1e-12 * np.max(np.abs(v_d)))


@sparse_only
def test_the_difference_is_conditioning_not_error(systems, case):
    """The claim the tolerances above rest on.

    The cross-terms agree to ~5e-16 relative, yet the amplitude moves by ~5e-7,
    because `F + H` has a condition number of ~1e10. If that is arithmetic
    rather than a defect in the backend, then perturbing the *dense*
    cross-terms randomly by the same 5e-16 must move the amplitude at least as
    far. It does -- by roughly ten times as far, which is the ordinary
    behaviour of a random direction against a structured one.
    """
    dense, sparse = systems
    positions = [(0.8, -0.7)]
    P = dense._stacked_columns([0.8], [-0.7])
    Pw = dense.w_stack[:, None] * P
    B_d, B_s = dense.mesh.cross(Pw), sparse.mesh.cross(Pw)
    scale = np.max(np.abs(B_s - B_d)) / np.max(np.abs(B_d))
    assert scale < 1e-14, "the cross-terms themselves must agree to rounding"

    a_ref = dense.solve(positions)[1]
    moved_sparse = abs(sparse.solve(positions)[1][0] - a_ref[0]) / abs(a_ref[0])

    rng = np.random.default_rng(0)
    inversion, dataset = case["dense"]
    moved_random = []
    for _ in range(3):
        class Jittered:
            n_vis, n_mesh = dense.mesh.n_vis, dense.mesh.n_mesh

            def cross(self, Pw):
                out = dense.mesh.cross(Pw)
                return out + rng.normal(
                    0.0, scale * np.max(np.abs(out)), out.shape)

            def forward(self, values):
                return dense.mesh.forward(values)

        jittered = AugmentedSystem(inversion, dataset)
        jittered.mesh = Jittered()
        moved_random.append(
            abs(jittered.solve(positions)[1][0] - a_ref[0]) / abs(a_ref[0]))

    assert moved_sparse <= max(moved_random), (
        f"the sparse backend moved the amplitude by {moved_sparse:.2e}, more "
        f"than rounding noise of the same size does ({max(moved_random):.2e})"
    )


# --- end to end -------------------------------------------------------------

@sparse_only
def test_the_whole_detection_finds_the_same_source(case):
    """`fit_point_sources` from end to end on both paths: same position, same
    flux, same error bar."""
    geom = case["geom"]
    out = {}
    for name in ("dense", "sparse"):
        inversion, ds = case[name]
        out[name] = fit_point_sources(
            inversion, ds, geom, beam_fwhm=_beam(ds, geom), retune=False,
        )
    d, s = out["dense"], out["sparse"]
    assert len(s.points) == len(d.points) >= 1
    for pd, ps in zip(d.points, s.points):
        assert ps.flux == pytest.approx(pd.flux, rel=1e-3)
        assert ps.flux_error == pytest.approx(pd.flux_error, rel=1e-3)
        assert ps.d_ra == pytest.approx(pd.d_ra, abs=1e-3)
        assert ps.d_dec == pytest.approx(pd.d_dec, abs=1e-3)
    assert s.chi_squared == pytest.approx(d.chi_squared, rel=1e-6)


@sparse_only
def test_the_recovered_flux_is_near_the_truth(case):
    """Not a parity check -- the sparse path must actually work."""
    inversion, ds = case["sparse"]
    solution = fit_point_sources(
        inversion, ds, case["geom"], beam_fwhm=_beam(ds, case["geom"]))
    truth = case["comps"]["compact"]
    assert solution.points, "no point detected on the sparse path"
    best = max(solution.points, key=lambda p: p.significance)
    dec, ra = truth["centre"]
    assert best.flux == pytest.approx(truth["flux"], rel=0.25)
    assert abs(best.d_ra - (-ra)) < 0.1 and abs(best.d_dec - dec) < 0.1
    assert best.significance > 5.0


# --- the detector's on-grid shortcut ----------------------------------------

@sparse_only
def test_the_on_grid_shortcut_matches_the_exact_columns(case):
    """`cross_on_grid` is the detector's whole-lattice shortcut. It must give
    the same numbers as the per-column route it replaces -- that route is the
    one already pinned against the dense matrix above, so this closes the
    chain."""
    system = AugmentedSystem(*case["sparse"])
    ys, xs = detection_lattice(case["geom"])
    fast = system.mesh.cross_on_grid(ys, xs)
    assert fast is not None, "the detection lattice must be recognised as on-grid"

    take = np.linspace(0, ys.size - 1, 12).astype(int)
    P = system._stacked_columns(ys[take], xs[take])
    exact = system.mesh.cross(system.w_stack[:, None] * P)
    assert np.allclose(
        fast[:, take], exact, rtol=0, atol=1e-12 * np.max(np.abs(exact)))


@sparse_only
def test_a_lattice_off_the_grid_refuses_the_shortcut(case):
    """Half a pixel off, the shortcut is wrong and must decline rather than
    quietly answer for the nearest pixel -- a point's whole reason for
    existing is that it is not on the grid."""
    system = AugmentedSystem(*case["sparse"])
    ys, xs = detection_lattice(case["geom"])
    half = 0.5 * case["geom"].pixel_scale
    assert system.mesh.cross_on_grid(ys + half, xs) is None
    assert system.mesh.cross_on_grid(ys, xs + 0.3 * case["geom"].pixel_scale) is None


@sparse_only
def test_neighbouring_pixels_get_different_columns(case):
    """An even-sided grid centres its pixels on half-integer multiples of the
    pixel scale, so a lookup keyed on whole pixels collides neighbours into
    one entry. It did, once."""
    system = AugmentedSystem(*case["sparse"])
    ys, xs = detection_lattice(case["geom"])
    index = system.mesh._slim_index_for(ys, xs)
    assert index is not None
    assert len(np.unique(index)) == index.size


@sparse_only
def test_the_dense_backend_has_no_shortcut_and_needs_none(case):
    """It is a `SparseMesh` method only; `scan` asks with `getattr`."""
    assert not hasattr(DenseMesh, "cross_on_grid")

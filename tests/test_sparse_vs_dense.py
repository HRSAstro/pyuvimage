"""The sparse inversion must give the same answer as the dense one.

This is the test that should have existed before `--inversion auto` started
preferring sparse on every real dataset. It does not exist yet in any
meaningful form: the one earlier attempt ran at a fixed coefficient of 1e8,
above the top of the search range, which nulls the model on **both** paths --
it compared two near-zero reconstructions and reported "identical to eight
significant figures". That claim is withdrawn.

The rules it teaches, all enforced below:

* fit at a coefficient that visibly moves the model, and assert that it does
  before comparing anything;
* compare the reconstructions, not just chi^2 -- a pure scale error is
  invisible in chi^2 and was exactly how the adjoint-scaling bug hid;
* score both against the known truth, because two paths can agree and both be
  wrong.

Everything here needs JAX and an autoarray with the w-tilde path, so it skips
where those are missing -- which includes the container these were written in.
It has therefore never run. Treat a first green run on a JAX machine as the
result, not as a regression check.
"""

import numpy as np
import pytest

from pyuvimage import fitting
from pyuvimage.mock import make_sparse_test_dataset

sparse_only = pytest.mark.skipif(
    fitting.sparse_inversion_diagnosis() is not None,
    reason=fitting.sparse_inversion_diagnosis() or "",
)

#: Where the two paths must agree. They compute the same matrix by different
#: routes -- one dense accumulation, one FFT convolution -- so they will not be
#: bit-identical; but a real disagreement (a scale error, a missing term) is
#: order 1, four decades above this.
AGREEMENT = 1e-3


@pytest.fixture(scope="module")
def problem():
    """A mock above the sparse threshold, with the truth kept.

    No point source: `--inversion sparse` refuses those today, so including
    one would test the dense path twice.

    The mock's **own** geometry is used rather than one re-resolved from the
    data. Two reasons. The truth is built on that mesh, so scoring against it
    needs the same grid -- re-resolving gave a 20x20 fit against a 24x24
    truth, and the truth comparison silently skipped, losing the one check
    that catches both paths being wrong together. And it removes the
    auto-resolution as a variable: this is a controlled comparison of two ways
    to compute one matrix, so everything else should be pinned.
    """
    uvd, truth, geom, _ = make_sparse_test_dataset(n_vis=8000, point_flux_jy=0.0)
    uv, d, n = uvd.flattened()
    assert len(d) >= fitting.SPARSE_AUTO_MIN_VISIBILITIES, (
        "the point of this dataset is that `auto` takes the sparse path on it"
    )
    truth = np.asarray(truth, dtype=float).ravel()
    assert truth.size == geom.mesh_shape[0] * geom.mesh_shape[1]
    return uv, d, n, geom, truth


def _fit(uv, d, n, geom, inversion, coefficient):
    transformer_cls = fitting.resolve_transformer(
        n_vis=len(d), transformer="auto",
        n_image_pixels=int(np.prod(geom.shape_native)),
        n_mesh_pixels=geom.mesh_shape[0] * geom.mesh_shape[1],
    )
    ds = fitting.make_dataset(
        uv, d, n, geom, transformer_cls, mask_shape="square")
    if inversion == "sparse":
        ds = fitting.with_sparse_operator(ds, uv, n, geom, cache_dir=None)
    return fitting.fit_at(
        ds, geom.mesh_shape, "matern", coefficient,
        positive_only=False, reg_scale=0.5, nu=1.5,
    )


#: A coefficient where the prior clearly bites but the model is still a model.
#: Measured on this mock (20x20 mesh, 8000 visibilities):
#:
#:     lambda   moves model   peak vs unregularised   error vs truth
#:     1e1        12.0%            0.906                  1.447
#:     1e2        56.8%            0.557                  0.965
#:     1e3        90.6%            0.345                  0.771
#:     1e4        96.7%            0.362                  0.749
#:
#: 1e3 is the choice: unmistakably regularised, peak still a third of the
#: unregularised solve, and far from the smoothed-to-nothing regime where two
#: paths agree for free. `test_the_coefficient_actually_moves_the_model` fails
#: first if a code change moves that range.
COEFFICIENT = 1e3


@sparse_only
def test_the_coefficient_actually_moves_the_model(problem):
    """Guard against the failure that made the last comparison worthless.

    If the reconstruction at COEFFICIENT is indistinguishable from the
    unregularised one, or has collapsed to nothing, then any agreement between
    the two paths below is agreement between two models that say nothing.
    """
    uv, d, n, geom, _ = problem
    weak = np.asarray(_fit(uv, d, n, geom, "dense", 1e-6).inversion.reconstruction)
    used = np.asarray(
        _fit(uv, d, n, geom, "dense", COEFFICIENT).inversion.reconstruction)

    change = np.linalg.norm(used - weak) / np.linalg.norm(weak)
    assert change > 0.02, (
        f"the prior barely moves the model at {COEFFICIENT:g} (by {change:.1%}); "
        "comparing the paths here would prove nothing"
    )
    assert np.abs(used).max() > 0.05 * np.abs(weak).max(), (
        "the model has been smoothed away to nothing at this coefficient"
    )


@sparse_only
def test_the_two_inversions_agree(problem):
    """The claim `--inversion auto` rests on."""
    uv, d, n, geom, _ = problem
    dense = np.asarray(
        _fit(uv, d, n, geom, "dense", COEFFICIENT).inversion.reconstruction)
    sparse = np.asarray(
        _fit(uv, d, n, geom, "sparse", COEFFICIENT).inversion.reconstruction)

    peak = max(np.abs(dense).max(), np.abs(sparse).max())
    assert peak > 0, "both reconstructions are zero; this is not agreement"
    assert np.abs(dense - sparse).max() / peak < AGREEMENT


@sparse_only
def test_the_two_inversions_agree_on_scale(problem):
    """Separated from the difference test on purpose.

    The adjoint-scaling bug made the sparse model a factor 10816 too small
    while leaving its morphology perfect. A difference norm catches that, but
    a ratio names it -- and names it in a way that points at the cause.
    """
    uv, d, n, geom, _ = problem
    dense = np.asarray(
        _fit(uv, d, n, geom, "dense", COEFFICIENT).inversion.reconstruction)
    sparse = np.asarray(
        _fit(uv, d, n, geom, "sparse", COEFFICIENT).inversion.reconstruction)

    assert np.abs(dense).max() / np.abs(sparse).max() == pytest.approx(1.0, abs=1e-3)
    assert dense.sum() / sparse.sum() == pytest.approx(1.0, abs=1e-3)


@sparse_only
def test_chi_squared_agrees_well_within_its_own_noise(problem):
    """chi^2 is a weak test -- it barely moves with the model on
    well-constrained data -- so it gets a loose bound and no more weight than
    that. Stated in units of sigma(chi^2) = sqrt(2N), which is the only scale
    on which a chi^2 difference means anything."""
    uv, d, n, geom, _ = problem
    n_data = 2 * len(d)
    a = fitting._chi_squared(_fit(uv, d, n, geom, "dense", COEFFICIENT))
    c = fitting._chi_squared(_fit(uv, d, n, geom, "sparse", COEFFICIENT))
    assert abs(a - c) / np.sqrt(2.0 * n_data) < 0.01


@sparse_only
def test_both_paths_sit_the_same_distance_from_the_truth(problem):
    """Two paths can agree and both be wrong, so check what they agree *on*.

    Note what this deliberately does not assert. The mock recovers only ~26%
    of the disc's flux, identically at every coefficient, because
    `random_uv_coverage` draws baselines from a Beta(1.5, 2.5) with no weight
    near zero -- there are no short baselines, so the extended flux is
    genuinely resolved out, exactly as it would be on a real array. That is a
    property of the coverage, not of the inversion, and an absolute
    flux-recovery assertion here would be testing the mock.

    What is meaningful: both paths must land the *same* distance from the sky
    (they compute the same matrix, so any gap is a bug), and both must be
    closer to it than an empty model is.
    """
    uv, d, n, geom, truth = problem
    errors = {}
    for inversion in ("dense", "sparse"):
        model = np.asarray(
            _fit(uv, d, n, geom, inversion, COEFFICIENT).inversion.reconstruction)
        errors[inversion] = (
            np.linalg.norm(model - truth) / np.linalg.norm(truth)
        )
    # better than reconstructing nothing at all, which scores exactly 1.0
    for inversion, error in errors.items():
        assert error < 0.9, (
            f"{inversion} is no closer to the sky than an empty model "
            f"(relative error {error:.3f})"
        )
    assert errors["dense"] == pytest.approx(errors["sparse"], abs=1e-3)


def test_the_mock_is_above_the_auto_threshold():
    """Runs without JAX: the dataset's whole purpose is to trip `auto`."""
    uvd, _, _, _ = make_sparse_test_dataset()
    assert uvd.n_samples >= fitting.SPARSE_AUTO_MIN_VISIBILITIES
    assert fitting.resolve_inversion(
        "auto", n_vis=uvd.n_samples,
        noise=np.full(8, 0.1 + 0.1j),
    ) in ("sparse", "dense")  # sparse where JAX exists, dense where it does not


def test_the_mock_carries_a_point_source_when_asked():
    """For when point components reach the sparse path -- the block methods
    exist upstream now. Until then `auto` sends any run with points to dense."""
    _, _, _, comps = make_sparse_test_dataset(point_flux_jy=0.004)
    assert comps["points"] and comps["points"][0]["flux"] == pytest.approx(0.004)
    assert fitting.resolve_inversion(
        "auto", n_vis=8000, point_sources=True) == "dense"

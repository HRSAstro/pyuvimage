"""The coefficient-invariant linear system behind the hyperparameter search.

`fitting.fit_dataset` used to build a fresh `ag.FitInterferometer` for every
trial of the search, and autoarray rebuilt the transformed mapping matrix for
every fit and F and D on every access: 43.6 s for a search whose linear
algebra took a third of a second. F and D are now read off one framework
inversion and every trial is a solve on them. These tests pin two things: that
a trial *is* the framework fit at that prior -- bitwise, not approximately --
and that nothing downstream goes back to autoarray for matrices it already
has.
"""

import logging

import numpy as np
import pytest

import autogalaxy as ag

from pyuvimage import fitting, mock
from pyuvimage.fitting import PriorScan, Trial


@pytest.fixture(scope="module")
def small():
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=200, mesh_n=10, seed=3)
    uv, d, n = uvd.flattened()
    return fitting.make_dataset(uv, d, n, geom, transformer="dft"), geom


@pytest.fixture(scope="module")
def system(small):
    dataset, geom = small
    return fitting.build_linear_system(dataset, geom.mesh_shape)


def _framework_evidence(fit):
    try:
        return float(fit.figure_of_merit)
    except np.linalg.LinAlgError:
        return -np.inf


# --- a trial is the framework fit -------------------------------------------

@pytest.mark.parametrize("kind, coefficient, positive, kwargs", [
    ("matern", 1e2, False, dict(reg_scale=0.5)),
    ("matern", 1e2, True, dict(reg_scale=0.5)),
    ("constant", 1e4, True, {}),
    ("gaussian", 1e3, False, dict(reg_scale=0.3, envelope={"fwhm": 1.0})),
    ("exponential", 10.0, False, dict(reg_scale=0.4)),
])
def test_a_trial_reproduces_the_framework_fit_bitwise(
    small, system, kind, coefficient, positive, kwargs
):
    """Same F, same D, same solver functions: the reconstruction, chi^2 and
    evidence must be the framework's to the last bit, or `SingleFit` could not
    report them without a second pass over the inversion."""
    dataset, geom = small
    fit = fitting.fit_at(
        dataset, geom.mesh_shape, kind, coefficient, positive_only=positive,
        **kwargs,
    )
    reg = fitting.make_regularization(
        kind, coefficient, kwargs.get("reg_scale"), 1.5, kwargs.get("envelope")
    )
    trial = system.trial(reg, positive=positive)
    assert np.array_equal(trial.reconstruction, np.asarray(fit.inversion.reconstruction))
    assert trial.chi_squared == float(fit.inversion.fast_chi_squared)
    assert trial.log_evidence == _framework_evidence(fit)


def test_positivity_zeroes_the_edge_as_the_framework_does(small, system):
    """Under `use_edge_zeroed_pixels` autoarray solves the non-negative system
    on the interior pixels only; a constrained trial must carry the same zero
    border or it is not the same model."""
    dataset, geom = small
    reg = fitting.make_regularization("matern", 10.0, 0.5, 1.5, None)
    constrained = system.trial(reg, positive=True).reconstruction.reshape(geom.mesh_shape)
    free = system.trial(reg, positive=False).reconstruction.reshape(geom.mesh_shape)
    assert np.all(constrained[0, :] == 0) and np.all(constrained[:, -1] == 0)
    assert np.any(free[0, :] != 0)
    assert np.all(constrained >= 0)


def test_the_evidence_fails_where_the_framework_fails(small, system, caplog):
    """A singular H (the constant scheme's) raises in autoarray's Cholesky and
    `_safe_evidence` has always turned that into -inf with a log line. A trial
    must do the same, not invent a number."""
    reg = fitting.make_regularization("constant", 1e4, None, 1.5, None)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        trial = system.trial(reg, positive=False)
    assert trial.log_evidence == -np.inf
    assert np.isfinite(trial.chi_squared)
    assert "evidence evaluation failed" in caplog.text


def test_the_sparse_class_is_never_asked_for_the_dense_matrix():
    """`operated_mapping_matrix` exists on the sparse inversion too, inherited,
    and answering it would build the n_vis x n_mesh matrix the sparse path
    exists to avoid. The system must go through the transformer instead."""
    calls = []

    class FakeTransformer:
        def visibilities_from(self, image, xp=np):
            calls.append(np.asarray(image.array).sum())
            return np.zeros(3, dtype=complex)

    class FakeSparseInversion:
        operated_mapping_matrix = property(
            lambda self: pytest.fail("the dense matrix was requested")
        )
        transformer = FakeTransformer()
        mask = ag.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)

    class Obj:
        mapping_matrix = np.ones((4, 2))

    system = fitting.LinearSystem(
        F=np.eye(2), D=np.ones(2), data_term=0.0, noise_normalization=0.0,
        linear_obj=Obj(), settings=None, keep=None,
        inversion=FakeSparseInversion(),
    )
    out = system.model_visibilities(np.array([1.0, 2.0]))
    assert out.shape == (3,)
    assert calls == [pytest.approx(12.0)]


# --- nothing downstream rebuilds what it already has ---------------------------

@pytest.fixture
def transform_counter(monkeypatch):
    calls = []
    original = ag.TransformerDFT.transform_mapping_matrix

    def counted(self, mapping_matrix, xp=np):
        calls.append(mapping_matrix.shape)
        return original(self, mapping_matrix, xp=xp)

    monkeypatch.setattr(ag.TransformerDFT, "transform_mapping_matrix", counted)
    return calls


def test_a_whole_fit_transforms_the_mapping_matrix_once(small, transform_counter):
    """One template inversion gives F and D; the delivered fit is seeded with
    the same A_t. Twenty-odd trials, one transform."""
    dataset, geom = small
    sf = fitting.fit_dataset(
        dataset, geom, reg_kind="matern", fixed_scale=0.5, positive_only=True,
        criterion="discrepancy",
    )
    assert len(transform_counter) == 1
    # ...and the products do not add any
    sf.chi_squared, sf.log_evidence, sf.model_visibilities
    sf.posterior_covariance, sf.model_uncertainty, sf.prior_systematic()
    sf.model_uncertainty_total()
    assert len(transform_counter) == 1
    assert len(sf.scan.trials) > 5


def test_chi2_evidence_and_residuals_of_a_bare_fit_cost_no_transform(
    small, transform_counter
):
    """`_chi_squared`, `_safe_evidence`, `structure_ratio` and
    `SingleFit.model_visibilities` on a fit this module did not build: the
    fit's own transform, then nothing more. autoarray's `fast_chi_squared`
    rebuilds F and D on every access and `model_data` re-transforms the
    mapping matrix."""
    from pyuvimage.beam import DirtyImager

    dataset, geom = small
    fit = fitting.fit_at(dataset, geom.mesh_shape, "matern", 10.0, reg_scale=0.5)
    fit.inversion.reconstruction
    assert len(transform_counter) == 1
    n_data = 2 * len(np.asarray(dataset.data))
    chi2 = fitting._chi_squared(fit)
    ev = fitting._safe_evidence(fit)
    ratio = fitting.structure_ratio(fit, DirtyImager(dataset), n_data)
    sf = fitting.SingleFit(fit=fit, geometry=geom, prior={"coefficient": 10.0})
    vis = sf.model_visibilities
    assert len(transform_counter) == 1
    # and they are the framework's numbers
    assert chi2 == float(fit.inversion.fast_chi_squared)
    assert ev == _framework_evidence(fit)
    np.testing.assert_allclose(vis, np.asarray(fit.model_data), rtol=0, atol=1e-12)
    assert np.isfinite(ratio)


def test_single_fit_products_are_computed_once(small):
    dataset, geom = small
    sf = fitting.fit_dataset(
        dataset, geom, reg_kind="matern", prior={"coefficient": 10.0, "scale": 0.5},
    )
    assert sf.posterior_covariance is sf.posterior_covariance
    assert sf.model_uncertainty is sf.model_uncertainty
    assert sf.prior_systematic() is sf.prior_systematic()
    assert sf.prior_systematic(0.3) is not sf.prior_systematic(0.5)
    assert sf.model_image is sf.model_image
    np.testing.assert_allclose(
        sf.posterior_covariance,
        np.linalg.inv(np.asarray(fit_curv := sf.fit.inversion.curvature_reg_matrix)),
    )
    assert fit_curv.shape == (100, 100)


def test_the_prior_systematic_of_a_positive_fit_uses_the_positive_solver(small):
    """`model_image_at_scale` re-solved unconstrained whatever the delivered
    fit did, so on a non-negative fit the "systematic" compared a non-negative
    model with two signed ones and booked the positivity constraint itself as
    prior systematic."""
    dataset, geom = small
    sf = fitting.fit_dataset(
        dataset, geom, reg_kind="matern", positive_only=True,
        prior={"coefficient": 10.0, "scale": 0.5},
    )
    assert sf.positive_only
    M = fitting.system_for(sf.fit).mapping_matrix
    mask = sf.fit.dataset.real_space_mask
    for factor in (10.0, 0.1):
        alt = sf.model_image_at_scale(factor)
        assert alt is not None
        H = factor * sf.regularization_matrix
        positive = sf.system.solve(H, positive=True)
        unconstrained = sf.system.solve(H, positive=False)
        assert np.all(positive >= 0) and np.any(unconstrained < 0)
        want = np.asarray(ag.Array2D(values=M @ positive, mask=mask).native)
        np.testing.assert_allclose(alt, want, rtol=0, atol=1e-15)
        wrong = np.asarray(ag.Array2D(values=M @ unconstrained, mask=mask).native)
        assert np.abs(alt - wrong).max() > 1e-6
    free = fitting.fit_dataset(
        dataset, geom, reg_kind="matern", positive_only=False,
        prior={"coefficient": 10.0, "scale": 0.5},
    )
    unconstrained = free.system.solve(0.1 * free.regularization_matrix, positive=False)
    np.testing.assert_allclose(
        free.model_image_at_scale(0.1),
        np.asarray(ag.Array2D(values=M @ unconstrained, mask=mask).native),
        rtol=0, atol=1e-15,
    )


# --- the fixes that rode along ------------------------------------------------

def test_optimising_the_envelope_width_no_longer_crashes(small):
    """A9: `--reg gaussian --envelope-fwhm optimise` made the second free
    hyperparameter the envelope width, and the reachability probe built a
    one-element vector for an `evaluate` that read `log_params[1]`."""
    dataset, geom = small
    sf = fitting.fit_dataset(
        dataset, geom, reg_kind="gaussian", fixed_scale=0.4,
        optimise_envelope=True, envelope={"fwhm": 1.0, "centre": (0.0, 0.0)},
        positive_only=False, criterion="discrepancy",
    )
    assert "envelope_fwhm" in sf.prior
    assert sf.scan.free_parameters == ["coefficient", "envelope_fwhm"]
    assert all("envelope_fwhm" in t for t in sf.scan.trials)


def test_effective_criterion_is_the_last_in_the_chain():
    assert PriorScan(criterion="discrepancy").effective_criterion == "discrepancy"
    assert PriorScan(
        criterion="structure->discrepancy (ratio unreachable)"
    ).effective_criterion == "discrepancy"
    assert PriorScan(
        criterion="discrepancy->evidence (unreachable target)"
    ).effective_criterion == "evidence"
    assert PriorScan(
        criterion="structure->discrepancy->evidence (unreachable target)"
    ).effective_criterion == "evidence"


# A controlled solver, for the paths that depend on what the solver returns.

class _FakeSystem:
    def __init__(self, chi2_of, n_pixels=64, n_vis=60):
        self.chi2_of = chi2_of
        self.seen = []
        self.n_pixels = n_pixels
        self.n_vis = n_vis

    def trial(self, regularization, positive):
        c = float(regularization.coefficient)
        self.seen.append((c, bool(positive)))
        chi2 = self.chi2_of(c, positive)
        rec = np.full(self.n_pixels, 1.0 + np.log10(c) / 20.0)
        return Trial(
            coefficient=c, positive=bool(positive), reconstruction=rec,
            chi_squared=chi2, log_evidence=-0.5 * chi2,
            regularization_matrix=np.eye(self.n_pixels),
        )

    def residual_visibilities(self, reconstruction):
        return np.zeros(self.n_vis, dtype=complex)


class _FakeFit:
    def __init__(self, chi2, rec):
        class Inv:
            pass

        self.inversion = Inv()
        self.inversion.fast_chi_squared = chi2
        self.inversion.reconstruction = rec
        self.figure_of_merit = -0.5 * chi2


def _install(monkeypatch, chi2_of, n_pixels=100):
    system = _FakeSystem(chi2_of, n_pixels=n_pixels)

    def fit_at(dataset, mesh_shape, reg_kind, coefficient, positive_only=True,
               **kwargs):
        t = system.trial(
            fitting.make_regularization("constant", coefficient), positive_only
        )
        system.seen.pop()  # the delivered fit is not a search trial
        return _FakeFit(t.chi_squared, t.reconstruction)

    monkeypatch.setattr(fitting, "build_linear_system", lambda *a, **k: system)
    monkeypatch.setattr(fitting, "fit_at", fit_at)
    return system


def test_the_positivity_probe_runs_at_the_chosen_coefficient(monkeypatch, small):
    """It used to probe at coefficient 1.0 -- the log-midpoint of the shipped
    bounds and otherwise arbitrary. Here the constrained solver is fine where
    the fit will run and far worse at 1.0: positivity must survive."""
    dataset, geom = small
    n_data = 2 * len(np.asarray(dataset.data))

    def chi2_of(c, positive):
        base = n_data * (0.95 + 0.5 * c / (c + 1e3))
        if positive and abs(np.log10(c)) < 0.5:
            return 10.0 * base   # a pathology only at c ~ 1
        return base

    system = _install(monkeypatch, chi2_of)
    sf = fitting.fit_dataset(dataset, geom, reg_kind="constant", positive_only=True)
    assert sf.positive_only is True
    chosen = sf.coefficient
    assert any(
        c == pytest.approx(chosen) and not pos for c, pos in system.seen
    ), "the unconstrained probe never ran at the chosen coefficient"
    assert any(c == pytest.approx(chosen) and pos for c, pos in system.seen)


def test_a_solver_that_fails_at_the_chosen_coefficient_is_caught(monkeypatch, small):
    dataset, geom = small
    n_data = 2 * len(np.asarray(dataset.data))

    def chi2_of(c, positive):
        base = n_data * (0.95 + 0.5 * c / (c + 1e3))
        # fine at 1.0, ten times worse everywhere the search will land
        return 10.0 * base if positive and c > 10.0 else base

    _install(monkeypatch, chi2_of)
    sf = fitting.fit_dataset(dataset, geom, reg_kind="constant", positive_only=True)
    assert sf.positive_only is False


def test_structure_handing_back_to_chi2_is_then_rebisected(monkeypatch, small):
    """A10: after the structure -> discrepancy fallback `fit_dataset` saw
    `criterion == "structure"` and skipped the constrained re-bisection that
    every chi^2 choice under positivity needs."""
    dataset, geom = small
    n_data = 2 * len(np.asarray(dataset.data))
    # the ratio never reaches 1, so structure gives up and hands to chi^2
    monkeypatch.setattr(fitting, "_structure_ratio", lambda *a, **k: 0.5)

    def chi2_of(c, positive):
        floor = 1.02 if positive else 0.95
        return n_data * (floor + 0.5 * c / (c + 1e3))

    system = _install(monkeypatch, chi2_of)
    sf = fitting.fit_dataset(
        dataset, geom, reg_kind="constant", positive_only=True,
        criterion="structure",
    )
    assert sf.scan.criterion == "structure->discrepancy (ratio unreachable)"
    assert sf.scan.effective_criterion == "discrepancy"
    # the delivered coefficient was chosen on the constrained solver
    constrained = [c for c, pos in system.seen if pos]
    assert sf.coefficient in constrained
    assert sf.chi_squared / n_data == pytest.approx(
        1.02 * (1.0 + fitting.chi2_floor_tolerance(n_data)), rel=0.02
    )


def test_an_inner_evidence_fallback_is_not_lost_in_the_chain(
    monkeypatch, small, caplog
):
    """structure -> discrepancy -> evidence: the middle step used to overwrite
    the inner marker, and the gate then re-bisected an evidence choice against
    a chi^2 target it had already given up on."""
    dataset, geom = small
    n_data = 2 * len(np.asarray(dataset.data))
    monkeypatch.setattr(fitting, "_structure_ratio", lambda *a, **k: 0.5)
    _install(
        monkeypatch, lambda c, positive: n_data * (2.0 + 0.5 * c / (c + 1e3))
    )
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        sf = fitting.fit_dataset(
            dataset, geom, reg_kind="constant", positive_only=True,
            criterion="structure",
        )
    assert sf.scan.criterion == (
        "structure->discrepancy->evidence (unreachable target)"
    )
    assert sf.scan.effective_criterion == "evidence"
    assert "re-optimising the coefficient with the constrained solver" not in caplog.text


def test_the_adaptive_first_pass_honours_warn_on_chi2(monkeypatch, small, caplog):
    dataset, geom = small
    n_data = 2 * len(np.asarray(dataset.data))
    # hopeless everywhere, so both passes end far above the target
    _install(monkeypatch, lambda c, positive: n_data * (2.0 + 0.5 * c / (c + 1e3)))
    envelope = {"floor": 1e-2, "power": 2.0}
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        fitting.fit_dataset(
            dataset, geom, reg_kind="adaptive", positive_only=False,
            fixed_scale=0.5, envelope=dict(envelope), warn_on_chi2=False,
        )
    assert "should not be trusted" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        fitting.fit_dataset(
            dataset, geom, reg_kind="adaptive", positive_only=False,
            fixed_scale=0.5, envelope=dict(envelope), warn_on_chi2=True,
        )
    assert caplog.text.count("should not be trusted") == 2


def test_the_delivered_fit_is_a_real_framework_fit(small):
    dataset, geom = small
    sf = fitting.fit_dataset(
        dataset, geom, reg_kind="matern", fixed_scale=0.5, criterion="evidence",
    )
    assert isinstance(sf.fit, ag.FitInterferometer)
    assert sf.chi_squared == float(sf.fit.inversion.fast_chi_squared)


# --- with_sparse_operator reads the dataset ------------------------------------

class _KernelDataset:
    def __init__(self, n=40, accepts_use_jax=True, jax_fails=False):
        rng = np.random.default_rng(0)
        self.uv_wavelengths = rng.normal(0, 1e4, (n, 2))
        self.noise_map = np.full(n, 0.1 + 0.1j)
        self.data = np.zeros(n, dtype=complex)
        self.transformer = ag.TransformerDFT(
            uv_wavelengths=self.uv_wavelengths,
            real_space_mask=ag.Mask2D.all_false(shape_native=(8, 8), pixel_scales=0.1),
        )
        self.calls = []
        self.sparse_operator = None
        self._jax_fails = jax_fails
        if accepts_use_jax:
            def build(chunk_k=2048, use_jax=False):
                self.calls.append(use_jax)
                if use_jax and self._jax_fails:
                    raise RuntimeError("no JAX here")
                return np.ones((16, 16))
        else:
            def build(chunk_k=2048):
                self.calls.append(None)
                return np.ones((16, 16))
        self.psf_precision_operator_from = build

    def apply_sparse_operator(self, nufft_precision_operator, batch_size):
        return self


class _Geom:
    shape_native = (8, 8)
    pixel_scale = 0.1
    mesh_shape = (4, 4)


@pytest.fixture
def sparse_allowed(monkeypatch):
    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    monkeypatch.setattr(fitting, "warn_on_single_precision", lambda: True)


def test_with_sparse_operator_reads_uv_and_noise_from_the_dataset(sparse_allowed):
    ds = _KernelDataset()
    out = fitting.with_sparse_operator(ds, geometry=_Geom())
    assert out is ds
    assert ds.calls == [True]


def test_a_pair_that_disagrees_in_shape_with_the_dataset_is_refused(sparse_allowed):
    ds = _KernelDataset()
    with pytest.raises(ValueError, match="shape"):
        fitting.with_sparse_operator(
            ds, ds.uv_wavelengths[:-1], ds.noise_map[:-1], _Geom()
        )


def test_a_pair_that_disagrees_in_value_is_warned_about(sparse_allowed, caplog):
    ds = _KernelDataset()
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        fitting.with_sparse_operator(
            ds, ds.uv_wavelengths, ds.noise_map * 1.5, _Geom()
        )
    assert "differs from the dataset" in caplog.text


def test_the_kernel_build_falls_back_to_numpy_when_jax_fails(sparse_allowed, caplog):
    ds = _KernelDataset(jax_fails=True)
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        fitting.with_sparse_operator(ds, geometry=_Geom())
    assert ds.calls == [True, False]
    assert "JAX build of the w-tilde kernel failed" in caplog.text


def test_the_kernel_build_omits_use_jax_where_autoarray_lacks_it(sparse_allowed):
    ds = _KernelDataset(accepts_use_jax=False)
    fitting.with_sparse_operator(ds, geometry=_Geom())
    assert ds.calls == [None]


def test_the_kernel_flag_can_be_switched_off(sparse_allowed, monkeypatch):
    monkeypatch.setattr(fitting, "SPARSE_KERNEL_USE_JAX", False)
    ds = _KernelDataset()
    fitting.with_sparse_operator(ds, geometry=_Geom())
    assert ds.calls == [False]


# --- resident memory is the current figure, not the high-water mark -----------

def test_current_memory_is_current():
    """`ru_maxrss` is the process's peak; a released mapping matrix must stop
    counting against the budget once it is gone."""
    import resource

    before = fitting.current_memory_gb()
    big = np.ones(int(3e7))          # 240 MB
    big += 1.0
    during = fitting.current_memory_gb()
    del big
    import gc

    gc.collect()
    after = fitting.current_memory_gb()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    assert during - before > 0.15
    assert after < during - 0.15, "the freed array is still being counted"
    assert after <= peak + 1e-3

"""Choosing the regularisation strength: unreachable targets, and the
residual-structure criterion.

Two failures on PJ0116 at 245 GHz motivate this file.

1. **An unreachable chi^2 target is not a near miss.** Positivity raises
   chi^2, so the constrained fit can floor *above* chi^2 = N while the
   unconstrained solve reaches it. Every bisection trial then reads "still
   too high", and the search walks the coefficient down to its lower bound --
   switching the prior off and delivering the noisiest model available. The
   fix is to measure the floor on the solver actually in use and, when it is
   only just out of reach, aim a hair above it instead.

2. **chi^2 can be flat in the coefficient.** Two decades of smoothing moved
   chi^2/N by 0.0008 on that dataset while the model went from white
   residuals to visibly over-smoothed. The residual *map's* structure ratio
   moved 0.67 -> 0.95 over the same range, so it can select what chi^2
   cannot.
"""

import numpy as np
import pytest

from pyuvimage import fitting, mock
from pyuvimage.fitting import (
    CHI2_FLOOR_TOLERANCE,
    CHI2_FLOOR_SIGMAS,
    chi2_floor_tolerance,
    CHI2_REBISECT_TOLERANCE,
    LOG_COEFFICIENT_BOUNDS,
    PriorScan,
    effective_chi2_target,
)


# --- effective_chi2_target -------------------------------------------------

def test_reachable_target_is_left_alone():
    assert effective_chi2_target(100.0, 90.0) == 100.0


def test_unreachable_target_is_raised_just_above_the_floor():
    got = effective_chi2_target(100.0, 110.0)
    assert got == pytest.approx(110.0 * (1.0 + CHI2_FLOOR_TOLERANCE))
    # a hair above, not a free pass: still within a few per cent of the floor
    assert got / 110.0 < 1.1


def test_an_unmeasured_floor_changes_nothing():
    assert effective_chi2_target(100.0, float("nan")) == 100.0


# --- PriorScan bookkeeping -------------------------------------------------

def test_scan_records_the_structure_ratio_and_the_solver():
    scan = PriorScan(n_data=100)
    scan.record({"coefficient": 1.0}, -1.0, 100.0, 0.8, positive=True)
    scan.record({"coefficient": 2.0}, -1.0, 100.0)
    assert scan.trials[0]["structure_ratio"] == 0.8
    assert scan.trials[0]["positive"] is True
    # a ratio that was never measured is absent, not NaN-filled
    assert "structure_ratio" not in scan.trials[1]
    assert "positive" not in scan.trials[1]


def test_unknown_criterion_is_rejected(demo_geometry):
    dataset, geometry = demo_geometry
    with pytest.raises(ValueError, match="criterion"):
        fitting.optimise_prior(dataset, geometry, criterion="vibes")


# --- the pathology, on a solver whose behaviour we control ------------------

@pytest.fixture(scope="module")
def demo_geometry():
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=60, mesh_n=8, seed=5)
    uv, d, n = uvd.flattened()
    return fitting.make_dataset(uv, d, n, geom, transformer="dft"), geom


class _FakeInversion:
    def __init__(self, chi2, reconstruction=None):
        self.fast_chi_squared = chi2
        # The positivity probe asks whether the *reconstruction* responds to
        # the coefficient, not whether chi^2 does -- chi^2 being flat in the
        # coefficient is normal on well-constrained data (Ruby: 0.4% across
        # twelve decades) and is not by itself a symptom. So the fake has to
        # carry a model too, or the probe sees nothing to judge.
        self.reconstruction = (
            np.zeros(4) if reconstruction is None else reconstruction
        )


class _FakeFit:
    """A fit whose chi^2 is a known, monotone function of the coefficient.

    chi^2 rises with the prior strength from a floor at zero smoothing, and
    the floor is *higher* under positivity -- which is the whole point: it
    reproduces the case where the constrained solve cannot reach the target
    and the unconstrained one can.
    """

    def __init__(self, chi2, reconstruction=None):
        self.inversion = _FakeInversion(chi2, reconstruction)
        self.figure_of_merit = -0.5 * chi2


def _fake_fit_at(n_data, floor_free, floor_positive, gain=0.5):
    """`gain` drives chi^2 *and* the model, together.

    A solver ignoring the prior returns the same answer at every coefficient,
    which is what `gain=0.0` now means: flat chi^2 and a frozen model. A
    healthy one moves both.
    """
    def fit_at(dataset, mesh_shape, reg_kind, coefficient, positive_only=True,
               **kwargs):
        floor = floor_positive if positive_only else floor_free
        c = float(coefficient)
        response = gain * c / (c + 1e3)
        return _FakeFit(
            n_data * (floor + response),
            reconstruction=np.full(4, 1.0 + response),
        )
    return fit_at


def _run(monkeypatch, dataset, geometry, floor_positive, **kwargs):
    n_data = 2 * len(np.asarray(dataset.data))
    monkeypatch.setattr(
        fitting, "fit_at",
        _fake_fit_at(n_data, floor_free=0.98, floor_positive=floor_positive),
    )
    sf = fitting.fit_dataset(
        dataset, geometry, reg_kind="constant", positive_only=True, **kwargs
    )
    return sf, n_data


def test_an_unreachable_target_does_not_switch_the_prior_off(
    monkeypatch, demo_geometry
):
    """The regression. Constrained chi^2 floors at 1.02 N against a target of
    N; the coefficient must not collapse to its lower bound."""
    dataset, geometry = demo_geometry
    sf, n_data = _run(monkeypatch, dataset, geometry, floor_positive=1.02)

    assert np.log10(sf.prior["coefficient"]) > LOG_COEFFICIENT_BOUNDS[0] + 1.0
    # it lands at the knee: just above the floor the solver can reach, by
    # however much chi^2/N is uncertain at *this* dataset size
    assert sf.chi_squared / n_data == pytest.approx(
        1.02 * (1.0 + chi2_floor_tolerance(n_data)), rel=0.02
    )
    assert np.isfinite(sf.scan.chi2_floor)
    assert sf.scan.chi2_floor / n_data == pytest.approx(1.02, rel=1e-3)


def test_the_floor_is_measured_on_the_solver_in_use(monkeypatch, demo_geometry):
    """Probing the weakest prior with the *unconstrained* solver was the bug:
    it reports a floor of 0.98 N, the target looks reachable, and the search
    then chases it with a solver that cannot get there."""
    dataset, geometry = demo_geometry
    seen = []
    n_data = 2 * len(np.asarray(dataset.data))
    inner = _fake_fit_at(n_data, floor_free=0.98, floor_positive=1.02)

    def spy(dataset_, mesh_shape, reg_kind, coefficient, positive_only=True,
            **kwargs):
        seen.append((float(coefficient), bool(positive_only)))
        return inner(dataset_, mesh_shape, reg_kind, coefficient,
                     positive_only=positive_only, **kwargs)

    monkeypatch.setattr(fitting, "fit_at", spy)
    fitting.fit_dataset(dataset, geometry, reg_kind="constant",
                        positive_only=True)

    weakest = 10.0 ** LOG_COEFFICIENT_BOUNDS[0]
    probes = [pos for c, pos in seen if c == pytest.approx(weakest)]
    assert probes, "the weakest prior was never probed"
    assert any(probes), "the reachability probe never used the constrained solver"


def test_a_reachable_target_still_lands_on_it(monkeypatch, demo_geometry):
    """The fix must not disturb the ordinary case."""
    dataset, geometry = demo_geometry
    sf, n_data = _run(monkeypatch, dataset, geometry, floor_positive=0.9)
    assert sf.chi_squared / n_data == pytest.approx(
        1.0, rel=CHI2_REBISECT_TOLERANCE
    )


def test_a_hopeless_target_falls_back_to_the_evidence(
    monkeypatch, demo_geometry
):
    """A floor far above the target is a different failure -- the model
    genuinely cannot reproduce the data -- and the knee would be meaningless.
    Fall back to maximum evidence rather than pretend."""
    dataset, geometry = demo_geometry
    sf, _ = _run(monkeypatch, dataset, geometry, floor_positive=1.6)
    assert "evidence" in sf.scan.criterion


# --- the structure criterion, on real (mock) data --------------------------

def test_structure_criterion_measures_every_trial_and_selects_on_it(
    demo_geometry,
):
    dataset, geometry = demo_geometry
    prior, scan = fitting.optimise_prior(
        dataset, geometry, reg_kind="constant", criterion="structure",
    )
    assert scan.trials, "the search recorded nothing"
    assert all("structure_ratio" in t for t in scan.trials)
    # the criterion is a *map* measurement, so it must move with the prior
    # even where chi^2 barely does
    ratios = [t["structure_ratio"] for t in scan.trials]
    assert np.nanmax(ratios) - np.nanmin(ratios) > 0.05
    assert prior["coefficient"] > 0


def test_structure_criterion_beats_the_weakest_prior_on_its_own_measure(
    demo_geometry,
):
    """Whatever it picks must leave a residual map closer to white than the
    unregularised solve does -- that is the entire claim."""
    from pyuvimage.beam import DirtyImager

    dataset, geometry = demo_geometry
    imager = DirtyImager(dataset)
    n_data = 2 * len(np.asarray(dataset.data))
    prior, _ = fitting.optimise_prior(
        dataset, geometry, reg_kind="constant", criterion="structure",
    )

    def ratio(coefficient):
        return fitting.structure_ratio(
            fitting.fit_at(dataset, geometry.mesh_shape, "constant",
                           coefficient, positive_only=False),
            imager, n_data,
        )

    chosen = ratio(prior["coefficient"])
    weakest = ratio(10.0 ** LOG_COEFFICIENT_BOUNDS[0])
    assert abs(chosen - 1.0) < abs(weakest - 1.0)


def test_the_structure_search_uses_the_solver_the_fit_will_use(demo_geometry):
    """Positivity changes the residual map, so a structure ratio measured on
    the unconstrained solve is not the one the delivered fit will have. Every
    ratio in the scan must therefore come from a constrained solve."""
    dataset, geometry = demo_geometry
    _, scan = fitting.optimise_prior(
        dataset, geometry, reg_kind="constant", criterion="structure",
        positive_only=True,
    )
    measured = [t for t in scan.trials if "structure_ratio" in t]
    assert measured, "no structure ratio was measured"
    assert all(t.get("positive") for t in measured)


# --- reporting which solver actually ran -----------------------------------

def test_the_delivered_solver_is_recorded_not_the_one_requested(
    monkeypatch, demo_geometry
):
    """`fit_parameters.json` read `positive_only: true` on a Ruby fit where
    the guard had disabled positivity. The fit was fine; the record lied about
    what produced it, which is the kind of thing that misleads a reader months
    later."""
    dataset, geometry = demo_geometry
    n_data = 2 * len(np.asarray(dataset.data))
    # a solver that ignores the prior entirely is what trips the guard
    monkeypatch.setattr(
        fitting, "fit_at",
        _fake_fit_at(n_data, floor_free=0.98, floor_positive=1.0, gain=0.0),
    )
    sf = fitting.fit_dataset(
        dataset, geometry, reg_kind="constant", positive_only=True
    )
    assert sf.positive_only is False


def test_an_untroubled_fit_records_positivity_as_asked(demo_geometry):
    dataset, geometry = demo_geometry
    sf = fitting.fit_dataset(
        dataset, geometry, reg_kind="constant", positive_only=True,
        prior={"coefficient": 1e3},
    )
    assert sf.positive_only is True


# --- controlling the positivity fallback -----------------------------------

def test_enforce_positive_keeps_positivity_through_a_bad_solver(
    monkeypatch, demo_geometry
):
    """By default a solver caught ignoring the prior gets switched off, which
    is right when the goal is a good image and wrong when a strictly
    non-negative model is the point. `enforce_positive` picks the other
    side."""
    dataset, geometry = demo_geometry
    n_data = 2 * len(np.asarray(dataset.data))
    monkeypatch.setattr(
        fitting, "fit_at",
        _fake_fit_at(n_data, floor_free=0.98, floor_positive=1.0, gain=0.0),
    )
    default = fitting.fit_dataset(
        dataset, geometry, reg_kind="constant", positive_only=True
    )
    forced = fitting.fit_dataset(
        dataset, geometry, reg_kind="constant", positive_only=True,
        enforce_positive=True,
    )
    assert default.positive_only is False
    assert forced.positive_only is True


def test_enforce_positive_still_says_the_solver_looked_wrong(
    monkeypatch, demo_geometry, caplog
):
    """Keeping positivity must not mean hiding why that is a compromise."""
    import logging

    dataset, geometry = demo_geometry
    n_data = 2 * len(np.asarray(dataset.data))
    monkeypatch.setattr(
        fitting, "fit_at",
        _fake_fit_at(n_data, floor_free=0.98, floor_positive=1.0, gain=0.0),
    )
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        fitting.fit_dataset(
            dataset, geometry, reg_kind="constant", positive_only=True,
            enforce_positive=True,
        )
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "unreliable" in said and "enforce_positive" in said


def test_enforce_positive_does_nothing_when_the_solver_is_fine(demo_geometry):
    dataset, geometry = demo_geometry
    sf = fitting.fit_dataset(
        dataset, geometry, reg_kind="constant", positive_only=True,
        prior={"coefficient": 1e3}, enforce_positive=True,
    )
    assert sf.positive_only is True


# --- how far above the floor still counts as "as good" ----------------------
#
# This was a flat 5% and it was the reason a Ruby fit at chi^2/N = 1.069 left
# the whole ring in the residual map at 60 sigma: 5% is 3.6 sigma at N = 10^4
# and 19 sigma at N = 3x10^5, so the same number means "the knee" on a small
# dataset and "smooth away the source" on a large one.


def test_the_tolerance_shrinks_as_the_dataset_grows():
    small, large = chi2_floor_tolerance(10_000), chi2_floor_tolerance(300_000)
    assert small > large
    # it is sqrt(2/N), so 30x the data is sqrt(30) times tighter
    assert small / large == pytest.approx(np.sqrt(30.0), rel=0.01)


def test_the_tolerance_is_a_fixed_number_of_sigma():
    for n in (10_316, 296_954, 328_524):
        sigma = np.sqrt(2.0 / n)
        assert chi2_floor_tolerance(n) == pytest.approx(CHI2_FLOOR_SIGMAS * sigma)


def test_the_ruby_case_lands_where_the_structure_criterion_independently_did():
    """The corroboration: on both large real datasets, aiming k=2 sigma above
    the floor puts the coefficient within 0.1% of where `structure` -- which
    knows nothing about chi^2 -- puts it. Ruby floors at 1.018 and `structure`
    chose 1.0225; 9io9 floors at 1.0298 by the same route."""
    got = 1.018 * (1.0 + chi2_floor_tolerance(296_954))
    assert got == pytest.approx(1.0233, abs=5e-4)
    assert abs(got - 1.0225) / 1.0225 < 0.002


def test_an_unknown_sample_count_falls_back_to_the_old_fixed_fraction():
    assert chi2_floor_tolerance(None) == CHI2_FLOOR_TOLERANCE
    assert chi2_floor_tolerance(0) == CHI2_FLOOR_TOLERANCE


def test_a_tiny_dataset_is_not_handed_unlimited_slack():
    """sqrt(2/N) grows without bound as N shrinks; the cap stops a 50-point
    dataset from being told that any prior fits essentially as well."""
    assert chi2_floor_tolerance(50) == fitting.CHI2_FLOOR_TOLERANCE_MAX


def test_the_target_uses_the_sample_count_when_it_is_given():
    """Same floor, same target, two dataset sizes -- and the large one gets a
    much tighter effective target."""
    small = effective_chi2_target(100.0, 110.0, 10_000)
    large = effective_chi2_target(100.0, 110.0, 300_000)
    assert 110.0 < large < small


def test_the_rebisection_gate_is_statistical_too():
    """Same flat-fraction mistake, one function along: 3% is 11 sigma at
    N = 3x10^5, which lets a badly-smoothed constrained fit through as
    "close enough" and skips the re-bisection entirely."""
    assert fitting.chi2_rebisect_tolerance(None) == fitting.CHI2_REBISECT_TOLERANCE
    assert fitting.chi2_rebisect_tolerance(296_954) < 0.01
    # never tighter than the absolute floor, however large the dataset
    assert fitting.chi2_rebisect_tolerance(10**9) == fitting.CHI2_REBISECT_FLOOR
    # ...and never *looser* than it was: this gate decides whether to
    # re-bisect at all, so a small dataset must not be let off the hook
    for n in (200, 1_400, 10_316):
        assert fitting.chi2_rebisect_tolerance(n) <= fitting.CHI2_REBISECT_TOLERANCE


# --- picking the criterion automatically -----------------------------------
#
# The last knob this needed. `structure` is the better criterion wherever it
# is calibrated -- at ratio 1.0 all three real datasets land at 3.9-5.0 sigma
# -- but it mis-selects on a weakly constrained fit, where the residual map is
# not white at chi^2 = N to begin with. Data per model pixel separates the two
# regimes cleanly, so `auto` decides on that and says which it took.


def test_auto_takes_structure_when_the_data_comfortably_outnumber_the_model():
    """Ruby: 296,954 data points over a 26x26 mesh is 439 per pixel."""
    assert fitting.resolve_criterion("auto", 296_954, 676) == "structure"
    assert fitting.resolve_criterion("auto", 328_524, 676) == "structure"


def test_auto_takes_discrepancy_on_a_weakly_constrained_fit():
    """The demo mock: 400 data points over 144 mesh pixels, 2.8 per pixel.
    Its structure ratio reads 0.49 at chi^2/N = 0.999, so driving the ratio
    to 1 over-smooths to chi^2/N = 1.59."""
    assert fitting.resolve_criterion("auto", 400, 144) == "discrepancy"
    # PJ0116 at 4.1 per pixel is the marginal case, and lands on discrepancy
    # -- which costs nothing there: 3.9 sigma either way, and it is faster.
    assert fitting.resolve_criterion("auto", 10_316, 2500) == "discrepancy"


def test_the_threshold_is_where_the_constant_says_it_is():
    n_mesh = 100
    k = fitting.CRITERION_AUTO_DATA_PER_PARAMETER
    assert fitting.resolve_criterion("auto", int(k * n_mesh), n_mesh) == "structure"
    assert fitting.resolve_criterion(
        "auto", int(k * n_mesh) - n_mesh, n_mesh
    ) == "discrepancy"


def test_an_explicit_criterion_is_never_overridden():
    for c in fitting.CRITERIA:
        assert fitting.resolve_criterion(c, 10**6, 100) == c


def test_auto_falls_back_safely_when_the_sizes_are_unknown():
    assert fitting.resolve_criterion("auto", None, None) == "discrepancy"
    assert fitting.resolve_criterion("auto", 0, 0) == "discrepancy"


def test_auto_says_which_it_took_and_why(caplog):
    """The point of an automatic choice is that it is not silent."""
    import logging

    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        fitting.resolve_criterion("auto", 296_954, 676)
    assert "auto -> structure" in caplog.text
    assert "--criterion discrepancy to override" in caplog.text


def test_a_flat_chi2_alone_does_not_disable_positivity(
    monkeypatch, demo_geometry, caplog
):
    """The regression, from Ruby 200 GHz on the sparse path.

    chi^2 being insensitive to the regularisation coefficient is a normal
    property of well-constrained data, not a broken solver. At 439 data points
    per model pixel Ruby's chi^2 moves 0.4% across twelve decades of
    coefficient -- under the old 1% threshold -- while the model changes out
    of all recognition over the same range, the structure ratio running from
    0.228 to 3.51. The probe concluded the solver was "ignoring the prior
    entirely" and silently switched positivity off on a fit where the prior
    was working perfectly.

    So: a model that responds means the prior is being applied, however flat
    chi^2 is.
    """
    import logging

    dataset, geometry = demo_geometry
    n_data = 2 * len(np.asarray(dataset.data))

    def fit_at(dataset_, mesh_shape, reg_kind, coefficient,
               positive_only=True, **kwargs):
        c = float(coefficient)
        # chi^2 moves 0.4% across the whole range -- Ruby's number, and below
        # the 1% the old test demanded
        chi2 = n_data * (1.018 + 0.004 * c / (c + 1e3))
        # ...but the model keeps moving, as a working prior must. Logarithmic
        # rather than saturating: a Matern prior goes on smoothing as the
        # coefficient grows, so the response must not flatten out at the top
        # of the range where the bisection actually looks.
        return _FakeFit(
            chi2, reconstruction=np.full(4, 10.0 + np.log10(max(c, 1e-30)))
        )

    monkeypatch.setattr(fitting, "fit_at", fit_at)
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        sf = fitting.fit_dataset(
            dataset, geometry, reg_kind="constant", positive_only=True
        )
    assert sf.positive_only is True, "positivity was disabled on a healthy fit"
    assert "ignoring the prior" not in "\n".join(
        r.getMessage() for r in caplog.records
    )


def test_a_frozen_model_still_disables_positivity(
    monkeypatch, demo_geometry, caplog
):
    """The genuine pathology the probe exists for: same answer at every
    coefficient. Fixing the false positive above must not cost the true one."""
    import logging

    dataset, geometry = demo_geometry
    n_data = 2 * len(np.asarray(dataset.data))
    monkeypatch.setattr(
        fitting, "fit_at",
        _fake_fit_at(n_data, floor_free=0.98, floor_positive=1.0, gain=0.0),
    )
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        sf = fitting.fit_dataset(
            dataset, geometry, reg_kind="constant", positive_only=True
        )
    assert sf.positive_only is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ignoring the prior" in text
    assert "reconstruction changes by only" in text

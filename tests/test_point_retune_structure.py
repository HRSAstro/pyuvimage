"""Point components on large data: the structure retune and the user-point checks.

The failure these pin came from a user's field (a bright compact feature at
the centre, 1.8e7 data, `--point=0,0`): structure ratio 0.40 and a point at
-4.3 mJy / 41.6 sigma, delivered without comment. Reproduced on mocks in
`claude/central-point-mock-reproduction.md`. Two causes:

1. The coefficient is chosen by a *mesh-only* search, which tunes the prior to
   let the mesh chase the point's residual. With the point taking that flux
   the prior is decades too weak. The re-tune that exists for this ran only
   under `discrepancy`, whose chi^2 is flat on big data -- and `auto` picks
   `structure` there.
2. User positions skipped both the sign check and the unresolved test that
   auto-detected candidates face. A negative point at a user position is the
   signature of a resolved compact source.
"""
import logging

import numpy as np
import pytest

from pyuvimage import fitting, mock
from pyuvimage.beam import DirtyImager, fit_beam
from pyuvimage.pointsource import (
    AugmentedSystem,
    augmented_structure_ratio,
    fit_point_sources,
    retune_regularization,
    sky_to_grid,
)

# weak enough that the mesh chases a point's residual: the mesh-only regime
WEAK = {"coefficient": 1.0, "scale": 0.25, "nu": 1.5}


def _field(points, extended=None, seed=5, n_vis=600):
    return mock.make_field_dataset(
        fov_arcsec=3.0, mesh_n=20, n_vis=n_vis, sigma_jy=3e-4, seed=seed,
        extended=extended or [(0.040, 0.70, (0.0, 0.0), 1.0, 0.0)],
        points=points, truth_on_mesh=False,
    )


def _fit(uvd, geom, prior=WEAK):
    uv, d, nz = uvd.flattened()
    ds = fitting.make_dataset(uv, d, nz, geom)
    fit = fitting.fit_dataset(ds, geom, reg_kind="matern", prior=prior,
                              positive_only=False)
    return fit, ds


def _beam(ds, geom):
    b = fit_beam(DirtyImager(ds).dirty_beam, geom.pixel_scale)
    return float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))


@pytest.fixture(scope="module")
def central_point():
    """A true point on the extended peak -- the user's configuration."""
    uvd, _, geom, _ = _field([(0.012, (0.0, 0.0))])
    fit, ds = _fit(uvd, geom)
    return fit, ds, geom


# --- the structure retune ---------------------------------------------------

def test_a_weak_prior_overfits_once_the_point_is_in(central_point):
    """The premise: at the mesh-only strength, adding the point leaves the
    combined residual far quieter than noise."""
    fit, ds, geom = central_point
    system = AugmentedSystem(fit.fit.inversion, ds)
    ratio = augmented_structure_ratio(system, [(0.0, 0.0)], DirtyImager(ds))
    assert ratio < 0.85


def test_the_structure_retune_brings_the_ratio_to_one(central_point):
    fit, ds, geom = central_point
    system = AugmentedSystem(fit.fit.inversion, ds)
    imager = DirtyImager(ds)
    factor = retune_regularization(
        system, [(0.0, 0.0)], criterion="structure", imager=imager)
    assert factor > 1.0, "the prior has to stiffen"
    assert augmented_structure_ratio(system, [(0.0, 0.0)], imager) == \
        pytest.approx(1.0, abs=0.03)


def test_the_retune_recovers_the_point_flux(central_point):
    """At the mesh-only strength the split between the point and the mesh
    under it is set by a prior too weak to set anything: on this mock the
    true 12 mJy point came back at -86 mJy, the mesh carrying the difference.
    That is Teresa's negative point from a true point source. Once the prior
    is re-tuned with the point in, the amplitude is the truth."""
    fit, ds, geom = central_point
    system = AugmentedSystem(fit.fit.inversion, ds)
    retune_regularization(system, [(0.0, 0.0)], criterion="structure",
                          imager=DirtyImager(ds))
    after = system.solve([(0.0, 0.0)])[1][0]
    assert after == pytest.approx(0.012, rel=0.1)


def test_fit_point_sources_retunes_on_structure_when_asked(central_point):
    fit, ds, geom = central_point
    sol = fit_point_sources(
        fit.fit.inversion, ds, geom, positions=[(0.0, 0.0)],
        beam_fwhm=_beam(ds, geom), retune=True, retune_criterion="structure",
    )
    assert sol.regularization_factor > 1.0
    ratio = augmented_structure_ratio(sol.system, sol.grid_positions,
                                      DirtyImager(ds))
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_a_structure_retune_without_an_imager_is_refused(central_point):
    fit, ds, geom = central_point
    system = AugmentedSystem(fit.fit.inversion, ds)
    with pytest.raises(ValueError, match="imager"):
        retune_regularization(system, [(0.0, 0.0)], criterion="structure")


def test_an_unknown_retune_criterion_is_refused(central_point):
    fit, ds, geom = central_point
    system = AugmentedSystem(fit.fit.inversion, ds)
    with pytest.raises(ValueError, match="evidence"):
        retune_regularization(system, [(0.0, 0.0)], criterion="evidence")


# --- user positions: sign ---------------------------------------------------

@pytest.fixture(scope="module")
def negative_at_user_position():
    """A *negative* delta in the truth, so the sign of the fit is not luck."""
    uvd, _, geom, _ = _field([(-0.006, (0.6, -0.5))], seed=9)
    fit, ds = _fit(uvd, geom, prior={"coefficient": 1e7, "scale": 0.25, "nu": 1.5})
    return fit, ds, geom


def test_a_negative_user_point_is_dropped_and_said(negative_at_user_position,
                                                   caplog):
    fit, ds, geom = negative_at_user_position
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        sol = fit_point_sources(fit.fit.inversion, ds, geom,
                                positions=[(0.6, -0.5)], retune=False)
    assert sol.points == []
    assert "negative flux" in caplog.text
    assert "has been dropped" in caplog.text


def test_every_delivered_user_point_is_positive(negative_at_user_position):
    """The invariant, alongside a real positive point that must be kept."""
    fit, ds, geom = negative_at_user_position
    sol = fit_point_sources(fit.fit.inversion, ds, geom,
                            positions=[(0.6, -0.5), (0.0, 0.0)], retune=False)
    assert all(p.flux > 0 for p in sol.points)


def test_cube_channels_keep_their_sign(negative_at_user_position):
    """Per-channel fits at the MFS positions must not drop a negative
    amplitude: in a line-free channel that is noise, and dropping it would
    bias the spectrum."""
    fit, ds, geom = negative_at_user_position
    sol = fit_point_sources(fit.fit.inversion, ds, geom,
                            positions=[(0.6, -0.5)], retune=False,
                            refine=False, check_positions=False)
    assert len(sol.points) == 1 and sol.points[0].flux < 0


def test_with_no_points_left_the_prior_is_restored(negative_at_user_position):
    """A retuned scale belongs to the points it was tuned with."""
    fit, ds, geom = negative_at_user_position
    sol = fit_point_sources(fit.fit.inversion, ds, geom,
                            positions=[(0.6, -0.5)], retune=True,
                            retune_criterion="structure")
    assert sol.points == []
    assert sol.regularization_factor == 1.0
    assert sol.system.h_scale == 1.0


# --- user positions: resolved -----------------------------------------------

def test_a_resolved_user_component_is_flagged(caplog):
    """A compact Gaussian half a beam wide is not a point: keep it (the user
    asked for it) but say so. The prior is stiff enough that the point, not
    the mesh, takes the flux -- with a soft one the mesh describes the whole
    Gaussian, the point gets ~0, and there is nothing to flag."""
    import pyuvimage.pointsource as ps

    uvd, _, geom, _ = mock.make_field_dataset(
        fov_arcsec=3.0, mesh_n=20, n_vis=600, sigma_jy=3e-4, seed=3,
        extended=[(0.040, 0.70, (0.0, 0.0), 1.0, 0.0)], points=[],
        truth_on_mesh=False,
    )
    beam = _beam(_fit(uvd, geom)[1], geom)
    y, x = sky_to_grid(0.5, 0.4)
    uvd.data[0] += 0.02 * ps.gaussian_visibilities(
        uvd.flattened()[0], y, x, 0.5 * beam / 2.3548)
    fit, ds = _fit(uvd, geom, prior={"coefficient": 1e9, "scale": 0.25,
                                     "nu": 1.5})
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        sol = fit_point_sources(fit.fit.inversion, ds, geom,
                                positions=[(0.5, 0.4)], beam_fwhm=beam,
                                retune=False)
    assert len(sol.points) == 1 and sol.points[0].flux > 0
    assert "does not look unresolved" in caplog.text


# --- api wiring --------------------------------------------------------------

def _captured_kwargs(monkeypatch, **run_kwargs):
    import pyuvimage
    from pyuvimage import pointsource

    seen = []
    real = pointsource.fit_point_sources

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(pointsource, "fit_point_sources", spy)
    uvd, _, geom, _ = mock.make_demo_dataset(point_flux_jy=0.004)
    res = pyuvimage.run(uvd, fov=3.0, point_sources=[(0.0, 0.0)],
                        uncertainty_map=False, write=False, **run_kwargs)
    return seen, res


def test_run_retunes_on_the_criterion_it_searched_with(monkeypatch):
    seen, _ = _captured_kwargs(monkeypatch, criterion="structure",
                               reg="matern")
    assert seen and all(k["retune"] for k in seen)
    assert all(k["retune_criterion"] == "structure" for k in seen)


def test_a_fixed_coefficient_is_neither_retuned_nor_reoptimised(monkeypatch):
    """--lambda used to be re-optimised on the adaptive refit and could be
    retuned after it; a fixed coefficient is the user's."""
    seen, res = _captured_kwargs(monkeypatch, reg="adaptive",
                                 coefficient=3.0e4)
    assert seen and not any(k["retune"] for k in seen)
    assert res.parameters["source_prior"]["coefficient"] == pytest.approx(3.0e4)

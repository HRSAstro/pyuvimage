"""Analytic point components: conventions, accuracy, and false-positive control."""

import numpy as np
import pytest

from pyuvimage import fitting, mock
from pyuvimage.pointsource import (
    AugmentedSystem,
    detection_lattice,
    fit_point_sources,
    gaussian_visibilities,
    grid_to_sky,
    point_visibilities,
    sky_to_grid,
)

PRIOR = {"coefficient": 1e7, "scale": 0.25, "nu": 1.5}


def test_sky_grid_roundtrip_and_sign():
    """dRA runs opposite to the grid x axis; getting this wrong mirrors the field."""
    assert sky_to_grid(0.3, -0.4) == (-0.4, -0.3)
    assert grid_to_sky(*sky_to_grid(0.3, -0.4)) == (0.3, -0.4)


def test_point_visibilities_match_the_transformer():
    """The analytic phase must reproduce autoarray's DFT of a delta exactly.

    A sign error here would put the source on the wrong side of the field
    while still fitting the amplitude, so it is checked against the framework
    rather than against itself.
    """
    import autogalaxy as ag

    uvd, _, geom = mock.make_demo_dataset(n_vis=60, seed=5)
    uv, d, nz = uvd.flattened()
    ds = fitting.make_dataset(uv, d, nz, geom)

    image = np.zeros(geom.shape_native)
    iy, ix = 12, 20
    image[iy, ix] = 1.0
    n = geom.shape_native[0]
    y = ((n - 1) / 2 - iy) * geom.pixel_scale
    x = (ix - (n - 1) / 2) * geom.pixel_scale

    arr = ag.Array2D(values=image, mask=ds.real_space_mask)
    framework = np.asarray(ds.transformer.visibilities_from(image=arr))
    analytic = point_visibilities(np.asarray(ds.uv_wavelengths), y, x)
    assert np.allclose(framework, analytic, rtol=1e-8, atol=1e-10)


def test_gaussian_visibilities_reduce_to_a_point():
    rng = np.random.default_rng(0)
    uv = rng.normal(0, 3e5, (200, 2))
    uv = uv[np.hypot(uv[:, 0], uv[:, 1]) > 2e5]   # drop the shortest spacings
    assert np.allclose(
        gaussian_visibilities(uv, 0.2, -0.1, 0.0),
        point_visibilities(uv, 0.2, -0.1),
    )
    # a 1" Gaussian is resolved out entirely on these baselines
    assert np.max(np.abs(gaussian_visibilities(uv, 0.0, 0.0, 1.0))) < 1e-3


@pytest.fixture(scope="module")
def knot_case():
    return mock.make_extended_plus_compact_dataset(
        n_vis=400, mesh_n=24, compact_flux=0.012, compact_centre=(0.8, -0.7))


def _system(uvd, geom):
    uv, d, nz = uvd.flattened()
    ds = fitting.make_dataset(uv, d, nz, geom)
    fit = fitting.fit_dataset(ds, geom, reg_kind="matern", prior=PRIOR,
                              positive_only=False)
    return AugmentedSystem(fit.fit.inversion, ds), ds, fit


def test_adding_a_point_at_the_truth_improves_the_fit(knot_case):
    """The augmented solve must reproduce the mesh-only fit when empty, and
    beat it once a point is placed on the real knot."""
    uvd, _, geom, comps = knot_case
    system, ds, fit = _system(uvd, geom)
    chi2_mesh = system.chi_squared([])
    assert chi2_mesh == pytest.approx(fit.chi_squared, rel=1e-6)

    dec, ra = comps["compact"]["centre"]
    chi2_point = system.chi_squared([sky_to_grid(-ra, dec)])
    assert chi2_point < chi2_mesh - 25.0


def _beam(ds, geom):
    from pyuvimage import beam as beam_mod

    imager = beam_mod.DirtyImager(ds)
    b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
    return float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))


def test_significance_map_peaks_on_the_real_knot(knot_case):
    """The detector is a matched filter, not a residual-image peak finder.

    Taking the peak of the residual dirty image fails: the mesh fit has
    already been driven to chi^2 = N and absorbed much of the knot, so what
    is left is sidelobe structure.  The scan asks the right question --
    how far would chi^2 drop if a point were added here -- and must peak on
    the knot.
    """
    uvd, _, geom, comps = knot_case
    system, ds, fit = _system(uvd, geom)
    ys, xs = detection_lattice(geom)
    amp, sig = system.scan([], ys, xs)
    j = int(np.argmax(np.where(amp > 0, sig, -np.inf)))
    d_ra, d_dec = grid_to_sky(ys[j], xs[j])
    dec, ra = comps["compact"]["centre"]
    assert np.hypot(d_ra - (-ra), d_dec - dec) < 0.15
    # the scan is exact: it must reproduce the full augmented solve there
    _, a, _, cov = system.solve([(ys[j], xs[j])])
    assert a[0] == pytest.approx(amp[j], rel=1e-6)   # ridge terms differ at 1e-8
    assert abs(a[0]) / np.sqrt(cov[0, 0]) == pytest.approx(sig[j], rel=1e-6)
    # and adding the point really does reduce chi^2
    assert system.chi_squared([(ys[j], xs[j])]) < system.chi_squared([])


def test_recovers_a_known_point_source(knot_case):
    """Auto-detection finds the knot, at the right place, with the right flux."""
    uvd, _, geom, comps = knot_case
    _, ds, fit = _system(uvd, geom)
    sol = fit_point_sources(
        fit.fit.inversion, ds, geom, beam_fwhm=_beam(ds, geom),
    )
    assert len(sol.points) == 1
    p = sol.points[0]
    dec, ra = comps["compact"]["centre"]
    assert abs(p.d_ra - (-ra)) < 0.1 and abs(p.d_dec - dec) < 0.1
    assert p.flux == pytest.approx(comps["compact"]["flux"], rel=0.25)
    assert p.significance > 5.0
    assert p.flux > 0, "a negative amplitude is never a source"


def test_no_false_positives_on_a_smooth_source():
    """A smooth exponential has a central cusp the mesh cannot render.

    Deltas will happily absorb it -- five of them, stacked inside one beam --
    unless detection enforces a minimum separation and tests that each
    candidate is genuinely unresolved.  This is the regression test for that.
    """
    uvd, _, geom, _ = mock.make_extended_plus_compact_dataset(
        n_vis=400, mesh_n=24, compact_flux=0.0)
    _, ds, fit = _system(uvd, geom)
    sol = fit_point_sources(
        fit.fit.inversion, ds, geom, beam_fwhm=_beam(ds, geom),
    )
    assert sol.points == []


def test_user_supplied_position_is_kept_and_refined(knot_case):
    """A position the user asked for is never dropped, and is improved on."""
    uvd, _, geom, comps = knot_case
    _, ds, fit = _system(uvd, geom)
    dec, ra = comps["compact"]["centre"]
    guess = (-ra + 0.15, dec - 0.15)
    sol = fit_point_sources(fit.fit.inversion, ds, geom, positions=[guess])
    assert len(sol.points) == 1
    p = sol.points[0]
    assert p.user_supplied
    moved = np.hypot(p.d_ra - guess[0], p.d_dec - guess[1])
    err = np.hypot(p.d_ra - (-ra), p.d_dec - dec)
    assert moved > 0.02 and err < np.hypot(0.15, 0.15)


def test_retune_restores_the_discrepancy_criterion(knot_case):
    """With a point carrying the knot, chi^2/N falls below the target.

    Off by default (it stiffens the prior a long way), but when asked for it
    must actually hit the target.
    """
    from pyuvimage.pointsource import retune_regularization

    uvd, _, geom, comps = knot_case
    system, ds, fit = _system(uvd, geom)
    dec, ra = comps["compact"]["centre"]
    pos = [sky_to_grid(-ra, dec)]
    assert system.chi_squared(pos) < 0.9 * system.n_data
    factor = retune_regularization(system, pos, chi2_target=1.0)
    assert factor > 1.0
    assert system.chi_squared(pos) == pytest.approx(system.n_data, rel=0.02)

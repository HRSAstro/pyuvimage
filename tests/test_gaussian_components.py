"""Elliptical Gaussian components with `--points-only` (`pointonly.fit_with_gaussians`).

The sky is drawn in the image plane with the PA written out by hand (major
axis from north toward east), not through the code's own taper, so a wrong
orientation convention cannot hide in a round trip.
"""
import json

import numpy as np
import pytest

from pyuvimage import api, mock, pointonly
from pyuvimage.pointsource import point_visibilities, sky_to_grid

F0 = 230e9
GAUSS = dict(dra=0.3, ddec=0.2, flux=10e-3, maj=0.6, mnr=0.3, pa=30.0)
POINT = (-0.8, -0.5, 3e-3)
META = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0, "dish_diameter_m": 12.0}


def _sky_image(n=96, pix=0.05):
    iy, ix = np.mgrid[0:n, 0:n]
    dra = -(ix - (n - 1) / 2) * pix
    ddec = ((n - 1) / 2 - iy) * pix
    pa = np.radians(GAUSS["pa"])
    sa, sb = GAUSS["maj"] / 2.3548, GAUSS["mnr"] / 2.3548
    a = (dra - GAUSS["dra"]) * np.sin(pa) + (ddec - GAUSS["ddec"]) * np.cos(pa)   # along the major axis
    b = (dra - GAUSS["dra"]) * np.cos(pa) - (ddec - GAUSS["ddec"]) * np.sin(pa)
    img = np.exp(-a ** 2 / (2 * sa ** 2) - b ** 2 / (2 * sb ** 2))
    return img * GAUSS["flux"] / img.sum(), pix


@pytest.fixture(scope="module")
def uvd():
    uvw = mock.random_uv_coverage(4000, 600e3 * 299792458.0 / F0, F0, seed=5)
    img, pix = _sky_image()
    out = mock.simulate(img, pix, uvw, np.array([F0]), sigma_jy=1e-3, seed=7, meta=dict(META))
    uv = mock.uv_of(uvw, F0)
    out.data = out.data + POINT[2] * point_visibilities(uv, *sky_to_grid(*POINT[:2]))[None, :]
    return out


@pytest.fixture(scope="module")
def fitted(uvd):
    # starting values off the truth: round, too small, shifted
    return api.run(uvd, fov=4.0, points_only=True, write=False,
                   point_sources=[(-POINT[0] + 0.05, POINT[1] - 0.04)],
                   gaussians=[(-0.25, 0.25, 0.4)])


def test_the_gaussian_and_the_point_are_recovered(fitted):
    (g,), (p,) = fitted.fit.gaussians, fitted.points
    for est, true, err in ((g.flux, GAUSS["flux"], g.flux_error), (g.d_ra, GAUSS["dra"], g.d_ra_error),
                           (g.d_dec, GAUSS["ddec"], g.d_dec_error),
                           (g.fwhm_major, GAUSS["maj"], g.fwhm_major_error),
                           (g.fwhm_minor, GAUSS["mnr"], g.fwhm_minor_error),
                           (g.pa_deg, GAUSS["pa"], g.pa_error_deg),
                           (p.flux, POINT[2], p.flux_error), (p.d_ra, POINT[0], p.d_ra_error),
                           (p.d_dec, POINT[1], p.d_dec_error)):
        assert abs(est - true) < 4 * err
    assert g.pa_error_deg < 2 and g.fwhm_minor_error < 0.01
    assert fitted.fit.chi2_per_n == pytest.approx(1.0, abs=0.05)
    assert np.abs(fitted.residual).max() / fitted.rms < 5


def test_a_resolved_gaussian_says_so(fitted):
    (g,) = fitted.fit.gaussians
    assert g.resolved == "yes"
    assert g.delta_chi2_vs_point > 100 and g.delta_chi2_vs_line > 100
    assert g.fwhm_upper_limit is None and g.fwhm_minor_upper_limit is None


def test_a_point_fitted_as_a_gaussian_is_unresolved():
    """As imfit's `ispoint`: a true point is consistent with a point, and the
    size gets a 95% upper limit rather than a value with an error."""
    import test_points_only as T

    uvd = T._dataset(1e-3, 1)
    res = api.run(uvd, fov=4.0, points_only=True, write=False,
                  gaussians=[(-0.53, -0.31, 0.3)], point_sources=[(0.92, 0.74)])
    (g,) = res.fit.gaussians
    assert g.resolved == pointonly.POSSIBLY_UNRESOLVED == "possibly unresolved"
    assert g.delta_chi2_vs_point < pointonly.RESOLVED_DELTA_CHI2
    assert 0 < g.fwhm_upper_limit < 0.2 * res.beam.bmaj_arcsec
    assert g.as_dict()["resolved"] == "possibly unresolved"


def test_the_residual_structure_ratio_flags_a_missing_size(uvd, fitted, caplog):
    """A Gaussian fitted as a point leaves structure: the image-plane
    structure ratio says so, where chi^2/N barely moves."""
    import logging

    fq = fitted.parameters["fit_quality"]
    assert abs(fq["structure_ratio"] - 1) < 4 * fq["structure_ratio_noise_sigma"]
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        bad = api.run(uvd, fov=4.0, points_only=True, write=False,
                      point_sources=[(-GAUSS["dra"], GAUSS["ddec"]), (-POINT[0], POINT[1])])
    fq = bad.parameters["fit_quality"]
    assert fq["structure_ratio"] > 1 + 3 * fq["structure_ratio_noise_sigma"]
    assert "structure they do not describe" in caplog.text
    assert fq["structure_region_pixels"] < 0.5 * np.prod(bad.residual.shape)


def test_the_jacobian_is_the_derivative(uvd):
    from pyuvimage import fitting
    from pyuvimage.grids import resolve_geometry

    uv, d, nz = uvd.flattened()
    geom = resolve_geometry(4.0, float(np.max(np.hypot(uv[:, 0], uv[:, 1]))))
    ds = fitting.make_dataset(uv, d, nz, geom)
    backend = pointonly.VisibilityPoints(ds, fitting.imager_for(ds))
    model = pointonly.JointModel(backend, ["point", "gaussian"], [False, False])
    theta = model.pack([(3e-3, (-0.5, 0.8), None), (1e-2, (0.2, -0.3), (0.25, 0.12, 0.5))])
    J = model.jacobian(theta)
    for i in range(theta.size):
        h = 1e-6 * max(abs(theta[i]), 1e-3)
        e = np.zeros_like(theta)
        e[i] = h
        num = (model.residual(theta + e) - model.residual(theta - e)) / (2 * h)
        np.testing.assert_allclose(J[:, i], num, rtol=0, atol=1e-5 * np.abs(num).max() + 1e-12)


def test_images_hold_the_flux(fitted):
    geom = fitted.geometry
    pix = geom.pixel_scale
    model = pointonly.gaussian_model_image(geom.shape_native, pix, fitted.fit.gaussians)
    assert model.sum() == pytest.approx(fitted.fit.gaussians[0].flux, rel=1e-3)
    b = fitted.beam
    beam_pix = 2 * np.pi * (b.bmaj_arcsec / 2.3548) * (b.bmin_arcsec / 2.3548) / pix ** 2
    restored = pointonly.restore_gaussians(geom.shape_native, pix, fitted.fit.gaussians, b)
    assert restored.sum() / beam_pix == pytest.approx(fitted.fit.gaussians[0].flux, rel=1e-3)


def test_points_are_detected_alongside_only_when_asked(uvd):
    res = api.run(uvd, fov=4.0, points_only=True, write=False, point_sources=True,
                  gaussians=[(-0.25, 0.25, 0.4)])
    assert len(res.fit.gaussians) == 1 and len(res.points) == 1
    assert (res.points[0].d_ra, res.points[0].d_dec) == pytest.approx(POINT[:2], abs=0.01)
    alone = api.run(uvd, fov=4.0, points_only=True, write=False, gaussians=[(-0.25, 0.25, 0.4)])
    assert alone.points == [] and alone.fit.chi2_inflation > 1.2     # the point is left over


def test_products_and_the_command_line(uvd, tmp_path):
    from pyuvimage import cli

    path = uvd.write(tmp_path / "ds")
    assert cli.main(["fit", str(path), "--fov", "4", "--points-only", "--point=0.85,-0.54",
                     "--gaussian=-0.25,0.25,0.5,0.3,20", "--out", str(tmp_path / "out")]) == 0
    rec = json.loads((tmp_path / "out" / "point_sources.json").read_text())
    (g,) = rec["gaussians"]
    assert g["pa_deg"] == pytest.approx(GAUSS["pa"], abs=3)
    assert len(rec["covariance_order"]) == 3 + 6
    par = json.loads((tmp_path / "out" / "fit_parameters.json").read_text())
    assert par["model"] == "points and Gaussians" and not par["data"]["streamed"]


def test_gaussians_need_points_only(uvd):
    with pytest.raises(NotImplementedError, match="points_only"):
        api.run(uvd, fov=4.0, write=False, gaussians=[(0, 0)])
    with pytest.raises(NotImplementedError, match="in memory"):
        api.run(uvd, fov=4.0, write=False, points_only=True, gaussians=[(0, 0)], streaming=True)

"""`--points-only`: point components with no mesh and no prior (`pointonly`).

Two unresolved sources at off-pixel positions, simulated analytically. The fit
finds both, the fluxes equal weighted least squares at the fitted positions,
the statistical errors describe the scatter over noise realisations, the
streamed terms give the in-memory answer, and the products carry the points.
"""
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from pyuvimage import api, mock
from pyuvimage import streaming as stm
from pyuvimage.pointsource import point_visibilities, sky_to_grid
from pyuvimage.uvdata import UVData

F0 = 230e9
FOV = 4.0
TRUTH = [(0.53, -0.31, 5e-3), (-0.92, 0.74, 1.5e-3)]     # dRA, dDec [arcsec], Jy
META = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0, "dish_diameter_m": 12.0}


def _dataset(sigma, seed, truth=TRUTH, n_vis=3000):
    uvw = mock.random_uv_coverage(n_vis, 400e3 * 299792458.0 / F0, F0, seed=3)
    uv = mock.uv_of(uvw, F0)
    vis = sum(a * point_visibilities(uv, *sky_to_grid(r, d)) for r, d, a in truth)
    rng = np.random.default_rng(seed)
    data = vis + rng.normal(0, sigma, vis.size) + 1j * rng.normal(0, sigma, vis.size)
    return UVData(uvw=uvw, frequencies=np.array([F0]), data=data[None, :],
                  noise=np.full((1, vis.size), sigma * (1 + 1j)), meta=dict(META))


def _matched(points):
    return sorted(points, key=lambda p: -p.flux)


@pytest.fixture(scope="module")
def fitted():
    uvd = _dataset(1e-3, 1)
    return uvd, api.run(uvd, fov=FOV, points_only=True, streaming=False, write=False)


def test_both_points_are_found_where_they_are(fitted):
    _, res = fitted
    pts = _matched(res.points)
    assert len(pts) == 2
    for p, (r, d, a) in zip(pts, TRUTH):
        assert abs(p.flux - a) < 4 * p.flux_error
        assert abs(p.d_ra - r) < 4 * p.d_ra_error and abs(p.d_dec - d) < 4 * p.d_dec_error
        assert p.d_ra_error < 0.01 and p.significance > 50
    assert res.fit.chi2_per_n == pytest.approx(1.0, abs=0.05)
    assert res.fit.chi2_inflation == pytest.approx(1.0, abs=0.05)
    assert np.abs(res.residual).max() / res.rms < 5


def test_the_fluxes_are_weighted_least_squares_at_the_fitted_positions(fitted):
    uvd, res = fitted
    uv, d, n = uvd.flattened()
    P = np.stack([point_visibilities(uv, *sky_to_grid(p.d_ra, p.d_dec)) for p in res.points], 1)
    A = np.vstack([P.real / n.real[:, None], P.imag / n.imag[:, None]])
    b = np.concatenate([d.real / n.real, d.imag / n.imag])
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose([p.flux for p in res.points], a, rtol=1e-8)


def test_the_errors_describe_the_scatter():
    """Pulls (estimate - truth) / error over noise realisations: unit width."""
    pulls = []
    for seed in range(16):
        res = api.run(_dataset(3e-3, 100 + seed), fov=FOV, points_only=True,
                      streaming=False, write=False, pb_correction=False)
        pts = _matched(res.points)
        assert len(pts) == 2
        pulls.append([(q - t) / e for p, (r, d, a) in zip(pts, TRUTH)
                      for q, t, e in ((p.flux, a, p.flux_error), (p.d_ra, r, p.d_ra_error),
                                      (p.d_dec, d, p.d_dec_error))])
    width = np.std(np.ravel(pulls))
    assert 0.75 < width < 1.3
    assert abs(np.mean(pulls)) < 0.3


def test_fixed_positions_and_a_missing_point_inflate_the_flux_error(fitted):
    uvd, res = fitted
    one = api.run(uvd, fov=FOV, points_only=True, streaming=False, write=False,
                  point_sources=[(-0.53, -0.31)], point_refine=False)   # image (x, y)
    p = one.points[0]
    assert p.user_supplied and p.d_ra_error is None and not one.fit.positions_fitted
    assert (p.d_ra, p.d_dec) == pytest.approx((0.53, -0.31))
    # the second source is left in the residual: chi^2/N > 1 and the error grows by its root
    assert one.fit.chi2_inflation > 1.5
    raw = p.flux_error / np.sqrt(one.fit.chi2_inflation)
    assert raw == pytest.approx(_matched(res.points)[0].flux_error, rel=0.05)


@pytest.mark.skipif(not stm.nufftax_available(), reason="streamed points need nufftax")
def test_the_streamed_terms_give_the_in_memory_fit(fitted, tmp_path):
    uvd, res = fitted
    path = uvd.write(tmp_path / "ds")
    s = api.run(str(path), fov=FOV, points_only=True, out=tmp_path / "out",
                kernel_cache=str(tmp_path / "cache"))
    assert s.parameters["data"]["streamed"]
    for p, q in zip(_matched(res.points), _matched(s.points)):
        assert (q.d_ra, q.d_dec) == pytest.approx((p.d_ra, p.d_dec), abs=1e-4)
        assert q.flux == pytest.approx(p.flux, rel=1e-6)
        assert q.d_ra_error == pytest.approx(p.d_ra_error, rel=1e-3)
    assert s.fit.chi_squared == pytest.approx(res.fit.chi_squared, rel=1e-8)


def test_the_products_carry_the_points(tmp_path):
    uvd = _dataset(1e-3, 1)
    res = api.run(uvd, fov=FOV, points_only=True, streaming=False, out=tmp_path)
    for name in ("model.fits", "dirty_image.fits", "model_reconvolved.fits",
                 "residual.fits", "fit_parameters.json", "point_sources.json",
                 "summary.png"):
        assert (tmp_path / name).exists(), name
    rec = json.loads((tmp_path / "fit_parameters.json").read_text())
    assert rec["model"] == "points only" and rec["point_sources"]["n_points"] == 2
    model = fits.getdata(tmp_path / "model.fits")
    assert model.sum() == pytest.approx(sum(p.flux for p in res.points), rel=1e-5)
    # RA / Dec: where the written WCS puts the restored peak of the brightest point
    hdr = fits.getheader(tmp_path / "model_reconvolved.fits")
    rest = fits.getdata(tmp_path / "model_reconvolved.fits")
    p = _matched(res.points)[0]
    row, col = np.unravel_index(np.argmax(rest), rest.shape)
    sky = WCS(hdr).celestial.pixel_to_world(col, row)
    pix = hdr["CDELT2"] * 3600
    assert abs(sky.ra.deg - p.ra_deg) * 3600 * np.cos(np.radians(2.0)) < pix
    assert abs(sky.dec.deg - p.dec_deg) * 3600 < pix
    assert p.ra_deg > META["phase_centre_ra_deg"]       # +dRA is east
    assert p.primary_beam < 1 and p.flux_pbcor > p.flux


def test_the_command_line(tmp_path):
    from pyuvimage import cli

    path = _dataset(1e-3, 1).write(tmp_path / "ds")
    assert cli.main(["fit", str(path), "--fov", "4", "--points-only", "--point=-0.53,-0.31",
                     "--fixed-points", "--out", str(tmp_path / "out"),
                     "--kernel-cache", str(tmp_path / "cache")]) == 0
    rec = json.loads((tmp_path / "out" / "point_sources.json").read_text())
    (p,) = rec["points"]
    assert (p["d_ra"], p["d_dec"]) == pytest.approx((0.53, -0.31))
    assert p["position"] == "user" and p["d_ra_error"] is None


def test_cube_mode_is_refused():
    with pytest.raises(NotImplementedError, match="MFS"):
        api.run(_dataset(1e-3, 1), fov=FOV, points_only=True, mode="cube", write=False)

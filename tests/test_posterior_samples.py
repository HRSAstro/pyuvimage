"""Posterior draws (`--posterior-samples`): their scatter is the posterior
covariance, they round-trip through posterior_samples.npz onto model.fits's
grid, and a cube run writes one set per channel."""

import numpy as np
import pytest
from astropy.io import fits

import pyuvimage
from pyuvimage import fitting, mock
from pyuvimage.posterior import PosteriorSamples


@pytest.fixture(scope="module")
def fit():
    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=250, mesh_n=16, sigma_jy=1e-3, seed=42)
    uv, d, nz = uvd.flattened()
    return fitting.fit_dataset(
        fitting.make_dataset(uv, d, nz, geom), geom, reg_kind="matern",
        prior={"coefficient": 1e7, "scale": 0.3, "nu": 1.5}, positive_only=False)


def _images(sf, post):
    shape = tuple(sf.geometry.shape_native)
    img = np.zeros((post["delta"].shape[0], shape[0] * shape[1]))
    img[:, post["unmasked"]] = (post["mapping"] @ post["delta"].T).T
    return img.reshape(-1, *shape)


def test_the_draws_scatter_as_the_posterior(fit):
    """Per pixel, sqrt(diag(M C M^T)) -- the undeblocked uncertainty map."""
    post = fit.posterior_deltas(4000, seed=1)
    sd = _images(fit, post).std(axis=0)
    ref = np.asarray(fit.model_uncertainty)
    on = ref > 0.2 * ref.max()
    assert np.median(sd[on] / ref[on]) == pytest.approx(1.0, abs=0.03)


def test_the_draws_carry_the_correlations(fit):
    """A sum over a region: the draws' scatter is the aperture error, not
    the per-pixel errors in quadrature (which overstate it here)."""
    post = fit.posterior_deltas(4000, seed=2)
    imgs = _images(fit, post)
    n = imgs.shape[1]
    region = np.zeros((n, n), bool)
    region[n // 2 - 3:n // 2 + 3, n // 2 - 3:n // 2 + 3] = True
    mc = imgs[:, region].sum(axis=1).std()
    exact = fit.aperture_uncertainty(region)
    quad = np.sqrt(np.sum(np.asarray(fit.model_uncertainty)[region] ** 2))
    assert mc == pytest.approx(exact, rel=0.05)
    assert abs(quad - exact) > 0.1 * exact


def test_the_bayesian_draws_use_the_marginal_covariance(fit):
    post = fit.posterior_deltas(3000, mode="bayesian", seed=3)
    sd = _images(fit, post).std(axis=0)
    ref = np.asarray(fit._propagate(fit.bayesian_posterior["cov"]))
    on = ref > 0.2 * ref.max()
    assert np.median(sd[on] / ref[on]) == pytest.approx(1.0, abs=0.04)


@pytest.mark.parametrize("streaming", [False, True])
def test_a_cube_run_writes_draws_on_model_fits_grid(tmp_path, streaming):
    from pyuvimage.grids import resolve_geometry

    fov, m, f0 = 3.0, 12, 230e9
    freqs = f0 * (1.0 + 1e-3 * np.arange(2))
    uvw = mock.random_uv_coverage(600, m / (2.0 * fov / 206265.0) * 299792458.0 / f0,
                                  f0, seed=7)
    geom = resolve_geometry(fov, float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max()
                                       / 299792458.0), mesh_shape=(m, m))
    # off-centre and asymmetric, so a flipped or transposed map would show
    truth = mock.exponential_image(geom.shape_native, geom.pixel_scale, flux_jy=0.05,
                                   r_eff_arcsec=0.3, centre_arcsec=(0.6, -0.3))
    uvd = mock.simulate(truth, geom.pixel_scale, uvw, freqs, sigma_jy=3e-4, seed=8)
    data = uvd.write(tmp_path / "ds") if streaming else uvd
    out = tmp_path / "o"
    # the adaptive prior makes the error follow the off-centre source
    pyuvimage.run(data, fov=fov, out=out, mode="cube", mesh_shape=(m, m), reg="adaptive",
                  criterion="discrepancy", pb_correction=False, streaming=streaming,
                  uncertainty="statistical", posterior_samples=300)
    post = PosteriorSamples(out / "posterior_samples.npz")
    model = fits.getdata(out / "model.fits")
    unc = fits.getdata(out / "uncertainty.fits")
    assert post.planes == [0, 1] and post.n_samples == 300 and post.mode == "statistical"
    assert post.shape == model.shape[1:]
    from pyuvimage.fitting import _deblock

    for c in range(2):
        # uncertainty.fits is the deblocked map: compare like with like
        sd = _deblock(post.offsets(c).std(axis=0), 2)
        r = lambda a: np.corrcoef(sd.ravel(), a.ravel())[0, 1]
        # same pixels, same orientation: a flip would decorrelate them
        assert r(unc[c]) > 0.6
        assert r(unc[c]) > r(np.flipud(unc[c])) + 0.1
        assert r(unc[c]) > r(np.fliplr(unc[c])) + 0.1
    one = post.draw(5, model)
    assert one.shape == model.shape
    np.testing.assert_allclose(one[1] - model[1], post.offsets(1, [5])[0], atol=1e-9)


def test_no_draws_unless_asked(tmp_path, fit):
    uvd = mock.make_demo_dataset(n_vis=600, mesh_n=12, fov_arcsec=3.0, seed=1)[0]
    pyuvimage.run(uvd, fov=3.0, out=tmp_path, mesh_shape=(12, 12), reg="matern",
                  criterion="discrepancy", pb_correction=False, uncertainty="statistical")
    assert not (tmp_path / "posterior_samples.npz").exists()


def _masked_run(tmp_path, mode, **kw):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from pyuvimage import mask
    from pyuvimage.grids import resolve_geometry

    fov, m, f0 = 3.0, 16, 230e9
    uvw = mock.random_uv_coverage(1500, m / (2.0 * fov / 206265.0) * 299792458.0 / f0,
                                  f0, seed=7)
    geom = resolve_geometry(fov, float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * f0
                                       / 299792458.0), mesh_shape=(m, m))
    truth = mock.exponential_image(geom.shape_native, geom.pixel_scale, flux_jy=0.05,
                                   r_eff_arcsec=0.4)
    meta = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0, "dish_diameter_m": 12.0}
    uvd = mock.simulate(truth, geom.pixel_scale, uvw, np.array([f0]), sigma_jy=3e-4,
                        seed=8, meta=meta)
    c = SkyCoord(150.0 * u.deg, 2.0 * u.deg)
    region = tmp_path / "c.crtf"
    region.write_text("#CRTFv0\ncircle [[%s, %s], 1.2arcsec] coord=J2000\n" % (
        c.ra.to_string(u.hour, sep=":", precision=4),
        c.dec.to_string(u.deg, sep=".", precision=3, alwayssign=True)))
    out = tmp_path / mode
    res = pyuvimage.run(uvd, fov=fov, out=out, mesh_shape=(m, m), pb_correction=False,
                        mesh_type="hilbert", mesh_points=120, emission_mask=str(region),
                        uncertainty=mode, posterior_samples=200, **kw)
    allowed = mask.load(region, res.products[0].model_image.shape, res.geometry.pixel_scale,
                        meta)
    return out, allowed, res


@pytest.mark.parametrize("mode", ["statistical", "bayesian"])
def test_a_mask_is_a_constraint_not_a_parameter(tmp_path, mode):
    """The zeroed ring (and positivity's active set) get no variance: the
    error outside the mask, where the model is held at zero, is zero, and
    the draws do not move there. Before, it read *larger* than inside."""
    out, allowed, res = _masked_run(tmp_path, mode)
    unc = np.flipud(fits.getdata(out / "uncertainty.fits"))
    assert np.nanmedian(unc[allowed]) > 0
    # zero outside, bar the first pixels past the edge that the triangles
    # straddling it interpolate into (as the model itself does)
    assert np.nanmedian(unc[~allowed]) == 0
    assert (unc[~allowed] > 0).mean() < 0.15
    post = PosteriorSamples(out / "posterior_samples.npz")
    sd = np.flipud(post.offsets(0).std(axis=0))
    assert np.median(sd[~allowed]) == 0 and (sd[~allowed] > 0).mean() < 0.15
    if mode == "bayesian":
        mb = np.flipud(fits.getdata(out / "model_bayesian.fits"))
        assert np.abs(mb[~allowed]).sum() < 0.01 * mb.sum()
        # the draws are centred on the marginal mean, not the delivered model
        model = np.flipud(fits.getdata(out / "model.fits"))
        centre = np.flipud(post.offsets(0).mean(axis=0)) + model
        assert np.abs(centre - mb).max() < 0.2 * np.abs(mb - model).max() + 1e-9
    terms = res.parameters["uncertainty"]["terms"]
    assert terms["chi2_inflation"] >= 1.0
    if mode == "statistical":
        assert 0 < terms["pinned_fraction"] < 1
    rec = res.parameters["source_flux"]
    assert rec["n_draws"] == 200 and rec["flux_error_draws_jy"] > 0


def test_chi2_inflation_scales_the_covariance():
    """A noise map that is too small leaves chi^2/N well above 1, and the
    covariance is (F+H)^-1 scaled by that factor; a good one leaves it alone."""
    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=250, mesh_n=16, sigma_jy=1e-3, seed=42)
    uv, d, nz = uvd.flattened()
    prior = {"coefficient": 1e7, "scale": 0.3, "nu": 1.5}
    ok = fitting.fit_dataset(fitting.make_dataset(uv, d, nz, geom), geom, reg_kind="matern",
                             prior=prior, positive_only=False)
    bad = fitting.fit_dataset(fitting.make_dataset(uv, d, nz / 2.0, geom), geom,
                              reg_kind="matern", prior=prior, positive_only=False)
    assert ok.chi2_inflation == pytest.approx(1.0, abs=0.25)
    assert bad.chi2_inflation > 2.0
    plain = np.linalg.inv(bad.system.F + bad.regularization_matrix)
    np.testing.assert_allclose(bad.posterior_covariance, plain * bad.chi2_inflation, rtol=1e-8)
    assert bad.free.all()

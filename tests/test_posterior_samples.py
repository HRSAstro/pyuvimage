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

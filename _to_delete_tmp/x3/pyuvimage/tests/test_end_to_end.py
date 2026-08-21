"""End-to-end regression tests on simulated data (slow-ish: ~1-2 min)."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import pyuvimage
from pyuvimage import mock
from pyuvimage.fitting import make_dataset
from pyuvimage.grids import resolve_geometry


@pytest.fixture(scope="module")
def demo():
    uvd, truth, geom, _ = mock.make_demo_dataset(n_vis=250, seed=2)
    return uvd, truth, geom


def test_transformer_adjoint(demo):
    """image_from must be the adjoint of visibilities_from:
    <R x, y> == <x, R^T y> (the load-bearing assumption of the whole fit)."""
    import autogalaxy as ag

    uvd, truth, geom = demo
    uv, d, n = uvd.flattened()
    ds = make_dataset(uv, d, n, geom, transformer="dft")
    tr = ds.transformer
    rng = np.random.default_rng(0)
    n_pix = int(np.sum(~ds.real_space_mask))
    x = rng.normal(size=n_pix)
    y = rng.normal(size=len(d)) + 1j * rng.normal(size=len(d))
    mask = ds.real_space_mask
    x_arr = ag.Array2D(values=x, mask=mask)
    Rx = np.asarray(tr.visibilities_from(image=x_arr))
    Rty = np.asarray(tr.image_from(visibilities=ag.Visibilities(y)))
    lhs = np.real(np.vdot(Rx, y))
    rhs = float(np.dot(x, Rty))
    assert lhs == pytest.approx(rhs, rel=1e-6)


def test_full_run_recovers_truth(demo, tmp_path):
    uvd, truth, geom = demo
    res = pyuvimage.run(uvd, fov=3.0, mesh_shape=geom.mesh_shape, out=tmp_path / "out")
    p = res.products[0]

    # The fit must reach the noise level without going through it. On these
    # small mocks the model has more pixels than data points, so the residual
    # *map* can sit below the image rms even at chi^2/N ~ 1 (a few
    # visibilities carry the chi^2 budget) -- chi^2 itself is the meaningful
    # check. See the "under-constrained fits" note in the README.
    n_data = 2 * uvd.n_vis
    assert 0.3 < p.chi_squared / n_data < 3.0
    # flux within 25%
    assert np.nansum(p.model_mesh) == pytest.approx(truth.sum(), rel=0.25)
    # morphology -- compared on the mesh grid, which is where the mock's
    # truth lives (building it on the finer product grid would put structure
    # in it that the model provably cannot represent)
    mm = np.nan_to_num(p.model_mesh)
    assert mm.shape == truth.shape
    corr = np.corrcoef(mm.ravel(), truth.ravel())[0, 1]
    assert corr > 0.85
    # peak position within one mesh pixel
    t = np.unravel_index(np.argmax(truth), truth.shape)
    m = np.unravel_index(np.argmax(mm), mm.shape)
    assert abs(t[0] - m[0]) <= 1 and abs(t[1] - m[1]) <= 1

    # products on disk with WCS + beam + rms headers
    for name in ["model.fits", "dirty_image.fits", "residual.fits",
                 "model_reconvolved.fits", "pb.fits", "model_pbcor.fits",
                 "model_reconvolved_pbcor.fits"]:
        path = tmp_path / "out" / name
        assert path.exists(), name
        h = fits.getheader(path)
        assert h["CTYPE1"] == "RA---SIN"
        WCS(h)  # parses
    h = fits.getheader(tmp_path / "out" / "residual.fits")
    assert h["RMS"] > 0
    assert "BMAJ" in h
    hc = fits.getheader(tmp_path / "out" / "model_reconvolved.fits")
    assert hc["BUNIT"] == "Jy/beam"
    hm = fits.getheader(tmp_path / "out" / "model.fits")
    assert hm["BUNIT"] == "Jy/pixel"


def test_dirty_rms_analytic_vs_empirical(demo):
    from pyuvimage.beam import DirtyImager

    uvd, truth, geom = demo
    uv, d, n = uvd.flattened()
    ds = make_dataset(uv, d, n, geom, transformer="dft")
    im = DirtyImager(ds)
    assert im.rms == pytest.approx(im.rms_empirical(), rel=0.5)


def test_default_prior_is_gibbs(tmp_path):
    """The default is the non-stationary Gibbs prior (best on compact
    emission); its correlation length still defaults to the beam."""
    uvd, truth, geom, _ = mock.make_demo_dataset(n_vis=250, seed=5)
    res = pyuvimage.run(
        uvd, fov=3.0, mesh_shape=geom.mesh_shape, out=tmp_path / "p",
        write=False
    )
    prior = res.parameters["source_prior"]
    assert prior["regularization"] == "gibbs"
    assert prior["optimised"] is True
    beam = res.products[0].beam
    expected = np.sqrt(beam.bmaj_arcsec * beam.bmin_arcsec)
    assert prior["scale"] == pytest.approx(expected, rel=0.05)


def test_no_overfitting_at_high_snr(tmp_path):
    """Regression: with few visibilities and very high S/N, the old evidence
    criterion drove chi^2 -> 0, absorbing noise and dirty-beam sidelobes into
    the model and leaving a blank residual map.  The discrepancy criterion
    must keep residuals noise-like."""
    uvd, truth, geom, _ = mock.make_demo_dataset(n_vis=200, sigma_jy=1e-5, seed=11)
    res = pyuvimage.run(
        uvd, fov=3.0, mesh_shape=geom.mesh_shape, out=tmp_path / "hs",
        write=False
    )
    p = res.products[0]
    n_data = 2 * uvd.n_vis
    assert p.chi_squared / n_data > 0.3, "model fitted below the noise floor"
    # the model must not be dominated by structure away from the source
    # (dirty-beam sidelobes absorbed into the model)
    on = truth > 0.05 * truth.max()
    mm = np.nan_to_num(p.model_mesh)
    assert np.nansum(np.abs(mm[~on])) < 0.6 * np.nansum(np.abs(mm))


def test_products_share_one_grid(tmp_path):
    uvd, truth, geom, _ = mock.make_demo_dataset(n_vis=200, seed=3)
    res = pyuvimage.run(
        uvd, fov=3.0, mesh_shape=geom.mesh_shape, out=tmp_path / "grid"
    )
    shapes = {
        fits.getdata(tmp_path / "grid" / f).shape
        for f in ["model.fits", "dirty_image.fits", "model_reconvolved.fits",
                  "residual.fits", "model_pbcor.fits", "pb.fits"]
    }
    assert len(shapes) == 1, f"products on different grids: {shapes}"
    scales = {
        round(abs(fits.getheader(tmp_path / "grid" / f)["CDELT1"]) * 3600, 8)
        for f in ["model.fits", "dirty_image.fits", "model_reconvolved.fits",
                  "residual.fits"]
    }
    assert len(scales) == 1
    # every product shares one pixel scale, but the model *mesh* is coarser
    # than that grid -- otherwise A^T W r = H s and the residual map stops
    # being a diagnostic of the data misfit
    assert res.geometry.pixel_scale < res.geometry.mesh_pixel_scale


def test_cube_mode(tmp_path):
    """Two-channel cube: per-channel images recover per-channel fluxes."""
    fov, f0 = 3.0, 230e9
    freqs = np.array([f0, f0 * 1.001])
    mesh_n = 24
    uv_max = mesh_n / (2.0 * (fov / 206265.0))
    uvw = mock.random_uv_coverage(200, uv_max * 299792458.0 / f0, f0, seed=7)
    geom = resolve_geometry(
        fov,
        float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max() / 299792458.0),
        mesh_shape=(20, 20),
    )
    truth = mock.exponential_image(
        geom.shape_native, geom.pixel_scale, flux_jy=0.05, r_eff_arcsec=0.35
    )
    uvd = mock.simulate(truth, geom.pixel_scale, uvw, freqs, sigma_jy=3e-4,
                        seed=8, meta={"dish_diameter_m": 12.0})
    res = pyuvimage.run(
        uvd, fov=fov, mode="cube", mesh_shape=(20, 20), out=tmp_path / "cube"
    )
    assert len(res.products) == 2
    for p in res.products:
        assert np.nansum(p.model_mesh) == pytest.approx(0.05, rel=0.25)
    cube = fits.getdata(tmp_path / "cube" / "model.fits")
    assert cube.ndim == 3 and cube.shape[0] == 2
    h = fits.getheader(tmp_path / "cube" / "model.fits")
    assert h["CTYPE3"] == "FREQ"


def test_envelope_prior_tracks_an_offset_source(tmp_path):
    """End-to-end check of the envelope prior's automatic placement: the
    envelope must land on the real source, not the phase centre. This is the
    regression test for the (y, x) sign conventions between the native image
    array and the mesh grid."""
    fov, f0 = 4.0, 230e9
    uv_max = 32 / (2 * (fov / 206265.0))
    uvw = mock.random_uv_coverage(220, uv_max * 299792458.0 / f0, f0, seed=13)
    geom = resolve_geometry(
        fov,
        float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * f0 / 299792458.0),
        mesh_shape=(24, 24),
    )
    true_centre = (0.6, 0.3)  # (y, x) arcsec
    truth = mock.exponential_image(
        geom.shape_native, geom.pixel_scale, flux_jy=0.05,
        r_eff_arcsec=0.25, centre_arcsec=true_centre,
    )
    uvd = mock.simulate(
        truth, geom.pixel_scale, uvw, np.array([f0]), sigma_jy=3e-4, seed=14,
        meta={"dish_diameter_m": 12.0},
    )
    res = pyuvimage.run(
        uvd, fov=fov, mesh_shape=(24, 24), reg="gaussian",
        out=tmp_path / "env", write=False,
    )
    prior = res.parameters["source_prior"]
    cy, cx = prior["envelope_centre_arcsec"]
    tol = 2 * geom.pixel_scale
    assert cy == pytest.approx(true_centre[0], abs=tol)
    assert cx == pytest.approx(true_centre[1], abs=tol)
    assert prior["envelope_fwhm_arcsec"] > 0
    # and the fit itself still recovers the source
    mi = np.nan_to_num(res.products[0].model_image)
    assert np.corrcoef(mi.ravel(), truth.ravel())[0, 1] > 0.9


def test_adaptive_prior_reduces_the_central_residual(tmp_path):
    """A bright compact core in fainter emission leaves a strong residual at
    the peak under a single global smoothing length. The two-stage adaptive
    prior, which smooths less where a first-pass model is bright, must reduce
    it."""
    uvd, truth, geom = mock.make_multi_component_dataset(n_vis=600, mesh_n=32)
    kw = dict(fov=3.0, mesh_shape=(32, 32), write=False)
    plain = pyuvimage.run(uvd, out=tmp_path / "m", reg="matern", **kw)
    adapt = pyuvimage.run(uvd, out=tmp_path / "a", reg="adaptive", **kw)

    n = geom.shape_native[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) * geom.pixel_scale
    core = r < 0.3

    def central(res):
        return float(np.nanmax(np.abs(res.products[0].residual_sigma[core])))

    assert central(adapt) < central(plain)
    # NB the model is block-replicated onto the finer product grid, so a
    # pixel-wise correlation against a smooth truth saturates around 0.9
    mi = np.nan_to_num(adapt.products[0].model_image)
    assert np.corrcoef(mi.ravel(), truth.ravel())[0, 1] > 0.85


def test_residual_map_is_not_structurally_nulled(tmp_path):
    """Regression for a real trap: if the model mesh spans the same grid the
    residual dirty image is computed on, the normal equations force
    A^T W r = H s, so the residual map collapses towards zero wherever the
    prior is weak -- it stops measuring the data misfit. The product grid must
    stay finer than the mesh, and the residual map must carry real signal."""
    uvd, truth, geom, _ = mock.make_demo_dataset(n_vis=250, seed=2)
    res = pyuvimage.run(
        uvd, fov=3.0, mesh_shape=geom.mesh_shape, out=tmp_path / "r",
        write=False,
    )
    p = res.products[0]
    assert res.geometry.pixel_scale < res.geometry.mesh_pixel_scale
    resid = np.asarray(p.residual_sigma)
    inside = np.abs(resid) > 0
    assert np.nanstd(resid[inside]) > 0.05, "residual map is structurally blank"


def test_uncertainty_sampling_formula_matches_monte_carlo(tmp_path):
    """The estimator covariance (F+H)^-1 F (F+H)^-1 must match the actual
    scatter of the model over noise realisations. This is the one uncertainty
    claim that can be checked from first principles, so it is checked."""
    import autogalaxy as ag
    from pyuvimage import fitting

    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=250, mesh_n=16, sigma_jy=1e-3, seed=41)
    uv, d, nz = uvd.flattened()
    prior = {"coefficient": 1e7, "scale": 0.3, "nu": 1.5}
    sf = fitting.fit_dataset(
        fitting.make_dataset(uv, d, nz, geom), geom, reg_kind="matern",
        prior=prior, positive_only=False)
    analytic = sf.model_uncertainty_sampling

    mask_t = ag.Mask2D.all_false(shape_native=truth.shape,
                                 pixel_scales=geom.mesh_pixel_scale)
    v_true = np.asarray(ag.TransformerDFT(
        uv_wavelengths=uv, real_space_mask=mask_t
    ).visibilities_from(image=ag.Array2D(values=truth, mask=mask_t)))

    rng = np.random.default_rng(3)
    imgs = []
    for _ in range(12):
        dk = v_true + rng.normal(0, nz.real) + 1j * rng.normal(0, nz.imag)
        sk = fitting.fit_dataset(
            fitting.make_dataset(uv, dk, nz, geom), geom, reg_kind="matern",
            prior=prior, positive_only=False)
        imgs.append(sk.model_image)
    mc = np.array(imgs).std(axis=0, ddof=1)

    ratio = float(np.median(mc / np.maximum(analytic, 1e-30)))
    # 12 realisations give ~21% sampling error on each std
    assert 0.7 < ratio < 1.35, f"sampling uncertainty is off by {ratio:.2f}x"


def test_total_uncertainty_map(tmp_path):
    """One map, built from two terms, with the checkerboard removed.

    The delivered map is sqrt(statistical^2 + systematic^2) with the
    mesh/image block pattern replaced by its upper envelope. Checked on a
    single fit so the terms are all from the same prior.
    """
    from pyuvimage import fitting
    from pyuvimage.fitting import _block_contrast

    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=250, mesh_n=16, sigma_jy=1e-3, seed=42)
    uv, d, nz = uvd.flattened()
    sf = fitting.fit_dataset(
        fitting.make_dataset(uv, d, nz, geom), geom, reg_kind="matern",
        prior={"coefficient": 1e7, "scale": 0.3, "nu": 1.5},
        positive_only=False)

    stat = sf.model_uncertainty
    sys_ = sf.prior_systematic()
    total, terms = sf.model_uncertainty_total()

    assert np.all(sys_ >= 0)
    assert terms["systematic_median"] > 0, "prior strength changes nothing?"
    assert np.all(stat >= sf.model_uncertainty_sampling - 1e-12)

    raw = np.hypot(stat, sys_)
    ovs = geom.oversample
    assert _block_contrast(raw, ovs) > 0.05, "no checkerboard to remove?"
    assert _block_contrast(total, ovs) < 0.4 * _block_contrast(raw, ovs)
    # the envelope is conservative: it must not sit below the raw map overall
    assert np.nanmedian(total) >= np.nanmedian(raw)
    # and it is still the same order of magnitude, not a runaway
    assert np.nanmedian(total) < 2.0 * np.nanmedian(raw)


def test_only_one_uncertainty_map_is_written(tmp_path):
    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=250, mesh_n=16, sigma_jy=1e-3, seed=42)
    pyuvimage.run(uvd, fov=3.0, mesh_shape=(16, 16), out=tmp_path / "u",
                  reg="matern")
    assert (tmp_path / "u" / "uncertainty.fits").exists()
    assert (tmp_path / "u" / "snr.fits").exists()
    assert not (tmp_path / "u" / "uncertainty_noise.fits").exists()
    hdr = fits.getheader(tmp_path / "u" / "uncertainty.fits")
    assert hdr["ERRTYPE"].strip() == "total"
    assert hdr["ERRSTAT"] > 0 and hdr["ERRDEBL"]


def test_uncertainty_is_data_independent_and_prior_shaped():
    """Two properties of the posterior covariance that are easy to get wrong.

    (1) C = (F+H)^-1 contains no data, so with the prior held fixed the
        uncertainty map must be *identical* for completely different data.
        It therefore cannot respond to how bright the source is.
    (2) Its spatial structure comes from the prior: a stationary kernel gives
        a flat map (F and H are both translation-invariant), a non-stationary
        one varies. A flat matern map is correct, not a bug.
    """
    from pyuvimage import fitting

    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=250, mesh_n=16, sigma_jy=1e-3, seed=51)
    uv, d, nz = uvd.flattened()
    prior = {"coefficient": 1e7, "scale": 0.3, "nu": 1.5}

    def unc(data, kind="matern", env=None):
        return fitting.fit_dataset(
            fitting.make_dataset(uv, data, nz, geom), geom, reg_kind=kind,
            prior=prior, positive_only=False, envelope=env,
        ).model_uncertainty

    rng = np.random.default_rng(2)
    other = d + rng.normal(0, nz.real) + 1j * rng.normal(0, nz.imag)
    u1, u2 = unc(d), unc(other)
    assert np.allclose(u1, u2, rtol=0, atol=0), "posterior depends on the data"

    # a stationary prior is flat in the interior (sample mesh nodes only, to
    # skip the node/interpolated checkerboard)
    n = u1.shape[0]
    sel = np.zeros_like(u1, dtype=bool)
    sel[n // 4:3 * n // 4:2, n // 4:3 * n // 4:2] = True
    assert np.std(u1[sel]) / np.mean(u1[sel]) < 0.25

    # the non-stationary prior must vary far more than the stationary one
    b = fitting.fit_dataset(
        fitting.make_dataset(uv, d, nz, geom), geom, reg_kind="matern",
        prior=prior, positive_only=False).model_mesh_image.ravel()
    ug = unc(d, "gibbs", {"brightness": np.clip(b, 0, None), "floor": 1e-2,
                          "ell_floor": 0.25})
    spread = lambda a: float(np.percentile(a, 99) / np.percentile(a, 50))
    assert spread(ug) > spread(u1), "non-stationary prior shows no structure"

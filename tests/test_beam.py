import numpy as np
import pytest

from pyuvimage.beam import (
    SIGMA_TO_FWHM,
    BeamFit,
    fit_beam,
    gaussian_kernel,
    restore,
)


def _gauss(shape, cy, cx, sy, sx):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    return np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))


def test_fit_beam_recovers_gaussian():
    pix = 0.05
    beam = _gauss((64, 64), 31.5, 31.5, sy=3.0, sx=2.0)
    bf = fit_beam(beam, pixel_scale=pix)
    assert bf.bmaj_arcsec == pytest.approx(3.0 * SIGMA_TO_FWHM * pix, rel=0.02)
    assert bf.bmin_arcsec == pytest.approx(2.0 * SIGMA_TO_FWHM * pix, rel=0.02)


def test_restore_preserves_centring():
    """A point model must restore to a beam centred at the same pixel
    (regression test for the prototype's kernel-shift bug)."""
    pix = 0.05
    bf = BeamFit(bmaj_arcsec=0.3, bmin_arcsec=0.2, bpa_deg=30.0)
    model = np.zeros((65, 65))
    model[40, 22] = 1.0
    out = restore(model, np.zeros_like(model), bf, pix)
    assert np.unravel_index(np.argmax(out), out.shape) == (40, 22)


def test_restored_units_point_source():
    """1 Jy point source -> peak 1 Jy/beam after restore."""
    pix = 0.05
    bf = BeamFit(bmaj_arcsec=0.3, bmin_arcsec=0.3, bpa_deg=0.0)
    model = np.zeros((129, 129))
    model[64, 64] = 1.0  # 1 Jy in one pixel
    out = restore(model, np.zeros_like(model), bf, pix)
    assert out.max() == pytest.approx(1.0, rel=1e-3)


def test_kernel_flux_scale():
    """Restoring a broad uniform disc conserves surface brightness:
    Jy/pix * beam_area_pix = Jy/beam."""
    pix = 0.1
    bf = BeamFit(bmaj_arcsec=0.5, bmin_arcsec=0.5, bpa_deg=0.0)
    model = np.full((101, 101), 2.0)  # 2 Jy/pix uniform
    out = restore(model, np.zeros_like(model), bf, pix)
    centre = out[50, 50]
    assert centre == pytest.approx(2.0 * bf.beam_area_pixels(pix), rel=0.01)


# --------------------------------------------------------------------------
# Sign and centring conventions (1 Sep 2026)
# --------------------------------------------------------------------------
#
# Three things were wrong at once and hid each other: the restoring kernel was
# evaluated half a pixel off the index `fftconvolve(mode="same")` centres on
# (invisible on the odd grids the tests above use, present on every production
# grid, which is even); `fit_beam` returned the sky position angle negated
# (nothing checked the sign); and `restore_points` used the opposite y
# convention from `gaussian_kernel`, so points were painted with a mirrored
# beam. All three are now in one north-up expression.


def _sky_psf(n, pa_deg, smaj, smin):
    """An elliptical beam at `pa_deg` east of north in the native frame:
    row 0 is north, column increases with image +x, which is west."""
    cy = cx = (n - 1) // 2
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    north, east = (cy - yy), -(xx - cx)
    t = np.radians(pa_deg)
    maj = north * np.cos(t) + east * np.sin(t)
    mnr = -north * np.sin(t) + east * np.cos(t)
    return np.exp(-0.5 * ((maj / smaj) ** 2 + (mnr / smin) ** 2))


@pytest.mark.parametrize("pa", [-75.0, -30.0, 0.0, 20.0, 45.0, 60.0, 89.0])
def test_fit_beam_position_angle_is_east_of_north(pa):
    """The CASA convention `BPA` claims. +30 used to come back as -30."""
    bf = fit_beam(_sky_psf(129, pa, smaj=6.0, smin=3.0), pixel_scale=1.0)
    wrapped = ((bf.bpa_deg - pa + 90.0) % 180.0) - 90.0
    assert wrapped == pytest.approx(0.0, abs=0.1)
    assert bf.bmaj_arcsec == pytest.approx(6.0 * SIGMA_TO_FWHM, rel=0.01)
    assert bf.bmin_arcsec == pytest.approx(3.0 * SIGMA_TO_FWHM, rel=0.01)


def test_fit_beam_handles_the_other_axis_being_major():
    """The fitter may converge with sx > sy; the returned angle must still be
    that of the major axis, east of north."""
    bf = fit_beam(_sky_psf(129, 60.0, smaj=3.0, smin=6.0), pixel_scale=1.0)
    # the *major* axis here is the one at 60 + 90 = 150 == -30 east of north
    wrapped = ((bf.bpa_deg + 30.0 + 90.0) % 180.0) - 90.0
    assert wrapped == pytest.approx(0.0, abs=0.1)


def test_kernel_regenerated_from_the_fit_matches_the_beam():
    """fit_beam -> gaussian_kernel must round-trip, or the restored image is
    convolved with a beam that is not the one in the header."""
    psf = _sky_psf(128, 35.0, smaj=6.0, smin=3.0)   # even grid on purpose
    bf = fit_beam(psf, pixel_scale=1.0)
    kernel = gaussian_kernel(bf, 1.0, psf.shape)
    assert np.abs(kernel - psf).max() < 1e-3


def test_restore_points_and_gaussian_kernel_agree():
    """Points and extended emission must be restored with the same beam."""
    from types import SimpleNamespace

    from pyuvimage.pointsource import restore_points

    bf = BeamFit(bmaj_arcsec=6.0, bmin_arcsec=3.0, bpa_deg=35.0)
    shape = (129, 129)
    painted = restore_points(
        shape, 1.0, [SimpleNamespace(flux=1.0, d_ra=0.0, d_dec=0.0)], bf
    )
    np.testing.assert_allclose(painted, gaussian_kernel(bf, 1.0, shape), atol=1e-12)


@pytest.mark.parametrize("n", [48, 49, 64, 65])
def test_restore_does_not_shift_on_even_grids(n):
    """A delta at (20, 30) restores to a beam centred at (20, 30) -- to the
    centroid, not just the peak pixel, on even and odd grids alike."""
    img = np.zeros((n, n))
    img[20, 30] = 1.0
    out = restore(img, np.zeros_like(img), BeamFit(0.3, 0.3, 0.0), 0.05)
    yy, xx = np.mgrid[0:n, 0:n]
    assert (out * yy).sum() / out.sum() == pytest.approx(20.0, abs=1e-6)
    assert (out * xx).sum() / out.sum() == pytest.approx(30.0, abs=1e-6)


def test_dirty_image_is_normalised_by_the_weight_sum():
    """A 1 Jy point at the phase centre reads 1 Jy/beam, and `rms` is the rms
    of a dirty image of pure noise. Normalising by the sampled beam peak, half
    a pixel off on an even grid, made both false by 5-9%."""
    import autogalaxy as ag

    from pyuvimage import fitting
    from pyuvimage.beam import DirtyImager
    from pyuvimage.mock import make_sparse_test_dataset
    from pyuvimage.pointsource import point_visibilities

    uvd, _, geom, _ = make_sparse_test_dataset(n_vis=3000)
    uv, d, n = uvd.flattened()
    ds = fitting.make_dataset(uv, d, n, geom, ag.TransformerDFT, mask_shape="square")
    im = DirtyImager(ds)
    assert geom.shape_native[0] % 2 == 0, "the case that was wrong is the even grid"

    assert im._norm == pytest.approx(im.weights.sum())
    # the sampled beam peak is below 1 on an even grid, and that is correct
    assert 0.85 < im.dirty_beam.max() < 1.0

    point = point_visibilities(uv, 0.0, 0.0)          # 1 Jy at the phase centre
    img = np.asarray(im.dirty_image(point))
    ny, nx = img.shape
    # the four pixels around the phase centre bracket 1 Jy/beam from below,
    # by the same sampling factor as the beam peak
    centre4 = img[ny // 2 - 1:ny // 2 + 1, nx // 2 - 1:nx // 2 + 1]
    assert centre4.max() == pytest.approx(im.dirty_beam.max(), rel=1e-6)

    assert im.rms_empirical(n_draws=12) == pytest.approx(im.rms, rel=0.05)


def test_dirty_images_are_on_the_true_adjoint_scale_whatever_the_transformer():
    """A pynufft-backed transformer's raw `image_from` is 4 N_y N_x low and
    only `use_adjoint_scaling=True` corrects it. When `_norm` was the sampled
    beam peak the factor cancelled; with the analytic sum(w) it must be asked
    for explicitly -- on 9io9 the structure ratio read 1.4e-05 for a day
    because it was not (2 Sep 2026)."""
    import autogalaxy as ag

    from pyuvimage import fitting
    from pyuvimage.beam import DirtyImager
    from pyuvimage.mock import make_sparse_test_dataset
    from pyuvimage.pointsource import point_visibilities

    class ScaledDownDFT(ag.TransformerDFT):
        """Behaves like the vendored pynufft transformer: raw adjoint low by
        4 N_y N_x, exact only when asked."""

        def image_from(self, visibilities, use_adjoint_scaling=False, **kw):
            img = super().image_from(visibilities=visibilities)
            if use_adjoint_scaling:
                return img
            ny, nx = img.native.shape
            return type(img)(values=np.asarray(img) / (4.0 * ny * nx),
                             mask=img.mask)

    uvd, _, geom, _ = make_sparse_test_dataset(n_vis=3000)
    uv, d, n = uvd.flattened()
    ds = fitting.make_dataset(uv, d, n, geom, ScaledDownDFT, mask_shape="square")
    ds_ref = fitting.make_dataset(uv, d, n, geom, ag.TransformerDFT, mask_shape="square")
    im, ref = DirtyImager(ds), DirtyImager(ds_ref)

    np.testing.assert_allclose(im.dirty_beam, ref.dirty_beam, rtol=1e-10)
    point = point_visibilities(uv, 0.0, 0.0)
    np.testing.assert_allclose(im.dirty_image(point), ref.dirty_image(point), rtol=1e-10)
    assert np.asarray(im.dirty_image(point)).max() > 0.85     # ~1 Jy/beam, not 1e-5
    assert im.rms_empirical(n_draws=8) == pytest.approx(im.rms, rel=0.05)


def test_a_mis_scaled_adjoint_is_refused_at_construction():
    """The failure mode above must be loud and immediate, not a structure
    ratio of 1e-5 discovered hours into a fit."""
    import autogalaxy as ag

    from pyuvimage import fitting
    from pyuvimage.beam import DirtyImager
    from pyuvimage.mock import make_sparse_test_dataset

    class BrokenScaleDFT(ag.TransformerDFT):
        """A raw adjoint 1/(4 N_y N_x) low that ignores the scaling request."""

        def image_from(self, visibilities, **kw):
            img = super().image_from(visibilities=visibilities)
            ny, nx = img.native.shape
            return type(img)(values=np.asarray(img) / (4.0 * ny * nx), mask=img.mask)

    uvd, _, geom, _ = make_sparse_test_dataset(n_vis=1500)
    uv, d, n = uvd.flattened()
    ds = fitting.make_dataset(uv, d, n, geom, BrokenScaleDFT, mask_shape="square")
    with pytest.raises(RuntimeError, match="not on the plain mathematical scale"):
        DirtyImager(ds)

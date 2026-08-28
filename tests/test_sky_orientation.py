"""Which way up is the sky?

Every astrometry test before this one was a round trip: build a mock with
`mock.simulate`, image it, check the source comes back where it was put. That
can never catch a mirrored sky, because the generator and the imager share
whatever convention is in force. pyuvimage's images were mirrored in
declination for weeks and every one of those tests passed.

So these tests do not use `mock.simulate`. They write the visibilities of a
point source out by hand, from a forward model stated explicitly here, and ask
where pyuvimage puts it. The forward model is the one measured against CASA on
Ruby CO(7-6) -- see `uvdata.V_SIGN` -- namely, for the (u, v) as a measurement
set stores them,

    V(u, v) = flux * exp[+2 pi i (u * dRA + v * dDec)]

with dRA positive East and dDec positive North, both in radians on the sky.
If pyuvimage's convention is ever flipped again, this fails; a round-trip test
would not.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import pyuvimage
from pyuvimage import beam as beam_mod
from pyuvimage.uvdata import C_M_S, UVData

ARCSEC = np.pi / 180.0 / 3600.0
F0 = 230e9
# deliberately asymmetric, and in the quadrant that was coming out wrong:
# East and South, which is where Ruby's ring actually is.
D_RA, D_DEC = +2.0, -1.4


def _visibilities_of_a_point(uvw_m, frequency_hz, d_ra, d_dec, flux=0.05):
    """The forward model, written out. Stored-MS (u, v); dRA East, dDec North."""
    uv = np.asarray(uvw_m)[:, :2] * (frequency_hz / C_M_S)
    phase = 2.0 * np.pi * (
        uv[:, 0] * d_ra * ARCSEC + uv[:, 1] * d_dec * ARCSEC
    )
    return flux * np.exp(1j * phase)


@pytest.fixture(scope="module")
def point_dataset():
    rng = np.random.default_rng(4)
    n_vis = 4000
    # a filled uv disc out to 200 klambda -> ~1" resolution
    r = 200e3 * np.sqrt(rng.random(n_vis))
    th = rng.uniform(0, 2 * np.pi, n_vis)
    uv_l = np.column_stack((r * np.cos(th), r * np.sin(th)))
    uvw = np.zeros((n_vis, 3))
    uvw[:, :2] = uv_l * (C_M_S / F0)
    vis = _visibilities_of_a_point(uvw, F0, D_RA, D_DEC)
    sigma = 2e-4
    vis = vis + rng.normal(0, sigma, n_vis) + 1j * rng.normal(0, sigma, n_vis)
    return UVData(
        uvw=uvw,
        frequencies=np.array([F0]),
        data=vis[None, :],
        noise=np.full((1, n_vis), sigma + 1j * sigma),
        meta={"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0,
              "dish_diameter_m": 12.0},
    )


def test_the_wide_field_survey_finds_it_in_the_right_quadrant(point_dataset):
    """The function every `--image-centre` recommendation comes from."""
    uv, d, n = point_dataset.flattened()
    npix, fov = 96, 12.0
    img, _ = beam_mod.wide_field_dirty_image(uv, d, n, fov_arcsec=fov,
                                             n_pixels=npix)
    c = (np.arange(npix) - (npix - 1) / 2.0) * (fov / npix)
    row, col = np.unravel_index(int(np.argmax(img)), img.shape)
    x, y = c[col], c[::-1][row]          # +x right, +y North (row 0 is North)
    assert x == pytest.approx(-D_RA, abs=0.4), f"x {x:+.2f} for dRA {D_RA:+.2f}"
    assert y == pytest.approx(D_DEC, abs=0.4), f"y {y:+.2f} for dDec {D_DEC:+.2f}"


def test_the_written_fits_puts_it_at_the_right_declination(point_dataset, tmp_path):
    """The one that matters to anyone comparing with CLEAN: absolute sky
    position, read back through the file's own WCS."""
    pyuvimage.run(
        point_dataset, fov=8.0, out=tmp_path, reg="matern", coefficient=1e3,
        reg_scale=1.0, uncertainty_map=False, pb_correction=False,
        mask_shape="square",
    )
    with fits.open(tmp_path / "model_reconvolved.fits") as hdul:
        h, data = hdul[0].header, np.asarray(hdul[0].data, float)
    plane = data[0] if data.ndim == 3 else data
    w = WCS(h).celestial
    row, col = np.unravel_index(int(np.nanargmax(plane)), plane.shape)
    ra, dec = w.all_pix2world([[col, row]], 0)[0]
    ra0, dec0 = 150.0, 2.0
    d_ra = (ra - ra0) * 3600.0 * np.cos(np.radians(dec0))
    d_dec = (dec - dec0) * 3600.0
    assert d_ra == pytest.approx(D_RA, abs=0.5), f"dRA {d_ra:+.2f}"
    assert d_dec == pytest.approx(D_DEC, abs=0.5), f"dDec {d_dec:+.2f}"


def test_recentring_on_the_source_actually_lands_on_it(point_dataset, tmp_path):
    """`--image-centre` takes image (x, y). Pointing it at the source must
    centre the field on the source *and* leave the WCS telling the truth --
    the two can be wrong together and look right."""
    pyuvimage.run(
        point_dataset, fov=4.0, out=tmp_path, reg="matern", coefficient=1e3,
        reg_scale=1.0, uncertainty_map=False, pb_correction=False,
        mask_shape="square", image_centre=(-D_RA, D_DEC),
    )
    with fits.open(tmp_path / "model_reconvolved.fits") as hdul:
        h, data = hdul[0].header, np.asarray(hdul[0].data, float)
    plane = data[0] if data.ndim == 3 else data
    row, col = np.unravel_index(int(np.nanargmax(plane)), plane.shape)
    cy, cx = (plane.shape[0] - 1) / 2.0, (plane.shape[1] - 1) / 2.0
    assert abs(row - cy) <= 3 and abs(col - cx) <= 3, "not centred on the source"
    w = WCS(h).celestial
    ra, dec = w.all_pix2world([[col, row]], 0)[0]
    d_ra = (ra - 150.0) * 3600.0 * np.cos(np.radians(2.0))
    d_dec = (dec - 2.0) * 3600.0
    assert d_ra == pytest.approx(D_RA, abs=0.5)
    assert d_dec == pytest.approx(D_DEC, abs=0.5)


def test_the_mock_generator_uses_the_same_frame():
    """`mock.uv_of` and `UVData.uv_wavelengths` must not drift apart -- the
    gap between them is precisely the hole this bug lived in."""
    from pyuvimage import mock

    rng = np.random.default_rng(1)
    uvw = np.zeros((7, 3))
    uvw[:, :2] = rng.normal(0, 300.0, (7, 2))
    d = UVData(
        uvw=uvw, frequencies=np.array([F0]),
        data=np.zeros((1, 7), dtype=complex),
        noise=np.full((1, 7), 1e-4 + 1e-4j),
    )
    assert np.allclose(mock.uv_of(uvw, F0), d.uv_wavelengths(0))

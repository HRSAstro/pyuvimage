import numpy as np
import pytest

from pyuvimage.primary_beam import pb_correct, pb_fwhm_arcsec, primary_beam_map


def test_fwhm_alma_band6():
    # 12 m dish at 230 GHz: ~1.13 * lambda/D ~ 25.3"
    fwhm = pb_fwhm_arcsec(230e9, 12.0)
    lam = 299792458.0 / 230e9
    assert fwhm == pytest.approx(1.13 * lam / 12.0 * 206264.8, rel=1e-3)


def test_map_peak_and_fwhm():
    pb = primary_beam_map((201, 201), 0.25, 230e9, 12.0)
    assert pb[100, 100] == pytest.approx(1.0)
    fwhm = pb_fwhm_arcsec(230e9, 12.0)
    r_half_pix = fwhm / 2 / 0.25
    val = pb[100, 100 + int(round(r_half_pix))]
    assert val == pytest.approx(0.5, abs=0.03)


def test_pbcor_blanks_low_response():
    pb = primary_beam_map((101, 101), 1.0, 230e9, 12.0)
    img = np.ones_like(pb)
    out = pb_correct(img, pb, cutoff=0.5)
    assert np.isnan(out[0, 0])
    assert out[50, 50] == pytest.approx(1.0)


def test_the_beam_follows_the_pointing_not_the_image_centre():
    """The PB is set by the instrument: its peak is at the phase centre.

    `--image-centre` moves the grid, not the dish. With the image centred
    (y0, x0) = (+1.0", -0.5") from the phase centre -- north and east of it --
    the phase centre sits south and west of the grid centre on the native
    array: larger row (row 0 is north), larger column (+x is west), by the
    offset in pixels. Before this the PB peaked on the grid centre and every
    recentred pbcor product was corrected as if the source sat at PB = 1.
    """
    shape, ps = (201, 201), 0.25
    pb = primary_beam_map(shape, ps, 230e9, 12.0, image_centre_offset_arcsec=(1.0, -0.5))
    row, col = np.unravel_index(int(np.argmax(pb)), shape)
    assert (row, col) == (100 + 4, 100 + 2)
    assert pb[row, col] == pytest.approx(1.0)
    # and the grid centre now reads the analytic response at that distance
    fwhm = pb_fwhm_arcsec(230e9, 12.0)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    r2 = 1.0**2 + 0.5**2
    assert pb[100, 100] == pytest.approx(np.exp(-0.5 * r2 / sigma**2), rel=1e-6)


def test_no_offset_means_the_grid_centre():
    a = primary_beam_map((101, 101), 0.5, 230e9, 12.0)
    b = primary_beam_map((101, 101), 0.5, 230e9, 12.0, image_centre_offset_arcsec=None)
    c = primary_beam_map((101, 101), 0.5, 230e9, 12.0, image_centre_offset_arcsec=(0.0, 0.0))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    assert a[50, 50] == pytest.approx(1.0)


def test_a_recentred_source_is_corrected_by_its_true_response():
    """The number that was wrong: ALMA 12 m at 245 GHz, source 4" off the
    pointing (both real datasets that motivated --image-centre). The PB there
    is ~0.93, and pbcor must divide by that, not by 1."""
    shape, ps = (64, 64), 0.125           # 8" field
    y0, x0 = 4.0, 0.0                     # image centred 4" north of the pointing
    pb = primary_beam_map(shape, ps, 245e9, 12.0, image_centre_offset_arcsec=(y0, x0))
    fwhm = pb_fwhm_arcsec(245e9, 12.0)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected = np.exp(-0.5 * (4.0 / sigma) ** 2)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    at_centre = pb[int(np.floor(cy)):int(np.ceil(cy)) + 1, int(np.floor(cx)):int(np.ceil(cx)) + 1].mean()
    assert 0.9 < expected < 0.96
    assert at_centre == pytest.approx(expected, rel=0.01)
    img = np.ones(shape)
    corrected = pb_correct(img, pb)
    assert corrected[32, 32] == pytest.approx(1.0 / pb[32, 32])

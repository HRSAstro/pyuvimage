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

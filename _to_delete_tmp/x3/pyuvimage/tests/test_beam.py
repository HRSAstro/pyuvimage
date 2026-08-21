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

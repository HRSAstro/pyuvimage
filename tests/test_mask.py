"""Emission masks (`mask.py`): region files and FITS masks onto the fit grid."""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from pyuvimage import mask

META = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0}
pytest.importorskip("regions")


def _sexa(c):
    return (c.ra.to_string(u.hour, sep=":", precision=4),
            c.dec.to_string(u.deg, sep=".", precision=3, alwayssign=True))


def test_crtf_ellipse_lands_north_east_where_the_sky_says(tmp_path):
    """1" east and 0.5" north of the phase centre: east is towards smaller
    column (x = -dRA), north is towards row 0 in native orientation."""
    c = SkyCoord(150.0 * u.deg, 2.0 * u.deg).spherical_offsets_by(1.0 * u.arcsec, 0.5 * u.arcsec)
    ra, dec = _sexa(c)
    f = tmp_path / "e.crtf"
    f.write_text("#CRTFv0\n"
                 f"ellipse [[{ra}, {dec}], [0.6arcsec, 0.3arcsec], 0deg] coord=J2000\n")
    m = mask.load(f, (64, 64), 0.05, META)
    yy, xx = np.nonzero(m)
    assert (31.5 - yy.mean()) * 0.05 == pytest.approx(0.5, abs=0.05)
    assert (xx.mean() - 31.5) * 0.05 == pytest.approx(-1.0, abs=0.05)
    # PA 0: the 0.6" semi-axis runs north-south
    assert (yy.max() - yy.min()) > 1.8 * (xx.max() - xx.min())


def test_exclusion_regions_are_cut_out(tmp_path):
    ra, dec = _sexa(SkyCoord(150.0 * u.deg, 2.0 * u.deg))
    f = tmp_path / "x.crtf"
    f.write_text("#CRTFv0\n"
                 f"circle [[{ra}, {dec}], 1arcsec] coord=J2000\n"
                 f"-circle [[{ra}, {dec}], 0.4arcsec] coord=J2000\n")
    m = mask.load(f, (64, 64), 0.05, META)
    assert not m[31:33, 31:33].any() and m.sum() > 0


def test_fits_round_trip_and_regrid(tmp_path):
    ra, dec = _sexa(SkyCoord(150.0 * u.deg, 2.0 * u.deg))
    f = tmp_path / "c.crtf"
    f.write_text(f"#CRTFv0\ncircle [[{ra}, {dec}], 0.5arcsec] coord=J2000\n")
    m = mask.load(f, (64, 64), 0.05, META)
    out = mask.write(tmp_path / "m.fits", m, 0.05, META)
    np.testing.assert_array_equal(mask.load(out, (64, 64), 0.05, META), m)
    fine = mask.load(out, (128, 128), 0.025, META)
    assert fine.sum() == 4 * m.sum()


def test_bad_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        mask.load(tmp_path / "missing.crtf", (8, 8), 0.1, META)
    (tmp_path / "m.txt").write_text("x")
    with pytest.raises(ValueError, match="unrecognised"):
        mask.load(tmp_path / "m.txt", (8, 8), 0.1, META)


def test_draw_writes_the_painted_mask_on_the_image_wcs(tmp_path, monkeypatch):
    """The GUI itself needs a display; the painter is replaced by one that
    'paints' a box, and what reaches the file is checked."""
    import autogalaxy.gui.scribbler as scrib

    painted = np.zeros((40, 40), bool)
    painted[5:12, 20:30] = True                     # native: near the top

    class FakePainter:
        def __init__(self, image, **kw):
            assert image.shape_native == (40, 40)
        def mask_from(self):
            return painted

    monkeypatch.setattr(scrib, "Scribbler", FakePainter)
    img = mask.write(tmp_path / "img.fits", np.ones((40, 40), bool), 0.1, META)
    out = mask.draw(img, tmp_path / "drawn.fits", backend="Agg")
    np.testing.assert_array_equal(mask.load(out, (40, 40), 0.1, META), painted)

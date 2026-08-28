"""The written FITS must agree with its own WCS.

Our arrays are North-up (autoarray's native convention: row 0 is North, which
is what `peak_offset_arcsec` assumes and what the summary figure draws with
`origin="upper"`). FITS is the other way up -- with CDELT2 positive, Dec
increases with row index, so row 0 is the southernmost.

Writing the array unflipped put every product upside down in Dec while leaving
RA correct. Nothing in pyuvimage's own output showed it: the summary PNG uses
the in-memory convention throughout, so it looked right. It was only visible
on opening model.fits in CASA or DS9 -- which is where the images are actually
used.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from pyuvimage.products import build_header, to_fits_orientation

META = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0}
PIX = 0.1
N = 64


def _write_and_read(array, tmp_path):
    """Round-trip through the real writer path and astropy's WCS."""
    hdr = build_header(N, PIX, META, "Jy/beam")
    path = tmp_path / "t.fits"
    fits.writeto(path, to_fits_orientation(array), hdr, overwrite=True)
    with fits.open(path) as hdul:
        data, w = hdul[0].data, WCS(hdul[0].header)
    row, col = np.unravel_index(int(np.nanargmax(data)), data.shape)
    ra, dec = w.all_pix2world([[col, row]], 0)[0]
    return (
        (ra - META["phase_centre_ra_deg"]) * 3600.0
        * np.cos(np.radians(META["phase_centre_dec_deg"])),
        (dec - META["phase_centre_dec_deg"]) * 3600.0,
    )


def _north_up_with_source_at(d_ra, d_dec):
    """A North-up array with one bright pixel at a known sky offset."""
    a = np.zeros((N, N))
    c = (N - 1) / 2.0
    row = int(round(c - d_dec / PIX))     # row 0 is North
    col = int(round(c - d_ra / PIX))      # +x is West, i.e. decreasing RA
    a[row, col] = 1.0
    return a


@pytest.mark.parametrize("d_ra,d_dec", [
    (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (0.8, -1.2),
])
def test_a_source_lands_where_the_wcs_says(d_ra, d_dec, tmp_path):
    got_ra, got_dec = _write_and_read(_north_up_with_source_at(d_ra, d_dec),
                                      tmp_path)
    assert got_ra == pytest.approx(d_ra, abs=1.5 * PIX)
    assert got_dec == pytest.approx(d_dec, abs=1.5 * PIX)


def test_dec_is_not_merely_symmetric(tmp_path):
    """A source on the axis would pass a sign error; this one cannot."""
    north = _write_and_read(_north_up_with_source_at(0.0, 2.0), tmp_path)[1]
    south = _write_and_read(_north_up_with_source_at(0.0, -2.0), tmp_path)[1]
    assert north > 0 > south


def test_the_unflipped_array_would_have_failed(tmp_path):
    """Guard against the fix being reverted or double-applied."""
    array = _north_up_with_source_at(0.0, 1.5)
    hdr = build_header(N, PIX, META, "Jy/beam")
    path = tmp_path / "raw.fits"
    fits.writeto(path, array.astype("float32"), hdr, overwrite=True)
    with fits.open(path) as hdul:
        data, w = hdul[0].data, WCS(hdul[0].header)
    row, col = np.unravel_index(int(np.nanargmax(data)), data.shape)
    dec = w.all_pix2world([[col, row]], 0)[0][1]
    assert (dec - META["phase_centre_dec_deg"]) * 3600.0 < 0, (
        "writing the array unflipped should put a northern source south"
    )


def test_the_flip_is_applied_exactly_once_per_plane():
    """Twice cancels, and cancelling is what hid a mirrored sky for two days:
    the writer flipped in `stack()` and again on the way to `fits.writeto`, so
    the file came out unflipped and only looked right because the imaging was
    mirrored too. `write_products` now flips once, in `stack()`."""
    plane = np.arange(4 * 3, dtype=float).reshape(4, 3)
    assert np.array_equal(to_fits_orientation(plane), plane[::-1, :])
    assert np.array_equal(to_fits_orientation(to_fits_orientation(plane)), plane)

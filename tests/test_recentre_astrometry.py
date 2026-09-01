"""`--image-centre` end to end, through the FITS flip.

Two axis conventions meet here and both were wrong at some point this week:
the recentring phase ramp (visibility domain) and the FITS row order (write
time). Either alone can be verified and still leave the pair broken, so this
tests the only thing that matters to a user: put a source at a known sky
position, recentre onto it, and ask the *written file* where the source is.

The check that makes it airtight is the WCS one. A fit whose grid and header
are both flipped puts the source at the image centre either way -- the picture
looks perfect. Only asking the WCS for the sky coordinate of the peak pixel
distinguishes "centred correctly" from "centred, and mislabelled".
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import pyuvimage
from pyuvimage import mock

# a source well off the phase centre, asymmetric in both axes so that no sign
# error can hide behind a coincidence. X, Y are what a user types (image axes);
# D_RA, D_DEC are the sky pair every product is written in.
X, Y = -0.9, -1.3
D_RA, D_DEC = 0.9, -1.3


@pytest.fixture(scope="module")
def recentred(tmp_path_factory):
    uvd, _, geom, _ = mock.make_demo_dataset(
        n_vis=700, mesh_n=20, seed=17, point_flux_jy=0.03,
        point_centre=(D_RA, D_DEC),      # mock takes (dRA, dDec)
    )
    out = tmp_path_factory.mktemp("recentred")
    pyuvimage.run(
        uvd, fov=2.0, out=out, image_centre=(X, Y),
        reg="matern", coefficient=1e4, uncertainty_map=False,
        pb_correction=False, mask_shape="square",
    )
    return out


def _peak_sky(path):
    """(dRA, dDec) of the brightest pixel, read through the file's own WCS."""
    with fits.open(path) as hdul:
        data, w = np.asarray(hdul[0].data, dtype=float), WCS(hdul[0].header)
        ra0 = hdul[0].header["CRVAL1"]
        dec0 = hdul[0].header["CRVAL2"]
    if data.ndim == 3:
        data, w = data[0], w.celestial
    row, col = np.unravel_index(int(np.nanargmax(data)), data.shape)
    ra, dec = w.all_pix2world([[col, row]], 0)[0]
    return (
        (ra - ra0) * 3600.0 * np.cos(np.radians(dec0)),
        (dec - dec0) * 3600.0,
        row, col, data.shape,
    )


def test_the_source_sits_at_the_centre_of_the_recentred_image(recentred):
    """The point of recentring: the source is in the middle of the field."""
    _, _, row, col, shape = _peak_sky(recentred / "model_reconvolved.fits")
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    assert abs(row - cy) <= 2 and abs(col - cx) <= 2


def test_the_wcs_puts_the_source_back_at_its_true_sky_position(recentred):
    """CRVAL moved with the grid, so the peak's *absolute* position is still
    the truth -- offset from the recentred CRVAL by ~0, and CRVAL itself
    displaced from the phase centre by exactly what was asked for."""
    d_ra, d_dec, _, _, _ = _peak_sky(recentred / "model_reconvolved.fits")
    assert abs(d_ra) < 0.25 and abs(d_dec) < 0.25


def test_crval_is_the_phase_centre_plus_the_requested_offset(recentred):
    with fits.open(recentred / "model_reconvolved.fits") as hdul:
        h = hdul[0].header
    ra0, dec0 = 150.0, 2.0                       # the mock's phase centre
    d_ra = (h["CRVAL1"] - ra0) * 3600.0 * np.cos(np.radians(dec0))
    d_dec = (h["CRVAL2"] - dec0) * 3600.0
    assert d_ra == pytest.approx(D_RA, abs=0.02)
    assert d_dec == pytest.approx(D_DEC, abs=0.02)
    assert h["IMCENOFF"] == f"{D_DEC:.4f},{-D_RA:.4f}"   # stored as grid (y, x)


def test_dec_is_not_flipped_by_the_recentring(recentred):
    """The composition that had to be checked: a southward request must not
    come back north once the array is flipped into FITS row order."""
    with fits.open(recentred / "model_reconvolved.fits") as hdul:
        h = hdul[0].header
    assert np.sign((h["CRVAL2"] - 2.0)) == np.sign(D_DEC)


@pytest.fixture(scope="module")
def recentred_with_pb(tmp_path_factory):
    """Same fit, with the primary-beam products switched on."""
    uvd, _, geom, _ = mock.make_demo_dataset(
        n_vis=700, mesh_n=20, seed=17, point_flux_jy=0.03,
        point_centre=(D_RA, D_DEC),
    )
    out = tmp_path_factory.mktemp("recentred_pb")
    pyuvimage.run(
        uvd, fov=2.0, out=out, image_centre=(X, Y),
        reg="matern", coefficient=1e4, uncertainty_map=False,
        pb_correction=True, dish_diameter=12.0, mask_shape="square",
    )
    return out


def test_the_primary_beam_peaks_at_the_phase_centre_not_the_image_centre(
    recentred_with_pb,
):
    """The dish pointed at the phase centre; --image-centre only moved the
    grid. So `pb.fits`, read through its own WCS, must peak at sky offset
    (0, 0) -- and must NOT peak at the image centre, which is what it did.

    The demo's 2" field at 230 GHz spans a tiny fraction of the 25" PB, so the
    peak pixel is a weak locator on its own; the gradient check is the sharp
    one. The PB must be *higher* on the side of the image facing the phase
    centre than on the far side, by the analytic ratio.
    """
    with fits.open(recentred_with_pb / "pb.fits") as hdul:
        pb = np.asarray(hdul[0].data, dtype=float)
        w = WCS(hdul[0].header)
        ra0, dec0 = hdul[0].header["CRVAL1"], hdul[0].header["CRVAL2"]
        cd = abs(hdul[0].header["CDELT2"]) * 3600.0
    if pb.ndim == 3:
        pb, w = pb[0], w.celestial
    ny, nx = pb.shape

    # where, on this grid, is the phase centre? ask the WCS for its pixel
    ra_pc = ra0 - D_RA / 3600.0 / np.cos(np.radians(dec0))   # CRVAL is the recentred position
    dec_pc = dec0 - D_DEC / 3600.0
    col_pc, row_pc = w.all_world2pix([[ra_pc, dec_pc]], 0)[0]
    # it lies off the grid centre by the recentring offset, in pixels
    assert abs(row_pc - (ny - 1) / 2) > 2 or abs(col_pc - (nx - 1) / 2) > 2

    # the PB is symmetric about the phase centre: two pixels equidistant from
    # it on opposite sides read the same, and a pixel nearer it reads higher
    from pyuvimage.primary_beam import pb_fwhm_arcsec
    sigma = pb_fwhm_arcsec(230e9, 12.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    rows = np.arange(ny)[:, None]
    cols = np.arange(nx)[None, :]
    r2 = ((rows - row_pc) ** 2 + (cols - col_pc) ** 2) * cd**2
    expected = np.exp(-0.5 * r2 / sigma**2)
    np.testing.assert_allclose(pb, expected, rtol=2e-3)

    # and it is definitely not centred on the image: that model is wrong by
    # more than the tolerance above somewhere in the field
    r2_grid = ((rows - (ny - 1) / 2) ** 2 + (cols - (nx - 1) / 2) ** 2) * cd**2
    wrong = np.exp(-0.5 * r2_grid / sigma**2)
    assert np.abs(pb - wrong).max() > 1e-3

"""Recentring the reconstruction on an off-centre source.

Cost goes as the square of the field of view, and both real ALMA datasets that
motivated this sit 3-4 arcsec off the phase centre: covering the source from
the phase centre needed an 8" field (32 GB, hours), while recentred it needs 3"
(4.4 GB, minutes) *at finer resolution*.

The operation is an exact phase ramp, V' = V exp(2 pi i (u x0 + v y0)). The way
it can do real damage is silently: if the WCS does not move with the grid,
every product is astrometrically wrong by exactly the amount shifted, and
nothing in the image looks off.
"""

import numpy as np
import pytest

from pyuvimage import fitting, mock
from pyuvimage.beam import DirtyImager, wide_field_dirty_image
from pyuvimage.cli import _parse_centre
from pyuvimage.envelope import peak_offset_arcsec
from pyuvimage.products import build_header
from pyuvimage.uvdata import shift_image_centre


@pytest.fixture(scope="module")
def offset_source():
    """A mock whose brightest feature is well away from the phase centre."""
    uvd, _, geom, _ = mock.make_demo_dataset(
        n_vis=600, mesh_n=24, seed=11, point_flux_jy=0.02,
        point_centre=(0.8, -0.6),
    )
    return uvd, geom


def _peak(uvd, geom):
    uv, d, n = uvd.flattened()
    imager = DirtyImager(fitting.make_dataset(uv, d, n, geom))
    return peak_offset_arcsec(imager.dirty_image(d), geom.pixel_scale)


# --- argument parsing ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("centre", "centre"), ("center", "centre"), ("auto", "auto"),
    # image x,y, passed through unconverted -- api.run does the conversion
    ("1.5,-2", (1.5, -2.0)), (" (0.5, 0.25) ", (0.5, 0.25)),
])
def test_parse_image_centre(text, expected):
    assert _parse_centre(text) == expected


def test_a_malformed_centre_is_refused():
    with pytest.raises(SystemExit, match="image-centre"):
        _parse_centre("over there")


def test_the_three_conventions_stay_straight():
    """Three coordinate pairs live in this codebase and two of them differ by
    a sign on the first number. Getting it wrong puts the field on the wrong
    side of the phase centre, and the fit still converges -- on empty sky.

        CLI and API  image (x, y)   +x right on summary.png, +y up
        sky          (dRA, dDec)    +RA East, +Dec North;  dRA = -x
        internal     grid (y, x)    y = dDec, x = -dRA
    """
    from pyuvimage.pointsource import image_to_sky, sky_to_grid

    assert _parse_centre("2,-1") == (2.0, -1.0)           # CLI: image x,y
    assert image_to_sky(2.0, -1.0) == (-2.0, -1.0)        # api.run -> sky
    assert sky_to_grid(-2.0, -1.0) == (-1.0, 2.0)         # sky -> grid


def test_an_explicit_centre_moves_the_source_to_the_middle(offset_source):
    """End to end through the API's own conversion, in sky coordinates."""
    from pyuvimage.api import _recentre
    from pyuvimage.pointsource import grid_to_sky

    uvd, geom = offset_source
    d_ra, d_dec = grid_to_sky(*_peak(uvd, geom))
    x, y = -d_ra, d_dec                      # the API takes image (x, y)
    moved = _recentre(uvd, (x, y), fov=1.0, dish_diameter=None)
    y, x = _peak(moved, geom)
    assert max(abs(y), abs(x)) <= geom.pixel_scale


def test_a_bad_image_centre_string_is_refused_by_the_api():
    from pyuvimage.api import _recentre

    with pytest.raises(ValueError, match=r"\(x, y\)"):
        _recentre(None, "middle", fov=1.0, dish_diameter=None)


# --- the shift itself ------------------------------------------------------

def test_the_source_moves_to_the_centre(offset_source):
    uvd, geom = offset_source
    y0, x0 = _peak(uvd, geom)
    assert max(abs(y0), abs(x0)) > 4 * geom.pixel_scale, "mock is not offset"
    moved = _peak(shift_image_centre(uvd, (y0, x0)), geom)
    assert max(abs(moved[0]), abs(moved[1])) <= geom.pixel_scale


def test_the_shift_moves_no_power(offset_source):
    """A phase ramp is unitary: every amplitude is untouched, so total flux
    and the noise level cannot change."""
    uvd, _ = offset_source
    before = uvd.flattened()[1]
    after = shift_image_centre(uvd, (0.8, -0.6)).flattened()[1]
    assert np.allclose(np.abs(before), np.abs(after))


def test_shifting_back_restores_the_data(offset_source):
    uvd, _ = offset_source
    there = shift_image_centre(uvd, (0.8, -0.6))
    back = shift_image_centre(there, (-0.8, 0.6))
    assert np.allclose(back.flattened()[1], uvd.flattened()[1], atol=1e-12)
    assert back.meta["image_centre_offset_arcsec"] == pytest.approx([0.0, 0.0])


def test_a_zero_shift_is_a_no_op(offset_source):
    uvd, _ = offset_source
    assert shift_image_centre(uvd, (0.0, 0.0)) is uvd


def test_the_offset_is_recorded_and_accumulates(offset_source):
    uvd, _ = offset_source
    once = shift_image_centre(uvd, (1.0, -0.5))
    twice = shift_image_centre(once, (0.25, 0.25))
    assert once.meta["image_centre_offset_arcsec"] == pytest.approx([1.0, -0.5])
    assert twice.meta["image_centre_offset_arcsec"] == pytest.approx([1.25, -0.25])


# --- the astrometry, which is the part that can be silently wrong ----------

def _crval(offset):
    meta = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0}
    if offset is not None:
        meta["image_centre_offset_arcsec"] = list(offset)
    h = build_header(64, 0.1, meta, "Jy/pixel")
    return h["CRVAL1"], h["CRVAL2"]


def test_the_wcs_follows_the_grid():
    ra0, dec0 = _crval(None)
    ra, dec = _crval((1.8, 0.0))            # 1.8" North
    assert dec - dec0 == pytest.approx(1.8 / 3600.0)
    assert ra == pytest.approx(ra0)


def test_the_wcs_ra_shift_carries_the_cos_dec():
    ra0, dec0 = _crval(None)
    ra, dec = _crval((0.0, 3.6))            # +x, i.e. decreasing RA
    assert dec == pytest.approx(dec0)
    assert ra0 - ra == pytest.approx(
        (3.6 / 3600.0) / np.cos(np.radians(dec0)), rel=1e-6
    )


def test_the_offset_is_stated_in_the_header():
    meta = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0,
            "image_centre_offset_arcsec": [1.0, -2.0]}
    h = build_header(64, 0.1, meta, "Jy/pixel")
    assert h["IMCENOFF"] == "1.0000,-2.0000"


# --- the wide-field finder -------------------------------------------------

def test_the_wide_field_image_agrees_with_the_transformer(offset_source):
    """It is a separate implementation, so it has to be pinned to the one the
    fit uses -- a flipped axis here would recentre in the wrong direction."""
    uvd, geom = offset_source
    uv, d, n = uvd.flattened()
    ref = DirtyImager(fitting.make_dataset(uv, d, n, geom)).dirty_image(d)
    fov = geom.pixel_scale * ref.shape[0]
    img, rms = wide_field_dirty_image(uv, d, n, fov, n_pixels=ref.shape[0])
    assert peak_offset_arcsec(img, fov / img.shape[0]) == pytest.approx(
        peak_offset_arcsec(ref, geom.pixel_scale)
    )
    assert rms > 0


def test_the_wide_field_image_sees_beyond_the_reconstruction(offset_source):
    """The whole point: find emission outside the field being fitted."""
    uvd, geom = offset_source
    uv, d, n = uvd.flattened()
    narrow = geom.pixel_scale * 4
    img, _ = wide_field_dirty_image(uv, d, n, narrow * 8, n_pixels=64)
    y, x = peak_offset_arcsec(img, narrow * 8 / 64)
    assert max(abs(y), abs(x)) > narrow / 2


# --- the noise under rotation ---------------------------------------------

def test_the_rotation_preserves_the_total_noise_variance():
    """sigma_re^2 + sigma_im^2 is what chi^2 actually consumes, and the
    quadrature mean preserves it exactly -- so recentring cannot move chi^2."""
    from pyuvimage.uvdata import UVData

    rng = np.random.default_rng(4)
    n_vis = 64
    uvw = rng.normal(scale=100.0, size=(n_vis, 3))
    data = rng.normal(size=(1, n_vis)) + 1j * rng.normal(size=(1, n_vis))
    # deliberately lopsided, far beyond anything real
    noise = (0.4 + 0.1 * rng.random((1, n_vis))) + 1j * (
        0.9 + 0.1 * rng.random((1, n_vis))
    )
    uvd = UVData(
        uvw=uvw, frequencies=np.array([2.3e11]), data=data, noise=noise,
    )
    before = np.asarray(uvd.noise)
    after = np.asarray(shift_image_centre(uvd, (1.0, 0.5)).noise)
    assert np.allclose(
        before.real**2 + before.imag**2, after.real**2 + after.imag**2
    )
    # and the two components come out equal, as thermal noise should be
    assert np.allclose(after.real, after.imag)


def test_a_lopsided_noise_map_is_called_out(caplog):
    """Scatter of a few per cent is the estimator and is reported quietly;
    a systematic difference is worth a warning."""
    import logging

    from pyuvimage.uvdata import REIM_ASYMMETRY_WARN, _report_reim_asymmetry

    n = np.full(64, 1.0) + 1j * np.full(64, 1.0 + 2 * REIM_ASYMMETRY_WARN)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _report_reim_asymmetry(n)
    assert any(r.levelno == logging.WARNING for r in caplog.records)

    caplog.clear()
    quiet = np.full(64, 1.0) + 1j * np.full(64, 1.02)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _report_reim_asymmetry(quiet)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_a_negative_offset_gets_a_usable_error():
    """argparse reads `--image-centre -2.3,0.3` as a missing argument and says
    "expected one argument", which tells you nothing. Half the sky is at
    negative dRA, so this is a normal thing to type."""
    from pyuvimage.cli import main

    with pytest.raises(SystemExit, match='image-centre="-2.3,0.3"'):
        main(["fit", "x.npz", "--fov", "5", "--image-centre", "-2.3,0.3"])


def test_the_equals_form_parses_fine():
    assert _parse_centre("-2.3,0.3") == (-2.3, 0.3)   # image x,y, verbatim


# --- the CLI takes image x,y; everything else speaks dRA/dDec --------------

def test_the_cli_takes_image_x_not_dra():
    """+x is right on summary.png and RA increases leftward, so x = -dRA.
    Typing the wrong one puts the field on the wrong side of the phase
    centre -- and the fit still converges, on empty sky."""
    from pyuvimage.pointsource import image_to_sky

    assert image_to_sky(2.0, 0.0) == (-2.0, 0.0)   # 2" right  = 2" West
    assert image_to_sky(-2.0, 0.0) == (2.0, 0.0)   # 2" left   = 2" East
    assert image_to_sky(0.0, 1.5) == (0.0, 1.5)    # y is Dec, unchanged
    assert _parse_centre("2.0,0.0") == (2.0, 0.0)  # CLI keeps image axes


def test_point_positions_use_the_same_convention_as_image_centre():
    """The two positional flags in one command must not disagree."""
    from pyuvimage.cli import _parse_pair

    assert _parse_pair("1.5,-0.5", "--point") == _parse_centre("1.5,-0.5")
    assert _parse_pair("1.5,-0.5", "--point") == (1.5, -0.5)


def test_a_negative_x_is_caught_for_point_too():
    from pyuvimage.cli import main

    with pytest.raises(SystemExit, match='--point="-1.2,0.4"'):
        main(["fit", "x.npz", "--fov", "5", "--point", "-1.2,0.4"])


# --- "0,0" is the default, and must be a true no-op -------------------------
#
# `shift_image_centre` pools sigma_re and sigma_im in quadrature, which is
# right when the phase ramp really does mix them and wrong when there is no
# ramp. So a zero offset must not go through it: doing so would quietly change
# the noise map on every default run, and with it chi^2 and which path
# `--inversion auto` takes (it declines sparse above 5% re/im asymmetry).

def _uneven_noise_dataset(n_vis=64, seed=0):
    """Deliberately unequal sigma_re and sigma_im, so pooling is detectable."""
    from pyuvimage.uvdata import UVData

    rng = np.random.default_rng(seed)
    return UVData(
        uvw=rng.normal(0, 100.0, (n_vis, 3)),
        frequencies=np.array([2.3e11]),
        data=rng.normal(size=(1, n_vis)) + 1j * rng.normal(size=(1, n_vis)),
        noise=np.full((1, n_vis), 0.10 + 0.13j),
        meta={},
    )


@pytest.mark.parametrize("spelling", ["0,0", "0.0,0.0", (0.0, 0.0), "centre"])
def test_a_zero_offset_leaves_the_dataset_untouched(spelling):
    from pyuvimage.api import _recentre

    uvd = _uneven_noise_dataset()
    out = _recentre(uvd, spelling, fov=3.0, dish_diameter=None)
    assert out is uvd, f"{spelling!r} went through the recentring machinery"
    np.testing.assert_allclose(np.asarray(out.noise), np.asarray(uvd.noise))


def test_a_real_offset_still_recentres_and_pools_the_noise():
    """The other half: the no-op must not have disabled recentring itself."""
    from pyuvimage.api import _recentre
    from pyuvimage.uvdata import reim_asymmetry

    uvd = _uneven_noise_dataset()
    assert reim_asymmetry(uvd.noise) > 0.2
    out = _recentre(uvd, (1.0, -0.5), fov=3.0, dish_diameter=None)
    assert out is not uvd
    assert reim_asymmetry(out.noise) == pytest.approx(0.0, abs=1e-12)


def test_the_api_default_is_the_phase_centre():
    """`run`'s default is the string "0,0", so it has to parse where the CLI's
    parsed tuple used to arrive."""
    import inspect

    from pyuvimage.api import _recentre, run

    assert inspect.signature(run).parameters["image_centre"].default == "0,0"
    uvd = _uneven_noise_dataset()
    assert _recentre(uvd, "0,0", fov=3.0, dish_diameter=None) is uvd


def test_a_numeric_string_offset_works_through_the_api():
    """The CLI spelling should mean the same thing passed to `run` directly."""
    from pyuvimage.api import _recentre

    uvd = _uneven_noise_dataset()
    from_string = _recentre(uvd, "1.0,-0.5", fov=3.0, dish_diameter=None)
    from_tuple = _recentre(uvd, (1.0, -0.5), fov=3.0, dish_diameter=None)
    np.testing.assert_allclose(
        np.asarray(from_string.data), np.asarray(from_tuple.data))


def test_an_unparseable_centre_says_what_is_allowed():
    from pyuvimage.api import _recentre

    with pytest.raises(ValueError, match="not understood"):
        _recentre(_uneven_noise_dataset(), "middle", fov=3.0, dish_diameter=None)

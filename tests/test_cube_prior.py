"""What the cube's shared prior is fitted on.

Cube mode fits each channel separately -- cheap -- but the prior they share
came from one MFS fit over *every* channel's visibilities. That pass is
n_chan times a single channel, and on Ruby CO(7-6) it is the whole reason an
otherwise affordable cube runs out of memory: 2.9 GB per channel against
20.1 GB for the prior pass.

The fix is not a memory trick, it is the right question: the prior is applied
to single channels, so it should be chosen on a dataset the size of a single
channel. Measured on Ruby 200 GHz continuum, full set against a random 1/8:

    criterion      full        1/8 raw     1/8 x 8    ratio to full
    structure      1.883e8     5.008e8     4.007e9    21.3
    discrepancy    2.920e8     3.991e7     3.193e8     1.09

The discrepancy row is the algebra working: (F + lambda C^-1)s = D with F and
D both sums over visibilities, so thinning by f and scaling lambda by f
reproduces the same model, and chi^2/N is invariant under that. The structure
row is why the coefficient must NOT be scaled back: a whiter residual map is
not a scale-invariant target, and a fit with an eighth of the data genuinely
wants a much stronger prior. Using the thinned coefficient as-is gives each
channel the prior a dataset of its size actually calls for.
"""

import logging

import numpy as np
import pytest

import pyuvimage
from pyuvimage import mock
from pyuvimage.grids import resolve_geometry

FOV, MESH, F0 = 3.0, 12, 230e9


@pytest.fixture(scope="module")
def cube():
    """A small four-channel cube -- enough that 1-in-n_chan is a real factor,
    small enough to fit in a couple of seconds."""
    freqs = F0 * (1.0 + 1e-3 * np.arange(4))
    uv_max = MESH / (2.0 * (FOV / 206265.0))
    uvw = mock.random_uv_coverage(400, uv_max * 299792458.0 / F0, F0, seed=7)
    geom = resolve_geometry(
        FOV,
        float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max() / 299792458.0),
        mesh_shape=(MESH, MESH),
    )
    truth = mock.exponential_image(
        geom.shape_native, geom.pixel_scale, flux_jy=0.05, r_eff_arcsec=0.4
    )
    return mock.simulate(truth, geom.pixel_scale, uvw, freqs, sigma_jy=3e-4,
                         seed=8)


def _run(uvd, out, **kw):
    return pyuvimage.run(
        uvd, fov=FOV, out=out, mode="cube", mesh_shape=(MESH, MESH),
        reg="matern", criterion="discrepancy", uncertainty_map=False,
        pb_correction=False, mask_shape="square", **kw,
    )


def test_the_default_fits_the_prior_on_one_channel_worth(cube, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        res = _run(cube, tmp_path)
    assert "random 1 visibility in 4" in caplog.text
    assert res.parameters["source_prior"][
        "prior_fitted_on_one_visibility_in"] == cube.n_chan


def test_the_coefficient_is_used_as_fitted_not_scaled_back(cube, tmp_path):
    """No rescaling anywhere on this path. The measurement in the module
    docstring is why: scaling by n_chan is right only if the criterion is
    scale-invariant, and `structure` -- the default on any well-constrained
    fit -- is not, by a factor of 21 on Ruby.

    Checked where it can be checked exactly: a coefficient supplied by hand
    describes the model, not the dataset, and must come back untouched."""
    res = _run(cube, tmp_path, coefficient=1e4, reg_scale=0.5)
    assert res.parameters["source_prior"]["coefficient"] == pytest.approx(1e4)
    assert res.parameters["source_prior"][
        "prior_fitted_on_one_visibility_in"] == cube.n_chan


def test_the_prior_pass_really_is_one_channel_worth(cube, tmp_path, caplog):
    """The memory claim, from the log the user will read."""
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _run(cube, tmp_path)
    line = next(l for l in caplog.text.splitlines() if "random 1 visibility" in l)
    per_chan = cube.n_samples // cube.n_chan
    assert f"{per_chan} of {cube.n_samples}" in line


def test_mfs_uses_every_visibility(cube, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        res = _run(cube, tmp_path, cube_prior="mfs")
    assert "random 1 visibility in" not in caplog.text
    assert "prior_fitted_on_one_visibility_in" not in res.parameters["source_prior"]


def test_an_unknown_setting_is_rejected(cube, tmp_path):
    with pytest.raises(ValueError, match="cube_prior"):
        _run(cube, tmp_path, cube_prior="sometimes")


def test_mfs_mode_ignores_it_entirely(cube, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        pyuvimage.run(
            cube, fov=FOV, out=tmp_path, mode="mfs", mesh_shape=(MESH, MESH),
            reg="matern", criterion="discrepancy", uncertainty_map=False,
            pb_correction=False, mask_shape="square", cube_prior="channel",
        )
    assert "random 1 visibility in" not in caplog.text


def test_the_adaptive_brightness_map_is_frozen_too(cube, tmp_path, caplog):
    """Without this, every channel re-runs the adaptive prior's first pass --
    and that first pass optimises its own hyperparameters, so an 8-channel
    cube costs nine full searches instead of one. It is also what "the
    channels share a prior" ought to mean."""
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        pyuvimage.run(
            cube, fov=FOV, out=tmp_path, mode="cube", mesh_shape=(MESH, MESH),
            reg="adaptive", criterion="discrepancy", uncertainty_map=False,
            pb_correction=False, mask_shape="square",
        )
    assert "brightness map is frozen" in caplog.text
    # one first pass in the whole run, not one per channel
    assert caplog.text.count("first pass (plain Matern)") == 1


def test_matern_cube_needs_no_brightness_map(cube, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _run(cube, tmp_path)
    assert "brightness map is frozen" not in caplog.text


def test_the_cube_summary_has_a_row_per_channel(cube, tmp_path):
    """It used to draw `products[0]` and stop, so the summary of an
    eight-channel cube was a picture of one channel -- and at a glance
    indistinguishable from an MFS image."""
    from PIL import Image

    _run(cube, tmp_path)
    single = tmp_path / "one"
    pyuvimage.run(
        cube, fov=FOV, out=single, mode="mfs", mesh_shape=(MESH, MESH),
        reg="matern", criterion="discrepancy", uncertainty_map=False,
        pb_correction=False, mask_shape="square",
    )
    h_cube = Image.open(tmp_path / "summary.png").size[1]
    h_one = Image.open(single / "summary.png").size[1]
    # four channels, so several times taller -- not the same figure
    assert h_cube > 2.5 * h_one, f"cube summary {h_cube}px vs mfs {h_one}px"


def test_a_long_cube_is_subsampled_rather_than_unrenderable(tmp_path):
    """Hundreds of channels would make a figure no one can read and
    matplotlib cannot render; an evenly spaced subset is drawn and the title
    says so."""
    from pyuvimage.products import MAX_SUMMARY_ROWS, _summary_png
    from unittest.mock import patch

    class _P:
        pass

    def fake(rows, geometry, path, freqs, ext, n_panels, note):
        fake.rows, fake.note = len(rows), note
        path.write_bytes(b"")

    n = MAX_SUMMARY_ROWS + 7
    products = [_P() for _ in range(n)]
    for q in products:
        q.uncertainty = None
    from types import SimpleNamespace

    with patch("pyuvimage.products._summary_png_cube", fake):
        _summary_png(products, SimpleNamespace(fov_arcsec=3.0),
                     tmp_path / "s.png", np.linspace(100e9, 101e9, n))
    assert fake.rows == MAX_SUMMARY_ROWS
    assert "evenly spaced" in fake.note


# --------------------------------------------------------------------------
# Point sources in a cube (1 Sep 2026)
# --------------------------------------------------------------------------
#
# Until this the MFS pass fitted the points and the channels were fitted with
# none: the point sat in every channel's residual (chi^2/N of 8-10 against an
# MFS of 1.0) while the MFS flux was stapled onto plane 0 of model.fits only.
# Now the MFS pass decides where the points are and every channel fits its own
# amplitude there.


@pytest.fixture(scope="module")
def cube_with_point():
    from pyuvimage.pointsource import point_visibilities, sky_to_grid

    freqs = F0 * (1.0 + 1e-3 * np.arange(3))
    uv_max = MESH / (2.0 * (FOV / 206265.0))
    uvw = mock.random_uv_coverage(500, uv_max * 299792458.0 / F0, F0, seed=11)
    geom = resolve_geometry(
        FOV,
        float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max() / 299792458.0),
        mesh_shape=(MESH, MESH),
    )
    truth = mock.exponential_image(
        geom.shape_native, geom.pixel_scale, flux_jy=0.05, r_eff_arcsec=0.4
    )
    uvd = mock.simulate(truth, geom.pixel_scale, uvw, freqs, sigma_jy=3e-4, seed=12)
    # a 20 mJy point off-centre in every channel. The spectral slope is kept
    # small: the MFS pass fits one amplitude across the band, and a slope
    # that moves the flux by many sigma per channel would (rightly) make the
    # MFS point fit look like a bad model rather than a point source.
    d_ra, d_dec = 0.7, -0.5
    y, x = sky_to_grid(d_ra, d_dec)
    fluxes = 0.020 * (1.0 + 0.01 * np.arange(len(freqs)))
    for c in range(len(freqs)):
        uvd.data[c] += fluxes[c] * point_visibilities(uvd.uv_wavelengths(c), y, x)
    return uvd, (d_ra, d_dec), fluxes


def test_points_are_carried_into_every_channel(cube_with_point, tmp_path):
    from astropy.io import fits

    uvd, (d_ra, d_dec), fluxes = cube_with_point
    res = _run(uvd, tmp_path, point_sources=[(-d_ra, d_dec)],  # image (x, y)
               cube_prior="mfs")
    n_chan = uvd.n_chan
    assert len(res.products) == n_chan
    for c, p in enumerate(res.products):
        assert p.points, f"channel {c} has no point component"
        assert len(p.points) == 1
        assert p.points[0].flux == pytest.approx(fluxes[c], rel=0.15), c
        # same position in every plane: the MFS decided it
        assert p.points[0].d_ra == pytest.approx(res.products[0].points[0].d_ra)
    # the per-channel chi^2 is honest now that the point is modelled
    per_datum = res.parameters["fit_quality"]["channel_chi2_per_datum"]
    assert len(per_datum) == n_chan
    assert all(v < 1.5 for v in per_datum), per_datum
    # the record describes what was fitted: n_data of the MFS pass, and the
    # transformer that actually ran
    assert res.parameters["fit_quality"]["n_data"] == 2 * uvd.n_samples
    assert res.parameters["solver"]["transformer"] != "auto"
    # the point flux is in every plane of model.fits, not just the first
    with fits.open(tmp_path / "model.fits") as hdul:
        cube = np.asarray(hdul[0].data, float)
        assert hdul[0].header["NPOINTS"] == 1
    peaks = cube.reshape(n_chan, -1).max(axis=1)
    assert peaks.min() > 0.5 * peaks.max(), peaks
    import json
    rec = json.loads((tmp_path / "point_sources.json").read_text())
    assert len(rec["channels"]) == n_chan
    assert rec["channels"][1]["points"][0]["flux_jy"] == pytest.approx(
        res.products[1].points[0].flux)

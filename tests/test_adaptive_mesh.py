"""Irregular meshes (`adaptive_mesh`): Hilbert points and hard masks."""
import logging

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from pyuvimage import adaptive_mesh, api, mask, mock


@pytest.fixture(scope="module")
def demo():
    return mock.make_demo_dataset(n_vis=2000, mesh_n=24, fov_arcsec=3.0, seed=1)


def test_hilbert_points_follow_positive_brightness_only():
    """Negative values are clipped: a negative bowl as deep as the source is
    bright draws no more points than empty sky."""
    n = 64
    img = np.zeros((n, n))
    img[20:28, 20:28] = 1.0                       # source
    img[40:48, 40:48] = -1.0                      # negative bowl
    pts = adaptive_mesh.hilbert_points(img, 0.1, 400, power=1.0, floor=0.01)
    rows = np.rint((n - 1) / 2 - pts[:, 0] / 0.1).astype(int)
    cols = np.rint(pts[:, 1] / 0.1 + (n - 1) / 2).astype(int)
    on_src = ((rows >= 20) & (rows < 28) & (cols >= 20) & (cols < 28)).sum()
    on_neg = ((rows >= 40) & (rows < 48) & (cols >= 40) & (cols < 48)).sum()
    # 64 source pixels at weight 1 against 4032 at the 0.01 floor: ~60% on the source
    assert on_src > 200 and on_neg < 10


def test_masked_mesh_has_free_points_inside_and_zeros_round_it(demo):
    _, _, geom, _ = demo
    n = geom.shape_native[0]
    yy, xx = np.mgrid[0:n, 0:n]
    allowed = np.hypot(yy - n / 2, xx - n / 2) < n / 4
    adapt = np.exp(-np.hypot(yy - n / 2, xx - n / 2) / 3.0)
    mesh = adaptive_mesh.build(geom, "hilbert", n_points=150, adapt=adapt, allowed=allowed)
    free, zero = mesh.points[:mesh.n_free], mesh.points[mesh.n_free:]
    r_free = np.hypot(free[:, 0], free[:, 1]) / geom.pixel_scale
    r_zero = np.hypot(zero[:, 0], zero[:, 1]) / geom.pixel_scale
    assert r_free.max() < n / 4 + 1
    assert mesh.n_zeroed > 0 and r_zero.min() >= n / 4 - 1
    lat = adaptive_mesh.build(geom, "rectangular", allowed=allowed)
    assert lat.n_free < geom.mesh_shape[0] ** 2 and lat.kind == "masked rectangular"
    assert adaptive_mesh.build(geom, "rectangular") is None


@pytest.mark.parametrize("streaming", [False, True])
def test_run_with_a_mask_holds_the_model_to_zero_outside(demo, tmp_path, streaming, caplog):
    uvd, _, _, _ = demo
    ra = SkyCoord(150.0 * u.deg, 2.0 * u.deg)
    region = tmp_path / "src.crtf"
    region.write_text("#CRTFv0\ncircle [[%s, %s], 1.5arcsec] coord=J2000\n" % (
        ra.ra.to_string(u.hour, sep=":", precision=4),
        ra.dec.to_string(u.deg, sep=".", precision=3, alwayssign=True)))
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        res = api.run(uvd, fov=3.0, out=tmp_path / "o", pb_correction=False,
                      streaming=streaming, mesh_type="hilbert", mesh_points=200,
                      emission_mask=str(region))
    assert "held at zero round the mask" in caplog.text
    model = np.asarray(res.products[0].model_image)            # native
    allowed = mask.load(region, model.shape, res.geometry.pixel_scale, uvd.meta)
    assert np.abs(model[~allowed]).max() < 0.01 * model.max()
    assert model.sum() == pytest.approx(0.05, rel=0.02)
    q = res.parameters["fit_quality"]
    assert q["chi_squared"] / q["n_data"] < 1.1


def test_mask_options_are_checked(demo, tmp_path):
    uvd, _, _, _ = demo
    with pytest.raises(ValueError, match="needs positivity"):
        api.run(uvd, fov=3.0, out=tmp_path / "x", emission_mask="m.crtf", positive_only=False,
                write=False)
    with pytest.raises(ValueError, match="mesh_type"):
        api.run(uvd, fov=3.0, out=tmp_path / "x", mesh_type="voronoi", write=False)

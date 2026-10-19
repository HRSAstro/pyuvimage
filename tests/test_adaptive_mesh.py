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


def test_chi2_message_tells_a_data_floor_from_a_bad_model():
    from pyuvimage import fitting

    at_floor = fitting.chi2_excess_message(1.355, 1.0, 1.353)
    assert "in the data" in at_floor and "--chi2-target 1.35" in at_floor
    assert "16% low" in at_floor
    bad = fitting.chi2_excess_message(1.6, 1.0, 1.05)
    assert "should not be trusted" in bad
    assert "should not be trusted" in fitting.chi2_excess_message(1.6, 1.0, None)


def test_structure_ratio_is_measured_inside_the_mask_region():
    """Outside a mask the residual holds what the model may not fit; the
    ratio is taken over the region the mesh carries."""
    from types import SimpleNamespace

    from pyuvimage import fitting

    rng = np.random.default_rng(1)
    resid = rng.normal(size=(40, 40))
    resid[:, :10] += 5.0 * np.sin(np.arange(40) / 3.0)[:, None]     # structure outside
    region = np.zeros((40, 40), bool)
    region[:, 12:] = True
    imager = SimpleNamespace(rms=1.0, inside=np.ones((40, 40), bool))
    whole = fitting._structure_ratio_from_map(resid, 1600.0, imager, 1600)
    imager.structure_region = region
    masked = fitting._structure_ratio_from_map(resid, 1600.0, imager, 1600)
    assert whole > 1.5 and masked == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("cube_mesh", ["shared", "channel"])
@pytest.mark.parametrize("streaming", [False, True])
def test_cube_channels_use_the_hilbert_mesh_and_mask(tmp_path, streaming, cube_mesh, caplog):
    from pyuvimage.grids import resolve_geometry

    fov, m, f0 = 3.0, 16, 230e9
    freqs = f0 * (1.0 + 1e-3 * np.arange(2))
    uvw = mock.random_uv_coverage(1500, m / (2.0 * fov / 206265.0) * 299792458.0 / f0, f0, seed=7)
    geom = resolve_geometry(fov, float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max()
                                       / 299792458.0), mesh_shape=(m, m))
    truth = mock.exponential_image(geom.shape_native, geom.pixel_scale, flux_jy=0.05,
                                   r_eff_arcsec=0.4)
    meta = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0, "dish_diameter_m": 12.0}
    uvd = mock.simulate(truth, geom.pixel_scale, uvw, freqs, sigma_jy=3e-4, seed=8, meta=meta)
    c = SkyCoord(150.0 * u.deg, 2.0 * u.deg)
    region = tmp_path / "c.crtf"
    region.write_text("#CRTFv0\ncircle [[%s, %s], 1.2arcsec] coord=J2000\n" % (
        c.ra.to_string(u.hour, sep=":", precision=4),
        c.dec.to_string(u.deg, sep=".", precision=3, alwayssign=True)))
    data = uvd.write(tmp_path / "ds") if streaming else uvd
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        res = api.run(data, fov=fov, out=tmp_path / "o", mode="cube", mesh_shape=(m, m),
                      pb_correction=False, mesh_type="hilbert", mesh_points=120,
                      emission_mask=str(region), streaming=streaming, write=False,
                      cube_mesh=cube_mesh)
    assert ("every channel is fitted on the MFS fit's hilbert" if cube_mesh == "shared"
            else "each channel gets its own Hilbert mesh") in caplog.text
    allowed = mask.load(region, res.products[0].model_image.shape, res.geometry.pixel_scale, meta)
    assert len(res.products) == 2
    for p in res.products:
        model = np.asarray(p.model_image)
        assert np.abs(model[~allowed]).max() < 0.01 * model.max()
        assert model.sum() == pytest.approx(0.05, rel=0.05)


@pytest.mark.parametrize("streaming", [False, True])
def test_cube_channels_can_have_their_own_hilbert_meshes(tmp_path, streaming, monkeypatch):
    """--cube-mesh channel: each channel's points follow that channel's
    emission (here a blob that moves across the field)."""
    from pyuvimage.grids import resolve_geometry
    from pyuvimage.uvdata import UVData

    fov, m, f0 = 4.0, 16, 230e9
    freqs = f0 * (1.0 + 1e-3 * np.arange(3))
    uvw = mock.random_uv_coverage(1200, m / (2.0 * fov / 206265.0) * 299792458.0 / f0, f0, seed=7)
    geom = resolve_geometry(fov, float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max()
                                       / 299792458.0), mesh_shape=(m, m))
    parts = []
    for i, x in enumerate((-1.0, 0.0, 1.0)):
        t = mock.exponential_image(geom.shape_native, geom.pixel_scale, flux_jy=0.02,
                                   r_eff_arcsec=0.2, centre_arcsec=(0.0, x))
        parts.append(mock.simulate(t, geom.pixel_scale, uvw, freqs[i:i + 1], sigma_jy=3e-4,
                                   seed=10 + i))
    meta = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0, "dish_diameter_m": 12.0}
    uvd = UVData(uvw=uvw, frequencies=freqs, data=np.concatenate([p.data for p in parts]),
                 noise=np.concatenate([p.noise for p in parts]), meta=meta)
    built = []
    orig = adaptive_mesh.build

    def spy(*a, **k):
        mesh = orig(*a, **k)
        built.append(mesh)
        return mesh

    monkeypatch.setattr(adaptive_mesh, "build", spy)
    data = uvd.write(tmp_path / "ds") if streaming else uvd
    res = api.run(data, fov=fov, out=tmp_path / "o", mode="cube", mesh_shape=(m, m),
                  pb_correction=False, reg="matern", coefficient=1e7, mesh_type="hilbert",
                  mesh_points=60, cube_mesh="channel", streaming=streaming, write=False)
    assert len(res.products) == 3 and len(built) == 4          # MFS + one per channel
    xs = [mesh.points[: mesh.n_free, 1].mean() for mesh in built[1:]]
    assert xs[0] < -0.1 < 0.1 < xs[2] and abs(xs[1]) < 0.15
    for p in res.products:
        assert np.asarray(p.model_image).sum() == pytest.approx(0.02, rel=0.1)
    with pytest.raises(ValueError, match="cube_mesh"):
        api.run(uvd, fov=fov, out=tmp_path / "x", mode="cube", mesh_shape=(m, m),
                pb_correction=False, mesh_type="hilbert", cube_mesh="per-plane", write=False)


def test_a_hilbert_run_is_sized_by_its_points():
    """The memory estimate and the criterion's data per pixel count the mesh
    points, not the rectangle they replace (a 3000-point NGC 4274 fit was
    estimated at 36 GB from the 192^2 rectangle)."""
    from pyuvimage.grids import resolve_geometry

    geom = resolve_geometry(3.0, 2e5, mesh_shape=(40, 40))
    assert api._model_pixels(geom, "rectangular", None) == 1600
    assert api._model_pixels(geom, "hilbert", None) == 400
    assert api._model_pixels(geom, "hilbert", 150) == 150


@pytest.mark.parametrize("streaming", [False, True])
def test_point_components_on_a_hilbert_mesh(tmp_path, streaming):
    """A point beside a disc, on a Hilbert-Delaunay mesh: found at its
    position, with its flux; refused with a mask."""
    import json

    uvd, truth, geom, comps = mock.make_demo_dataset(
        n_vis=4000, mesh_n=32, fov_arcsec=3.0, sigma_jy=1.56e-3, point_flux_jy=0.004,
        point_centre=(0.85, -0.65), seed=0)
    data = uvd.write(tmp_path / "ds") if streaming else uvd
    api.run(data, fov=3.0, out=tmp_path / "o", mesh_shape=(32, 32), reg="gaussian",
            criterion="structure", point_sources=True, pb_correction=False,
            mesh_type="hilbert", streaming=streaming)
    pts = json.loads((tmp_path / "o" / "point_sources.json").read_text())["points"]
    assert len(pts) == 1
    p = pts[0]
    assert abs(p["d_ra_arcsec"] - 0.85) < 0.05 and abs(p["d_dec_arcsec"] + 0.65) < 0.05
    assert p["flux_jy"] == pytest.approx(0.004, rel=0.05)
    with pytest.raises(NotImplementedError, match="--mask"):
        api.run(uvd, fov=3.0, out=tmp_path / "x", mesh_shape=(32, 32), point_sources=True,
                pb_correction=False, emission_mask=str(tmp_path / "none.crtf"), write=False)

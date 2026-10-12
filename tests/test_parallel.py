"""`--workers`: cube channels fitted in parallel processes.

The channels share a frozen prior and are otherwise independent, so the
parallel run must reproduce the serial one channel for channel.
"""
import numpy as np
import pytest

import pyuvimage
from pyuvimage import mock, parallel
from pyuvimage.grids import resolve_geometry

FOV, MESH, F0 = 3.0, 12, 230e9


@pytest.fixture(scope="module")
def cube():
    freqs = F0 * (1.0 + 1e-3 * np.arange(4))
    uv_max = MESH / (2.0 * (FOV / 206265.0))
    uvw = mock.random_uv_coverage(400, uv_max * 299792458.0 / F0, F0, seed=7)
    geom = resolve_geometry(
        FOV, float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * freqs.max() / 299792458.0),
        mesh_shape=(MESH, MESH))
    truth = mock.exponential_image(geom.shape_native, geom.pixel_scale,
                                   flux_jy=0.05, r_eff_arcsec=0.4)
    return mock.simulate(truth, geom.pixel_scale, uvw, freqs, sigma_jy=3e-4, seed=8)


def _run(uvd, out, **kw):
    return pyuvimage.run(
        uvd, fov=FOV, out=out, mode="cube", mesh_shape=(MESH, MESH),
        reg="matern", criterion="discrepancy", uncertainty="statistical",
        pb_correction=False, mask_shape="square", write=False, **kw)


def test_parallel_channels_reproduce_the_serial_cube(cube, tmp_path):
    serial = _run(cube, tmp_path / "s")
    par = _run(cube, tmp_path / "p", workers=2)
    assert len(par.products) == len(serial.products) == cube.n_chan
    for a, b in zip(serial.products, par.products):
        np.testing.assert_allclose(b.model_image, a.model_image, rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(b.uncertainty, a.uncertainty, rtol=1e-8, atol=1e-14)
    assert (par.parameters["fit_quality"]["channel_chi2_per_datum"]
            == pytest.approx(serial.parameters["fit_quality"]["channel_chi2_per_datum"]))


def test_worker_count_is_capped(monkeypatch):
    assert parallel.resolve_workers(1, 8) == 1
    assert parallel.resolve_workers(None, 8) == 1
    assert parallel.resolve_workers(16, 3) == 3                 # by the tasks
    monkeypatch.setattr(parallel, "cpu_count", lambda: 6)
    assert parallel.resolve_workers("auto", 10) == 6           # by the cores
    assert parallel.resolve_workers(0, 4) == 4
    from pyuvimage import fitting
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 4.0)
    # (1.4 + 0.6) GB per worker in 4 GB: two
    assert parallel.resolve_workers(8, 8, per_task_gb=1.4) == 2
    assert parallel.resolve_workers(8, 8, per_task_gb=100.0) == 1
    with pytest.raises(ValueError):
        parallel.resolve_workers(-2, 4)


def test_parallel_streamed_cube_reproduces_the_serial_one(tmp_path):
    """The streamed path ships each channel's accumulated terms to a worker."""
    from pyuvimage import api, streaming as stm

    if not stm.nufftax_available():
        pytest.skip("streaming needs nufftax")
    uvd, _, _ = mock.make_multi_spw_dataset(n_vis=500, mesh_n=10, fov_arcsec=3.0, seed=4)
    kw = dict(fov=3.0, mode="cube", cube_prior="mfs", reg="matern", criterion="discrepancy",
              inversion="sparse", pb_correction=False, write=False, uncertainty="statistical",
              streaming=True, chunk_k=256)
    serial = api.run(uvd, out=tmp_path / "s", **kw)
    par = api.run(uvd, out=tmp_path / "p", workers=2, **kw)
    assert len(par.products) == len(serial.products) == uvd.n_chan
    for a, b in zip(serial.products, par.products):
        np.testing.assert_allclose(b.model_image, a.model_image, rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(b.uncertainty, a.uncertainty, rtol=1e-8, atol=1e-14)

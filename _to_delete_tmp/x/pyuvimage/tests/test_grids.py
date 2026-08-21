import numpy as np
import pytest

from pyuvimage.grids import nyquist_pixel_scale_arcsec, resolve_geometry


def test_nyquist_formula():
    # 0.5 / b_max rad -> arcsec
    assert np.isclose(
        nyquist_pixel_scale_arcsec(1e6), 0.5 / 1e6 * 180 / np.pi * 3600
    )


def test_geometry_auto_is_nyquist_mesh_on_a_finer_product_grid():
    g = resolve_geometry(fov_arcsec=4.0, max_baseline_wavelengths=5e5)
    nyq = nyquist_pixel_scale_arcsec(5e5)
    assert g.nyquist_pixel_scale == pytest.approx(nyq)
    assert g.mesh_pixel_scale <= nyq * 1.05
    # the product grid must be strictly finer than the model mesh, otherwise
    # the residual dirty image collapses to the prior's pull (A^T W r = H s)
    assert g.pixel_scale < g.mesh_pixel_scale
    assert g.shape_native[0] % g.mesh_shape[0] == 0
    # image grid is an exact integer multiple of the mesh
    assert g.shape_native[0] % g.mesh_shape[0] == 0
    assert g.fov_arcsec == pytest.approx(g.mesh_shape[0] * g.mesh_pixel_scale)


def test_geometry_explicit_mesh():
    g = resolve_geometry(4.0, 5e5, mesh_shape=(40, 40), oversample=2)
    assert g.mesh_shape == (40, 40)
    assert g.shape_native == (80, 80)
    assert g.pixel_scale == pytest.approx(4.0 / 80)


def test_coarse_pixel_scale_warns():
    nyq = nyquist_pixel_scale_arcsec(5e5)
    with pytest.warns(UserWarning):
        resolve_geometry(4.0, 5e5, pixel_scale=nyq * 3)

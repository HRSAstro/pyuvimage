"""Unit tests for the simplified log-sky reconstructor helpers."""

import numpy as np
import pytest

from src.deconv.log_sky import (
    _neighbor_edges,
    _radial_edge_mask,
    _smooth_energy_and_grad,
)


def test_brightness_is_positive_exp():
    i0 = 2.5
    m = np.array([-2.0, 0.0, 1.5])
    image = i0 * np.exp(m)
    assert np.all(image > 0.0)
    assert image[1] == pytest.approx(i0)


def test_neighbor_edges_and_smooth_grad():
    kept = np.zeros((3, 3), dtype=bool)
    kept[1, 0] = True
    kept[1, 1] = True
    kept[1, 2] = True
    edge_i, edge_j = _neighbor_edges(kept)
    assert edge_i.size >= 2
    # Indices are native ravel positions, not packed slim.
    assert set(edge_i.tolist() + edge_j.tolist()) <= {3, 4, 5}
    m = np.zeros(9, dtype=float)
    m[3] = 0.0
    m[4] = 1.0
    m[5] = 0.0
    energy, grad = _smooth_energy_and_grad(m, edge_i, edge_j)
    assert energy > 0.0
    assert grad.shape == m.shape
    eps = 1e-6
    m2 = m.copy()
    m2[3] += eps
    e2, _ = _smooth_energy_and_grad(m2, edge_i, edge_j)
    assert grad[3] == pytest.approx((e2 - energy) / eps, rel=1e-4, abs=1e-4)


def test_rectangular_border_mask():
    from src.deconv.log_sky import _rectangular_border_mask

    border = _rectangular_border_mask((5, 5))
    assert border[0, :].all()
    assert border[-1, :].all()
    assert border[:, 0].all()
    assert border[:, -1].all()
    assert not border[2, 2]
    assert int(border.sum()) == 5 * 4 - 4  # corners counted once


def test_radial_edge_mask_outer_tenth():
    mask = _radial_edge_mask((40, 40), 0.1)
    assert mask.shape == (40, 40)
    assert not mask[0, 0]  # corner outside disk is excluded
    assert not mask[20, 20]  # centre is not
    frac = mask.mean()
    assert 0.10 < frac < 0.30
    assert mask[20, 38]
    assert not mask[20, 35]


def test_resolve_log_grid_mask_default():
    from src.deconv.log_sky import _resolve_log_grid

    class _Mask:
        shape_native = (120, 120)

    settings = {
        "mask_pixel_scale": 0.0390625,
        "mask_fov": 4.6875,
        "nyquist_pixel_scale": 0.1474,
        "log_sky": {"pixel_scale": "mask"},
    }
    grid = _resolve_log_grid(settings, _Mask())
    assert grid["log_shape"] == (120, 120)
    assert grid["upsample_order"] == 0


def test_resolve_log_grid_coarser_than_nyquist_uses_bilinear():
    from src.deconv.log_sky import _resolve_log_grid, _upsample_to_fine

    class _Mask:
        shape_native = (120, 120)

    settings = {
        "mask_pixel_scale": 0.0390625,
        "mask_fov": 4.6875,
        "nyquist_pixel_scale": 0.1474,
        "log_sky": {"pixel_scale": 0.3},
    }
    grid = _resolve_log_grid(settings, _Mask())
    assert grid["recon_pixel_scale"] > settings["nyquist_pixel_scale"]
    assert grid["upsample_order"] == 1
    assert grid["log_shape"][0] < 120
    coarse = np.zeros(grid["log_shape"], dtype=float)
    coarse[0, 0] = 1.0
    fine = _upsample_to_fine(coarse, (120, 120), order=1)
    assert fine.shape == (120, 120)
    # Bilinear spreads the impulse; NN would be a flat block of ones.
    assert fine[0, 0] > 0
    assert np.count_nonzero(fine > 1e-12) > 1


def test_upsample_nearest_integer_blocks():
    from src.deconv.log_sky import _upsample_to_fine

    coarse = np.array([[1.0, 2.0], [3.0, 4.0]])
    fine = _upsample_to_fine(coarse, (4, 4), order=0)
    assert fine.shape == (4, 4)
    assert fine[0, 0] == 1.0 and fine[0, 1] == 1.0
    assert fine[0, 2] == 2.0

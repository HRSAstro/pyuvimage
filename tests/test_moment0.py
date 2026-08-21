"""Unit tests for MFS channel collapse."""

import numpy as np
import pytest

from src.deconv.moment0 import (
    collapse_uv_to_mfs,
    collapse_visibilities_to_mfs,
    mfs_arrays_from,
)


def test_collapse_visibilities_mean():
    # 2 channels, 3 vis, real/imag
    vis = np.zeros((2, 3, 2))
    vis[0, :, 0] = 2.0
    vis[1, :, 0] = 4.0
    sigma = np.ones((2, 3, 2))
    vis_mfs, sigma_mfs = collapse_visibilities_to_mfs(vis, sigma)
    assert vis_mfs.shape == (3,)
    assert np.allclose(vis_mfs.real, 3.0)
    # independent_mean: sqrt(1+1)/2 = sqrt(2)/2
    assert sigma_mfs[0, 0] == pytest.approx(np.sqrt(2.0) / 2.0)


def test_collapse_uv_average():
    uv = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    out = collapse_uv_to_mfs(uv, uv_mode="average")
    assert out.shape == (1, 2)
    assert np.allclose(out, [[2.0, 3.0]])


def test_mfs_arrays_from():
    uv = np.ones((4, 5, 2))
    vis = np.ones((4, 5, 2))
    sigma = np.ones((4, 5, 2))
    vis_out, sigma_out, uv_out = mfs_arrays_from(uv, vis, sigma)
    assert vis_out.shape == (5,)
    assert sigma_out.shape == (5, 2)
    assert uv_out.shape == (5, 2)

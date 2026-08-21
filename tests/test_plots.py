"""Unit tests for shared sky-grid plotting helpers."""

import numpy as np
import pytest

from src.deconv.plots import (
    require_common_sky_grid,
    sky_extent_arcsec,
)


def test_sky_extent_centred():
    extent = sky_extent_arcsec((10, 10), 0.5)
    assert extent == [-2.5, 2.5, -2.5, 2.5]


def test_require_common_sky_grid_ok():
    img = np.zeros((8, 8))
    shape, extent = require_common_sky_grid(
        {"truth": img, "dirty": img, "recon": img, "resid": img},
        pixel_scale=0.1,
        expected_shape=(8, 8),
    )
    assert shape == (8, 8)
    assert extent == sky_extent_arcsec((8, 8), 0.1)


def test_require_common_sky_grid_rejects_mismatch():
    with pytest.raises(ValueError, match="share one image"):
        require_common_sky_grid(
            {"truth": np.zeros((8, 8)), "dirty": np.zeros((10, 10))},
            pixel_scale=0.1,
        )


def test_require_common_sky_grid_rejects_wrong_expected():
    with pytest.raises(ValueError, match="expected mask grid"):
        require_common_sky_grid(
            {"truth": np.zeros((8, 8))},
            pixel_scale=0.1,
            expected_shape=(16, 16),
        )

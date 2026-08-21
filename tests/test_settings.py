"""Unit tests for runner settings validation."""

import copy

import pytest

from src.deconv.settings import validate_settings

_BASE = {
    "fov": 5.0,
    "config_path": "./config",
    "output_path": "./output/test/",
    "data_directory": "./data/test",
    "data_patterns": {
        "frequencies": "frequencies_{uid}_width_{width}.fits",
        "uv_wavelengths": "uv_wavelengths_{uid}_width_{width}.fits",
        "visibilities": "visibilities_{uid}_width_{width}.fits",
        "sigma": "sigma_statwt_{uid}_width_{width}.fits",
    },
}


def test_validate_sets_defaults():
    settings = validate_settings(copy.deepcopy(_BASE))
    assert settings["mode"] == "mfs"
    assert settings["mesh_type"] == "rectangular_uniform"
    assert settings["regularization"]["type"] == "constant"
    assert settings["cube"]["regularization"] == "from_mfs"
    assert settings["source_pixel_scale"] == "nyquist"


def test_validate_requires_fov():
    settings = copy.deepcopy(_BASE)
    del settings["fov"]
    with pytest.raises(KeyError, match="fov"):
        validate_settings(settings)


def test_validate_rejects_bad_mode():
    settings = copy.deepcopy(_BASE)
    settings["mode"] = "clean"
    with pytest.raises(ValueError, match="mode"):
        validate_settings(settings)


def test_validate_delaunay_adapt_split():
    settings = copy.deepcopy(_BASE)
    settings["mesh_type"] = "delaunay"
    settings["regularization"] = {"type": "adapt_split"}
    out = validate_settings(settings)
    assert out["mesh_type"] == "delaunay"
    assert out["regularization"]["type"] == "adapt_split"


def test_validate_rejects_adapt_split_on_rectangular():
    settings = copy.deepcopy(_BASE)
    settings["regularization"] = {"type": "adapt_split"}
    with pytest.raises(ValueError, match="delaunay"):
        validate_settings(settings)


def test_validate_cube_mode():
    settings = copy.deepcopy(_BASE)
    settings["mode"] = "cube"
    settings["cube"] = {"regularization": "per_channel"}
    out = validate_settings(settings)
    assert out["mode"] == "cube"
    assert out["cube"]["regularization"] == "per_channel"


def test_validate_log_sky_defaults():
    settings = copy.deepcopy(_BASE)
    settings["reconstructor"] = "log_sky"
    out = validate_settings(settings)
    assert out["reconstructor"] == "log_sky"
    assert out["log_sky"]["i0"] == "auto"
    assert out["log_sky"]["smooth"] == 1.0
    assert out["log_sky"]["maxiter"] == 200


def test_validate_linear_sky_defaults():
    settings = copy.deepcopy(_BASE)
    settings["reconstructor"] = "linear_sky"
    out = validate_settings(settings)
    assert out["reconstructor"] == "linear_sky"
    assert out["linear_sky"]["smooth"] == 1.0
    assert out["linear_sky"]["maxiter"] == 200
    assert out["linear_sky"]["pixel_scale"] == "mask"
    assert "i0" not in out["linear_sky"]


def test_validate_auto_reconstructor_defaults():
    settings = copy.deepcopy(_BASE)
    settings["reconstructor"] = "auto"
    out = validate_settings(settings)
    assert out["reconstructor"] == "auto"
    assert out["sky_auto"]["snr_threshold"] == 100.0
    assert out["log_sky"]["smooth"] == "auto"
    assert out["log_sky"]["optimize_smooth"] is True
    assert out["linear_sky"]["smooth"] == "auto"
    assert out["linear_sky"]["optimize_smooth"] is True
    assert out["log_sky"]["smooth_init"] == 1.0e4
    assert out["linear_sky"]["smooth_init"] == 1.0e6


def test_validate_smooth_auto_on_explicit_log_sky():
    settings = copy.deepcopy(_BASE)
    settings["reconstructor"] = "log_sky"
    settings["log_sky"] = {"smooth": "auto", "edge_frac": 0.1}
    out = validate_settings(settings)
    assert out["log_sky"]["optimize_smooth"] is True
    assert out["log_sky"]["edge_prior_ratio"] == 100.0


def test_validate_rejects_bad_reconstructor():
    settings = copy.deepcopy(_BASE)
    settings["reconstructor"] = "clean"
    with pytest.raises(ValueError, match="reconstructor"):
        validate_settings(settings)


def test_validate_gaussian_kernel_regularization():
    settings = copy.deepcopy(_BASE)
    settings["regularization"] = {
        "type": "gaussian_kernel",
        "prior_type": "fixed",
        "coefficient": 50.0,
        "scale": 0.2,
    }
    out = validate_settings(settings)
    assert out["regularization"]["type"] == "gaussian_kernel"
    assert out["regularization"]["scale"] == 0.2


def test_validate_matern_kernel_regularization():
    settings = copy.deepcopy(_BASE)
    settings["regularization"] = {
        "type": "matern_kernel",
        "prior_type": "fixed",
        "coefficient": 50.0,
        "scale": 0.2,
        "nu": 2.5,
    }
    out = validate_settings(settings)
    assert out["regularization"]["type"] == "matern_kernel"
    assert out["regularization"]["nu"] == 2.5

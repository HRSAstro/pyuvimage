"""Tests for CLEAN-style restore helpers."""

import numpy as np
import pytest

from src.deconv.restore import (
    FWHM_TO_AREA,
    SIGMA_TO_FWHM,
    fit_clean_beam_gaussian,
    gaussian_beam_area_pixels,
    restore_clean_image,
)


def test_gaussian_beam_area_matches_2pi_sigma():
    sx, sy = 2.0, 3.0
    area = gaussian_beam_area_pixels(sx, sy)
    assert area == pytest.approx(FWHM_TO_AREA * (SIGMA_TO_FWHM * sx) * (SIGMA_TO_FWHM * sy))
    assert area == pytest.approx(2.0 * np.pi * sx * sy)


def test_fit_clean_beam_gaussian_recovers_peak_and_area():
    yy, xx = np.mgrid[0:41, 0:41]
    beam = np.exp(-0.5 * (((xx - 20) / 3.0) ** 2 + ((yy - 20) / 2.5) ** 2))
    kernel, params = fit_clean_beam_gaussian(beam, window_frac=0.5, pixel_scale=0.1)
    assert kernel.shape == beam.shape
    assert pytest.approx(kernel.max(), rel=1e-3) == 1.0
    assert abs(params["x_mean"] - 20.0) < 0.5
    assert abs(params["y_mean"] - 20.0) < 0.5
    assert params["beam_area_pixels"] == pytest.approx(
        gaussian_beam_area_pixels(params["x_stddev"], params["y_stddev"])
    )
    assert params["beam_area_arcsec2"] == pytest.approx(
        params["beam_area_pixels"] * 0.1**2
    )


def test_restore_clean_image_jy_per_beam():
    model = np.zeros((21, 21))
    model[10, 10] = 2.0  # Jy in one pixel
    residual = np.ones((21, 21)) * 0.5  # dirty units
    beam = np.zeros((21, 21))
    beam[10, 10] = 1.0  # peak-normalized delta
    beam_area = 4.0
    dirty_peak = 5.0
    clean = restore_clean_image(
        model,
        residual,
        beam,
        beam_area_pixels=beam_area,
        dirty_beam_peak=dirty_peak,
    )
    # restored = 2 * (4/1) = 8 Jy/beam at centre; residual = 0.5/5 = 0.1
    assert clean[10, 10] == pytest.approx(8.0 + 0.1)
    assert clean[0, 0] == pytest.approx(0.1)

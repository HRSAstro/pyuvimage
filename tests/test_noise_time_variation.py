"""The noise is not stationary over a real track, and `difference` cannot say so.

`sigma_from_time_differences` returns one sigma per baseline, pooled over the
whole observation -- the quadratic mean of sigma(t), with no time resolution.
That is harmless when the noise is stationary and actively harmful when it is
not: as a target rises and sets, airmass and Tsys follow, and sigma can change
by ~1.9x between transit and the ends of a 30-70 degree track. The quadratic
mean then over-weights the noisiest data.
"""

import numpy as np
import pytest

from pyuvimage.noise import (
    noise_time_variation,
    scale_relative_sigma,
    sigma_from_time_differences,
)

N_ANT = 16


def _track(n_time=60, elevation=True, seed=0, sigma_floor=0.004):
    """A source plus noise whose level follows airmass over the track."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(N_ANT, k=1)
    n_base = a1.size

    if elevation:
        el = np.radians(30 + 40 * np.sin(np.pi * np.arange(n_time) / max(n_time - 1, 1)))
        airmass = 1.0 / np.sin(el)
        profile = airmass / airmass.min()          # 1.0 at transit, ~1.9 at the ends
    else:
        profile = np.ones(n_time)

    tsys_ant = rng.uniform(1.0, 1.6, N_ANT)
    per_base = np.sqrt(tsys_ant[a1] * tsys_ant[a2])
    sigma_true = sigma_floor * np.outer(profile, per_base).ravel()[None, :]

    ant1 = np.tile(a1, n_time)
    ant2 = np.tile(a2, n_time)
    time = np.repeat(np.arange(float(n_time)), n_base)
    sky = np.tile(rng.normal(0.0, 0.05, n_base), n_time)[None, :]   # 12x the noise
    data = (
        sky
        + rng.normal(0, 1, sigma_true.shape) * sigma_true
        + 1j * rng.normal(0, 1, sigma_true.shape) * sigma_true
    )
    relative = 7.0 * sigma_true * (1 + 1j)   # the weight column tracks Tsys(t)
    return data, relative, sigma_true, ant1, ant2, time, profile


def test_a_stationary_track_reports_a_ratio_near_one():
    data, _, _, a1, a2, t, _ = _track(elevation=False)
    ratio, blocks = noise_time_variation(data, a1, a2, t)
    assert ratio == pytest.approx(1.0, abs=0.15)
    assert np.all(np.isfinite(blocks))


def test_an_elevation_track_is_detected():
    data, _, _, a1, a2, t, _ = _track(elevation=True)
    ratio, blocks = noise_time_variation(data, a1, a2, t)
    assert ratio > 1.25
    # thirds of a rise-transit-set track: high, low, high
    assert blocks[1] < blocks[0] and blocks[1] < blocks[2]


def test_the_ratio_is_a_lower_bound_on_the_real_variation():
    """Blocks average over their own span, so the report is conservative."""
    data, _, sigma_true, a1, a2, t, profile = _track(elevation=True)
    ratio, _ = noise_time_variation(data, a1, a2, t)
    assert 1.0 < ratio < profile.max() / profile.min()


def test_difference_delivers_the_quadratic_mean_and_no_time_dependence():
    data, _, sigma_true, a1, a2, t, profile = _track(elevation=True)
    est = sigma_from_time_differences(data, a1, a2, t).real

    n_base = np.triu_indices(N_ANT, k=1)[0].size
    first_baseline = est.reshape(-1)[np.arange(len(profile)) * n_base]
    assert np.allclose(first_baseline, first_baseline[0])   # flat in time

    truth = sigma_true.reshape(-1)[np.arange(len(profile)) * n_base]
    assert first_baseline[0] == pytest.approx(
        np.sqrt(np.mean(truth**2)), rel=0.1
    )


def test_the_cost_is_over_weighting_the_noisiest_data():
    """The reason this matters: it is the weights the fit uses that go wrong."""
    data, relative, sigma_true, a1, a2, t, profile = _track(elevation=True)
    n_time = len(profile)
    n_base = np.triu_indices(N_ANT, k=1)[0].size

    def weight_error(est):
        r = ((1.0 / est**2) / (1.0 / sigma_true**2)).reshape(n_time, n_base)
        return r[0].mean(), r[n_time // 2].mean()   # low elevation, transit

    low_d, mid_d = weight_error(sigma_from_time_differences(data, a1, a2, t).real)
    low_s, mid_s = weight_error(scale_relative_sigma(data, relative, a1, a2, t).real)

    # difference: the noisiest data gets far MORE weight than it deserves
    assert low_d > 2.0
    assert mid_d < 0.85
    # scaled: the weight column carries the time dependence, so both are right
    assert low_s == pytest.approx(1.0, rel=0.15)
    assert mid_s == pytest.approx(1.0, rel=0.15)


def test_too_little_data_reports_nan_rather_than_a_number():
    data, _, _, a1, a2, t, _ = _track(n_time=2)
    ratio, _ = noise_time_variation(data, a1, a2, t, n_bins=6)
    assert not np.isfinite(ratio)

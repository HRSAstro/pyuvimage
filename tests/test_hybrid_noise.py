"""Weights and differencing are strong on different axes. Use each where it is.

The weight column is radiometric -- Tsys, bandwidth, integration time, flagged
fraction. Elevation moves Tsys for the whole array at once, so the column
tracks the *time* axis exactly. It cannot see anything that happens after the
radiometry: residual phase errors, which grow with baseline length as the
atmosphere decorrelates faster over longer separations, or an antenna whose
calibration is worse than its Tsys suggests. Two baselines with equal Tsys look
identical to it however differently they behave.

Differencing is the reverse. It measures whatever really makes the data
scatter, decorrelation included, so the *baseline* axis is honest -- but it
pools each baseline's whole track into one number, so it is blind in time.
"""

import numpy as np
import pytest

from pyuvimage.noise import (
    baseline_weight_disagreement,
    hybrid_sigma,
    scale_relative_sigma,
    sigma_from_time_differences,
)

N_ANT = 24
N_TIME = 40


def _observation(elevation=True, decorrelation=True, bad_antenna=True, seed=0):
    """Both effects at once: what the weights know, and what they do not."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(N_ANT, k=1)
    n_base = a1.size

    pos = rng.normal(0, 300, (N_ANT, 2))
    pos[:6] *= 0.05                                    # a compact core
    length = np.hypot(*(pos[a1] - pos[a2]).T)
    length = 15 + (length - length.min()) / np.ptp(length) * 1485   # 15 m .. 1.5 km

    if elevation:
        el = np.radians(30 + 40 * np.sin(np.pi * np.arange(N_TIME) / (N_TIME - 1)))
        profile = 1.0 / np.sin(el)
        profile = profile / profile.min()
    else:
        profile = np.ones(N_TIME)

    tsys_ant = rng.uniform(1.0, 1.5, N_ANT)
    radiometric = np.sqrt(tsys_ant[a1] * tsys_ant[a2])   # the weights know this

    extra = np.ones(n_base)
    if decorrelation:
        extra = extra * (1.0 + 1.2 * (length / length.max()) ** 1.5)
    if bad_antenna:
        one_bad = np.ones(N_ANT)
        one_bad[7] = 2.5
        extra = extra * np.sqrt(one_bad[a1] * one_bad[a2])

    sigma_true = 0.004 * np.outer(profile, radiometric * extra).ravel()[None, :]
    claimed = 0.004 * np.outer(profile, radiometric).ravel()[None, :]

    ant1 = np.tile(a1, N_TIME)
    ant2 = np.tile(a2, N_TIME)
    time = np.repeat(np.arange(float(N_TIME)), n_base)
    sky = np.tile(rng.normal(0.0, 0.05, n_base), N_TIME)[None, :]
    data = (
        sky
        + rng.normal(0, 1, sigma_true.shape) * sigma_true
        + 1j * rng.normal(0, 1, sigma_true.shape) * sigma_true
    )
    relative = 7.0 * claimed * (1 + 1j)
    return data, relative, sigma_true, ant1, ant2, time, np.tile(length, N_TIME)


def _err(est, truth):
    return float(np.median(np.abs(est / truth - 1.0)))


def test_neither_alone_wins_when_both_effects_are_present():
    data, rel, truth, a1, a2, t, _ = _observation()
    d = _err(sigma_from_time_differences(data, a1, a2, t).real, truth)
    s = _err(scale_relative_sigma(data, rel, a1, a2, t).real, truth)
    # they land within a few per cent of each other: no free lunch either way
    assert abs(d - s) < 0.05


def test_hybrid_beats_both():
    data, rel, truth, a1, a2, t, _ = _observation()
    d = _err(sigma_from_time_differences(data, a1, a2, t).real, truth)
    s = _err(scale_relative_sigma(data, rel, a1, a2, t).real, truth)
    h = _err(hybrid_sigma(data, rel, a1, a2, t).real, truth)
    assert h < 0.75 * min(d, s)


def test_hybrid_is_difference_when_the_weights_carry_no_time_information():
    """It can only help: a flat time profile leaves the estimate untouched."""
    data, rel, _, a1, a2, t, _ = _observation(elevation=False)
    d = sigma_from_time_differences(data, a1, a2, t).real
    h = hybrid_sigma(data, rel, a1, a2, t).real
    assert np.allclose(h, d, rtol=0.05)


def test_hybrid_keeps_the_baseline_structure_difference_measured():
    data, rel, truth, a1, a2, t, length = _observation()
    d = sigma_from_time_differences(data, a1, a2, t).real
    h = hybrid_sigma(data, rel, a1, a2, t).real
    # averaged over the track, the hybrid must reproduce the differenced level
    n_base = np.triu_indices(N_ANT, k=1)[0].size
    d_b = d.reshape(-1)[:n_base]
    h_b = np.sqrt((h.reshape(N_TIME, n_base) ** 2).mean(axis=0))
    assert np.allclose(h_b, d_b, rtol=0.02)


def test_the_disagreement_diagnostic_finds_hidden_baseline_structure():
    data, rel, _, a1, a2, t, length = _observation(decorrelation=True)
    ratio, short, long_ = baseline_weight_disagreement(data, rel, a1, a2, t, length)
    assert ratio > 1.2
    assert long_ > short


def test_the_diagnostic_is_quiet_when_the_weights_are_right():
    data, rel, _, a1, a2, t, length = _observation(
        decorrelation=False, bad_antenna=False
    )
    ratio, _, _ = baseline_weight_disagreement(data, rel, a1, a2, t, length)
    assert ratio == pytest.approx(1.0, abs=0.2)


def test_the_diagnostic_degrades_to_nan_rather_than_guessing():
    data, rel, _, a1, a2, t, _ = _observation()
    ratio, _, _ = baseline_weight_disagreement(
        data, rel, a1, a2, t, np.full(data.shape[1], np.nan)
    )
    assert not np.isfinite(ratio)

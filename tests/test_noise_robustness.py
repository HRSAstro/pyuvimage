"""The noise estimator must survive bad visibilities.

On the first real dataset pyuvimage was pointed at, **one unflagged NaN in
6930 visibilities** turned every sigma into NaN: `np.std` of any window
containing it is NaN, and the global fallback was pooled from the same
poisoned differences, so there was nothing left to fall back to. The whole
noise map was unusable and the dataset would not even load.
"""

import numpy as np
import pytest

from pyuvimage.noise import sigma_from_time_differences


def _rows(n_ant=10, n_time=20):
    a1 = np.repeat(np.arange(n_ant), n_time)
    a2 = np.repeat(np.arange(n_ant, 2 * n_ant), n_time)
    t = np.tile(np.arange(float(n_time)), n_ant)
    return a1, a2, t


def _data(n_chan=7, sigma=1e-3, seed=0, n_ant=10, n_time=20):
    rng = np.random.default_rng(seed)
    n = n_ant * n_time
    return (rng.normal(0, sigma, (n_chan, n))
            + 1j * rng.normal(0, sigma, (n_chan, n)))


def test_recovers_the_input_sigma():
    a1, a2, t = _rows()
    s = sigma_from_time_differences(_data(sigma=1e-3), a1, a2, t)
    assert np.all(np.isfinite(s.real)) and np.all(s.real > 0)
    assert np.median(s.real) == pytest.approx(1e-3, rel=0.15)


def test_one_nan_does_not_poison_the_whole_map():
    """The exact regression: a single unflagged NaN."""
    a1, a2, t = _rows()
    clean = _data()
    dirty = clean.copy()
    dirty[3, 57] = np.nan

    s_clean = sigma_from_time_differences(clean, a1, a2, t)
    s_dirty = sigma_from_time_differences(dirty, a1, a2, t)
    assert np.all(np.isfinite(s_dirty.real)), "one NaN wiped out the noise map"
    assert np.all(s_dirty.real > 0)
    # and it must barely move the answer
    assert np.median(s_dirty.real) == pytest.approx(
        np.median(s_clean.real), rel=0.05)


def test_survives_a_heavily_corrupted_baseline():
    a1, a2, t = _rows()
    d = _data()
    d[:, 0:20] = np.nan          # one baseline entirely non-finite
    s = sigma_from_time_differences(d, a1, a2, t)
    assert np.all(np.isfinite(s.real)) and np.all(s.real > 0)
    # the dead baseline falls back to the global estimate, not to zero
    assert np.median(s[:, 0:20].real) == pytest.approx(
        np.median(s[:, 20:].real), rel=0.25)


def test_per_channel_mode_is_equally_robust():
    a1, a2, t = _rows()
    d = _data()
    d[2, 100] = np.nan
    s = sigma_from_time_differences(d, a1, a2, t, per_channel=True)
    assert np.all(np.isfinite(s.real)) and np.all(s.real > 0)


def test_single_integration_falls_back_to_robust_scatter():
    """No time differences available at all."""
    a1 = np.arange(50)
    a2 = np.arange(50, 100)
    t = np.zeros(50)
    d = _data(n_chan=3, n_ant=1, n_time=50, sigma=2e-3)
    s = sigma_from_time_differences(d, a1, a2, t)
    assert np.all(np.isfinite(s.real)) and np.all(s.real > 0)
    assert np.median(s.real) == pytest.approx(2e-3, rel=0.4)


def test_all_data_non_finite_raises_something_actionable():
    a1, a2, t = _rows()
    d = np.full((7, 200), np.nan + 1j * np.nan)
    with pytest.raises(ValueError, match="cannot estimate a noise level"):
        sigma_from_time_differences(d, a1, a2, t)


def test_constant_estimator_ignores_non_finite_samples():
    """The no-metadata fallback needs the same protection.

    Flagged samples are passed to it as NaN, so it sees them by design.
    """
    from pyuvimage.noise import sigma_constant_from_differences

    rng = np.random.default_rng(1)
    d = rng.normal(0, 5e-4, (7, 500)) + 1j * rng.normal(0, 5e-4, (7, 500))
    clean = sigma_constant_from_differences(d)
    d[2, 11] = np.nan
    d[5, 300] = np.nan + 1j * np.nan
    dirty = sigma_constant_from_differences(d)
    assert np.isfinite(dirty.real) and dirty.real > 0
    assert dirty.real == pytest.approx(clean.real, rel=0.05)


# ---------------------------------------------------------------- the pool

def test_few_differences_per_baseline_still_feed_the_global_pool():
    """MIN_DIFFS must gate the per-baseline sigma, never the pooled estimate.

    PJ0116 at 245 GHz has four timestamps, so every baseline had three rows and
    two usable differences -- under MIN_DIFFS. The export skipped each baseline
    *before* adding its differences to the pool, the pool came out empty, and
    the fallback returned the robust scatter of the visibilities: 5.111 mJy
    where the pooled differences give 3.696 mJy. That is a measurement of the
    source, not the noise, and it made the discrepancy principle stop at
    chi2/N = 1 when the true chi2/N was 0.52.
    """
    rng = np.random.default_rng(7)
    n_baselines, n_times = 200, 3          # 2 differences each, under MIN_DIFFS
    sigma_true = 0.004
    rows = n_baselines * n_times
    ant1 = np.repeat(np.arange(n_baselines), n_times)
    ant2 = ant1 + 1
    time = np.tile(np.arange(float(n_times)), n_baselines)
    # A real source varies from baseline to baseline, which is exactly why the
    # scatter of the visibilities is not a noise estimate: here it is ~13x the
    # true noise, and that is the number the broken path returned.
    source = np.repeat(rng.normal(0.0, 0.05, n_baselines), n_times)
    data = (
        source
        + rng.normal(0, sigma_true, rows)
        + 1j * rng.normal(0, sigma_true, rows)
    )[None, :]

    sig = sigma_from_time_differences(data, ant1, ant2, time)

    assert np.all(np.isfinite(sig.real)) and np.all(sig.real > 0)
    assert np.median(sig.real) == pytest.approx(sigma_true, rel=0.15)
    # the failure mode, stated as the tool would see it
    scatter = 1.4826 * np.median(np.abs(data.real - np.median(data.real)))
    assert scatter > 5 * sigma_true, "test is not exercising the failure mode"
    assert np.median(sig.real) < 0.25 * scatter

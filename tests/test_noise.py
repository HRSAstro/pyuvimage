import numpy as np

from pyuvimage.noise import sigma_from_time_differences


def test_recovers_known_sigma():
    rng = np.random.default_rng(0)
    n_baselines, n_times, n_chan = 6, 200, 2
    sigma_true = 0.05
    rows = n_baselines * n_times
    ant1 = np.repeat(np.arange(n_baselines), n_times)
    ant2 = ant1 + 1
    time = np.tile(np.arange(n_times, dtype=float), n_baselines)
    # constant sky signal per baseline + noise
    signal = np.repeat(rng.normal(0, 1.0, n_baselines), n_times)
    data = (
        signal
        + rng.normal(0, sigma_true, rows)
        + 1j * rng.normal(0, sigma_true, rows)
    )
    data = np.tile(data, (n_chan, 1))
    sig = sigma_from_time_differences(data, ant1, ant2, time)
    assert np.allclose(np.median(sig.real), sigma_true, rtol=0.15)
    assert np.allclose(np.median(sig.imag), sigma_true, rtol=0.15)


def test_signal_immunity():
    """A strong constant source must not inflate the noise estimate."""
    rng = np.random.default_rng(1)
    n_times = 500
    ant1 = np.zeros(n_times, dtype=int)
    ant2 = np.ones(n_times, dtype=int)
    time = np.arange(n_times, dtype=float)
    data = (100.0 + rng.normal(0, 0.01, n_times) * (1 + 0j))[None, :]
    sig = sigma_from_time_differences(data, ant1, ant2, time)
    assert np.median(sig.real) < 0.02  # not ~100

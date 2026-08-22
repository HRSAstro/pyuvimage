"""How `--noise difference` resolves the noise in time.

Hannah's suggestion, and it is not a separate mode: `difference` chunks the
track where there are enough integrations to support it and collapses to one
sigma per baseline where there is not, so the two reduce to the same thing.
`sigma_in_time_chunks` is the implementation and `sigma_from_time_differences`
is the limit it falls back to.

Why it is worth doing at all: pooling a baseline's whole track cannot see
elevation. The weight column can, but only radiometrically -- it stops at Tsys
and is blind to decorrelation. Since decorrelation is driven by the *same*
airmass as Tsys, the weights get the direction of the time dependence right and
the amplitude wrong. Chunking measures the real scatter inside each block, so
it sees Tsys and phase together, and needs no weights.

The cost is estimation noise: a sigma from `n` differences carries ~1/sqrt(2n),
so a block has to be long enough to hold enough of them.
"""

import numpy as np
import pytest

from pyuvimage.noise import (
    DEFAULT_CHUNK_SECONDS,
    sigma_from_time_differences,
    sigma_in_time_chunks,
)

N_ANT = 20


def _track(n_time, integration=6.0, seed=0, elevation=True):
    """Decorrelation driven by the same airmass as Tsys, as in reality."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(N_ANT, k=1)
    n_base = a1.size

    pos = rng.normal(0, 300, (N_ANT, 2))
    pos[:5] *= 0.05
    length = np.hypot(*(pos[a1] - pos[a2]).T)
    length = 15 + (length - length.min()) / np.ptp(length) * 1485

    if elevation:
        el = np.radians(30 + 40 * np.sin(np.pi * np.arange(n_time) / max(n_time - 1, 1)))
        airmass = 1.0 / np.sin(el)
        airmass = airmass / airmass.min()
    else:
        airmass = np.ones(n_time)

    # phase rms rises with airmass and with baseline length; sigma inflates by
    # exp(phi^2/2), so Tsys and decorrelation move together
    phi = 0.55 * np.outer(airmass**1.5, (length / length.max()) ** 0.6)
    sigma_true = 0.004 * np.outer(airmass, np.ones(n_base)) * np.exp(0.5 * phi**2)
    sigma_true = sigma_true.ravel()[None, :]

    ant1 = np.tile(a1, n_time)
    ant2 = np.tile(a2, n_time)
    time = np.repeat(np.arange(n_time) * float(integration), n_base)
    sky = np.tile(rng.normal(0.0, 0.05, n_base), n_time)[None, :]
    data = (
        sky
        + rng.normal(0, 1, sigma_true.shape) * sigma_true
        + 1j * rng.normal(0, 1, sigma_true.shape) * sigma_true
    )
    return data, sigma_true, ant1, ant2, time


def _err(est, truth):
    return float(np.median(np.abs(est / truth - 1.0)))


def test_chunking_beats_pooling_the_whole_track():
    """Two hours of 6 s integrations: 200 differences per 1200 s chunk."""
    data, truth, a1, a2, t = _track(n_time=1200, integration=6.0)
    whole = _err(sigma_from_time_differences(data, a1, a2, t).real, truth)
    chunked = _err(sigma_in_time_chunks(data, a1, a2, t, 1200.0).real, truth)
    assert chunked < 0.4 * whole


def test_a_chunk_as_long_as_the_track_reduces_to_difference():
    """The guarantee that makes the mode safe: it can only add resolution."""
    data, _, a1, a2, t = _track(n_time=200, integration=6.0)
    span = t.max() - t.min()
    whole = sigma_from_time_differences(data, a1, a2, t).real
    chunked = sigma_in_time_chunks(data, a1, a2, t, span * 2).real
    assert np.allclose(chunked, whole)


def test_it_follows_the_track_rather_than_flattening_it():
    n_time = 1200
    data, truth, a1, a2, t = _track(n_time=n_time, integration=6.0)
    est = sigma_in_time_chunks(data, a1, a2, t, 1200.0).real
    n_base = np.triu_indices(N_ANT, k=1)[0].size

    est_bt = est.reshape(n_time, n_base)
    true_bt = truth.reshape(n_time, n_base)

    # `difference` would give a perfectly flat column for every baseline
    spread = est_bt.max(axis=0) / est_bt.min(axis=0)
    assert np.median(spread) > 1.4

    # and the shape it recovers is the real one, high at the ends, low at
    # transit. A 2 h track in 1200 s chunks is a six-step staircase against a
    # smooth curve, so the correlation is capped well below 1 by the
    # discretisation rather than by estimation noise -- ~0.83 here.
    corr = [
        np.corrcoef(est_bt[:, b], true_bt[:, b])[0, 1]
        for b in range(0, n_base, 7)
    ]
    assert np.median(corr) > 0.75
    assert np.median(spread) < np.median(true_bt.max(axis=0) / true_bt.min(axis=0))


def test_too_short_a_chunk_is_worse_than_a_sensible_one():
    """Estimation noise: ~1/sqrt(2n) on a sigma from n differences."""
    data, truth, a1, a2, t = _track(n_time=120, integration=60.0)
    short = _err(sigma_in_time_chunks(data, a1, a2, t, 300.0).real, truth)
    sensible = _err(sigma_in_time_chunks(data, a1, a2, t, 1200.0).real, truth)
    assert sensible < short


def test_a_stationary_track_is_not_made_worse():
    data, truth, a1, a2, t = _track(n_time=600, integration=6.0, elevation=False)
    whole = _err(sigma_from_time_differences(data, a1, a2, t).real, truth)
    chunked = _err(sigma_in_time_chunks(data, a1, a2, t, 1200.0).real, truth)
    assert chunked < whole * 1.6


def test_too_few_integrations_to_chunk_falls_back_cleanly():
    """PJ0116's regime: four timestamps. There is nothing to chunk."""
    data, _, a1, a2, t = _track(n_time=4, integration=60.0)
    whole = sigma_from_time_differences(data, a1, a2, t).real
    chunked = sigma_in_time_chunks(data, a1, a2, t, 1200.0).real
    assert np.allclose(chunked, whole)


def test_the_default_chunk_suits_an_alma_execution():
    """1-1.5 h EB, 30-40% on calibrators -> ~45-60 min of target to divide."""
    assert DEFAULT_CHUNK_SECONDS == 600.0
    on_source_minutes = 50
    n_chunks = on_source_minutes * 60 / DEFAULT_CHUNK_SECONDS
    assert 4 <= n_chunks <= 8

"""MS weights are relative, not absolute -- so recompute the scale, keep the shape.

SIGMA is nominally 1/sqrt(2 dnu dt), but that only holds if the calibration put
it on an absolute scale. The ALMA pipeline sets weights proportional to the true
inverse variance without being equal to it, and split / mstransform / averaging
rescale them again (CASA memo on data weights,
https://casa.nrao.edu/Memos/CASA-data-weights.pdf).

What the column *does* carry is real structure: Tsys differences between
antennas, band edges, atmospheric lines. `scale_relative_sigma` keeps that and
replaces only the scale.
"""

import numpy as np
import pytest

from pyuvimage.noise import (
    MIN_DIFFS,
    scale_relative_sigma,
    sigma_from_time_differences,
)


def _observation(n_baselines=200, n_times=3, n_chan=1, seed=0,
                 sigma_floor=0.004, shape_range=(1.0, 3.0), weight_scale=7.0):
    """A source plus noise whose level varies per baseline by a known shape."""
    rng = np.random.default_rng(seed)
    rows = n_baselines * n_times
    ant1 = np.repeat(np.arange(n_baselines), n_times)
    ant2 = ant1 + 1
    time = np.tile(np.arange(float(n_times)), n_baselines)

    shape = np.repeat(rng.uniform(*shape_range, n_baselines), n_times)
    shape = np.broadcast_to(shape, (n_chan, rows))
    true_sigma = sigma_floor * shape

    # a real source: constant per baseline, different between them
    source = np.repeat(rng.normal(0.0, 0.05, n_baselines), n_times)
    source = np.broadcast_to(source, (n_chan, rows))
    data = (
        source
        + rng.normal(0, 1, (n_chan, rows)) * true_sigma
        + 1j * rng.normal(0, 1, (n_chan, rows)) * true_sigma
    )
    # the weight column knows the shape but is off by a constant factor
    relative = weight_scale * true_sigma * (1 + 1j)
    return data, relative, true_sigma, ant1, ant2, time


def test_recovers_the_absolute_scale():
    data, relative, true_sigma, a1, a2, t = _observation()
    out = scale_relative_sigma(data, relative, a1, a2, t)
    assert np.median(out.real / true_sigma) == pytest.approx(1.0, rel=0.05)


def test_preserves_the_relative_shape_exactly():
    """The whole point: only one number changes."""
    data, relative, _, a1, a2, t = _observation()
    out = scale_relative_sigma(data, relative, a1, a2, t)
    factor = out.real / relative.real
    assert np.allclose(factor, factor.flat[0])


def test_beats_plain_differencing_when_baselines_are_short():
    """PJ0116's regime: too few integrations for a per-baseline sigma.

    Every baseline falls below MIN_DIFFS, so `sigma_from_time_differences`
    gives one flat number and all per-baseline structure is lost. Whitening
    first keeps the structure and pools the same differences for the scale.
    """
    n_times = 3          # 2 differences per baseline
    assert n_times - 1 < MIN_DIFFS
    data, relative, true_sigma, a1, a2, t = _observation(n_times=n_times)

    flat = sigma_from_time_differences(data, a1, a2, t)
    scaled = scale_relative_sigma(data, relative, a1, a2, t)

    # the plain estimate really has collapsed to a single value
    assert np.allclose(flat.real, flat.real.flat[0])

    err_flat = np.median(np.abs(flat.real / true_sigma - 1.0))
    err_scaled = np.median(np.abs(scaled.real / true_sigma - 1.0))
    assert err_scaled < 0.25 * err_flat


def test_a_wrong_shape_is_not_rescued():
    """Honesty check: if the weights' shape is wrong, the answer is wrong.

    `scaled` assumes the column's *shape* is right and only its scale is not.
    That assumption is not verified and cannot be rescued after the fact -- a
    scrambled shape gives both the wrong per-cell noise and, because the
    whitened variance is then not uniform, the wrong overall level too. This
    is the documented cost of the mode, and `difference` is the fallback when
    the weight column is not trusted at all.
    """
    rng = np.random.default_rng(3)
    data, _, true_sigma, a1, a2, t = _observation(seed=3)
    scrambled = rng.permutation(true_sigma.ravel()).reshape(true_sigma.shape)
    out = scale_relative_sigma(data, scrambled * (1 + 1j), a1, a2, t)
    per_cell_error = np.median(np.abs(out.real / true_sigma - 1.0))
    assert per_cell_error > 0.2

    # and `difference`, which ignores the column, is unharmed by it
    flat = sigma_from_time_differences(data, a1, a2, t)
    assert np.median(flat.real) == pytest.approx(np.median(true_sigma), rel=0.2)


def test_shape_mismatch_is_rejected():
    data, relative, _, a1, a2, t = _observation()
    with pytest.raises(ValueError, match="relative_sigma shape"):
        scale_relative_sigma(data, relative[:, :-1], a1, a2, t)


def test_flagged_and_nonfinite_cells_do_not_set_the_scale():
    data, relative, true_sigma, a1, a2, t = _observation(seed=5)
    data = data.copy()
    relative = relative.copy()
    data[0, ::17] = np.nan          # unflagged NaNs, as a real MS can carry
    relative[0, ::23] = 0.0         # zero weight
    out = scale_relative_sigma(data, relative, a1, a2, t)
    good = np.isfinite(out.real) & (out.real > 0)
    assert np.median(out.real[good] / true_sigma[good]) == pytest.approx(1.0, rel=0.1)


def test_multichannel_shape_is_carried_through():
    """Band edges: per-channel weights must survive into the noise map."""
    n_chan = 8
    data, relative, true_sigma, a1, a2, t = _observation(n_chan=n_chan, seed=9)
    relative = relative.copy()
    edge = np.ones(n_chan)
    edge[0] = edge[-1] = 3.0        # noisy band edges
    relative *= edge[:, None]
    out = scale_relative_sigma(data, relative, a1, a2, t)
    interior = np.median(out.real[1:-1])
    assert np.median(out.real[0]) == pytest.approx(3.0 * interior, rel=0.05)
    assert np.median(out.real[-1]) == pytest.approx(3.0 * interior, rel=0.05)


def test_weights_that_change_between_integrations_do_not_leak_the_sky():
    """Difference first, normalise after -- the order is the whole point.

    Whitening first and differencing the result gives
        d(V/s) = dV/s_bar + V d(1/s)
    and the second term is the sky coming back in whenever the weight changes
    between two integrations, which flagging routinely makes it do. That is
    precisely what differencing exists to prevent.

    Measured on this mock, with a source 12x the noise: whitening first was
    harmless at constant weights, tripled the error at 5% weight jitter, and at
    20% was an order of magnitude worse than ignoring the weight column
    entirely. Differencing the raw visibilities and dividing by the pair's
    combined sigma cancels the sky in the numerator exactly.
    """
    rng = np.random.default_rng(11)
    n_baselines, n_times = 66, 10
    rows = n_baselines * n_times
    ant1 = np.repeat(np.arange(n_baselines), n_times)
    ant2 = ant1 + 1
    time = np.tile(np.arange(float(n_times)), n_baselines)

    sigma_true = 0.004 * np.repeat(rng.uniform(1.0, 2.5, n_baselines), n_times)
    sigma_true = sigma_true[None, :]
    # a source 12x the noise: the term that leaks is proportional to it
    sky = np.repeat(rng.normal(0.0, 0.05, n_baselines), n_times)[None, :]
    data = (
        sky
        + rng.normal(0, 1, sigma_true.shape) * sigma_true
        + 1j * rng.normal(0, 1, sigma_true.shape) * sigma_true
    )

    def error(jitter):
        rel = 7.0 * sigma_true * (1 + 1j)
        rel = rel * (1.0 + jitter * rng.normal(0, 1, rel.shape))
        out = scale_relative_sigma(data, rel, ant1, ant2, time)
        return np.median(np.abs(out.real / sigma_true - 1.0))

    steady = error(0.0)
    jittered = error(0.20)
    plain = np.median(
        np.abs(
            sigma_from_time_differences(data, ant1, ant2, time).real / sigma_true
            - 1.0
        )
    )

    assert steady < 0.05
    # 20% jitter means the shape really is ~20% wrong, so some loss is honest.
    # What must not happen is blowing past the estimator that ignores weights.
    assert jittered < 1.5 * plain

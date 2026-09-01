"""The vectorised estimators must give the per-baseline loop's answer exactly.

Every estimator in `noise.py` used to walk the baselines in Python: find the
rows, sort them by time, difference neighbours, drop pairs that straddle a
calibrator gap, take the standard deviation. That block was written five
times, once more in casa_export.py, and on 2e5 rows it was the run time. It
is now one helper, `adjacent_pairs`, feeding a blocked two-pass reduction,
`grouped_std`.

The tests here pin the new code to a deliberately naive re-implementation of
the old loop -- flags as NaN, sporadic NaNs, calibrator gaps, baselines with
different row counts, rows arriving in no particular order -- and demand
agreement to 1e-12. Anything looser would let a wrong grouping through as
"close enough".
"""

import warnings

import numpy as np
import pytest

from pyuvimage import noise as noise_mod
from pyuvimage.noise import (
    MIN_DIFFS,
    adjacent_pairs,
    auto_max_gap,
    baseline_sigma_from_pairs,
    baseline_weight_disagreement,
    hybrid_sigma,
    noise_time_variation,
    scale_relative_sigma,
    sigma_from_time_differences,
    sigma_in_time_chunks,
)


def _observation(n_ant=12, n_time=40, n_chan=5, seed=0, gaps=True, ragged=True):
    """Shuffled rows, calibrator gaps, ragged baselines, NaNs: the real thing."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(n_ant, k=1)
    n_base = a1.size
    t_int = np.arange(n_time) * 6.0
    if gaps:
        t_int = t_int + 120.0 * (np.arange(n_time) // 10)   # a visit every 10
    ant1 = np.tile(a1, n_time)
    ant2 = np.tile(a2, n_time)
    time = np.repeat(t_int, n_base)
    perm = rng.permutation(ant1.size)
    ant1, ant2, time = ant1[perm], ant2[perm], time[perm]
    if ragged:
        keep = rng.random(ant1.size) > 0.2
        keep[:3] = True
        ant1, ant2, time = ant1[keep], ant2[keep], time[keep]
    n = ant1.size
    sigma = 0.004 * (1 + (ant1 * 7 + ant2) % 5 / 5.0)
    sky = 0.05 * np.sin(0.37 * (ant1 * 100 + ant2))
    data = (
        sky[None, :]
        + rng.normal(0, 1, (n_chan, n)) * sigma
        + 1j * rng.normal(0, 1, (n_chan, n)) * sigma
    )
    data[:, rng.random(n) < 0.05] = np.nan            # flagged rows, as NaN
    data[rng.random((n_chan, n)) < 0.03] = np.nan     # sporadic bad cells
    rel = (7.0 * sigma[None, :] * (1 + 0.1 * rng.random((n_chan, n)))) * (1 + 1j)
    return data, ant1, ant2, time, rel


# ------------------------------------------------ the loop, written plainly

def _loop_pairs(ant1, ant2, time, gap):
    """Per baseline, sort by time, take neighbours within the gap."""
    baseline = ant1.astype(np.int64) * 100000 + ant2.astype(np.int64)
    out = []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(time[rows], kind="stable")]
        for i in range(rows.size - 1):
            if time[rows[i + 1]] - time[rows[i]] <= gap:
                out.append((rows[i], rows[i + 1], b))
    return out


def _loop_sigma(data, ant1, ant2, time, per_channel=False):
    """`sigma_from_time_differences` as the original loop computed it."""
    data = np.atleast_2d(data)
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    gap = auto_max_gap(time)
    baseline = ant1.astype(np.int64) * 100000 + ant2.astype(np.int64)
    sigma = np.full(data.shape, np.nan + 1j * np.nan)
    pool_re, pool_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(time[rows], kind="stable")]
        if rows.size < 2:
            continue
        diff = np.diff(data[:, rows], axis=1)
        ok = finite[:, rows]
        usable = ok[:, 1:] & ok[:, :-1] & (np.diff(time[rows]) <= gap)[None, :]
        d_re = np.where(usable, diff.real, np.nan)
        d_im = np.where(usable, diff.imag, np.nan)
        if per_channel:
            n = usable.sum(axis=1)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                s_re = np.where(n >= MIN_DIFFS, np.nanstd(d_re, axis=1) / np.sqrt(2), np.nan)
                s_im = np.where(n >= MIN_DIFFS, np.nanstd(d_im, axis=1) / np.sqrt(2), np.nan)
            sigma[:, rows] = (s_re + 1j * s_im)[:, None]
        elif usable.sum() >= MIN_DIFFS:
            sigma[:, rows] = np.nanstd(d_re) / np.sqrt(2) + 1j * np.nanstd(d_im) / np.sqrt(2)
        if usable.any():
            pool_re.append(d_re[usable])
            pool_im.append(d_im[usable])
    g = np.std(np.concatenate(pool_re)) / np.sqrt(2) + 1j * (
        np.std(np.concatenate(pool_im)) / np.sqrt(2))
    bad = ~np.isfinite(sigma.real) | (sigma.real <= 0) | ~np.isfinite(sigma.imag) | (sigma.imag <= 0)
    sigma[bad] = g
    return sigma


def _loop_time_variation(data, ant1, ant2, time, n_bins=3):
    data = np.atleast_2d(data)
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    t = np.asarray(time, dtype=float)
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    pools = [[] for _ in range(n_bins)]
    for i, j, _ in _loop_pairs(ant1, ant2, t, auto_max_gap(t)):
        k = min(max(np.searchsorted(edges, t[i], "right") - 1, 0), n_bins - 1)
        ok = finite[:, i] & finite[:, j]
        pools[k].append((data[ok, j] - data[ok, i]).real)
    sig = np.full(n_bins, np.nan)
    for k, parts in enumerate(pools):
        v = np.concatenate(parts) if parts else np.empty(0)
        if v.size >= MIN_DIFFS:
            sig[k] = np.std(v) / np.sqrt(2)
    good = sig[np.isfinite(sig) & (sig > 0)]
    return (good.max() / good.min() if good.size >= 2 else np.nan), sig


def _loop_chunks(data, ant1, ant2, time, chunk_seconds):
    """`sigma_in_time_chunks` as the nested baselines x chunks loop had it."""
    data = np.atleast_2d(data)
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    t = np.asarray(time, dtype=float)
    whole = _loop_sigma(data, ant1, ant2, time)
    gap = auto_max_gap(t)
    n_chunk = int(np.ceil((t.max() - t.min()) / chunk_seconds))
    which = np.clip(((t - t.min()) / chunk_seconds).astype(int), 0, n_chunk - 1)
    baseline = ant1.astype(np.int64) * 100000 + ant2.astype(np.int64)
    per_bc, pooled = {}, [[] for _ in range(n_chunk)]
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(t[rows], kind="stable")]
        if rows.size < 2:
            continue
        diff = np.diff(data[:, rows], axis=1)
        ok = finite[:, rows]
        good = ok[:, 1:] & ok[:, :-1]
        c0, c1 = which[rows[:-1]], which[rows[1:]]
        inside = (c0 == c1) & (np.diff(t[rows]) <= gap)
        for c in np.unique(c0[inside]):
            sel = good & ((c0 == c) & inside)[None, :]
            if not sel.any():
                continue
            vals = np.concatenate([diff.real[sel], diff.imag[sel]])
            pooled[c].append(vals)
            if sel.sum() >= MIN_DIFFS:
                per_bc[(b, c)] = np.std(vals) / np.sqrt(2)
    level = np.full(n_chunk, np.nan)
    for c, parts in enumerate(pooled):
        v = np.concatenate(parts) if parts else np.empty(0)
        if v.size >= 2 * MIN_DIFFS:
            level[c] = np.std(v) / np.sqrt(2)
    use = np.isfinite(level) & (level > 0)
    if use.sum() < 2:
        return whole
    g = np.where(use, level / np.sqrt(np.mean(level[use] ** 2)), 1.0)
    out = whole.copy()
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        for c in np.unique(which[rows]):
            cols = rows[which[rows] == c]
            own = per_bc.get((b, c))
            if own is not None and own > 0:
                out[:, cols] = own + 1j * own
            else:
                out[:, cols] = whole[:, cols] * g[c]
    return out


# ----------------------------------------------------------------- the pairs

def test_adjacent_pairs_matches_the_loop_including_gaps_and_ragged_baselines():
    _, a1, a2, t, _ = _observation()
    pairs = adjacent_pairs(a1, a2, t)
    expected = _loop_pairs(a1, a2, t, pairs.max_gap)
    got = sorted(zip(pairs.first.tolist(), pairs.second.tolist()))
    assert got == sorted((i, j) for i, j, _ in expected)
    assert pairs.max_gap == auto_max_gap(t)
    # every row has a baseline id, even the ones with no pairs
    assert pairs.row_baseline.shape == a1.shape
    assert pairs.n_baselines == np.unique(a1 * 100000 + a2).size
    # the pairs really are contiguous per baseline, which the reductions rely on
    assert np.all(np.diff(pairs.pair_baseline) >= 0)


def test_the_gap_guard_drops_exactly_the_pairs_that_span_a_visit():
    _, a1, a2, t, _ = _observation(ragged=False)
    guarded = adjacent_pairs(a1, a2, t)
    everything = adjacent_pairs(a1, a2, t, max_gap=np.inf)
    dt = t[everything.second] - t[everything.first]
    assert guarded.first.size == np.count_nonzero(dt <= guarded.max_gap)
    assert guarded.first.size < everything.first.size


# ------------------------------------------------------------ the estimators

@pytest.mark.parametrize("per_channel", [False, True])
def test_sigma_from_time_differences_is_the_loop_to_1e12(per_channel):
    data, a1, a2, t, _ = _observation()
    got = sigma_from_time_differences(data, a1, a2, t, per_channel=per_channel)
    want = _loop_sigma(data, a1, a2, t, per_channel=per_channel)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_sigma_in_time_chunks_is_the_nested_loop_to_1e12():
    data, a1, a2, t, _ = _observation(n_time=80)
    got = sigma_in_time_chunks(data, a1, a2, t, chunk_seconds=300.0)
    want = _loop_chunks(data, a1, a2, t, 300.0)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    # and it really resolved something, or the test is vacuous
    assert not np.allclose(got, sigma_from_time_differences(data, a1, a2, t))


def test_noise_time_variation_is_the_loop_to_1e12():
    data, a1, a2, t, _ = _observation()
    ratio, blocks = noise_time_variation(data, a1, a2, t)
    want_ratio, want_blocks = _loop_time_variation(data, a1, a2, t)
    np.testing.assert_allclose(blocks, want_blocks, rtol=1e-12)
    assert ratio == pytest.approx(want_ratio, rel=1e-12)


def test_blocking_does_not_change_the_answer():
    """The sweep is blocked to bound memory; the block size must be invisible."""
    data, a1, a2, t, rel = _observation(n_chan=7)
    reference = sigma_in_time_chunks(data, a1, a2, t, 300.0)
    hybrid_ref = hybrid_sigma(data, rel, a1, a2, t)
    saved = noise_mod.BLOCK_ELEMENTS
    try:
        for block in (7, 64, 1000):          # one column, a few, many
            noise_mod.BLOCK_ELEMENTS = block
            np.testing.assert_allclose(
                sigma_in_time_chunks(data, a1, a2, t, 300.0), reference,
                rtol=1e-12, atol=0,
            )
            np.testing.assert_allclose(
                hybrid_sigma(data, rel, a1, a2, t), hybrid_ref, rtol=1e-12, atol=0,
            )
    finally:
        noise_mod.BLOCK_ELEMENTS = saved


def test_baseline_sigma_from_pairs_is_what_casa_export_needs():
    """casa_export.py builds its noise map from this, not from its own loop."""
    data, a1, a2, t, _ = _observation()
    pairs = adjacent_pairs(a1, a2, t)
    est = baseline_sigma_from_pairs(data, pairs)
    want = _loop_sigma(data, a1, a2, t)
    # the pooled value is what fills baselines below MIN_DIFFS, and the
    # per-baseline values are those baselines' own sigma
    per_row = (est.sigma_re + 1j * est.sigma_im)[pairs.row_baseline]
    filled = np.where(np.isfinite(per_row), per_row, est.pool_re + 1j * est.pool_im)
    np.testing.assert_allclose(np.broadcast_to(filled, data.shape), want, rtol=1e-12)
    assert est.count.shape == (pairs.n_baselines,)
    assert est.pool_count == int(est.count.sum())


# ---------------------------------------------- the time-variation gap guard

def test_noise_time_variation_no_longer_reads_calibrator_gaps_as_variation():
    """Item 6: this diagnostic was the one estimator without the gap guard.

    A stationary track whose scans are separated by calibrator visits: the
    one pair per baseline that straddles each visit carries the source's
    change across it. Without the guard those pairs inflated whichever block
    held the scan boundary, and a perfectly stationary noise level was
    reported as varying.
    """
    rng = np.random.default_rng(3)
    n_ant, n_time = 16, 60
    a1, a2 = np.triu_indices(n_ant, k=1)
    n_base = a1.size
    t_int = np.arange(n_time) * 6.0 + 300.0 * (np.arange(n_time) // 20)
    ant1, ant2 = np.tile(a1, n_time), np.tile(a2, n_time)
    time = np.repeat(t_int, n_base)
    # the source changes across every visit, the noise never does
    scan = np.repeat(np.arange(n_time) // 20, n_base)
    source = 0.5 * np.sin(1.3 * scan + 0.1 * (ant1 * 10 + ant2))
    sigma = 0.004
    data = (source + rng.normal(0, sigma, ant1.size)
            + 1j * rng.normal(0, sigma, ant1.size))[None, :]

    ratio, blocks = noise_time_variation(data, ant1, ant2, time)
    assert ratio == pytest.approx(1.0, abs=0.1)
    assert np.allclose(blocks, sigma, rtol=0.1)

    # the failure mode, stated: without the guard the straddling pairs lift
    # a block by far more than the stationary noise allows
    _, unguarded = _loop_time_variation_no_guard(data, ant1, ant2, time)
    assert unguarded.max() / unguarded.min() > 1.25


def _loop_time_variation_no_guard(data, ant1, ant2, time, n_bins=3):
    t = np.asarray(time, dtype=float)
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    pools = [[] for _ in range(n_bins)]
    for i, j, _ in _loop_pairs(ant1, ant2, t, np.inf):
        k = min(max(np.searchsorted(edges, t[i], "right") - 1, 0), n_bins - 1)
        pools[k].append((data[:, j] - data[:, i]).real)
    sig = np.array([np.std(np.concatenate(p)) / np.sqrt(2) for p in pools])
    return sig.max() / sig.min(), sig


# ------------------------------------------- the whole-track estimate, once

def test_a_precomputed_whole_track_estimate_is_used_as_is():
    """Item 9: the import computes it once and hands it to the others."""
    data, a1, a2, t, rel = _observation(n_time=80)
    whole = sigma_from_time_differences(data, a1, a2, t)
    length = np.hypot(a1 - a2, a1 + a2).astype(float) + 1.0

    for own, given in [
        (sigma_in_time_chunks(data, a1, a2, t, 300.0),
         sigma_in_time_chunks(data, a1, a2, t, 300.0, whole_track_sigma=whole)),
        (hybrid_sigma(data, rel, a1, a2, t),
         hybrid_sigma(data, rel, a1, a2, t, whole_track_sigma=whole)),
        (baseline_weight_disagreement(data, rel, a1, a2, t, length),
         baseline_weight_disagreement(data, rel, a1, a2, t, length,
                                      whole_track_sigma=whole)),
    ]:
        np.testing.assert_allclose(np.asarray(given), np.asarray(own), rtol=1e-12)

    # and it really is used, not recomputed: a wrong one changes the answer
    wrong = whole * 3.0
    assert not np.allclose(
        hybrid_sigma(data, rel, a1, a2, t, whole_track_sigma=wrong),
        hybrid_sigma(data, rel, a1, a2, t),
    )
    with pytest.raises(ValueError, match="whole_track_sigma shape"):
        sigma_in_time_chunks(data, a1, a2, t, 300.0, whole_track_sigma=whole[:, :-1])


def test_scale_relative_sigma_matches_a_plain_pooled_computation():
    data, a1, a2, t, rel = _observation()
    out = scale_relative_sigma(data, rel, a1, a2, t)
    usable = (np.isfinite(data.real) & np.isfinite(data.imag)
              & np.isfinite(rel.real) & (rel.real > 0))
    pairs = _loop_pairs(a1, a2, t, auto_max_gap(t))
    pool_re, pool_im = [], []
    for i, j, _ in pairs:
        ok = usable[:, i] & usable[:, j]
        d = data[ok, j] - data[ok, i]
        bar = np.sqrt(0.5 * (rel.real[ok, i] ** 2 + rel.real[ok, j] ** 2))
        pool_re.append(d.real / bar)
        pool_im.append(d.imag / bar)
    k_re = np.std(np.concatenate(pool_re)) / np.sqrt(2)
    k_im = np.std(np.concatenate(pool_im)) / np.sqrt(2)
    np.testing.assert_allclose(out.real, rel.real * k_re, rtol=1e-12)
    np.testing.assert_allclose(out.imag, rel.imag * k_im, rtol=1e-12)

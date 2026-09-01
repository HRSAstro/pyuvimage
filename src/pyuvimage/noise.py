"""Noise estimation from the data themselves.

The estimator differences visibilities that are adjacent in time on the same
baseline (and correlation/channel): the sky signal is essentially identical in
consecutive integrations, so the difference is pure noise with variance
2 sigma^2 (cf. https://github.com/tikk3r/maser).  Real and imaginary parts are
kept separate.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np


# casa_export.py imports `adjacent_pairs` and `baseline_sigma_from_pairs` from
# here and runs under CASA's own python, so this module must stay numpy-only:
# no astropy, nothing from the rest of the package.

logger = logging.getLogger("pyuvimage")

MIN_DIFFS = 4          # fewer than this and a per-baseline sigma is noise itself


def _robust_sigma(values: np.ndarray) -> float:
    """Median absolute deviation, scaled to a Gaussian sigma.

    Used only as a last resort, and deliberately robust: by the time we need
    it the data have already shown they contain something pathological.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


GAP_TOLERANCE = 3.0     # a pair is "adjacent" out to this many median steps


def auto_max_gap(time: np.ndarray, tolerance: float = GAP_TOLERANCE) -> float:
    """Largest time step that still counts as two adjacent integrations.

    Differencing assumes the sky is the same in both samples. That holds
    between consecutive integrations and fails across a **calibrator visit**:
    a typical ALMA execution spends 30-40% of its 1-1.5 h on calibrators, so
    the target's timestamps come in scans separated by gaps of a minute or
    two. Over 90 s the earth turns a 1.5 km baseline through ~8 klambda, which
    is a real change in the visibility of anything bigger than an arcsecond --
    so a difference spanning the gap measures the source, not the noise, and
    inflates sigma.

    The guard is relative, so it adapts to whatever integration time the data
    was averaged to, and it is deliberately loose: three median steps keeps
    genuinely consecutive samples even when the spacing is a little ragged.

    Returns `inf` when there is nothing to go on, which keeps every pair.
    """
    t = np.unique(np.asarray(time, dtype=float))
    if t.size < 3:
        return float("inf")
    steps = np.diff(t)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return float("inf")
    return float(tolerance * np.median(steps))


def _cols(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`x[:, idx]` for a (n_chan, n_vis) array, only fast.

    Fancy indexing along the second axis is a slow path in numpy -- on a
    (64, 1.7e5) complex array `data[:, idx]` took 0.8 s where `np.take` with
    `axis=1` took 0.12 s for the same gather. Every estimator gathers its pair
    endpoints this way, so the difference is most of their run time.
    """
    return np.take(x, idx, axis=1)


class AdjacentPairs(NamedTuple):
    """Every pair of rows that may be differenced, and the grouping behind it.

    `first[k]` and `second[k]` index the original rows of pair `k`: the same
    baseline, consecutive in time, no more than `max_gap` apart. Pairs come
    out sorted by baseline and then by time, so a baseline's pairs are one
    contiguous run -- which is what lets `np.bincount` replace a loop.

    `row_baseline` is a dense 0..n-1 id per *row* (not per pair), in the order
    of `np.unique((antenna1, antenna2))`, and `pair_baseline` the same id per
    pair. A baseline with a single row, or none within the gap, simply has no
    pairs; it still has an id, so callers can fill it from the pool.
    """

    first: np.ndarray
    second: np.ndarray
    pair_baseline: np.ndarray
    row_baseline: np.ndarray
    n_baselines: int
    max_gap: float


def adjacent_pairs(
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    max_gap: float | None = None,
) -> AdjacentPairs:
    """Rows on the same baseline that are adjacent in time.

    This is the one place the "group by baseline, sort by time, take
    consecutive pairs, drop those spanning a gap" logic lives. It used to be
    written out in every estimator and once more in casa_export.py -- six
    copies, each a per-baseline Python loop with a fancy-indexed
    `data[:, rows]` gather inside it. At a few channels that loop was most of
    the cost (2e5 rows by 8 channels: 2-3.5x faster without it); at 64
    channels both versions are bound by memory traffic over the (n_chan,
    n_pairs) differences and the gain is 1-2.5x. The import used to run the
    whole-track estimator three times over on the same data; it now computes
    it once and passes it to the estimators that need it (`whole_track_sigma`).

    Returns the pairs as index arrays so that every estimator can gather its
    (n_chan, n_pairs) differences in one vectorised step and reduce them per
    baseline with `grouped_std`.

    `max_gap=None` means `auto_max_gap(time)`; pass `np.inf` to keep every
    consecutive pair regardless of spacing.

    Ties in time within a baseline are ordered by row (stable sort), which is
    the one respect in which this can differ from the old per-baseline
    `np.argsort`: a real MS never has two rows of one baseline at the same
    TIME within one DATA_DESC_ID, so in practice the pairs are the same.
    """
    a1 = np.asarray(antenna1).astype(np.int64, copy=False)
    a2 = np.asarray(antenna2).astype(np.int64, copy=False)
    t = np.asarray(time, dtype=float)
    key = a1 * 100000 + a2
    uniq, row_baseline = np.unique(key, return_inverse=True)
    row_baseline = np.asarray(row_baseline).ravel()
    gap = auto_max_gap(t) if max_gap is None else float(max_gap)

    order = np.lexsort((t, row_baseline))          # by baseline, then time
    b_sorted = row_baseline[order]
    t_sorted = t[order]
    # consecutive in the sorted order, same baseline, close enough in time
    pair = b_sorted[1:] == b_sorted[:-1]
    with np.errstate(invalid="ignore"):
        pair &= (t_sorted[1:] - t_sorted[:-1]) <= gap
    k = np.flatnonzero(pair)
    return AdjacentPairs(
        first=order[k],
        second=order[k + 1],
        pair_baseline=b_sorted[k],
        row_baseline=row_baseline,
        n_baselines=int(uniq.size),
        max_gap=gap,
    )


#: Values per temporary in the blocked reductions. 1M float64 is 8 MB: under
#: glibc's 32 MB mmap threshold, so freed temporaries are recycled from the
#: heap instead of being handed back to the kernel and page-faulted in again
#: on the next block -- measured at 4M values the same estimator swung
#: between 1.4 and 3.0 s from run to run, at 1M it sat at 1.1-1.3 s. Still
#: large enough that the Python loop over blocks is negligible (ten blocks
#: for 2e5 rows by 64 channels).
BLOCK_ELEMENTS = 1 << 20

#: The second sweep re-reads the same values as the first. Up to this many
#: bytes they are kept from the first sweep instead of being gathered again,
#: which halves the gather cost; above it (a spectral cube: 3840 channels by
#: 2e5 rows is 12 GB of differences) they are recomputed so that memory stays
#: bounded by the block size whatever the input.
CACHE_BYTES = 512 << 20


def _blocks(n: int, n_chan: int):
    step = max(1, BLOCK_ELEMENTS // max(int(n_chan), 1))
    for start in range(0, int(n), step):
        yield slice(start, min(int(n), start + step))


class Grouping(NamedTuple):
    """How `grouped_std` should reduce the pairs.

    `ids` gives each pair's group; `per_channel` keeps the channel axis (the
    ids must then be sorted, as `adjacent_pairs` delivers them, because the
    per-channel path reduces contiguous runs); `pool_components` pools the
    real and imaginary parts into one sigma, as the chunked estimator wants;
    `mask`, if given, drops pairs from this grouping only -- so the chunked
    estimator can reduce its within-chunk pairs and the whole-track pairs in
    the same sweep.
    """

    ids: np.ndarray
    n_groups: int
    per_channel: bool = False
    pool_components: bool = False
    mask: np.ndarray | None = None


def grouped_std(
    n_pairs: int,
    n_chan: int,
    n_comp: int,
    block_values,
    groupings: list[Grouping],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Count and population standard deviation within groups of pairs.

    `block_values(sl)` must return `(values, usable)` for the pairs in slice
    `sl`: `values` real, shape (n_comp, n_chan, B) -- the components being
    the real and imaginary differences, say -- and `usable` (n_chan, B).
    Every grouping is reduced in the same sweep, so the gathers behind
    `block_values` are shared between, e.g., the per-baseline sigma and the
    global pool.

    Two passes, like `np.std`: the mean first, then the squared deviations.
    The one-pass E[x^2] - E[x]^2 form would save the second sweep but loses
    digits whenever the mean is not small against the scatter, and a
    differenced visibility with a residual source-variation term is exactly
    that case. Groups with no usable values come back as NaN, never as 0.

    Why blocked: the estimators used to loop over baselines, gathering a
    (n_chan, ~200) block each -- slow in Python, but tiny in memory. A first
    vectorised version materialised every (n_chan, n_pairs) temporary at
    once, and at 64 channels spent longer page-faulting fresh 160 MB arrays
    than the loop had spent looping; on a 3840-channel cube it would have
    needed several times the data's own size. Sweeping the pairs in blocks
    of `BLOCK_ELEMENTS` values keeps every temporary at a few MB whatever the
    channel count, and the per-block reductions are plain `np.bincount`
    (channels pooled) or `np.add.reduceat` (per channel). The blocks from the
    first sweep are kept for the second while they total under `CACHE_BYTES`,
    so the gathers -- most of the remaining cost -- happen once on anything
    but a cube.

    Returns
    -------
    For each grouping, `(count, std)`. Shapes: `count` is (n_groups,) or
    (n_chan, n_groups); `std` has a leading component axis of length
    `n_comp` unless `pool_components`, in which case it matches `count` and
    `count` counts real and imaginary values together.
    """
    n_pairs = int(n_pairs)
    acc = []
    for g in groupings:
        shape = (n_chan, g.n_groups) if g.per_channel else (g.n_groups,)
        acc.append({
            "count": np.zeros(shape),
            "total": np.zeros((n_comp,) + shape),
            "ss": np.zeros((n_comp,) + shape),
        })

    def _reduce(g: Grouping, x: np.ndarray, u: np.ndarray, sl: slice, out, key: str):
        """Add the block's sums of `x` (n_comp, n_chan, B) into `out[key]`."""
        ids = g.ids[sl]
        if not g.per_channel:
            xs = x.sum(axis=1)                              # (n_comp, B)
            for c in range(n_comp):
                out[key][c] += np.bincount(ids, weights=xs[c], minlength=g.n_groups)
            if u is not None:
                out["count"] += np.bincount(ids, weights=u.sum(axis=0),
                                            minlength=g.n_groups)
            return
        if ids.size and np.any(ids[1:] < ids[:-1]):
            raise ValueError("a per-channel grouping needs its ids sorted")
        present, starts = np.unique(ids, return_index=True)
        if present.size:
            # every listed segment is non-empty, so reduceat's "empty
            # segment returns the element itself" quirk cannot bite here
            out[key][:, :, present] += np.add.reduceat(x, starts, axis=2)
            if u is not None:
                out["count"][:, present] += np.add.reduceat(u, starts, axis=1)

    def _masked(g: Grouping, usable: np.ndarray, sl: slice) -> np.ndarray:
        return usable if g.mask is None else usable & g.mask[sl][None, :]

    total_bytes = n_comp * n_chan * n_pairs * 8
    cache: list | None = [] if total_bytes <= CACHE_BYTES else None

    # pass 1: counts and sums
    for sl in _blocks(n_pairs, n_chan):
        vals, usable = block_values(sl)
        if cache is not None:
            cache.append((vals, usable))
        for g, a in zip(groupings, acc):
            ug = _masked(g, usable, sl)
            _reduce(g, np.where(ug[None], vals, 0.0), ug.astype(float), sl, a, "total")

    # the means, pooled over components where asked, expanded for pass 2
    means = []
    for g, a in zip(groupings, acc):
        count, total = a["count"], a["total"]
        if g.pool_components:
            count = count * n_comp
            total = np.broadcast_to(total.sum(axis=0)[None], total.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        a["count"] = count
        means.append(mean)

    # pass 2: squared deviations about the group mean
    for i, sl in enumerate(_blocks(n_pairs, n_chan)):
        vals, usable = cache[i] if cache is not None else block_values(sl)
        for g, a, mean in zip(groupings, acc, means):
            ids = g.ids[sl]
            # `np.take`, not `mean[..., ids]`: fancy indexing on a trailing
            # axis is the slow path (see `_cols`)
            if g.per_channel:
                m = np.take(mean, ids, axis=2)
            else:
                m = np.take(mean, ids, axis=1)[:, None, :]
            dev = np.where(_masked(g, usable, sl)[None], vals - m, 0.0)
            dev *= dev
            _reduce(g, dev, None, sl, a, "ss")

    results = []
    for g, a in zip(groupings, acc):
        count, ss = a["count"], a["ss"]
        if g.pool_components:
            ss = ss.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.where(count > 0, np.sqrt(ss / np.maximum(count, 1)), np.nan)
        results.append((count, std))
    return results


def _difference_blocks(data: np.ndarray, finite: np.ndarray,
                       first: np.ndarray, second: np.ndarray):
    """`block_values` for the plain differences: real and imaginary parts as
    two components, usable where *both* endpoints are finite."""
    def block(sl):
        f, s = first[sl], second[sl]
        d = _cols(data, s) - _cols(data, f)
        vals = np.empty((2,) + d.shape)
        vals[0] = d.real
        vals[1] = d.imag
        return vals, _cols(finite, s) & _cols(finite, f)
    return block


class BaselineSigma(NamedTuple):
    """Per-baseline and pooled differenced sigma, before any fallback."""

    count: np.ndarray        # usable differences per baseline (per channel if asked)
    sigma_re: np.ndarray     # (n_baselines,) or (n_chan, n_baselines); NaN if too few
    sigma_im: np.ndarray
    pool_count: int          # usable differences in the whole pool
    pool_re: float           # pooled sigma over every baseline; NaN if too few
    pool_im: float


def baseline_sigma_from_pairs(
    data: np.ndarray,
    pairs: AdjacentPairs,
    per_channel: bool = False,
    extra: list[Grouping] | None = None,
) -> BaselineSigma | tuple[BaselineSigma, list[tuple[np.ndarray, np.ndarray]]]:
    """The differenced sigma of every baseline, and of all of them pooled.

    This is the arithmetic shared by `sigma_from_time_differences` and by
    casa_export.py, which used to carry its own copy of the loop. `MIN_DIFFS`
    gates only the per-baseline values: a baseline with fewer usable
    differences gets NaN here and is left for the caller to fill from the
    pool, to which its differences have still contributed. Skipping the pool
    as well was a real bug once (see `sigma_from_time_differences`).

    `extra` groupings of the same pairs are reduced in the same sweep and
    their `(count, std)` returned alongside, so a caller that needs more
    than the per-baseline sigma -- the chunked estimator -- pays for the
    gathers once.
    """
    data = np.atleast_2d(data)
    n_chan = data.shape[0]
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    n_pairs = pairs.first.size
    (count, std), (g_count, g_std), *others = grouped_std(
        n_pairs, n_chan, 2, _difference_blocks(data, finite, pairs.first, pairs.second),
        [
            Grouping(pairs.pair_baseline, pairs.n_baselines, per_channel),
            Grouping(np.zeros(n_pairs, dtype=np.intp), 1),
        ] + list(extra or []),
    )
    enough = count >= MIN_DIFFS
    s_re = np.where(enough, std[0] / np.sqrt(2.0), np.nan)
    s_im = np.where(enough, std[1] / np.sqrt(2.0), np.nan)
    pool_n = int(g_count[0])
    if pool_n >= MIN_DIFFS:
        pool_re = float(g_std[0, 0]) / np.sqrt(2.0)
        pool_im = float(g_std[1, 0]) / np.sqrt(2.0)
    else:
        pool_re = pool_im = float("nan")
    est = BaselineSigma(count, s_re, s_im, pool_n, pool_re, pool_im)
    return (est, others) if extra is not None else est


def _pair_or_raise(s_re: float, s_im: float, context: str) -> tuple[float, float]:
    """Return a usable (sigma_re, sigma_im) pair, or raise.

    Real and imaginary noise are equal to well within a per cent in any real
    interferometric dataset, so if exactly one part is unusable -- zero scatter,
    NaN -- the other stands in for it. That keeps a run alive on data where one
    hand is degenerate (a purely real column, a constant imaginary part) instead
    of failing the whole noise map for it. Only when *neither* part is usable is
    there genuinely nothing to go on.
    """
    ok_re = bool(np.isfinite(s_re) and s_re > 0)
    ok_im = bool(np.isfinite(s_im) and s_im > 0)
    if ok_re and ok_im:
        return float(s_re), float(s_im)
    if ok_re:
        return float(s_re), float(s_re)
    if ok_im:
        return float(s_im), float(s_im)
    raise ValueError(context)


def sigma_from_time_differences(
    data: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    per_channel: bool = False,
    max_gap: float | None = None,
) -> np.ndarray:
    """Estimate per-visibility noise sigma by pairwise time differencing.

    Non-finite visibilities are excluded rather than propagated. That is not a
    nicety: on the first real dataset this was run on, **a single unflagged NaN
    in 6930 visibilities** turned every sigma into NaN, because `np.std` of a
    window containing it is NaN and the global fallback was computed from the
    same poisoned pool. The whole noise map was unusable.

    Parameters
    ----------
    data
        Complex visibilities, shape (n_chan, n_vis).
    antenna1, antenna2, time
        Row metadata, shape (n_vis,).
    per_channel
        If True, estimate sigma per (baseline, channel); otherwise pool the
        channels of a baseline (more samples, more robust -- recommended
        unless the noise varies strongly across the band).

    Returns
    -------
    sigma : complex ndarray (n_chan, n_vis) -- sigma_re + 1j * sigma_im,
        constant within each baseline (and channel if per_channel).
    """
    data = np.atleast_2d(data)
    pairs = adjacent_pairs(antenna1, antenna2, time, max_gap)
    # per baseline: a sigma from its own differences, or NaN if too few --
    # both endpoints finite and adjacent in time (see `auto_max_gap`), all
    # baselines at once
    est = baseline_sigma_from_pairs(data, pairs, per_channel)
    return _sigma_from_estimate(
        data, pairs, est, antenna1, antenna2, time, per_channel, max_gap
    )


def _sigma_from_estimate(
    data, pairs, est, antenna1, antenna2, time, per_channel, max_gap,
) -> np.ndarray:
    """Turn a `BaselineSigma` into the per-visibility map, with the fallbacks.

    The second half of `sigma_from_time_differences`, split out so that
    `sigma_in_time_chunks` can build its whole-track estimate from the same
    sweep as its per-chunk one.
    """
    n_chan, n_vis = data.shape
    gap = pairs.max_gap
    if per_channel:
        sigma = (_cols(est.sigma_re, pairs.row_baseline)
                 + 1j * _cols(est.sigma_im, pairs.row_baseline))
    else:
        sigma = np.broadcast_to(
            (est.sigma_re + 1j * est.sigma_im)[pairs.row_baseline][None, :],
            (n_chan, n_vis),
        ).copy()
    g_re, g_im = est.pool_re, est.pool_im
    if (
        max_gap is None
        and np.isfinite(gap)
        and not (np.isfinite(g_re) and g_re > 0)
    ):
        # the automatic guard was too strict for this data -- irregular
        # sampling, one integration per scan. Better a contaminated estimate
        # than none, but say so.
        logger.warning(
            "no usable time differences within %.4g s of each other; "
            "re-estimating the noise without the adjacency guard, so it may "
            "include real source variation between samples", gap,
        )
        return sigma_from_time_differences(
            data, antenna1, antenna2, time, per_channel, max_gap=np.inf
        )
    if not (np.isfinite(g_re) and g_re > 0) and not (np.isfinite(g_im) and g_im > 0):
        # single-integration data, or too little usable: overall scatter
        finite = np.isfinite(data.real) & np.isfinite(data.imag)
        g_re = _robust_sigma(data.real[finite])
        g_im = _robust_sigma(data.imag[finite])
    g_re, g_im = _pair_or_raise(
        g_re,
        g_im,
        "cannot estimate a noise level from these visibilities: no "
        "usable time differences and no finite scatter. Check the data "
        "column and the flags, or supply the noise yourself "
        "(--noise sigma to trust the MS SIGMA column).",
    )

    bad = (
        ~np.isfinite(sigma.real) | (sigma.real <= 0)
        | ~np.isfinite(sigma.imag) | (sigma.imag <= 0)
    )
    sigma[bad] = g_re + 1j * g_im
    return sigma


def sigma_constant_from_differences(data: np.ndarray) -> complex:
    """Crude fallback when no time/baseline metadata exists (e.g. legacy
    exports): difference consecutive rows, which are usually adjacent in
    time on the same baseline in an MS-ordered export.

    Non-finite values are excluded, and flagged samples may be passed in as
    NaN. Like the per-baseline estimator, this must not let a handful of bad
    visibilities decide the noise level for the whole dataset.
    """
    data = np.atleast_2d(data)
    diff = np.diff(data, axis=1)
    good = np.isfinite(diff.real) & np.isfinite(diff.imag)
    if good.sum() >= MIN_DIFFS:
        s_re = float(np.std(diff.real[good])) / np.sqrt(2.0)
        s_im = float(np.std(diff.imag[good])) / np.sqrt(2.0)
    else:
        s_re = _robust_sigma(data.real)
        s_im = _robust_sigma(data.imag)
    if not (np.isfinite(s_re) and s_re > 0) and not (np.isfinite(s_im) and s_im > 0):
        s_re = _robust_sigma(data.real)
        s_im = _robust_sigma(data.imag)
    s_re, s_im = _pair_or_raise(
        s_re,
        s_im,
        "cannot estimate a noise level from these visibilities: too few "
        "finite samples to difference.",
    )
    return s_re + 1j * s_im


def scale_relative_sigma(
    data: np.ndarray,
    relative_sigma: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Calibrate a *relative* noise map against time-differenced visibilities.

    An MS WEIGHT / WEIGHT_SPECTRUM column carries real information about how
    sensitivity varies -- between antennas via Tsys, across the band via edges
    and atmospheric lines, between integrations via flagged fraction. What it
    does not reliably carry is the absolute scale. The ALMA pipeline in
    particular sets weights that are proportional to the true inverse variance
    without being equal to it, and every `split`, `mstransform` or averaging
    step rescales them again (CASA memo on data weights,
    https://casa.nrao.edu/Memos/CASA-data-weights.pdf).

    So: keep the shape, replace the scale. The visibilities are divided by the
    relative sigma, which should make them unit-variance up to one unknown
    factor; that factor is then measured by differencing in time exactly as
    `sigma_from_time_differences` does, and folded back in.

    This is strictly better than a plain differenced estimate when a baseline
    has too few integrations to measure its own sigma. PJ0116 at 245 GHz has
    four timestamps, so every baseline had two differences -- below MIN_DIFFS,
    so all 646 of them collapsed to a single pooled number with no
    per-baseline structure at all. Normalising first pools those same 1292
    differences into one very well determined scalar and keeps the weights'
    structure underneath it.

    Parameters
    ----------
    data
        Complex visibilities, shape (n_chan, n_vis).
    relative_sigma
        Noise map of the same shape, correct up to one overall factor.
        Non-positive or non-finite entries are ignored when measuring the
        factor and left for the caller to fill.
    antenna1, antenna2, time
        Row metadata, shape (n_vis,).

    Returns
    -------
    sigma : complex ndarray (n_chan, n_vis), `relative_sigma` multiplied by the
        measured factor. The factor itself is available as the ratio of any
        output entry to its input.
    """
    data = np.atleast_2d(data)
    rel = np.atleast_2d(np.asarray(relative_sigma))
    if rel.shape != data.shape:
        raise ValueError(
            f"relative_sigma shape {rel.shape} != data shape {data.shape}"
        )
    # `rel.real` is a strided view of the complex array, and `np.take` on a
    # strided view runs ~4x slower than on contiguous memory -- the block
    # sweep below gathers the relative sigma four times per block, which made
    # this the slowest estimator by far until the views were made contiguous
    # once here (one copy each, the size of the output anyway).
    rel_re = np.ascontiguousarray(rel.real, dtype=float)
    rel_im = rel_re
    if np.iscomplexobj(rel) and np.any(rel.imag > 0):
        rel_im = np.ascontiguousarray(rel.imag, dtype=float)

    usable = (
        np.isfinite(data.real) & np.isfinite(data.imag)
        & np.isfinite(rel_re) & (rel_re > 0)
        & np.isfinite(rel_im) & (rel_im > 0)
    )

    # DIFFERENCE FIRST, NORMALISE AFTER. The order matters and getting it wrong
    # quietly undoes the whole reason for differencing.
    #
    # Whitening first and differencing the result gives
    #     d(V/s) = dV/s_bar + V d(1/s)
    # whose second term is the *sky* leaking back in whenever the weight
    # changes between two integrations -- which it does routinely, since
    # flagging changes a row's weight. Measured on a mock with a source 12x the
    # noise: harmless at constant weights, but 5% weight jitter tripled the
    # error and 20% made it worse than ignoring the weights altogether.
    #
    # Differencing the raw visibilities and dividing by the pair's combined
    # sigma cancels the sky in the numerator exactly, whatever the weights do:
    #     Var(V_t+1 - V_t) = s_t^2 + s_t+1^2 = 2 s_bar^2
    # so the normalised difference has variance 2, and the scale factor is
    # std/sqrt(2) exactly as in `sigma_from_time_differences`.
    pairs = adjacent_pairs(antenna1, antenna2, time)
    n_pairs = pairs.first.size

    def block(sl):
        f, sec = pairs.first[sl], pairs.second[sl]
        good = _cols(usable, sec) & _cols(usable, f)
        d = _cols(data, sec) - _cols(data, f)
        # rms of the pair, so the difference has variance 2 sigma_bar^2
        bar_re = np.sqrt(0.5 * (_cols(rel_re, sec) ** 2 + _cols(rel_re, f) ** 2))
        bar_im = np.sqrt(0.5 * (_cols(rel_im, sec) ** 2 + _cols(rel_im, f) ** 2))
        vals = np.empty((2,) + d.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(d.real, bar_re, out=vals[0])
            np.divide(d.imag, bar_im, out=vals[1])
        return vals, good

    [(n_good, k)] = grouped_std(
        n_pairs, data.shape[0], 2, block,
        [Grouping(np.zeros(n_pairs, dtype=np.intp), 1)],
    )
    if int(n_good[0]) >= MIN_DIFFS:
        k_re = float(k[0, 0]) / np.sqrt(2.0)
        k_im = float(k[1, 0]) / np.sqrt(2.0)
    else:
        # No usable differences at all -- a single integration. Only here does
        # the whitened *scatter* get used, and it is an upper limit because it
        # still contains the source, exactly as the plain std would.
        with np.errstate(invalid="ignore", divide="ignore"):
            k_re = _robust_sigma(np.where(usable, data.real / rel_re, np.nan))
            k_im = _robust_sigma(np.where(usable, data.imag / rel_im, np.nan))
    k_re, k_im = _pair_or_raise(
        k_re,
        k_im,
        "cannot calibrate the relative noise map: no usable time differences "
        "and no finite scatter after whitening. Check the data column, the "
        "flags, and the WEIGHT column.",
    )
    # written straight into the output's real and imaginary halves rather than
    # as `rel_re * k_re + 1j * (rel_im * k_im)`, which allocates two full-size
    # real temporaries on the way to the same complex array
    out = np.empty(rel_re.shape, dtype=complex)
    np.multiply(rel_re, k_re, out=out.real)
    np.multiply(rel_im, k_im, out=out.imag)
    return out


def noise_time_variation(
    data: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    n_bins: int = 3,
) -> tuple[float, np.ndarray]:
    """How much the noise level changes over the track.

    `sigma_from_time_differences` returns **one sigma per baseline**, pooled
    over the whole observation -- the quadratic mean of sigma(t). It has no
    time resolution at all. That is fine when the noise is stationary and
    actively harmful when it is not: the target rises and sets, airmass and
    Tsys go with it, and on a track from 30 to 70 degrees elevation sigma can
    change by a factor of ~1.9 between transit and the ends. Assigning the
    quadratic mean everywhere then **over-weights the noisiest data**: measured
    on a mock of exactly that track, low-elevation visibilities were given 2.5x
    more weight than they deserve and transit ones 0.72x.

    Nothing in the fit can see this, so it is worth saying out loud. Pool the
    time differences into `n_bins` blocks of the track and report the spread.

    Like every other estimator here, a difference that spans a calibrator gap
    is dropped (`auto_max_gap`). This one lacked that guard for a while: on a
    track with scans a few minutes apart, the one pair per baseline that
    straddled each gap carried the source's change across it, and since those
    pairs land in whichever block holds the scan boundary, the blocks did not
    inflate evenly -- the "variation" it reported was partly the gaps.

    Returns
    -------
    (ratio, sigmas) : the max/min of the per-block sigma, and the blocks. A
        ratio near 1 means the noise is stationary. NaN if there is too little
        to measure.
    """
    data = np.atleast_2d(data)
    finite = np.isfinite(data.real) & np.isfinite(data.imag)

    t = np.asarray(time, dtype=float)
    lo, hi = float(np.min(t)), float(np.max(t))
    if not np.isfinite(lo) or hi <= lo or n_bins < 2:
        return float("nan"), np.full(max(n_bins, 1), np.nan)
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    pairs = adjacent_pairs(antenna1, antenna2, time)
    # a difference belongs to the block holding its first endpoint
    which = np.clip(
        np.searchsorted(edges, t[pairs.first], "right") - 1, 0, n_bins - 1
    )
    def block(sl):
        # the real part alone, as the diagnostic always reported
        f, sec = pairs.first[sl], pairs.second[sl]
        d = _cols(data, sec) - _cols(data, f)
        return d.real[None], _cols(finite, sec) & _cols(finite, f)

    [(count, std)] = grouped_std(
        pairs.first.size, data.shape[0], 1, block, [Grouping(which, n_bins)],
    )
    sigmas = np.where(count >= MIN_DIFFS, std[0] / np.sqrt(2.0), np.nan)

    good_s = sigmas[np.isfinite(sigmas) & (sigmas > 0)]
    if good_s.size < 2:
        return float("nan"), sigmas
    return float(good_s.max() / good_s.min()), sigmas


def _time_profile_from_weights(
    relative_sigma: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Isolate the *time* dependence of a relative noise map, per visibility.

    Each baseline's own sigma series is normalised to unit quadratic mean, so
    whatever is constant about that baseline drops out and only the shape in
    time survives. The surviving shapes are then averaged over baselines --
    elevation is common to the whole array at a given instant, so this is a
    well determined curve even when any single baseline's is not.

    Returns a factor with mean square 1, to be multiplied into a per-baseline
    sigma without changing its level.
    """
    rel = np.asarray(np.atleast_2d(relative_sigma).real, dtype=float)
    ok = np.isfinite(rel) & (rel > 0)
    t_vals, t_index = np.unique(np.asarray(time, dtype=float), return_inverse=True)
    t_index = np.asarray(t_index).ravel()
    # the same dense baseline id `adjacent_pairs` uses; no pairs needed here
    row_baseline = adjacent_pairs(antenna1, antenna2, time, np.inf).row_baseline
    n_base = int(row_baseline.max()) + 1 if row_baseline.size else 0

    # collapse channels (the mean over usable ones, as nanmean would), then
    # normalise each baseline's own time series
    n_ok = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_row = np.where(ok, rel**2, 0.0).sum(axis=0) / n_ok    # (n_vis,)
    finite = np.isfinite(per_row) & (per_row > 0)
    n_rows = np.bincount(row_baseline, weights=finite.astype(float), minlength=n_base)
    sums = np.bincount(row_baseline, weights=np.where(finite, per_row, 0.0),
                       minlength=n_base)
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where(n_rows > 0, sums / np.maximum(n_rows, 1), np.nan)
    # a baseline needs two usable rows to have a time series at all
    use = finite & (n_rows[row_baseline] >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(use, per_row / level[row_baseline], 0.0)
    total = np.bincount(t_index, weights=normalised, minlength=t_vals.size)
    count = np.bincount(t_index, weights=use.astype(float), minlength=t_vals.size)

    with np.errstate(invalid="ignore", divide="ignore"):
        profile_sq = np.where(count > 0, total / np.maximum(count, 1e-30), np.nan)
    if not np.any(np.isfinite(profile_sq) & (profile_sq > 0)):
        return np.ones_like(np.atleast_2d(relative_sigma).real)
    fill = np.nanmedian(profile_sq[np.isfinite(profile_sq) & (profile_sq > 0)])
    profile_sq = np.where(np.isfinite(profile_sq) & (profile_sq > 0), profile_sq, fill)
    profile_sq = profile_sq / np.mean(profile_sq)          # mean square 1
    g = np.sqrt(profile_sq)[t_index]                       # (n_vis,)
    return np.broadcast_to(g, np.atleast_2d(relative_sigma).shape).copy()


def hybrid_sigma(
    data: np.ndarray,
    relative_sigma: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    whole_track_sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Baseline structure from the data, time structure from the weight column.

    Each source of information is used only where it is strong, because they
    are strong in different places.

    **The weight column is reliable along the time axis and blind along the
    baseline axis.** It is radiometric: Tsys, bandwidth, integration time,
    flagged fraction. Elevation moves Tsys for the whole array at once, so the
    column tracks it exactly. What it cannot know is anything that happens
    after the radiometry -- residual phase errors, which grow with baseline
    length because the atmosphere decorrelates faster over longer separations,
    or an antenna whose calibration is simply worse than its Tsys suggests. To
    the weight column two baselines with the same Tsys look identical however
    differently they actually behave.

    **Differencing is the reverse.** It measures whatever really makes the data
    scatter, decorrelation included, so it sees the baseline axis honestly. But
    `sigma_from_time_differences` pools a baseline's whole track into one
    number -- the quadratic mean of sigma(t) -- so it is blind along the time
    axis. On a 30-70 degree track that assigns the noisiest data ~2.3x more
    weight than it deserves.

    So: take the per-baseline level from the differences, and the time profile
    from the weights, normalised to leave that level untouched.

    On a mock carrying both effects -- 1.9x elevation variation plus 3.1x of
    baseline-dependent decorrelation and one bad antenna, with the weight
    column knowing only the first -- the median error in sigma was 16.7% for
    `difference`, 17.0% for `scaled` (neither wins) and **10.2%** here.

    Use `baseline_weight_disagreement` to see whether a given dataset actually
    has structure the weights are blind to.

    `whole_track_sigma` is the output of `sigma_from_time_differences` on the
    same `data`, if the caller already has it -- the import computes it once
    and hands it to every estimator and diagnostic that needs it rather than
    paying for it three times over.
    """
    per_baseline = _whole_track(data, antenna1, antenna2, time, whole_track_sigma)
    g = _time_profile_from_weights(relative_sigma, antenna1, antenna2, time)
    return per_baseline.real * g + 1j * (per_baseline.imag * g)


def _whole_track(data, antenna1, antenna2, time, precomputed) -> np.ndarray:
    """`sigma_from_time_differences`, unless the caller already ran it."""
    if precomputed is None:
        return sigma_from_time_differences(data, antenna1, antenna2, time)
    out = np.atleast_2d(np.asarray(precomputed))
    if out.shape != np.atleast_2d(data).shape:
        raise ValueError(
            f"whole_track_sigma shape {out.shape} != data shape "
            f"{np.atleast_2d(data).shape}"
        )
    return out


def baseline_weight_disagreement(
    data: np.ndarray,
    relative_sigma: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    baseline_length: np.ndarray,
    whole_track_sigma: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Does the measured noise depend on baseline length beyond the weights?

    The weight column is radiometric and knows nothing about decorrelation,
    which grows with baseline length. Comparing the differenced sigma against
    what the column claims, quartile by quartile in baseline length, exposes
    exactly that: a rising ratio means there is real structure `--noise scaled`
    would throw away.

    Returns
    -------
    (ratio, short, long) : the long/short quartile ratio and the two means.
        A ratio above ~1.2 says the weights are blind to something
        baseline-dependent. NaN if it cannot be measured.
    """
    measured = _whole_track(data, antenna1, antenna2, time, whole_track_sigma).real
    claimed = np.asarray(np.atleast_2d(relative_sigma).real, dtype=float)
    ok = (
        np.isfinite(measured) & (measured > 0)
        & np.isfinite(claimed) & (claimed > 0)
    )
    if not np.any(ok):
        return float("nan"), float("nan"), float("nan")

    # the mean over usable channels of each row (nanmean, without its warning
    # on rows where every channel is flagged)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_map = np.where(ok, measured / claimed, 0.0)
        per_vis = ratio_map.sum(axis=0) / ok.sum(axis=0)   # (n_vis,)

    length = np.asarray(baseline_length, dtype=float)
    good = np.isfinite(per_vis) & np.isfinite(length) & (length > 0)
    if good.sum() < 8:
        return float("nan"), float("nan"), float("nan")

    lo_cut = np.percentile(length[good], 25)
    hi_cut = np.percentile(length[good], 75)
    short = per_vis[good & (length <= lo_cut)]
    long_ = per_vis[good & (length >= hi_cut)]
    if short.size == 0 or long_.size == 0:
        return float("nan"), float("nan"), float("nan")
    s, l = float(np.mean(short)), float(np.mean(long_))
    if not (np.isfinite(s) and s > 0 and np.isfinite(l)):
        return float("nan"), s, l
    return l / s, s, l


# Ten minutes. A typical ALMA execution is 1-1.5 h *including* calibrator
# visits, which take 30-40% of it, so the target gets roughly 45-60 minutes --
# and that is what has to be divided up. 600 s leaves 5-6 chunks of it, which
# is enough to follow elevation without measuring each sigma from too few
# differences. Measured on a 75 min execution with interleaved calibrators
# (median error in sigma, 6 s integrations): 300 s 7.9%, 450 s 7.1%,
# **600 s 7.1%**, 900 s 8.1%, 1200 s 8.2%, 1800 s 10.7%, against 22.4% for
# pooling the whole track.
DEFAULT_CHUNK_SECONDS = 600.0


def sigma_in_time_chunks(
    data: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    whole_track_sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Time-differenced sigma, resolved into chunks of the track.

    `sigma_from_time_differences` pools a baseline's whole observation into one
    number, so it cannot represent noise that changes as the target rises and
    sets. The weight column can, but only radiometrically -- it stops at Tsys
    and is blind to decorrelation, which grows with baseline length and, being
    driven by the same airmass, gets worse at exactly the times Tsys does.

    Chunking recovers the time axis **from the data**, so it captures the
    atmosphere's effect on phase as well as on Tsys, and needs no weight column
    at all. Twenty minutes is long enough to hold plenty of differences and
    short enough that elevation has not moved much.

    Where a baseline has too few differences inside a chunk to measure its own
    sigma, it falls back to a separable estimate: its whole-track sigma scaled
    by that chunk's level pooled over *all* baselines. The pooled level is
    always well determined -- every baseline in the array contributes to it --
    so the time dependence survives even when the per-baseline detail cannot.

    Parameters
    ----------
    chunk_seconds
        Chunk width in the units of `time` (an MS stores seconds).
    whole_track_sigma
        `sigma_from_time_differences(data, ...)`, if the caller has it
        already; computed here otherwise.

    Returns
    -------
    sigma : complex ndarray (n_chan, n_vis).
    """
    data = np.atleast_2d(data)
    n_chan, n_vis = data.shape
    t = np.asarray(time, dtype=float)

    span = float(np.max(t) - np.min(t)) if t.size else 0.0
    if not np.isfinite(span) or span <= 0 or chunk_seconds <= 0:
        return _whole_track(data, antenna1, antenna2, time, whole_track_sigma)
    n_chunk = max(1, int(np.ceil(span / float(chunk_seconds))))
    if n_chunk < 2:
        # the whole track is one chunk: nothing to resolve
        return _whole_track(data, antenna1, antenna2, time, whole_track_sigma)
    which = np.clip(
        ((t - t.min()) / float(chunk_seconds)).astype(int), 0, n_chunk - 1
    )

    # Every adjacent pair at once. A difference belongs to the chunk holding
    # its first endpoint, and is dropped if it straddles a chunk boundary --
    # the calibrator-gap guard is already inside `adjacent_pairs`.
    pairs = adjacent_pairs(antenna1, antenna2, time)
    c0 = which[pairs.first]
    inside = c0 == which[pairs.second]
    # The (baseline, chunk) cells that actually hold pairs, densely numbered.
    # Pairs arrive sorted by baseline then time and a chunk index grows with
    # time, so the cell ids are sorted too.
    key = pairs.pair_baseline * n_chunk + c0
    cells, cell_of_pair = np.unique(key[inside], return_inverse=True)
    cell_of_pair = np.asarray(cell_of_pair).ravel()
    n_cells = int(cells.size)
    cell_ids = np.zeros(pairs.first.size, dtype=np.intp)
    cell_ids[inside] = cell_of_pair
    # Real and imaginary differences are pooled into one sigma per cell and
    # per chunk, so a cell's count is of real *and* imaginary values and needs
    # 2 * MIN_DIFFS of them -- i.e. MIN_DIFFS complex differences.
    groupings = [
        Grouping(cell_ids, n_cells, pool_components=True, mask=inside),
        Grouping(c0, n_chunk, pool_components=True, mask=inside),  # every baseline
    ]
    if whole_track_sigma is None:
        # the whole-track estimate from the same sweep over the same pairs,
        # rather than a second pass through `sigma_from_time_differences`
        est, [(count2, own), (chunk_count, chunk_level)] = (
            baseline_sigma_from_pairs(data, pairs, extra=groupings)
        )
        whole = _sigma_from_estimate(
            data, pairs, est, antenna1, antenna2, time, False, None
        )
    else:
        whole = _whole_track(data, antenna1, antenna2, time, whole_track_sigma)
        finite = np.isfinite(data.real) & np.isfinite(data.imag)
        (count2, own), (chunk_count, chunk_level) = grouped_std(
            pairs.first.size, n_chan, 2,
            _difference_blocks(data, finite, pairs.first, pairs.second),
            groupings,
        )
    own = np.where(count2 >= 2 * MIN_DIFFS, own / np.sqrt(2.0), np.nan)
    chunk_level = np.where(
        chunk_count >= 2 * MIN_DIFFS, chunk_level / np.sqrt(2.0), np.nan
    )
    usable_levels = chunk_level[np.isfinite(chunk_level) & (chunk_level > 0)]
    if usable_levels.size < 2:
        return whole
    reference = float(np.sqrt(np.mean(usable_levels**2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(
            np.isfinite(chunk_level) & (chunk_level > 0),
            chunk_level / reference,
            1.0,
        )

    # each row looks up its own (baseline, chunk) cell; rows whose cell has
    # no usable sigma take the separable fallback -- this baseline's level,
    # this chunk's shape
    row_key = pairs.row_baseline * n_chunk + which
    pos = np.searchsorted(cells, row_key)
    pos_ok = pos < n_cells
    hit = np.zeros(n_vis, dtype=bool)
    hit[pos_ok] = cells[pos[pos_ok]] == row_key[pos_ok]
    row_own = np.full(n_vis, np.nan)
    row_own[hit] = own[pos[hit]]
    use_own = np.isfinite(row_own) & (row_own > 0)
    out_re = np.where(use_own[None, :], row_own[None, :], whole.real * g[which][None, :])
    out_im = np.where(use_own[None, :], row_own[None, :], whole.imag * g[which][None, :])

    bad = (
        ~np.isfinite(out_re) | (out_re <= 0)
        | ~np.isfinite(out_im) | (out_im <= 0)
    )
    out_re[bad] = whole.real[bad]
    out_im[bad] = whole.imag[bad]
    return out_re + 1j * out_im

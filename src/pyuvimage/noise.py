"""Noise estimation from the data themselves.

The estimator differences visibilities that are adjacent in time on the same
baseline (and correlation/channel): the sky signal is essentially identical in
consecutive integrations, so the difference is pure noise with variance
2 sigma^2 (cf. https://github.com/tikk3r/maser).  Real and imaginary parts are
kept separate.
"""

from __future__ import annotations

import logging

import numpy as np


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
    n_chan, n_vis = data.shape
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)
    sigma = np.zeros((n_chan, n_vis), dtype=complex)
    t = np.asarray(time, dtype=float)
    gap = auto_max_gap(t) if max_gap is None else float(max_gap)

    global_re, global_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(t[rows])]
        if rows.size < 2:
            sigma[:, rows] = np.nan  # filled from the global estimate below
            continue
        sel = data[:, rows]                       # (n_chan, n_rows)
        ok = finite[:, rows]
        diff = np.diff(sel, axis=1)
        # a difference is usable only if *both* of its endpoints are finite,
        # and only if they really are adjacent in time -- see `auto_max_gap`
        usable = ok[:, 1:] & ok[:, :-1]
        adjacent = np.diff(t[rows]) <= gap
        usable = usable & adjacent[None, :]
        d_re = np.where(usable, diff.real, np.nan)
        d_im = np.where(usable, diff.imag, np.nan)

        if per_channel:
            counts = usable.sum(axis=1)
            with np.errstate(invalid="ignore"):
                s_re = np.nanstd(d_re, axis=1) / np.sqrt(2.0)
                s_im = np.nanstd(d_im, axis=1) / np.sqrt(2.0)
            s_re = np.where(counts >= MIN_DIFFS, s_re, np.nan)
            s_im = np.where(counts >= MIN_DIFFS, s_im, np.nan)
            sigma[:, rows] = (s_re + 1j * s_im)[:, None]
        else:
            if usable.sum() >= MIN_DIFFS:
                with np.errstate(invalid="ignore"):
                    s_re = np.nanstd(d_re) / np.sqrt(2.0)
                    s_im = np.nanstd(d_im) / np.sqrt(2.0)
            else:
                s_re = s_im = np.nan
            sigma[:, rows] = s_re + 1j * s_im
        if np.any(usable):
            global_re.append(d_re[usable])
            global_im.append(d_im[usable])

    g_re = g_im = float("nan")
    if global_re:
        pool_re = np.concatenate(global_re)
        pool_im = np.concatenate(global_im)
        if pool_re.size >= MIN_DIFFS:
            g_re = float(np.nanstd(pool_re)) / np.sqrt(2.0)
            g_im = float(np.nanstd(pool_im)) / np.sqrt(2.0)
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
    rel_re = np.asarray(rel.real, dtype=float)
    rel_im = np.asarray(rel.imag, dtype=float) if np.iscomplexobj(rel) else rel_re
    if not np.any(rel_im > 0):
        rel_im = rel_re

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
    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)
    gap = auto_max_gap(time)
    pool_re, pool_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        if rows.size < 2:
            continue
        rows = rows[np.argsort(time[rows])]
        ok = usable[:, rows]
        good = ok[:, 1:] & ok[:, :-1]
        good = good & (np.diff(np.asarray(time, dtype=float)[rows]) <= gap)[None, :]
        if not np.any(good):
            continue
        d_re = np.diff(data.real[:, rows], axis=1)
        d_im = np.diff(data.imag[:, rows], axis=1)
        r = rel_re[:, rows]
        i_ = rel_im[:, rows]
        # rms of the pair, so the difference has variance 2 sigma_bar^2
        bar_re = np.sqrt(0.5 * (r[:, 1:] ** 2 + r[:, :-1] ** 2))
        bar_im = np.sqrt(0.5 * (i_[:, 1:] ** 2 + i_[:, :-1] ** 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            pool_re.append((d_re / bar_re)[good])
            pool_im.append((d_im / bar_im)[good])

    if pool_re and np.concatenate(pool_re).size >= MIN_DIFFS:
        k_re = float(np.std(np.concatenate(pool_re))) / np.sqrt(2.0)
        k_im = float(np.std(np.concatenate(pool_im))) / np.sqrt(2.0)
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
    return rel_re * k_re + 1j * (rel_im * k_im)


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

    Returns
    -------
    (ratio, sigmas) : the max/min of the per-block sigma, and the blocks. A
        ratio near 1 means the noise is stationary. NaN if there is too little
        to measure.
    """
    data = np.atleast_2d(data)
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)

    t = np.asarray(time, dtype=float)
    lo, hi = float(np.min(t)), float(np.max(t))
    if not np.isfinite(lo) or hi <= lo or n_bins < 2:
        return float("nan"), np.full(max(n_bins, 1), np.nan)
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    pools: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        if rows.size < 2:
            continue
        rows = rows[np.argsort(t[rows])]
        ok = finite[:, rows]
        good = ok[:, 1:] & ok[:, :-1]
        if not np.any(good):
            continue
        diff = np.diff(data[:, rows], axis=1)
        # a difference belongs to the block holding its first endpoint
        which = np.clip(np.searchsorted(edges, t[rows[:-1]], "right") - 1, 0, n_bins - 1)
        for k in range(n_bins):
            sel = good & (which[None, :] == k)
            if np.any(sel):
                pools[k].append(diff.real[sel])

    sigmas = np.full(n_bins, np.nan)
    for k, parts in enumerate(pools):
        if not parts:
            continue
        v = np.concatenate(parts)
        if v.size >= MIN_DIFFS:
            sigmas[k] = float(np.std(v)) / np.sqrt(2.0)

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
    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)
    t_vals, t_index = np.unique(np.asarray(time, dtype=float), return_inverse=True)

    total = np.zeros(t_vals.size)
    count = np.zeros(t_vals.size)
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        sel = rel[:, rows]
        good = ok[:, rows]
        if not np.any(good):
            continue
        # collapse channels, then normalise this baseline's own time series
        with np.errstate(invalid="ignore"):
            series = np.where(good, sel**2, np.nan)
            per_row = np.nanmean(series, axis=0)          # (n_rows,)
        finite = np.isfinite(per_row) & (per_row > 0)
        if finite.sum() < 2:
            continue
        per_row = per_row / np.mean(per_row[finite])
        idx = t_index[rows][finite]
        np.add.at(total, idx, per_row[finite])
        np.add.at(count, idx, 1.0)

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
    """
    per_baseline = sigma_from_time_differences(data, antenna1, antenna2, time)
    g = _time_profile_from_weights(relative_sigma, antenna1, antenna2, time)
    return per_baseline.real * g + 1j * (per_baseline.imag * g)


def baseline_weight_disagreement(
    data: np.ndarray,
    relative_sigma: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    baseline_length: np.ndarray,
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
    measured = sigma_from_time_differences(data, antenna1, antenna2, time).real
    claimed = np.asarray(np.atleast_2d(relative_sigma).real, dtype=float)
    ok = (
        np.isfinite(measured) & (measured > 0)
        & np.isfinite(claimed) & (claimed > 0)
    )
    if not np.any(ok):
        return float("nan"), float("nan"), float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_map = np.where(ok, measured / claimed, np.nan)
    per_vis = np.nanmean(ratio_map, axis=0)                # (n_vis,)

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

    Returns
    -------
    sigma : complex ndarray (n_chan, n_vis).
    """
    data = np.atleast_2d(data)
    n_chan, n_vis = data.shape
    t = np.asarray(time, dtype=float)
    finite = np.isfinite(data.real) & np.isfinite(data.imag)
    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)

    whole = sigma_from_time_differences(data, antenna1, antenna2, time)
    gap = auto_max_gap(t)

    span = float(np.max(t) - np.min(t)) if t.size else 0.0
    if not np.isfinite(span) or span <= 0 or chunk_seconds <= 0:
        return whole
    n_chunk = max(1, int(np.ceil(span / float(chunk_seconds))))
    if n_chunk < 2:
        return whole          # the whole track is one chunk: nothing to resolve
    which = np.clip(
        ((t - t.min()) / float(chunk_seconds)).astype(int), 0, n_chunk - 1
    )

    # per (baseline, chunk) differences, and the pooled level of each chunk
    per_bc: dict[tuple[int, int], float] = {}
    pooled: list[list[np.ndarray]] = [[] for _ in range(n_chunk)]
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        if rows.size < 2:
            continue
        rows = rows[np.argsort(t[rows])]
        ok = finite[:, rows]
        good = ok[:, 1:] & ok[:, :-1]
        if not np.any(good):
            continue
        diff = np.diff(data[:, rows], axis=1)
        # a difference belongs to the chunk holding its first endpoint, and is
        # dropped if it straddles a chunk boundary *or* a calibrator gap
        c0 = which[rows[:-1]]
        c1 = which[rows[1:]]
        inside = (c0 == c1) & (np.diff(t[rows]) <= gap)
        for c in np.unique(c0[inside]):
            sel = good & ((c0 == c) & inside)[None, :]
            if not np.any(sel):
                continue
            vals = np.concatenate([diff.real[sel], diff.imag[sel]])
            pooled[int(c)].append(vals)
            if sel.sum() * 2 >= 2 * MIN_DIFFS:
                per_bc[(int(b), int(c))] = float(np.std(vals)) / np.sqrt(2.0)

    chunk_level = np.full(n_chunk, np.nan)
    for c, parts in enumerate(pooled):
        if not parts:
            continue
        v = np.concatenate(parts)
        if v.size >= 2 * MIN_DIFFS:
            chunk_level[c] = float(np.std(v)) / np.sqrt(2.0)
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

    out_re = np.array(whole.real, dtype=float, copy=True)
    out_im = np.array(whole.imag, dtype=float, copy=True)
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        for c in np.unique(which[rows]):
            cols = rows[which[rows] == int(c)]
            own = per_bc.get((int(b), int(c)))
            if own is not None and own > 0:
                out_re[:, cols] = own
                out_im[:, cols] = own
            else:
                # separable fallback: this baseline's level, this chunk's shape
                out_re[:, cols] = whole.real[:, cols] * g[int(c)]
                out_im[:, cols] = whole.imag[:, cols] * g[int(c)]

    bad = (
        ~np.isfinite(out_re) | (out_re <= 0)
        | ~np.isfinite(out_im) | (out_im <= 0)
    )
    out_re[bad] = whole.real[bad]
    out_im[bad] = whole.imag[bad]
    return out_re + 1j * out_im

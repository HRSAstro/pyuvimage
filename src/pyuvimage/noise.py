"""Noise estimation from the data themselves.

The estimator differences visibilities that are adjacent in time on the same
baseline (and correlation/channel): the sky signal is essentially identical in
consecutive integrations, so the difference is pure noise with variance
2 sigma^2 (cf. https://github.com/tikk3r/maser).  Real and imaginary parts are
kept separate.
"""

from __future__ import annotations

import numpy as np


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

    global_re, global_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(time[rows])]
        if rows.size < 2:
            sigma[:, rows] = np.nan  # filled from the global estimate below
            continue
        sel = data[:, rows]                       # (n_chan, n_rows)
        ok = finite[:, rows]
        diff = np.diff(sel, axis=1)
        # a difference is usable only if *both* of its endpoints are finite
        usable = ok[:, 1:] & ok[:, :-1]
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
    # whiten: if the shape is right, this is unit variance times one factor
    with np.errstate(invalid="ignore", divide="ignore"):
        white = np.where(usable, data.real / rel_re, np.nan) + 1j * np.where(
            usable, data.imag / rel_im, np.nan
        )

    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)
    pool_re, pool_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        if rows.size < 2:
            continue
        rows = rows[np.argsort(time[rows])]
        sel = white[:, rows]
        ok = usable[:, rows]
        good = ok[:, 1:] & ok[:, :-1]
        if not np.any(good):
            continue
        diff = np.diff(sel, axis=1)
        pool_re.append(diff.real[good])
        pool_im.append(diff.imag[good])

    if pool_re and np.concatenate(pool_re).size >= MIN_DIFFS:
        k_re = float(np.std(np.concatenate(pool_re))) / np.sqrt(2.0)
        k_im = float(np.std(np.concatenate(pool_im))) / np.sqrt(2.0)
    else:
        # no usable differences at all: the whitened scatter is an upper limit
        # because it still contains the source. The caller is told.
        k_re = _robust_sigma(white.real[usable])
        k_im = _robust_sigma(white.imag[usable])
    k_re, k_im = _pair_or_raise(
        k_re,
        k_im,
        "cannot calibrate the relative noise map: no usable time differences "
        "and no finite scatter after whitening. Check the data column, the "
        "flags, and the WEIGHT column.",
    )
    return rel_re * k_re + 1j * (rel_im * k_im)

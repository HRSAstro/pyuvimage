"""Noise estimation from the data themselves.

The estimator differences visibilities that are adjacent in time on the same
baseline (and correlation/channel): the sky signal is essentially identical in
consecutive integrations, so the difference is pure noise with variance
2 sigma^2 (cf. https://github.com/tikk3r/maser).  Real and imaginary parts are
kept separate.
"""

from __future__ import annotations

import numpy as np


def sigma_from_time_differences(
    data: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    per_channel: bool = False,
) -> np.ndarray:
    """Estimate per-visibility noise sigma by pairwise time differencing.

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
    baseline = antenna1.astype(np.int64) * 100000 + antenna2.astype(np.int64)
    sigma = np.zeros((n_chan, n_vis), dtype=complex)

    global_re, global_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(time[rows])]
        sel = data[:, rows]  # (n_chan, n_rows)
        if rows.size < 2:
            sigma[:, rows] = np.nan  # fill from global estimate below
            continue
        diff = np.diff(sel, axis=1)  # (n_chan, n_rows-1)
        if per_channel:
            s_re = np.std(diff.real, axis=1) / np.sqrt(2.0)
            s_im = np.std(diff.imag, axis=1) / np.sqrt(2.0)
            sigma[:, rows] = (s_re + 1j * s_im)[:, None]
        else:
            s_re = np.std(diff.real) / np.sqrt(2.0)
            s_im = np.std(diff.imag) / np.sqrt(2.0)
            sigma[:, rows] = s_re + 1j * s_im
        global_re.append(diff.real.ravel())
        global_im.append(diff.imag.ravel())

    if global_re:
        g_re = np.std(np.concatenate(global_re)) / np.sqrt(2.0)
        g_im = np.std(np.concatenate(global_im)) / np.sqrt(2.0)
    else:  # single-integration data: fall back to overall scatter
        g_re = np.std(data.real)
        g_im = np.std(data.imag)
    bad = ~np.isfinite(sigma.real) | (sigma.real <= 0) | (sigma.imag <= 0)
    sigma[bad] = g_re + 1j * g_im
    return sigma


def sigma_constant_from_differences(data: np.ndarray) -> complex:
    """Crude fallback when no time/baseline metadata exists (e.g. legacy
    exports): difference consecutive rows, which are usually adjacent in
    time on the same baseline in an MS-ordered export."""
    data = np.atleast_2d(data)
    diff = np.diff(data, axis=1)
    return (
        float(np.std(diff.real) / np.sqrt(2.0))
        + 1j * float(np.std(diff.imag) / np.sqrt(2.0))
    )

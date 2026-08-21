"""Standalone CASA export script -- for users without python-casacore.

Run inside CASA (which bundles casatools)::

    casa --nologger --nogui -c casa_export.py obs.ms mydata.npz [field] [spw]

It writes a single ``.npz`` file that ``pyuvimage fit`` reads directly (or
convert it to a dataset directory with ``pyuvimage convert mydata.npz
mydata/``).

This file is deliberately self-contained (numpy + casatools only) so it can
be copied anywhere and run under CASA's own python.
"""

from __future__ import annotations

import json
import sys

import numpy as np

PARALLEL_HANDS = {5: "RR", 8: "LL", 9: "XX", 12: "YY", 1: "I"}


def export(ms_path, out_path, field=0, spw=0, data_column="auto"):
    from casatools import table as table_tool

    tb = table_tool()

    tb.open(ms_path)
    q = tb.query(
        "FIELD_ID == %d AND DATA_DESC_ID == %d AND NOT FLAG_ROW"
        % (int(field), int(spw))
    )
    colnames = tb.colnames()
    if data_column == "auto":
        data_column = "CORRECTED_DATA" if "CORRECTED_DATA" in colnames else "DATA"
    # casatools returns (n_corr, n_chan, n_row)
    data = q.getcol(data_column)
    flag = q.getcol("FLAG")
    uvw = q.getcol("UVW").T  # -> (n_row, 3)
    weight = q.getcol("WEIGHT").T  # -> (n_row, n_corr)
    ant1 = q.getcol("ANTENNA1")
    ant2 = q.getcol("ANTENNA2")
    time = q.getcol("TIME")
    q.close()
    tb.close()

    tb.open(ms_path + "/DATA_DESCRIPTION")
    spw_id = int(tb.getcol("SPECTRAL_WINDOW_ID")[int(spw)])
    pol_id = int(tb.getcol("POLARIZATION_ID")[int(spw)])
    tb.close()
    tb.open(ms_path + "/SPECTRAL_WINDOW")
    frequencies = np.atleast_1d(
        np.asarray(tb.getcol("CHAN_FREQ")[:, spw_id], dtype=float).ravel()
    )
    tb.close()
    tb.open(ms_path + "/POLARIZATION")
    corr_types = np.atleast_1d(tb.getcol("CORR_TYPE")[:, pol_id]).ravel()
    tb.close()
    tb.open(ms_path + "/FIELD")
    phase_dir = np.asarray(tb.getcol("PHASE_DIR")[:, :, int(field)]).ravel()
    field_name = str(tb.getcol("NAME")[int(field)])
    tb.close()
    tb.open(ms_path + "/ANTENNA")
    dish_m = float(np.median(tb.getcol("DISH_DIAMETER")))
    tb.close()
    telescope = ""
    try:
        tb.open(ms_path + "/OBSERVATION")
        telescope = str(tb.getcol("TELESCOPE_NAME")[0])
        tb.close()
    except Exception:
        pass

    # data axes (n_corr, n_chan, n_row) -> select parallel hands
    par = [i for i, ct in enumerate(corr_types) if int(ct) in PARALLEL_HANDS]
    if not par:
        raise ValueError("no parallel-hand correlations (CORR_TYPE=%s)" % corr_types)
    names = [PARALLEL_HANDS[int(corr_types[i])] for i in par]

    d = data[par]  # (n_par, n_chan, n_row)
    f = flag[par]
    w = np.maximum(weight[:, par].T, 0.0)[:, None, :] * (~f)
    wsum = w.sum(axis=0)  # (n_chan, n_row)
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = np.where(wsum > 0, (d * w).sum(axis=0) / wsum, 0.0)  # (n_chan, n_row)
    flags = wsum <= 0

    # noise from pairwise time differences per baseline
    baseline = ant1.astype(np.int64) * 100000 + ant2.astype(np.int64)
    sigma = np.zeros(vis.shape, dtype=complex)
    diffs_re, diffs_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(time[rows])]
        if rows.size < 2:
            sigma[:, rows] = np.nan
            continue
        diff = np.diff(vis[:, rows], axis=1)
        s_re = np.std(diff.real) / np.sqrt(2.0)
        s_im = np.std(diff.imag) / np.sqrt(2.0)
        sigma[:, rows] = s_re + 1j * s_im
        diffs_re.append(diff.real.ravel())
        diffs_im.append(diff.imag.ravel())
    if diffs_re:
        g = np.std(np.concatenate(diffs_re)) / np.sqrt(2.0) + 1j * (
            np.std(np.concatenate(diffs_im)) / np.sqrt(2.0)
        )
    else:
        g = np.std(vis.real) + 1j * np.std(vis.imag)
    bad = ~np.isfinite(sigma.real) | (sigma.real <= 0)
    sigma[bad] = g

    meta = {
        "source_ms": ms_path,
        "field": int(field),
        "field_name": field_name,
        "spw": int(spw),
        "data_column": data_column,
        "stokes": "I",
        "correlations_used": names,
        "phase_centre_ra_deg": float(np.degrees(phase_dir[0])) % 360.0,
        "phase_centre_dec_deg": float(np.degrees(phase_dir[1])),
        "dish_diameter_m": dish_m,
        "telescope": telescope,
        "noise_estimate": "difference",
    }
    np.savez_compressed(
        out_path,
        uvw=np.asarray(uvw, dtype=np.float64),
        frequencies=frequencies,
        data_re=vis.real, data_im=vis.imag,
        noise_re=sigma.real, noise_im=sigma.imag,
        flags=flags.astype(np.uint8),
        meta=json.dumps(meta),
    )
    print("pyuvimage export written to %s (%d vis x %d chan, %s)" % (
        out_path, vis.shape[1], vis.shape[0], "+".join(names)))


if __name__ == "__main__":
    # CASA passes its own args first; ours follow the script name.
    argv = sys.argv
    if "-c" in argv:
        argv = argv[argv.index("-c") + 2 :]
    else:
        argv = argv[1:]
    if len(argv) < 2:
        print("usage: casa -c casa_export.py <ms> <out.npz> [field] [spw] [column]")
        sys.exit(1)
    export(
        argv[0], argv[1],
        field=int(argv[2]) if len(argv) > 2 else 0,
        spw=int(argv[3]) if len(argv) > 3 else 0,
        data_column=argv[4] if len(argv) > 4 else "auto",
    )

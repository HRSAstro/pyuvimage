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

# Fewer differences than this and a *per-baseline* sigma is mostly noise
# itself, so that baseline takes the pooled value instead. It never stops
# those differences joining the pool.
MIN_DIFFS = 4


def _cell(tb, column, row):
    """Read one row of a subtable column.

    Use this, never ``getcol``, for the MS subtable columns whose shape varies
    from row to row -- ``POLARIZATION.CORR_TYPE``, ``SPECTRAL_WINDOW.CHAN_FREQ``
    and ``FIELD.PHASE_DIR``. ``getcol`` has to return one rectangular array for
    the whole column, so on a real MS with two polarisation setups, or spectral
    windows with different channel counts, it fails with

        RuntimeError: Table DataManager error: Internal error:
        StManIndArray::get/put shapes not conforming

    ``getcell`` reads a single row and needs no common shape. It also removes a
    trap: casatools' ``getcol`` puts the row axis last while python-casacore's
    puts it first, so the index differed between the two bindings; a cell has
    no row axis at all.
    """
    try:
        return np.asarray(tb.getcell(column, int(row)))
    except Exception:
        # very old bindings, or a column that genuinely has no per-row shape
        col = np.asarray(tb.getcol(column))
        return np.asarray(col[..., int(row)] if col.ndim > 1 else col[int(row)])


def _export_one(ms_path, field, spw, data_column):
    """Read one spectral window; returns (arrays, meta)."""
    from casatools import table as table_tool

    tb = table_tool()

    tb.open(ms_path)
    # ANTENNA1 != ANTENNA2 drops autocorrelations: they sit at u = v = 0 and
    # carry total power, not a visibility, so an unflagged one would act as a
    # bogus zero-spacing constraint on the total flux.
    q = tb.query(
        "FIELD_ID == %d AND DATA_DESC_ID == %d AND ANTENNA1 != ANTENNA2 "
        "AND NOT FLAG_ROW" % (int(field), int(spw))
    )
    colnames = tb.colnames()
    if data_column == "auto":
        data_column = "CORRECTED_DATA" if "CORRECTED_DATA" in colnames else "DATA"
    # casatools returns (n_corr, n_chan, n_row)
    data = q.getcol(data_column)
    flag = q.getcol("FLAG")
    uvw = q.getcol("UVW").T  # -> (n_row, 3)
    weight = q.getcol("WEIGHT").T  # -> (n_row, n_corr)
    # WEIGHT_SPECTRUM is optional (CASA 4.3+) and is the only place per-channel
    # sensitivity lives: band edges, atmospheric lines, spectral Tsys.
    try:
        weight_spectrum = q.getcol("WEIGHT_SPECTRUM")  # (n_corr, n_chan, n_row)
        print("  using WEIGHT_SPECTRUM for per-channel weights")
    except Exception:
        weight_spectrum = None
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
        np.asarray(_cell(tb, "CHAN_FREQ", spw_id), dtype=float).ravel()
    )
    tb.close()
    tb.open(ms_path + "/POLARIZATION")
    corr_types = np.atleast_1d(_cell(tb, "CORR_TYPE", pol_id)).ravel()
    tb.close()
    tb.open(ms_path + "/FIELD")
    phase_dir = np.asarray(_cell(tb, "PHASE_DIR", int(field))).ravel()
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
    # A non-finite sample must not contribute even with zero weight: NaN * 0
    # is NaN, so a single bad visibility would poison the weighted sum for
    # that cell. Treat non-finite as flagged and zero it before weighting.
    finite = np.isfinite(d.real) & np.isfinite(d.imag)
    f = f | ~finite
    d = np.where(finite, d, 0.0)
    if weight_spectrum is not None:
        w = np.maximum(weight_spectrum[par], 0.0) * (~f)
    else:
        w = np.maximum(weight[:, par].T, 0.0)[:, None, :] * (~f)
    wsum = w.sum(axis=0)  # (n_chan, n_row)
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = np.where(wsum > 0, (d * w).sum(axis=0) / wsum, 0.0)  # (n_chan, n_row)
    flags = wsum <= 0
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        print("  %d non-finite visibilities treated as flagged" % n_nonfinite)

    # Noise from pairwise time differences per baseline. Flagged and
    # non-finite cells are excluded throughout: on the first real dataset this
    # was run on, one unflagged NaN in 6930 visibilities made every sigma NaN,
    # because np.std of a window containing it is NaN and the global fallback
    # was pooled from the same poisoned differences.
    # A difference is only meaningful between samples adjacent in time. Across
    # a calibrator visit -- 30-40% of a typical ALMA execution -- the earth has
    # turned a long baseline through several klambda, so the difference
    # measures the source, not the noise. Three median steps, matching
    # pyuvimage.noise.auto_max_gap.
    unique_t = np.unique(np.asarray(time, dtype=float))
    if unique_t.size >= 3:
        steps = np.diff(unique_t)
        steps = steps[np.isfinite(steps) & (steps > 0)]
        max_gap = 3.0 * float(np.median(steps)) if steps.size else np.inf
    else:
        max_gap = np.inf

    baseline = ant1.astype(np.int64) * 100000 + ant2.astype(np.int64)
    sigma = np.zeros(vis.shape, dtype=complex)
    usable_cell = ~flags & np.isfinite(vis.real) & np.isfinite(vis.imag)
    diffs_re, diffs_im = [], []
    for b in np.unique(baseline):
        rows = np.where(baseline == b)[0]
        rows = rows[np.argsort(time[rows])]
        if rows.size < 2:
            sigma[:, rows] = np.nan
            continue
        diff = np.diff(vis[:, rows], axis=1)
        ok = usable_cell[:, rows]
        good = ok[:, 1:] & ok[:, :-1]   # both endpoints usable
        good = good & (np.diff(time[rows]) <= max_gap)[None, :]
        # Contribute to the global pool FIRST, whatever the per-baseline count.
        # MIN_DIFFS decides only whether *this baseline's own* sigma can be
        # trusted; the differences themselves are still perfectly good noise
        # samples for the pooled estimate. Skipping the append as well was a
        # real bug: PJ0116 at 245 GHz has four timestamps, so every baseline
        # had two differences, every baseline was skipped, the pool came out
        # empty, and the code fell through to the MAD of the visibilities --
        # which measures the *source*, not the noise. It returned 5.111 mJy
        # where the pooled differences give 3.696 mJy, a 1.38x overestimate,
        # and the discrepancy principle duly stopped at chi2/N = 1 while the
        # true chi2/N was 0.52, leaving the whole source in the residual map.
        if good.any():
            diffs_re.append(diff.real[good])
            diffs_im.append(diff.imag[good])
        if good.sum() < MIN_DIFFS:
            sigma[:, rows] = np.nan   # filled from the pooled estimate below
            continue
        s_re = np.std(diff.real[good]) / np.sqrt(2.0)
        s_im = np.std(diff.imag[good]) / np.sqrt(2.0)
        sigma[:, rows] = s_re + 1j * s_im
    if diffs_re and np.concatenate(diffs_re).size >= MIN_DIFFS:
        g = np.std(np.concatenate(diffs_re)) / np.sqrt(2.0) + 1j * (
            np.std(np.concatenate(diffs_im)) / np.sqrt(2.0)
        )
    else:
        # Nothing to difference at all -- a single integration, say. The
        # robust scatter of the visibilities is an *upper limit*: it contains
        # the source. Say so, because a noise map that is really a source
        # measurement makes every chi^2-based decision downstream wrong.
        mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
        g = mad(vis.real[usable_cell]) + 1j * mad(vis.imag[usable_cell])
        print(
            "  WARNING: spw %s has no usable time differences, so the noise "
            "is the robust scatter of the visibilities (%.4g Jy). That "
            "INCLUDES THE SOURCE and is an upper limit -- supply the noise "
            "yourself if you can." % (spw, g.real)
        )
    if not (np.isfinite(g.real) and g.real > 0 and np.isfinite(g.imag) and g.imag > 0):
        raise ValueError(
            "cannot estimate a noise level for spw %s: no usable time "
            "differences and no finite scatter. Check the data column and "
            "the flags." % spw
        )
    bad = (
        ~np.isfinite(sigma.real) | (sigma.real <= 0)
        | ~np.isfinite(sigma.imag) | (sigma.imag <= 0)
    )
    sigma[bad] = g

    # What the weight column would have claimed, for comparison. MS weights are
    # relative, not absolute -- the ALMA pipeline sets them proportional to the
    # true inverse variance without being equal to it, and split / mstransform
    # / averaging rescale them again (CASA memo on data weights,
    # https://casa.nrao.edu/Memos/CASA-data-weights.pdf). The noise is always
    # recomputed here; printing the ratio makes it obvious when the column is
    # far off, which is the normal case rather than the alarming one.
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_from_weights = np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan)
    ok_w = np.isfinite(sigma_from_weights) & (sigma_from_weights > 0)
    if np.any(ok_w):
        med_w = float(np.median(sigma_from_weights[ok_w]))
        med_s = float(np.median(sigma.real[np.isfinite(sigma.real)]))
        print(
            "  noise check: MS weights imply %.4g Jy, the visibilities give "
            "%.4g Jy (ratio %.3f); using the visibilities"
            % (med_w, med_s, med_w / med_s if med_s > 0 else float("nan"))
        )

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
    arrays = {
        "uvw": np.asarray(uvw, dtype=np.float64),
        "frequencies": frequencies,
        "data_re": vis.real, "data_im": vis.imag,
        "noise_re": sigma.real, "noise_im": sigma.imag,
        "flags": flags.astype(np.uint8),
        # kept so the noise can be re-estimated later without re-reading the
        # MS -- the first real export had an unusable noise map and nothing
        # in the file to rebuild it from
        "antenna1": np.asarray(ant1, dtype=np.int32),
        "antenna2": np.asarray(ant2, dtype=np.int32),
        "time": np.asarray(time, dtype=np.float64),
        # The weight column's *relative* sigma, 1/sqrt(sum w) over the hands
        # actually averaged. Its scale is not trusted (pipeline weights are
        # relative), but its shape carries Tsys, band edges and atmospheric
        # lines, which --noise hybrid and scaled use. Stored so that choice
        # does not require going back to the MS.
        "weight_sigma_re": np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan),
        "weight_sigma_im": np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan),
    }
    print("  spw %s: %d vis x %d chan, %s" % (
        spw, vis.shape[1], vis.shape[0], "+".join(names)))
    return arrays, meta


def _resolve_spws(ms_path, spw):
    """"all" -> every DATA_DESC_ID present; a list stays a list."""
    from casatools import table as table_tool

    if isinstance(spw, str):
        if spw.strip().lower() != "all":
            return [int(x) for x in spw.replace("-", ",").split(",") if x != ""]
        tb = table_tool()
        tb.open(ms_path)
        ids = sorted({int(v) for v in tb.getcol("DATA_DESC_ID")})
        tb.close()
        return ids
    if isinstance(spw, (list, tuple, set)):
        return sorted({int(x) for x in spw})
    return [int(spw)]


def export(ms_path, out_path, field=0, spw=0, data_column="auto"):
    """Export one or more spectral windows to a single .npz.

    Several windows are stored side by side (`spw000_uvw`, `spw001_uvw`, ...)
    rather than stacked, because they are ragged: different channel counts and
    different row counts. pyuvimage images them together with MFS.
    """
    wanted = _resolve_spws(ms_path, spw)
    print("exporting %d spectral window(s): %s" % (
        len(wanted), ", ".join(str(w) for w in wanted)))

    exported = []
    for w in wanted:
        try:
            exported.append((w,) + _export_one(ms_path, field, w, data_column))
        except Exception as e:  # an empty or unusable window is normal
            if len(wanted) == 1:
                raise
            print("  spw %d skipped: %s" % (w, e))
    if not exported:
        raise ValueError("no usable data in spws %s for field %d" % (wanted, field))

    if len(exported) == 1:
        _, arrays, meta = exported[0]
        np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    else:
        payload = {"n_spw": np.array(len(exported))}
        metas = []
        for i, (_, arrays, meta) in enumerate(exported):
            for k, v in arrays.items():
                payload["spw%03d_%s" % (i, k)] = v
            metas.append(meta)
        combined = dict(metas[0])
        combined["spw"] = [m["spw"] for m in metas]
        combined["per_spw_meta"] = metas
        payload["meta"] = json.dumps(combined)
        np.savez_compressed(out_path, **payload)
    print("pyuvimage export written to %s (%d spectral window(s))" % (
        out_path, len(exported)))


if __name__ == "__main__":
    # CASA passes its own args first; ours follow the script name.
    argv = sys.argv
    if "-c" in argv:
        argv = argv[argv.index("-c") + 2 :]
    else:
        argv = argv[1:]
    if len(argv) < 2:
        print("usage: casa -c casa_export.py <ms> <out.npz> [field] [spw] [column]")
        print('  spw may be an id (0), a list (0,2), a range (0-3), or "all"')
        sys.exit(1)
    spw_arg = argv[3] if len(argv) > 3 else "0"
    if spw_arg.strip().lower() != "all":
        ids = []
        for part in spw_arg.split(","):
            part = part.strip()
            if "-" in part[1:]:
                lo, hi = part.split("-", 1)
                ids.extend(range(int(lo), int(hi) + 1))
            elif part:
                ids.append(int(part))
        spw_arg = ids[0] if len(ids) == 1 else sorted(set(ids))
    export(
        argv[0], argv[1],
        field=int(argv[2]) if len(argv) > 2 else 0,
        spw=spw_arg,
        data_column=argv[4] if len(argv) > 4 else "auto",
    )

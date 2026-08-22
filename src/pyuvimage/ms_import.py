"""Convert a CASA MeasurementSet to a pyuvimage dataset -- no CASA required.

Uses python-casacore (``pip install pyuvimage[ms]``), which reads an MS
directly.  If python-casacore is unavailable on your platform, use the
bundled CASA script instead::

    casa -c $(python -m pyuvimage.casa_export) obs.ms outdir/

Both produce the same dataset directory (see uvdata.py).

What the importer does:
  * selects one field / spectral window / data column,
  * forms Stokes I from the parallel-hand correlations (weighted mean),
  * respects FLAG / FLAG_ROW,
  * recomputes the per-visibility noise from the data on every import --
    pairwise time-differenced visibilities per baseline (maser-style),
    optionally keeping the weight column's *shape* (--noise scaled). MS
    weights are relative, not absolute, so the column's scale is never
    trusted unless --noise sigma is asked for explicitly,
  * records the phase centre and dish diameter for WCS / primary-beam use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import noise as noise_mod
from .uvdata import MultiSpwUVData, UVData

logger = logging.getLogger("pyuvimage")

# CORR_TYPE codes from the MS v2 standard / Stokes.h
PARALLEL_HANDS = {5: "RR", 8: "LL", 9: "XX", 12: "YY", 1: "I"}


def import_ms(
    ms_path: str | Path,
    out_path: str | Path | None = None,
    data_column: str = "auto",
    field: int = 0,
    spw: int | str | Sequence[int] = 0,
    noise_estimate: str = "difference",
    overwrite: bool = False,
) -> "UVData | MultiSpwUVData":
    """Read an MS and (optionally) write a pyuvimage dataset directory.

    Parameters
    ----------
    data_column
        "auto" (CORRECTED_DATA if present, else DATA), "data" or "corrected".
    spw
        A single DATA_DESC_ID (the default), a sequence of them, or "all".
        Several windows are imported into a `MultiSpwUVData` and imaged
        together by MFS -- each keeps its own channels, rows and noise
        estimate, since in a measurement set all three differ between windows.
    noise_estimate
        How to set the per-visibility noise. The MS weights are relative, not
        absolute (see the noise section below), so all three recompute the
        *scale* from the data:

        "difference" (default)
            Pairwise time differences per baseline. Ignores the weight column
            entirely. Falls back to one pooled number when a baseline has too
            few integrations to measure its own sigma.
        "scaled"
            Take the *shape* from WEIGHT / WEIGHT_SPECTRUM -- which does carry
            real Tsys, band-edge and atmospheric structure -- and the absolute
            scale from the same time differences. Better than "difference"
            when baselines have few integrations, since the shape survives.
        "sigma"
            Trust the MS SIGMA column as an absolute level. Warns, because it
            usually is not one.
    """
    try:
        from casacore.tables import table, taql  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "python-casacore is required to read measurement sets: "
            "pip install pyuvimage[ms]\n"
            "If casacore is unavailable on your platform, export with CASA "
            "instead: see pyuvimage/casa_export.py"
        ) from e

    ms_path = Path(ms_path)
    main = table(str(ms_path), readonly=True, ack=False)
    try:
        wanted = _resolve_spws(spw, main)
        if len(wanted) == 1:
            uvd = _import_from_open_ms(
                ms_path, main, table, data_column, field, wanted[0],
                noise_estimate,
            )
        else:
            logger.info("importing %d spectral windows: %s",
                        len(wanted), ", ".join(str(w) for w in wanted))
            spws = []
            for w in wanted:
                try:
                    spws.append(_import_from_open_ms(
                        ms_path, main, table, data_column, field, w,
                        noise_estimate,
                    ))
                except ValueError as e:
                    # an empty or fully flagged window is normal in a real MS;
                    # skipping it beats failing the whole import
                    logger.warning("skipping spw %d: %s", w, e)
            if not spws:
                raise ValueError(
                    f"no usable data in spws {wanted} for field {field}"
                )
            uvd = MultiSpwUVData(spws=spws)
            f = uvd.frequencies
            logger.info(
                "%d spectral windows: %d channels, %d samples, "
                "%.6g-%.6g GHz (fractional bandwidth %.1f%%)",
                uvd.n_spw, uvd.n_chan, uvd.n_samples,
                float(f.min()) / 1e9, float(f.max()) / 1e9,
                100 * uvd.fractional_bandwidth,
            )
    finally:
        main.close()

    if out_path is not None:
        uvd.write(out_path, overwrite=overwrite)
        logger.info("dataset written to %s", Path(out_path).resolve())
    return uvd


def _resolve_spws(spw, main) -> list[int]:
    """Turn the `spw` argument into a concrete list of DATA_DESC_IDs."""
    if isinstance(spw, str):
        if spw.strip().lower() != "all":
            raise ValueError(f"spw must be an int, a sequence or 'all', not {spw!r}")
        ids = sorted({int(v) for v in main.getcol("DATA_DESC_ID")})
        if not ids:
            raise ValueError("the measurement set contains no DATA_DESC_IDs")
        return ids
    if isinstance(spw, (list, tuple, set, np.ndarray)):
        ids = sorted({int(v) for v in spw})
        if not ids:
            raise ValueError("empty spw list")
        return ids
    return [int(spw)]



def _cell(tab, column: str, row: int) -> np.ndarray:
    """Read one row of a subtable column.

    Use this, never ``getcol``, for the MS subtable columns whose shape varies
    from row to row -- ``POLARIZATION.CORR_TYPE``, ``SPECTRAL_WINDOW.CHAN_FREQ``
    and ``FIELD.PHASE_DIR``. ``getcol`` must return one rectangular array for
    the whole column, so on a real MS with two polarisation setups, or spectral
    windows with different channel counts, it fails with

        Table DataManager error: Internal error:
        StManIndArray::get/put shapes not conforming

    ``getcell`` reads a single row and needs no common shape.
    """
    try:
        return np.asarray(tab.getcell(column, int(row)))
    except Exception:
        col = np.asarray(tab.getcol(column))
        return np.asarray(col[int(row)])


def _import_from_open_ms(ms_path, main, table, data_column, field, spw, noise_estimate):
    # ANTENNA1 != ANTENNA2 drops autocorrelations: they sit at u = v = 0 and
    # carry total power, not a visibility, so an unflagged one would act as a
    # bogus zero-spacing constraint on the total flux.
    sel = main.query(
        f"FIELD_ID == {int(field)} AND DATA_DESC_ID == {int(spw)} "
        "AND ANTENNA1 != ANTENNA2 AND NOT FLAG_ROW"
    )
    n_row = sel.nrows()
    if n_row == 0:
        raise ValueError(f"no unflagged rows for field {field} spw {spw}")
    logger.info("reading %d rows (field %d, spw %d)", n_row, field, spw)

    colnames = main.colnames()
    if data_column == "auto":
        data_column = (
            "CORRECTED_DATA" if "CORRECTED_DATA" in colnames else "DATA"
        )
    else:
        data_column = {"data": "DATA", "corrected": "CORRECTED_DATA"}.get(
            data_column.lower(), data_column.upper()
        )
    logger.info("using %s column", data_column)

    data = sel.getcol(data_column)  # (n_row, n_chan, n_corr)
    flag = sel.getcol("FLAG")
    uvw = sel.getcol("UVW")  # (n_row, 3) metres
    weight = sel.getcol("WEIGHT")  # (n_row, n_corr)
    ant1 = sel.getcol("ANTENNA1")
    ant2 = sel.getcol("ANTENNA2")
    time = sel.getcol("TIME")
    sigma_col = sel.getcol("SIGMA")  # (n_row, n_corr)
    # WEIGHT_SPECTRUM is optional (CASA 4.3+) and is the only place per-channel
    # sensitivity lives: band edges, atmospheric lines, spectral Tsys. Without
    # it every channel of a row shares one weight, which is wrong at the band
    # edges of any real ALMA spw.
    try:
        weight_spectrum = sel.getcol("WEIGHT_SPECTRUM")  # (n_row, n_chan, n_corr)
        logger.info("using WEIGHT_SPECTRUM for per-channel weights")
    except Exception:
        weight_spectrum = None

    # ---------------------------------------------------------------- tables
    spw_tab = table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False)
    # DATA_DESC_ID -> SPECTRAL_WINDOW_ID
    dd_tab = table(str(ms_path / "DATA_DESCRIPTION"), readonly=True, ack=False)
    spw_id = int(dd_tab.getcol("SPECTRAL_WINDOW_ID")[int(spw)])
    pol_id = int(dd_tab.getcol("POLARIZATION_ID")[int(spw)])
    frequencies = np.atleast_1d(
        np.asarray(_cell(spw_tab, "CHAN_FREQ", spw_id), dtype=float).ravel()
    )
    dd_tab.close()
    spw_tab.close()

    pol_tab = table(str(ms_path / "POLARIZATION"), readonly=True, ack=False)
    corr_types = np.atleast_1d(_cell(pol_tab, "CORR_TYPE", pol_id)).ravel()
    pol_tab.close()

    field_tab = table(str(ms_path / "FIELD"), readonly=True, ack=False)
    phase_dir = np.asarray(_cell(field_tab, "PHASE_DIR", int(field))).ravel()
    field_name = str(field_tab.getcol("NAME")[int(field)])
    field_tab.close()
    ra_deg = float(np.degrees(phase_dir[0])) % 360.0
    dec_deg = float(np.degrees(phase_dir[1]))

    ant_tab = table(str(ms_path / "ANTENNA"), readonly=True, ack=False)
    dish_m = float(np.median(ant_tab.getcol("DISH_DIAMETER")))
    telescope = ""
    try:
        obs_tab = table(str(ms_path / "OBSERVATION"), readonly=True, ack=False)
        telescope = str(obs_tab.getcol("TELESCOPE_NAME")[0])
        obs_tab.close()
    except Exception:
        pass
    ant_tab.close()

    # ------------------------------------------------------------- Stokes I
    par = [i for i, ct in enumerate(corr_types) if int(ct) in PARALLEL_HANDS]
    if not par:
        raise ValueError(
            f"no parallel-hand correlations found (CORR_TYPE={corr_types})"
        )
    names = [PARALLEL_HANDS[int(corr_types[i])] for i in par]
    logger.info("forming Stokes I from %s", "+".join(names))

    d = data[:, :, par]  # (n_row, n_chan, n_par)
    f = flag[:, :, par]
    # A non-finite sample must not contribute even at zero weight: NaN * 0 is
    # NaN, so one bad visibility would poison its cell's weighted sum. Treat
    # non-finite as flagged and zero it before weighting.
    finite = np.isfinite(d.real) & np.isfinite(d.imag)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        logger.warning(
            "%d non-finite visibilities in %s: treating them as flagged",
            n_nonfinite, data_column,
        )
    f = f | ~finite
    d = np.where(finite, d, 0.0)
    if weight_spectrum is not None:
        w = np.maximum(weight_spectrum[:, :, par], 0.0) * (~f)
    else:
        w = np.maximum(weight[:, par], 0.0)[:, None, :] * (~f)  # zero-weight flagged
    wsum = w.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        stokes_i = np.where(wsum > 0, (d * w).sum(axis=2) / wsum, 0.0)
    chan_flag = wsum <= 0  # (n_row, n_chan): all hands flagged

    # transpose to (n_chan, n_vis)
    vis = np.ascontiguousarray(stokes_i.T)
    flags = np.ascontiguousarray(chan_flag.T)

    # ----------------------------------------------------------------- noise
    #
    # The MS weights are *relative*, not absolute. SIGMA is nominally
    # 1/sqrt(2 dnu dt), but that only holds if the calibration put it on an
    # absolute scale; the ALMA pipeline sets weights proportional to the true
    # inverse variance without being equal to it, and split / mstransform /
    # averaging rescale them again. See the CASA memo on data weights,
    # https://casa.nrao.edu/Memos/CASA-data-weights.pdf. So the noise is
    # recomputed from the visibilities on every import, and the column is used
    # -- at most -- for its shape.
    #
    # Stokes I is a *weighted* average, so its variance is 1/sum(w), not
    # sum(sigma^2)/n^2. Those agree only when the hands carry equal weight;
    # with unequal weights the old expression was simply the variance of a
    # different estimator than the one actually formed above.
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_rel_2d = np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan)
    sigma_rel = np.ascontiguousarray(sigma_rel_2d.T)   # (n_chan, n_vis)

    differenced = noise_mod.sigma_from_time_differences(
        vis, antenna1=ant1, antenna2=ant2, time=time
    )
    med_diff = float(np.median(differenced.real))

    # Always report how far the column is from the data. This is the cheapest
    # possible check that the weights mean what they claim, and it costs one
    # estimate we have already made.
    finite_rel = np.isfinite(sigma_rel) & (sigma_rel > 0)
    med_rel = float(np.median(sigma_rel[finite_rel])) if np.any(finite_rel) else float("nan")
    if np.isfinite(med_rel) and med_rel > 0:
        ratio = med_rel / med_diff
        logger.info(
            "noise check: MS weights imply median sigma %.4g Jy, "
            "time-differenced visibilities give %.4g Jy (ratio %.3f)",
            med_rel, med_diff, ratio,
        )
        if not 0.8 < ratio < 1.25:
            logger.warning(
                "the MS weight column disagrees with the data by a factor "
                "%.2f. That is normal -- pipeline weights are relative, not "
                "absolute -- and is why the noise is recomputed here. It "
                "does mean --noise sigma would be wrong by that factor.",
                ratio,
            )

    if noise_estimate == "sigma":
        logger.warning(
            "--noise sigma trusts the MS SIGMA column as an absolute noise "
            "level. Pipeline weights are usually relative only, so this is "
            "very likely wrong in scale -- and chi^2, the discrepancy "
            "criterion and every uncertainty downstream scale with it. Use "
            "the default, or --noise scaled to keep the column's shape and "
            "take the scale from the data."
        )
        noise = np.where(finite_rel, sigma_rel, np.nan)
        noise = noise + 1j * noise
        bad = ~np.isfinite(noise.real) | (noise.real <= 0)
        noise[bad] = med_diff * (1 + 1j)
    elif noise_estimate == "scaled":
        noise = noise_mod.scale_relative_sigma(
            vis, sigma_rel, antenna1=ant1, antenna2=ant2, time=time
        )
        bad = (
            ~np.isfinite(noise.real) | (noise.real <= 0)
            | ~np.isfinite(noise.imag) | (noise.imag <= 0)
        )
        noise[bad] = differenced[bad]
        logger.info(
            "noise from MS weight shape rescaled to the data: median sigma "
            "%.4g Jy (%d cells fell back to the differenced estimate)",
            float(np.median(noise.real)), int(bad.sum()),
        )
    else:
        noise = differenced
        logger.info(
            "noise from time-differenced visibilities: median sigma %.4g Jy",
            med_diff,
        )

    meta = {
        "source_ms": str(ms_path),
        "field": int(field),
        "field_name": field_name,
        "spw": int(spw),
        "data_column": data_column,
        "stokes": "I",
        "correlations_used": names,
        "phase_centre_ra_deg": ra_deg,
        "phase_centre_dec_deg": dec_deg,
        "dish_diameter_m": dish_m,
        "telescope": telescope,
        "noise_estimate": noise_estimate,
    }
    return UVData(
        uvw=np.asarray(uvw, dtype=float),
        frequencies=frequencies,
        data=vis.astype(complex),
        noise=noise,
        flags=flags if np.any(flags) else None,
        meta=meta,
    )

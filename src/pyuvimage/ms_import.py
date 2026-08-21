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
  * estimates the per-visibility noise from pairwise time-differenced
    visibilities per baseline (maser-style), unless --noise=sigma requests
    the MS SIGMA column,
  * records the phase centre and dish diameter for WCS / primary-beam use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import noise as noise_mod
from .uvdata import UVData

logger = logging.getLogger("pyuvimage")

# CORR_TYPE codes from the MS v2 standard / Stokes.h
PARALLEL_HANDS = {5: "RR", 8: "LL", 9: "XX", 12: "YY", 1: "I"}


def import_ms(
    ms_path: str | Path,
    out_path: str | Path | None = None,
    data_column: str = "auto",
    field: int = 0,
    spw: int = 0,
    noise_estimate: str = "difference",
    overwrite: bool = False,
) -> UVData:
    """Read an MS and (optionally) write a pyuvimage dataset directory.

    Parameters
    ----------
    data_column
        "auto" (CORRECTED_DATA if present, else DATA), "data" or "corrected".
    noise_estimate
        "difference" (pairwise time differences, recommended) or "sigma"
        (trust the MS SIGMA column, e.g. after statwt).
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
        uvd = _import_from_open_ms(
            ms_path, main, table, data_column, field, spw, noise_estimate
        )
    finally:
        main.close()

    if out_path is not None:
        uvd.write(out_path, overwrite=overwrite)
        logger.info("dataset written to %s", Path(out_path).resolve())
    return uvd


def _import_from_open_ms(ms_path, main, table, data_column, field, spw, noise_estimate):
    sel = main.query(
        f"FIELD_ID == {int(field)} AND DATA_DESC_ID == {int(spw)} "
        "AND NOT FLAG_ROW"
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

    # ---------------------------------------------------------------- tables
    spw_tab = table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False)
    # DATA_DESC_ID -> SPECTRAL_WINDOW_ID
    dd_tab = table(str(ms_path / "DATA_DESCRIPTION"), readonly=True, ack=False)
    spw_id = int(dd_tab.getcol("SPECTRAL_WINDOW_ID")[int(spw)])
    pol_id = int(dd_tab.getcol("POLARIZATION_ID")[int(spw)])
    frequencies = np.atleast_1d(
        np.asarray(spw_tab.getcol("CHAN_FREQ")[spw_id], dtype=float)
    )
    dd_tab.close()
    spw_tab.close()

    pol_tab = table(str(ms_path / "POLARIZATION"), readonly=True, ack=False)
    corr_types = np.atleast_1d(pol_tab.getcol("CORR_TYPE")[pol_id])
    pol_tab.close()

    field_tab = table(str(ms_path / "FIELD"), readonly=True, ack=False)
    phase_dir = np.asarray(field_tab.getcol("PHASE_DIR")[int(field)]).ravel()
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
    w = np.maximum(weight[:, par], 0.0)[:, None, :] * (~f)  # zero-weight flagged
    wsum = w.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        stokes_i = np.where(wsum > 0, (d * w).sum(axis=2) / wsum, 0.0)
    chan_flag = wsum <= 0  # (n_row, n_chan): all hands flagged

    # transpose to (n_chan, n_vis)
    vis = np.ascontiguousarray(stokes_i.T)
    flags = np.ascontiguousarray(chan_flag.T)

    # ----------------------------------------------------------------- noise
    if noise_estimate == "sigma":
        sig = sigma_col[:, par]
        var_i = np.where(
            wsum.mean(axis=1, keepdims=True) > 0,
            (sig**2).sum(axis=1, keepdims=True) / len(par) ** 2,
            np.inf,
        )
        s = np.sqrt(np.broadcast_to(var_i, stokes_i.shape)).T
        noise = s + 1j * s
    else:
        noise = noise_mod.sigma_from_time_differences(
            vis, antenna1=ant1, antenna2=ant2, time=time
        )
        med = np.median(noise.real)
        logger.info(
            "noise from time-differenced visibilities: median sigma %.4g Jy",
            med,
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

"""Real-data check of the structure-fit systematic window: 1 sigma vs 5 sigma.

    python scripts/test_structure_window.py           # fit, then plot
    python scripts/test_structure_window.py --plot    # re-plot from the cache

A `structure` fit's prior-systematic window is the range of strengths over
which the structure ratio stays within `STRUCTURE_WINDOW_N_SIGMA` times its
own noise scatter. At 1 (the shipped value) the window never opens at low
S/N and the demo mock's quoted error falls ~10x short of the actual error at
5 sigma; 5 brings the mock's coverage to ~1. This runs both on real data:

  J0116 (245 GHz continuum), J1446 (220 GHz continuum), Ruby (200 GHz
  continuum) from testing_data/, and REBELS-25 [CII] (MFS).

Each dataset is fitted with `criterion="structure"` (and its earlier run's
field of view, mesh and prior), once per setting. The figure has one row per
dataset: the restored image, the residual, the total 1-sigma map at 1 and at
5 sigma (same scale), and the S/N at the restoring beam with 5 sigma. The
window and the source-region flux S/N are written on the panels.

A second figure splits the uncertainty into its statistical and prior-systematic
parts, for both windows, as maps and as the source-flux error budget.

Writes figures/structure_window_{real,budget}.{pdf,png} and
figures/structure_window_real.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

import figure_style as fs

DATA = Path(os.environ.get("PYUVIMAGE_TEST_DATA", "/mnt/user-data/uploads/Work"))
DATASETS = [
    # name, path, fov [arcsec], mesh, prior
    ("J0116 245 GHz", DATA / "pyuvimage/testing_data/J0116_245GHz/PJ0116_245GHz_cont.npz", 8.0, 50, "adaptive"),
    ("J1446 220 GHz", DATA / "pyuvimage/testing_data/J1446_220GHz/PJ1446_220GHz_cont.npz", 3.0, 32, "adaptive"),
    ("Ruby 200 GHz", DATA / "pyuvimage/testing_data/Ruby_200GHz/Ruby_200GHz_cont.npz", 3.0, 26, "adaptive"),
    ("REBELS-25 [CII]", DATA / "REBELS_25/Rebels25.concat.CII.npz", 3.5, None, "adaptive"),
]
SIGMAS = [1.0, 5.0]
CACHE = Path("/tmp/pyuvimage_structure_window")
TEXT_BUMP_PT = 2.0


def fit_all(names=None):
    from pyuvimage import api, fitting
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, path, fov, mesh, reg in DATASETS:
        if names and name not in names:
            continue
        for k in SIGMAS:
            f = CACHE / f"{name.split()[0]}_{k:g}.pkl"
            if f.exists():
                continue
            fitting.STRUCTURE_WINDOW_N_SIGMA = k
            t0 = time.time()
            res = api.run(str(path), fov=fov, mode="mfs", reg=reg, criterion="structure",
                          mesh_shape=(mesh, mesh) if mesh else None, write=False,
                          uncertainty_map=True, pb_correction=False,
                          kernel_cache=str(CACHE / "kernels"))
            p = res.products[0]
            rec = {
                "name": name, "n_sigma": k, "seconds": time.time() - t0,
                "pixel_scale": res.geometry.pixel_scale,
                "restored": np.asarray(p.reconvolved), "dirty": np.asarray(p.dirty_image),
                "model": np.asarray(p.model_image),
                "resid": np.asarray(p.residual_sigma), "unc": np.asarray(p.uncertainty),
                "snr": None if p.snr is None else np.asarray(p.snr),
                "rms": float(p.rms), "beam": p.beam, "coefficient": float(p.coefficient),
                "terms": dict(p.uncertainty_terms or {}), "source_flux": p.source_flux,
                "criterion": (res.scan or {}).get("criterion"),
            }
            f.write_bytes(pickle.dumps(rec))
            sf = rec["source_flux"] or {}
            print(f"{name:<16} x{k:g}  [{rec['criterion']}]  window {rec['terms'].get('systematic_window_dex')}  "
                  f"metric {rec['terms'].get('systematic_window_metric')}  flux S/N {sf.get('snr')}  "
                  f"stat-only {sf.get('snr_stat')}  [{rec['seconds']:.0f} s]", flush=True)
            for q in Path(".").glob("pyuvimage-*.wtilde.npy"):
                q.unlink()


def load():
    out = {}
    for name, *_ in DATASETS:
        recs = {}
        for k in SIGMAS:
            f = CACHE / f"{name.split()[0]}_{k:g}.pkl"
            if f.exists():
                recs[k] = pickle.loads(f.read_bytes())
        if len(recs) == len(SIGMAS):
            out[name] = recs
    return out


def summary(runs) -> list[dict]:
    rows = []
    for name, recs in runs.items():
        for k, r in recs.items():
            sf = r["source_flux"] or {}
            rows.append({
                "dataset": name, "n_sigma": k, "criterion": r["criterion"],
                "coefficient": r["coefficient"],
                "window_dex": r["terms"].get("systematic_window_dex"),
                "window_metric": r["terms"].get("systematic_window_metric"),
                "systematic_median": r["terms"].get("systematic_median"),
                "statistical_median": r["terms"].get("statistical_median"),
                "flux_jy": sf.get("flux_jy"), "flux_error_jy": sf.get("flux_error_jy"),
                "flux_error_sys_jy": sf.get("flux_error_sys_jy"),
                "flux_snr": sf.get("snr"), "flux_snr_stat": sf.get("snr_stat"),
                "peak_snr_beam": None if r["snr"] is None else float(np.nanmax(r["snr"])),
            })
    return rows


def budget_all(names=None):
    """Statistical and prior-systematic maps, from one fit per dataset.

    The fit is captured from inside `api.run` (the delivered `SingleFit`), so
    both windows are evaluated on the same fit."""
    from pyuvimage import api, fitting
    from pyuvimage.fitting import _deblock
    for name, path, fov, mesh, reg in DATASETS:
        if names and name not in names:
            continue
        f = CACHE / f"{name.split()[0]}_budget.pkl"
        if f.exists():
            continue
        captured, orig = {}, fitting.fit_dataset

        def capture(*a, **kw):
            out = orig(*a, **kw)
            if kw.get("reg_kind") == reg:
                captured["fit"] = out
            return out

        fitting.fit_dataset = capture
        try:
            fitting.STRUCTURE_WINDOW_N_SIGMA = 1.0
            api.run(str(path), fov=fov, mode="mfs", reg=reg, criterion="structure",
                    mesh_shape=(mesh, mesh) if mesh else None, write=False, uncertainty_map=False,
                    pb_correction=False, kernel_cache=str(CACHE / "kernels"))
        finally:
            fitting.fit_dataset = orig
        fit = captured["fit"]
        k = fit.geometry.oversample
        rec = {"stat": _deblock(np.asarray(fit.model_uncertainty), k)}
        for n in SIGMAS:
            fitting.STRUCTURE_WINDOW_N_SIGMA = n
            fit.__dict__.pop("_prior_systematic_cache", None)
            rec[f"sys{n:g}"] = _deblock(np.asarray(fit.prior_systematic()), k)
            rec[f"window{n:g}"] = fit.__dict__.get("_systematic_window_dex")
        fitting.STRUCTURE_WINDOW_N_SIGMA = 1.0
        f.write_bytes(pickle.dumps(rec))
        print(f"{name}: budget done; windows {rec['window1']} / {rec['window5']}", flush=True)
        for q in Path(".").glob("pyuvimage-*.wtilde.npy"):
            q.unlink()


def _grid(fig, n_r, n_c, top=0.95, bottom=0.07, left=0.10, right=0.93, extra=None):
    """Image panels with a narrow colour-bar column after each."""
    ratios = []
    for j in range(n_c):
        ratios += [1.0, 0.06, 0.42]
    return fig.add_gridspec(n_r, 3 * n_c - 1, width_ratios=ratios[:-1], left=left, right=right,
                            top=top, bottom=bottom, wspace=0.0, hspace=0.12)


def _cbar(fig, im, cax, label, ticks=None):
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    cb.set_label(label, fontsize=6.0, labelpad=1.5)
    return cb


def plot(runs, budgets):
    import matplotlib.pyplot as plt
    from pyuvimage.products import to_fits_orientation as fo

    fs.use_paper_style()
    names = list(runs)
    n_r = len(names)

    # ---- figure 1: the fits, with the 5x window -------------------------
    n_c = 4
    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.235 * n_r + 0.4))
    gs = _grid(fig, n_r, n_c)
    titles = ["restored [Jy/beam]", "residual [σ]", "total 1σ (±5 scatter)", "S/N at beam (±5)"]
    for i, name in enumerate(names):
        r1, r5 = runs[name][1.0], runs[name][5.0]
        pix = r1["pixel_scale"]
        ext = fs.sky_extent(r1["restored"].shape[0], pix)
        ax = [fig.add_subplot(gs[i, 3 * j]) for j in range(n_c)]
        cax = [fig.add_subplot(gs[i, 3 * j + 1]) for j in range(n_c)]
        vmax = float(np.nanmax(r5["restored"]))
        im = fs.show_image(ax[0], fo(r5["restored"]), ext,
                           norm=fs.asinh_norm(vmax, vmin=float(min(0.0, np.nanmin(r5["restored"]))),
                                              linear_width=2 * r5["rms"]))
        fs.add_beam(ax[0], r5["beam"], ext)
        _cbar(fig, im, cax[0], "")
        rm = 2.0 * np.ceil(float(np.nanmax(np.abs(r5["resid"]))) / 2.0)
        im = fs.show_image(ax[1], fo(r5["resid"]), ext, cmap=fs.RESIDUAL_CMAP, norm=fs.symmetric_norm(rm))
        _cbar(fig, im, cax[1], "")
        u = r5["unc"] / pix**2
        im = fs.show_image(ax[2], fo(u), ext, cmap=fs.UNCERTAINTY_CMAP,
                           norm=fs.asinh_norm(float(np.nanpercentile(u, 99.5)),
                                              linear_width=0.5 * float(np.nanpercentile(u, 50))))
        _cbar(fig, im, cax[2], "Jy arcsec$^{-2}$")
        sf1, sf5 = r1["source_flux"] or {}, r5["source_flux"] or {}
        w = r5["terms"].get("systematic_window_dex") or [np.nan, np.nan]
        fs.panel_label(ax[2], f"window [{w[0]:+.1f}, {w[1]:+.1f}] dex", loc="upper left", size=5.2)
        if sf5.get("snr") and sf1.get("snr"):
            fs.panel_label(ax[2], f"flux S/N {sf5['snr']:.1f} (was {sf1['snr']:.1f})", loc="lower left",
                           size=5.2)
        snr = r5["snr"]
        smax = float(np.nanmax(snr))
        im = fs.show_image(ax[3], fo(snr), ext, cmap="cividis",
                           norm=fs.asinh_norm(smax, vmin=float(min(0.0, np.nanmin(snr))), linear_width=3.0))
        _cbar(fig, im, cax[3], "")
        fs.panel_label(ax[3], f"peak {smax:.1f}", loc="lower left", size=5.2)
        crit = r5["criterion"].split(" ")[0]
        ax[0].set_ylabel(f"{name}\n{crit}", fontsize=6.3, color=fs.INK, labelpad=3)
        if i == 0:
            for a_, t in zip(ax, titles):
                a_.set_title(t, pad=2.5, color=fs.INK, fontsize=6.5)
    for q in fs.save(fig, "structure_window_real"):
        print("wrote", q)

    # ---- figure 2: statistical vs prior-systematic ----------------------
    n_c = 3
    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.235 * n_r + 0.4))
    gs = _grid(fig, n_r, n_c, right=0.72)
    gs_bar = fig.add_gridspec(n_r, 1, left=0.83, right=0.99, top=0.95, bottom=0.07, hspace=0.12)
    titles = ["statistical 1σ", "prior syst., ±1 scatter", "prior syst., ±5 scatter",
              "source-flux error"]
    for i, name in enumerate(names):
        b = budgets.get(name)
        if b is None:
            continue
        r1, r5 = runs[name][1.0], runs[name][5.0]
        pix = r1["pixel_scale"]
        ext = fs.sky_extent(b["stat"].shape[0], pix)
        maps = [b["stat"], b["sys1"], b["sys5"]]
        vmax = max(float(np.nanpercentile(m, 99.5)) for m in maps) / pix**2
        lw = 0.5 * float(np.nanpercentile(b["stat"], 50)) / pix**2
        norm = fs.asinh_norm(vmax, linear_width=lw)
        for j, m in enumerate(maps):
            a_ = fig.add_subplot(gs[i, 3 * j])
            im = fs.show_image(a_, fo(m) / pix**2, ext, cmap=fs.UNCERTAINTY_CMAP, norm=norm)
            if j > 0:
                w = b[f"window{SIGMAS[j - 1]:g}"] or (np.nan, np.nan)
                fs.panel_label(a_, f"[{w[0]:+.1f}, {w[1]:+.1f}] dex", loc="upper left", size=5.2)
            # share of the total variance on the source region
            if j == 0:
                a_.set_ylabel(f"{name}", fontsize=6.3, color=fs.INK, labelpad=3)
            if i == 0:
                a_.set_title(titles[j], pad=2.5, color=fs.INK, fontsize=6.5)
            if j < 2:
                fig.add_subplot(gs[i, 3 * j + 1]).set_visible(False)
        _cbar(fig, im, fig.add_subplot(gs[i, 3 * 2 + 1]), "Jy arcsec$^{-2}$")
        # flux error budget: stat and sys, per window, as fractions of the flux
        a_ = fig.add_subplot(gs_bar[i, 0])
        f = (r1["source_flux"] or {}).get("flux_jy") or np.nan
        stat = (r1["source_flux"] or {}).get("flux_error_stat_jy", np.nan) / f * 100
        sys1 = (r1["source_flux"] or {}).get("flux_error_sys_jy", np.nan) / f * 100
        sys5 = (r5["source_flux"] or {}).get("flux_error_sys_jy", np.nan) / f * 100
        tot1, tot5 = np.hypot(stat, sys1), np.hypot(stat, sys5)
        xs = np.arange(2)
        a_.bar(xs - 0.2, [stat, stat], 0.18, color="#2a78d6", label="statistical")
        a_.bar(xs, [sys1, sys5], 0.18, color="#eb6834", label="prior systematic")
        a_.bar(xs + 0.2, [tot1, tot5], 0.18, color=fs.MUTED, label="total")
        a_.set_ylim(0, 1.45 * max(tot1, tot5))
        a_.set_xticks(xs)
        a_.set_xticklabels(["±1", "±5"], fontsize=6)
        a_.set_ylabel("% of source flux", fontsize=6)
        a_.tick_params(labelsize=5.5, length=1.5, width=0.4)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        if i == 0:
            a_.set_title(titles[3], pad=2.5, color=fs.INK, fontsize=6.5)
            a_.legend(fontsize=5, frameon=False, loc="upper center", ncol=3, handlelength=0.8, columnspacing=0.6)
    for q in fs.save(fig, "structure_window_budget"):
        print("wrote", q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    if not a.plot:
        fit_all(a.only)
        budget_all(a.only)
    runs = load()
    budgets = {name: pickle.loads((CACHE / f"{name.split()[0]}_budget.pkl").read_bytes())
               for name in runs if (CACHE / f"{name.split()[0]}_budget.pkl").exists()}
    rows = summary(runs)
    Path("figures/structure_window_real.json").write_text(json.dumps(rows, indent=2, default=float))
    for r in rows:
        print(r)
    plot(runs, budgets)


if __name__ == "__main__":
    main()

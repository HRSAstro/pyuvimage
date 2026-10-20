"""Matrix-free against w-tilde on the real datasets.

    python scripts/bench_real_solvers.py                 # every dataset, every path
    python scripts/bench_real_solvers.py --only ruby 9io9
    python scripts/bench_real_solvers.py --paths matfree_pos   # one solver, rest kept
    python scripts/bench_real_solvers.py --plot          # re-plot from the JSON

Each dataset is fitted on the streamed path, with the default prior
(`adaptive`) and criterion (`auto`), by three solvers at the same mesh:

  wtilde       the w-tilde algebra with positivity (the default below 100^2)
  wtilde_free  the same, unconstrained (--no-positive): like for like with
               the matrix-free solve, which has no positivity
  matfree      the matrix-free CG solve with the sparse SPDE Matern prior
               (nu = 1 against the dense path's 1.5), unconstrained
  matfree_pos  the same with positivity (an interior point): like for like
               with `wtilde`

The switch between the two paths is forced by setting
`matfree.MATRIX_FREE_MIN_MESH_PIXELS` in a fresh process per fit, so each
fit's wall time and peak memory are its own. The streaming pass is cached
and shared: a priming fit per dataset builds it (its time is reported as
the pass), so every timed fit reads no visibilities.

Ruby and PJ1446 are run at --fov 4 with --image-centre auto: at --fov 3 the
recentred field clips the source against the edge pixels the non-negative
solver holds at zero (Ruby: chi^2/N 1.055 and structure ratio 5.8 with
positivity, against 1.02 and 1.0 without).

Recorded per fit: fit time, peak RSS, chosen strength, chi^2/N, structure
ratio, peak residual, total flux; and per dataset the agreement of each
model with `wtilde`'s at the restoring beam's resolution.

Writes figures/bench_real_solvers.{json,pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

DATA = Path(os.environ.get("PYUVIMAGE_TESTING_DATA",
                           "/mnt/user-data/uploads/Work/pyuvimage/testing_data"))
WORK = Path(os.environ.get("PYUVIMAGE_WORK", "/mnt/user-data/uploads/Work"))
OUT = Path("/tmp/pyuvimage_bench_real")

#: name -> (dataset, fov, image centre, mesh or None for the default)
DATASETS = {
    "pj0116": (DATA / "J0116_245GHz/PJ0116_245GHz_cont.npz", 8.0, None, None),
    "pj1446": (DATA / "J1446_220GHz/PJ1446_220GHz_cont.npz", 4.0, "auto", None),
    "ruby": (DATA / "Ruby_200GHz/Ruby_200GHz_cont.npz", 4.0, "auto", None),
    "9io9": (DATA / "J0209_135GHz/9io9_135GHz_cont.npz", 8.0, None, None),
    "rebels25": (WORK / "REBELS_25/Rebels25.concat.CII.npz", 3.5, None, None),
    "ngc4274_64": (WORK / "NGC4274/NGC4274.split.npz", 35.0, None, 64),
    # the default mesh: matrix-free only (w-tilde's n_mesh^2 matrices do not fit)
    "ngc4274": (WORK / "NGC4274/NGC4274.split.npz", 35.0, None, None),
}
PATHS = ["wtilde", "wtilde_free", "matfree", "matfree_pos"]


def worker(name: str, path: str, out: Path) -> None:
    """One fit, in this (fresh) process."""
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from pyuvimage import api, matfree

    dataset, fov, centre, mesh = DATASETS[name]
    matfree.MATRIX_FREE_MIN_MESH_PIXELS = (0 if path in ("matfree", "matfree_pos", "prime")
                                           else 10 ** 9)
    t0 = time.time()
    res = api.run(str(dataset), fov=fov, out=out, mode="mfs",
                  mesh_shape=None if mesh is None else (mesh, mesh),
                  image_centre=centre, streaming=True,
                  positive_only=path in ("wtilde", "matfree_pos"),
                  kernel_cache=str(OUT / name / "cache"), uncertainty="none",
                  pb_correction=False)
    wall = time.time() - t0
    p = res.products[0]
    fp = res.parameters
    chi2n = fp["fit_quality"]["chi_squared"] / fp["fit_quality"]["n_data"]
    r = np.asarray(p.residual_sigma)
    rec = {
        "dataset": name, "path": path, "wall_s": wall,
        "streaming_s": float((fp.get("streaming") or {}).get("seconds_streaming", 0.0)),
        "mesh": list(res.geometry.mesh_shape), "n_samples": int(fp["data"]["n_samples"]),
        "coefficient": float(p.coefficient), "chi2_per_n": float(chi2n),
        "criterion": fp["source_prior"].get("criterion"),
        "structure_ratio": float(np.std(r) / np.sqrt(chi2n)),
        "peak_residual_sigma": float(np.nanmax(np.abs(r))),
        "flux_jy": float(np.nansum(p.model_image)),
        "min_model": float(np.nanmin(p.model_image)),
        "rms_jy_beam": float(p.rms),
        "beam": [p.beam.bmaj_arcsec, p.beam.bmin_arcsec, p.beam.bpa_deg],
        "pixel_scale": float(res.geometry.pixel_scale),
    }
    np.save(out / "model_native.npy", np.asarray(p.model_image))
    rec["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    (out / "record.json").write_text(json.dumps(rec, indent=2))


def run_all(names, paths=None, old=()):
    """``paths``: only these solvers, the rest of each dataset's rows kept
    from ``old`` (the JSON) -- the agreement still needs `wtilde`'s model."""
    records = []
    paths = paths or PATHS
    for name in names:
        records += [r for r in old if r["dataset"] == name and r["path"] not in paths]
        # build the streamed terms once, untimed: every fit below reuses them
        prime = OUT / name / "prime"
        prime.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        subprocess.run([sys.executable, __file__, "--worker", name, "prime", str(prime)],
                       capture_output=True, text=True)
        pass_s = time.time() - t0
        print(f"{name:<11} streaming pass + priming fit {pass_s:.0f} s", flush=True)
        for path in paths:
            out = OUT / name / path
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            proc = subprocess.run([sys.executable, __file__, "--worker", name, path, str(out)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"{name:<11} {path:<12} FAILED\n{proc.stderr[-2000:]}", flush=True)
                records.append({"dataset": name, "path": path, "failed": proc.stderr[-500:]})
                continue
            rec = json.loads((out / "record.json").read_text())
            rec["pass_s"] = pass_s
            records.append(rec)
            print(f"{name:<11} {path:<12} mesh {rec['mesh'][0]}²  fit {rec['wall_s']:7.1f} s  "
                  f"{rec['max_rss_gb']:.2f} GB  "
                  f"λ {rec['coefficient']:.3g}  χ²/N {rec['chi2_per_n']:.4f}  ρ {rec['structure_ratio']:.2f}  "
                  f"peak {rec['peak_residual_sigma']:.1f}σ  flux {1e3 * rec['flux_jy']:.3g} mJy  "
                  f"[{rec['criterion']}]", flush=True)
        agree(name, [r for r in records if r["dataset"] == name and "failed" not in r])
    return records


def agree(name, recs):
    """Each model against `wtilde`'s, at the restoring beam's resolution."""
    from scipy.ndimage import gaussian_filter

    ref = next((r for r in recs if r["path"] == "wtilde"), None)
    if ref is None:
        return
    pix = ref["pixel_scale"]
    sig = np.sqrt(ref["beam"][0] * ref["beam"][1]) / 2.3548 / pix
    m0 = gaussian_filter(np.load(OUT / name / "wtilde" / "model_native.npy"), sig)
    for r in recs:
        m = gaussian_filter(np.load(OUT / name / r["path"] / "model_native.npy"), sig)
        r["diff_vs_wtilde"] = float(np.sqrt(np.mean((m - m0) ** 2)) / m0.max())
        r["corr_vs_wtilde"] = float(np.corrcoef(m.ravel(), m0.ravel())[0, 1])
        print(f"    {r['path']:<12} vs wtilde at the beam: rms diff {100 * r['diff_vs_wtilde']:.2f}% of peak, "
              f"corr {r['corr_vs_wtilde']:.4f}", flush=True)


def plot(records):
    import matplotlib.pyplot as plt
    import figure_style as fs

    fs.use_paper_style()
    names = [n for n in DATASETS if any(r["dataset"] == n and "failed" not in r for r in records)]
    colours = {"wtilde": "#2a78d6", "wtilde_free": "#8fb8ea", "matfree": "#f4a582",
               "matfree_pos": "#d6501f"}
    labels = {"wtilde": "w-tilde (positive)", "wtilde_free": "w-tilde (unconstrained)",
              "matfree": "matrix-free (unconstrained)", "matfree_pos": "matrix-free (positive)"}
    panels = [("fit time [s]", lambda r: r["wall_s"], True),
              ("peak memory [GB]", lambda r: r["max_rss_gb"], False),
              ("peak residual [σ]", lambda r: r["peak_residual_sigma"], False),
              ("model vs w-tilde at the beam\n[rms, % of peak]", lambda r: 100 * r.get("diff_vs_wtilde", np.nan), False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.30))
    x = np.arange(len(names))
    w = 0.8 / len(PATHS)
    for k, path in enumerate(PATHS):
        for ax, (lab, fn, logy) in zip(axes, panels):
            vals = []
            for n in names:
                r = next((r for r in records if r["dataset"] == n and r["path"] == path
                          and "failed" not in r), None)
                vals.append(fn(r) if r else np.nan)
            ax.bar(x + (k - (len(PATHS) - 1) / 2) * w, vals, w, color=colours[path], label=labels[path])
            ax.set_ylabel(lab, fontsize=6.5)
            if logy:
                ax.set_yscale("log")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(PATHS), fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for q in fs.save(fig, "bench_real_solvers"):
        print("wrote", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("NAME", "PATH", "OUT"))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--paths", nargs="+", default=None, choices=PATHS)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], a.worker[1], Path(a.worker[2]))
        return
    js = Path("figures/bench_real_solvers.json")
    if a.plot:
        records = json.loads(js.read_text())
    else:
        names = a.only or list(DATASETS)
        old = json.loads(js.read_text()) if js.exists() else []
        records = run_all(names, a.paths, old if a.paths else ())
        records = [r for r in old if r["dataset"] not in names] + records
        order = {n: i for i, n in enumerate(DATASETS)}
        records.sort(key=lambda r: (order.get(r["dataset"], 99), PATHS.index(r["path"])))
        js.parent.mkdir(exist_ok=True)
        js.write_text(json.dumps(records, indent=2))
    plot(records)


if __name__ == "__main__":
    main()

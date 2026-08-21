"""Summary-style panels for every generalisation case.

`scripts/generalisation_tests.py` scores the cases numerically; this shows
what the fits actually look like, in the same layout as the `summary.png` a
run writes: dirty image, model, model reconvolved, residual, uncertainty.
One row per case, sharing that case's colour scales.

Usage:  python scripts/generalisation_figures.py [crowded|resolution|snr|outside|all]
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyuvimage import api, beam as beam_mod, fitting
from pyuvimage.mock import make_field_dataset
from pyuvimage.products import model_with_points

sys.path.insert(0, "scripts")
from generalisation_tests import CROWDED, beam_fwhm, score  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def cases(which):
    base = dict(CROWDED)
    out = []
    if which in ("all", "crowded"):
        out.append(("crowded field", dict(base)))
    if which in ("all", "resolution"):
        b = dict(base); b.pop("fov_arcsec"); b.pop("mesh_n")
        for label, fov, mesh, bmax in [
            ("coarse beam (b_max 300 m)", 4.0, 20, 3.0e2),
            ("nominal (b_max 800 m)", 4.0, 32, 8.0e2),
            ("fine beam (b_max 2.5 km)", 4.0, 48, 2.5e3),
            ("wide field, same array", 8.0, 48, 8.0e2),
        ]:
            out.append((f"resolution: {label}",
                        dict(fov_arcsec=fov, mesh_n=mesh,
                             max_baseline_m=bmax, **b)))
    if which in ("all", "snr"):
        b = dict(base); b.pop("sigma_jy")
        for label, sig in [("high S/N", 3e-5), ("nominal", 3e-4),
                           ("low S/N", 1.5e-3)]:
            out.append((f"snr: {label} (sigma {sig:.0e})",
                        dict(sigma_jy=sig, **b)))
    if which in ("all", "outside"):
        b = dict(base); b.pop("fov_arcsec"); b.pop("mesh_n")
        out.append(('outside field: fov 2" for a 3" field',
                    dict(fov_arcsec=2.0, mesh_n=32, max_baseline_m=2.5e3, **b)))
    return out


def run_one(kw):
    uvd, truth, geom, comps = make_field_dataset(**kw)
    fw = beam_fwhm(uvd, geom)
    res = api.run(uvd, fov=kw.get("fov_arcsec", 4.0),
                  mesh_shape=geom.mesh_shape, point_sources=True,
                  uncertainty_map=True, write=False)
    p = res.products[0]
    truth_points = [(c["flux"], c["centre"]) for c in comps["points"]]
    sc = score(p.points or [], truth_points, tol=0.5 * fw)
    return p, geom, comps, sc, 2 * uvd.data.size, fw


def draw(which="all"):
    todo = cases(which)
    scores = []
    fig, axes = plt.subplots(len(todo), 5,
                             figsize=(21, 4.0 * len(todo)),
                             squeeze=False)
    for row, (label, kw) in enumerate(todo):
        p, geom, comps, sc, n_data, fw = run_one(kw)
        half = geom.fov_arcsec / 2.0
        ext = [half, -half, -half, half]
        resid = np.asarray(p.residual_sigma)
        rmax = max(float(np.nanmax(np.abs(resid))), 1e-6)
        peak_sigma = float(np.nanmax(p.reconvolved)) / p.rms
        model_panel = model_with_points(p, geom.pixel_scale)
        ext_max = float(np.nanmax(p.model_image))

        panels = [
            (p.dirty_image, "dirty image", "Jy/beam", "inferno", None),
            (model_panel,
             f"model (+{len(p.points or [])} point, scaled to the extended)",
             "Jy/pixel", "inferno", (0.0, ext_max) if ext_max > 0 else None),
            (p.reconvolved, "model reconvolved", "Jy/beam", "inferno", None),
            (resid,
             f"residual  peak {rmax:.1f}$\\sigma$ = "
             f"{100 * rmax / max(peak_sigma, 1e-9):.2f}% of the source peak",
             r"$\sigma$", "RdBu_r", (-rmax, rmax)),
            (p.uncertainty, "total 1$\\sigma$", "Jy/pixel", "viridis", None),
        ]
        for ax, (img, title, unit, cmap, clim) in zip(axes[row], panels):
            if img is None:
                ax.axis("off")
                continue
            im = ax.imshow(np.asarray(img), origin="upper", extent=ext,
                           cmap=cmap)
            if clim is not None:
                im.set_clim(*clim)
            # truth points in green, what the fit found in cyan
            for c in comps["points"]:
                ax.plot(c["centre"][0], c["centre"][1], "x", color="lime",
                        ms=8, mew=1.6)
            for q in (p.points or []):
                ax.plot(q.d_ra, q.d_dec, "o", mfc="none", mec="cyan", ms=12,
                        mew=1.3)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('dRA ["]', fontsize=8, labelpad=1)
            ax.tick_params(labelsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(unit, fontsize=8)
            cb.ax.tick_params(labelsize=7)
        n_fp = len(sc["false_positives"])
        axes[row][0].set_ylabel(
            f"{label}\n"
            f"beam {fw:.2f}\"  mesh {geom.mesh_shape[0]}  "
            f"chi2/N {p.chi_squared / n_data:.3f}\n"
            f"points {len(sc['matched'])}/{sc['n_truth']}"
            + (f", {n_fp} false" if n_fp else ""),
            fontsize=8)
        scores.append({
            "case": label, "beam": fw, "mesh": geom.mesh_shape[0],
            "chi2_N": p.chi_squared / n_data,
            "resid_peak": float(rmax),
            "resid_pct_of_peak": 100 * rmax / max(peak_sigma, 1e-9),
            "total_flux_ratio": (
                (float(np.nansum(p.model_image))
                 + sum(q.flux for q in (p.points or [])))
                / (sum(c["flux"] for c in comps["extended"])
                   + sum(c["flux"] for c in comps["points"]))),
            **sc,
        })
        print(f"drew {label}: chi2/N {p.chi_squared / n_data:.3f}, "
              f"{len(sc['matched'])}/{sc['n_truth']} points, "
              f"{len(sc['false_positives'])} false", flush=True)

    fig.suptitle(
        "pyuvimage generalisation tests, fit by fit. Green x = true point "
        "source, cyan circle = fitted point.\n"
        "Model panels are stretched to the extended emission, so a point "
        "component saturates its own pixel by design.",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"figures/generalisation_summaries_{which}.png"
    fig.savefig(out, dpi=100)
    print("wrote", out)
    import json
    with open(f"/tmp/genfig_{which}.json", "w") as f:
        json.dump(scores, f, indent=1, default=float)
    print(f"wrote /tmp/genfig_{which}.json")


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "all")

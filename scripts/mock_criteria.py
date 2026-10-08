"""How should the prior strength be chosen: discrepancy, structure or evidence?

    python scripts/mock_criteria.py               # fit, then plot
    python scripts/mock_criteria.py --plot        # re-plot from the cache

The three-structure mock of `mock_matern_nu.py` (Gaussian blob, cusped
exponential disc, arc of half-beam knots; truth on the product grid) at a high
and a low signal-to-noise, three realisations each. For `matern` and
`adaptive`, the full pipeline is run with each criterion, positivity on as by
default:

  discrepancy   chi^2 = N (raised to the constrained floor where needed)
  structure     residual-map structure ratio = 1
  evidence      maximum Bayesian evidence

Scored against the truth: rms error at the beam's resolution, peak residual,
structure ratio, chosen strength, and fluxes compared at the beam's resolution
(so that the model's finite resolution moving flux between neighbouring
components does not count as an error), plus the flux the model puts where
the truth has none.

Writes figures/criteria_mock_gallery.{pdf,png} (adaptive, one realisation per
S/N), figures/criteria_mock_metrics.{pdf,png} and figures/criteria_mock.json.
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

import mock_matern_nu as mn
import figure_style as fs

CRITERIA = [("discrepancy", "discrepancy"), ("structure", "structure"), ("evidence", "evidence")]
REGS = [("matern", "Matérn"), ("adaptive", "adaptive")]
COMPONENTS = ["smooth blob", "exponential disc", "knot arc"]
CACHE = Path("/tmp/pyuvimage_mock_criteria")


def fit_all(seeds):
    from pyuvimage import api, fitting
    from pyuvimage import beam as beam_mod

    CACHE.mkdir(parents=True, exist_ok=True)
    records, gallery = [], {}
    for label, sigma in mn.NOISE.items():
        for seed in seeds:
            uvd, truth, parts, geom = mn.make_mock(seed, sigma)
            uv, d, n = uvd.flattened()
            ds = fitting.make_dataset(uv, d, n, geom)
            b = beam_mod.fit_beam(beam_mod.DirtyImager(ds).dirty_beam, geom.pixel_scale)
            bpx = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec)) / geom.pixel_scale
            sm = lambda x: mn.smooth(x, bpx)
            t_s = sm(truth)
            masks = {k: sm(v) > 0.1 * sm(v).max() for k, v in parts.items()}
            empty = ~np.any(list(masks.values()), axis=0)
            print(f"\n[{label}, seed {seed}]", flush=True)
            for reg, _ in REGS:
                for crit, _ in CRITERIA:
                    t0 = time.time()
                    res = api.run(uvd, fov=mn.FOV, mesh_shape=(mn.MESH, mn.MESH), reg=reg,
                                  criterion=crit, write=False, uncertainty_map=False,
                                  pb_correction=False, kernel_cache=str(CACHE / "kernels"))
                    p = res.products[0]
                    m = np.asarray(p.model_image)
                    r = np.asarray(p.residual_sigma)
                    chi2n = float(p.chi_squared / (2 * uvd.n_samples))
                    m_s = sm(m)
                    used = (res.scan or {}).get("criterion")
                    rec = {
                        "label": label, "seed": seed, "reg": reg, "criterion": crit,
                        "criterion_used": used,
                        "chi2_per_n": chi2n,
                        "coefficient": float(p.coefficient),
                        "log_evidence": float(p.log_evidence),
                        "rms_err_beam": float(np.sqrt(np.mean((m_s - t_s) ** 2)) / t_s.max()),
                        "peak_residual_sigma": float(np.max(np.abs(r))),
                        "structure_ratio": float(np.std(r) / np.sqrt(chi2n)),
                        "flux_beam": {k: float(m_s[mk].sum() / t_s[mk].sum()) for k, mk in masks.items()},
                        "flux_empty_mjy": float(1e3 * m[empty].sum()),
                        "total_flux": float(m.sum() / truth.sum()),
                        "seconds": time.time() - t0,
                    }
                    records.append(rec)
                    if seed == seeds[0] and reg == "adaptive":
                        gallery[(label, crit)] = (m, r, chi2n, rec["coefficient"], p.beam, float(p.rms),
                                                  np.asarray(p.dirty_image), truth, geom.pixel_scale)
                    print(f"  {reg:<8} {crit:<11} [{used}] λ {rec['coefficient']:.2g}  χ²/N {chi2n:.4f}  "
                          f"ρ {rec['structure_ratio']:.2f}  err {100 * rec['rms_err_beam']:5.2f}%  "
                          f"peak {rec['peak_residual_sigma']:4.1f}σ  flux@beam "
                          + " ".join(f"{rec['flux_beam'][k]:.3f}" for k in COMPONENTS)
                          + f"  empty {rec['flux_empty_mjy']:+.2f} mJy  [{rec['seconds']:.0f} s]",
                          flush=True)
    (CACHE / "records.json").write_text(json.dumps(records, indent=2))
    with open(CACHE / "gallery.pkl", "wb") as f:
        pickle.dump(gallery, f)
    return records, gallery


def summarise(records):
    print("\nMean over seeds")
    for label in mn.NOISE:
        print(f"  {label}")
        for reg, rname in REGS:
            for crit, _ in CRITERIA:
                rs = [r for r in records if r["label"] == label and r["reg"] == reg
                      and r["criterion"] == crit]
                if not rs:
                    continue
                f = lambda key: np.mean([r[key] for r in rs])
                print(f"    {rname:<9} {crit:<11} log λ {np.mean(np.log10([r['coefficient'] for r in rs])):5.2f}  "
                      f"χ²/N {f('chi2_per_n'):.4f}  ρ {f('structure_ratio'):.2f}  "
                      f"err {100 * f('rms_err_beam'):5.2f}±{100 * np.std([r['rms_err_beam'] for r in rs]):.2f}%  "
                      f"peak {f('peak_residual_sigma'):4.1f}σ  flux@beam "
                      + " ".join(f"{np.mean([r['flux_beam'][k] for r in rs]):.3f}" for k in COMPONENTS)
                      + f"  empty {f('flux_empty_mjy'):+.2f} mJy")


def plot_gallery(gallery):
    import matplotlib.pyplot as plt
    from pyuvimage.products import to_fits_orientation

    fs.use_paper_style()
    labels = list(mn.NOISE)
    n_c = len(CRITERIA)
    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.27 * (n_c + 1) + 0.5))
    top, bar_h = 0.965, 0.010
    row_h = (top - 0.07) / (n_c + 1)
    gs_ref = fig.add_gridspec(1, 4, left=0.06, right=0.995, top=top,
                              bottom=top - row_h * 0.90, wspace=0.05)
    gs = fig.add_gridspec(n_c, 4, left=0.06, right=0.995, top=top - row_h * 1.08,
                          bottom=0.07, wspace=0.05, hspace=0.05)
    for j, label in enumerate(labels):
        _, _, _, _, beam, rms, dirty, truth, pix = gallery[(label, CRITERIA[0][0])]
        ext = fs.sky_extent(truth.shape[0], pix)
        beam_area = beam.beam_area_pixels(1.0)
        truth_sb = to_fits_orientation(truth) / pix**2
        models = {c: to_fits_orientation(gallery[(label, c)][0]) / pix**2 for c, _ in CRITERIA}
        vmax = max(float(truth_sb.max()), *[float(v.max()) for v in models.values()])
        norm_i = fs.asinh_norm(vmax, linear_width=2.0 * rms / beam_area)
        rmax = max(float(np.max(np.abs(gallery[(label, c)][1]))) for c, _ in CRITERIA)
        norm_r = fs.symmetric_norm(2.0 * np.ceil(rmax / 2.0))
        ax_t = fig.add_subplot(gs_ref[0, 2 * j])
        ax_d = fig.add_subplot(gs_ref[0, 2 * j + 1])
        fs.show_image(ax_t, truth_sb, ext, norm=norm_i)
        fs.show_image(ax_d, to_fits_orientation(dirty) / rms, ext, cmap=fs.INTENSITY_CMAP)
        ax_t.set_title(f"true sky ({label})", pad=2.5, color=fs.INK)
        ax_d.set_title("dirty image", pad=2.5, color=fs.INK)
        fs.add_beam(ax_d, beam, ext)
        fs.panel_label(ax_d, f"peak {np.nanmax(dirty) / rms:.0f}$\\sigma$", loc="lower left", size=5.6)
        for row, (crit, cname) in enumerate(CRITERIA):
            m, r, chi2n, coeff = gallery[(label, crit)][:4]
            ax_m = fig.add_subplot(gs[row, 2 * j])
            ax_r = fig.add_subplot(gs[row, 2 * j + 1])
            im_i = fs.show_image(ax_m, models[crit], ext, norm=norm_i)
            im_r = fs.show_image(ax_r, to_fits_orientation(r), ext, cmap=fs.RESIDUAL_CMAP, norm=norm_r)
            fs.add_beam(ax_m, beam, ext)
            if row == 0:
                ax_m.set_title("model", pad=2.5, color=fs.INK)
                ax_r.set_title("residual", pad=2.5, color=fs.INK)
            if j == 0:
                ax_m.set_ylabel(cname, color=fs.INK, fontsize=7.5, labelpad=3)
            fs.panel_label(ax_r, f"$\\chi^2/N$ {chi2n:.3f}", loc="upper right", color=fs.INK, size=5.4)
            fs.panel_label(ax_r, f"peak {np.max(np.abs(r)):.1f}$\\sigma$", loc="lower right",
                           color=fs.INK, size=5.4)
            fs.panel_label(ax_r, f"$\\lambda$ = {coeff:.2g}", loc="lower left", color=fs.INK, size=5.2)
            if row == n_c - 1:
                fs.sky_axes(ax_m, ext, ticks=(-2.0, 0.0, 2.0))
                if j == 0:   # sky_axes replaces the y label; keep the criterion's name
                    ax_m.set_ylabel(cname + "\n" + ax_m.get_ylabel(), color=fs.INK,
                                    fontsize=7.5, labelpad=3)
                fs.hcolorbar(fig, im_i, [gs[n_c - 1, 2 * j]], "Jy arcsec$^{-2}$", bar_h,
                             ticks=[0.0, 1e-3, 1e-2, 1e-1])
                fs.hcolorbar(fig, im_r, [gs[n_c - 1, 2 * j + 1]], "residual [$\\sigma$]", bar_h)
    for p in fs.save(fig, "criteria_mock_gallery"):
        print("wrote", p)


def plot_metrics(records):
    import matplotlib.pyplot as plt

    fs.use_paper_style()
    x = np.arange(len(CRITERIA))
    fig, axes = plt.subplots(1, 4, figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.27))
    styles = {("high S/N", "matern"): ("#1f77b4", "o"), ("high S/N", "adaptive"): ("#1f77b4", "s"),
              ("low S/N", "matern"): ("#d62728", "o"), ("low S/N", "adaptive"): ("#d62728", "s")}
    panels = [
        (lambda r: 100 * r["rms_err_beam"], "rms error at beam res. [% of peak]"),
        (lambda r: r["peak_residual_sigma"], "peak residual [σ]"),
        (lambda r: r["structure_ratio"], "structure ratio ρ"),
        (lambda r: np.log10(r["coefficient"]), "log$_{10}$ λ"),
    ]
    for k, ((label, reg), (c, mk)) in enumerate(styles.items()):
        off = (k - 1.5) * 0.12
        for ax, (fn, ylab) in zip(axes, panels):
            vals = [[fn(r) for r in records if r["label"] == label and r["reg"] == reg
                     and r["criterion"] == crit] for crit, _ in CRITERIA]
            ax.errorbar(x + off, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals],
                        fmt=mk, ms=3, color=c, mfc=c if reg == "adaptive" else "white",
                        capsize=1.5, lw=0.8, label=f"{reg}, {label}")
            ax.set_ylabel(ylab, fontsize=6.5)
    axes[2].axhline(1.0, color=fs.MUTED, lw=0.5)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in CRITERIA], rotation=30, ha="right", fontsize=6)
    axes[0].legend(fontsize=5.5)
    fig.tight_layout()
    for p in fs.save(fig, "criteria_mock_metrics"):
        print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    if a.plot:
        records = json.loads((CACHE / "records.json").read_text())
        with open(CACHE / "gallery.pkl", "rb") as f:
            gallery = pickle.load(f)
    else:
        records, gallery = fit_all(a.seeds)
    summarise(records)
    Path("figures").mkdir(exist_ok=True)
    Path("figures/criteria_mock.json").write_text(json.dumps(records, indent=2))
    plot_gallery(gallery)
    plot_metrics(records)


if __name__ == "__main__":
    main()

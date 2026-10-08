"""Which source prior reconstructs the mock best?

    python scripts/mock_priors.py                 # fit, then plot
    python scripts/mock_priors.py --plot          # re-plot from the cache

The same mock as `mock_matern_nu.py` (a smooth Gaussian blob, a cusped
exponential disc and an arc of six half-beam knots, truth on the product
grid), at a high and a low signal-to-noise, three noise and uv realisations
each. Every prior is run through the full pipeline with criterion
`structure` and positivity on, and scored against the truth: rms error at the
beam's resolution, peak residual, structure ratio, and flux in one large
aperture around all the emission (`large_aperture`). Per-component apertures
are recorded too, but they pick up flux the model's resolution spreads from
neighbouring components, so they over-report compact components.

(Until Oct 2026 this also ran `--criterion evidence --no-positive` for ln Z;
see git history.)

Writes figures/priors_mock_gallery.{pdf,png} (one realisation, model and
residual per prior, as priors_comparison.pdf) and
figures/priors_mock_metrics.{pdf,png} (all realisations), plus the JSON.
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

import mock_matern_nu as mn           # the mock itself, shared
import figure_style as fs

PRIORS = [
    ("matern", "Matérn", "stationary, ν = 1.5"),
    ("exponential", "exponential", "Matérn ν = 0.5"),
    ("gaussian", "Gaussian envelope", "width tapered off the centre"),
    ("gibbs", "Gibbs", "length short where bright"),
    ("adaptive", "adaptive", "amplitude follows brightness"),
]
CACHE = Path("/tmp/pyuvimage_mock_priors_structure")
COMPONENTS = ["smooth blob", "exponential disc", "knot arc"]
#: added to every font size in the gallery figure
TEXT_BUMP_PT = 2.0


def large_aperture(truth_smoothed, beam_px):
    """The source's 1%-of-peak contour (at the beam's resolution), grown by one beam.

    One aperture around all the emission, so flux the model's resolution
    spreads between components stays inside it.
    """
    from scipy.ndimage import binary_dilation
    core = truth_smoothed > 0.01 * truth_smoothed.max()
    r = int(np.ceil(beam_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return binary_dilation(core, structure=np.hypot(yy, xx) <= beam_px)


def fit_all(seeds):
    from pyuvimage import api
    from pyuvimage import beam as beam_mod
    from pyuvimage import fitting

    CACHE.mkdir(parents=True, exist_ok=True)
    records, gallery = [], None
    for label, sigma in mn.NOISE.items():
        for seed in seeds:
            uvd, truth, parts, geom = mn.make_mock(seed, sigma)
            uv, d, n = uvd.flattened()
            ds = fitting.make_dataset(uv, d, n, geom)
            imager = beam_mod.DirtyImager(ds)
            b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
            beam_px = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec)) / geom.pixel_scale
            t_s = mn.smooth(truth, beam_px)
            masks = {k: mn.smooth(v, beam_px) > 0.1 * mn.smooth(v, beam_px).max()
                     for k, v in parts.items()}
            aperture = large_aperture(t_s, beam_px)
            print(f"\n[{label}, seed {seed}]", flush=True)
            fits_here = {}
            for reg, _, _ in PRIORS:
                rec = {"label": label, "seed": seed, "reg": reg}
                for crit in ("structure",):
                    t0 = time.time()
                    res = api.run(uvd, fov=mn.FOV, mesh_shape=(mn.MESH, mn.MESH), reg=reg,
                                  criterion=crit, write=False, uncertainty_map=False,
                                  pb_correction=False, kernel_cache=str(CACHE / "kernels"))
                    p = res.products[0]
                    n_d = 2 * uvd.n_samples
                    chi2n = float(p.chi_squared / n_d)
                    m = np.asarray(p.model_image)
                    r = np.asarray(p.residual_sigma)
                    rec.update({
                        "chi2_per_n": chi2n,
                        "coefficient": float(p.coefficient),
                        "rms_err_beam": float(np.sqrt(np.mean((mn.smooth(m, beam_px) - t_s) ** 2))
                                              / t_s.max()),
                        "peak_residual_sigma": float(np.max(np.abs(r))),
                        "structure_ratio": float(np.std(r) / np.sqrt(chi2n)),
                        "flux": {k: float(m[mk].sum() / truth[mk].sum()) for k, mk in masks.items()},
                        # one aperture around all the emission: no blending
                        "flux_aperture": float(m[aperture].sum() / truth[aperture].sum()),
                        "flux_field": float(m.sum() / truth.sum()),
                        "seconds": time.time() - t0,
                    })
                    fits_here[reg] = (m, r, chi2n, float(p.coefficient), p.beam, p.rms,
                                      np.asarray(p.dirty_image))
                records.append(rec)
                print(f"  {reg:<11} err {100 * rec['rms_err_beam']:5.2f}%  peak {rec['peak_residual_sigma']:4.1f}σ  "
                      f"ρ {rec['structure_ratio']:.2f}  flux "
                      + " ".join(f"{rec['flux'][k]:.2f}" for k in COMPONENTS)
                      + f"  aperture {rec['flux_aperture']:.3f}  [{rec['seconds']:.0f} s]", flush=True)
            if gallery is None and label == "high S/N":
                gallery = {"truth": truth, "fits": fits_here,
                           "pixel_scale": geom.pixel_scale}
    (CACHE / "records.json").write_text(json.dumps(records, indent=2))
    with open(CACHE / "gallery.pkl", "wb") as f:
        pickle.dump(gallery, f)
    return records, gallery


def summarise(records):
    print("\nMean over seeds (± scatter); criterion structure.")
    for label in mn.NOISE:
        print(f"  {label}")
        for reg, name, _ in PRIORS:
            rs = [r for r in records if r["label"] == label and r["reg"] == reg]
            err = np.array([r["rms_err_beam"] for r in rs]) * 100
            print(f"    {name:<18} err {err.mean():5.2f}±{err.std():.2f}%  "
                  f"peak {np.mean([r['peak_residual_sigma'] for r in rs]):4.1f}σ  "
                  f"ρ {np.mean([r['structure_ratio'] for r in rs]):.2f}  flux "
                  + " ".join(f"{np.mean([r['flux'][k] for r in rs]):.2f}" for k in COMPONENTS)
                  + f"  aperture {np.mean([r['flux_aperture'] for r in rs]):.3f}"
                    f"±{np.std([r['flux_aperture'] for r in rs]):.3f}"
                  + f"  field {np.mean([r['flux_field'] for r in rs]):.3f}")


def plot_gallery(gallery):
    import matplotlib.pyplot as plt
    from pyuvimage.products import to_fits_orientation

    fs.use_paper_style()
    pix = gallery["pixel_scale"]
    truth_sb = to_fits_orientation(gallery["truth"]) / pix**2
    fits_ = gallery["fits"]
    first = fits_[PRIORS[0][0]]
    beam, rms, dirty = first[4], float(first[5]), to_fits_orientation(first[6])
    ext = fs.sky_extent(truth_sb.shape[0], pix)
    beam_area = beam.beam_area_pixels(1.0)
    models_sb = {reg: to_fits_orientation(fits_[reg][0]) / pix**2 for reg, _, _ in PRIORS}
    vmax_i = max(float(truth_sb.max()), *[float(m.max()) for m in models_sb.values()])
    norm_i = fs.asinh_norm(vmax_i, linear_width=2.0 * rms / beam_area)
    rmax = max(float(np.max(np.abs(fits_[reg][1]))) for reg, _, _ in PRIORS)
    norm_r = fs.symmetric_norm(2.0 * np.ceil(rmax / 2.0))

    n_p = len(PRIORS)
    fig = plt.figure(figsize=(fs.ONE_COLUMN, fs.ONE_COLUMN * 0.54 * (n_p + 1) + 0.55))
    top, bar_h = 0.96, 0.012
    row_h = (top - 0.075) / (n_p + 1)
    gs_ref = fig.add_gridspec(1, 2, left=0.115, right=0.995, top=top,
                              bottom=top - row_h * 0.90, wspace=0.05)
    gs = fig.add_gridspec(n_p, 2, left=0.115, right=0.995, top=top - row_h * 1.06,
                          bottom=0.075, wspace=0.05, hspace=0.05)
    ax_t, ax_d = fig.add_subplot(gs_ref[0, 0]), fig.add_subplot(gs_ref[0, 1])
    fs.show_image(ax_t, truth_sb, ext, norm=norm_i)
    fs.show_image(ax_d, dirty / rms, ext, cmap=fs.INTENSITY_CMAP)
    ax_t.set_title("true sky", pad=2.5, color=fs.INK)
    ax_d.set_title("dirty image", pad=2.5, color=fs.INK)
    fs.add_beam(ax_d, beam, ext)
    fs.panel_label(ax_d, f"peak {np.nanmax(dirty) / rms:.0f}$\\sigma$", loc="lower left", size=5.6)
    for row, (reg, name, _) in enumerate(PRIORS):
        m, r, chi2n, coeff = fits_[reg][:4]
        ax_m, ax_r = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        im_i = fs.show_image(ax_m, models_sb[reg], ext, norm=norm_i)
        im_r = fs.show_image(ax_r, to_fits_orientation(r), ext, cmap=fs.RESIDUAL_CMAP, norm=norm_r)
        fs.add_beam(ax_m, beam, ext)
        if row == 0:
            ax_m.set_title("model", pad=2.5, color=fs.INK)
            ax_r.set_title("residual", pad=2.5, color=fs.INK)
        ax_m.set_ylabel(name, color=fs.INK, fontsize=7.5, labelpad=3)
        fs.panel_label(ax_r, f"$\\chi^2/N$ {chi2n:.3f}", loc="upper right", color=fs.INK, size=5.6)
        fs.panel_label(ax_r, f"peak {np.max(np.abs(r)):.1f}$\\sigma$", loc="lower right",
                       color=fs.INK, size=5.6)
        fs.panel_label(ax_r, f"$\\lambda$ = {coeff:.2g}", loc="lower left", color=fs.INK, size=5.4)
        if row == n_p - 1:
            fs.sky_axes(ax_m, ext, ticks=(-2.0, 0.0, 2.0))
            # sky_axes replaces the y label; keep the prior's name on it
            ax_m.set_ylabel(name + "\n" + ax_m.get_ylabel(), color=fs.INK, fontsize=7.5, labelpad=3)
    fs.hcolorbar(fig, im_i, [gs[n_p - 1, 0]], "Jy arcsec$^{-2}$", bar_h, ticks=[0.0, 1e-3, 1e-2, 1e-1])
    fs.hcolorbar(fig, im_r, [gs[n_p - 1, 1]], "residual [$\\sigma$]", bar_h)
    # every piece of text 2 pt larger than the paper style's default
    fig.canvas.draw()
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        if t.get_text():
            t.set_fontsize(t.get_fontsize() + TEXT_BUMP_PT)
    for p in fs.save(fig, "priors_mock_gallery"):
        print("wrote", p)


def plot_metrics(records):
    import matplotlib.pyplot as plt

    fs.use_paper_style()
    names = [n for _, n, _ in PRIORS]
    x = np.arange(len(PRIORS))
    fig, axes = plt.subplots(1, 4, figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.27))
    colours = {"high S/N": "#1f77b4", "low S/N": "#d62728"}
    for k, label in enumerate(mn.NOISE):
        off = (k - 0.5) * 0.18
        def stat(fn):
            vals = [[fn(r) for r in records if r["label"] == label and r["reg"] == reg]
                    for reg, _, _ in PRIORS]
            return np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])
        panels = [
            (lambda r: 100 * r["rms_err_beam"], "rms error [% of peak]"),
            (lambda r: r["peak_residual_sigma"], "peak residual [σ]"),
            (lambda r: r["flux_aperture"], "flux in source aperture / true"),
            # the structure criterion holds ρ at 1, so it is not shown; the
            # aperture holds 99.99% of the true flux, so field − aperture is
            # the model's flux in empty sky
            (lambda r: 100 * (r["flux_field"] - r["flux_aperture"]),
             "flux outside aperture [% of true]"),
        ]
        for ax, (fn, ylab) in zip(axes, panels):
            mu, sd = stat(fn)
            ax.errorbar(x + off, mu, yerr=sd, fmt="o", ms=3, color=colours[label],
                        label=label, capsize=1.5, lw=0.8)
            ax.set_ylabel(ylab, fontsize=6.5)
    axes[2].axhline(1.0, color=fs.MUTED, lw=0.5)
    axes[3].axhline(0.0, color=fs.MUTED, lw=0.5)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for p in fs.save(fig, "priors_mock_metrics"):
        print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--plot", action="store_true", help="re-plot from the cache")
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
    Path("figures/priors_mock.json").write_text(json.dumps(records, indent=2))
    plot_gallery(gallery)
    plot_metrics(records)


if __name__ == "__main__":
    main()

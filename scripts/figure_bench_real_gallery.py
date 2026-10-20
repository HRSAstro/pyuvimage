"""Visual summary of the real-data solver benchmark (`bench_real_solvers.py`).

    python scripts/figure_bench_real_gallery.py            # every dataset
    python scripts/figure_bench_real_gallery.py ruby 9io9

One figure per dataset, one row per solver (w-tilde positive / unconstrained,
matrix-free unconstrained / positive), on shared colour scales per column:

  model           Jy/pixel, asinh stretch; cyan contours where it is negative
                  (-2% and -5% of the peak)
  reconvolved     the model convolved with the restoring beam [Jy/beam]
  residual        dirty image of the residual visibilities [sigma]
  vs reference    reconvolved minus the reference fit's reconvolved model, in
                  units of the image rms: how far apart two solvers' answers
                  are compared with the noise (reference: w-tilde positive,
                  or matrix-free positive where w-tilde cannot run). The
                  scale is +-3 sigma, widened to the 99th percentile where
                  the solvers differ more (on extended emission, in the
                  large scales the data do not constrain)

Reads the products the benchmark wrote under /tmp/pyuvimage_bench_real and
writes figures/bench_real_summary_<dataset>.{pdf,png}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import figure_style as fs
from bench_real_solvers import DATASETS, OUT, PATHS

LABELS = {"wtilde": "w-tilde, positive", "wtilde_free": "w-tilde, unconstrained",
          "matfree": "matrix-free, unconstrained", "matfree_pos": "matrix-free, positive"}
RESID_MAX = 5.0          # sigma
DIFF_MAX = 3.0           # sigma


def load(name, path):
    d = OUT / name / path
    if not (d / "record.json").exists():
        return None
    rec = json.loads((d / "record.json").read_text())
    img = {k: np.squeeze(fits.getdata(d / f"{k}.fits")).astype(float)
           for k in ("model", "model_reconvolved", "residual")}
    pix = abs(fits.getheader(d / "model.fits")["CDELT1"]) * 3600.0
    return rec, img, pix


def figure(name):
    runs = {p: r for p in PATHS if (r := load(name, p)) is not None}
    if not runs:
        print(f"{name}: no runs under {OUT / name}")
        return
    paths = list(runs)
    ref = "wtilde" if "wtilde" in runs else "matfree_pos"
    rec0, img0, pix = runs[ref]
    n = img0["model"].shape[0]
    ext = fs.sky_extent(n, pix)
    beam = SimpleNamespace(bmaj_arcsec=rec0["beam"][0], bmin_arcsec=rec0["beam"][1],
                           bpa_deg=rec0["beam"][2])
    rms = rec0["rms_jy_beam"]
    mmax = max(np.nanpercentile(r[1]["model"], 99.95) for r in runs.values())
    cmax = max(np.nanmax(r[1]["model_reconvolved"]) for r in runs.values())
    mnorm = fs.asinh_norm(mmax, linear_width_frac=0.02)
    # the difference scale: 3 sigma, or wider where the solvers disagree on
    # scales the data do not measure (extended emission: NGC 4274)
    diffs = [np.abs(r[1]["model_reconvolved"] - img0["model_reconvolved"]) / rms
             for p, r in runs.items() if p != ref]
    dmax = max([DIFF_MAX] + [float(np.nanpercentile(d, 99)) for d in diffs])
    dmax = float(np.ceil(dmax)) if dmax < 10 else float(10 * np.ceil(dmax / 10))

    nr = len(paths)
    fig = plt.figure(figsize=(fs.TWO_COLUMN, 0.27 * fs.TWO_COLUMN * nr + 0.9))
    gs = fig.add_gridspec(nr, 4, left=0.07, right=0.98, top=1 - 0.75 / fig.get_figheight(),
                          bottom=0.75 / fig.get_figheight(), wspace=0.06, hspace=0.42)
    cells = [[gs[i, j] for j in range(4)] for i in range(nr)]
    ims = [None] * 4
    for i, p in enumerate(paths):
        rec, img, _ = runs[p]
        axs = [fig.add_subplot(cells[i][j]) for j in range(4)]
        m = img["model"]
        ims[0] = fs.show_image(axs[0], m, ext, norm=mnorm)
        if np.nanmin(m) < -0.02 * mmax:
            axs[0].contour(m, levels=[-0.05 * mmax, -0.02 * mmax], colors=["#00e5ff", "#7fdfff"],
                           linewidths=0.4, origin="lower", extent=ext)
        ims[1] = fs.show_image(axs[1], img["model_reconvolved"], ext, vmin=-0.05 * cmax, vmax=cmax)
        fs.add_beam(axs[1], beam, ext)
        ims[2] = fs.show_image(axs[2], img["residual"], ext, cmap=fs.RESIDUAL_CMAP,
                               norm=fs.symmetric_norm(RESID_MAX))
        if p == ref:
            axs[3].axis("off")
            axs[3].text(0.5, 0.5, "reference", ha="center", va="center", fontsize=6.5,
                        color=fs.MUTED, transform=axs[3].transAxes)
        else:
            diff = (img["model_reconvolved"] - img0["model_reconvolved"]) / rms
            ims[3] = fs.show_image(axs[3], diff, ext, cmap=fs.RESIDUAL_CMAP,
                                   norm=fs.symmetric_norm(dmax))
            axs[3].text(0.97, 0.04, f"max {np.nanmax(np.abs(diff)):.1f}σ", ha="right",
                        va="bottom", fontsize=5.5, color=fs.INK, transform=axs[3].transAxes)
        axs[2].text(0.97, 0.04, f"peak {rec['peak_residual_sigma']:.1f}σ", ha="right",
                    va="bottom", fontsize=5.5, color=fs.INK, transform=axs[2].transAxes)
        axs[0].text(0.0, 1.06, f"{LABELS[p]}", fontweight="bold", fontsize=6.3,
                    transform=axs[0].transAxes, ha="left", va="bottom", color=fs.INK)
        axs[1].text(0.08, 1.06,
                    f"{rec['wall_s']:.1f} s · {rec['max_rss_gb']:.2f} GB · λ {rec['coefficient']:.3g}"
                    f" · χ²/N {rec['chi2_per_n']:.4f} · structure ratio {rec['structure_ratio']:.2f}"
                    f" · flux {1e3 * rec['flux_jy']:.3g} mJy",
                    fontsize=6.0, transform=axs[1].transAxes, ha="left", va="bottom",
                    color=fs.MUTED)
        if i == 0:
            for ax, t in zip(axs, ["model", "reconvolved", "residual", "vs reference"]):
                ax.set_title(t, fontsize=7, color=fs.INK, pad=19)
        if i == nr - 1:
            fs.sky_axes(axs[0], ext, ticks=_ticks(ext))
    last = cells[-1]
    e = int(np.floor(np.log10(mmax)))
    fs.hcolorbar(fig, ims[0], [last[0]], "Jy / pixel", pad=0.045 / max(nr / 4, 0.6), shrink=0.9,
                 ticks=[0.0, 10.0 ** (e - 1), 10.0 ** e])
    fs.hcolorbar(fig, ims[1], [last[1]], "Jy / beam", pad=0.045 / max(nr / 4, 0.6), shrink=0.9)
    fs.hcolorbar(fig, ims[2], [last[2]], "σ", pad=0.045 / max(nr / 4, 0.6), shrink=0.9)
    if ims[3] is not None:
        fs.hcolorbar(fig, ims[3], [last[3]], "Δ reconvolved [σ]", pad=0.045 / max(nr / 4, 0.6),
                     shrink=0.9)
    mesh = runs[ref][0]["mesh"]
    fig.suptitle(f"{name}  ({mesh[0]}×{mesh[1]} mesh, beam {beam.bmaj_arcsec:.2f}″ × "
                 f"{beam.bmin_arcsec:.2f}″, rms {1e6 * rms:.3g} μJy/beam)",
                 fontsize=7.5, color=fs.INK, y=1 - 0.12 / fig.get_figheight())
    for q in fs.save(fig, f"bench_real_summary_{name}"):
        print("wrote", q)
    plt.close(fig)


def _ticks(ext):
    half = abs(ext[0])
    step = 1.0 if half <= 3 else (2.0 if half <= 6 else 10.0)
    k = int(half // step)
    return tuple(step * np.arange(-k, k + 1))


if __name__ == "__main__":
    fs.use_paper_style()
    for name in (sys.argv[1:] or list(DATASETS)):
        figure(name)

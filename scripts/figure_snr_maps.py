"""Per-pixel vs beam-resolution S/N on mock discs: why snr.fits is at the
restoring beam's resolution (docs/uncertainty.md).

    python scripts/figure_snr_maps.py      # -> figures/snr_maps_mock.{png,pdf}

Prints, per mock, the four maps' peak S/N and the source-region flux S/N.
"""
import sys, logging
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import figure_style as fs
from pyuvimage import api, mock, beam as beam_mod
from pyuvimage.products import to_fits_orientation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

captured = {}
_orig = api._products_for
def _grab(sf, dataset, *a, **k):
    captured["sf"] = sf
    return _orig(sf, dataset, *a, **k)
api._products_for = _grab


def case(sigma_jy):
    uvd, truth, geom32, _ = mock.make_demo_dataset(n_vis=4800, sigma_jy=sigma_jy, seed=3)
    res = api.run(uvd, fov=3.0, write=False, pb_correction=False)
    sf, pr = captured["sf"], res.products[0]
    g = sf.geometry
    snr_pix = np.where(pr.uncertainty > 0, sf.model_image / pr.uncertainty, 0.0)
    snr_beam = pr.snr                     # what snr.fits holds (`SingleFit.beam_snr`)
    kern = beam_mod.gaussian_kernel(pr.beam, g.pixel_scale, g.shape_native)
    snr_restored = pr.reconvolved / pr.rms
    # the truth on the product grid (area-preserving), smoothed by the same beam
    t = ndimage.zoom(truth / geom32.mesh_pixel_scale**2, g.shape_native[0] / truth.shape[0],
                     order=1, grid_mode=True, mode="grid-constant") * g.pixel_scale**2
    snr_true = fftconvolve(t, kern, mode="same") / pr.rms
    return dict(g=g, beam=pr.beam, true=snr_true, pix=snr_pix, beam_snr=snr_beam,
                restored=snr_restored, crit=res.parameters["source_prior"]["criterion"],
                flux=res.parameters.get("source_flux"))


fs.use_paper_style()
cases = [("bright source", 0.011), ("faint source", 0.045)]
results = [case(s) for _, s in cases]

titles = ["truth $\\otimes$ beam / rms\n(what we want to see)",
          "model / uncertainty\nper pixel (the old snr.fits)",
          "snr.fits\nbeam resolution, full covariance",
          "restored image / rms\n(CLEAN-style)"]
keys = ["true", "pix", "beam_snr", "restored"]
colors = ["#111111", "#d55e00", "#0072b2", "#8a8a8a"]
styles = ["-", "-", "-", "--"]

fig = plt.figure(figsize=(fs.TWO_COLUMN * 1.4, fs.TWO_COLUMN * 0.7))
gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.06, 0.42, 1.3], wspace=0.08,
                      hspace=0.3, left=0.06, right=0.985, top=0.89, bottom=0.17)
for r, ((label, _), res) in enumerate(zip(cases, results)):
    g = res["g"]; N = g.shape_native[0]
    ext = fs.sky_extent(N, g.pixel_scale)
    vmax = float(np.nanmax(res["true"]))
    axes = []
    for c, (k, title) in enumerate(zip(keys, titles)):
        ax = fig.add_subplot(gs[r, c]); axes.append(ax)
        img = to_fits_orientation(res[k])
        im = fs.show_image(ax, img, ext, cmap="magma", vmin=-2, vmax=vmax)
        ax.contour(img, levels=[3, 5], colors=["#9ecae1", "white"], linewidths=[0.5, 0.7],
                   origin="lower", extent=ext)
        if r == 0:
            ax.set_title(title, fontsize=6.3, color=fs.INK, pad=3)
        fs.panel_label(ax, f"peak {np.nanmax(res[k]):.1f}", loc="upper right", size=5.8)
        if c == 0:
            fs.add_beam(ax, res["beam"], ext)
            fs.sky_axes(ax, ext)
            fs.panel_label(ax, label, loc="upper left", size=6.3)
    cax = fig.add_subplot(gs[r, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(fs.MUTED)
    cb.ax.tick_params(labelsize=5.5, length=1.6, width=0.4, color=fs.MUTED)
    cb.set_label("S/N", fontsize=6.0, color=fs.INK, labelpad=2)
    # a cut through the peak of the truth, along RA
    cut = fig.add_subplot(gs[r, 6])
    cy, cx = np.unravel_index(np.nanargmax(res["true"]), res["true"].shape)
    x = (np.arange(N) - (N - 1) / 2) * g.pixel_scale
    for k, col, ls in zip(keys, colors, styles):
        cut.plot(-x, res[k][cy, :], color=col, ls=ls, lw=1.1)
    for lev in (3, 5):
        cut.axhline(lev, color=fs.FAINT, lw=0.6, zorder=0)
    cut.set_xlim(1.5, -1.5)
    cut.set_xlabel(r"$\Delta$RA [arcsec]", fontsize=6, color=fs.INK)
    cut.set_ylabel("S/N", fontsize=6, color=fs.INK)
    cut.tick_params(labelsize=5.5)
    for s in ("top", "right"):
        cut.spines[s].set_visible(False)
    f = res["flux"]
    cut.set_title("cut through the peak" + (f"  (region flux {f['snr']:.1f}$\\sigma$)" if f else ""),
                  fontsize=6.3, color=fs.INK, pad=3)
handles = [Line2D([], [], color=c_, ls=l_, lw=1.1) for c_, l_ in zip(colors, styles)]
fig.legend(handles, ["truth $\\otimes$ beam / rms", "per pixel (old snr.fits)",
                     "beam resolution (snr.fits)", "restored image / rms"],
           fontsize=6, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
fig.suptitle("Mock exponential disc, 4800 visibilities: contours at S/N 3 and 5; "
             "one colour scale per row", fontsize=7, color=fs.INK)
_out = __import__("pathlib").Path(__file__).resolve().parent.parent / "figures"
_out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(_out / f"snr_maps_mock.{ext}", dpi=250)
for (label, s), r_ in zip(cases, results):
    print(label, s, r_["crit"], {k: round(float(np.nanmax(r_[k])), 1) for k in keys}, r_["flux"] and round(r_["flux"]["snr"], 1))

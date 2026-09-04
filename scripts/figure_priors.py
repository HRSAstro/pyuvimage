"""Figure: what each source prior does to the same data.

One mock, one criterion, four priors. The mock is the extended disc with an
unresolved compact knot (`mock.make_extended_plus_compact_dataset`), because
that is the case that separates them: on a smooth disc every prior looks
much the same, and a gallery of four near-identical panels is not an
argument.

Everything except `--reg` is held fixed, including the criterion. The
coefficient is still fitted per prior -- holding *that* fixed would compare
four differently-converged fits rather than four priors.

    python scripts/figure_priors.py            # fit, then plot
    python scripts/figure_priors.py --plot     # re-plot from the cache

Writes figures/priors_comparison.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

import figure_style as fs  # noqa: E402  (same directory)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CACHE = Path("/tmp/pyuvimage_fig_priors")

#: The four priors that behave differently. `exponential` is `matern` with
#: nu = 0.5 and `constant` is rank-deficient, so neither adds a row.
PRIORS = [
    ("matern", "Matérn", "stationary; correlation length = beam"),
    ("gaussian", "Gaussian envelope", "prior width tapered off the centre"),
    ("gibbs", "Gibbs", "correlation length short where bright"),
    ("adaptive", "adaptive", "prior amplitude follows the brightness"),
]

# Well-constrained on purpose: 8000 data points against 1024 mesh pixels is
# 7.8 per pixel, comfortably out of the regime where the faint structure is
# set by the prior rather than the data (and where the residual-structure
# statistic is not calibrated -- see docs/design-notes.md).
N_VIS = 4000
MESH_N = 32
FOV = 3.0
# Peak signal-to-noise ~400, which is the range the real ALMA datasets sit in
# (117-285 sigma). Deeper than that and the fit starts to be limited by the
# mesh's edge pixels, which the non-negative solver holds at zero: any flux
# the source has at the field boundary then shows up as a bright rim in the
# residual, which is a statement about the field of view rather than about
# the prior.
SIGMA_JY = 1.5e-3
CRITERION = "discrepancy"

# The compact component is deliberately *resolved*: r_eff 0.18" against a
# 0.25" beam, so it spans a couple of beams rather than sitting inside one.
# At the mock's default 0.03" it is unresolved and therefore indistinguishable
# from a delta function -- which is the subject of the point-source figure,
# not this one. What this figure is about is a source with two very different
# scales in it (0.18" and 0.70"), which is the situation a single stationary
# smoothing length cannot serve.
COMPACT_R_EFF = 0.18
COMPACT_FLUX = 0.012
# and the disc is smaller than the mock's 0.70" default, so that it is
# contained by the 3" field: at 0.70" nearly 4% of the flux sits in the outer
# two mesh rings, and zeroed edge pixels turned that into an 11 sigma rim
# around every residual panel.
EXTENDED_R_EFF = 0.35


def fit_all() -> dict:
    from pyuvimage import mock
    import pyuvimage

    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True)

    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=N_VIS, mesh_n=MESH_N, fov_arcsec=FOV, sigma_jy=SIGMA_JY,
        compact_r_eff=COMPACT_R_EFF, compact_flux=COMPACT_FLUX,
        extended_r_eff=EXTENDED_R_EFF,
    )
    np.save(CACHE / "truth.npy", np.asarray(truth, dtype=float))
    (CACHE / "geometry.json").write_text(json.dumps({
        "fov": FOV,
        "pixel_scale": geom.pixel_scale,
        "mesh_pixel_scale": geom.mesh_pixel_scale,
        "mesh_shape": list(geom.mesh_shape),
        "n_data": 2 * int(uvd.n_samples),
        "components": {
            "extended_flux": comps["extended"]["flux"],
            "compact_flux": comps["compact"]["flux"],
            "compact_centre": list(comps["compact"]["centre"]),
            "compact_r_eff": comps["compact"]["r_eff"],
            "extended_r_eff": comps["extended"]["r_eff"],
        },
    }, indent=2))

    for reg, _, _ in PRIORS:
        out = CACHE / reg
        t = time.time()
        pyuvimage.run(
            uvd, fov=FOV, out=out, reg=reg, criterion=CRITERION,
            mesh_shape=(MESH_N, MESH_N),
            uncertainty_map=False, pb_correction=False, mask_shape="square",
        )
        print(f"{reg:>10s}  {time.time() - t:5.1f} s", flush=True)
    return {}


def _load(reg: str):
    d = CACHE / reg
    model = fits.getdata(d / "model.fits").astype(float)
    resid = fits.getdata(d / "residual.fits").astype(float)
    hdr = fits.getheader(d / "model_reconvolved.fits")
    params = json.loads((d / "fit_parameters.json").read_text())
    return model, resid, hdr, params


def plot() -> None:
    import matplotlib.pyplot as plt
    from pyuvimage.beam import BeamFit
    from pyuvimage.products import to_fits_orientation

    fs.use_paper_style()
    geo = json.loads((CACHE / "geometry.json").read_text())
    truth_mesh = np.load(CACHE / "truth.npy")
    pix, mpix = geo["pixel_scale"], geo["mesh_pixel_scale"]

    # Jy/pixel on two different grids is not comparable; surface brightness
    # is. Every intensity panel below is Jy/arcsec^2 on one shared scale.
    truth_sb = to_fits_orientation(truth_mesh) / mpix**2
    dirty = fits.getdata(CACHE / PRIORS[0][0] / "dirty_image.fits").astype(float)

    fitted = {reg: _load(reg) for reg, _, _ in PRIORS}
    models_sb = {reg: fitted[reg][0] / pix**2 for reg, _, _ in PRIORS}

    hdr0 = fitted[PRIORS[0][0]][2]
    beam = BeamFit(bmaj_arcsec=hdr0["BMAJ"] * 3600.0,
                   bmin_arcsec=hdr0["BMIN"] * 3600.0,
                   bpa_deg=hdr0["BPA"])
    rms = float(hdr0["RMS"])
    beam_area = beam.beam_area_pixels(1.0)          # arcsec^2 per beam

    ext_img = fs.sky_extent(dirty.shape[0], pix)
    ext_mesh = fs.sky_extent(truth_sb.shape[0], mpix)

    # One intensity scale for truth and every model. The stretch is linear
    # below the image-plane noise and logarithmic above it, so the 0.7" disc
    # (a few times the noise) and the compact knot are both legible; a linear
    # scale shows the knot and nothing else.
    vmax_i = max(float(np.nanmax(truth_sb)),
                 *[float(np.nanmax(m)) for m in models_sb.values()])
    norm_i = fs.asinh_norm(vmax_i, linear_width=2.0 * rms / beam_area)

    # One residual scale, set by the *worst* panel and not clipped: clipping
    # it would flatten the difference between a 4 sigma residual and a 12
    # sigma one, which is the entire content of this column.
    rmax = float(np.nanmax([np.nanmax(np.abs(fitted[r][1])) for r, _, _ in PRIORS]))
    rmax = 2.0 * np.ceil(rmax / 2.0)
    norm_r = fs.symmetric_norm(rmax)

    # Two blocks: a reference row (what is there, what the telescope sees),
    # then one row per prior. Separate gridspecs so the reference row can
    # carry its own titles without pretending to be a model and a residual.
    n_p = len(PRIORS)
    fig = plt.figure(figsize=(fs.ONE_COLUMN, fs.ONE_COLUMN * 0.54 * (n_p + 1) + 0.55))
    top, bar_h = 0.955, 0.014
    row_h = (top - 0.085) / (n_p + 1)
    gs_ref = fig.add_gridspec(1, 2, left=0.115, right=0.995,
                              top=top, bottom=top - row_h * 0.90, wspace=0.05)
    gs = fig.add_gridspec(n_p, 2, left=0.115, right=0.995,
                          top=top - row_h * 1.06, bottom=0.085,
                          wspace=0.05, hspace=0.05)

    ax_t, ax_d = fig.add_subplot(gs_ref[0, 0]), fig.add_subplot(gs_ref[0, 1])
    im_i = fs.show_image(ax_t, truth_sb, ext_mesh, norm=norm_i)
    fs.show_image(ax_d, dirty / rms, ext_img, cmap=fs.INTENSITY_CMAP)
    ax_t.set_title("true sky", pad=2.5, color=fs.INK)
    ax_d.set_title("dirty image", pad=2.5, color=fs.INK)
    fs.add_beam(ax_d, beam, ext_img)
    fs.panel_label(ax_t, f"{1e3 * float(np.nansum(truth_mesh)):.1f} mJy",
                   loc="lower left", size=5.6)
    fs.panel_label(ax_d, f"peak {np.nanmax(dirty) / rms:.0f}$\\sigma$",
                   loc="lower left", size=5.6)

    for row, (reg, label, blurb) in enumerate(PRIORS):
        model, resid, _, params = fitted[reg]
        ax_m = fig.add_subplot(gs[row, 0])
        ax_r = fig.add_subplot(gs[row, 1])
        im_i = fs.show_image(ax_m, models_sb[reg], ext_img, norm=norm_i)
        im_r = fs.show_image(ax_r, resid, ext_img, cmap=fs.RESIDUAL_CMAP,
                             norm=norm_r)
        fs.add_beam(ax_m, beam, ext_img)
        if row == 0:
            ax_m.set_title("model", pad=2.5, color=fs.INK)
            ax_r.set_title("residual", pad=2.5, color=fs.INK)

        chi2 = params["fit_quality"]["chi_squared"] / params["fit_quality"]["n_data"]
        peak = float(np.nanmax(np.abs(resid)))
        flux = float(np.nansum(model))
        # mathtext swallows spaces, so the descriptor lives in the caption
        # rather than in a rotated label nobody can read anyway
        ax_m.set_ylabel(label, color=fs.INK, fontsize=7.5, labelpad=3)
        fs.panel_label(ax_r, f"$\\chi^2/N$ {chi2:.3f}", loc="upper right",
                       color=fs.INK, size=5.6)
        fs.panel_label(ax_r, f"peak {peak:.1f}$\\sigma$", loc="lower right",
                       color=fs.INK, size=5.6)
        fs.panel_label(ax_m, f"{1e3 * flux:.1f} mJy", loc="lower left",
                       size=5.6)
        if row == n_p - 1:
            fs.sky_axes(ax_m, ext_img)

        fs.panel_label(ax_r, f"$\\lambda$ = {params['source_prior']['coefficient']:.2g}",
                       loc="lower left", color=fs.INK, size=5.4)

    fs.hcolorbar(fig, im_i, [gs[n_p - 1, 0]], "Jy arcsec$^{-2}$", bar_h,
                 ticks=[0.0, 1e-3, 1e-2, 1e-1])
    fs.hcolorbar(fig, im_r, [gs[n_p - 1, 1]], "residual [$\\sigma$]", bar_h)

    for p in fs.save(fig, "priors_comparison"):
        print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true",
                    help="skip fitting and re-plot from the cache")
    args = ap.parse_args()
    if not args.plot:
        fit_all()
    plot()

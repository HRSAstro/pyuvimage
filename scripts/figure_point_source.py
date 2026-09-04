"""Figure: an extended disc with a true point source, fitted both ways.

The same data twice. The top row is the pixelised model on its own; the
bottom row adds an analytic delta component solved in the same linear system
(`--point-sources`). Columns are the four products a user actually looks at:
what the telescope saw, what was fitted, what is left over, and how well each
pixel is known.

A point source is the one thing a pixel grid cannot represent, so the top row
is not a straw man -- it is what the reconstruction does when the model has no
term for the thing in the data.

    python scripts/figure_point_source.py            # fit, then plot
    python scripts/figure_point_source.py --plot     # re-plot from the cache

Writes figures/point_source.{pdf,png}.
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

CACHE = Path("/tmp/pyuvimage_fig_point")

N_VIS = 4000
MESH_N = 32
FOV = 3.0
#: chosen so the point sits at a few tens of sigma -- clearly real, but not so
#: bright that recovering it is trivial
SIGMA_JY = 1.0e-3
POINT_FLUX = 0.004
POINT_CENTRE = (0.85, -0.65)      # (dRA, dDec) arcsec

#: The labels name the flag, because the figure's job is partly to tell a
#: reader what to type. The bottom row auto-detects the position -- nothing
#: is supplied with `--point x,y` -- so the recovered offset is a result,
#: not an input.
RUNS = [("mesh", "default\n(pixelised model only)", False),
        ("point", "--point-sources\n(one delta fitted)", True)]


def fit_all() -> None:
    from pyuvimage import mock
    import pyuvimage

    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True)

    uvd, truth, geom, comps = mock.make_demo_dataset(
        n_vis=N_VIS, mesh_n=MESH_N, fov_arcsec=FOV, sigma_jy=SIGMA_JY,
        point_flux_jy=POINT_FLUX, point_centre=POINT_CENTRE,
    )
    np.save(CACHE / "truth.npy", np.asarray(truth, dtype=float))
    (CACHE / "geometry.json").write_text(json.dumps({
        "pixel_scale": geom.pixel_scale,
        "mesh_pixel_scale": geom.mesh_pixel_scale,
        "n_data": 2 * int(uvd.n_samples),
        "point_flux": POINT_FLUX,
        "point_centre": list(POINT_CENTRE),
        "extended_flux": comps["extended"][0]["flux"],
    }, indent=2))

    for tag, _, points in RUNS:
        t = time.time()
        pyuvimage.run(
            uvd, fov=FOV, out=CACHE / tag, mesh_shape=(MESH_N, MESH_N),
            reg="adaptive", criterion="discrepancy",
            point_sources=bool(points),
            uncertainty_map=True, pb_correction=False, mask_shape="square",
        )
        print(f"{tag:>6s}  {time.time() - t:5.1f} s", flush=True)


def _load(tag: str):
    d = CACHE / tag
    out = {
        "dirty": fits.getdata(d / "dirty_image.fits").astype(float),
        "model": fits.getdata(d / "model.fits").astype(float),
        "resid": fits.getdata(d / "residual.fits").astype(float),
        "unc": fits.getdata(d / "uncertainty.fits").astype(float),
        "hdr": fits.getheader(d / "model_reconvolved.fits"),
        "params": json.loads((d / "fit_parameters.json").read_text()),
    }
    pj = d / "point_sources.json"
    out["points"] = json.loads(pj.read_text())["points"] if pj.exists() else []
    return out


def plot() -> None:
    import matplotlib.pyplot as plt
    from pyuvimage.beam import BeamFit
    from pyuvimage.products import to_fits_orientation

    fs.use_paper_style()
    geo = json.loads((CACHE / "geometry.json").read_text())
    pix, mpix = geo["pixel_scale"], geo["mesh_pixel_scale"]
    truth_sb = to_fits_orientation(np.load(CACHE / "truth.npy")) / mpix**2

    runs = {tag: _load(tag) for tag, _, _ in RUNS}
    ref = runs["point"]
    beam = BeamFit(bmaj_arcsec=ref["hdr"]["BMAJ"] * 3600.0,
                   bmin_arcsec=ref["hdr"]["BMIN"] * 3600.0,
                   bpa_deg=ref["hdr"]["BPA"])
    rms = float(ref["hdr"]["RMS"])
    beam_area = beam.beam_area_pixels(1.0)

    ext = fs.sky_extent(ref["dirty"].shape[0], pix)
    ext_mesh = fs.sky_extent(truth_sb.shape[0], mpix)

    # shared scales across both rows, so the two fits are comparable
    models_sb = {t: runs[t]["model"] / pix**2 for t, _, _ in RUNS}
    vmax_i = max(float(np.nanmax(truth_sb)),
                 *[float(np.nanmax(m)) for m in models_sb.values()])
    norm_i = fs.asinh_norm(vmax_i, linear_width=2.0 * rms / beam_area)
    rmax = 2.0 * np.ceil(max(float(np.nanmax(np.abs(runs[t]["resid"])))
                             for t, _, _ in RUNS) / 2.0)
    norm_r = fs.symmetric_norm(rmax)
    unc_sb = {t: runs[t]["unc"] / pix**2 for t, _, _ in RUNS}
    umax = max(float(np.nanpercentile(u, 99.5)) for u in unc_sb.values())

    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.53))
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.995, top=0.930,
                          bottom=0.215, wspace=0.05, hspace=0.05)

    for row, (tag, label, _) in enumerate(RUNS):
        r = runs[tag]
        ax0 = fig.add_subplot(gs[row, 0])
        if row == 0:
            im_i = fs.show_image(ax0, r["dirty"] / rms, ext,
                                 cmap=fs.INTENSITY_CMAP)
            ax0.set_title("data / truth", pad=2.5, color=fs.INK)
            fs.panel_label(ax0, "dirty image")
            fs.panel_label(ax0, f"peak {np.nanmax(r['dirty']) / rms:.0f}$\\sigma$",
                           loc="lower left", size=5.6)
        else:
            fs.show_image(ax0, truth_sb, ext_mesh, norm=norm_i)
            fs.panel_label(ax0, "true sky")
            fs.panel_label(
                ax0,
                f"{1e3 * (geo['extended_flux'] + geo['point_flux']):.1f} mJy",
                loc="lower left", size=5.6)
            fs.sky_axes(ax0, ext_mesh)
        fs.add_beam(ax0, beam, ext)

        ax_m = fig.add_subplot(gs[row, 1])
        im_i = fs.show_image(ax_m, models_sb[tag], ext, norm=norm_i)
        ax_r = fig.add_subplot(gs[row, 2])
        im_r = fs.show_image(ax_r, r["resid"], ext, cmap=fs.RESIDUAL_CMAP,
                             norm=norm_r)
        ax_u = fig.add_subplot(gs[row, 3])
        im_u = fs.show_image(ax_u, unc_sb[tag], ext, cmap=fs.UNCERTAINTY_CMAP,
                             vmin=0.0, vmax=umax)
        if row == 0:
            for ax, t in ((ax_m, "model"), (ax_r, "residual"),
                          (ax_u, "1$\\sigma$ uncertainty")):
                ax.set_title(t, pad=2.5, color=fs.INK)

        ax_m.set_ylabel(label, color=fs.INK, fontsize=6.8, labelpad=3,
                        linespacing=1.6)
        chi2 = (r["params"]["fit_quality"]["chi_squared"]
                / r["params"]["fit_quality"]["n_data"])
        fs.panel_label(ax_r, f"$\\chi^2/N$ {chi2:.3f}", loc="upper right",
                       color=fs.INK, size=5.6)
        fs.panel_label(ax_r, f"peak {np.nanmax(np.abs(r['resid'])):.1f}$\\sigma$",
                       loc="lower right", color=fs.INK, size=5.6)

        # model.fits already carries each point's flux in its nearest pixel
        # (`products.model_with_points`), so this total is the whole model --
        # adding the point flux again would double-count it.
        total = float(np.nansum(r["model"]))
        pt_flux = sum(p["flux_jy"] for p in r["points"])
        detail = (f"\n{1e3 * (total - pt_flux):.1f} mesh + "
                  f"{1e3 * pt_flux:.1f} point") if r["points"] else ""
        fs.panel_label(ax_m, f"{1e3 * total:.1f} mJy{detail}",
                       loc="lower left", size=5.6)
        if r["points"]:
            p0 = r["points"][0]
            off = np.hypot(p0["d_ra_arcsec"] - geo["point_centre"][0],
                           p0["d_dec_arcsec"] - geo["point_centre"][1])
            fs.panel_label(
                ax_m,
                f"point {1e3 * p0['flux_jy']:.2f} $\\pm$ "
                f"{1e3 * p0['flux_error_jy']:.2f} mJy\n"
                f"{1e3 * off:.0f} mas from truth",
                loc="upper right", size=5.4)
            ax_m.plot(p0["d_ra_arcsec"], p0["d_dec_arcsec"], "+",
                      color="#7ef9ff", markersize=4, markeredgewidth=0.7)

    fs.hcolorbar(fig, im_i, [gs[1, 0], gs[1, 1]], "Jy arcsec$^{-2}$", 0.022,
                 pad=0.135, ticks=[0.0, 1e-3, 1e-2, 1e-1, 1e0])
    fs.hcolorbar(fig, im_r, [gs[1, 2]], "residual [$\\sigma$]", 0.022,
                 pad=0.135)
    fs.hcolorbar(fig, im_u, [gs[1, 3]], "Jy arcsec$^{-2}$", 0.022, pad=0.135)

    for p in fs.save(fig, "point_source"):
        print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true",
                    help="skip fitting and re-plot from the cache")
    args = ap.parse_args()
    if not args.plot:
        fit_all()
    plot()

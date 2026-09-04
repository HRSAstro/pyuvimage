"""Figure: how the reconstruction degrades as the signal-to-noise falls.

The same sky, the same uv coverage and the same noise *realisation* at six
amplitudes, so signal-to-noise is the only thing that changes between
columns. The source is the extended disc with a true point component used in
`figure_point_source.py`, fitted the same way, which gives two things to
track: whether the extended flux survives, and whether the point is still
detected and its error bar still covers the truth.

The series is deliberately run past the point where the method works. The
faintest column is below the 5 sigma detection cut, and the point is lost --
a figure that only shows the comfortable regime does not tell a reader where
to stop trusting the tool.

    python scripts/figure_snr_series.py            # fit, then plot
    python scripts/figure_snr_series.py --plot     # re-plot from the cache

Writes figures/snr_series.{pdf,png}.
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

CACHE = Path("/tmp/pyuvimage_fig_snr")

#: the envelope prior, as in `figure_point_source.py` -- see the note there
REG = "gaussian"

N_VIS = 4000
MESH_N = 32
FOV = 3.0
POINT_FLUX = 0.004
POINT_CENTRE = (0.85, -0.65)
EXTENDED_FLUX = 0.050

#: Noise amplitudes giving a dirty-image peak of roughly 300, 130, 60, 25,
#: 11 and 5 sigma. Calibrated from a measured 468 sigma at sigma_jy = 1e-3;
#: the peak S/N actually achieved is measured per run and used for the
#: labels, so this list only has to be about right.
SIGMAS = [1.56e-3, 3.55e-3, 8.07e-3, 1.80e-2, 4.25e-2, 9.36e-2]

#: `api.run`'s default significance cut, redrawn on the trend panel
DETECTION_CUT = 5.0


def fit_all() -> None:
    from pyuvimage import mock
    import pyuvimage

    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True)

    records = []
    for i, sigma in enumerate(SIGMAS):
        # One seed throughout: the uv coverage and the *normalised* noise draw
        # are then identical across the series and only the amplitude changes,
        # so a difference between columns is a difference in signal-to-noise
        # and not in which noise happened to land where.
        uvd, truth, geom, comps = mock.make_demo_dataset(
            n_vis=N_VIS, mesh_n=MESH_N, fov_arcsec=FOV, sigma_jy=sigma,
            point_flux_jy=POINT_FLUX, point_centre=POINT_CENTRE, seed=0,
        )
        if i == 0:
            np.save(CACHE / "truth.npy", np.asarray(truth, dtype=float))
            (CACHE / "geometry.json").write_text(json.dumps({
                "pixel_scale": geom.pixel_scale,
                "mesh_pixel_scale": geom.mesh_pixel_scale,
                "point_flux": POINT_FLUX,
                "point_centre": list(POINT_CENTRE),
                "extended_flux": EXTENDED_FLUX,
                "reg": REG,
            }, indent=2))

        out = CACHE / f"snr{i}"
        t = time.time()
        try:
            pyuvimage.run(
                uvd, fov=FOV, out=out, mesh_shape=(MESH_N, MESH_N),
                reg=REG, criterion="discrepancy", point_sources=True,
                uncertainty_map=True, pb_correction=False,
                mask_shape="square",
            )
            failed = ""
        except Exception as e:                    # a fit may legitimately fail
            failed = f"{type(e).__name__}: {e}"   # at the faint end; record it
        records.append({"index": i, "sigma_jy": sigma, "failed": failed,
                        "seconds": time.time() - t})
        print(f"  sigma {sigma:.2e}  {time.time() - t:5.1f} s  {failed}",
              flush=True)
    (CACHE / "runs.json").write_text(json.dumps(records, indent=2))


def _load(i: int):
    d = CACHE / f"snr{i}"
    if not (d / "model.fits").exists():
        return None
    hdr = fits.getheader(d / "model_reconvolved.fits")
    pj = d / "point_sources.json"
    params = json.loads((d / "fit_parameters.json").read_text())
    dirty = fits.getdata(d / "dirty_image.fits").astype(float)
    rms = float(hdr["RMS"])
    return {
        "dirty": dirty,
        "model": fits.getdata(d / "model.fits").astype(float),
        # the restored image: at six columns across a page a one-pixel delta
        # is smaller than a rendered pixel and simply disappears, so the top
        # row shows the model convolved with the beam, which is also what a
        # reader compares against a CLEAN image
        "restored": fits.getdata(d / "model_reconvolved.fits").astype(float),
        "resid": fits.getdata(d / "residual.fits").astype(float),
        "unc": fits.getdata(d / "uncertainty.fits").astype(float),
        "rms": rms,
        "peak_snr": float(np.nanmax(dirty)) / rms,
        "hdr": hdr,
        "params": params,
        "points": (json.loads(pj.read_text())["points"] if pj.exists() else []),
    }


def plot() -> None:
    import matplotlib.pyplot as plt
    from pyuvimage.beam import BeamFit
    from pyuvimage.products import to_fits_orientation

    fs.use_paper_style()
    geo = json.loads((CACHE / "geometry.json").read_text())
    pix, mpix = geo["pixel_scale"], geo["mesh_pixel_scale"]
    truth_sb = to_fits_orientation(np.load(CACHE / "truth.npy")) / mpix**2

    runs = [(i, _load(i)) for i in range(len(SIGMAS))]
    runs = [(i, r) for i, r in runs if r is not None]
    n = len(runs)

    ref = runs[0][1]
    beam = BeamFit(bmaj_arcsec=ref["hdr"]["BMAJ"] * 3600.0,
                   bmin_arcsec=ref["hdr"]["BMIN"] * 3600.0,
                   bpa_deg=ref["hdr"]["BPA"])
    beam_area = beam.beam_area_pixels(1.0)
    ext = fs.sky_extent(ref["dirty"].shape[0], pix)

    # Intensity: one scale for every panel, in Jy/beam. The stretch's linear
    # width is the *highest* S/N run's noise, so the faint columns are not
    # flattened into the linear regime -- they are noisier, and seeing that
    # is the point of the figure.
    models_sb = [r["restored"] for _, r in runs]
    vmax_i = max(float(np.nanmax(m)) for m in models_sb)
    norm_i = fs.asinh_norm(vmax_i, linear_width=2.0 * ref["rms"])
    # Residuals are already in units of each column's own sigma, so one scale
    # is meaningful across the series without renormalising anything.
    rmax = 2.0 * np.ceil(max(float(np.nanmax(np.abs(r["resid"])))
                             for _, r in runs) / 2.0)
    norm_r = fs.symmetric_norm(min(rmax, 12.0))
    # Uncertainty on ONE scale across the series, asinh-stretched over the
    # 60x range the noise spans. Normalising each panel to itself would have
    # made the row look constant, which is the opposite of what happens: the
    # uncertainty grows with the noise, and that growth is the row's content.
    uncs = [r["unc"] / pix**2 for _, r in runs]
    umax = max(float(np.nanpercentile(u, 99.8)) for u in uncs)
    umin = min(float(np.nanpercentile(u, 50.0)) for u in uncs)
    norm_u = fs.asinh_norm(umax, linear_width=0.5 * umin)

    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.80))
    gs_img = fig.add_gridspec(3, n, left=0.075, right=0.995, top=0.955,
                              bottom=0.475, wspace=0.04, hspace=0.04)
    gs_tr = fig.add_gridspec(1, 3, left=0.065, right=0.995, top=0.365,
                             bottom=0.065, wspace=0.34)

    for col, (i, r) in enumerate(runs):
        ax_m = fig.add_subplot(gs_img[0, col])
        ax_r = fig.add_subplot(gs_img[1, col])
        ax_u = fig.add_subplot(gs_img[2, col])
        im_i = fs.show_image(ax_m, models_sb[col], ext, norm=norm_i)
        im_r = fs.show_image(ax_r, r["resid"], ext, cmap=fs.RESIDUAL_CMAP,
                             norm=norm_r)
        im_u = fs.show_image(ax_u, uncs[col], ext, cmap=fs.UNCERTAINTY_CMAP,
                             norm=norm_u)
        ax_m.set_title(f"{r['peak_snr']:.0f}$\\sigma$", pad=2.5, color=fs.INK)
        if col == 0:
            for ax, t in ((ax_m, "model $\\otimes$ beam"), (ax_r, "residual"),
                          (ax_u, "1$\\sigma$ uncertainty")):
                ax.set_ylabel(t, color=fs.INK, fontsize=6.6, labelpad=3)
            fs.add_beam(ax_m, beam, ext)
        if r["points"]:
            p0 = r["points"][0]
            ax_m.plot(p0["d_ra_arcsec"], p0["d_dec_arcsec"], "+",
                      color="#7ef9ff", markersize=3.5, markeredgewidth=0.6)
        else:
            fs.panel_label(ax_m, "no point", loc="lower right", size=5.2)

    fs.hcolorbar(fig, im_i, [gs_img[2, 0], gs_img[2, 1]],
                 "model $\\otimes$ beam [Jy/beam]", 0.013, pad=0.055,
                 shrink=0.86, ticks=[0.0, 1e-4, 1e-3])
    fs.hcolorbar(fig, im_r, [gs_img[2, 2], gs_img[2, 3]],
                 "residual [$\\sigma$]", 0.013, pad=0.055, shrink=0.86)
    fs.hcolorbar(fig, im_u, [gs_img[2, 4], gs_img[2, n - 1]],
                 "uncertainty [Jy arcsec$^{-2}$]", 0.013, pad=0.055,
                 shrink=0.86, ticks=[0.0, 1e-4, 1e-3, 1e-2])

    # ---- what the panels cannot show: numbers, with error bars ----------
    snr = np.array([r["peak_snr"] for _, r in runs])
    truth_ext = geo["extended_flux"]
    mesh_ratio, pt_ratio, pt_err, sig, detected = [], [], [], [], []
    for _, r in runs:
        total = float(np.nansum(r["model"]))
        pf = sum(p["flux_jy"] for p in r["points"])
        mesh_ratio.append((total - pf) / truth_ext)
        if r["points"]:
            p0 = r["points"][0]
            pt_ratio.append(p0["flux_jy"] / geo["point_flux"])
            pt_err.append(p0["flux_error_jy"] / geo["point_flux"])
            sig.append(p0["significance"])
            detected.append(True)
        else:
            pt_ratio.append(np.nan); pt_err.append(np.nan)
            sig.append(np.nan); detected.append(False)

    ax_a = fig.add_subplot(gs_tr[0, 0])
    ax_a.axhline(1.0, color=fs.MUTED, lw=0.6, ls="--", zorder=1)
    ax_a.plot(snr, mesh_ratio, "o-", color="#2a78d6", ms=3.2, lw=1.0,
              label="extended (mesh)", zorder=3)
    ok = np.array(detected)
    ax_a.errorbar(snr[ok], np.array(pt_ratio)[ok], yerr=np.array(pt_err)[ok],
                  fmt="s-", color="#eb6834", ms=3.0, lw=1.0, elinewidth=0.8,
                  capsize=1.6, label="point component", zorder=3)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("peak signal-to-noise [$\\sigma$]", fontsize=6.5)
    ax_a.set_ylabel("recovered / true flux", fontsize=6.5)
    ax_a.legend(fontsize=5.6, frameon=False, loc="lower right")

    ax_b = fig.add_subplot(gs_tr[0, 1])
    ax_b.axhline(DETECTION_CUT, color=fs.MUTED, lw=0.6, ls="--", zorder=1)
    ax_b.text(snr.max() * 1.5, DETECTION_CUT * 1.15,
              f"{DETECTION_CUT:.0f}$\\sigma$ detection cut", fontsize=5.6,
              color=fs.MUTED, va="bottom", ha="left")
    ax_b.plot(snr[ok], np.array(sig)[ok], "s-", color="#eb6834", ms=3.0,
              lw=1.0, zorder=3)
    for s, d in zip(snr, detected):
        if not d:
            ax_b.plot([s], [DETECTION_CUT * 0.60], "x", color=fs.MUTED,
                      ms=4.0, mew=0.9)
            ax_b.text(s, DETECTION_CUT * 0.72, "not detected", fontsize=5.2,
                      color=fs.MUTED, ha="center", va="bottom", rotation=90)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlabel("peak signal-to-noise [$\\sigma$]", fontsize=6.5)
    ax_b.set_ylabel("point-source significance [$\\sigma$]", fontsize=6.5)

    # ---- and the question the uncertainty row raises ---------------------
    # The quoted uncertainty *falls* as the data get worse, because a weaker
    # dataset gets a stronger prior and a strong prior shrinks the posterior
    # variance. The actual error does the opposite. Where the two cross is
    # where `uncertainty.fits` stops being a usable error bar, and it is not
    # a detail a caption can be trusted to carry.
    k = ref["model"].shape[0] // truth_sb.shape[0]
    truth_img = to_fits_orientation(
        np.kron(np.load(CACHE / "truth.npy"), np.ones((k, k))) / k**2)
    src = truth_img > 0.02 * truth_img.max()
    quoted, actual = [], []
    for _, r in runs:
        m = r["model"].copy()
        if r["points"]:                     # the delta is not extended emission
            p0 = r["points"][0]
            npx = m.shape[0]
            half = 0.5 * npx * pix
            yy, xx = np.mgrid[0:npx, 0:npx]
            dra = half - (xx + 0.5) * pix
            ddec = -half + (yy + 0.5) * pix
            near = np.hypot(dra - p0["d_ra_arcsec"],
                            ddec - p0["d_dec_arcsec"]) < 1.5 * pix
            m[near] = truth_img[near]
        quoted.append(float(np.nanmedian(r["unc"][src])))
        actual.append(float(np.sqrt(np.nanmean((m - truth_img)[src] ** 2))))

    ax_c = fig.add_subplot(gs_tr[0, 2])
    ax_c.plot(snr, quoted, "o-", color="#1baf7a", ms=3.2, lw=1.0,
              label="quoted 1$\\sigma$ (median)", zorder=3)
    ax_c.plot(snr, actual, "^-", color="#4a3aa7", ms=3.4, lw=1.0,
              label="actual rms error", zorder=3)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xlabel("peak signal-to-noise [$\\sigma$]", fontsize=6.5)
    ax_c.set_ylabel("extended model [Jy/pixel]", fontsize=6.5)
    ax_c.legend(fontsize=5.6, frameon=False, loc="lower left")

    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(labelsize=5.8, length=2.0, width=0.4, color=fs.MUTED)
        for sp in ax.spines.values():
            sp.set_color(fs.MUTED); sp.set_linewidth(0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlim(snr.max() * 1.7, snr.min() * 0.55)   # bright on the left,
                                                         # matching the panels

    for p in fs.save(fig, "snr_series"):
        print("wrote", p)

    # the numbers behind the trend panel, for the caption
    print(f"{'peak S/N':>9} {'mesh/true':>10} {'point/true':>11} {'sig':>7} "
          f"{'chi2/N':>7} {'resid':>7}")
    for k, (_, r) in enumerate(runs):
        c = (r["params"]["fit_quality"]["chi_squared"]
             / r["params"]["fit_quality"]["n_data"])
        print(f"{snr[k]:9.0f} {mesh_ratio[k]:10.3f} {pt_ratio[k]:11.3f} "
              f"{sig[k]:7.1f} {c:7.3f} {np.nanmax(np.abs(r['resid'])):6.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true",
                    help="skip fitting and re-plot from the cache")
    args = ap.parse_args()
    if not args.plot:
        fit_all()
    plot()

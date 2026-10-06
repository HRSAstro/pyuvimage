"""Does folding the primary beam into the model change the reconstruction?

    python scripts/mock_pb_in_model.py [--reg adaptive matern] [--out figures/pb_in_model]

A mock field whose emission reaches the outer primary beam: a central disc
(PB ~ 1) and three extended clumps at PB 0.5, 0.33 and 0.3. The visibilities
are of the apparent sky (truth x PB) plus noise. Each prior is fitted twice --
the default (prior on the apparent sky) and `pb_in_model=True` (prior on the
true sky) -- and the PB-corrected models are compared with the truth:
aperture fluxes per component, the error map, chi^2 and the chosen strength.

The PB is made small (FWHM 8") by giving the mock a 25 m dish at 345 GHz, so
the field reaches PB 0.2 at a 12" field of view on a 40 x 40 mesh. Only the
ratio of PB to beam matters for what is being tested.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
import numpy as np

from pyuvimage import api, mock, primary_beam
from pyuvimage.grids import resolve_geometry
from pyuvimage.uvdata import UVData

FREQ = 345e9
PB_FWHM = 8.0                                         # arcsec
DISH = primary_beam.DEFAULT_PB_FACTOR * primary_beam.C_M_S / FREQ / (
    PB_FWHM / primary_beam.RAD_TO_ARCSEC)
FOV = 12.0
MESH = 40
SIGMA = 15e-3                                         # Jy per visibility (re and im)
N_VIS = 20000

# (name, flux Jy, r_eff ", (dRA, dDec) ", axis ratio, angle)
COMPONENTS = [
    ("disc", 0.020, 1.5, (0.0, 0.0), 0.6, 30.0),
    ("clump E", 0.006, 0.8, (4.0, 1.0), 1.0, 0.0),
    ("clump N", 0.003, 1.0, (-1.0, 5.0), 0.7, 80.0),
    ("clump SW", 0.004, 0.6, (-3.5, -4.0), 1.0, 0.0),
]


def make_mock(seed=11):
    uvw = mock.random_uv_coverage(N_VIS, 400.0, FREQ, seed=seed)
    bmax = float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * FREQ / mock.C_M_S)
    geom = resolve_geometry(FOV, max_baseline_wavelengths=bmax, mesh_shape=(MESH, MESH))
    shape, pix = geom.shape_native, geom.pixel_scale
    truth = np.zeros(shape)
    parts = {}
    for name, flux, r_eff, (d_ra, d_dec), q, angle in COMPONENTS:
        img = mock.exponential_image(shape, pix, flux_jy=flux, r_eff_arcsec=r_eff,
                                     centre_arcsec=(d_dec, -d_ra), axis_ratio=q,
                                     angle_deg=angle)
        parts[name] = img
        truth += img
    pb = primary_beam.primary_beam_map(shape, pix, FREQ, DISH)
    # the DFT in chunks: one call over every visibility holds a
    # N_vis x N_pixel complex matrix (2 GB here)
    vis = np.concatenate([
        mock.simulate(truth * pb, pix, uvw[i:i + 1000], np.array([FREQ]),
                      sigma_jy=0.0).data[0]
        for i in range(0, N_VIS, 1000)
    ])
    rng = np.random.default_rng(seed + 1)
    vis = vis + rng.normal(0, SIGMA, N_VIS) + 1j * rng.normal(0, SIGMA, N_VIS)
    uvd = UVData(
        uvw=uvw, frequencies=np.array([FREQ]), data=vis[None, :],
        noise=np.full((1, N_VIS), SIGMA + 1j * SIGMA),
        meta={"telescope": "mock", "dish_diameter_m": DISH,
              "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0},
    )
    return uvd, truth, parts, pb, geom


def aperture(shape, pix, centre_sky, radius):
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x = (xx - cx) * pix
    y = (cy - yy) * pix
    d_ra, d_dec = centre_sky
    return np.hypot(x + d_ra, y - d_dec) <= radius


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", nargs="+", default=["adaptive", "matern"])
    ap.add_argument("--out", default="figures/pb_in_model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15],
                    help="noise and uv realisations (the sky is the same)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

    runs = {}            # label -> list of rows, one per seed
    first = None
    for seed in a.seeds:
        uvd, truth, parts, pb, geom = make_mock(seed)
        pix = geom.pixel_scale
        apertures = {
            name: aperture(truth.shape, pix, c, max(2.0 * r, 1.2))
            for name, _, r, c, _, _ in COMPONENTS
        }
        seen = pb >= 0.2                       # where a pbcor image is trusted
        if first is None:
            print(f"mock: PB FWHM {PB_FWHM}\" (dish {DISH:.1f} m at {FREQ/1e9:.0f} GHz), "
                  f"field {FOV}\", mesh {MESH}^2 at {geom.mesh_pixel_scale:.3f}\"")
            for name, _, _, c, _, _ in COMPONENTS:
                r = np.hypot(*c)
                print(f"  {name:<9} at {r:4.1f}\": PB "
                      f"{np.exp(-4*np.log(2)*(r/PB_FWHM)**2):.2f}")
        maps = {}
        for reg in a.reg:
            for in_model in (False, True):
                label = f"{reg}, PB {'in model' if in_model else 'after (default)'}"
                t0 = time.time()
                res = api.run(uvd, fov=FOV, mesh_shape=(MESH, MESH), reg=reg,
                              pb_in_model=in_model, write=False, uncertainty_map=True)
                p = res.products[0]
                model = np.nan_to_num(p.model_pbcor)
                unc = p.uncertainty / pb if p.uncertainty is not None else None
                row = {
                    "seed": seed,
                    "seconds": time.time() - t0,
                    "chi2_per_n": float(p.chi_squared / (2 * uvd.n_samples)),
                    "coefficient": float(p.coefficient),
                    "log_evidence": float(p.log_evidence),
                    "flux_pb_ge_0.2": float(model[seen].sum()),
                    "true_pb_ge_0.2": float(truth[seen].sum()),
                    "rms_error": float(np.sqrt(np.mean((model - truth)[seen] ** 2))),
                    "rms_error_outer": float(np.sqrt(np.mean(
                        (model - truth)[seen & (pb < 0.5)] ** 2))),
                    "components": {},
                }
                for name, m in apertures.items():
                    t, f = float(truth[m].sum()), float(model[m].sum())
                    row["components"][name] = {"true": t, "model": f, "ratio": f / t}
                runs.setdefault(label, []).append(row)
                maps[label] = (model, unc, p.residual_sigma)
                print(f"seed {seed} {label:<28} chi2/N {row['chi2_per_n']:.4f} "
                      f"coeff {row['coefficient']:.3g} logZ {row['log_evidence']:.1f} "
                      f"rms err {row['rms_error']*1e6:.2f} uJy/pix "
                      + " ".join(f"{c['ratio']:.2f}" for c in row["components"].values()),
                      flush=True)
        if first is None:
            first = (maps, truth, pb, geom)

    print("\nsummary over", len(a.seeds), "realisations: recovered / true flux "
          "(mean +/- scatter)")
    names = [c[0] for c in COMPONENTS]
    print(f"{'':<28}" + "".join(f"{n:>15}" for n in names)
          + f"{'flux PB>0.2':>15}{'rms err':>10}{'outer':>8}")
    for label, rows in runs.items():
        cells = []
        for n in names:
            r = np.array([row["components"][n]["ratio"] for row in rows])
            cells.append(f"{r.mean():8.2f}+/-{r.std():.2f}")
        tot = np.array([row["flux_pb_ge_0.2"] / row["true_pb_ge_0.2"] for row in rows])
        err = np.mean([row["rms_error"] for row in rows]) * 1e6
        out_ = np.mean([row["rms_error_outer"] for row in rows]) * 1e6
        print(f"{label:<28}" + "".join(f"{c:>15}" for c in cells)
              + f"{tot.mean():8.2f}+/-{tot.std():.2f}{err:10.2f}{out_:8.2f}")
    for reg in a.reg:
        lo = runs[f"{reg}, PB after (default)"]
        hi = runs[f"{reg}, PB in model"]
        dz = [h["log_evidence"] - l["log_evidence"] for l, h in zip(lo, hi)]
        print(f"{reg}: log evidence, PB in model minus default: "
              + ", ".join(f"{d:+.1f}" for d in dz))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(runs, indent=2))
    maps, truth, pb, geom = first
    figure({k: {"_maps": v} for k, v in maps.items()}, truth, pb, geom, out)
    print(f"\nwritten {out}.json and {out}.png")


def figure(results, truth, pb, geom, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    half = 0.5 * geom.shape_native[0] * geom.pixel_scale
    ext = [half, -half, -half, half]
    labels = list(results)
    n = len(labels) // 2
    fig, axes = plt.subplots(n, 5, figsize=(17, 3.4 * n + 0.4), squeeze=False)
    smooth = 0.6 / geom.pixel_scale        # ~ beam-scale smoothing, for display
    t_s = gaussian_filter(truth, smooth)
    vmax = float(np.percentile(t_s, 99.8))
    for i in range(n):
        lo, hi = labels[2 * i], labels[2 * i + 1]
        m_lo = gaussian_filter(results[lo]["_maps"][0], smooth)
        m_hi = gaussian_filter(results[hi]["_maps"][0], smooth)
        dmax = float(np.percentile(np.abs(m_lo - t_s), 99.5))
        panels = [
            (t_s, "truth (smoothed)", "magma", 0, vmax),
            (m_lo, lo + "\nmodel_pbcor", "magma", 0, vmax),
            (m_hi, hi + "\nmodel_pbcor", "magma", 0, vmax),
            (m_lo - t_s, "error, " + lo.split(",")[1].strip(), "RdBu_r", -dmax, dmax),
            (m_hi - t_s, "error, " + hi.split(",")[1].strip(), "RdBu_r", -dmax, dmax),
        ]
        for ax, (img, title, cmap, v0, v1) in zip(axes[i], panels):
            ax.imshow(img, origin="upper", extent=ext, cmap=cmap, vmin=v0, vmax=v1)
            ax.contour(pb, levels=[0.3, 0.5], colors="w" if cmap == "magma" else "k",
                       linewidths=0.6, linestyles=["--", ":"], extent=ext, origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("dRA [\"]")
        axes[i][0].set_ylabel("dDec [\"]")
    fig.suptitle("PB folded into the model vs applied afterwards "
                 "(contours: PB 0.5 dotted, 0.3 dashed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()

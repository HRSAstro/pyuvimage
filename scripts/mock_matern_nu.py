"""Which Matern smoothness, nu, do the data prefer?

    python scripts/mock_matern_nu.py [--out figures/matern_nu] [--seeds 1 2 3]

A mock field with three kinds of structure, built on the product grid (finer
than the model mesh, so the truth is not exactly representable):

  a smooth Gaussian blob, an exponential disc (cusped centre) and an arc of
  six compact knots (about half a beam across).

Two tests, at a high and a low signal-to-noise:

1. Evidence. For each nu, the Bayesian evidence ln Z (unconstrained solve) is
   maximised over the prior strength and the correlation length (0.25-2 beams),
   so each nu is compared at its own best settings.
2. The full default pipeline (`api.run`, positivity on, criterion auto) with
   `--reg matern` and `--reg adaptive` at each nu, scored against the truth:
   rms error after smoothing to the beam, fluxes per component, and the
   residual structure ratio.
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

from pyuvimage import api, fitting, mock
from pyuvimage import beam as beam_mod
from pyuvimage.grids import resolve_geometry
from pyuvimage.uvdata import UVData

FREQ = 230e9
FOV = 6.0
MESH = 40
N_VIS = 20000
MAX_BASELINE_M = 800.0
NOISE = {"high S/N": 5e-3, "low S/N": 25e-3}      # Jy per visibility (re and im)
NUS_EVIDENCE = [0.5, 1.0, 1.5, 2.5, 3.5]
NUS_PIPELINE = [0.5, 1.5, 2.5]
SCALE_FACTORS = [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]


def gaussian_image(shape, pix, flux, fwhm, centre_sky):
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x = (xx - cx) * pix
    y = (cy - yy) * pix
    d_ra, d_dec = centre_sky
    s = fwhm / 2.3548
    img = np.exp(-0.5 * ((x + d_ra) ** 2 + (y - d_dec) ** 2) / s**2)
    return img / img.sum() * flux


def make_mock(seed, sigma):
    uvw = mock.random_uv_coverage(N_VIS, MAX_BASELINE_M, FREQ, seed=seed)
    bmax = float(np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * FREQ / mock.C_M_S)
    geom = resolve_geometry(FOV, max_baseline_wavelengths=bmax, mesh_shape=(MESH, MESH))
    shape, pix = geom.shape_native, geom.pixel_scale
    parts = {
        "smooth blob": gaussian_image(shape, pix, 0.015, 1.5, (-1.0, 0.8)),
        "exponential disc": mock.exponential_image(
            shape, pix, flux_jy=0.015, r_eff_arcsec=0.6, centre_arcsec=(-0.5, -1.0),
            axis_ratio=0.6, angle_deg=30.0),
    }
    knots = np.zeros(shape)
    for ang in np.radians([200, 225, 250, 275, 300, 325]):
        knots += gaussian_image(shape, pix, 0.0015, 0.25,
                                (1.8 * np.cos(ang), 1.8 * np.sin(ang) + 0.3))
    parts["knot arc"] = knots
    truth = sum(parts.values())
    vis = np.concatenate([
        mock.simulate(truth, pix, uvw[i:i + 1000], np.array([FREQ]), sigma_jy=0.0).data[0]
        for i in range(0, N_VIS, 1000)
    ])
    rng = np.random.default_rng(seed + 100)
    vis = vis + rng.normal(0, sigma, N_VIS) + 1j * rng.normal(0, sigma, N_VIS)
    uvd = UVData(uvw=uvw, frequencies=np.array([FREQ]), data=vis[None, :],
                 noise=np.full((1, N_VIS), sigma + 1j * sigma),
                 meta={"telescope": "mock", "dish_diameter_m": 12.0,
                       "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0})
    return uvd, truth, parts, geom


def best_evidence(system, nu, beam_size):
    """max over (lambda, scale) of the unconstrained ln Z."""
    best = (-np.inf, None, None)
    for f in SCALE_FACTORS:
        scale = f * beam_size
        def lnz(log_c):
            reg = fitting.make_regularization("matern", 10.0 ** log_c, scale, nu)
            return system.trial(reg, positive=False).log_evidence
        grid = np.arange(-4.0, 14.01, 0.5)
        vals = np.array([lnz(g) for g in grid])
        i = int(np.nanargmax(vals))
        fine = np.arange(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)] + 1e-9, 0.1)
        fvals = np.array([lnz(g) for g in fine])
        j = int(np.nanargmax(fvals))
        if fvals[j] > best[0]:
            best = (float(fvals[j]), float(fine[j]), float(f))
    from pyuvimage.envelope import clear_covariance_cache
    clear_covariance_cache()
    return best


def smooth(img, beam_px):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, beam_px / 2.3548)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/matern_nu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip-pipeline", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

    results = {"evidence": [], "pipeline": []}
    first_maps = None
    for label, sigma in NOISE.items():
        for seed in a.seeds:
            uvd, truth, parts, geom = make_mock(seed, sigma)
            uv, d, n = uvd.flattened()
            ds = fitting.make_dataset(uv, d, n, geom)
            # the w-tilde (sparse) path, as api.run takes at this size
            ds = fitting.with_sparse_operator(ds, uv, n, geom,
                                              cache_dir="/tmp/claude-0/nu_cache")
            imager = beam_mod.DirtyImager(ds)
            b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
            beam_size = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))
            peak_snr = float(np.max(imager.dirty_image(np.asarray(ds.data))) / imager.rms)
            print(f"\n[{label}, seed {seed}] beam {beam_size:.3f}\", dirty peak S/N {peak_snr:.0f}",
                  flush=True)

            # 1. evidence over nu
            t0 = time.time()
            system = fitting.build_linear_system(ds, geom.mesh_shape)
            row = {"label": label, "seed": seed, "beam": beam_size, "peak_snr": peak_snr, "nu": {}}
            for nu in NUS_EVIDENCE:
                lnz, logc, f = best_evidence(system, nu, beam_size)
                row["nu"][str(nu)] = {"lnZ": lnz, "log_coeff": logc, "scale_beams": f}
            ref = row["nu"]["1.5"]["lnZ"]
            print("  evidence  " + "  ".join(
                f"nu={k}: {v['lnZ'] - ref:+7.1f} (l={v['scale_beams']}b)"
                for k, v in row["nu"].items()) + f"   [{time.time() - t0:.0f} s]", flush=True)
            results["evidence"].append(row)
            del system

            # 2. the default pipeline
            if a.skip_pipeline:
                continue
            beam_px = beam_size / geom.pixel_scale
            t_s = smooth(truth, beam_px)
            masks = {k: smooth(v, beam_px) > 0.1 * smooth(v, beam_px).max() for k, v in parts.items()}
            maps = {}
            for reg in ("matern", "adaptive"):
                for nu in NUS_PIPELINE:
                    t0 = time.time()
                    res = api.run(uvd, fov=FOV, mesh_shape=(MESH, MESH), reg=reg, nu=nu,
                                  write=False, uncertainty_map=False)
                    p = res.products[0]
                    m = np.asarray(p.model_image)
                    m_s = smooth(m, beam_px)
                    rec = {
                        "label": label, "seed": seed, "reg": reg, "nu": nu,
                        "chi2_per_n": float(p.chi_squared / (2 * uvd.n_samples)),
                        "coefficient": float(p.coefficient),
                        "rms_err_beam": float(np.sqrt(np.mean((m_s - t_s) ** 2)) / t_s.max()),
                        "peak_residual_sigma": float(np.max(np.abs(p.residual_sigma))),
                        "flux": {k: float(m[mk].sum() / truth[mk].sum()) for k, mk in masks.items()},
                        "seconds": time.time() - t0,
                    }
                    results["pipeline"].append(rec)
                    maps[(reg, nu)] = m
                    print(f"  {reg:<8} nu={nu}: chi2/N {rec['chi2_per_n']:.4f}  "
                          f"rms err (beam) {100 * rec['rms_err_beam']:.2f}% of peak  "
                          f"peak resid {rec['peak_residual_sigma']:.1f}σ  flux "
                          + " ".join(f"{v:.2f}" for v in rec["flux"].values())
                          + f"  [{rec['seconds']:.0f} s]", flush=True)
            if first_maps is None and label == "high S/N":
                first_maps = (truth, maps, geom)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    summarise(results)
    figure(results, first_maps, out)
    print(f"\nwritten {out}.json and {out}.png")


def summarise(results):
    print("\nEvidence relative to nu = 1.5 (mean over seeds; each nu at its best lambda and scale)")
    for label in NOISE:
        rows = [r for r in results["evidence"] if r["label"] == label]
        line = f"  {label:<9}"
        for nu in NUS_EVIDENCE:
            d = [r["nu"][str(nu)]["lnZ"] - r["nu"]["1.5"]["lnZ"] for r in rows]
            line += f"  nu={nu}: {np.mean(d):+7.1f} ± {np.std(d):4.1f}"
        print(line)
    if not results["pipeline"]:
        return
    print("\nPipeline (mean over seeds): rms error at beam resolution, % of peak | flux ratios")
    for label in NOISE:
        for reg in ("matern", "adaptive"):
            for nu in NUS_PIPELINE:
                rs = [r for r in results["pipeline"]
                      if r["label"] == label and r["reg"] == reg and r["nu"] == nu]
                if not rs:
                    continue
                err = np.mean([r["rms_err_beam"] for r in rs]) * 100
                fl = {k: np.mean([r["flux"][k] for r in rs]) for k in rs[0]["flux"]}
                print(f"  {label:<9} {reg:<8} nu={nu}: {err:5.2f}%  "
                      + "  ".join(f"{k} {v:.2f}" for k, v in fl.items()))


def figure(results, first_maps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_maps = first_maps is not None
    fig = plt.figure(figsize=(15, 8.5 if has_maps else 4))
    ax = fig.add_subplot(2 if has_maps else 1, 3, 1)
    for label, c in zip(NOISE, ("C0", "C3")):
        rows = [r for r in results["evidence"] if r["label"] == label]
        for r in rows:
            ax.plot(NUS_EVIDENCE, [r["nu"][str(nu)]["lnZ"] - r["nu"]["1.5"]["lnZ"]
                                   for nu in NUS_EVIDENCE], "o-", color=c, alpha=0.6,
                    label=label if r is rows[0] else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Matérn ν")
    ax.set_ylabel("ln Z − ln Z(ν=1.5)")
    ax.set_title("Evidence (best λ and length per ν)")
    ax.legend(fontsize=8)
    if results["pipeline"]:
        ax2 = fig.add_subplot(2 if has_maps else 1, 3, 2)
        for label, ls in zip(NOISE, ("-", "--")):
            for reg, c in (("matern", "C0"), ("adaptive", "C2")):
                ys = [np.mean([r["rms_err_beam"] for r in results["pipeline"]
                               if r["label"] == label and r["reg"] == reg and r["nu"] == nu]) * 100
                      for nu in NUS_PIPELINE]
                ax2.plot(NUS_PIPELINE, ys, "o" + ls, color=c, label=f"{reg}, {label}")
        ax2.set_xlabel("Matérn ν")
        ax2.set_ylabel("rms error at beam resolution (% of peak)")
        ax2.set_title("Default pipeline vs truth")
        ax2.legend(fontsize=8)
        ax3 = fig.add_subplot(2 if has_maps else 1, 3, 3)
        comps = list(results["pipeline"][0]["flux"])
        for k, mk in zip(comps, ("o", "s", "^")):
            for reg, c in (("matern", "C0"), ("adaptive", "C2")):
                ys = [np.mean([r["flux"][k] for r in results["pipeline"]
                               if r["label"] == "high S/N" and r["reg"] == reg and r["nu"] == nu])
                      for nu in NUS_PIPELINE]
                ax3.plot(NUS_PIPELINE, ys, mk + "-", color=c, label=f"{k} ({reg})")
        ax3.axhline(1, color="k", lw=0.5)
        ax3.set_xlabel("Matérn ν")
        ax3.set_ylabel("recovered / true flux (high S/N)")
        ax3.set_title("Component fluxes")
        ax3.legend(fontsize=7)
    if has_maps:
        truth, maps, geom = first_maps
        half = 0.5 * geom.shape_native[0] * geom.pixel_scale
        ext = [half, -half, -half, half]
        vmax = truth.max()
        panels = [("truth", truth)] + [(f"matern ν={nu}", maps[("matern", nu)]) for nu in NUS_PIPELINE]
        for k, (title, img) in enumerate(panels):
            axm = fig.add_subplot(2, 4, 5 + k)
            axm.imshow(img, origin="upper", extent=ext, cmap="magma", vmin=0, vmax=vmax)
            axm.set_title(title, fontsize=9)
            axm.set_xlabel("dRA [\"]")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()

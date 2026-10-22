"""chi^2/N and the structure ratio when the model is right, across S/N.

    python scripts/null_structure_chi2.py            # run and plot
    python scripts/null_structure_chi2.py --plot     # plot the saved results

Two single-source mocks, each fitted with its own (right) model:

  point      10 mJy at (dRA, dDec) = (0.3, 0.2)", fitted with a point
  Gaussian   10 mJy, FWHM 0.6" x 0.3", PA 30 deg at the same place, fitted
             with an elliptical Gaussian (start: round, 0.4", 0.07" off)

The visibilities are analytic (no pixelised sky), with 4000 samples on the
uv coverage of `tests/test_gaussian_components.py`, so any excess is the
statistics' own and not a mismatch in the mock. Noise per visibility from
0.1 to 256 mJy, 20 realisations each. For each fit:

  chi^2/N excess     (chi^2/N - 1) / sqrt(2/N)
  structure excess   (rho - 1) / sigma_rho, rho measured within two beams of
                     the source (`pointonly.structure_region`)

A right model should give excesses scattered about 0 with unit width, and
false alarms (> 3 sigma, where the fit warns) rarely. The same 20 noise
realisations are scaled at every level, so the levels differ only in S/N.

Result (Oct 2026): S/N 2-6300, 240 fits, no false alarm from either. chi^2/N
excess -0.35 +- 0.95 at every S/N for both models; structure ratio 0.93-0.94
(point) and 0.96-0.97 (Gaussian), excess -0.5 +- 0.7 and -0.3 +- 0.8: about
half a sigma low, and its sigma ~25% conservative, so the 3 sigma warning is
nearer 4 true sigma. Writes
figures/null_structure_chi2.{json,pdf,png}.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

SIGMAS = (0.1e-3, 1e-3, 4e-3, 16e-3, 64e-3, 256e-3)
SEEDS = 20
F0 = 230e9
FLUX = 10e-3
POS = (0.3, 0.2)                                   # dRA, dDec [arcsec]
SHAPE = dict(maj=0.6, mnr=0.3, pa=30.0)
META = {"phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0, "dish_diameter_m": 12.0}
OUT = Path("figures") / "null_structure_chi2.json"
COL = {"point": "#2c6fbb", "gaussian": "#c0392b"}


def dataset(kind, sigma, seed):
    from pyuvimage import mock, pointonly
    from pyuvimage.pointsource import point_visibilities, sky_to_grid
    from pyuvimage.uvdata import UVData

    uvw = mock.random_uv_coverage(4000, 600e3 * 299792458.0 / F0, F0, seed=5)
    uv = mock.uv_of(uvw, F0)
    vis = FLUX * point_visibilities(uv, *sky_to_grid(*POS))
    if kind == "gaussian":
        shape = (SHAPE["maj"] / pointonly.FWHM_PER_SIGMA, SHAPE["mnr"] / pointonly.FWHM_PER_SIGMA,
                 np.radians(SHAPE["pa"]))
        vis = vis * pointonly.gaussian_uv_terms(uv, shape)[0]
    rng = np.random.default_rng(1000 + seed)
    data = vis + sigma * (rng.normal(size=vis.size) + 1j * rng.normal(size=vis.size))
    return UVData(uvw=uvw, frequencies=np.array([F0]), data=data[None, :],
                  noise=np.full((1, vis.size), sigma * (1 + 1j)), meta=dict(META))


def fit(kind, u):
    from pyuvimage import api

    kw = dict(fov=4.0, points_only=True, write=False, pb_correction=False)
    if kind == "point":
        return api.run(u, point_sources=[(-POS[0], POS[1])], **kw)
    return api.run(u, gaussians=[(-POS[0] + 0.05, POS[1] + 0.05, 0.4)], **kw)


def run():
    logging.getLogger("pyuvimage").setLevel(logging.ERROR)
    rows = []
    for kind in ("point", "gaussian"):
        for sigma in SIGMAS:
            for seed in range(SEEDS):
                res = fit(kind, dataset(kind, sigma, seed))
                q = res.parameters["fit_quality"]
                c = (res.fit.gaussians or res.points)[0]
                n = q["n_data"]
                rows.append(dict(
                    kind=kind, sigma=sigma, seed=seed, snr=c.flux / c.flux_error,
                    chi2n=q["chi2_per_n"], chi2_excess=(q["chi2_per_n"] - 1) / np.sqrt(2.0 / n),
                    rho=q["structure_ratio"], rho_sigma=q["structure_ratio_noise_sigma"],
                    rho_excess=(q["structure_ratio"] - 1) / q["structure_ratio_noise_sigma"],
                    rho_field=q["structure_ratio_field"],
                    resolved=getattr(c, "resolved", None)))
            print(kind, sigma, flush=True)
    OUT.write_text(json.dumps(rows))
    return rows


def summary(rows):
    out = []
    for kind in ("point", "gaussian"):
        for sigma in SIGMAS:
            r = [x for x in rows if x["kind"] == kind and x["sigma"] == sigma]
            c = np.array([x["chi2_excess"] for x in r])
            p = np.array([x["rho_excess"] for x in r])
            res = [x["resolved"] for x in r if x["resolved"] is not None]
            out.append(dict(
                kind=kind, sigma=sigma, snr=float(np.median([x["snr"] for x in r])),
                chi2_mean=c.mean(), chi2_sd=c.std(), chi2_alarm=int(np.sum(c > 3)),
                rho=float(np.mean([x["rho"] for x in r])),
                rho_mean=p.mean(), rho_sd=p.std(), rho_alarm=int(np.sum(p > 3)),
                n=len(r), resolved=(sum(v == "yes" for v in res), len(res)) if res else None))
    return out


def plot(rows):
    import matplotlib.pyplot as plt
    import figure_style as fs

    fs.use_paper_style()
    s = summary(rows)
    fig, axs = plt.subplots(1, 2, figsize=(fs.TWO_COLUMN, 0.36 * fs.TWO_COLUMN), sharey=True)
    for ax, (key, title) in zip(axs, (("chi2", "χ²/N excess"), ("rho", "structure ratio excess"))):
        for kind, lab in (("point", "point fitted with a point"),
                          ("gaussian", "Gaussian fitted with a Gaussian")):
            ss = [x for x in s if x["kind"] == kind]
            x = np.array([v["snr"] for v in ss]) * (1.06 if kind == "gaussian" else 1 / 1.06)
            ex = f"{key}_excess"
            pts = [(v["snr"], v[ex]) for v in rows if v["kind"] == kind]
            ax.scatter([p[0] * (1.06 if kind == "gaussian" else 1 / 1.06) for p in pts],
                       [p[1] for p in pts], s=3, color=COL[kind], alpha=0.25, lw=0)
            ax.errorbar(x, [v[f"{key}_mean"] for v in ss], yerr=[v[f"{key}_sd"] for v in ss],
                        color=COL[kind], marker="o", ms=3, lw=1, capsize=2, label=lab)
        for y, ls in ((0, "-"), (3, "--"), (-3, "--")):
            ax.axhline(y, color=fs.MUTED, lw=0.6, ls=ls)
        ax.text(0.98, 3.2, "warns above 3σ", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=5.5, color=fs.MUTED)
        ax.set_xscale("log")
        ax.set_xlabel("source S/N (flux / flux error)", fontsize=6.5)
        ax.set_title(title, fontsize=7, color=fs.INK)
    axs[0].set_ylabel("excess over noise [σ of the statistic]", fontsize=6.5)
    axs[0].set_ylim(-5, 6)
    axs[0].legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for q in fs.save(fig, "null_structure_chi2"):
        print("wrote", q)
    print(f"{'model':9s} {'noise':>7s} {'S/N':>7s} | {'chi2 excess':>14s} {'>3σ':>4s} | "
          f"{'rho':>5s} {'rho excess':>14s} {'>3σ':>4s} | resolved")
    for v in s:
        print(f"{v['kind']:9s} {1e3 * v['sigma']:5g}mJy {v['snr']:7.1f} | "
              f"{v['chi2_mean']:+6.2f} ± {v['chi2_sd']:4.2f} {v['chi2_alarm']:2d}/{v['n']} | "
              f"{v['rho']:5.3f} {v['rho_mean']:+6.2f} ± {v['rho_sd']:4.2f} {v['rho_alarm']:2d}/{v['n']} | "
              f"{'' if v['resolved'] is None else '%d/%d' % v['resolved']}")


if __name__ == "__main__":
    rows = json.loads(OUT.read_text()) if "--plot" in sys.argv else run()
    plot(rows)

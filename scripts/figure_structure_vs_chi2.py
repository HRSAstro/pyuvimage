"""Residuals of a wrong and a right component model: chi^2/N vs the structure ratio.

    python scripts/figure_structure_vs_chi2.py

The mock of `tests/test_gaussian_components.py`: an elliptical Gaussian
(0.6" x 0.3", PA 30 deg, 10 mJy) and a 3 mJy point, 4000 visibilities. Two
points-only fits at each noise level:

  wrong model   two points (the Gaussian fitted as a point)
  right model   a point and an elliptical Gaussian

Columns, one row per noise level:

  1, 2  residual dirty image [sigma] of the wrong / right model, with chi^2/N and
        the structure ratio rho and their excesses over noise:
            chi^2/N:  (chi^2/N - 1) / sqrt(2/N)
            rho:      (rho - 1) / sigma_rho, measured within two beams of the
                      components (`pointonly.structure_region`)
  3     |r / sigma|^2 / 2 against uv distance in bins (1 for noise). A point's
        flux is a compromise between the short baselines, which see all of the
        Gaussian, and the long ones, which see less: too low on one end, too
        high on the other, and close to right in between. chi^2/N averages the
        misfit with every baseline and every channel of noise; the residual
        image adds it up coherently at the source, so rho stays significant
        to much lower S/N

A last panel follows the two excesses of the wrong model over a finer series
of noise levels. Every level uses the same noise realisation scaled, so the
right model's normalised residual is the same at every level.
Writes figures/structure_vs_chi2_residuals.{pdf,png}.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import figure_style as fs
from pyuvimage import api, fitting, mock, pointonly
from pyuvimage.pointsource import point_visibilities, sky_to_grid
from test_gaussian_components import F0, GAUSS, META, POINT, _sky_image

MAP_SIGMAS = (1e-3, 4e-3, 8e-3)
CURVE_SIGMAS = (1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 12e-3, 16e-3)
RESID_MAX = 5.0
N_BINS = 12
WRONG, RIGHT = "#c0392b", "#2c6fbb"


def dataset(sigma):
    uvw = mock.random_uv_coverage(4000, 600e3 * 299792458.0 / F0, F0, seed=5)
    img, pix = _sky_image()
    out = mock.simulate(img, pix, uvw, np.array([F0]), sigma_jy=sigma, seed=7, meta=dict(META))
    uv = mock.uv_of(uvw, F0)
    out.data = out.data + POINT[2] * point_visibilities(uv, *sky_to_grid(*POINT[:2]))[None, :]
    return out


def fit(uvd, right):
    kw = dict(fov=4.0, points_only=True, write=False, pb_correction=False)
    if right:
        return api.run(uvd, point_sources=[(-POINT[0], POINT[1])],
                       gaussians=[(-0.25, 0.25, 0.4)], **kw)
    return api.run(uvd, point_sources=[(-GAUSS["dra"], GAUSS["ddec"]), (-POINT[0], POINT[1])], **kw)


def scores(res):
    q = res.parameters["fit_quality"]
    n, c = q["n_data"], q["chi_squared"] / q["n_data"]
    rho, s = q["structure_ratio"], q["structure_ratio_noise_sigma"]
    return dict(chi2n=c, chi2_sig=(c - 1) / np.sqrt(2.0 / n), rho=rho, rho_sig=(rho - 1) / s)


def model_vis(res, uv):
    f = res.fit
    ys = [y for y, _ in f.grid_positions] + [yx[0] for yx, _ in f.gaussian_components]
    xs = [x for _, x in f.grid_positions] + [yx[1] for yx, _ in f.gaussian_components]
    shapes = [None] * len(f.grid_positions) + [s for _, s in f.gaussian_components]
    amp = np.concatenate([np.ravel(f.amplitudes), np.ravel(f.gaussian_amplitudes)])
    cols = pointonly._stacked_columns(uv, ys, xs, shapes) @ amp
    n = uv.shape[0]
    return cols[:n] + 1j * cols[n:]


def binned_misfit(uvd, res, edges):
    uv, d, noise = uvd.flattened()
    r = d - model_vis(res, uv)
    z = 0.5 * ((r.real / noise.real) ** 2 + (r.imag / noise.imag) ** 2)
    q = np.hypot(uv[:, 0], uv[:, 1]) / 1e3
    k = np.clip(np.digitize(q, edges) - 1, 0, len(edges) - 2)
    cnt = np.bincount(k, minlength=len(edges) - 1)
    mean = np.bincount(k, weights=z, minlength=len(edges) - 1) / np.maximum(cnt, 1)
    return mean, 1.0 / np.sqrt(np.maximum(cnt, 1))      # noise: mean of chi^2_2/2 has sd 1/sqrt(n)


def main():
    logging.getLogger("pyuvimage").setLevel(logging.ERROR)
    fs.use_paper_style()
    curve = {s: (scores(fit(dataset(s), False)), scores(fit(dataset(s), True)))
             for s in CURVE_SIGMAS}

    nr = len(MAP_SIGMAS)
    fig = plt.figure(figsize=(fs.TWO_COLUMN, 0.30 * fs.TWO_COLUMN * nr + 2.3))
    H = fig.get_figheight()
    top = fig.add_gridspec(nr, 3, left=0.07, right=0.98, top=1 - 0.45 / H, bottom=2.35 / H,
                           wspace=0.32, hspace=0.42, width_ratios=[1, 1, 1.25])
    ims = None
    for i, s in enumerate(MAP_SIGMAS):
        uvd = dataset(s)
        runs = {"wrong": fit(uvd, False), "right": fit(uvd, True)}
        n = runs["wrong"].residual.shape[0]
        pix = 4.0 / n
        ext = fs.sky_extent(n, pix)
        for j, (key, title) in enumerate((("wrong", "two points"), ("right", "point + Gaussian"))):
            res = runs[key]
            ax = fig.add_subplot(top[i, j])
            ims = fs.show_image(ax, np.flipud(res.residual) / res.rms, ext, cmap=fs.RESIDUAL_CMAP,
                                norm=fs.symmetric_norm(RESID_MAX))
            sc = scores(res)
            ax.text(0.03, 0.96, f"χ²/N {sc['chi2n']:.3f}  ({sc['chi2_sig']:+.1f}σ)\n"
                                f"ρ {sc['rho']:.2f}  ({sc['rho_sig']:+.1f}σ)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=fs.INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
            if i == 0:
                ax.set_title(f"residual: {title}", fontsize=7, color=fs.INK)
            if j == 0:
                ax.text(-0.32, 0.5, f"σ = {1e3 * s:g} mJy", rotation=90, ha="center",
                        va="center", fontsize=7, fontweight="bold", color=fs.INK,
                        transform=ax.transAxes)
                if i == nr - 1:
                    fs.sky_axes(ax, ext, ticks=(-2, -1, 0, 1, 2))
        ax = fig.add_subplot(top[i, 2])
        uv, _, _ = uvd.flattened()
        q = np.hypot(uv[:, 0], uv[:, 1]) / 1e3
        edges = np.quantile(q, np.linspace(0, 1, N_BINS + 1))
        mid = 0.5 * (edges[1:] + edges[:-1])
        for key, col, lab in (("wrong", WRONG, "two points"), ("right", RIGHT, "point + Gaussian")):
            m, e = binned_misfit(uvd, runs[key], edges)
            ax.errorbar(mid, m, yerr=e, color=col, marker="o", ms=2.2, lw=0.9, capsize=0,
                        elinewidth=0.6, label=lab)
        ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter(""))
        ax.set_ylabel("⟨|r/σ|²⟩ / 2", fontsize=6.5)
        if i == nr - 1:
            ax.set_xlabel("uv distance [kλ]", fontsize=6.5)
        else:
            ax.tick_params(labelbottom=False)
        if i == 0:
            ax.set_title("misfit by baseline length", fontsize=7, color=fs.INK)
            ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    cb = fig.add_axes([0.07, 2.0 / H, 0.55, 0.012 * 9 / H])
    fig.colorbar(ims, cax=cb, orientation="horizontal").set_label("residual [σ]", fontsize=6)
    cb.tick_params(labelsize=5.5)

    ax = fig.add_axes([0.09, 0.42 / H, 0.86, 1.2 / H])
    x = 1e3 * np.array(CURVE_SIGMAS)
    for k, lab, mk, ls in (("chi2_sig", "χ²/N excess", "s", ":"),
                           ("rho_sig", "structure ratio excess", "o", "-")):
        ax.plot(x, [max(curve[s][0][k], 0.05) for s in CURVE_SIGMAS], color=WRONG, marker=mk,
                ms=3, lw=1, ls=ls, label=f"{lab} (two points)")
    ax.axhline(3.0, color=fs.MUTED, lw=0.6, ls="--")
    ax.text(x[0], 3.3, "3σ", fontsize=5.5, color=fs.MUTED, ha="left", va="bottom")
    r = curve[CURVE_SIGMAS[0]][1]
    ax.text(0.02, 0.06, f"point + Gaussian at every level: χ²/N {r['chi2_sig']:+.1f}σ, "
                        f"ρ {r['rho_sig']:+.1f}σ", transform=ax.transAxes, fontsize=5.5,
            color=RIGHT, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 800)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel("visibility noise σ [mJy]", fontsize=6.5)
    ax.set_ylabel("excess over noise [σ]", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.xaxis.set_minor_formatter(FormatStrFormatter(""))
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    for q in fs.save(fig, "structure_vs_chi2_residuals"):
        print("wrote", q)
    for s in CURVE_SIGMAS:
        w, r = curve[s]
        print(f"{1e3 * s:5g} mJy  wrong chi2 {w['chi2_sig']:+7.1f}  rho {w['rho_sig']:+7.1f}   "
              f"right chi2 {r['chi2_sig']:+5.1f}  rho {r['rho_sig']:+5.1f}")


if __name__ == "__main__":
    main()

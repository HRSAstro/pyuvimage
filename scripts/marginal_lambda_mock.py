"""Marginalising over the prior strength: does it fix the low-S/N error bars?

    python scripts/marginal_lambda_mock.py

For a linear-Gaussian model with one hyperparameter the posterior on log(lambda)
is p(log lambda | d) ~ Z(lambda) (flat prior on log lambda), and the image
posterior marginalised over it is a mixture of the conditional Gaussians:

    mean  = sum_k w_k s_k
    var   = sum_k w_k [ sigma_k^2 + (s_k - mean)^2 ]      (law of total variance)

with w_k ~ Z(lambda_k) on a grid in log lambda. One dimension needs no sampler:
a grid is exact to its spacing and every point is one solve plus one inverse.
(A sampler such as nautilus pays off only once the kernel scale, nu or the
envelope are sampled too.)

Run on the demo S/N series (the disc without its point source, the gaussian
envelope prior) and scored against the truth on the source pixels:
coverage = median quoted 1-sigma / rms error, and the fraction of source
pixels whose error is within their 1-sigma (0.68 if calibrated). Compared with
the structure fit and its shipped window, and with the evidence optimum alone.

Unconstrained throughout: the evidence and the Gaussian posterior are exact
only without positivity.

Writes figures/bayesian_vs_statistical.{pdf,png}: per S/N, the structure
fit's model, its statistical 1-sigma and (model - truth)/sigma on the source,
then the same for the Bayesian mean and 1-sigma; and the calibration
(fraction within 1 sigma, median sigma / rms error, flux) against S/N for
statistical only, statistical + prior systematic, and Bayesian.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
logging.basicConfig(level=logging.WARNING)
import numpy as np

from pyuvimage import beam as beam_mod
from pyuvimage import envelope as envelope_mod
from pyuvimage import fitting, mock
from pyuvimage.fitting import _deblock

sys.path.insert(0, str(Path(__file__).parent))
CACHE = Path("/tmp/pyuvimage_marginal_lambda.pkl")

SIGMAS = [1.56e-3, 3.55e-3, 8.07e-3, 1.80e-2, 4.25e-2, 9.36e-2]
DEX = np.arange(-4.0, 2.01, 0.25)          # relative to the structure fit's lambda


def score(mean, std, truth, src):
    err = (mean - truth)[src]
    rms = float(np.sqrt(np.mean(err ** 2)))
    return dict(rms=rms, cover=float(np.median(std[src])) / rms,
                within=float(np.mean(np.abs(err) <= std[src])),
                flux=float(mean[src].sum() / truth[src].sum()))


def fit_all():
    rows, maps = [], []
    for sigma in SIGMAS:
        uvd, truth, geom, comps = mock.make_demo_dataset(
            n_vis=4000, mesh_n=32, fov_arcsec=3.0, sigma_jy=sigma, point_flux_jy=0.0, seed=0)
        uv, d, n = uvd.flattened()
        ds = fitting.make_dataset(uv, d, n, geom)
        imager = beam_mod.DirtyImager(ds)
        b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
        bs = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))
        centre, fwhm = envelope_mod.estimate_envelope(
            imager.dirty_image(np.asarray(ds.data)), pixel_scale=geom.pixel_scale,
            rms=imager.rms, beam_fwhm=bs, max_fwhm=geom.fov_arcsec / 2.0)
        env = {"fwhm": fwhm, "floor": 1e-2, "centre": centre}
        fit = fitting.fit_dataset(ds, geom, reg_kind="gaussian", criterion="structure",
                                  fixed_scale=bs, envelope=env, warn_on_chi2=False)
        fit.imager = imager
        sysm, H0, F = fit.system, np.asarray(fit.regularization_matrix), np.asarray(fit.system.F)
        k = fit.model_image.shape[0] // truth.shape[0]
        t_img = np.kron(np.asarray(truth), np.ones((k, k))) / k**2
        src = t_img > 0.02 * t_img.max()
        peak = float(np.max(imager.dirty_image(np.asarray(ds.data))) / imager.rms)

        ovs = geom.oversample
        # shipped: structure fit, positivity, its measured window
        total, _ = fit.model_uncertainty_total()
        shipped = score(np.asarray(fit.model_image), np.asarray(total), t_img, src)
        # the statistical term alone, at the structure fit's strength
        stat = _deblock(np.asarray(fit.model_uncertainty), ovs)
        stat_only = score(np.asarray(fit.model_image), stat, t_img, src)

        means, stds, lnz = [], [], []
        scale = fit.prior.get("scale", bs)
        for dex in DEX:
            c = 10.0 ** dex * fit.coefficient
            H = 10.0 ** dex * H0
            v = sysm.solve(H, positive=False)
            cov = np.linalg.inv(F + H)
            means.append(np.asarray(fit._image_from_values(v)))
            stds.append(np.asarray(fit._propagate(cov)))
            reg = fitting.make_regularization("gaussian", c, scale, fitting.DEFAULT_NU, env)
            lnz.append(float(sysm.trial(reg, positive=False).log_evidence))
        lnz = np.array(lnz)
        w = np.exp(lnz - lnz.max()); w /= w.sum()
        m = sum(wi * mi for wi, mi in zip(w, means))
        var = sum(wi * (si ** 2 + (mi - m) ** 2) for wi, si, mi in zip(w, stds, means))
        sd_marg = _deblock(np.sqrt(var), ovs)
        marg = score(m, sd_marg, t_img, src)
        kbest = int(np.argmax(lnz))
        evid = score(means[kbest], stds[kbest], t_img, src)
        # spread of the posterior on log10 lambda
        mu = float(np.sum(w * DEX)); sd = float(np.sqrt(np.sum(w * (DEX - mu) ** 2)))
        # share of the marginal variance that comes from lambda (between-lambda term)
        between = sum(wi * (mi - m) ** 2 for wi, mi in zip(w, means))
        frac_between = float(np.median(between[src] / var[src]))
        row = dict(peak=peak, post_mean_dex=mu, post_sd_dex=sd, frac_var_from_lambda=frac_between,
                   stat_only=stat_only, shipped=shipped, evidence=evid, marginal=marg)
        maps.append(dict(peak=peak, truth=t_img, src=src, pix=geom.pixel_scale, beam=b,
                         model=np.asarray(fit.model_image), stat=stat, total=np.asarray(total),
                         bayes=m, bayes_sd=sd_marg))
        rows.append(row)
        print(f"peak {peak:4.0f}  log10 lam posterior {mu:+.2f} +- {sd:.2f} dex (rel. structure)  "
              f"lambda share of variance {frac_between:.2f}", flush=True)
        for name, r in (("stat only", stat_only), ("shipped", shipped), ("evidence", evid), ("marginal", marg)):
            print(f"    {name:<9} rms {r['rms']:.2e}  coverage {r['cover']:.2f}  "
                  f"within-1sigma {r['within']:.2f}  flux {r['flux']:.2f}", flush=True)
    Path("figures").mkdir(exist_ok=True)
    Path("figures/marginal_lambda_mock.json").write_text(json.dumps(rows, indent=2))
    CACHE.write_bytes(pickle.dumps((rows, maps)))
    return rows, maps


def plot(rows, maps):
    """Statistical-only (structure fit) against Bayesian (evidence, marginalised over lambda)."""
    import matplotlib.pyplot as plt
    import matplotlib.text as mtext
    import figure_style as fs
    from pyuvimage.products import to_fits_orientation as fo

    fs.use_paper_style()
    n = len(maps)
    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 1.18))
    gs = fig.add_gridspec(6, n, left=0.085, right=0.90, top=0.965, bottom=0.315,
                          wspace=0.04, hspace=0.05)
    gs_tr = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.245, bottom=0.055, wspace=0.36)
    pix = maps[0]["pix"]
    ext = fs.sky_extent(maps[0]["truth"].shape[0], pix)
    vmax = max(float(mp["truth"].max()) for mp in maps) / pix**2
    norm_i = fs.asinh_norm(vmax, linear_width=0.02 * vmax)   # negatives of the unconstrained mean show as 0
    smax = max(float(np.nanpercentile(np.r_[mp["stat"].ravel(), mp["bayes_sd"].ravel()], 99.5))
               for mp in maps) / pix**2
    smin = min(float(np.nanpercentile(mp["stat"], 50)) for mp in maps) / pix**2
    norm_s = fs.asinh_norm(smax, linear_width=0.5 * smin)
    norm_z = fs.symmetric_norm(4.0)
    rows_lab = ["model\n(structure)", "statistical 1σ", "(model − truth)\n/ statistical 1σ",
                "mean\n(Bayesian)", "Bayesian 1σ", "(mean − truth)\n/ Bayesian 1σ"]
    for j, mp in enumerate(maps):
        src = mp["src"]
        panels = [(mp["model"] / pix**2, dict(norm=norm_i)),
                  (mp["stat"] / pix**2, dict(cmap=fs.UNCERTAINTY_CMAP, norm=norm_s)),
                  (np.where(src, (mp["model"] - mp["truth"]) / mp["stat"], np.nan),
                   dict(cmap=fs.RESIDUAL_CMAP, norm=norm_z)),
                  (mp["bayes"] / pix**2, dict(norm=norm_i)),
                  (mp["bayes_sd"] / pix**2, dict(cmap=fs.UNCERTAINTY_CMAP, norm=norm_s)),
                  (np.where(src, (mp["bayes"] - mp["truth"]) / mp["bayes_sd"], np.nan),
                   dict(cmap=fs.RESIDUAL_CMAP, norm=norm_z))]
        for i, (img, kw) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            ax.set_facecolor("#f0f0f0")
            im = fs.show_image(ax, fo(img), ext, **kw)
            if i in (2, 5):
                frac = float(np.nanmean(np.abs(img[src]) <= 1.0))
                fs.panel_label(ax, f"{frac:.2f}", loc="lower right", color=fs.INK, size=5.4)
            if i == 0:
                ax.set_title(f"{mp['peak']:.0f}σ", pad=2.5, color=fs.INK)
                if j == 0:
                    fs.add_beam(ax, mp["beam"], ext)
            if j == 0:
                ax.set_ylabel(rows_lab[i], fontsize=6.0, color=fs.INK, labelpad=2)
            if j == n - 1:
                fs.colorbar(fig, im, ax, ["Jy arcsec$^{-2}$", "Jy arcsec$^{-2}$", "σ"][i % 3],
                            pad=0.06, width=0.010)
    snr = np.array([r["peak"] for r in rows])
    series = [("stat_only", "statistical only (structure)", "#2a78d6", "o"),
              ("shipped", "statistical + prior systematic", "#8a8a8a", "s"),
              ("marginal", "Bayesian (evidence, λ-marginalised)", "#eb6834", "^")]
    panels = [("within", "fraction within 1σ", 0.6827),
              ("cover", "median 1σ / rms error", 1.0),
              ("flux", "recovered / true flux", 1.0)]
    for k, (key, ylab, ref) in enumerate(panels):
        ax = fig.add_subplot(gs_tr[0, k])
        ax.axhline(ref, color=fs.MUTED, lw=0.6, ls="--")
        for name, lab, c, mk in series:
            ax.plot(snr, [r[name][key] for r in rows], marker=mk, color=c, ms=3, lw=1, label=lab)
        ax.set_xscale("log")
        ax.set_xlim(snr.max() * 1.6, snr.min() * 0.6)
        ax.set_xlabel("peak S/N [σ]", fontsize=6.5)
        ax.set_ylabel(ylab, fontsize=6.5)
        ax.tick_params(labelsize=5.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.axes[-3].legend(fontsize=5.2, frameon=False, loc="lower left")
    fig.canvas.draw()
    for t in fig.findobj(mtext.Text):
        if t.get_text():
            t.set_fontsize(t.get_fontsize() + 1.0)
    for q in fs.save(fig, "bayesian_vs_statistical"):
        print("wrote", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    rows, maps = pickle.loads(CACHE.read_bytes()) if a.plot else fit_all()
    plot(rows, maps)


if __name__ == "__main__":
    main()

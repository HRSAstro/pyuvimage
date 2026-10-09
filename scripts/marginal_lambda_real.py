"""Statistical-only against Bayesian (evidence, lambda-marginalised) errors on real data.

    python scripts/marginal_lambda_real.py                       # REBELS-25 [CII] MFS
    python scripts/marginal_lambda_real.py --plot                # re-plot from the cache
    python scripts/marginal_lambda_real.py --data X.npz --fov 3 --name X

The real-data counterpart of `marginal_lambda_mock.py`. There is no truth, so
what can be compared is where the evidence puts lambda against the structure
criterion, how wide its posterior is, the error maps, and the source flux
with each kind of error bar:

  * statistical only -- sqrt(w^T C w) at the structure fit's lambda;
  * statistical + prior systematic -- what the pipeline reports
    (`source_flux` in fit_parameters.json, 1x structure window);
  * Bayesian -- the posterior marginalised over log lambda on an evidence-
    weighted grid (law of total variance), unconstrained.

The prior's shape is held at the structure fit's (for `adaptive`, its
brightness map); only its strength varies. The source region is the
pipeline's: model (x) restoring beam > 3 rms, connected to the peak.

Writes figures/bayesian_vs_statistical_<name>.{pdf,png} and a JSON beside it.
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT = "/mnt/user-data/uploads/Work/REBELS_25/Rebels25.concat.CII.npz"
DEX = np.arange(-4.0, 2.01, 0.25)          # relative to the structure fit's lambda


def fit_all(path, fov, mesh, reg, name):
    import autogalaxy as ag
    from scipy import ndimage

    from pyuvimage import api, fitting
    from pyuvimage import beam as beam_mod
    from pyuvimage.fitting import _deblock

    captured, orig = {}, fitting.fit_dataset

    def capture(*a, **kw):
        out = orig(*a, **kw)
        if kw.get("reg_kind") == reg:
            captured["fit"] = out
        return out

    fitting.fit_dataset = capture
    try:
        res = api.run(path, fov=fov, mode="mfs", reg=reg, criterion="structure",
                      mesh_shape=(mesh, mesh) if mesh else None, write=False,
                      uncertainty_map=True, pb_correction=False,
                      kernel_cache=f"/tmp/pyuvimage_marginal_{name}_kernels")
    finally:
        fitting.fit_dataset = orig
    fit, p = captured["fit"], res.products[0]
    geom = fit.geometry
    ovs, pix = geom.oversample, geom.pixel_scale
    sysm, H0, F = fit.system, np.asarray(fit.regularization_matrix), np.asarray(fit.system.F)

    # the pipeline's source region
    model = np.asarray(fit.model_image)
    smooth = beam_mod.restore(model, np.zeros_like(model), p.beam, pix)
    above = smooth > api.SOURCE_REGION_SIGMA * p.rms
    labels, _ = ndimage.label(above)
    peak = np.unravel_index(np.nanargmax(np.where(above, smooth, -np.inf)), smooth.shape)
    region = labels == labels[peak]
    # g: the region's flux as a linear function of the mesh values
    mask = fit.fit.dataset.real_space_mask
    slim_index = np.asarray(ag.Array2D(values=np.arange(int(np.sum(~np.asarray(mask)))),
                                       mask=mask).native).astype(int)
    in_region = slim_index[region & ~np.asarray(mask)]
    (obj,) = list(fit.fit.inversion.reconstruction_dict)
    M = np.asarray(obj.mapping_matrix)
    g = M[in_region].sum(axis=0)

    C0 = np.linalg.inv(F + H0)
    stat_flux = float(np.sqrt(g @ C0 @ g))
    flux_struct = float(model[region].sum())
    sf = p.source_flux or {}

    means, sds, fluxes, fvars, lnz = [], [], [], [], []
    for dex in DEX:
        H = 10.0 ** dex * H0
        v = sysm.solve(H, positive=False)
        cov = np.linalg.inv(F + H)
        means.append(np.asarray(fit._image_from_values(v)))
        sds.append(np.asarray(fit._propagate(cov)))
        fluxes.append(float(g @ v))
        fvars.append(float(g @ cov @ g))
        lnz.append(float(sysm.log_evidence(v, H)))
        print(f"  dex {dex:+5.2f}  lnZ {lnz[-1]:.6g}  flux {fluxes[-1]:.4g}", flush=True)
    lnz = np.array(lnz)
    w = np.exp(lnz - lnz.max()); w /= w.sum()
    mean = sum(wi * mi for wi, mi in zip(w, means))
    var = sum(wi * (si ** 2 + (mi - mean) ** 2) for wi, si, mi in zip(w, sds, means))
    fl = np.array(fluxes)
    f_mean = float(np.sum(w * fl))
    f_var = float(np.sum(w * (np.array(fvars) + (fl - f_mean) ** 2)))
    between = float(np.sum(w * (fl - f_mean) ** 2))
    mu = float(np.sum(w * DEX)); sd = float(np.sqrt(np.sum(w * (DEX - mu) ** 2)))
    out = {
        "name": name, "coefficient_structure": fit.coefficient,
        "post_mean_dex": mu, "post_sd_dex": sd, "lnz": lnz.tolist(), "dex": DEX.tolist(),
        "region_pixels": int(region.sum()),
        "flux_structure": flux_struct, "stat_flux_err": stat_flux,
        "pipeline_flux_err": sf.get("flux_error_jy"), "pipeline_sys_err": sf.get("flux_error_sys_jy"),
        "pipeline_window": (p.uncertainty_terms or {}).get("systematic_window_dex"),
        "flux_bayes": f_mean, "bayes_flux_err": float(np.sqrt(f_var)),
        "bayes_flux_err_from_lambda": float(np.sqrt(between)),
    }
    maps = dict(pix=pix, beam=p.beam, region=region, model=model,
                stat=_deblock(np.asarray(fit.model_uncertainty), ovs),
                total=np.asarray(p.uncertainty), bayes=mean, bayes_sd=_deblock(np.sqrt(var), ovs))
    Path(f"/tmp/pyuvimage_marginal_{name}.pkl").write_bytes(pickle.dumps((out, maps)))
    Path(f"figures/bayesian_vs_statistical_{name}.json").write_text(json.dumps(out, indent=2, default=float))
    for q in Path(".").glob("pyuvimage-*.wtilde.npy"):
        q.unlink()
    return out, maps


def plot(out, maps, name, label):
    import matplotlib.pyplot as plt
    import figure_style as fs
    from pyuvimage.products import to_fits_orientation as fo

    fs.use_paper_style()
    pix = maps["pix"]
    ext = fs.sky_extent(maps["model"].shape[0], pix)
    fig = plt.figure(figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.62))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.66, top=0.93, bottom=0.08, wspace=0.05, hspace=0.12)
    gs_r = fig.add_gridspec(2, 1, left=0.76, right=0.99, top=0.93, bottom=0.10, hspace=0.55)
    vmax = float(max(maps["model"].max(), maps["bayes"].max())) / pix**2
    norm_i = fs.asinh_norm(vmax, linear_width=0.03 * vmax)
    s_all = np.r_[maps["stat"].ravel(), maps["total"].ravel(), maps["bayes_sd"].ravel()] / pix**2
    norm_s = fs.asinh_norm(float(np.nanpercentile(s_all, 99.5)),
                           linear_width=0.5 * float(np.nanpercentile(maps["stat"], 50)) / pix**2)
    panels = [
        (0, 0, maps["model"], "model (structure)", dict(norm=norm_i)),
        (0, 1, maps["stat"], "statistical 1σ", dict(cmap=fs.UNCERTAINTY_CMAP, norm=norm_s)),
        (0, 2, maps["total"], "statistical + prior syst.", dict(cmap=fs.UNCERTAINTY_CMAP, norm=norm_s)),
        (1, 0, maps["bayes"], "mean (Bayesian)", dict(norm=norm_i)),
        (1, 1, maps["bayes_sd"], "Bayesian 1σ", dict(cmap=fs.UNCERTAINTY_CMAP, norm=norm_s)),
    ]
    for r, c, img, title, kw in panels:
        ax = fig.add_subplot(gs[r, c])
        im = fs.show_image(ax, fo(img) / pix**2, ext, **kw)
        ax.contour(fo(maps["region"]).astype(float), levels=[0.5], extent=ext, colors="w",
                   linewidths=0.5, origin="lower")
        ax.set_title(title, pad=2.5, color=fs.INK, fontsize=6.5)
        if (r, c) == (0, 0):
            fs.add_beam(ax, maps["beam"], ext)
        if (r, c) == (1, 0):
            fs.sky_axes(ax, ext, ticks=(-1.0, 0.0, 1.0))
    ax_n = fig.add_subplot(gs[1, 2]); ax_n.axis("off")
    ax_n.text(0.0, 0.95,
              f"{label}\nλ (structure) = {out['coefficient_structure']:.3g}\n"
              f"λ posterior: {out['post_mean_dex']:+.2f} ± {out['post_sd_dex']:.2f} dex\n"
              f"  (relative to structure)\n"
              f"source region: {out['region_pixels']} px (white)",
              transform=ax_n.transAxes, va="top", fontsize=5.8, color=fs.INK)
    # evidence over lambda
    ax = fig.add_subplot(gs_r[0, 0])
    lnz = np.array(out["lnz"]); dex = np.array(out["dex"])
    ax.plot(dex, lnz - lnz.max(), "o-", ms=2.5, lw=1.0, color="#eb6834")
    ax.axvline(0.0, color=fs.MUTED, lw=0.6, ls="--")
    ax.text(0.05, 0.08, "structure", transform=ax.get_xaxis_transform(), fontsize=5.5, color=fs.MUTED)
    ax.set_ylim(max(-60, float((lnz - lnz.max()).min())), 5)
    ax.set_xlabel("log₁₀ λ − log₁₀ λ$_{\\rm structure}$", fontsize=6)
    ax.set_ylabel("ln Z − max", fontsize=6)
    ax.tick_params(labelsize=5.5)
    # flux with the three error bars
    ax = fig.add_subplot(gs_r[1, 0])
    rows = [("statistical only", out["flux_structure"], out["stat_flux_err"], "#2a78d6"),
            ("stat. + prior syst.", out["flux_structure"], out["pipeline_flux_err"], "#8a8a8a"),
            ("Bayesian", out["flux_bayes"], out["bayes_flux_err"], "#eb6834")]
    for k, (lab, f, e, c) in enumerate(rows):
        if e is None:
            continue
        ax.errorbar([1e3 * f], [k], xerr=[1e3 * e], fmt="o", color=c, ms=3, capsize=2, lw=1)
        ax.text(1e3 * (f + e), k + 0.18, f" S/N {f / e:.1f}", fontsize=5.3, color=c, va="bottom")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("source flux [mJy]", fontsize=6)
    ax.tick_params(labelsize=5.5)
    for q in fs.save(fig, f"bayesian_vs_statistical_{name}"):
        print("wrote", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT)
    ap.add_argument("--fov", type=float, default=3.5)
    ap.add_argument("--mesh", type=int, default=None)
    ap.add_argument("--reg", default="adaptive")
    ap.add_argument("--name", default="rebels25")
    ap.add_argument("--label", default="REBELS-25 [CII] (MFS)")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    if a.plot:
        out, maps = pickle.loads(Path(f"/tmp/pyuvimage_marginal_{a.name}.pkl").read_bytes())
    else:
        out, maps = fit_all(a.data, a.fov, a.mesh, a.reg, a.name)
    print(json.dumps(out | {"lnz": None, "dex": None}, indent=1, default=float))
    plot(out, maps, a.name, a.label)


if __name__ == "__main__":
    main()

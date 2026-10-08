"""Flux recovery by injection: how much of a known source does each prior return?

    python scripts/mock_injection.py              # fit, then plot
    python scripts/mock_injection.py --plot       # re-plot from the cache

Measuring a component's flux inside an aperture on the model mixes in flux
that the model's finite resolution spreads from its neighbours, so the
priors mock over-reports the knots at high S/N. Injection avoids that:

  1. Add known sources to the mock visibilities (same noise realisation).
  2. Fit the injected data with the default pipeline choices (criterion
     `structure`, positivity on). This fixes the prior: coefficient, length
     and, for `adaptive`/`gibbs`, the brightness map, which therefore *does*
     contain the injected sources -- as it would for a real source.
  3. Fit the un-injected data with that same prior held fixed.
  4. The difference of the two models is the reconstruction's response to the
     injected sources alone: the rest of the field and the noise cancel
     (exactly without positivity; very nearly with it, because the prior is
     the same in both fits).

Recovered flux = sum of the difference image (total), and within an
aperture around each injected source (per position: on the blob, on the
disc, next to a knot, two in empty sky). Positions are well separated, so
apertures do not overlap.

Injections: a point source and a Gaussian two beams across, at 5, 15 and 50
times the dirty-image rms, all five positions at once. Same base mock as
`mock_matern_nu.py` / `mock_priors.py`, at high and low S/N.

Writes figures/injection_mock_{gallery,recovery}.{pdf,png} and
figures/injection_mock.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

import mock_matern_nu as mn
import figure_style as fs

PRIORS = [
    ("matern", "Matérn"),
    ("exponential", "exponential"),
    ("gaussian", "Gaussian\nenvelope"),
    ("gibbs", "Gibbs"),
    ("adaptive", "adaptive"),
]
CRITERION = "structure"
#: injection sites, sky offsets as `mn.gaussian_image` takes them
POSITIONS = {
    "on blob": (-1.0, 0.8),
    "on disc": (-0.6, -1.0),
    "by knot": (0.9, -0.6),
    "empty 1": (1.8, 1.8),
    "empty 2": (-2.0, -2.2),
}
SIZES = {"point": 0.0, "2-beam Gaussian": 2.0}     # FWHM in beams (0 = point)
SNRS = [5.0, 15.0, 50.0]                           # peak flux / dirty rms (point)
#: aperture radius [arcsec]: just under half the closest pair's separation
APERTURE_RADIUS = 0.72
CACHE = Path("/tmp/pyuvimage_mock_injection")
TEXT_BUMP_PT = 2.0


def injection_image(shape, pix, flux, fwhm_arcsec):
    img = np.zeros(shape)
    for c in POSITIONS.values():
        img += mn.gaussian_image(shape, pix, flux, max(fwhm_arcsec, 0.5 * pix), c)
    return img


def apertures(shape, pix, radius_arcsec):
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x, y = (xx - cx) * pix, (cy - yy) * pix
    # the same convention as `mn.gaussian_image`: centred at x = -d_ra
    return {k: np.hypot(x + c[0], y - c[1]) <= radius_arcsec for k, c in POSITIONS.items()}


def visibilities(img, pix, uvw):
    from pyuvimage import mock
    return np.concatenate([
        mock.simulate(img, pix, uvw[i:i + 1000], np.array([mn.FREQ]), sigma_jy=0.0).data[0]
        for i in range(0, len(uvw), 1000)
    ])


def dataset_for(uvd, geom):
    from pyuvimage import fitting
    uv, d, n = uvd.flattened()
    ds = fitting.make_dataset(uv, d, n, geom)
    return fitting.with_sparse_operator(ds, uv, n, geom, cache_dir=str(CACHE / "kernels"))


def initial_envelope(reg, ds, geom, imager, beam_size):
    """What `api.run` builds before its MFS fit (its defaults)."""
    from pyuvimage import envelope as envelope_mod
    from pyuvimage import fitting
    if reg in fitting.ADAPTIVE_REGULARIZATIONS:
        env = {"floor": 1e-2, "power": float(fitting.ADAPT_POWER)}
        if reg == "gibbs":
            env["ell_floor"] = fitting.GIBBS_ELL_FLOOR
        return env
    if reg in fitting.ENVELOPE_REGULARIZATIONS:
        centre, fwhm = envelope_mod.estimate_envelope(
            imager.dirty_image(np.asarray(ds.data)), pixel_scale=geom.pixel_scale,
            rms=imager.rms, beam_fwhm=beam_size, max_fwhm=geom.fov_arcsec / 2.0)
        return {"fwhm": fwhm, "floor": 1e-2, "centre": centre}
    return None


def fit_pair(reg, ds_inj, ds_base, geom, imager_inj, beam_size, positive=True):
    """Step 2 and 3: prior chosen on the injected data, held for the base."""
    from pyuvimage import fitting
    env = initial_envelope(reg, ds_inj, geom, imager_inj, beam_size)
    nu = 0.5 if reg == "exponential" else fitting.DEFAULT_NU
    kind = "matern" if reg == "exponential" else reg
    if reg in fitting.ADAPTIVE_REGULARIZATIONS:
        # the first pass, done here so its brightness map can be reused
        first = fitting.fit_dataset(ds_inj, geom, reg_kind="matern", criterion=CRITERION,
                                    positive_only=positive, nu=nu, fixed_scale=beam_size, warn_on_chi2=False)
        env["brightness"] = np.clip(first.model_mesh_image.ravel(), 0.0, None)
        del first
    fa = fitting.fit_dataset(ds_inj, geom, reg_kind=kind, criterion=CRITERION, nu=nu,
                             positive_only=positive,
                             fixed_scale=beam_size, envelope=env, warn_on_chi2=False)
    fb = fitting.fit_dataset(ds_base, geom, reg_kind=kind, prior=dict(fa.prior),
                             positive_only=bool(fa.positive_only), criterion=CRITERION,
                             nu=nu, fixed_scale=beam_size, envelope=env,
                             window_criterion=fa.criterion_used, warn_on_chi2=False)
    return fa, fb


def fit_all(seeds):
    from pyuvimage import beam as beam_mod
    from pyuvimage.uvdata import UVData

    CACHE.mkdir(parents=True, exist_ok=True)
    rec_path = CACHE / "records.json"
    records = json.loads(rec_path.read_text()) if rec_path.exists() else []
    done = {(r["label"], r["seed"], r["size"], r["snr"], r["reg"]) for r in records}
    gal_path = CACHE / "gallery.pkl"
    gallery = pickle.loads(gal_path.read_bytes()) if gal_path.exists() else {}

    for label, sigma in mn.NOISE.items():
        for seed in seeds:
            uvd, truth, parts, geom = mn.make_mock(seed, sigma)
            ds_base = dataset_for(uvd, geom)
            imager = beam_mod.DirtyImager(ds_base)
            b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
            beam_size = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))
            rms = float(imager.rms)                     # Jy / beam
            pix = geom.pixel_scale
            ap = apertures(geom.shape_native, pix, APERTURE_RADIUS)
            print(f"\n[{label}, seed {seed}] beam {beam_size:.3f}\", rms {1e6 * rms:.1f} uJy/beam",
                  flush=True)
            base_vis = uvd.data[0]
            for size, fwhm_beams in SIZES.items():
                for snr in SNRS:
                    flux = snr * rms                    # total flux of each injection
                    inj = injection_image(geom.shape_native, pix, flux, fwhm_beams * beam_size)
                    uvd_inj = UVData(uvw=uvd.uvw, frequencies=uvd.frequencies,
                                     data=(base_vis + visibilities(inj, pix, uvd.uvw))[None, :],
                                     noise=uvd.noise, meta=uvd.meta)
                    ds_inj = dataset_for(uvd_inj, geom)
                    imager_inj = beam_mod.DirtyImager(ds_inj)
                    for reg, _ in PRIORS:
                        key = (label, seed, size, snr, reg)
                        if key in done:
                            continue
                        t0 = time.time()
                        fa, fb = fit_pair(reg, ds_inj, ds_base, geom, imager_inj, beam_size)
                        diff = np.asarray(fa.model_image) - np.asarray(fb.model_image)
                        rec = {
                            "label": label, "seed": seed, "size": size, "snr": snr, "reg": reg,
                            "flux_injected": flux, "rms": rms, "beam": beam_size,
                            "coefficient": float(fa.prior["coefficient"]),
                            "positive_only": bool(fa.positive_only),
                            "total": float(diff.sum() / (flux * len(POSITIONS))),
                            # per position: against the injected flux inside the same aperture
                            "per_position": {k: float(diff[m].sum() / inj[m].sum())
                                             for k, m in ap.items()},
                            "outside": float(diff[~np.any(list(ap.values()), axis=0)].sum()
                                             / (flux * len(POSITIONS))),
                            "seconds": time.time() - t0,
                        }
                        records.append(rec)
                        done.add(key)
                        print(f"  {size:<15} {snr:>4.0f}σ  {reg:<11} total {rec['total']:.3f}  "
                              + "  ".join(f"{v:.2f}" for v in rec["per_position"].values())
                              + f"  [{rec['seconds']:.0f} s]", flush=True)
                        if seed == seeds[0] and snr == 15.0:
                            gallery[(label, size, reg)] = (diff, inj, beam_size, pix, b)
                        rec_path.write_text(json.dumps(records, indent=2))
                        gal_path.write_bytes(pickle.dumps(gallery))
                        for p in Path(".").glob("pyuvimage-*.wtilde.npy"):
                            p.unlink()
    return records, gallery


def summarise(records):
    print("\nRecovered / injected flux, mean over seeds. Columns: total, then per position.")
    labels = list(mn.NOISE)
    for label in labels:
        for size in SIZES:
            print(f"\n{label}, {size}")
            print(f"  {'prior':<11} {'S/N':>4}  {'total':>6}  "
                  + "  ".join(f"{k:>8}" for k in POSITIONS))
            for reg, _ in PRIORS:
                for snr in SNRS:
                    rs = [r for r in records if (r["label"], r["size"], r["snr"], r["reg"])
                          == (label, size, snr, reg)]
                    if not rs:
                        continue
                    t = np.mean([r["total"] for r in rs])
                    pp = [np.mean([r["per_position"][k] for r in rs]) for k in POSITIONS]
                    print(f"  {reg:<11} {snr:>4.0f}  {t:6.3f}  " + "  ".join(f"{v:8.3f}" for v in pp))


def _bump_text(fig):
    import matplotlib.text as mtext
    fig.canvas.draw()
    for t in fig.findobj(mtext.Text):
        if t.get_text():
            t.set_fontsize(t.get_fontsize() + TEXT_BUMP_PT)


def plot_recovery(records):
    import matplotlib.pyplot as plt
    fs.use_paper_style()
    labels, sizes = list(mn.NOISE), list(SIZES)
    groups = {"total": None, "on emission": ["on blob", "on disc", "by knot"],
              "empty sky": ["empty 1", "empty 2"]}
    fig, axes = plt.subplots(len(labels) * len(sizes), len(groups), sharex=True, sharey=True,
                             figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.95))
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(PRIORS)))
    for i, (label, size) in enumerate([(l, s) for l in labels for s in sizes]):
        for j, (gname, keys) in enumerate(groups.items()):
            ax = axes[i, j]
            ax.axhline(1.0, color=fs.FAINT, lw=0.8, zorder=0)
            for c, (reg, name) in zip(colours, PRIORS):
                ys, es = [], []
                for snr in SNRS:
                    rs = [r for r in records if (r["label"], r["size"], r["snr"], r["reg"])
                          == (label, size, snr, reg)]
                    v = [r["total"] if keys is None
                         else np.mean([r["per_position"][k] for k in keys]) for r in rs]
                    ys.append(np.mean(v) if v else np.nan)
                    es.append(np.std(v) if len(v) > 1 else 0.0)
                ax.errorbar(SNRS, ys, yerr=es, color=c, marker="o", ms=2.5, lw=0.9,
                            capsize=1.5, label=name.replace("\n", " "))
            ax.set_xscale("log")
            ax.set_xticks(SNRS)
            ax.set_xticklabels([f"{s:.0f}" for s in SNRS])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_xlim(3.8, 66.0)
            if i == 0:
                ax.set_title(gname, pad=2.5, color=fs.INK)
            if j == 0:
                ax.set_ylabel(f"{label}\n{size}\nrecovered / injected", fontsize=6.5)
            if i == len(labels) * len(sizes) - 1:
                ax.set_xlabel("injected flux / dirty rms", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.07, hspace=0.12, wspace=0.06)
    _bump_text(fig)
    for p in fs.save(fig, "injection_mock_recovery"):
        print("wrote", p)


def plot_gallery(gallery):
    """Difference images (the response to the injections), 15σ, first seed."""
    import matplotlib.pyplot as plt
    from pyuvimage.products import to_fits_orientation
    fs.use_paper_style()
    rows = [(l, s) for l in mn.NOISE for s in SIZES]
    n_r, n_c = len(rows), len(PRIORS) + 1
    W = fs.TWO_COLUMN
    left, right, gap = 0.55, 0.75, 0.05
    p = (W - left - right - (n_c - 1) * gap) / n_c
    H = 0.3 + n_r * (p + gap) + 0.45
    fig = plt.figure(figsize=(W, H))
    for i, (label, size) in enumerate(rows):
        diff0, inj, beam_size, pix, beam = gallery[(label, size, PRIORS[0][0])]
        ext = fs.sky_extent(inj.shape[0], pix)
        # scaled to the reconstructions: an injected point is one pixel and
        # would set a scale none of the models come near
        vmax = max(float(np.abs(gallery[(label, size, reg)][0]).max()) for reg, _ in PRIORS) / pix**2
        norm = fs.symmetric_norm(vmax)
        panels = [("injected", inj)] + [(name, gallery[(label, size, reg)][0]) for reg, name in PRIORS]
        for j, (name, img) in enumerate(panels):
            ax = fig.add_axes([(left + j * (p + gap)) / W, 1 - (0.3 + i * (p + gap) + p) / H,
                               p / W, p / H])
            im = fs.show_image(ax, to_fits_orientation(img) / pix**2, ext,
                               cmap=fs.RESIDUAL_CMAP, norm=norm)
            if i == 0:
                ax.set_title(name, pad=2.5, color=fs.INK)
            if j == 0:
                ax.set_ylabel(f"{label}\n{size}", color=fs.INK, fontsize=6.5, labelpad=3)
                fs.add_beam(ax, beam, ext, color=fs.INK)
            if j > 0:
                fs.panel_label(ax, f"{img.sum() / inj.sum():.2f}", loc="lower right",
                               color=fs.INK, size=5.6)
            if i == n_r - 1 and j == 0:
                fs.sky_axes(ax, ext, ticks=(-2.0, 0.0, 2.0))
                ax.set_ylabel(f"{label}\n{size}\n" + ax.get_ylabel(), color=fs.INK,
                              fontsize=6.5, labelpad=3)
        fs.colorbar(fig, im, ax, "Jy arcsec$^{-2}$", pad=0.05, width=0.012)
    _bump_text(fig)
    for p_ in fs.save(fig, "injection_mock_gallery"):
        print("wrote", p_)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--plot", action="store_true", help="re-plot from the cache only")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    if a.plot:
        records = json.loads((CACHE / "records.json").read_text())
        gallery = pickle.loads((CACHE / "gallery.pkl").read_bytes())
    else:
        records, gallery = fit_all(a.seeds)
    summarise(records)
    Path("figures/injection_mock.json").write_text(json.dumps(records, indent=2))
    plot_recovery(records)
    plot_gallery(gallery)


if __name__ == "__main__":
    main()

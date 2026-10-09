#!/usr/bin/env python
"""MacKay (1992) on the demo S/N series: gamma, the lambda posterior width, and
the strength where chi^2 = N - gamma, compared with structure's choice and the truth."""
import os, sys, json
os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
sys.path.insert(0, "/tmp/claude-0/diag")
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
from pyuvimage import mock, fitting, beam as beam_mod, envelope as envelope_mod
from pyuvimage.fitting import _structure_ratio_from_map

SIGMAS = [1.56e-3, 3.55e-3, 8.07e-3, 1.80e-2, 4.25e-2, 9.36e-2]
DEX = np.arange(-3.0, 1.01, 0.25)


def gamma_of(F, H):
    A = F + H
    return float(H.shape[0] - np.trace(np.linalg.solve(A, H)))


rows = []
for sigma in SIGMAS:
    uvd, truth, geom, comps = mock.make_demo_dataset(n_vis=4000, mesh_n=32, fov_arcsec=3.0, sigma_jy=sigma,
                                                     point_flux_jy=0.0, seed=0)
    uv, d, n = uvd.flattened()
    ds = fitting.make_dataset(uv, d, n, geom)
    imager = beam_mod.DirtyImager(ds)
    b = beam_mod.fit_beam(imager.dirty_beam, geom.pixel_scale)
    bs = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))
    centre, fwhm = envelope_mod.estimate_envelope(imager.dirty_image(np.asarray(ds.data)), pixel_scale=geom.pixel_scale,
                                                  rms=imager.rms, beam_fwhm=bs, max_fwhm=geom.fov_arcsec / 2.0)
    env = {"fwhm": fwhm, "floor": 1e-2, "centre": centre}
    fit = fitting.fit_dataset(ds, geom, reg_kind="gaussian", criterion="structure", fixed_scale=bs, envelope=env,
                              warn_on_chi2=False)
    sysm, H0 = fit.system, np.asarray(fit.regularization_matrix)
    F = np.asarray(sysm.F)
    N = sysm.n_data
    k_ = fit.model_image.shape[0] // truth.shape[0]
    t_img = np.kron(np.asarray(truth), np.ones((k_, k_))) / k_**2
    src = t_img > 0.02 * t_img.max()
    peak = float(np.max(imager.dirty_image(np.asarray(ds.data))) / imager.rms)
    curve = []
    for dex in DEX:
        H = 10.0 ** dex * H0
        v = sysm.solve(H, positive=True, warm_start=True)
        vu = sysm.solve(H, positive=False)
        g = gamma_of(F, H)
        img = np.asarray(fit._image_from_values(v))
        err = float(np.sqrt(np.mean((img - t_img)[src] ** 2)))
        flux = float(img[src].sum() / t_img[src].sum())
        chi2 = float(sysm.chi_squared(v)); chi2u = float(sysm.chi_squared(vu))
        rho = float(_structure_ratio_from_map(sysm.residual_dirty_image(v, imager), chi2, imager, N))
        lnz = float(sysm.trial(fitting.make_regularization("gaussian", 10.0 ** dex * fit.coefficient,
                                                           fit.prior.get("scale", bs), 1.5, env),
                               positive=False).log_evidence)
        curve.append(dict(dex=float(dex), gamma=g, chi2=chi2, chi2u=chi2u, rho=rho, err=err, flux=flux, lnz=lnz))
    c = {k: np.array([r[k] for r in curve]) for k in curve[0]}
    # MacKay: chi^2 = N - gamma (unconstrained chi^2, as in the derivation)
    f = c["chi2u"] - (N - c["gamma"])
    i = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    dex_mk = float(np.interp(0.0, f[i[0]:i[0] + 2], c["dex"][i[0]:i[0] + 2])) if len(i) else float("nan")
    best = int(np.argmin(c["err"]))
    i0 = int(np.argmin(abs(c["dex"])))
    imk = int(np.argmin(abs(c["dex"] - dex_mk))) if np.isfinite(dex_mk) else i0
    iev = int(np.argmax(c["lnz"]))
    g0 = c["gamma"][i0]
    row = dict(peak=peak, N=N, n_mesh=H0.shape[0], gamma=g0, sigma_dex=float(np.sqrt(2 / g0) / np.log(10)),
               dex_truth=float(c["dex"][best]), dex_mackay=dex_mk, dex_evidence=float(c["dex"][iev]),
               rho_mackay=float(c["rho"][imk]), rho_truth=float(c["rho"][best]),
               err_structure=float(c["err"][i0]), err_mackay=float(c["err"][imk]), err_truth=float(c["err"][best]),
               flux_structure=float(c["flux"][i0]), flux_mackay=float(c["flux"][imk]), flux_truth=float(c["flux"][best]),
               curve=curve)
    rows.append(row)
    print(f"peak {peak:4.0f}  gamma {g0:6.1f} of {H0.shape[0]}  sigma(log10 lam) {row['sigma_dex']:.3f}  "
          f"dex: truth {row['dex_truth']:+.2f} mackay {dex_mk:+.2f} evidence {row['dex_evidence']:+.2f}  "
          f"rho@mackay {row['rho_mackay']:.3f} rho@truth {row['rho_truth']:.3f}  "
          f"err struct/mackay/best {row['err_structure']:.2e}/{row['err_mackay']:.2e}/{row['err_truth']:.2e}  "
          f"flux {row['flux_structure']:.2f}/{row['flux_mackay']:.2f}/{row['flux_truth']:.2f}", flush=True)
json.dump(rows, open("/tmp/claude-0/diag/mackay.json", "w"), indent=1, default=float)

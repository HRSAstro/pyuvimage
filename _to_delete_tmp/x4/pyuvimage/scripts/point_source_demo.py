"""Point components vs mesh alone, on the extended + knot mock.

Produces figures/point_sources.png and prints the numbers quoted in README.
"""
import json, logging, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyuvimage import api, beam as beam_mod
from pyuvimage.mock import make_extended_plus_compact_dataset
from pyuvimage.pointsource import restore_points

logging.basicConfig(level=logging.WARNING, format="%(message)s")

FOV, MESH, NVIS = 3.0, 32, 600
TRUE_CENTRE = (0.7, 0.8)      # (dRA, dDec) arcsec
TRUE_FLUX = 0.012


def run(compact_flux, points, retune=False):
    uvd, truth, geom, comps = make_extended_plus_compact_dataset(
        n_vis=NVIS, mesh_n=MESH, compact_flux=compact_flux,
        compact_centre=(0.8, -0.7))
    res = api.run(uvd, fov=FOV, mesh_shape=(MESH, MESH), reg="gibbs",
                  point_sources=points, point_retune=retune,
                  uncertainty_map=False, write=False)
    return res, truth, geom


def report(tag, res):
    p = res.products[0]
    resid = p.residual_sigma
    print(f"{tag:24s} chi2={p.chi_squared:8.1f}  "
          f"resid peak={np.nanmax(np.abs(resid)):5.2f} sigma  "
          f"chi2/N={p.chi_squared/1200:6.3f}  "
          f"flux={np.nansum(p.model_image) + sum(q.flux for q in (p.points or [])):.6f}")
    for q in (p.points or []):
        print(f"    point {q.flux:.5f} +- {q.flux_error:.1e}  "
              f"{q.significance:5.1f} sigma  dRA {q.d_ra:+.3f} dDec {q.d_dec:+.3f}")
    return p


def _with_points(p, pixel=None):
    """Extended model with each point dropped into its nearest pixel."""
    img = np.array(p.model_image)
    n = img.shape[0]
    pix = pixel or (FOV / n)
    for q in (p.points or []):
        row = int(round((n - 1) / 2 - q.d_dec / pix))
        col = int(round((n - 1) / 2 - q.d_ra / pix))
        if 0 <= row < n and 0 <= col < n:
            img[row, col] += q.flux
    return img


if __name__ == "__main__":
    out = {}
    print("\n-- extended + 0.012 Jy knot --")
    r_mesh, truth, geom = run(0.012, False)
    p_mesh = report("mesh only", r_mesh)
    r_pt, _, _ = run(0.012, True)
    p_pt = report("mesh + point", r_pt)
    r_rt, _, _ = run(0.012, True, retune=True)
    p_rt = report("mesh + point, retuned", r_rt)

    print("\n-- extended only (control) --")
    r_ctl, _, _ = run(0.0, True)
    p_ctl = report("mesh + point", r_ctl)

    # a shared colour scale, set by the smoothest extended model, so the
    # three reconstructions can actually be compared
    vmax = float(np.nanmax(p_rt.model_image)) * 1.3
    k = (p_mesh.model_image.shape[0] // truth.shape[0]) ** 2
    truth_disp = np.kron(truth, np.ones((2, 2))) / k   # display only

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    ext = [FOV / 2, -FOV / 2, -FOV / 2, FOV / 2]
    top = [
        (truth_disp, "truth (disc + knot)"),
        (p_mesh.model_image, "model: mesh only"),
        (p_pt.model_image, "extended model: mesh + point"),
        (p_rt.model_image, "extended model: point + retune"),
    ]
    for ax, (img, title) in zip(axes[0], top):
        im = ax.imshow(img, origin="upper", extent=ext, cmap="inferno",
                       vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label="Jy/pixel")
    bottom = [
        (p_mesh.dirty_image, "dirty image", "Jy/beam"),
        (p_mesh.residual_sigma,
         f"residual: mesh only (chi2/N {p_mesh.chi_squared / 1200:.2f})", "sigma"),
        (p_pt.residual_sigma,
         f"residual: mesh + point (chi2/N {p_pt.chi_squared / 1200:.2f})", "sigma"),
        (p_rt.residual_sigma,
         f"residual: point + retune (chi2/N {p_rt.chi_squared / 1200:.2f})", "sigma"),
    ]
    for ax, (img, title, unit) in zip(axes[1], bottom):
        if unit == "sigma":
            v = np.nanmax(np.abs(img))
            im = ax.imshow(img, origin="upper", extent=ext, cmap="RdBu_r",
                           vmin=-v, vmax=v)
        else:
            im = ax.imshow(img, origin="upper", extent=ext, cmap="inferno")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label=unit)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(TRUE_CENTRE[0], TRUE_CENTRE[1], "x", color="lime", ms=9, mew=2)
        src = {2: p_pt, 3: p_rt, 6: p_pt, 7: p_rt}.get(j)
        for q in (getattr(src, "points", None) or []):
            ax.plot(q.d_ra, q.d_dec, "o", mfc="none", mec="cyan", ms=13, mew=1.6)
        ax.set_xlabel('dRA ["]'); ax.set_ylabel('dDec ["]')
    fig.suptitle(
        "Analytic point component on the extended + knot mock. Green x = true "
        "knot, cyan circle = fitted point.\nThe point's flux is not on the "
        "mesh, so panels 3-4 show the extended model alone; the knot "
        "saturates panels 1-2 on this shared scale.", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/point_sources.png", dpi=125)
    print("\nwrote figures/point_sources.png")

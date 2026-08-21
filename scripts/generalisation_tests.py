"""Does any of this generalise?

Three families of test, all scored the same way:

  1. a crowded field  -- several true point sources of different brightness
                         sitting on several extended components
  2. resolution / FOV -- the same sky sampled by different arrays
  3. SNR              -- the same sky at noise levels spanning ~2 decades

The points are injected analytically at sub-pixel positions, so nothing in
the truth is on the model's grid.  Scoring: a detection is matched to a truth
point if it lands within half a beam; anything else is a false positive.
"""
from __future__ import annotations

import json
import logging
import sys
import numpy as np

from pyuvimage import api, beam as beam_mod
from pyuvimage.mock import make_field_dataset

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("pyuvimage")


def beam_fwhm(uvd, geom):
    from pyuvimage import fitting
    uv, d, nz = uvd.flattened()
    ds = fitting.make_dataset(uv, d, nz, geom)
    b = beam_mod.fit_beam(
        beam_mod.DirtyImager(ds).dirty_beam, geom.pixel_scale)
    return float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))


def score(found, truth_points, tol):
    """Match detections to truth within `tol`; return a scoring dict."""
    unmatched = list(range(len(truth_points)))
    matched, false_pos = [], []
    for p in sorted(found, key=lambda q: -abs(q.flux)):
        best, best_d = None, np.inf
        for i in unmatched:
            tf, (tra, tdec) = truth_points[i]
            d = np.hypot(p.d_ra - tra, p.d_dec - tdec)
            if d < best_d:
                best, best_d = i, d
        if best is not None and best_d < tol:
            unmatched.remove(best)
            tf, _ = truth_points[best]
            matched.append({"truth_flux": tf, "flux": p.flux,
                            "err": p.flux_error, "sig": p.significance,
                            "dist": best_d, "ratio": p.flux / tf,
                            "pull": (p.flux - tf) / p.flux_error})
        else:
            false_pos.append({"flux": p.flux, "sig": p.significance,
                              "d_ra": p.d_ra, "d_dec": p.d_dec})
    return {"n_truth": len(truth_points), "matched": matched,
            "missed": [truth_points[i] for i in unmatched],
            "false_positives": false_pos}


def run_case(name, retune=True, **kw):
    uvd, truth, geom, comps = make_field_dataset(**kw)
    fw = beam_fwhm(uvd, geom)
    res = api.run(uvd, fov=kw.get("fov_arcsec", 4.0),
                  mesh_shape=geom.mesh_shape, reg="gibbs",
                  point_sources=True, point_retune=retune,
                  uncertainty_map=False, write=False)
    p = res.products[0]
    truth_points = [(c["flux"], c["centre"]) for c in comps["points"]]
    sc = score(p.points or [], truth_points, tol=0.5 * fw)
    n_data = 2 * uvd.data.size
    ext_flux = sum(c["flux"] for c in comps["extended"])
    pt_flux = sum(c["flux"] for c in comps["points"])
    model_flux = float(np.nansum(p.model_image)) + sum(
        q.flux for q in (p.points or []))
    # Total flux is the meaningful photometric check.  Comparing the mesh
    # against the *extended* truth alone is misleading whenever a point is
    # not detected: with a beam much coarser than the mesh pixel the mesh
    # represents an unresolved source perfectly well and simply absorbs it,
    # which is correct behaviour, not a 33% flux error.
    out = {
        "case": name, "beam": fw, "pixel": geom.pixel_scale,
        "mesh": geom.mesh_shape[0], "n_vis": uvd.data.shape[1],
        "chi2_N": p.chi_squared / n_data,
        "resid_peak": float(np.nanmax(np.abs(p.residual_sigma))),
        "resid_rms": float(np.nanstd(p.residual_sigma)),
        "total_flux_ratio": model_flux / (ext_flux + pt_flux),
        "mesh_over_data": geom.mesh_shape[0] ** 2 / float(n_data),
        "coefficient": float(p.coefficient),
        **sc,
    }
    return out


def show(r):
    print(f"\n### {r['case']}")
    print(f"  beam {r['beam']:.3f}\"  pixel {r['pixel']:.4f}\"  mesh {r['mesh']}"
          f"  nvis {r['n_vis']}  chi2/N {r['chi2_N']:.3f}"
          f"  resid peak {r['resid_peak']:.1f} sigma"
          f"  total flux x{r['total_flux_ratio']:.3f}"
          f"  pixels/data {r['mesh_over_data']:.2f}")
    for m in r["matched"]:
        print(f"    MATCH  truth {m['truth_flux']:.5f} -> {m['flux']:.5f} "
              f"(x{m['ratio']:.3f}) +-{m['err']:.1e} {m['sig']:.0f}sig "
              f"at {m['dist']*1000:.0f} mas  pull {m['pull']:+.1f}")
    for f, c in r["missed"]:
        print(f"    MISS   truth {f:.5f} at dRA {c[0]:+.2f} dDec {c[1]:+.2f}")
    for f in r["false_positives"]:
        print(f"    FALSE  {f['flux']:.5f} ({f['sig']:.0f}sig) "
              f"at dRA {f['d_ra']:+.2f} dDec {f['d_dec']:+.2f}")


# ---------------------------------------------------------------- test suites
CROWDED = dict(
    fov_arcsec=4.0, mesh_n=32, n_vis=800, sigma_jy=3e-4, seed=77,
    extended=[
        (0.040, 0.80, (0.0, 0.0), 1.0, 0.0),      # big faint disc
        (0.020, 0.25, (-1.0, -0.9), 0.6, 40.0),   # small brighter blob
        (0.008, 0.50, (1.1, 0.6), 0.8, -20.0),    # faint mid-size
    ],
    points=[
        (0.0120, (1.30, -1.20)),   # bright, isolated
        (0.0060, (-0.35, 0.55)),   # medium, on the big disc
        (0.0030, (-1.05, -0.85)),  # faint, on the bright blob
        (0.0015, (0.20, 1.45)),    # very faint, isolated
    ],
)


def suite_crowded():
    return [run_case("crowded field", **CROWDED)]


def suite_resolution():
    """The same sky seen by different arrays.

    The field of view is held wide enough to contain every component in all
    three cases -- otherwise the test measures the well-known out-of-field
    failure rather than resolution.  That case is exercised separately by
    `suite_outside_field`.  What changes is the longest baseline (so the beam)
    and the mesh, i.e. how well the array actually resolves the sky.
    """
    out = []
    base = dict(CROWDED)
    base.pop("fov_arcsec"); base.pop("mesh_n")
    for label, fov, mesh, bmax in [
        ("coarse beam (b_max 300 m)", 4.0, 20, 3.0e2),
        ("nominal (b_max 800 m)", 4.0, 32, 8.0e2),
        ("fine beam (b_max 2.5 km), small pixels", 4.0, 48, 2.5e3),
        ("wide field, same array", 8.0, 48, 8.0e2),
    ]:
        out.append(run_case(
            f"resolution: {label}", fov_arcsec=fov, mesh_n=mesh,
            max_baseline_m=bmax, **base))
    return out


def suite_outside_field():
    """A field of view that does not contain the emission.

    Known failure mode, kept as a test that it fails *loudly*: the fit cannot
    reproduce visibilities from sources it has no pixels for, chi^2/N runs
    away, and the model flux becomes meaningless.  The check is that the
    warning fires, not that the answer is good.
    """
    base = dict(CROWDED)
    base.pop("fov_arcsec"); base.pop("mesh_n")
    return [run_case("outside field: fov 2\" for a 3\" field",
                     fov_arcsec=2.0, mesh_n=32, max_baseline_m=2.5e3, **base)]


def suite_snr():
    out = []
    base = dict(CROWDED)
    base.pop("sigma_jy")
    for label, sig in [("high SNR", 3e-5), ("nominal", 3e-4),
                       ("low SNR", 1.5e-3)]:
        out.append(run_case(f"snr: {label} (sigma {sig:.0e})",
                            sigma_jy=sig, **base))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = []
    if which in ("all", "crowded"):
        results += suite_crowded()
    if which in ("all", "resolution"):
        results += suite_resolution()
    if which in ("all", "outside"):
        results += suite_outside_field()
    if which in ("all", "snr"):
        results += suite_snr()
    for r in results:
        show(r)
    with open(f"/tmp/gen_{which}.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nwrote /tmp/gen_{which}.json")

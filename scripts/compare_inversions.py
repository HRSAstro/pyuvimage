"""Do the dense and sparse inversions give the same answer?

This is the comparison that has been owed since the sparse path was written.
The earlier attempt (`wtilde_probe.py`) ran at a fixed coefficient of 1e8,
above the top of the search range, which nulls the model on **both** paths --
it compared two near-zero reconstructions, found them equal, and proved
nothing. The rules that follow from that mistake:

* Fit at a coefficient that visibly moves the model, and say what fraction of
  the dirty image the reconstruction actually reaches.
* Compare the reconstructions, not just chi^2. A pure scale error is invisible
  in chi^2 and obvious in the ratio of peaks.
* Score both against the **known truth** where there is one, not only against
  each other. Two paths can agree and both be wrong.
* Refuse to report agreement between two models that are essentially zero.

Run it on the built-in mock (truth known):

    python scripts/compare_inversions.py --mock

or on a real dataset:

    python scripts/compare_inversions.py data.npz --fov 3 \\
        --image-centre="-2.0,-2.0" --mesh 26

Both paths run in this process, one after the other, on one dataset built
once. Peak memory is reported so a dense OOM is distinguishable from a crash.
"""

from __future__ import annotations

import argparse
import logging
import resource
import time

import numpy as np

from pyuvimage import fitting, uvdata
from pyuvimage.grids import resolve_geometry
from pyuvimage.pointsource import image_to_sky, sky_to_grid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("compare")


def peak_gb() -> float:
    """Peak RSS in GB (ru_maxrss is kB on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if r > 1e8 else r / 1e6


def build(args):
    """Returns (uvdata, truth, geometry-or-None).

    The mock hands back its own geometry, and it is used as-is. Re-resolving
    it from the data gives a mesh that does not match the grid the truth was
    built on, and the truth comparison -- the only check that catches both
    paths being wrong together -- then silently drops out.
    """
    if args.mock:
        from pyuvimage.mock import make_sparse_test_dataset

        uvd, truth, geom, _ = make_sparse_test_dataset(
            n_vis=args.n_vis, point_flux_jy=args.point_flux
        )
        return uvd, truth, geom
    uvd = uvdata.read_dataset(args.dataset)
    if args.image_centre:
        x, y = (float(v) for v in args.image_centre.split(","))
        uvd = uvdata.shift_image_centre(uvd, sky_to_grid(*image_to_sky(x, y)))
    return uvd, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default=None)
    ap.add_argument("--mock", action="store_true", help="use the built-in mock")
    ap.add_argument("--n-vis", type=int, default=8000, help="mock only")
    ap.add_argument("--point-flux", type=float, default=0.0, help="mock only")
    ap.add_argument("--fov", type=float, default=3.0)
    ap.add_argument("--image-centre", default=None)
    ap.add_argument("--mesh", type=int, default=None)
    ap.add_argument(
        "--coefficient", type=float, default=None,
        help="fixed prior strength. Default: chosen from a short scan so it "
        "sits where the model actually responds -- NOT a large value, which "
        "nulls the model on both paths and makes them agree for free",
    )
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    if not args.mock and args.dataset is None:
        ap.error("give a dataset path, or --mock")

    uvd, truth, mock_geom = build(args)
    uv, d, n = uvd.flattened()
    if mock_geom is not None and args.mesh is None:
        geom = mock_geom
    else:
        b = np.hypot(uv[:, 0], uv[:, 1])
        geom = resolve_geometry(
            fov_arcsec=args.fov,
            max_baseline_wavelengths=float(b.max()),
            effective_baseline_wavelengths=float(np.percentile(b, 95)),
            mesh_shape=(args.mesh, args.mesh) if args.mesh else None,
        )
        if mock_geom is not None:
            truth = None  # a forced mesh no longer matches the truth's grid
    n_data = 2 * len(d)
    log.info(
        "%d visibilities | mesh %s | image %s | %.1f data per model pixel",
        len(d), geom.mesh_shape, geom.shape_native,
        n_data / (geom.mesh_shape[0] * geom.mesh_shape[1]),
    )

    reason = fitting.sparse_inversion_diagnosis()
    if reason is not None:
        log.error("the sparse path is unavailable here: %s", reason)
        return

    transformer_cls = fitting.resolve_transformer(
        n_vis=len(d), transformer="auto",
        n_image_pixels=int(np.prod(geom.shape_native)),
        n_mesh_pixels=geom.mesh_shape[0] * geom.mesh_shape[1],
    )

    def dataset_for(inversion):
        ds = fitting.make_dataset(
            uv, d, n, geom, transformer_cls, mask_shape="square")
        if inversion == "sparse":
            ds = fitting.with_sparse_operator(ds, uv, n, geom, cache_dir=None)
        return ds

    def fit(ds, coefficient):
        return fitting.fit_at(
            ds, geom.mesh_shape, "matern", coefficient,
            positive_only=False, reg_scale=args.scale, nu=1.5,
        )

    # ---- pick a coefficient where the model actually responds -----------
    dense_ds = dataset_for("dense")
    coefficient = args.coefficient
    if coefficient is None:
        log.info("finding a coefficient that moves the model...")
        weak = np.asarray(fit(dense_ds, 1e-6).inversion.reconstruction)
        best, best_change = 1.0, 0.0
        for trial in (1e0, 1e2, 1e4, 1e6, 1e8):
            m = np.asarray(fit(dense_ds, trial).inversion.reconstruction)
            change = np.linalg.norm(m - weak) / max(np.linalg.norm(weak), 1e-30)
            log.info("   lambda=%8.0e  moves the model by %.1f%%",
                     trial, 100 * change)
            # the sweet spot: the prior bites, but has not flattened the model
            if 0.05 < change < 0.8 and change > best_change:
                best, best_change = trial, change
        coefficient = best
        log.info("using coefficient=%.4g (model moves %.1f%% from unregularised)",
                 coefficient, 100 * best_change)

    # ---- both paths -----------------------------------------------------
    results = {}
    for inversion in ("dense", "sparse"):
        ds = dataset_for(inversion)
        t = time.time()
        f = fit(ds, coefficient)
        model = np.asarray(f.inversion.reconstruction, dtype=float)
        chi2 = fitting._chi_squared(f)
        results[inversion] = dict(
            model=model, chi2=chi2, seconds=time.time() - t,
            cls=type(f.inversion).__name__,
        )
        log.info(
            "%-7s chi2 = %.10g  chi2/N = %.6f  sum(model) = %.8g  %.1f s  "
            "peak %.2f GB  [%s]",
            inversion, chi2, chi2 / n_data, model.sum(),
            results[inversion]["seconds"], peak_gb(), results[inversion]["cls"],
        )
        del ds, f

    a, c = results["dense"]["model"], results["sparse"]["model"]
    peak = max(np.abs(a).max(), np.abs(c).max())
    if peak <= 0:
        log.error(
            "both reconstructions are zero. This is NOT agreement -- it is "
            "the failure mode that made the last comparison worthless. "
            "Lower --coefficient."
        )
        return

    rel = np.abs(a - c).max() / peak
    ratio = np.abs(a).max() / max(np.abs(c).max(), 1e-30)
    flux = a.sum() / c.sum() if c.sum() != 0 else float("inf")
    sigma_chi2 = np.sqrt(2.0 * n_data)
    dchi2 = (results["dense"]["chi2"] - results["sparse"]["chi2"]) / sigma_chi2

    log.info("")
    log.info("max|dense - sparse| / peak   = %.3e", rel)
    log.info("max|dense| / max|sparse|     = %.8f   (1 if the scales match)",
             ratio)
    log.info("sum(dense) / sum(sparse)     = %.8f", flux)
    log.info("(chi2_dense - chi2_sparse)   = %.3f sigma(chi2)", dchi2)
    log.info("speed                        = %.2fx",
             results["dense"]["seconds"] / max(results["sparse"]["seconds"], 1e-9))

    if truth is not None:
        t_flat = np.asarray(truth, dtype=float).ravel()
        if t_flat.size == a.size:
            for name in ("dense", "sparse"):
                m = results[name]["model"]
                err = np.linalg.norm(m - t_flat) / np.linalg.norm(t_flat)
                log.info("%-7s vs truth: relative error %.4f, flux %.6g "
                         "against %.6g", name, err, m.sum(), t_flat.sum())
        else:
            log.info("truth is on a %d-pixel grid and the model on %d; "
                     "skipping the truth comparison", t_flat.size, a.size)

    log.info("")
    if rel < 1e-6:
        log.info("VERDICT: the two paths agree to %.0e -- interchangeable.", rel)
    elif rel < 1e-3:
        log.info("VERDICT: agreement at the %.0e level. Good enough for "
                 "imaging; not bit-identical.", rel)
    else:
        log.info("VERDICT: they DISAGREE at the %.0e level. Do not treat "
                 "sparse as a drop-in until this is understood.", rel)


if __name__ == "__main__":
    main()

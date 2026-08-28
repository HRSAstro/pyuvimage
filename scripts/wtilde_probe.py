"""Try autoarray's w-tilde (sparse) inversion path on a real dataset.

Needs a working JAX -- the operator wrapper imports `jax.numpy`.

    python scripts/wtilde_probe.py testing_data/Ruby_200GHz/Ruby_200GHz_cont.npz \
        --fov 3 --image-centre=-2,-2 --mesh 16 --path both

What it is for
--------------
pyuvimage's inversion builds a dense `n_vis x n_mesh` transformed mapping
matrix, so memory scales with the number of visibilities. CASA's tclean never
does that: it accumulates visibilities into a fixed-size uv grid and FFTs
once, so its memory scales with the *image*, not the data.

autoarray already ships the equivalent for a regularised inversion --
`Interferometer.apply_sparse_operator()` precomputes a translation-invariant
W~ kernel on the image grid, and `InversionInterferometerSparse` assembles
`F = A^T W~ A` from sparse mapping triplets with no dense matrix anywhere.
The kernel is `(2 Ny, 2 Nx)` floats: sub-megabyte for every dataset we have,
against 1-22 GB for the mapping matrix it replaces.

Each path runs in its **own process** (`--path both` re-invokes this script).
That matters on a laptop: the dense path is the one that does not fit, JAX
holds its own arena, and running both in one process made the dense leg take
the sparse result down with it. Separate processes also make the reported
peak RSS mean something per path.

What to look for
----------------
1. `sparse` finishing where `dense` cannot -- that is the whole point;
2. with `--path both`, the two reconstructions agreeing. Nothing should be
   switched over until they do.

`F` depends only on the mesh geometry and the uv coverage, not on the
regularisation coefficient, so if this works it is built once and every
hyperparameter trial after it is a Cholesky on `n_mesh x n_mesh`.
"""

from __future__ import annotations

import argparse
import logging
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pyuvimage import beam as beam_mod, fitting, uvdata
from pyuvimage.grids import resolve_geometry
from pyuvimage.pointsource import image_to_sky, sky_to_grid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("wtilde")


def peak_gb() -> float:
    """Peak RSS in GB (ru_maxrss is kB on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if r > 1e8 else r / 1e6


def build(args):
    """The dataset and geometry, shared by both paths."""
    uvd = uvdata.read_dataset(args.dataset)
    if args.image_centre:
        x, y = (float(v) for v in args.image_centre.split(","))
        uvd = uvdata.shift_image_centre(uvd, sky_to_grid(*image_to_sky(x, y)))
    uv, d, n = uvd.flattened()
    b = np.hypot(uv[:, 0], uv[:, 1])
    geom = resolve_geometry(
        fov_arcsec=args.fov,
        max_baseline_wavelengths=float(b.max()),
        effective_baseline_wavelengths=float(np.percentile(b, 95)),
        mesh_shape=(args.mesh, args.mesh) if args.mesh else None,
    )
    return uv, d, n, geom


def run_one(args) -> None:
    """One path, in its own process, writing the reconstruction to --save."""
    uv, d, n, geom = build(args)
    n_mesh = geom.mesh_shape[0] * geom.mesh_shape[1]
    have = fitting.available_memory_gb()
    log.info(
        "%s | %d visibilities | mesh %s | image grid %s | %.1f GB available",
        args.path, len(d), geom.mesh_shape, geom.shape_native, have or float("nan"),
    )
    ds = fitting.make_dataset(uv, d, n, geom, "auto", mask_shape="square")

    if args.path == "sparse":
        t = time.time()
        kernel = np.asarray(ds.psf_precision_operator_from(chunk_k=args.chunk_k))
        log.info(
            "  W~ kernel %s, %.2f MB, built in %.1f s  <-- the only "
            "n_vis-dependent cost, and cacheable to disk",
            kernel.shape, kernel.nbytes / 1e6, time.time() - t,
        )
        t = time.time()
        ds = ds.apply_sparse_operator(
            nufft_precision_operator=kernel, batch_size=args.batch_size
        )
        log.info("  sparse operator attached in %.1f s", time.time() - t)
    else:
        log.info(
            "  the dense mapping matrix wants ~%.1f GB",
            fitting.estimate_peak_memory_gb(len(d), n_mesh),
        )

    t = time.time()
    fit = fitting.fit_at(
        ds, geom.mesh_shape, "matern", args.coefficient,
        positive_only=False, reg_scale=args.scale, nu=1.5,
    )
    rec = np.asarray(fit.inversion.reconstruction)
    # The dirty image's peak is the scale the reconstruction has to live on.
    # Print the ratio: a model that is right in shape and wrong in amplitude
    # -- the failure mode that cost a day -- is invisible in chi^2 and
    # obvious here.
    dirty = np.asarray(beam_mod.DirtyImager(ds).dirty_image)
    log.info(
        "  %-7s chi2 = %.8g   sum(recon) = %.8g   max(recon)/max(dirty) = "
        "%.3g   %.1f s   peak %.2f GB   [%s]",
        args.path, fitting._chi_squared(fit), rec.sum(),
        np.abs(rec).max() / max(np.abs(dirty).max(), 1e-30),
        time.time() - t, peak_gb(), type(fit.inversion).__name__,
    )
    if np.abs(rec).max() < 1e-4 * np.abs(dirty).max():
        log.warning(
            "  the reconstruction is negligible next to the dirty image. "
            "Either --coefficient is far too large, or the data vector and "
            "the curvature matrix are on different scales. Comparing two "
            "nulled models proves nothing -- lower --coefficient and re-run."
        )
    if args.save:
        np.save(args.save, rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--fov", type=float, required=True)
    ap.add_argument("--image-centre", default=None, help='image "x,y" in arcsec')
    ap.add_argument("--mesh", type=int, default=None)
    ap.add_argument(
        "--coefficient", type=float, default=1e2,
        help="regularisation strength. NOT 1e8: a coefficient far above the "
        "search range nulls the model on BOTH paths, and two near-zero "
        "reconstructions agree to any precision you care to measure. That "
        "false pass is what this script reported on 2026-08-28.",
    )
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--chunk-k", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--path", choices=["sparse", "dense", "both"], default="sparse",
        help="'both' re-invokes this script once per path, in separate "
        "processes, and compares the two reconstructions",
    )
    ap.add_argument("--save", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.path != "both":
        run_one(args)
        return

    out = {}
    for path in ("sparse", "dense"):
        save = Path(f".wtilde_probe_{path}.npy")
        cmd = [sys.executable, __file__, args.dataset, "--fov", str(args.fov),
               "--coefficient", str(args.coefficient), "--scale", str(args.scale),
               "--chunk-k", str(args.chunk_k), "--batch-size", str(args.batch_size),
               "--path", path, "--save", str(save)]
        if args.image_centre:
            cmd.append(f"--image-centre={args.image_centre}")
        if args.mesh:
            cmd += ["--mesh", str(args.mesh)]
        rc = subprocess.call(cmd)
        if rc != 0:
            log.warning(
                "  the %s path exited with %d%s", path, rc,
                " (killed -- out of memory)" if rc in (-9, 137) else "",
            )
        elif save.exists():
            out[path] = np.load(save)
            save.unlink()

    if len(out) == 2:
        a, c = out["dense"], out["sparse"]
        peak = max(np.max(np.abs(a)), np.max(np.abs(c)))
        rel = np.max(np.abs(a - c)) / max(peak, 1e-30)
        # A pure scale error -- the way this actually failed -- shows up here
        # and nowhere in the difference norm.
        ratio = (
            np.max(np.abs(a)) / np.max(np.abs(c))
            if np.max(np.abs(c)) > 0 else float("inf")
        )
        log.info("max |dense - sparse| / peak = %.3g", rel)
        log.info("max|dense| / max|sparse|     = %.6g  (1.0 if scales match)",
                 ratio)
        if peak <= 0 or not np.isfinite(ratio):
            log.warning(
                "at least one reconstruction is identically zero. This is NOT "
                "agreement -- it is two null models. Lower --coefficient."
            )
        elif rel < 1e-3:
            log.info("the two agree: the sparse path is a drop-in")
        else:
            log.info("they DISAGREE -- do not switch until this is understood")
    else:
        log.info(
            "only %s completed, so there is nothing to compare. If it was "
            "`dense` that died, that is the result: try a smaller --mesh for "
            "a like-for-like check.", " and ".join(out) or "neither path",
        )


if __name__ == "__main__":
    main()

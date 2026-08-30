"""Why is chi^2 independent of the regularisation coefficient?

Answers, in order, the questions that separate the candidate causes:

  1. Which transformer did `auto` actually resolve? (`fit_parameters.json`
     records the *request*, "auto", not the answer.)
  2. Does the sparse operator's stored `dirty_image` match the **scaled** or
     the **unscaled** adjoint? This is the whole ballgame: the operator's data
     vector is `L^T dirty_image`, and `W~` is accumulated straight from
     1/sigma^2, so the two are only on a common scale if
     `use_adjoint_scaling=True` reached the transformer.
  3. Which inversion class does the framework actually build?
  4. Does chi^2 move with the coefficient at all, and is the reconstruction
     a sane size next to the dirty image?

Run from the repository root:

    python scripts/sparse_diagnose.py testing_data/Ruby_200GHz/Ruby_200GHz_cont.npz \
        --fov 3 --image-centre="-2.0,-2.0" --mesh 26
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from pyuvimage import beam as beam_mod, fitting, uvdata
from pyuvimage.grids import resolve_geometry
from pyuvimage.pointsource import image_to_sky, sky_to_grid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diagnose")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--fov", type=float, required=True)
    ap.add_argument("--image-centre", default=None)
    ap.add_argument("--mesh", type=int, default=None)
    ap.add_argument("--chunk-k", type=int, default=4096)
    args = ap.parse_args()

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
    log.info(
        "%d visibilities | mesh %s | image grid %s",
        len(d), geom.mesh_shape, geom.shape_native,
    )

    # ---- 0. what this autoarray provides --------------------------------
    import autoarray
    from autoarray.inversion.inversion.interferometer import (
        inversion_interferometer_util as iiu,
        sparse as isparse,
    )
    from autoarray.dataset.interferometer.dataset import Interferometer

    log.info("0. autoarray %s", getattr(autoarray, "__version__", "unknown"))
    op_cls = iiu.InterferometerSparseOperator
    for name in ("curvature_matrix_diag_from",
                 "curvature_matrix_off_diag_from",
                 "curvature_matrix_off_diag_func_list_from"):
        log.info("   InterferometerSparseOperator.%-42s %s", name,
                 "yes" if hasattr(op_cls, name) else "NO")
    src = __import__("inspect").getsource(isparse)
    log.info("   interferometer/sparse.py dispatches on func lists:   %s",
             "yes" if "AbstractLinearObjFuncList" in src else "NO")
    passes = "use_adjoint_scaling=True" in __import__("inspect").getsource(
        Interferometer.apply_sparse_operator
    )
    log.info("   apply_sparse_operator passes use_adjoint_scaling:    %s",
             "yes" if passes else "NO (pyuvimage repairs this)")

    # ---- 1. which transformer -------------------------------------------
    cls = fitting.resolve_transformer(
        n_vis=len(d), transformer="auto",
        n_image_pixels=int(np.prod(geom.shape_native)),
        n_mesh_pixels=geom.mesh_shape[0] * geom.mesh_shape[1],
    )
    log.info("1. transformer resolved by `auto`: %s", cls.__name__)

    ds = fitting.make_dataset(uv, d, n, geom, cls, mask_shape="square")
    tr = ds.transformer
    log.info("   dataset.transformer is:          %s", type(tr).__name__)
    log.info(
        "   honours use_adjoint_scaling:     %s",
        "use_adjoint_scaling" in __import__("inspect")
        .signature(tr.image_from).parameters,
    )

    # ---- 2. the operator's dirty image, against both references ---------
    weighted = (
        np.asarray(ds.data).real * np.asarray(ds.noise_map).real ** -2.0
        + 1j * np.asarray(ds.data).imag * np.asarray(ds.noise_map).imag ** -2.0
    )
    from autoarray.structures.visibilities import Visibilities

    v = Visibilities(visibilities=weighted)
    unscaled = np.abs(np.asarray(tr.image_from(visibilities=v).native)).max()
    scaled = np.abs(
        np.asarray(tr.image_from(visibilities=v, use_adjoint_scaling=True).native)
    ).max()
    log.info(
        "2. adjoint peaks: unscaled %.6e | scaled %.6e | ratio %.1f",
        unscaled, scaled, scaled / max(unscaled, 1e-300),
    )

    # Build the kernel once, then attach it the way autoarray does -- raw --
    # so we can see what it hands back before pyuvimage touches it.
    kernel = np.asarray(ds.psf_precision_operator_from(chunk_k=args.chunk_k))
    raw = ds.apply_sparse_operator(
        nufft_precision_operator=kernel, batch_size=fitting.SPARSE_BATCH_SIZE
    )
    op = np.abs(np.asarray(raw.sparse_operator.dirty_image)).max()
    log.info("   autoarray's operator dirty image peak: %.6e", op)
    which = (
        "SCALED (correct)" if abs(op - scaled) < 0.01 * scaled
        else "UNSCALED (autoarray is not passing use_adjoint_scaling)"
        if abs(op - unscaled) < 0.01 * unscaled
        else "NEITHER -- something else is going on"
    )
    log.info("   => %s", which)

    # ---- 3. pyuvimage's repair pass -------------------------------------
    ds = fitting.repair_sparse_dirty_image(raw, ds)
    fixed = np.abs(np.asarray(ds.sparse_operator.dirty_image)).max()
    log.info(
        "3. after pyuvimage's repair: %.6e (%s)", fixed,
        "on scale" if abs(fixed - scaled) < 0.01 * scaled else "STILL WRONG",
    )

    # ---- 4. the inversion, and whether lambda does anything ------------
    log.info("4. chi^2 against the coefficient:")
    dirty_peak = np.abs(
        np.asarray(beam_mod.DirtyImager(ds).dirty_image(np.asarray(ds.data)))
    ).max()
    for coefficient in (1e-2, 1e2, 1e4, 1e6):
        fit = fitting.fit_at(
            ds, geom.mesh_shape, "matern", coefficient,
            positive_only=False, reg_scale=0.5, nu=1.5,
        )
        rec = np.asarray(fit.inversion.reconstruction)
        log.info(
            "   lambda=%8.0e  chi2 = %.10g   max|recon|/max|dirty| = %.4g   [%s]",
            coefficient, fitting._chi_squared(fit),
            np.abs(rec).max() / max(dirty_peak, 1e-300),
            type(fit.inversion).__name__,
        )


if __name__ == "__main__":
    main()

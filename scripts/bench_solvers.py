"""Benchmark: dense vs sparse (w-tilde) vs matrix-free -- precision and time.

One mock (the demo disc, 4000 visibilities), fitted at mesh sizes m with the
field growing at a fixed mesh pixel (`fov_for`), one process per (m, stage)
so memory and timings are clean. Two stages:

  solve   one fixed-strength unconstrained solve per path: the build time of
          F (dense: A^T W A from the transformed mapping matrix; sparse: from
          the w-tilde kernel; matrix-free: no F, only operators), the solve,
          the evidence, and the agreement of F, D, s and ln Z between paths.
  fit     the whole default pipeline (`api.run`, adaptive prior, structure
          criterion, positivity where the path has it), timed, and the
          delivered images compared with the dense path's.

    python scripts/bench_solvers.py solve 32      # one row, appended to the json
    python scripts/bench_solvers.py all           # every row, then plot
    python scripts/bench_solvers.py plot
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
import numpy as np

OUT = Path("figures/bench_solvers.json")
N_VIS, FOV, LAM, SCALE, NU = int(os.environ.get("BENCH_NVIS", 4000)), 3.0, 1e8, 0.3, 1.5
DENSE_MAX, SPARSE_MAX = 64, 80            # beyond these the n_mesh^2 paths do not fit here
SOLVE_MESHES = [16, 24, 32, 48, 64, 80, 96, 128, 192, 256]
FIT_MESHES = [16, 24, 32, 48, 64, 96, 128, 192]
FIT_DENSE_MAX, FIT_SPARSE_MAX = 32, 64


#: the sky is simulated once, on this grid, and fitted at every mesh size:
#: simulating at each m would make the data differ between rows (and the
#: 192-pixel mock's own DFT alone needs 2.4 GB)
TRUTH_MESH = 48


def fov_for(m: int) -> float:
    """The field grows with the mesh at a fixed mesh pixel (FOV / TRUTH_MESH,
    the 48-pixel grid's): the realistic way a mesh gets large -- a wider
    field at the same resolution, as NGC 4274 at --fov 35. Shrinking the
    pixel instead over-resolves the beam and measures something else."""
    return m * FOV / TRUTH_MESH


def _mock(m):
    """(data, truth on the fitted image grid, geometry for an m x m mesh)."""
    from pyuvimage import mock
    from pyuvimage.grids import resolve_geometry

    uvd, truth, g0, _ = mock.make_demo_dataset(n_vis=N_VIS, mesh_n=TRUTH_MESH,
                                               fov_arcsec=FOV, seed=0)
    geom = resolve_geometry(fov_for(m), max_baseline_wavelengths=uvd.max_baseline_wavelengths,
                            mesh_shape=(m, m))
    # the truth was the exponential disc sampled on the 48-pixel grid; on the
    # fitted grid, the same disc sampled at that grid's pixels (flux per pixel)
    t_img = mock.exponential_image(geom.shape_native, geom.pixel_scale, flux_jy=0.05,
                                   r_eff_arcsec=FOV / 8)
    return uvd, t_img, geom


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def stage_solve(m: int) -> dict:
    import logging
    logging.disable(logging.WARNING)
    import autogalaxy as ag
    from scipy.linalg import cho_factor, cho_solve
    from pyuvimage import fitting, matfree, streaming as stm

    uvd, truth, geom = _mock(m)
    uv, d, n = uvd.flattened()
    mask = fitting.make_mask(geom, "square")
    row = {"stage": "solve", "mesh": m, "n_mesh": m * m, "n_vis": int(len(d))}

    def timed(f):
        t = time.perf_counter(); out = f(); return out, time.perf_counter() - t

    # the w-tilde terms: shared by the sparse and matrix-free paths
    # the per-chunk DFT is n_image x chunk complex: keep it under ~0.5 GB
    n_img = int(np.prod(geom.shape_native))
    chunk = int(max(64, min(4096, 0.5e9 / (16 * n_img))))
    terms, row["t_terms"] = timed(lambda: stm.sparse_terms_for(
        uvd, geom, mask, ag.TransformerDFT, pool_noise=True, chunk_k=chunk))
    stub = stm.stub_dataset_from_terms(terms, geom, mask)
    F_d = F_s = None
    if m <= DENSE_MAX:
        ds = fitting.make_dataset(uv, d, n, geom, transformer="dft")
        fit_d = fitting.fit_at(ds, geom.mesh_shape, "matern", LAM, reg_scale=SCALE, nu=NU,
                               positive_only=False)
        sys_d, row["t_build_dense"] = timed(lambda: fitting.system_for(fit_d))
        F_d, D_d = np.asarray(sys_d.F), np.asarray(sys_d.D)
        H = np.asarray(fit_d.inversion.regularization_matrix)
    if m <= SPARSE_MAX:
        fit_s = fitting.fit_at(stub, geom.mesh_shape, "matern", LAM, reg_scale=SCALE, nu=NU,
                               positive_only=False)
        sys_s, row["t_build_sparse"] = timed(lambda: fitting.system_for(fit_s))
        F_s, D_s = np.asarray(sys_s.F), np.asarray(sys_s.D)
        if F_d is None:
            H = np.asarray(fit_s.inversion.regularization_matrix)
    mapper = fitting.fit_at(stub, geom.mesh_shape, "constant", 1.0,
                            positive_only=False).inversion.linear_obj_list[0]

    def chol_solve(F, D, Hm):
        c = cho_factor(F + Hm, lower=True)
        s = cho_solve(c, D)
        return s, 2 * np.sum(np.log(np.diag(c[0])))

    if F_d is not None:
        (s_d, ld_d), row["t_solve_dense"] = timed(lambda: chol_solve(F_d, D_d, H))
    if F_s is not None:
        (s_s, ld_s), row["t_solve_sparse"] = timed(lambda: chol_solve(F_s, D_s, H))
    if F_d is not None and F_s is not None:
        row["F_rel_diff"] = float(np.abs(F_s - F_d).max() / np.abs(F_d).max())
        row["D_rel_diff"] = float(np.abs(D_s - D_d).max() / np.abs(D_d).max())
        row["s_rel_diff_sparse_dense"] = float(np.abs(s_s - s_d).max() / np.abs(s_d).max())
        row["logdet_diff_sparse_dense"] = float(ld_s - ld_d)

    # matrix-free: operators, SPDE prior, CG, evidence
    def setup():
        data = matfree.DataOperator(terms, matfree.sparse_mapping(mapper),
                                    ~np.asarray(mask.native, bool))
        order, shp, pix = matfree.mesh_grid_order(np.asarray(mapper.source_plane_mesh_grid.array))
        prior = matfree.SparseMaternPrior(order, shp, pix, SCALE)
        return data, prior, matfree.MatrixFreeSystem(data, prior)
    (data, prior, mfs), row["t_build_mf"] = timed(setup)
    sol, row["t_solve_mf"] = timed(lambda: mfs.solve(LAM))
    row["cg_matvecs"] = sol.iterations
    lnz_mf, row["t_evidence_mf"] = timed(lambda: mfs.log_evidence(sol))
    row["lnZ_mf"] = lnz_mf
    if F_s is not None:
        # the same SPDE system, dense: what the matrix-free answer should equal
        Hs = LAM * prior.Qw.toarray()
        (s_ref, ld_ref), row["t_solve_dense_spde"] = timed(lambda: chol_solve(F_s, D_s, Hs))
        ld_H = np.linalg.slogdet(Hs)[1]
        lnz_ref = -0.5 * (data.chi_squared(s_ref) + s_ref @ Hs @ s_ref + ld_ref - ld_H
                          + data.noise_normalization)
        row["s_rel_diff_mf"] = float(np.abs(sol.s - s_ref).max() / np.abs(s_ref).max())
        row["lnZ_rel_diff_mf"] = float((lnz_mf - lnz_ref) / abs(lnz_ref))
        row["lnZ_ref"] = float(lnz_ref)
    row["peak_rss_gb"] = _rss_gb()
    return row


def stage_fit(m: int, path: str) -> dict:
    import logging
    logging.disable(logging.INFO)
    from pyuvimage import api, matfree

    uvd, t_img, geom = _mock(m)
    kw = dict(fov=fov_for(m), mesh_shape=(m, m), reg="adaptive", criterion="structure",
              pb_correction=False, write=True, out=f"/tmp/bench_fit/{path}_{m}")
    n_img = (2 * m) ** 2
    if path != "dense":
        kw["chunk_k"] = int(max(64, min(4096, 0.5e9 / (16 * n_img))))
    if path == "dense":
        kw.update(inversion="dense", streaming=False, transformer="dft")
    elif path == "sparse":
        kw.update(inversion="sparse", streaming=True)
        matfree.MATRIX_FREE_MIN_MESH_PIXELS = 10 ** 9
    else:
        kw.update(inversion="sparse", streaming=True)
        matfree.MATRIX_FREE_MIN_MESH_PIXELS = 0
    t = time.perf_counter()
    res = api.run(uvd, **kw)
    p = res.products[0]
    np.save(f"/tmp/bench_fit/{path}_{m}/model.npy", p.model_image)
    src = t_img > 0.02 * t_img.max()
    return {"stage": "fit", "mesh": m, "path": path, "seconds": time.perf_counter() - t,
            "coefficient": float(p.coefficient), "chi2_per_datum": float(p.chi_squared) / (2 * uvd.n_samples),
            "flux_ratio": float(np.nansum(p.model_image) / np.sum(t_img)),
            "rms_err_src": float(np.sqrt(np.mean((p.model_image - t_img)[src] ** 2)) / t_img.max()),
            "peak_rss_gb": _rss_gb()}


def run_all():
    OUT.parent.mkdir(exist_ok=True)
    rows = []
    jobs = [("solve", m, None) for m in SOLVE_MESHES]
    for m in FIT_MESHES:
        for path in ("dense", "sparse", "mf"):
            if (path == "dense" and m > FIT_DENSE_MAX) or (path == "sparse" and m > FIT_SPARSE_MAX):
                continue
            jobs.append(("fit", m, path))
    for stage, m, path in jobs:
        args = [sys.executable, __file__, stage, str(m)] + ([path] if path else [])
        t = time.time()
        r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        line = [l for l in r.stdout.splitlines() if l.startswith("ROW ")]
        if line:
            rows.append(json.loads(line[-1][4:]))
            print(stage, m, path or "", f"{time.time() - t:.0f}s", flush=True)
        else:
            print(stage, m, path or "", "FAILED", r.stderr[-400:], flush=True)
        OUT.write_text(json.dumps(rows, indent=1))
    return rows

MF_SOLVE_KEYS = ("t_build_mf", "t_solve_mf", "cg_matvecs", "t_evidence_mf", "lnZ_mf",
                 "s_rel_diff_mf", "lnZ_rel_diff_mf")
MF_FIT_KEYS = ("seconds", "coefficient", "chi2_per_datum", "flux_ratio", "rms_err_src",
               "peak_rss_gb")


def rerun_mf():
    """Re-run the matrix-free rows after a solver change, keeping the
    previous values beside them (suffix _sparseP: Powell's sparse
    preconditioner for every prior, before the radial-profile one)."""
    rows = json.loads(OUT.read_text())
    for r in rows:
        if r["stage"] == "solve":
            args = [sys.executable, __file__, "solve", str(r["mesh"])]
            keys = MF_SOLVE_KEYS
        elif r["path"] == "mf":
            args = [sys.executable, __file__, "fit", str(r["mesh"]), "mf"]
            keys = MF_FIT_KEYS
        else:
            continue
        t = time.time()
        out = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        line = [l for l in out.stdout.splitlines() if l.startswith("ROW ")]
        if not line:
            print(r["stage"], r["mesh"], "FAILED", out.stderr[-400:], flush=True)
            continue
        new = json.loads(line[-1][4:])
        for k in keys:
            if k in r and k + "_sparseP" not in r:
                r[k + "_sparseP"] = r[k]
            if k in new:
                r[k] = new[k]
        print(r["stage"], r["mesh"], f"{time.time() - t:.0f}s", flush=True)
        OUT.write_text(json.dumps(rows, indent=1))
    return rows


COLORS = {"dense": "#2a78d6", "sparse": "#eb6834", "mf": "#1baf7a"}
LABELS = {"dense": "dense (A^T W A)", "sparse": "sparse (w-tilde)", "mf": "matrix-free"}


def plot(rows=None):
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).parent))
    import figure_style as fs

    rows = rows if rows is not None else json.loads(OUT.read_text())
    fs.use_paper_style()
    sol = sorted((r for r in rows if r["stage"] == "solve"), key=lambda r: r["n_mesh"])
    fit = [r for r in rows if r["stage"] == "fit"]

    def series(rs, key):
        x = np.array([r["n_mesh"] for r in rs if key in r], float)
        y = np.array([r[key] for r in rs if key in r], float)
        return x, y

    fig, axs = plt.subplots(2, 2, figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.78))
    # (a) one solve at fixed strength: building F, and the solve itself
    ax = axs[0, 0]
    for path, build, solve in (("dense", "t_build_dense", "t_solve_dense"),
                               ("sparse", "t_build_sparse", "t_solve_sparse"),
                               ("mf", "t_build_mf", "t_solve_mf")):
        x, y = series(sol, build)
        ax.plot(x, y, "o--", color=COLORS[path], ms=2.8, lw=0.8, alpha=0.7)
        x, y = series(sol, solve)
        ax.plot(x, y, "o-", color=COLORS[path], ms=3.2, lw=1.1, label=LABELS[path])
    x, y = series(sol, "t_solve_mf_sparseP")
    if len(x):
        ax.plot(x, y, "o:", color=COLORS["mf"], ms=2.2, lw=0.8, alpha=0.5,
                label="matrix-free, sparse P only (before)")
    for r in sol:
        if "cg_matvecs" in r:
            unconv = r["cg_matvecs"] >= 5000
            ax.annotate(f'{r["cg_matvecs"]}' + (" (cap)" if unconv else ""),
                        (r["n_mesh"], r["t_solve_mf"]), textcoords="offset points",
                        xytext=(-4, 4), ha="right", fontsize=4.3, color=COLORS["mf"])
    x, y = series(sol, "t_terms")
    ax.plot(x, y, "s:", color=fs.MUTED, ms=2.5, lw=0.8, label="w-tilde kernel pass (shared)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mesh pixels"); ax.set_ylabel("seconds")
    ax.set_title("one solve at fixed strength (numbers: CG iterations)\nsolid = solve, dashed = build F / operators", fontsize=6.5)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left")
    # (b) agreement
    ax = axs[0, 1]
    for key, lab, c, mk in (("F_rel_diff", "F: sparse vs dense", COLORS["sparse"], "s"),
                            ("s_rel_diff_sparse_dense", "solution: sparse vs dense", COLORS["sparse"], "o"),
                            ("s_rel_diff_mf", "solution: matrix-free vs exact (same prior)", COLORS["mf"], "o"),
                            ("lnZ_rel_diff_mf", "ln Z: matrix-free vs exact", COLORS["mf"], "^")):
        x, y = series(sol, key)
        ax.plot(x, np.abs(y), mk + "-", color=c, ms=3, lw=1, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mesh pixels"); ax.set_ylabel("|relative difference|")
    ax.set_title("precision at fixed strength", fontsize=6.5)
    ax.legend(fontsize=5.0, frameon=False, loc="center right")
    # (c) whole default fit: wall time
    ax = axs[1, 0]
    for path in ("dense", "sparse", "mf"):
        rs = sorted((r for r in fit if r["path"] == path), key=lambda r: r["mesh"])
        ax.plot([r["mesh"] ** 2 for r in rs], [r["seconds"] for r in rs], "o-",
                color=COLORS[path], ms=3, lw=1, label=LABELS[path])
    rs = sorted((r for r in fit if r["path"] == "mf" and "seconds_sparseP" in r),
                key=lambda r: r["mesh"])
    if rs:
        ax.plot([r["mesh"] ** 2 for r in rs], [r["seconds_sparseP"] for r in rs], "o:",
                color=COLORS["mf"], ms=2.2, lw=0.8, alpha=0.5,
                label="matrix-free, sparse P only (before)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mesh pixels"); ax.set_ylabel("seconds")
    ax.set_title("whole fit (adaptive, structure; positivity on dense/sparse)", fontsize=6.5)
    ax.axvline(1e4, color=fs.MUTED, lw=0.6, ls="--")
    ax.text(1.05e4, ax.get_ylim()[0] * 1.5, "switch", fontsize=5.5, color=fs.MUTED)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left")
    # (d) whole fit: peak memory
    ax = axs[1, 1]
    for path in ("dense", "sparse", "mf"):
        rs = sorted((r for r in fit if r["path"] == path), key=lambda r: r["mesh"])
        ax.plot([r["mesh"] ** 2 for r in rs], [r["peak_rss_gb"] for r in rs], "o-",
                color=COLORS[path], ms=3, lw=1, label=LABELS[path])
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel("mesh pixels"); ax.set_ylabel("peak memory [GB]")
    ax.set_title("whole fit: peak resident memory (~0.8 GB is the bare process)", fontsize=6.5)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left")
    from matplotlib.ticker import LogLocator, NullFormatter
    for a in axs.ravel():
        a.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        a.xaxis.set_minor_formatter(NullFormatter())
        a.tick_params(labelsize=5.8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    for q in fs.save(fig, "bench_solvers"):
        print("wrote", q)
    plot_images(fit)


def plot_images(fit):
    """The delivered models side by side at the meshes all three paths ran,
    and each against the dense (or, past it, sparse) one."""
    import matplotlib.pyplot as plt
    import figure_style as fs
    from pyuvimage.products import to_fits_orientation as fo

    meshes = sorted({r["mesh"] for r in fit if r["path"] == "sparse"})
    meshes = [m for m in meshes if m in (24, 32, 64)] or meshes[:2]
    fig, axs = plt.subplots(len(meshes), 5, figsize=(fs.TWO_COLUMN, fs.TWO_COLUMN * 0.21 * len(meshes) + 0.3),
                            squeeze=False)
    for i, m in enumerate(meshes):
        imgs = {}
        for path in ("dense", "sparse", "mf"):
            f = Path(f"/tmp/bench_fit/{path}_{m}/model.npy")
            if f.exists():
                imgs[path] = np.load(f)
        ref_name = "dense" if "dense" in imgs else "sparse"
        ref = imgs[ref_name]
        vmax = float(ref.max())
        panels = [(imgs.get("dense"), "dense"), (imgs.get("sparse"), "sparse"), (imgs.get("mf"), "matrix-free"),
                  (None if "dense" not in imgs else imgs["sparse"] - imgs["dense"], "sparse - dense"),
                  (imgs["mf"] - ref, "matrix-free - reference")]
        for j, (img, title) in enumerate(panels):
            ax = axs[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{m}x{m}", fontsize=6)
            if img is None:
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(0.5, 0.5, "dense not run\n(too large)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=5.5, color=fs.MUTED)
                continue
            if j < 3:
                ax.imshow(fo(img), origin="lower", cmap="magma", vmin=-0.05 * vmax, vmax=vmax)
            else:
                lim = 0.1 * vmax
                ax.imshow(fo(img), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.text(0.04, 0.04, f"max {np.abs(img).max() / vmax:.1e}", transform=ax.transAxes,
                        fontsize=5, color=fs.INK)
            if i == 0:
                ax.set_title(title, fontsize=6)
            if j == 4 and ref_name != "dense":
                ax.text(0.04, 0.9, f"vs {ref_name}", transform=ax.transAxes, fontsize=5, color=fs.INK)
    fig.text(0.5, 0.005, "difference maps on +-10% of the reference (dense, else sparse) peak; "
             "matrix-free differs by design (nu = 1 prior, no positivity)",
             ha="center", fontsize=5.5, color=fs.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for q in fs.save(fig, "bench_solvers_images"):
        print("wrote", q)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "solve":
        print("ROW " + json.dumps(stage_solve(int(sys.argv[2]))))
    elif what == "fit":
        Path("/tmp/bench_fit").mkdir(exist_ok=True)
        print("ROW " + json.dumps(stage_fit(int(sys.argv[2]), sys.argv[3])))
    elif what == "all":
        plot(run_all())
    elif what == "plot":
        plot()
    elif what == "rerun-mf":
        plot(rerun_mf())

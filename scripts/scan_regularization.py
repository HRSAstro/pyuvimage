"""Compare regularization / mesh options on the exponential mock."""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    _SCRIPT_FILE = __file__
except NameError:
    _SCRIPT_FILE = None

if _SCRIPT_FILE is not None:
    _search_paths = Path(_SCRIPT_FILE).resolve().parents
else:
    _search_paths = [Path.cwd(), *Path.cwd().parents]

for _parent in _search_paths:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        _repo_str = str(_parent)
        if _repo_str not in sys.path:
            sys.path.insert(0, _repo_str)
        break

from scripts.bootstrap import bootstrap_script
from src.deconv.invert import mfs_dataset_from, run_inversion
from src.deconv.model import coefficient_from_instance, reconstruction_mask_from_settings
from src.deconv.plots import dirty_images_from_fit, robust_rms
from src.deconv.settings import load_settings, validate_settings
from src.utils.grids import resolve_grids, transformer_class_from_settings

ROOT = bootstrap_script(_SCRIPT_FILE)

# (label, mesh_type, mesh_shape, regularization dict)
CASES = [
    (
        "rect_constant_1e2",
        "rectangular_uniform",
        [48, 48],
        {"type": "constant", "prior_type": "fixed", "value": 100.0},
    ),
    (
        "rect_constant_1e3",
        "rectangular_uniform",
        [48, 48],
        {"type": "constant", "prior_type": "fixed", "value": 1000.0},
    ),
    (
        "rect_adapt_i1_o100",
        "rectangular_uniform",
        [48, 48],
        {
            "type": "adapt",
            "prior_type": "fixed",
            "inner_coefficient": {"value": 1.0},
            "outer_coefficient": {"value": 100.0},
            "signal_scale": {"value": 3.0},
        },
    ),
    (
        "del_constant_split_1e3",
        "delaunay",
        [32, 32],
        {"type": "constant_split", "prior_type": "fixed", "value": 1000.0},
    ),
    (
        "del_adapt_split_r1000",
        "delaunay",
        [32, 32],
        {
            "type": "adapt_split",
            "prior_type": "fixed",
            "inner_coefficient": {"value": 0.1},
            "outer_inner_ratio": 1000,
            "signal_scale": {"value": 3.0},
        },
    ),
    (
        "del_adapt_split_zeroth",
        "delaunay",
        [32, 32],
        {
            "type": "adapt_split_zeroth",
            "prior_type": "fixed",
            "inner_coefficient": {"value": 0.1},
            "outer_inner_ratio": 1000,
            "signal_scale": {"value": 3.0},
            "zeroth_coefficient": {"value": 10.0},
            "zeroth_signal_scale": {"value": 1.0},
        },
    ),
]


def _edge_ring_metric(recon, mask_frac=0.85):
    """RMS in outer annulus / RMS in inner disk (edge artefact score)."""
    arr = np.asarray(recon, dtype=float)
    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    inner = arr[r < mask_frac * 0.6]
    outer = arr[(r >= mask_frac * 0.85) & (r <= mask_frac)]
    inner = inner[np.isfinite(inner)]
    outer = outer[np.isfinite(outer)]
    if inner.size == 0 or outer.size == 0:
        return np.nan
    i_rms = float(np.sqrt(np.mean(inner**2)))
    o_rms = float(np.sqrt(np.mean(outer**2)))
    return o_rms / i_rms if i_rms > 0 else np.nan


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import autofit as af
    import autolens as al
    from astropy.io import fits

    base = validate_settings(
        load_settings("settings/runners/mock_exponential_mfs.json", repo_root=ROOT)
    )
    out_dir = Path(base["output_path"]) / "reg_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = fits.getdata(Path(base["data_directory"]) / "truth_image.fits")
    truth = np.asarray(truth, dtype=float)

    from src.deconv.data import load_cube_data

    freq, uv, vis, sigma = load_cube_data(base)
    resolve_grids(base, uv)
    transformer_class = transformer_class_from_settings(base)

    rows = []
    fig, axes = plt.subplots(len(CASES), 3, figsize=(9.5, 2.8 * len(CASES)))
    if len(CASES) == 1:
        axes = np.asarray([axes])

    for i, (label, mesh_type, mesh_shape, reg_cfg) in enumerate(CASES):
        settings = copy.deepcopy(base)
        settings["mesh_type"] = mesh_type
        settings["mesh_shape"] = list(mesh_shape)
        settings["regularization"] = copy.deepcopy(reg_cfg)
        settings["search"] = dict(settings["search"])
        settings["search"]["unique_tag"] = label
        settings["output_path"] = str(out_dir / label)
        settings = validate_settings(settings)
        resolve_grids(settings, uv)

        mask = reconstruction_mask_from_settings(settings)
        dataset = mfs_dataset_from(
            uv, vis, sigma, mask, settings, transformer_class=transformer_class
        )
        af.conf.instance.push(
            new_path=settings["config_path"], output_path=settings["output_path"]
        )

        t0 = time.time()
        try:
            result, recon = run_inversion(settings, dataset, search_name=label)
            elapsed = time.time() - t0
            fit = result.max_log_likelihood_fit
            dirty, _, _, residual_sigma, noise_rms = dirty_images_from_fit(fit)
            try:
                coeff = coefficient_from_instance(result.max_log_likelihood_instance)
            except AttributeError:
                coeff = None
            fom = float(fit.log_likelihood_with_regularization)
            resid = np.asarray(residual_sigma, dtype=float)
            resid_finite = resid[np.isfinite(resid)]
            metrics = {
                "label": label,
                "mesh_type": mesh_type,
                "mesh_shape": mesh_shape,
                "reg_type": reg_cfg["type"],
                "coefficient": coeff,
                "fom": fom,
                "recon_peak": float(np.nanmax(recon)),
                "truth_peak": float(np.nanmax(truth)),
                "recon_sum": float(np.nansum(recon)),
                "truth_sum": float(np.nansum(truth)),
                "resid_rms": float(robust_rms(resid_finite)),
                "resid_p95": float(np.nanpercentile(np.abs(resid_finite), 95)),
                "resid_max": float(np.nanmax(np.abs(resid_finite))),
                "edge_ratio": _edge_ring_metric(recon),
                "elapsed_s": elapsed,
                "ok": True,
                "error": None,
            }
        except Exception as exc:
            elapsed = time.time() - t0
            logging.exception("Case %s failed", label)
            dirty = np.zeros_like(truth)
            recon = np.zeros_like(truth)
            residual_sigma = np.zeros_like(truth)
            noise_rms = np.nan
            metrics = {
                "label": label,
                "mesh_type": mesh_type,
                "mesh_shape": mesh_shape,
                "reg_type": reg_cfg["type"],
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": elapsed,
            }

        rows.append(metrics)
        logging.info("Finished %s: %s", label, metrics)

        for ax, img, title in zip(
            axes[i],
            [truth, recon, residual_sigma],
            ["Truth", "Reconstruction", "Residual / σ"],
        ):
            data = np.asarray(img, dtype=float)
            if "Residual" in title:
                vmax = np.nanpercentile(np.abs(data), 99.5)
                vmax = max(float(vmax), 1.0)
                im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:
                vmax = np.nanpercentile(np.abs(truth), 99.5)
                im = ax.imshow(
                    data, origin="lower", cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-12)
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[i, 0].set_ylabel(label, fontsize=8)

    fig.suptitle("Regularization scan — mock exponential MFS", fontsize=12)
    fig.tight_layout()
    plot_path = out_dir / "reg_scan_compare.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")
    print(
        f"{'label':28s} {'reg':18s} {'peak':>8s} {'edge':>7s} {'resid_p95':>10s} {'fom':>12s}"
    )
    for m in rows:
        if not m.get("ok"):
            print(f"{m['label']:28s} FAILED {m.get('error')}")
            continue
        print(
            f"{m['label']:28s} {m['reg_type']:18s} "
            f"{m['recon_peak']:8.3g} {m['edge_ratio']:7.3f} "
            f"{m['resid_p95']:10.3g} {m['fom']:12.4g}"
        )


if __name__ == "__main__":
    main()

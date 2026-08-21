"""End-to-end mock: generate exponential data, fit MFS, write diagnostic plots."""

import argparse
import logging
import sys
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
from src.deconv.mock import make_exponential_mock
from src.deconv.model import coefficient_from_instance
from src.deconv.pipeline import run_deconv
from src.deconv.plots import (
    dirty_images_from_fit,
    load_truth_matching_grid,
    plot_fit_summary,
    require_common_sky_grid,
)
from src.deconv.settings import load_settings, validate_settings

ROOT = bootstrap_script(_SCRIPT_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Make exponential mock, run MFS deconv, and plot the fit."
    )
    parser.add_argument(
        "--settings",
        default="settings/runners/mock_exponential_mfs.json",
        help="Runner settings JSON",
    )
    parser.add_argument(
        "--skip-mock",
        action="store_true",
        help="Reuse existing data under data/mock_exponential",
    )
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="Exponential source intensity (Autolens lp.Exponential)",
    )
    parser.add_argument(
        "--uv-path",
        default=None,
        help="Reuse UV coverage FITS from another mock (same geometry)",
    )
    parser.add_argument("--n-vis", type=int, default=200)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = validate_settings(load_settings(path=args.settings, repo_root=ROOT))
    data_dir = Path(settings["data_directory"])

    science_n = int(settings.get("mask_n_pixels", 64))
    pad = int(settings.get("mask_pad_pixels", 0))
    total_n = science_n + 2 * pad
    science_fov = float(settings["fov"])
    # Match resolve_grids: pixel scale from science FOV; mock covers padded mask.
    mock_fov = science_fov * (total_n / science_n)

    truth = None
    if not args.skip_mock:
        mock = make_exponential_mock(
            output_dir=data_dir,
            fov=mock_fov,
            n_pixels=total_n,
            n_vis=args.n_vis,
            noise_sigma=args.noise_sigma,
            intensity=args.intensity,
            uv_path=args.uv_path,
        )
        truth = mock["truth_image"]
        print(
            f"Mock written to {data_dir} (fov={mock_fov:.4g}\", n={total_n}, "
            f"intensity={args.intensity:g}, noise_sigma={args.noise_sigma:g})"
        )

    result = run_deconv(settings)
    reconstruction = result["source_image"]
    reconstructor = result.get("reconstructor", settings.get("reconstructor", "pixelization"))

    if reconstructor in {"log_sky", "linear_sky"}:
        sky_result = result["result"]
        dirty_data = sky_result.dirty_data
        dirty_model = sky_result.dirty_model
        residual_sigma = sky_result.residual_sigma
        noise_rms = sky_result.noise_rms
        coeff = None
        label = "log-sky" if reconstructor == "log_sky" else "linear-sky"
        smooth_disp = getattr(sky_result, "smooth_best", None) or sky_result.smooth
        title = f"Mock Exponential — {label} (smooth={smooth_disp:g})"
        print_coeff = float(smooth_disp)
    else:
        fit = result["result"].max_log_likelihood_fit
        dirty_data, dirty_model, _, residual_sigma, noise_rms = dirty_images_from_fit(
            fit
        )
        try:
            coeff = coefficient_from_instance(
                result["result"].max_log_likelihood_instance
            )
        except AttributeError:
            coeff = None
        title = "Mock Exponential — MFS deconv"
        print_coeff = coeff

    # After resolve_grids, mask_* are the authoritative shared sky grid.
    pixel_scale = float(settings["mask_pixel_scale"])
    mask_n = int(settings["mask_n_pixels"])
    expected_shape = (mask_n, mask_n)
    nyquist_pixel_scale = float(
        settings.get("nyquist_pixel_scale", settings["source_pixel_scale"])
    )

    if truth is None:
        truth_path = data_dir / "truth_image.fits"
        if truth_path.is_file():
            truth = load_truth_matching_grid(
                truth_path, shape=expected_shape, pixel_scale=pixel_scale
            )
    else:
        # Fresh mock truth: still enforce mask-grid identity.
        require_common_sky_grid(
            {"truth": truth},
            pixel_scale=pixel_scale,
            expected_shape=expected_shape,
        )

    named = {
        "dirty": dirty_data,
        "reconstruction": reconstruction,
        "residual": residual_sigma,
    }
    if truth is not None:
        named["truth"] = truth
    require_common_sky_grid(
        named, pixel_scale=pixel_scale, expected_shape=expected_shape
    )

    out_dir = Path(settings["output_path"])
    products = result.get("products") or {}
    product_paths = products.get("paths") or {}

    # Pipeline already wrote fit_summary.png + clean.png + FITS products.
    # Keep an optional truth-comparison plot for mock demos only.
    truth_plot_path = None
    if truth is not None:
        truth_plot_path = plot_fit_summary(
            dirty_image=dirty_data,
            reconstruction=reconstruction,
            truth_image=truth,
            residual_sigma=residual_sigma,
            noise_rms=noise_rms,
            output_path=out_dir / "fit_truth_compare.png",
            title=title,
            coefficient=coeff,
            pixel_scale=pixel_scale,
            nyquist_pixel_scale=nyquist_pixel_scale,
            expected_shape=expected_shape,
        )

    print(f"Fit complete: {result['output_path']}")
    if product_paths:
        print(f"Fit summary: {product_paths.get('fit_summary')}")
        print(f"Clean plot:  {product_paths.get('clean_plot')}")
        print(f"FITS: dirty_image / dirty_model / residual_sigma / clean_image / reconstruction")
        if products.get("beam_params"):
            bp = products["beam_params"]
            print(
                f"Clean beam: BMAJ={bp.get('bmaj_arcsec', float('nan')):.4g}\" "
                f"BMIN={bp.get('bmin_arcsec', float('nan')):.4g}\" "
                f"BPA={bp.get('bpa_deg', float('nan')):.2f}° E of N"
            )
        if products.get("noise_rms_jybeam") is not None:
            print(f"Noise RMS: {products['noise_rms_jybeam']:.4g} Jy/beam")
    if truth_plot_path is not None:
        print(f"Truth compare: {truth_plot_path}")
    if print_coeff is not None:
        label = "smooth" if reconstructor in {"log_sky", "linear_sky"} else "λ"
        print(f"Regularization {label} = {print_coeff:.6g}")
    if result.get("dirty_snr") is not None:
        snr = result["dirty_snr"]
        print(
            f"Dirty SNR: peak={snr['peak']:.4g} / rms={snr['noise_rms']:.4g} "
            f"= {snr['snr']:.4g} (threshold={snr['threshold']:.4g}) → {reconstructor}"
        )
    if reconstructor == "log_sky":
        r = result["result"]
        print(
            f"Log-sky: I0={r.i0:.4g} chi2={r.chi2:.4g} "
            f"nit={r.n_iter} success={r.success}"
        )
        if getattr(r, "optimize_smooth", False):
            print(
                f"  smooth search: init={r.smooth_init:.4g} best={r.smooth_best:.4g} "
                f"LLWR={r.llwr:.6g} trials={len(r.smooth_trials or [])}"
            )
    elif reconstructor == "linear_sky":
        r = result["result"]
        print(
            f"Linear-sky: chi2={r.chi2:.4g} "
            f"nit={r.n_iter} success={r.success}"
        )
        if getattr(r, "optimize_smooth", False):
            print(
                f"  smooth search: init={r.smooth_init:.4g} best={r.smooth_best:.4g} "
                f"LLWR={r.llwr:.6g} trials={len(r.smooth_trials or [])}"
            )
    return result


if __name__ == "__main__":
    main()

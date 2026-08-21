"""Command-line interface.

    pyuvimage import obs.ms mydata/            # MS -> dataset (needs casacore)
    pyuvimage fit mydata/ --fov 3.0            # reconstruct
    pyuvimage convert export.npz mydata/       # CASA-script export -> dataset
    pyuvimage demo demo_out/                   # self-contained demo run
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyuvimage",
        description="Image reconstruction of radio interferometric data by "
        "forward modelling in the uv-plane.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_imp = sub.add_parser("import", help="convert a CASA MS to a dataset")
    p_imp.add_argument("ms")
    p_imp.add_argument("out")
    p_imp.add_argument("--field", type=int, default=0)
    p_imp.add_argument("--spw", type=int, default=0)
    p_imp.add_argument(
        "--column", default="auto", choices=["auto", "data", "corrected"]
    )
    p_imp.add_argument(
        "--noise", default="difference", choices=["difference", "sigma"],
        help="noise from time-differenced visibilities (default) or the MS "
        "SIGMA column",
    )
    p_imp.add_argument("--overwrite", action="store_true")

    p_conv = sub.add_parser(
        "convert", help="convert a casa_export.py .npz to a dataset directory"
    )
    p_conv.add_argument("npz")
    p_conv.add_argument("out")
    p_conv.add_argument("--overwrite", action="store_true")

    p_fit = sub.add_parser("fit", help="reconstruct an image or cube")
    p_fit.add_argument("dataset", help="dataset directory or .npz")
    p_fit.add_argument(
        "--fov", type=float, required=True,
        help="full field of view [arcsec]; must cover all emission",
    )
    p_fit.add_argument("--mode", default="mfs", choices=["mfs", "cube"])
    p_fit.add_argument("--out", default="pyuvimage_out")
    p_fit.add_argument(
        "--pixel-scale", default="auto",
        help='pixel scale of every output product: "auto" (half-Nyquist, '
        '~4 pixels per beam), "nyquist" (~2 per beam, cheaper), or arcsec',
    )
    p_fit.add_argument(
        "--reg", default="adaptive",
        choices=["adaptive", "gibbs", "matern", "gaussian", "exponential",
                 "constant"],
        help="source prior. adaptive (default): two-stage, prior amplitude "
        "tracks a first-pass model as b^power — best overall on extended "
        "emission. gibbs: non-stationary GP whose correlation *length* "
        "shortens where the source is bright — sharpest on compact features. "
        "matern/exponential: stationary GP (PyAutoLabs' default). gaussian: "
        "matern with a spatial envelope, for sparse visibilities. constant: "
        "nearest-neighbour gradient (rank-deficient; evidence ill-behaved).",
    )
    p_fit.add_argument(
        "--adapt-power", type=float, default=None,
        help="exponent in the adaptive/gibbs brightness weighting (default 2)",
    )
    p_fit.add_argument(
        "--envelope-fwhm", default="auto",
        help='for --reg gaussian: FWHM [arcsec] of the Gaussian envelope, '
        '"auto" to size it from the dirty image, or "optimise" to fit it',
    )
    p_fit.add_argument(
        "--envelope-centre", default="auto",
        help='for --reg gaussian: "auto" (the dirty-image peak), "centre" '
        '(the phase centre), or "dy,dx" in arcsec',
    )
    p_fit.add_argument(
        "--envelope-floor", type=float, default=1e-2,
        help="for --reg gaussian: prior width far from the centre relative "
        "to the centre (default 0.01)",
    )
    p_fit.add_argument(
        "--scale", dest="reg_scale", default="auto",
        help='kernel correlation length: "auto" (the synthesised beam size, '
        'recommended), a value in arcsec, or "optimise" to fit it',
    )
    p_fit.add_argument(
        "--nu", type=float, default=1.5,
        help="Matern smoothness (0.5 = exponential, higher = smoother)",
    )
    p_fit.add_argument(
        "--lambda", dest="coefficient", default="auto",
        help='regularisation strength: "auto" or a value',
    )
    p_fit.add_argument(
        "--criterion", default="discrepancy",
        choices=["evidence", "discrepancy"],
        help="how the source-prior hyperparameters are optimised: "
        "fit-to-the-noise-level (default) or maximum Bayesian evidence",
    )
    p_fit.add_argument(
        "--chi2-target", type=float, default=1.0,
        help="target chi^2/N for the discrepancy criterion (default 1.0)",
    )
    p_fit.add_argument(
        "--no-positive", action="store_true",
        help="allow negative flux in the model (faster, less robust)",
    )
    p_fit.add_argument(
        "--dish-diameter", type=float, default=None,
        help="antenna diameter [m] for the primary beam (default: from MS)",
    )
    p_fit.add_argument("--no-pb", action="store_true", help="skip primary-beam products")
    p_fit.add_argument(
        "--transformer", default="auto", choices=["auto", "dft", "nufft"]
    )
    p_fit.add_argument("--mesh", type=int, default=None, help="mesh pixels per side")
    p_fit.add_argument(
        "--no-uncertainty", action="store_true",
        help="skip the per-pixel 1 sigma posterior map",
    )
    p_fit.add_argument(
        "--point-sources", action="store_true",
        help="fit analytic point components; auto-detect their positions",
    )
    p_fit.add_argument(
        "--point", action="append", metavar="dRA,dDec", default=None,
        help="fit a point at this offset [arcsec]; the position is refined. "
             "Repeatable. Implies --point-sources and disables auto-detection",
    )
    p_fit.add_argument(
        "--point-significance", type=float, default=5.0,
        help="keep auto-detected points above this significance (default 5)",
    )
    p_fit.add_argument(
        "--no-point-retune", action="store_true",
        help="keep the mesh-only regularisation when point components are "
             "fitted, instead of re-tuning to chi^2 = N (looser mesh, wider "
             "point error bars)",
    )
    p_fit.add_argument(
        "--max-points", type=int, default=5,
        help="most auto-detected point components to keep (default 5)",
    )

    p_demo = sub.add_parser("demo", help="run a self-contained mock demo")
    p_demo.add_argument("out", nargs="?", default="pyuvimage_demo")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    from ._jax_guard import report_if_disabled

    report_if_disabled()

    if args.command == "import":
        from .ms_import import import_ms

        import_ms(
            args.ms, args.out, data_column=args.column, field=args.field,
            spw=args.spw, noise_estimate=args.noise, overwrite=args.overwrite,
        )
        return 0

    if args.command == "convert":
        from .uvdata import UVData

        UVData.read(args.npz).write(args.out, overwrite=args.overwrite)
        print(f"dataset written to {args.out}")
        return 0

    if args.command == "fit":
        from .api import run

        try:
            pixel_scale = float(args.pixel_scale)
        except ValueError:
            pixel_scale = args.pixel_scale
        try:
            coefficient = float(args.coefficient)
        except ValueError:
            coefficient = args.coefficient
        try:
            reg_scale = float(args.reg_scale)
        except ValueError:
            reg_scale = args.reg_scale
        # explicit positions win; otherwise --point-sources means auto-detect
        if args.point:
            point_sources = [
                tuple(float(v) for v in p.split(",")) for p in args.point
            ]
        else:
            point_sources = bool(args.point_sources)
        run(
            args.dataset,
            fov=args.fov,
            mode=args.mode,
            out=args.out,
            pixel_scale=pixel_scale,
            mesh_shape=(args.mesh, args.mesh) if args.mesh else None,
            reg=args.reg,
            coefficient=coefficient,
            reg_scale=reg_scale,
            nu=args.nu,
            envelope_fwhm=(
                args.envelope_fwhm
                if args.envelope_fwhm in ("auto", "optimise")
                else float(args.envelope_fwhm)
            ),
            envelope_centre=(
                args.envelope_centre
                if args.envelope_centre in ("auto", "centre")
                else tuple(float(v) for v in args.envelope_centre.split(","))
            ),
            envelope_floor=args.envelope_floor,
            **({} if args.adapt_power is None
               else {'adapt_power': args.adapt_power}),
            criterion=args.criterion,
            chi2_target=args.chi2_target,
            positive_only=not args.no_positive,
            transformer=args.transformer,
            pb_correction=not args.no_pb,
            dish_diameter=args.dish_diameter,
            uncertainty_map=not args.no_uncertainty,
            point_sources=point_sources,
            point_significance=args.point_significance,
            max_points=args.max_points,
            point_retune=not args.no_point_retune,
        )
        return 0

    if args.command == "demo":
        from .api import run
        from .mock import make_demo_dataset

        uvd, truth, _, comps = make_demo_dataset(point_flux_jy=0.004)
        # the demo shows the tool as configured, point components included:
        # its mock contains one true point source that no pixel grid can hold
        res = run(uvd, fov=3.0, out=args.out, point_sources=True)
        pts = res.products[0].points or []
        truth_pt = comps["points"][0] if comps["points"] else None
        print(f"demo products written to {args.out}")
        if truth_pt:
            print(f"  true point source: {truth_pt['flux']:.4f} Jy at "
                  f"dRA {truth_pt['centre'][0]:+.2f}\", "
                  f"dDec {truth_pt['centre'][1]:+.2f}\"")
        for p in pts:
            print(f"  recovered:         {p.flux:.4f} +- {p.flux_error:.4f} Jy "
                  f"at dRA {p.d_ra:+.2f}\", dDec {p.d_dec:+.2f}\" "
                  f"({p.significance:.0f} sigma)")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

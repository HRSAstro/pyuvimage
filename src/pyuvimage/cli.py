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

import numpy as np



def _all_sigma(uvd):
    """Median-able view of a dataset's sigma, single- or multi-spw."""
    import numpy as _np

    return _np.concatenate([_np.asarray(s.noise).real.ravel() for s in uvd.spws])


def _parse_pair(text: str, flag: str) -> tuple[float, float]:
    parts = [
        q for q in text.replace("(", "").replace(")", "").split(",") if q.strip()
    ]
    if len(parts) != 2:
        raise SystemExit(
            f"{flag}: expected 'x,y' in arcsec from the phase centre, "
            f"got {text!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise SystemExit(f"{flag}: {text!r} is not a pair of numbers in arcsec")


def _parse_centre(text):
    """"centre" / "auto" pass through; "x,y" becomes a (float, float).

    Image axes, +x right and +y up, arcsec from the phase centre -- the same
    convention as `--point` and as `api.run`, which converts to sky
    coordinates via `pointsource.image_to_sky`.
    """
    if not isinstance(text, str):
        return text
    t = text.strip().lower()
    if t in ("centre", "center", "none", ""):
        return "centre"
    if t == "auto":
        return "auto"
    return _parse_pair(t, "--image-centre")


def _parse_spw(text: str):
    """"0" -> 0;  "all" -> "all";  "0,2" / "0-3" / "0-1,4" -> [0, 2] / ... ."""
    text = str(text).strip()
    if text.lower() == "all":
        return "all"
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:  # allow a leading minus to fail as an int instead
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    if not ids:
        raise ValueError(f"could not parse --spw {text!r}")
    return ids[0] if len(ids) == 1 else sorted(set(ids))


def _hint_negative_centre(argv: list[str]) -> None:
    """argparse reads `--image-centre -2.3,0.3` as a missing argument.

    A leading minus makes argparse treat the value as another option flag, and
    the resulting "expected one argument" says nothing about the fix. Negative
    offsets are entirely normal -- half the sky is at negative dRA -- so catch
    it before argparse does.
    """
    for i, a in enumerate(argv):
        if a in ("--image-centre", "--point") and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and "," in nxt:
                raise SystemExit(
                    f'{a}: write {a}="{nxt}" (with the "=") when the '
                    f"offset starts with a minus sign, or argparse reads it "
                    f"as another flag."
                )


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
    p_imp.add_argument(
        "--spw", default="0",
        help='spectral window(s): a DATA_DESC_ID ("0"), a comma-separated '
             'list or range ("0,2", "0-3"), or "all". Several windows are '
             "imaged together by MFS.",
    )
    p_imp.add_argument(
        "--column", default="auto", choices=["auto", "data", "corrected"]
    )
    p_imp.add_argument(
        "--noise", default="difference",
        choices=["difference", "hybrid", "scaled", "sigma"],
        help="how to set the per-visibility noise. MS weights are relative, "
        "not absolute, so the scale is always recomputed from the data. "
        '"difference" (default) times-differences the visibilities and uses '
        'no weights; "hybrid" adds the weight column\'s time profile; '
        '"scaled" takes the whole shape from the weights; "sigma" trusts the '
        "SIGMA column and warns. See docs/noise.md",
    )
    p_imp.add_argument(
        "--noise-chunk", type=float, default=600.0,
        help="how finely --noise difference resolves the noise in time, in "
        "seconds (default 600). Blocks with too few integrations fall back to "
        "one sigma per baseline automatically; 0 forces that everywhere. "
        "See docs/noise.md",
    )
    p_imp.add_argument("--overwrite", action="store_true")

    p_conv = sub.add_parser(
        "convert", help="convert a casa_export.py .npz to a dataset directory"
    )
    p_conv.add_argument("npz")
    p_conv.add_argument("out")
    p_conv.add_argument("--overwrite", action="store_true")
    p_conv.add_argument(
        "--noise", default="keep",
        choices=["keep", "difference", "hybrid", "scaled"],
        help="re-estimate the noise while converting, and store the result so "
        "no later run pays for it again. Default 'keep' uses the map already "
        "in the .npz. Needs the antenna/time columns the export stores; "
        "'hybrid' and 'scaled' additionally need the weight column",
    )
    p_conv.add_argument(
        "--noise-chunk", type=float, default=None,
        help="how finely --noise difference resolves the noise in time, in "
        "seconds (default 600); 0 for one sigma per baseline",
    )

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
        help='model-mesh scale: "auto" (Nyquist of the baseline 95%% of '
        'samples fall within -- what the bulk of the data supports), '
        '"nyquist" (Nyquist of the *longest* baseline: finer, much slower, '
        'and more mesh than a sparse long-baseline tail constrains), '
        '"fine" (half that again), or a value in arcsec. Products are '
        'written on a grid --oversample times finer.',
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
        choices=["discrepancy", "structure", "evidence"],
        help="how the source-prior hyperparameters are optimised: "
        "fit to the noise level (chi^2 = N, default), make the residual map "
        "look like noise (structure ratio = 1), or maximum Bayesian evidence",
    )
    p_fit.add_argument(
        "--chi2-target", type=float, default=1.0,
        help="target chi^2/N for the discrepancy criterion (default 1.0)",
    )
    p_fit.add_argument(
        "--enforce-positive", action="store_true",
        help="keep positivity even if the non-negative solver looks "
        "unreliable. By default pyuvimage probes the solver and falls back to "
        "the unconstrained solve when it is ignoring the prior or fitting far "
        "worse -- right for a good image, wrong when a strictly non-negative "
        "model is the point. The fit logs which solver actually ran, and "
        "fit_parameters.json records it.",
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
        "--transformer", default="auto",
        choices=["auto", "dft", "nufft", "pynufft"],
        help=(
            "Fourier transform backend. auto: the direct DFT while it is "
            "affordable, then pynufft or the JAX NUFFT, whichever fits in "
            "memory -- on large datasets that is pynufft, because the JAX "
            "one transforms the whole mapping matrix in one batch (see "
            "docs/parameters.md)"
        ),
    )
    p_fit.add_argument("--mesh", type=int, default=None, help="mesh pixels per side")
    p_fit.add_argument(
        "--image-centre", default="centre", metavar="centre|auto|x,y",
        help='where to centre the reconstruction: "centre" (the phase '
        'centre, default), "auto" (the brightest peak in a wide-field dirty '
        'image), or an offset in arcsec from the phase centre as "x,y" in '
        "image axes -- +x right and +y up, as you would read it off "
        "summary.png (RA increases leftward, so x = -dRA), the same "
        "convention as --point. A negative x needs the "
        '"=" form -- --image-centre="-2.3,0.3" -- because argparse reads a '
        "leading minus as a flag. Cost goes as the square of --fov, so "
        "recentring on a source a few arcsec off the phase centre is much "
        "cheaper than growing the field to reach it. The output WCS follows.",
    )
    p_fit.add_argument(
        "--no-uncertainty", action="store_true",
        help="skip the per-pixel 1 sigma posterior map",
    )
    p_fit.add_argument(
        "--point-sources", action="store_true",
        help="fit analytic point components; auto-detect their positions",
    )
    p_fit.add_argument(
        "--point", action="append", metavar="x,y", default=None,
        help="fit a point at this offset [arcsec] from the phase centre, in "
             "image axes: +x right and +y up, as you would read it off "
             "summary.png. (RA increases leftward, so x = -dRA; products are "
             "written in dRA/dDec.) The position is refined. Repeatable. "
             "Implies --point-sources and disables auto-detection. A negative "
             'x needs the "=" form: --point="-1.2,0.4"',
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

    _hint_negative_centre(list(sys.argv[1:] if argv is None else argv))
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
            spw=_parse_spw(args.spw), noise_estimate=args.noise,
            noise_chunk_seconds=args.noise_chunk,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "convert":
        from .uvdata import read_dataset, recompute_noise

        uvd = read_dataset(args.npz)
        if args.noise != "keep":
            before = float(np.median(_all_sigma(uvd)))
            uvd = recompute_noise(uvd, args.noise, args.noise_chunk)
            after = float(np.median(_all_sigma(uvd)))
            print(
                "noise re-estimated (%s): median sigma %.4g -> %.4g Jy "
                "(x%.3f); stored in the dataset, so fits do not repeat it"
                % (args.noise, before, after, after / max(before, 1e-30))
            )
        uvd.write(args.out, overwrite=args.overwrite)
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
            # image x,y, straight through -- api.run converts
            point_sources = [_parse_pair(p, "--point") for p in args.point]
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
            enforce_positive=args.enforce_positive,
            transformer=args.transformer,
            image_centre=_parse_centre(args.image_centre),
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

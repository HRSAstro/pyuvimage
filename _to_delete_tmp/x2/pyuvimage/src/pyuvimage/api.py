"""High-level pipeline: the one function most users call.

    import pyuvimage
    result = pyuvimage.run("mydata/", fov=3.0)

or from the command line:

    pyuvimage fit mydata/ --fov 3.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import beam as beam_mod
from . import fitting, primary_beam
from .grids import ImageGeometry, resolve_geometry
from .products import ProductSet, upsample_model, write_products
from .uvdata import UVData

logger = logging.getLogger("pyuvimage")


@dataclass
class RunResult:
    geometry: ImageGeometry
    products: list[ProductSet]
    written: dict
    scan: dict | None
    uvdata: UVData
    parameters: dict | None = None

    @property
    def model(self) -> np.ndarray:
        return self.products[0].model_mesh

    @property
    def rms(self) -> float:
        return self.products[0].rms


def run(
    dataset: str | Path | UVData,
    fov: float,
    mode: str = "mfs",
    out: str | Path = "pyuvimage_out",
    pixel_scale: float | str = "auto",
    mesh_shape: tuple[int, int] | None = None,
    reg: str = "gibbs",
    coefficient: float | str = "auto",
    reg_scale: float | str = "auto",
    nu: float = fitting.DEFAULT_NU,
    envelope_fwhm: float | str = "auto",
    envelope_centre: tuple[float, float] | str = "auto",
    envelope_floor: float = 1e-2,
    criterion: str = "discrepancy",
    chi2_target: float = 1.0,
    positive_only: bool = True,
    transformer: str = "auto",
    mask_shape: str = "square",
    oversample: int = 2,
    pb_correction: bool = True,
    dish_diameter: float | None = None,
    pb_factor: float = primary_beam.DEFAULT_PB_FACTOR,
    uncertainty_map: bool = True,
    point_sources: bool | list = False,
    point_significance: float = 5.0,
    max_points: int = 5,
    point_retune: bool = True,
    write: bool = True,
) -> RunResult:
    """Reconstruct an image (mfs) or image cube (cube) from visibilities.

    Parameters (all optional except ``fov``):
        dataset: path of a pyuvimage dataset directory (from ``pyuvimage
            import``), a legacy pyuvimage_dev export directory, or a UVData.
        fov: full field of view of the reconstruction [arcsec].
        mode: "mfs" (single image from all channels) or "cube" (per-channel).
        pixel_scale: "auto" (Nyquist, 0.5/b_max) or arcsec.
        reg: the source prior (regularisation scheme):
            matern | exponential (Gaussian-process priors with a correlation
            length, the PyAutoLabs default), constant (nearest-neighbour
            gradient), or adapt (two-stage brightness-adaptive).
        coefficient: "auto" (optimised) or a fixed prior strength.
        reg_scale: "auto" (optimised) or the kernel correlation length in
            arcsec (kernel priors only).
        nu: Matern smoothness (0.5 = exponential, higher = smoother).
        envelope_fwhm: for reg="gaussian", the FWHM [arcsec] of the Gaussian
            envelope on the prior width; "auto" sizes it from the extent of
            the significant emission in the dirty image, "optimise" fits it
            as a free hyperparameter alongside the coefficient.
        envelope_centre: (y, x) arcsec offset of the envelope; "auto" places
            it at the dirty-image peak (not necessarily the phase centre),
            "centre" forces the phase centre.
        envelope_floor: prior width far from the centre, relative to the
            centre (smaller suppresses distant structure more strongly).
        criterion: how the prior hyperparameters are optimised: "evidence"
            (maximise the Bayesian evidence, as PyAutoLabs does) or
            "discrepancy" (drive chi^2 to chi2_target * N).
        chi2_target: target chi^2/N for the discrepancy criterion.
        positive_only: constrain the source to non-negative flux.
        dish_diameter: antenna diameter [m] for the primary beam; defaults to
            the value stored at import.
    """
    if mode not in ("mfs", "cube"):
        raise ValueError("mode must be 'mfs' or 'cube'")
    uvd = dataset if isinstance(dataset, UVData) else UVData.read(dataset)
    logger.info(
        "dataset: %d visibilities x %d channel(s), central frequency %.6g GHz",
        uvd.n_vis, uvd.n_chan, uvd.central_frequency / 1e9,
    )

    geometry = resolve_geometry(
        fov_arcsec=fov,
        max_baseline_wavelengths=uvd.max_baseline_wavelengths,
        pixel_scale=pixel_scale,
        mesh_shape=mesh_shape,
        oversample=oversample,
    )
    logger.info(
        "mesh %dx%d at %.4g\"/pix (Nyquist %.4g\"), image grid %dx%d",
        *geometry.mesh_shape, geometry.mesh_pixel_scale,
        geometry.nyquist_pixel_scale, *geometry.shape_native,
    )
    n_pix = geometry.mesh_shape[0] * geometry.mesh_shape[1]
    n_data_all = 2 * uvd.n_vis * uvd.n_chan
    if n_pix > n_data_all:
        logger.warning(
            "the model has more pixels (%d) than data points (%d): the "
            "reconstruction is under-constrained and its faint structure is "
            "set mainly by the source prior, not the data. Residuals can look "
            "smaller than the noise even at chi^2/N = 1. Consider a smaller "
            "--fov or a coarser --pixel-scale if the source allows it.",
            n_pix, n_data_all,
        )

    dish = dish_diameter or uvd.meta.get("dish_diameter_m")
    if pb_correction and not dish:
        logger.warning(
            "no dish diameter known: skipping primary-beam products "
            "(pass dish_diameter=... to enable)"
        )
        pb_correction = False

    # ----------------------------------------------------------------- MFS
    uv, d, n = uvd.flattened()
    mfs_dataset = fitting.make_dataset(
        uv, d, n, geometry, transformer, mask_shape=mask_shape
    )
    logger.info("fitting MFS image (%d visibility samples)...", len(d))
    # The natural correlation length of a Gaussian-process source prior is
    # the resolution element: structure finer than the synthesised beam is
    # not constrained by the data. Measure it before fitting.
    beam_scale = None
    envelope = None
    needs_beam = bool(point_sources) or (
        reg in fitting.KERNEL_REGULARIZATIONS
        and (reg_scale == "auto" or reg in fitting.ENVELOPE_REGULARIZATIONS)
    )
    beam_size = None
    if needs_beam:
        b = beam_mod.fit_beam(
            beam_mod.DirtyImager(mfs_dataset).dirty_beam, geometry.pixel_scale
        )
        beam_size = float(np.sqrt(b.bmaj_arcsec * b.bmin_arcsec))
    if reg in fitting.KERNEL_REGULARIZATIONS:
        if reg_scale == "auto":
            beam_scale = beam_size
            logger.info(
                "source prior correlation length set to the beam size: "
                "%.4g arcsec", beam_scale,
            )
        elif reg_scale != "optimise":
            beam_scale = float(reg_scale)
    if reg in fitting.ADAPTIVE_REGULARIZATIONS:
        envelope = {"floor": float(envelope_floor)}
        if reg == "gibbs":
            envelope["ell_floor"] = fitting.GIBBS_ELL_FLOOR
    if reg in fitting.ENVELOPE_REGULARIZATIONS:
        from .envelope import estimate_envelope

        imager = beam_mod.DirtyImager(mfs_dataset)
        auto_centre, auto_fwhm = estimate_envelope(
            imager.dirty_image(np.asarray(mfs_dataset.data)),
            pixel_scale=geometry.pixel_scale,
            rms=imager.rms,
            beam_fwhm=beam_size,
            max_fwhm=geometry.fov_arcsec / 2.0,
        )
        optimise_env = envelope_fwhm == "optimise"
        fwhm = (
            auto_fwhm if envelope_fwhm in ("auto", "optimise")
            else float(envelope_fwhm)
        )
        if envelope_centre == "auto":
            centre = auto_centre
        elif envelope_centre == "centre":
            centre = (0.0, 0.0)
        else:
            centre = (float(envelope_centre[0]), float(envelope_centre[1]))
        envelope = {"fwhm": fwhm, "floor": float(envelope_floor),
                    "centre": centre}
        logger.info(
            "Gaussian envelope prior: FWHM %.4g arcsec centred on "
            "(dy=%.3g, dx=%.3g) arcsec [dirty-image peak] (floor %.3g)",
            fwhm, centre[0], centre[1], envelope_floor,
        )

    fixed_prior = _fixed_prior(reg, coefficient, reg_scale, nu, beam_scale)
    mfs_fit = fitting.fit_dataset(
        mfs_dataset, geometry, reg_kind=reg, prior=fixed_prior,
        positive_only=positive_only, criterion=criterion,
        nu=nu, fixed_scale=beam_scale, envelope=envelope,
        optimise_envelope=locals().get("optimise_env", False),
        chi2_target=chi2_target,
    )
    # Optional analytic point components, solved in the same linear system.
    # Opt-in: never added unless asked for, and auto-detected candidates are
    # kept only above `point_significance`.
    point_solution = None
    if point_sources and mfs_fit.chi_squared / (2 * len(d)) > 2.0 * chi2_target:
        # The coefficient search drives chi^2 to the target, so a mesh fit
        # still far above it means the model cannot describe the data at all
        # -- emission outside --fov being the usual cause.  The residual is
        # then model error, not sky, and fitting points to it produces
        # nonsense: on the out-of-field test it returned an 11.5 Jy "source"
        # in a 0.09 Jy field, at 76 sigma.
        logger.warning(
            "skipping point-source fitting: the pixelized model sits at "
            "chi^2/N = %.4g against a target of %.3g, so the residual is "
            "model error rather than sky and any point fitted to it would be "
            "spurious. Fix the fit first (usually --fov).",
            mfs_fit.chi_squared / (2 * len(d)), chi2_target,
        )
    elif point_sources:
        from .pointsource import fit_point_sources

        positions = point_sources if isinstance(point_sources, list) else None
        logger.info(
            "fitting analytic point sources (%s)...",
            "user positions" if positions else
            f"auto-detect above {point_significance:.1f} sigma",
        )
        point_solution = fit_point_sources(
            mfs_fit.fit.inversion, mfs_dataset, geometry, positions=positions,
            significance=point_significance, max_points=max_points,
            dirty_imager=beam_mod.DirtyImager(mfs_dataset),
            beam_fwhm=beam_size,
            retune=bool(point_retune) and criterion == "discrepancy",
            chi2_target=chi2_target,
        )
        if point_solution.points:
            from .pointsource import PointAugmentedFit

            mfs_fit = PointAugmentedFit(mfs_fit, point_solution)
            logger.info(
                "  %d point source(s), total %.4g Jy; chi2/N now %.3f",
                len(point_solution.points), point_solution.total_point_flux,
                point_solution.chi_squared / (2 * len(d)),
            )
        else:
            logger.info("  no point source passed the significance cut")

    scan = mfs_fit.scan.as_dict() if mfs_fit.scan is not None else None
    n_data = 2 * len(d)
    if mfs_fit.chi_squared / n_data > 1.3 * chi2_target:
        logger.warning(
            "the delivered model sits at chi^2/N = %.3g against a target of "
            "%.3g: it does not reproduce the data, and every product below "
            "(fluxes, residuals, uncertainties, point sources) should be "
            "treated as unreliable. Common causes: the field of view (--fov) "
            "does not cover all the emission, the mesh is too coarse for the "
            "S/N, or the noise map underestimates the true noise.",
            mfs_fit.chi_squared / n_data, chi2_target,
        )

    products: list[ProductSet] = []
    if mode == "mfs":
        products.append(
            _products_for(mfs_fit, mfs_dataset, geometry, uvd,
                          uvd.central_frequency, pb_correction, dish, pb_factor,
                          oversample, uncertainty_map)
        )
        freqs = np.atleast_1d(uvd.central_frequency)
    else:
        frozen = mfs_fit.prior
        logger.info(
            "cube mode: source prior frozen from the MFS fit (%s)",
            ", ".join(f"{k}={v:.4g}" for k, v in frozen.items()),
        )
        for c in range(uvd.n_chan):
            ch = uvd.select(channel=c)
            uv_c, d_c, n_c = ch.flattened()
            ds_c = fitting.make_dataset(
                uv_c, d_c, n_c, geometry, transformer, mask_shape=mask_shape
            )
            sf = fitting.fit_dataset(
                ds_c, geometry, reg_kind=reg, prior=frozen,
                positive_only=positive_only, criterion=criterion, nu=nu,
                envelope=envelope,
            )
            logger.info(
                "channel %d/%d: chi2=%.5g", c + 1, uvd.n_chan, sf.chi_squared
            )
            products.append(
                _products_for(sf, ds_c, geometry, uvd, float(ch.frequencies[0]),
                              pb_correction, dish, pb_factor, oversample,
                              uncertainty_map)
            )
        freqs = uvd.frequencies

    if point_solution is not None:
        products[0].points = point_solution.points
    parameters = _parameter_record(
        uvd, geometry, mode, reg, criterion, chi2_target, positive_only,
        transformer, oversample, dish, pb_factor, pb_correction, mfs_fit, scan,
        envelope=envelope, point_solution=point_solution,
    )
    written = {}
    if write:
        written = write_products(
            products, geometry, uvd.meta, freqs, out, scan=scan,
            parameters=parameters,
        )
        logger.info("products written to %s", Path(out).resolve())
    return RunResult(
        geometry=geometry, products=products, written=written, scan=scan,
        uvdata=uvd, parameters=parameters,
    )


def _fixed_prior(reg, coefficient, reg_scale, nu, beam_scale=None) -> dict | None:
    """Assemble a fully specified prior, or None if anything must be fitted."""
    kernel = reg in fitting.KERNEL_REGULARIZATIONS
    if coefficient == "auto":
        return None
    prior = {"coefficient": float(coefficient)}
    if kernel:
        if reg_scale == "optimise":
            return None
        prior["scale"] = float(beam_scale if reg_scale == "auto" else reg_scale)
        prior["nu"] = float(nu)
    return prior


def _parameter_record(
    uvd, geometry, mode, reg, criterion, chi2_target, positive_only,
    transformer, oversample, dish, pb_factor, pb_correction, fit, scan,
    envelope=None, point_solution=None,
) -> dict:
    """Every parameter that defined this run, for the record and for reuse."""
    return {
        "pyuvimage_version": __import__("pyuvimage").__version__,
        "data": {
            "n_visibilities": int(uvd.n_vis),
            "n_channels": int(uvd.n_chan),
            "central_frequency_hz": float(uvd.central_frequency),
            "max_baseline_wavelengths": float(uvd.max_baseline_wavelengths),
            "noise_estimate": uvd.meta.get("noise_estimate", "from file"),
            "median_sigma_jy": float(np.median(np.asarray(uvd.noise).real)),
            "stokes": uvd.meta.get("stokes", "I"),
        },
        "geometry": {
            "mode": mode,
            "fov_arcsec": geometry.fov_arcsec,
            "mesh_shape": list(geometry.mesh_shape),
            "mesh_pixel_scale_arcsec": geometry.mesh_pixel_scale,
            "image_shape": list(geometry.shape_native),
            "image_pixel_scale_arcsec": geometry.pixel_scale,
            "nyquist_pixel_scale_arcsec": geometry.nyquist_pixel_scale,
            "oversample": int(oversample),
            "mask_radius_arcsec": geometry.mask_radius,
        },
        "source_prior": {
            "regularization": reg,
            **({k: v for k, v in {
                "envelope_fwhm_arcsec": envelope.get("fwhm"),
                "envelope_centre_arcsec": (
                    list(envelope["centre"]) if "centre" in envelope else None
                ),
                "envelope_floor": envelope.get("floor"),
                "adaptive_power": envelope.get("power"),
            }.items() if v is not None} if envelope else {}),
            **{k: float(v) for k, v in fit.prior.items()},
            "optimised": scan is not None,
            "criterion": criterion if scan is not None else "fixed",
            "chi2_target": float(chi2_target),
            "n_evaluations": (scan or {}).get("n_evaluations", 0),
        },
        "solver": {
            "positive_only": bool(positive_only),
            "transformer": transformer,
            "jax": fitting.jax_available(),
        },
        "primary_beam": {
            "applied": bool(pb_correction),
            "dish_diameter_m": float(dish) if dish else None,
            "pb_factor": float(pb_factor),
        },
        "point_sources": (
            point_solution.as_dict() if point_solution is not None
            else {"enabled": False}
        ),
        "fit_quality": {
            "chi_squared": fit.chi_squared,
            "n_data": int(2 * uvd.n_vis * uvd.n_chan),
            "log_evidence": fit.log_evidence,
        },
    }


def _products_for(
    sf: fitting.SingleFit,
    dataset,
    geometry: ImageGeometry,
    uvd: UVData,
    frequency_hz: float,
    pb_correction: bool,
    dish: float | None,
    pb_factor: float,
    oversample: int,
    uncertainty_map: bool = True,
) -> ProductSet:
    imager = beam_mod.DirtyImager(dataset)
    data = np.asarray(dataset.data)
    model_vis = sf.model_visibilities
    resid_vis = data - model_vis

    model_mesh = sf.model_mesh_image
    model_image = sf.model_image  # exact: via the mapping matrices
    uncertainty = uncertainty_sampling = None
    if uncertainty_map:
        try:
            uncertainty = sf.model_uncertainty
            uncertainty_sampling = sf.model_uncertainty_sampling
        except Exception as e:  # never let the error bar kill the fit
            logger.warning("uncertainty map failed: %s: %s", type(e).__name__, e)
    dirty_image = imager.dirty_image(data)
    dirty_model = imager.dirty_image(model_vis)
    resid_dirty = imager.dirty_image(resid_vis)
    rms = imager.rms
    bf = beam_mod.fit_beam(imager.dirty_beam, geometry.pixel_scale)
    clean = beam_mod.restore(model_image, resid_dirty, bf, geometry.pixel_scale)
    points = list(getattr(sf, "points", []) or [])
    if points:
        # points are restored analytically at their fitted sub-pixel position,
        # which is exact: the restoring beam is smooth
        from .pointsource import restore_points

        clean = restore_points(
            clean.shape, geometry.pixel_scale, points, bf, existing=clean
        )

    pb = model_pbcor = None
    if pb_correction and dish:
        pb = primary_beam.primary_beam_map(
            geometry.shape_native, geometry.pixel_scale, frequency_hz, dish,
            pb_factor,
        )
        model_pbcor = primary_beam.pb_correct(model_image, pb)

    return ProductSet(
        model_mesh=model_mesh,
        model_image=model_image,
        uncertainty=uncertainty,
        uncertainty_sampling=uncertainty_sampling,
        dirty_image=dirty_image,
        dirty_model=dirty_model,
        residual_sigma=resid_dirty / rms,
        clean=clean,
        pb=pb,
        model_pbcor=model_pbcor,
        beam=bf,
        points=points,
        rms=rms,
        log_evidence=sf.log_evidence,
        chi_squared=sf.chi_squared,
        coefficient=sf.coefficient,
    )

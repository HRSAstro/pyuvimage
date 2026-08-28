"""Output products: FITS images with proper WCS, and a PNG summary figure.

All FITS products carry a celestial WCS (SIN projection about the phase
centre) so they overlay correctly in CASA / DS9 / CARTA.  Units follow radio
convention: the model is Jy/pixel, dirty/residual/reconvolved images are
Jy/beam,
the residual map is additionally normalised by the image RMS (stored in the
header as RMS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from .beam import BeamFit
from .grids import ImageGeometry

logger = logging.getLogger("pyuvimage")


def to_fits_orientation(native: np.ndarray) -> np.ndarray:
    """autoarray 'native' arrays put +y (North) in row 0; FITS wants row 0 at
    the bottom (South), to match the positive ``CDELT2`` `build_header`
    writes. Flip vertically -- **once**.

    Once is the whole story. On 26 Aug a second flip was added at write time
    to fix declinations that came out mirrored; two flips cancel, so the
    written file was then *unflipped* -- and it looked correct only because it
    cancelled a third mirror, in the imaging itself, where `v` had the wrong
    sign (see `uvdata.V_SIGN`). Fixing that sign made the double flip visible
    immediately. Pinned by `tests/test_sky_orientation.py`, which places a
    source from a written-out forward model rather than a round trip.
    """
    return np.flipud(np.asarray(native))


def build_header(
    n_pix: int,
    pixel_scale_arcsec: float,
    meta: dict,
    bunit: str,
    beam: BeamFit | None = None,
    frequencies_hz: np.ndarray | None = None,
    extra: dict | None = None,
) -> fits.Header:
    h = fits.Header()
    ra = meta.get("phase_centre_ra_deg")
    dec = meta.get("phase_centre_dec_deg")
    if ra is None or dec is None:
        ra, dec = 0.0, 0.0
        logger.warning(
            "no phase centre in dataset metadata; writing WCS centred on (0,0)"
        )
    cd = pixel_scale_arcsec / 3600.0
    # The reconstruction may be centred off the phase centre (--image-centre).
    # The image grid moved, so CRVAL has to move with it or every product is
    # astrometrically wrong by exactly the shift -- the one way this feature
    # could do real damage.
    offset = meta.get("image_centre_offset_arcsec")
    if offset:
        y0, x0 = float(offset[0]), float(offset[1])
        dec = float(dec) + y0 / 3600.0
        # column index increases with +x and CDELT1 is negative, so +x is
        # decreasing RA
        cosd = max(np.cos(np.radians(float(dec))), 1e-6)
        ra = float(ra) - (x0 / 3600.0) / cosd
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CUNIT1"] = h["CUNIT2"] = "deg"
    h["CRVAL1"], h["CRVAL2"] = float(ra), float(dec)
    if offset:
        h["IMCENOFF"] = (
            f"{offset[0]:.4f},{offset[1]:.4f}",
            "image centre offset (y,x) arcsec from phase centre",
        )
    h["CRPIX1"] = h["CRPIX2"] = (n_pix + 1) / 2.0
    h["CDELT1"], h["CDELT2"] = -cd, cd
    h["RADESYS"] = "ICRS"
    h["BUNIT"] = bunit
    if frequencies_hz is not None and len(frequencies_hz) > 1:
        f = np.asarray(frequencies_hz, dtype=float)
        h["CTYPE3"] = "FREQ"
        h["CUNIT3"] = "Hz"
        h["CRPIX3"] = 1.0
        h["CRVAL3"] = f[0]
        steps = np.diff(f)
        h["CDELT3"] = float(steps[0])
        # A linear frequency axis is a lie for a cube built from several
        # spectral windows: the channels are not evenly spaced, so CRVAL3 +
        # n*CDELT3 puts later planes at frequencies that were never observed.
        # Record the truth per plane and say so, rather than quietly shipping
        # a WCS that mislabels the data.
        if not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
            h["CDELT3"] = float(np.median(steps))
            h["FREQIRR"] = (
                True, "channel spacing is irregular; see FRQnnnn / .json"
            )
            for i, freq in enumerate(f):
                if i < 999:
                    h[f"FRQ{i:04d}"] = (float(freq), f"plane {i + 1} freq [Hz]")
            # logged once per run by write_products, not once per file
    elif frequencies_hz is not None:
        h["RESTFRQ"] = float(np.mean(frequencies_hz))
    if beam is not None:
        h["BMAJ"] = beam.bmaj_arcsec / 3600.0
        h["BMIN"] = beam.bmin_arcsec / 3600.0
        h["BPA"] = beam.bpa_deg
    h["TELESCOP"] = str(meta.get("telescope", "unknown"))
    h["ORIGIN"] = "pyuvimage"
    for k, v in (extra or {}).items():
        value = v[0] if isinstance(v, tuple) else v
        if isinstance(value, float) and not np.isfinite(value):
            # FITS headers cannot hold inf/nan; record the failure instead
            v = ("undefined", v[1]) if isinstance(v, tuple) else "undefined"
        h[k] = v
    return h


def upsample_model(mesh_image: np.ndarray, factor: int) -> np.ndarray:
    """Block-replicate a mesh image onto the (factor x finer) image grid,
    dividing by factor^2 so the unit stays Jy per (new) pixel."""
    return np.kron(mesh_image, np.ones((factor, factor))) / factor**2


@dataclass
class ProductSet:
    """In-memory products for one channel (or the MFS image)."""

    model_mesh: np.ndarray        # Jy / mesh pixel (native orientation)
    model_image: np.ndarray       # Jy / image pixel, same grid as the rest
    dirty_image: np.ndarray       # Jy / beam
    dirty_model: np.ndarray       # Jy / beam
    residual_sigma: np.ndarray    # (data - model) dirty / rms
    reconvolved: np.ndarray       # Jy / beam: model (x) beam + residuals
    uncertainty: np.ndarray | None    # total 1-sigma on the model, Jy/pixel
    pb: np.ndarray | None         # primary beam (image grid)
    model_pbcor: np.ndarray | None
    beam: BeamFit
    rms: float
    log_evidence: float
    chi_squared: float
    coefficient: float
    reconvolved_pbcor: np.ndarray | None = None
    points: list = None          # analytic point components, if fitted
    uncertainty_terms: dict | None = None   # the pieces of `uncertainty`



def model_with_points(p, pixel_scale: float) -> np.ndarray:
    """Extended model plus each point's flux in its nearest pixel.

    The image grid cannot hold a sub-pixel delta, so `model.fits` is
    flux-correct but positionally quantised; `point_sources.json` carries the
    exact fitted positions, and `model_reconvolved.fits` places them
    analytically. Shared by the FITS writer and the summary panel so the two
    can never disagree -- the summary used to drop points entirely.
    """
    img = np.array(p.model_image)
    if not p.points:
        return img
    n = img.shape[0]
    for pt in p.points:
        row = int(round((n - 1) / 2 - pt.d_dec / pixel_scale))
        col = int(round((n - 1) / 2 - pt.d_ra / pixel_scale))
        if 0 <= row < n and 0 <= col < img.shape[1]:
            img[row, col] += pt.flux
    return img


def write_products(
    products: list[ProductSet],
    geometry: ImageGeometry,
    meta: dict,
    frequencies_hz: np.ndarray,
    out_dir: str | Path,
    scan: dict | None = None,
    parameters: dict | None = None,
    overwrite: bool = True,
) -> dict:
    """Write the full FITS product set. `products` has one entry per channel
    (cube) or a single entry (mfs)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_cube = len(products) > 1
    freqs = np.asarray(frequencies_hz, dtype=float)
    beam0 = products[0].beam
    med_beam = BeamFit(
        bmaj_arcsec=float(np.median([p.beam.bmaj_arcsec for p in products])),
        bmin_arcsec=float(np.median([p.beam.bmin_arcsec for p in products])),
        bpa_deg=float(np.median([p.beam.bpa_deg for p in products])),
    )
    rms_all = float(np.median([p.rms for p in products]))

    def stack(attr: str, orient=True) -> np.ndarray:
        planes = [getattr(p, attr) for p in products]
        planes = [to_fits_orientation(x) if orient else x for x in planes]
        return np.stack(planes) if is_cube else planes[0]

    def hdr(n_pix, scale, bunit, beam=None, extra=None):
        return build_header(
            n_pix, scale, meta, bunit, beam=beam,
            frequencies_hz=freqs if is_cube else np.atleast_1d(np.mean(freqs)),
            extra=extra,
        )

    written = {}

    def w(name, data, header):
        path = out / name
        fits.writeto(path, np.asarray(data, dtype=np.float32), header,
                     overwrite=overwrite)
        written[name] = path

    n_img = geometry.shape_native[0]
    extra_common = {
        "REGCOEF": (products[0].coefficient, "source prior coefficient"),
        "LOGEV": (products[0].log_evidence, "log evidence (MFS/first plane)"),
        "PIXMESH": (geometry.mesh_pixel_scale, "model pixel scale [arcsec]"),
        "NYQUIST": (geometry.nyquist_pixel_scale, "Nyquist pixel scale [arcsec]"),
    }

    pts = products[0].points or []
    if pts:
        extra_common["NPOINTS"] = (len(pts), "analytic point components")
        extra_common["PTFLUX"] = (
            float(sum(p.flux for p in pts)), "total point flux [Jy]"
        )

    # All image products share one grid (the Nyquist-oversampled image grid)
    # so they overlay pixel-for-pixel.
    model_planes = [
        to_fits_orientation(model_with_points(p, geometry.pixel_scale))
        for p in products
    ]
    w("model.fits",
      np.stack(model_planes) if is_cube else model_planes[0],
      hdr(n_img, geometry.pixel_scale, "Jy/pixel", extra=extra_common))
    w("dirty_image.fits", stack("dirty_image"),
      hdr(n_img, geometry.pixel_scale, "Jy/beam", beam=med_beam))
    w("dirty_model.fits", stack("dirty_model"),
      hdr(n_img, geometry.pixel_scale, "Jy/beam", beam=med_beam))
    w("residual.fits", stack("residual_sigma"),
      hdr(n_img, geometry.pixel_scale, "sigma", beam=med_beam,
          extra={"RMS": (rms_all, "image-plane rms noise [Jy/beam]")}))
    w("model_reconvolved.fits", stack("reconvolved"),
      hdr(n_img, geometry.pixel_scale, "Jy/beam", beam=med_beam,
          extra={"RMS": (rms_all, "image-plane rms noise [Jy/beam]")}))
    if products[0].uncertainty is not None:
        terms = products[0].uncertainty_terms or {}
        unc_extra = {
            "ERRTYPE": ("total", "statistical + prior systematic"),
            "ERRSTAT": (terms.get("statistical_median", 0.0),
                        "median statistical 1-sigma [Jy/pixel]"),
            "ERRSYS": (terms.get("systematic_median", 0.0),
                       "median prior-strength systematic [Jy/pixel]"),
            "ERRSPRD": (terms.get("systematic_spread_dex", 0.0),
                        "systematic probed over +/- this many dex"),
            "ERRDEBL": (bool(terms.get("deblocked", False)),
                        "checkerboard replaced by its envelope"),
        }
        w("uncertainty.fits", stack("uncertainty"),
          hdr(n_img, geometry.pixel_scale, "Jy/pixel", extra=unc_extra))
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = [
                np.where(p.uncertainty > 0, p.model_image / p.uncertainty, 0.0)
                for p in products
            ]
        snr_stack = (np.stack([to_fits_orientation(s_) for s_ in snr])
                     if is_cube else to_fits_orientation(snr[0]))
        w("snr.fits", snr_stack, hdr(n_img, geometry.pixel_scale, ""))
    if products[0].pb is not None:
        w("pb.fits", stack("pb"),
          hdr(n_img, geometry.pixel_scale, ""))
        w("model_pbcor.fits", stack("model_pbcor"),
          hdr(n_img, geometry.pixel_scale, "Jy/pixel", extra=extra_common))
        if products[0].reconvolved_pbcor is not None:
            w("model_reconvolved_pbcor.fits", stack("reconvolved_pbcor"),
              hdr(n_img, geometry.pixel_scale, "Jy/beam", beam=med_beam,
                  extra={"RMS": (rms_all,
                                 "rms before PB correction [Jy/beam]")}))

    import json

    if pts:
        (out / "point_sources.json").write_text(json.dumps(
            {"points": [p.as_dict() for p in pts]}, indent=2))
    if is_cube:
        even = bool(
            len(freqs) < 3
            or np.allclose(np.diff(freqs), np.diff(freqs)[0], rtol=1e-6)
        )
        if not even:
            logger.warning(
                "the cube's channels are not evenly spaced (they span several "
                "spectral windows), so the linear FITS frequency axis cannot "
                "describe them: CDELT3 is the median step and is only "
                "indicative. The true per-plane frequencies are in each "
                "header as FRQ0000... and in frequencies.json."
            )
        (out / "frequencies.json").write_text(json.dumps(
            {"frequencies_hz": [float(x) for x in freqs],
             "evenly_spaced": even},
            indent=2))
    if scan is not None:
        (out / "prior_scan.json").write_text(json.dumps(scan, indent=2))
    if parameters is not None:
        (out / "fit_parameters.json").write_text(json.dumps(parameters, indent=2))

    # summary figure -- one row per channel in cube mode
    try:
        _summary_png(products, geometry, out / "summary.png", freqs)
        written["summary.png"] = out / "summary.png"
    except Exception as e:  # plotting must never kill a fit
        logger.warning("summary figure failed: %s", e)
    return written


# A cube can have hundreds of channels; past this many the figure stops being
# readable (and matplotlib stops being able to render it), so an evenly spaced
# subset is drawn and the title says so.
MAX_SUMMARY_ROWS = 24


def _summary_png_cube(rows, geometry, path, freqs, ext, n_panels, note):
    """One row per channel, on colour scales shared down each column.

    Shared scales are the point: per-plane autoscaling makes every channel
    look the same and hides the very thing a cube is for -- where the emission
    moves and how its brightness changes with velocity.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    stacks = {
        "dirty": [np.asarray(r.dirty_image, float) for r in rows],
        "model": [model_with_points(r, geometry.pixel_scale) for r in rows],
        "recon": [np.asarray(r.reconvolved, float) for r in rows],
        "resid": [np.asarray(r.residual_sigma, float) for r in rows],
    }
    if rows[0].uncertainty is not None:
        stacks["unc"] = [np.asarray(r.uncertainty, float) for r in rows]

    def lim(key, symmetric=False):
        a = np.concatenate([x[np.isfinite(x)].ravel() for x in stacks[key]])
        if a.size == 0:
            return None
        if symmetric:
            m = max(float(np.max(np.abs(a))), 1e-6)
            return (-m, m)
        return (0.0, float(np.max(a)))

    # the model panel is stretched to the *extended* model, as in the single
    # -channel figure, so a point component cannot saturate it
    ext_max = max(
        (float(np.nanmax(r.model_image)) for r in rows
         if np.isfinite(r.model_image).any()), default=0.0
    )
    cols = [
        ("dirty", "dirty image", "Jy/beam", "inferno", lim("dirty")),
        ("model", "model", "Jy/pixel", "inferno",
         (0.0, ext_max) if ext_max > 0 else None),
        ("recon", "model reconvolved", "Jy/beam", "inferno", lim("recon")),
        ("resid", "residual", r"$\sigma$", "RdBu_r", lim("resid", True)),
    ]
    if "unc" in stacks:
        cols.append(("unc", r"1$\sigma$ uncertainty", "Jy/pixel", "viridis",
                     lim("unc")))
    cols = cols[:n_panels]

    fig, axes = plt.subplots(
        n, len(cols), figsize=(3.6 * len(cols), 3.3 * n), squeeze=False,
        constrained_layout=True,
    )
    f0 = float(np.mean(freqs)) if freqs is not None else None
    for i, r in enumerate(rows):
        for j, (key, title, unit, cmap, clim) in enumerate(cols):
            ax = axes[i][j]
            im = ax.imshow(stacks[key][i], origin="upper", extent=ext, cmap=cmap)
            if clim is not None:
                im.set_clim(*clim)
            for pt in (r.points or []):
                ax.plot(pt.d_ra, pt.d_dec, "o", mfc="none", mec="cyan", ms=9,
                        mew=1.1)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if i == n - 1:
                ax.set_xlabel("dRA [arcsec]")
            else:
                ax.set_xticklabels([])
            if j == 0:
                label = f"channel {i + 1}"
                if freqs is not None:
                    v = 299792.458 * (f0 - freqs[i]) / f0
                    label = f"{freqs[i] / 1e9:.4f} GHz\n{v:+.0f} km/s"
                ax.set_ylabel(label, fontsize=9)
            else:
                ax.set_yticklabels([])
            if key == "resid":
                rmax = float(np.nanmax(np.abs(stacks["resid"][i])))
                ax.text(0.03, 0.03, rf"max {rmax:.1f}$\sigma$",
                        transform=ax.transAxes, fontsize=8, color="k",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))

    # one colour bar per column, spanning it, after every panel exists
    for j, (key, title, unit, cmap, clim) in enumerate(cols):
        cb = fig.colorbar(axes[0][j].images[0], ax=[axes[i][j] for i in range(n)],
                          fraction=0.035, pad=0.02, location="right")
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    b = rows[0].beam
    fig.suptitle(
        f"{n} channels, colour scales shared down each column  |  "
        f"beam {b.bmaj_arcsec:.3g}\" x {b.bmin_arcsec:.3g}\" "
        f"pa {b.bpa_deg:.0f} deg  |  prior coefficient "
        f"{rows[0].coefficient:.3g}{note}",
        fontsize=11,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)



def _summary_png(
    products, geometry: ImageGeometry, path: Path, frequencies_hz=None
) -> None:
    """One row of panels per channel; a single row for an MFS fit.

    Cube mode used to draw `products[0]` and nothing else, so a summary of an
    eight-channel cube was a picture of one channel -- indistinguishable, at a
    glance, from the MFS image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(products, ProductSet):
        products = [products]
    rows = list(products)
    freqs = (np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
             if frequencies_hz is not None else None)
    note = ""
    if len(rows) > MAX_SUMMARY_ROWS:
        keep = np.linspace(0, len(rows) - 1, MAX_SUMMARY_ROWS).astype(int)
        note = (f"  |  showing {MAX_SUMMARY_ROWS} of {len(rows)} channels, "
                "evenly spaced")
        rows = [rows[i] for i in keep]
        freqs = freqs[keep] if freqs is not None and len(freqs) > max(keep) else None
    if freqs is not None and len(freqs) != len(rows):
        freqs = None

    p = rows[0]
    half = geometry.fov_arcsec / 2.0
    ext = [half, -half, -half, half]  # RA increases leftward
    n_panels = 5 if p.uncertainty is not None else 4
    if len(rows) > 1:
        _summary_png_cube(rows, geometry, path, freqs, ext, n_panels, note)
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(4.25 * n_panels, 4.4),
                             constrained_layout=True)

    resid = np.asarray(p.residual_sigma)
    rmax = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else 1.0
    rmax = max(rmax, 1e-6)

    # The model panel shows what model.fits holds, points included -- but a
    # point carries its whole flux in one pixel, so on a linear scale set by
    # the image maximum it saturates the panel and the extended emission
    # disappears entirely.  Stretch to the *extended* model instead and mark
    # each point, so both are visible.
    model_panel = model_with_points(p, geometry.pixel_scale)
    n_pt = len(p.points or [])
    ext_max = float(np.nanmax(p.model_image)) if np.isfinite(
        p.model_image).any() else 0.0
    model_clim = (0.0, ext_max) if ext_max > 0 else None
    model_title = "model"
    if n_pt:
        model_title += (f" (+{n_pt} point source{'s' if n_pt > 1 else ''}, "
                        "circled; colour scale set by the extended model)")

    panels = [
        (p.dirty_image, "dirty image", "Jy/beam", "inferno", None),
        (model_panel, model_title, "Jy/pixel", "inferno", model_clim),
        (p.reconvolved, "model reconvolved", "Jy/beam", "inferno", None),
        (resid, "residual", r"$\sigma$", "RdBu_r", (-rmax, rmax)),
    ]
    if p.uncertainty is not None:
        panels.append((p.uncertainty, "1$\\sigma$ uncertainty", "Jy/pixel",
                       "viridis", None))
    for ax, (img, title, unit, cmap, clim) in zip(axes, panels):
        im = ax.imshow(np.asarray(img), origin="upper", extent=ext, cmap=cmap)
        if clim is not None:
            im.set_clim(*clim)
        for pt in (p.points or []):
            ax.plot(pt.d_ra, pt.d_dec, "o", mfc="none", mec="cyan", ms=11,
                    mew=1.3)
        ax.set_title(title, fontsize=10 if len(title) > 24 else 11)
        ax.set_xlabel("dRA [arcsec]")
        if ax is axes[0]:
            ax.set_ylabel("dDec [arcsec]")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # the residual panel is the one people judge the fit by: state its scale
    peak_sigma = float(np.nanmax(p.reconvolved)) / p.rms if p.rms > 0 else 0.0
    pct = 100.0 * rmax / peak_sigma if peak_sigma > 0 else float("nan")
    axes[3].set_title(
        f"residual  (peak {rmax:.1f}$\\sigma$ = {pct:.2f}% of the source "
        f"peak, rms {np.nanstd(resid):.2f}$\\sigma$)", fontsize=10
    )
    fig.suptitle(
        f"beam {p.beam.bmaj_arcsec:.3g}\" x {p.beam.bmin_arcsec:.3g}\" "
        f"pa {p.beam.bpa_deg:.0f} deg | rms {p.rms:.3g} Jy/beam | "
        f"prior coefficient {p.coefficient:.3g} | chi2 {p.chi_squared:.5g} | "
        f"peak {peak_sigma:.0f}$\\sigma$",
        fontsize=10,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)

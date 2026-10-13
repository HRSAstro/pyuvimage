"""User-supplied emission masks (`--mask`).

A mask says where the sky may hold emission. With it the fit is *hard*
masked: the model is zero outside (`api` drops or zeroes the mesh there), so
structure the data cannot attribute elsewhere -- above all the large-scale
sidelobes of a bright extended source, which a model with positivity will
otherwise absorb as broad emission across the field
(docs/masks.md) -- has to be explained by emission inside it.

Two ways to make one:

* a **region file**: CASA's CRTF (``.crtf``, as written by the CASA viewer
  or CARTA) or a DS9 ``.reg`` file, read with `regions` (an optional
  dependency: ``pip install regions``). Sky regions are placed through the
  fit grid's WCS; pixel regions are taken as pixels of that grid in FITS
  orientation. Regions marked as exclusions are cut out of the union of the
  rest.
* a **FITS mask**: any image with a celestial WCS, non-zero where emission is
  allowed -- one drawn with ``pyuvimage mask`` (autogalaxy's GUI painter over
  a dirty image), or one exported from CASA / CARTA. It is resampled onto the
  fit grid by WCS, nearest pixel.

Masks are handled here in **native** orientation (row 0 = +y, North), as
every array inside the fit; FITS files hold the vertical flip
(`products.to_fits_orientation`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("pyuvimage")

REGION_SUFFIXES = (".crtf", ".reg")
FITS_SUFFIXES = (".fits", ".fit", ".fts", ".mask")


def fit_wcs(shape_native: tuple[int, int], pixel_scale: float, meta: dict):
    """The celestial WCS of the fit's image grid, in FITS orientation: the
    header every product is written with (`products.build_header`)."""
    from astropy.wcs import WCS

    from .products import build_header

    ny, nx = (int(v) for v in shape_native)
    if ny != nx:
        raise ValueError("pyuvimage image grids are square")
    h = build_header(nx, float(pixel_scale), meta, "")
    return WCS(h).celestial


def load(path: str | Path, shape_native, pixel_scale: float, meta: dict) -> np.ndarray:
    """The mask at ``path`` on the fit grid: a native-orientation boolean
    array, True where emission is allowed."""
    path = Path(path)
    wcs = fit_wcs(shape_native, pixel_scale, meta)
    mask = np.flipud(onto(path, wcs, tuple(shape_native)))
    frac = float(mask.mean())
    if not mask.any():
        raise ValueError(f"the mask {path.name} covers none of the fitted field")
    logger.info("mask %s: %.1f%% of the %d x %d field may hold emission",
                path.name, 100 * frac, shape_native[0], shape_native[1])
    return mask


def onto(path: str | Path, wcs, shape) -> np.ndarray:
    """The mask at ``path`` on the grid of ``wcs`` (FITS orientation,
    ``shape`` = (ny, nx)), True where emission is allowed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mask file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in REGION_SUFFIXES:
        return _from_regions(path, wcs, tuple(shape))
    if suffix in FITS_SUFFIXES:
        return _from_fits(path, wcs, tuple(shape))
    raise ValueError(
        f"unrecognised mask file {path.name!r}: give a CASA region file "
        f"(.crtf), a DS9 region file (.reg) or a FITS mask (.fits)")


def _from_regions(path: Path, wcs, shape) -> np.ndarray:
    try:
        from regions import PixelRegion, Regions
    except ImportError as e:                    # pragma: no cover - optional
        raise ImportError(
            "reading region files needs the `regions` package: "
            "pip install regions") from e
    fmt = "crtf" if path.suffix.lower() == ".crtf" else "ds9"
    regs = Regions.read(str(path), format=fmt)
    if len(regs) == 0:
        raise ValueError(f"no regions in {path}")
    include = np.zeros(shape, dtype=bool)
    exclude = np.zeros(shape, dtype=bool)
    n_inc = 0
    for reg in regs:
        pix = reg if isinstance(reg, PixelRegion) else reg.to_pixel(wcs)
        m = pix.to_mask(mode="center").to_image(shape)
        if m is None:                           # wholly outside the field
            logger.warning("mask: a %s region lies outside the fitted field",
                           type(reg).__name__)
            continue
        m = m.astype(bool)
        if reg.meta.get("include", True):
            include |= m
            n_inc += 1
        else:
            exclude |= m
    if n_inc == 0:
        raise ValueError(f"{path} holds only exclusion regions")
    return include & ~exclude


def _from_fits(path: Path, wcs, shape) -> np.ndarray:
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None)
        data = np.asarray(hdu.data)
        src = WCS(hdu.header).celestial
    # a CASA mask is (stokes, freq, y, x): any plane with emission allowed
    while data.ndim > 2:
        data = np.any(data != 0, axis=0)
    data = np.nan_to_num(np.asarray(data, dtype=float)) != 0
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs.pixel_to_world(xx.ravel(), yy.ravel())
    sx, sy = src.world_to_pixel(world)
    ix, iy = np.rint(sx).astype(int), np.rint(sy).astype(int)
    ok = (ix >= 0) & (ix < data.shape[1]) & (iy >= 0) & (iy < data.shape[0])
    out = np.zeros(ny * nx, dtype=bool)
    out[ok] = data[iy[ok], ix[ok]]
    return out.reshape(ny, nx)


def write(path: str | Path, mask_native: np.ndarray, pixel_scale: float, meta: dict) -> Path:
    """Write a native-orientation boolean mask as a FITS image (1 = emission
    allowed) with the fit grid's WCS."""
    from astropy.io import fits

    from .products import build_header

    mask_native = np.asarray(mask_native, dtype=bool)
    h = build_header(mask_native.shape[1], float(pixel_scale), meta, "")
    h["COMMENT"] = "pyuvimage emission mask: 1 = emission allowed"
    path = Path(path)
    fits.PrimaryHDU(np.flipud(mask_native).astype(np.int16), header=h).writeto(
        path, overwrite=True)
    return path


def draw(image_path: str | Path, out: str | Path, proposal: str | Path | None = None,
         backend: str = "TkAgg") -> Path:
    """Paint a mask over a FITS image (a dirty image, e.g. a fit's
    ``dirty_image.fits``) with autogalaxy's `Scribbler`, and write it to
    ``out`` on that image's grid and WCS.

    Brush '1' adds emission, '2' erases, '=' / '-' resize, 'z' undoes, Esc
    finishes. ``proposal`` starts from an existing mask (FITS or region file).
    """
    import autogalaxy as ag
    from astropy.io import fits
    from astropy.wcs import WCS
    from autogalaxy.gui.scribbler import Scribbler

    with fits.open(image_path) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None)
        img = np.asarray(hdu.data, dtype=float)
        header = hdu.header
    while img.ndim > 2:
        img = img[0]
    img = np.nan_to_num(img)
    from astropy.wcs.utils import proj_plane_pixel_scales

    wcs = WCS(header).celestial
    scale = float(abs(proj_plane_pixel_scales(wcs)[1])) * 3600.0
    native = np.flipud(img)
    array = ag.Array2D.no_mask(values=native, pixel_scales=scale)
    start = None
    if proposal is not None:
        start = np.flipud(onto(proposal, wcs, img.shape))
    vmax = float(np.nanpercentile(native, 99.9))
    painter = Scribbler(image=array, cmap="inferno", vmin=-0.1 * vmax, vmax=vmax,
                        backend=backend, proposal=start)
    drawn = np.asarray(painter.mask_from(), dtype=bool)
    if not drawn.any():
        raise ValueError("nothing was painted: no mask written")
    out = Path(out)
    fits.PrimaryHDU(np.flipud(drawn).astype(np.int16), header=_mask_header(header)).writeto(
        out, overwrite=True)
    logger.info("mask written to %s (%.1f%% of the image)", out, 100 * drawn.mean())
    return out


def _mask_header(header):
    from astropy.wcs import WCS

    h = WCS(header).celestial.to_header()
    h["BUNIT"] = ""
    h["COMMENT"] = "pyuvimage emission mask: 1 = emission allowed"
    return h

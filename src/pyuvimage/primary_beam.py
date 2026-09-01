"""Primary beam model and correction.

v1 uses a Gaussian primary beam with FWHM = pb_factor * lambda / D radians
(pb_factor ~ 1.13 for a uniformly illuminated 12-m ALMA antenna; 1.02-1.22
depending on illumination taper).  The reconstruction is of the *apparent*
sky (PB x I); dividing by the PB gives the flux-correct image, exactly as
CASA's pbcor does.
"""

from __future__ import annotations

import numpy as np

C_M_S = 299792458.0
RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0

DEFAULT_PB_FACTOR = 1.13
DEFAULT_PB_CUTOFF = 0.1


def pb_fwhm_arcsec(
    frequency_hz: float, dish_diameter_m: float, pb_factor: float = DEFAULT_PB_FACTOR
) -> float:
    lam = C_M_S / frequency_hz
    return pb_factor * lam / dish_diameter_m * RAD_TO_ARCSEC


def primary_beam_map(
    shape: tuple[int, int],
    pixel_scale: float,
    frequency_hz: float,
    dish_diameter_m: float,
    pb_factor: float = DEFAULT_PB_FACTOR,
    image_centre_offset_arcsec: tuple[float, float] | None = None,
) -> np.ndarray:
    """Gaussian PB, peak 1 at the **phase centre** -- where the dish pointed.

    The primary beam is a property of the instrument, not of the image: it is
    centred on the pointing, and that is the phase centre of the data. When
    the reconstruction has been recentred (``--image-centre``) the grid centre
    is no longer the phase centre, and the PB peak has to move off the grid
    centre by the same offset in the opposite direction. Without that, a
    source 4 arcsec off the pointing at 245 GHz (12 m dish, PB FWHM ~24") was
    "corrected" as if it sat at PB = 1.0 when the true response there is 0.93
    -- both ALMA datasets this feature was built for.

    ``image_centre_offset_arcsec`` is ``(y0, x0)`` as `uvdata.shift_image_centre`
    records it in ``meta["image_centre_offset_arcsec"]``: the image centre's
    offset from the phase centre, ``y0`` toward north (+Dec) and ``x0`` along
    image +x (west, decreasing RA). On the native grid row 0 is north and the
    column index increases with +x, so the phase centre sits at
    ``row = cy + y0 / pixel_scale, col = cx - x0 / pixel_scale``.
    """
    fwhm = pb_fwhm_arcsec(frequency_hz, dish_diameter_m, pb_factor)
    sigma_pix = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pixel_scale
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    if image_centre_offset_arcsec:
        y0, x0 = (float(v) for v in image_centre_offset_arcsec)
        cy += y0 / pixel_scale
        cx -= x0 / pixel_scale
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-0.5 * r2 / sigma_pix**2)


def pb_correct(
    image: np.ndarray, pb: np.ndarray, cutoff: float = DEFAULT_PB_CUTOFF
) -> np.ndarray:
    """image / PB, blanked (NaN) where the PB is below `cutoff`."""
    out = np.full_like(np.asarray(image, dtype=float), np.nan)
    good = pb >= cutoff
    out[good] = image[good] / pb[good]
    return out

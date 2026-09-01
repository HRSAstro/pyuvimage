"""Dirty images, the dirty beam, and the CLEAN-style restored image.

All dirty images use natural weighting (w = 1/sigma^2) and are normalised by
sum(w), the analytic peak of the dirty beam at zero offset -- which makes
their unit Jy/beam, directly comparable to CASA products, and makes the
image-plane RMS exactly sigma_im = sqrt(sum w^2 sigma^2) / sum w = 1/sqrt(sum w).

Normalising by the *sampled* peak instead, as this module did until 1 Sep
2026, is wrong on every production grid: `resolve_geometry` always produces an
even number of pixels, so the phase centre falls between four pixel centres
and the sampled peak sits half a pixel off it, 5-9% below sum(w) (0.92 on the
mock, 0.945 on the demo). Dirty-image values were then that much *high*
relative to `rms`, and the structure ratio -- the `--criterion structure`
statistic -- read 1.06-1.09 for a residual that was actually white.

Three frames meet here and the sign conventions are easy to get wrong. On
the native image array row 0 is north and the column index increases with
image +x, which is west (decreasing RA). A position angle is measured east of
north (the CASA convention), so every Gaussian in this module is written in
north-up, east-left coordinates: dy = (cy - row), dx = (col - cx) points west,
and the major axis at angle theta points along (east, north) = (sin, cos).
`pointsource.restore_points` uses the same expression and must keep doing so,
or point sources and extended emission are restored with mirrored beams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

import autogalaxy as ag

logger = logging.getLogger("pyuvimage")

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class BeamFit:
    """Elliptical-Gaussian fit to the dirty beam main lobe."""

    bmaj_arcsec: float  # FWHM major axis
    bmin_arcsec: float  # FWHM minor axis
    bpa_deg: float      # position angle, East of North (CASA convention)

    def beam_area_pixels(self, pixel_scale: float) -> float:
        """Beam solid angle in units of image pixels: 2 pi sx sy / dpix^2."""
        sx = self.bmin_arcsec / SIGMA_TO_FWHM
        sy = self.bmaj_arcsec / SIGMA_TO_FWHM
        return 2.0 * np.pi * sx * sy / pixel_scale**2


class DirtyImager:
    """Natural-weighting dirty images through the dataset's transformer."""

    def __init__(self, dataset: ag.Interferometer):
        self.dataset = dataset
        self.transformer = dataset.transformer
        sigma = np.asarray(dataset.noise_map)
        # scalar weight per visibility from the mean re/im variance
        var = 0.5 * (sigma.real**2 + sigma.imag**2)
        self.weights = 1.0 / var
        self._beam_native, self._norm = self._make_beam()

    def _image_from(self, values: np.ndarray) -> np.ndarray:
        vis = ag.Visibilities(np.asarray(values, dtype=complex))
        img = self.transformer.image_from(visibilities=vis)
        return np.asarray(img.native)

    def _make_beam(self) -> tuple[np.ndarray, float]:
        """The dirty beam and the normalisation every image is divided by.

        The normalisation is sum(w): the value the dirty beam takes at exactly
        zero offset, where every visibility's phase is zero. It is *not* the
        sampled maximum -- on an even grid the phase centre lies between pixel
        centres, so the sampled peak is half a pixel off and 5-9% low, and
        dividing by it made every dirty image that much too bright next to
        `rms`, which has always assumed sum(w). With sum(w) a 1 Jy point at
        the phase centre reads 1 Jy/beam, and `rms` is exact.
        """
        raw = self._image_from(self.weights.astype(complex))
        norm = float(np.sum(self.weights))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("dirty beam has non-positive weight sum")
        return raw / norm, norm

    @property
    def dirty_beam(self) -> np.ndarray:
        """Dirty beam (PSF) on the image grid, unit response at zero offset.

        Its sampled maximum is a little below 1 on an even grid, because no
        pixel centre sits exactly on the phase centre. That is the truth of
        the sampling, not an error, and `fit_beam` fits the amplitude freely.
        """
        return self._beam_native

    def dirty_image(self, visibilities: np.ndarray) -> np.ndarray:
        """Naturally weighted dirty image [Jy/beam] of arbitrary visibilities."""
        return self._image_from(visibilities * self.weights) / self._norm

    @property
    def rms(self) -> float:
        """Analytic image-plane RMS for natural weighting [Jy/beam].

        The dirty image is D(x) = sum_k w_k Re[V_k e^{2 pi i u_k.x}] / sum_k w_k
        (normalised so the dirty beam peaks at 1), so with w = 1/var,
        Var[D] = sum w^2 var / (sum w)^2 = 1 / sum w.
        Verified against `rms_empirical`.
        """
        return float(1.0 / np.sqrt(np.sum(self.weights)))

    @property
    def inside(self) -> np.ndarray:
        """Public alias of `_inside`: where the image plane is defined."""
        return self._inside

    @property
    def _inside(self) -> np.ndarray:
        """Boolean map of pixels inside the real-space mask (native shape)."""
        return ~np.asarray(self.dataset.real_space_mask.native if hasattr(
            self.dataset.real_space_mask, "native"
        ) else self.dataset.real_space_mask).astype(bool)

    def rms_empirical(self, n_draws: int = 16, seed: int = 0) -> float:
        """Monte-Carlo cross-check of `rms`.

        Only unmasked pixels are used: the transformer returns zeros outside
        the mask, which would otherwise bias the estimate low.
        """
        rng = np.random.default_rng(seed)
        sigma = np.asarray(self.dataset.noise_map)
        inside = self._inside
        vals = []
        for _ in range(n_draws):
            noise = rng.normal(size=sigma.shape) * sigma.real + 1j * (
                rng.normal(size=sigma.shape) * sigma.imag
            )
            img = np.asarray(self.dirty_image(noise))[inside]
            vals.append(np.std(img))
        return float(np.median(vals))


def fit_beam(
    dirty_beam: np.ndarray, pixel_scale: float, window_frac: float = 0.35
) -> BeamFit:
    """Fit an elliptical Gaussian to the main lobe of the dirty beam."""
    beam = np.nan_to_num(np.asarray(dirty_beam, dtype=float))
    ny, nx = beam.shape
    cy, cx = np.unravel_index(np.nanargmax(beam), beam.shape)
    half = max(3, int(round(min(ny, nx) * window_frac / 2)))
    y0, y1 = max(cy - half, 0), min(cy + half + 1, ny)
    x0, x1 = max(cx - half, 0), min(cx + half + 1, nx)
    patch = beam[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)

    # Only fit the positive main lobe.
    m = patch > 0.1
    if m.sum() < 8:
        m = patch > 0

    def model(p):
        # north-up coordinates: dy points north (row 0 is north), dx points
        # west (+x). See the module docstring; `gaussian_kernel` and
        # `pointsource.restore_points` use the identical expression, which is
        # what makes the fitted angle the one they restore with.
        amp, px, py, sx, sy, theta = p
        ct, st = np.cos(theta), np.sin(theta)
        dx, dy = xx - px, py - yy
        xr = dx * ct + dy * st
        yr = -dx * st + dy * ct
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    def resid(p):
        return (model(p) - patch)[m]

    p0 = [1.0, float(cx), float(cy), 2.0, 2.0, 0.0]
    sol = least_squares(resid, p0, method="lm", max_nfev=5000)
    amp, px, py, sx, sy, theta = sol.x
    sx, sy = abs(sx), abs(sy)
    # Position angle east of north. The yr axis lies along (dx, dy) =
    # (-sin, cos), i.e. (east, north) = (sin, cos): angle theta east of north.
    # The xr axis is 90 degrees from it. Until 1 Sep 2026 the model used
    # row-down dy and the returned angle was the sky PA negated; no test
    # checked the sign, and the comment claiming CASA agreement was untrue.
    if sy >= sx:
        smaj, smin = sy, sx
        pa = np.degrees(theta)
    else:
        smaj, smin = sx, sy
        pa = np.degrees(theta) - 90.0
    pa = ((pa + 90.0) % 180.0) - 90.0
    return BeamFit(
        bmaj_arcsec=float(smaj * SIGMA_TO_FWHM * pixel_scale),
        bmin_arcsec=float(smin * SIGMA_TO_FWHM * pixel_scale),
        bpa_deg=float(pa),
    )


def gaussian_kernel(beam: BeamFit, pixel_scale: float, shape: tuple[int, int]) -> np.ndarray:
    """Peak-normalised restoring beam, centred where `fftconvolve` expects it.

    `restore` convolves with ``mode="same"``, which centres the output on
    kernel index ``(n - 1) // 2`` -- an integer pixel. The kernel has to be
    evaluated about that same pixel, not about the geometric centre
    ``(n - 1) / 2``: the two agree for odd ``n`` and differ by half a pixel
    for even ``n``, and `resolve_geometry` always produces an even number of
    pixels. Until 1 Sep 2026 this used the geometric centre, so on every
    production grid the restored extended emission was shifted by (+0.5,
    +0.5) pixels relative to the residual and to the point sources, which are
    not convolved. The tests used odd grids and never saw it.
    """
    ny, nx = shape
    cy, cx = (ny - 1) // 2, (nx - 1) // 2
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    smaj = beam.bmaj_arcsec / SIGMA_TO_FWHM / pixel_scale
    smin = beam.bmin_arcsec / SIGMA_TO_FWHM / pixel_scale
    theta = np.radians(beam.bpa_deg)
    ct, st = np.cos(theta), np.sin(theta)
    # north-up, like `fit_beam.model` and `pointsource.restore_points`
    dx, dy = xx - cx, cy - yy
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    return np.exp(-0.5 * ((xr / smin) ** 2 + (yr / smaj) ** 2))


def restore(
    model_image_jy_pix: np.ndarray,
    residual_dirty_jy_beam: np.ndarray,
    beam: BeamFit,
    pixel_scale: float,
) -> np.ndarray:
    """CLEAN-style restored image [Jy/beam]:

    (model [Jy/pix] convolved with the peak-normalised restoring beam)
    + residual dirty image [Jy/beam].
    """
    kernel = gaussian_kernel(beam, pixel_scale, model_image_jy_pix.shape)
    restored = fftconvolve(
        np.nan_to_num(model_image_jy_pix), kernel, mode="same"
    )
    return restored + np.nan_to_num(residual_dirty_jy_beam)


def wide_field_dirty_image(
    uv_wavelengths: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    fov_arcsec: float,
    n_pixels: int = 96,
    chunk: int = 4096,
) -> tuple[np.ndarray, float]:
    """A naturally weighted dirty image by direct summation, in chunks.

    Deliberately does not go through a transformer. The point is to look at a
    field far larger than the one being reconstructed -- to find where the
    emission actually is -- and building an autoarray dataset for that would
    allocate the ``n_pixels^2 x n_vis`` temporary the whole exercise is meant
    to avoid. Here memory is bounded by ``chunk``.

    On a real ALMA dataset (164k visibilities, 96x96 pixels) this takes about
    a minute and a few hundred MB.

    The returned array follows the same convention as every other native image
    here: row 0 is North (+y), column index increases with +x, so
    `envelope.peak_offset_arcsec` reads it directly.

    Returns (image [Jy/beam], analytic rms [Jy/beam]).
    """
    uv = np.asarray(uv_wavelengths, dtype=float)
    vis = np.asarray(data)
    sig = np.asarray(noise)
    w = 1.0 / (0.5 * (sig.real**2 + sig.imag**2))
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("no usable weights for the wide-field dirty image")
    arcsec = np.pi / 180.0 / 3600.0
    step = fov_arcsec / n_pixels
    coord = (np.arange(n_pixels) - (n_pixels - 1) / 2.0) * step * arcsec
    x_of_col = coord                 # +x with increasing column
    y_of_row = coord[::-1]           # +y (North) at row 0
    img = np.zeros((n_pixels, n_pixels))
    for s in range(0, len(vis), chunk):
        u = uv[s:s + chunk, 0]
        v = uv[s:s + chunk, 1]
        d = vis[s:s + chunk]
        wc = w[s:s + chunk]
        for r, yy in enumerate(y_of_row):
            ph = 2.0 * np.pi * (u[None, :] * x_of_col[:, None] + v[None, :] * yy)
            # Re[V e^{+i phi}] -- the sign that matches `DirtyImager`; the
            # other one returns the image flipped in both axes, which is easy
            # to miss on a centrally peaked source and wrong on every other
            img[r] += (wc * (d.real * np.cos(ph) - d.imag * np.sin(ph))).sum(1)
    return img / total, float(1.0 / np.sqrt(total))


def wide_field_image(
    uv_wavelengths: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    fov_arcsec: float,
    n_pixels: int = 96,
    transformer: str = "auto",
) -> tuple[np.ndarray, float]:
    """Dirty image of a field much wider than the one being reconstructed.

    Used to find where the emission actually is before committing to an
    ``--fov``. Takes the NUFFT when one is available -- a few seconds -- and
    falls back to direct summation otherwise.

    The fallback is exact but costs ``n_pixels^2 x n_vis``: about 1.4 billion
    operations for a 96x96 field over 148k visibilities, which is ~90 s on a
    fast machine and several minutes on a slow one. That is long enough to
    look like a hang, so this reports what it is doing before starting.

    Returns (image [Jy/beam], rms [Jy/beam]) on the native grid convention.
    """
    from . import fitting
    from .grids import resolve_geometry

    uv = np.asarray(uv_wavelengths, dtype=float)
    n_vis = len(uv)
    cls = fitting.resolve_transformer(
        n_vis, transformer=transformer, n_image_pixels=n_pixels * n_pixels
    )
    if cls is not ag.TransformerDFT or n_vis * n_pixels**2 <= fitting.DFT_MAX_PRODUCT:
        b_max = float(np.max(np.hypot(uv[:, 0], uv[:, 1])))
        half = max(1, n_pixels // 2)
        geometry = resolve_geometry(
            fov_arcsec=fov_arcsec,
            max_baseline_wavelengths=b_max,
            mesh_shape=(half, half),
            oversample=2,
        )
        dataset = fitting.make_dataset(
            uv, data, noise, geometry, transformer=transformer
        )
        imager = DirtyImager(dataset)
        return np.asarray(imager.dirty_image(np.asarray(data))), imager.rms

    logger.info(
        "  no NUFFT available, so the wide-field image is a direct sum over "
        "%d visibilities: this takes a couple of minutes. `pip install "
        "pynufft` makes it seconds.", n_vis,
    )
    return wide_field_dirty_image(
        uv, data, noise, fov_arcsec, n_pixels=n_pixels
    )

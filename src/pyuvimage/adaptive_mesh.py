"""Irregular (Delaunay) meshes: brightness-adaptive Hilbert points, and masks.

The rectangular mesh spends its pixels evenly over the field. Two things here
spend them where the emission is instead, on a Delaunay triangulation
(autoarray's ``Delaunay`` mesh, barycentric interpolation onto the image):

* **Hilbert points** (``--mesh-type hilbert``), after PyAutoLens: a cheap
  first-pass model is read along a Hilbert space-filling curve over the image
  and points are drawn with density following its brightness, so a compact
  source in a wide field needs far fewer mesh pixels for the same fidelity
  (docs/adaptive-mesh.md: a 1064-point mesh matched a 4096-pixel rectangle on
  a compact mock). Unlike PyAutoLens, which weights by ``|adapt|``, negative
  values are clipped: an interferometric first pass has negative bowls and
  sidelobes, and ``|adapt|`` put dense points on them.
* **An emission mask** (``--mask``, `mask.py`): mesh points only inside it,
  and a ring of points held at zero around it (autoarray's
  ``zeroed_pixels``), so the model is zero outside. With a rectangular
  ``--mesh-type`` the points are the rectangle's own centres that fall in the
  mask; with Hilbert they are drawn inside the mask only.

The zeroed points are held at zero by the non-negative solver (autoarray
applies ``zeroed_pixels`` there), so a mask needs positivity on these paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("pyuvimage")

MESH_TYPES = ("rectangular", "hilbert")

#: first-pass mesh for the Hilbert adapt image: this fraction of the nominal
#: mesh per side, within [`ADAPT_MESH_MIN`, `ADAPT_MESH_MAX`]
ADAPT_MESH_FRACTION = 0.25
ADAPT_MESH_MIN = 16
ADAPT_MESH_MAX = 48

#: Hilbert weight map: (brightness / peak)^power, floored -- PyAutoLens's
#: ``weight_power`` and ``weight_floor``
HILBERT_POWER = 1.0
HILBERT_FLOOR = 0.02


@dataclass(eq=False)
class IrregularMesh:
    """Mesh points (y, x) [arcsec, the image grid's frame]; the last
    ``n_zeroed`` are held at zero (a mask's boundary ring)."""

    points: np.ndarray
    n_zeroed: int = 0
    kind: str = "hilbert"
    #: where the model can act (a mask dilated by a beam), native image
    #: grid: the structure ratio is measured here, since the residual outside
    #: a mask holds what the model is forbidden to fit (`fitting`)
    region: np.ndarray | None = None

    @property
    def pixels(self) -> int:
        return int(len(self.points))

    @property
    def n_free(self) -> int:
        return self.pixels - int(self.n_zeroed)

    def describe(self) -> str:
        z = f", {self.n_zeroed} held at zero round the mask" if self.n_zeroed else ""
        return f"{self.kind} Delaunay mesh of {self.n_free} points{z}"


# ---------------------------------------------------------------------------
# geometry helpers: native arrays (row 0 = +y) <-> (y, x) arcsec
# ---------------------------------------------------------------------------

def _pixel_yx(rows, cols, shape, pixel_scale):
    ny, nx = shape
    return np.stack([((ny - 1) / 2.0 - np.asarray(rows, float)) * pixel_scale,
                     (np.asarray(cols, float) - (nx - 1) / 2.0) * pixel_scale], axis=1)


def _field_ring(shape, pixel_scale, spacing):
    """Points on a square just outside the field, so every image pixel lies
    inside the triangulation."""
    h = (shape[0] / 2.0 + 0.5) * pixel_scale
    t = np.arange(-h, h, spacing)
    ring = np.concatenate([np.stack([np.full_like(t, h), t], 1),
                           np.stack([t, np.full_like(t, -h)], 1),
                           np.stack([np.full_like(t, -h), -t], 1),
                           np.stack([-t, np.full_like(t, h)], 1)])
    return np.unique(np.round(ring, 9), axis=0)


def _mask_boundary(allowed, pixel_scale, spacing):
    """Zero points round the mask: lattice points (step ``spacing``) within
    about two steps outside it, so no triangle reaches from the mask's
    interior across empty sky."""
    from scipy.ndimage import binary_dilation, distance_transform_edt

    step = max(1, int(round(spacing / pixel_scale)))
    dist = distance_transform_edt(~allowed)            # pixels from the mask
    band = (dist > 0) & (dist <= 2.0 * step)
    rows, cols = np.nonzero(band)
    keep = (rows % step == 0) & (cols % step == 0)
    # and the pixels just outside the edge, thinned, so the zero line hugs it
    edge = binary_dilation(allowed, iterations=1) & ~allowed
    er, ec = np.nonzero(edge)
    thin = (er // step + ec // step) % 2 == 0
    r = np.concatenate([rows[keep], er[thin]])
    c = np.concatenate([cols[keep], ec[thin]])
    pts = _pixel_yx(r, c, allowed.shape, pixel_scale)
    return np.unique(np.round(pts, 9), axis=0)


# ---------------------------------------------------------------------------
# point sets
# ---------------------------------------------------------------------------

def hilbert_points(adapt, pixel_scale: float, n: int, power: float = HILBERT_POWER,
                   floor: float = HILBERT_FLOOR, allowed=None) -> np.ndarray:
    """``n`` points along the Hilbert curve of the (square, native) adapt
    image, with density proportional to max((adapt+ / peak)^power, floor),
    and zero outside ``allowed``."""
    from autoarray.inversion.mesh.image_mesh.hilbert import (
        gilbert2d, inverse_transform_sampling_interpolated,
    )

    adapt = np.asarray(adapt, dtype=float)
    ny, nx = adapt.shape
    if ny != nx:
        raise ValueError("Hilbert points need a square image")
    pos = np.clip(adapt, 0.0, None)
    if allowed is not None:
        pos = np.where(allowed, pos, 0.0)
    peak = pos.max()
    w = (pos / peak) ** power if peak > 0 else np.ones_like(pos)
    w = np.maximum(w, floor)
    if allowed is not None:
        w = np.where(allowed, w, 0.0)
    curve = np.array(list(gilbert2d(nx, ny)))           # (x, y) along the curve
    wc = w[curve[:, 1], curve[:, 0]]
    wc = wc / wc.sum()
    _, xs, ys = inverse_transform_sampling_interpolated(
        wc, int(n), curve[:, 0].astype(float), curve[:, 1].astype(float))
    return _pixel_yx(ys, xs, adapt.shape, pixel_scale)


def lattice_points(geometry, allowed=None) -> np.ndarray:
    """The rectangular mesh's pixel centres (inside ``allowed``)."""
    m = geometry.mesh_shape[0]
    k = geometry.shape_native[0] / m
    centres = (np.arange(m) + 0.5) * k - 0.5             # image-pixel coordinates
    rr, cc = np.meshgrid(centres, centres, indexing="ij")
    pts = _pixel_yx(rr.ravel(), cc.ravel(), geometry.shape_native, geometry.pixel_scale)
    if allowed is None:
        return pts
    ir = np.clip(np.rint(rr.ravel()).astype(int), 0, allowed.shape[0] - 1)
    ic = np.clip(np.rint(cc.ravel()).astype(int), 0, allowed.shape[1] - 1)
    return pts[allowed[ir, ic]]


def build(geometry, mesh_type: str, *, n_points: int | None = None, adapt=None,
          allowed=None, power: float = HILBERT_POWER,
          floor: float = HILBERT_FLOOR, beam: float | None = None) -> IrregularMesh | None:
    """The irregular mesh for these options, or None for the plain rectangle."""
    if mesh_type not in MESH_TYPES:
        raise ValueError(f"mesh type must be one of {MESH_TYPES}, not {mesh_type!r}")
    if mesh_type == "rectangular" and allowed is None:
        return None
    spacing = geometry.mesh_pixel_scale
    if mesh_type == "hilbert":
        if adapt is None:
            raise ValueError("a Hilbert mesh needs an adapt image")
        n = int(n_points or default_points(geometry))
        inner = hilbert_points(adapt, geometry.pixel_scale, n, power, floor, allowed)
        kind = "hilbert"
    else:
        inner = lattice_points(geometry, allowed)
        kind = "masked rectangular"
    if len(inner) < 3:
        raise ValueError("the mask leaves fewer than 3 mesh points: widen it")
    if allowed is None:
        ring = _field_ring(geometry.shape_native, geometry.pixel_scale, 4 * spacing)
        return IrregularMesh(np.concatenate([inner, ring]), 0, kind)
    zero = np.concatenate([
        _mask_boundary(allowed, geometry.pixel_scale, spacing),
        _field_ring(geometry.shape_native, geometry.pixel_scale, 4 * spacing),
    ])
    zero = np.unique(np.round(zero, 9), axis=0)
    return IrregularMesh(np.concatenate([inner, zero]), len(zero), kind,
                         region=model_region(allowed, geometry.pixel_scale,
                                             beam or 2.0 * spacing))


def model_region(allowed, pixel_scale: float, beam: float) -> np.ndarray:
    """The mask dilated by one beam: where a masked model's residual is
    measured (its sidelobes reach about that far)."""
    from scipy.ndimage import binary_dilation

    n = max(1, int(round(beam / pixel_scale)))
    return binary_dilation(np.asarray(allowed, bool), iterations=n)


def default_points(geometry) -> int:
    """`--mesh-points` when not given: a quarter of the nominal mesh."""
    return int(geometry.mesh_shape[0] * geometry.mesh_shape[1] // 4)


def adapt_mesh_shape(geometry) -> tuple[int, int]:
    """The first pass's rectangular mesh for a Hilbert adapt image."""
    m = int(round(geometry.mesh_shape[0] * ADAPT_MESH_FRACTION))
    m = max(ADAPT_MESH_MIN, min(ADAPT_MESH_MAX, m))
    return (m, m)

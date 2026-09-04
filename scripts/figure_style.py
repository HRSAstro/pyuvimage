"""Shared style for the paper figures.

Journal figures, not dashboards: the conventions here are the ones an
interferometry reader expects, and the two that matter most are

* **one shared colour scale per column.** If every panel self-scales, a model
  that has thrown half the flux away looks exactly as good as one that has
  not -- which is precisely the comparison these figures exist to make.
* **perceptually uniform, colour-vision-safe maps.** ``magma`` for intensity
  (single hue family, light to dark), ``RdBu_r`` centred on zero for
  residuals (two hues with a neutral midpoint). No rainbow: a rainbow map
  invents structure at its hue boundaries, which on a residual map is
  indistinguishable from the thing we are looking for.

Sizes are A&A: 8.8 cm one column, 17.6 cm two columns.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, Normalize
from matplotlib.patches import Ellipse

CM = 1.0 / 2.54
ONE_COLUMN = 8.8 * CM
TWO_COLUMN = 17.6 * CM

INK = "#111111"
MUTED = "#8a8a8a"
FAINT = "#d5d5d5"

INTENSITY_CMAP = "magma"
RESIDUAL_CMAP = "RdBu_r"
UNCERTAINTY_CMAP = "viridis"


def use_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
    })


def sky_extent(n_pixels: int, pixel_scale: float) -> list[float]:
    """imshow extent in arcsec with RA increasing to the left.

    Pair with ``origin="lower"`` and an array in FITS orientation (row 0 =
    south), which is what `products.write_products` writes.
    """
    half = 0.5 * n_pixels * pixel_scale
    return [half, -half, -half, half]


def show_image(ax, data, extent, *, cmap=INTENSITY_CMAP, norm=None,
               vmin=None, vmax=None):
    im = ax.imshow(
        np.asarray(data, dtype=float), origin="lower", extent=extent,
        cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5)
    return im


def asinh_norm(vmax: float, vmin: float = 0.0, *, linear_width=None,
               linear_width_frac: float = 0.01):
    """Stretch that shows faint extended emission next to a bright peak.

    A linear scale on a source with a compact knot shows the knot and nothing
    else; a log scale cannot hold the noise, which is signed. asinh is the
    usual compromise and is what the reader expects on a Jy/arcsec^2 panel.

    Pass `linear_width` as the image-plane noise in the panel's own units and
    the transition sits exactly where the data stop being noise -- linear
    through the noise, logarithmic through the source. Choosing it as a
    fraction of `vmax` instead (the fallback) ties it to the brightest pixel,
    which on a source with a compact component buries the extended emission
    in the linear part and renders it black.
    """
    if linear_width is None:
        linear_width = linear_width_frac * float(vmax)
    return AsinhNorm(
        linear_width=max(float(linear_width), 1e-30), vmin=vmin, vmax=vmax,
    )


def symmetric_norm(vmax: float):
    return Normalize(vmin=-abs(vmax), vmax=abs(vmax))


def add_beam(ax, beam, extent, *, frac=0.16, color="white"):
    """The synthesised beam, bottom-left, as an ellipse of the right size.

    Without it a reader cannot tell whether a feature is resolved, which is
    the first question anyone asks of a reconstruction.
    """
    if beam is None:
        return
    x0, x1, y0, y1 = extent
    span = abs(x1 - x0)
    cx = x0 - np.sign(x0 - x1) * frac * span * 0.0 + (x0 - x1) * 0.0
    # bottom-left corner in sky coordinates: RA increases leftwards, so the
    # left edge of the panel is x0 (the larger dRA)
    cx = x0 + (x1 - x0) * 0.13
    cy = y0 + (y1 - y0) * 0.13
    # The panel's x axis runs east (left) to west (right), so north is +y and
    # east is -x. A major axis at position angle theta east of north points
    # along (-sin theta, cos theta), which is theta counter-clockwise from
    # +y -- and the ellipse's `height` starts along +y, so `angle` is +theta.
    ax.add_patch(Ellipse(
        (cx, cy), width=beam.bmin_arcsec, height=beam.bmaj_arcsec,
        angle=beam.bpa_deg,
        facecolor="none", edgecolor=color, linewidth=0.6,
    ))


def _tick_text(v: float) -> str:
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0**e
    if abs(m - 1.0) < 1e-9:
        return rf"$10^{{{e}}}$"
    return rf"${m:g}\times10^{{{e}}}$"


def sky_axes(ax, extent, *, ticks=(-1.0, 0.0, 1.0)):
    """Put arcsec ticks and axis labels back on one panel of a grid.

    Every panel shares the same grid, so labelling one of them is enough and
    labelling all of them is clutter -- but labelling none leaves the reader
    with no angular scale at all, which is the commoner mistake.
    """
    x0, x1, y0, y1 = extent
    keep = [t for t in ticks if min(x1, x0) <= t <= max(x0, x1)]
    ax.set_xticks(keep); ax.set_yticks(keep)
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=5.5,
                   length=1.8, width=0.4, color=MUTED, pad=1.5)
    ax.set_xlabel(r"$\Delta$RA [arcsec]", fontsize=6.0, color=INK, labelpad=1.5)
    ax.set_ylabel(r"$\Delta$Dec [arcsec]", fontsize=6.0, color=INK, labelpad=1.5)


def hcolorbar(fig, im, cells, label, height=0.014, *, pad=0.030, ticks=None):
    """A horizontal colourbar under the column that `cells` spans.

    Positioned from the gridspec cell rather than from a rendered axes, so it
    cannot land on top of a neighbouring panel -- which is what happens when
    a colourbar is placed relative to an axes whose right-hand neighbour is
    another panel.
    """
    boxes = [c.get_position(fig) for c in cells]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    cax = fig.add_axes([x0, y0 - pad, x1 - x0, height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    if ticks is not None:
        # an asinh/log locator left to itself puts a half-visible decade at
        # the end of the bar; naming the ticks avoids it
        cb.ax.set_xticklabels([_tick_text(t) for t in ticks])
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(labelsize=5.5, length=1.6, width=0.4, color=MUTED,
                      pad=1.5)
    cb.set_label(label, fontsize=6.0, color=INK, labelpad=1.5)
    return cb


def colorbar(fig, im, ax, label, *, pad=0.02, width=0.018):
    """A thin colourbar hugging the right edge of `ax`."""
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + pad * box.width, box.y0,
                        width, box.height])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(labelsize=5.5, length=1.6, width=0.4, color=MUTED)
    cb.set_label(label, fontsize=6.0, color=INK, labelpad=2)
    return cb


def panel_label(ax, text, *, loc="upper left", color="white", size=6.5):
    x, y, ha, va = {
        "upper left": (0.04, 0.96, "left", "top"),
        "upper right": (0.96, 0.96, "right", "top"),
        "lower left": (0.04, 0.04, "left", "bottom"),
        "lower right": (0.96, 0.04, "right", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=size, color=color)


def save(fig, stem: str, outdir="figures"):
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    return paths

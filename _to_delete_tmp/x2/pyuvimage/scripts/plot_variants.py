"""Comparison figures for the source-prior variants."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

D = np.load("/tmp/variants.npz", allow_pickle=True)
truth = D["truth"]; models = D["models"]; resids = D["resids"]
order = [str(t) for t in D["order"]]
metrics = json.loads(str(D["metrics"]))
p = float(D["pixel_scale"]); fov = float(D["fov"]); knot = D["knot"]; beam = float(D["beam"])
# the mock stores the knot centre in array coordinates (x increases with column);
# the sky axis dRA runs the other way, so flip the sign for plotting
knot_dra, knot_ddec = -float(knot[1]), float(knot[0])
n = truth.shape[0]
half = fov / 2.0
EXT = [half, -half, -half, half]          # RA increases leftward

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
SERIES = {"matern": "#2a78d6", "adaptive": "#eb6834",
          "hybrid": "#1baf7a", "gibbs": "#4a3aa7"}
def family(tag):
    if tag.startswith("hybrid"): return "hybrid"
    if tag.startswith("adaptive"): return "adaptive"
    if tag.startswith("gibbs"): return "gibbs"
    return "matern"

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values(): s.set_color(MUTED); s.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=2.5)

# ---------------------------------------------------------------- figure 1
# Model images. One shared asinh stretch so panels are directly comparable:
# without it each panel self-scales and every variant looks equally good.
vmax = float(np.nanmax(truth))
norm = AsinhNorm(linear_width=0.05 * vmax, vmin=0.0, vmax=vmax)
panels = [("truth", truth)] + [(t, models[i]) for i, t in enumerate(order)]
fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.6), constrained_layout=True)
for ax, (tag, img) in zip(axes.ravel(), panels):
    im = ax.imshow(img, origin="upper", extent=EXT, cmap="magma", norm=norm)
    ax.add_patch(Rectangle((knot_dra + 0.30, knot_ddec - 0.30), -0.60, 0.60,
                           fill=False, ec="#7ce0ff", lw=1.0))
    m = metrics.get(tag)
    ax.set_title(
        tag if m is None else f"{tag}\npeak {m['cmp_peak']:.2f}  corr {m['corr']:.2f}",
        fontsize=9, color=INK)
    style(ax); ax.set_xticks([1, 0, -1]); ax.set_yticks([-1, 0, 1])
axes[1, 0].set_xlabel("dRA [arcsec]", fontsize=8, color=MUTED)
axes[1, 0].set_ylabel("dDec [arcsec]", fontsize=8, color=MUTED)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("Jy / pixel  (asinh stretch, shared scale)", fontsize=8, color=MUTED)
cb.ax.tick_params(colors=MUTED, labelsize=7)
fig.suptitle("Model images — extended exponential + unresolved offset knot "
             "(cyan box); all fitted to chi2 = N", fontsize=11, color=INK)
fig.savefig("/tmp/fig_models.png", dpi=145, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Residual maps, shared symmetric diverging scale, neutral at zero.
rmax = float(np.nanpercentile(np.abs(resids), 99.8))
fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.6), constrained_layout=True)
for ax, tag, r in zip(axes.ravel(), order, resids):
    im = ax.imshow(r, origin="upper", extent=EXT, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax))
    ax.add_patch(Rectangle((knot_dra + 0.30, knot_ddec - 0.30), -0.60, 0.60,
                           fill=False, ec="#111111", lw=1.0))
    m = metrics[tag]
    ax.set_title(f"{tag}\nknot {m['resid_cmp']:.1f}$\\sigma$  rms {m['resid_rms']:.2f}$\\sigma$",
                 fontsize=9, color=INK)
    style(ax); ax.set_xticks([1, 0, -1]); ax.set_yticks([-1, 0, 1])
axes.ravel()[-1].axis("off")
axes[1, 0].set_xlabel("dRA [arcsec]", fontsize=8, color=MUTED)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("residual [$\\sigma$]", fontsize=8, color=MUTED)
cb.ax.tick_params(colors=MUTED, labelsize=7)
fig.suptitle("Residual maps (data - model, dirty, / rms) — shared scale. "
             "A dipole on the knot means the compact source is mis-fitted",
             fontsize=11, color=INK)
fig.savefig("/tmp/fig_residuals.png", dpi=145, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# Cut through the knot. Small multiples rather than 9 lines on one axis.
row = int(round((n - 1) / 2 - knot_ddec / p))
xs = -((np.arange(n) - (n - 1) / 2) * p)      # array-x -> dRA
sel = np.abs(xs - knot_dra) < 0.75
fig, axes = plt.subplots(3, 3, figsize=(12, 8.4), constrained_layout=True,
                         sharex=True, sharey=True)
def ridge(img):
    """Max over +-1 pixel in dDec, so the curve shows the same quantity the
    scorecard's peak does (a single row can miss the model's peak by a pixel)."""
    return img[max(row - 1, 0):row + 2].max(axis=0)

for ax, tag, mdl in zip(axes.ravel(), order, models):
    c = SERIES[family(tag)]
    ax.plot(xs[sel], ridge(truth)[sel], color=MUTED, lw=2.4, label="truth")
    ax.plot(xs[sel], ridge(mdl)[sel], color=c, lw=2.0, label=tag)
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_title(f"{tag}   peak {metrics[tag]['cmp_peak']:.2f}x",
                 fontsize=9.5, color=INK)
    style(ax)
    ax.text(0.03, 0.92, "truth", transform=ax.transAxes, fontsize=8,
            color=MUTED, ha="left", va="top")
    ax.text(0.03, 0.80, tag, transform=ax.transAxes, fontsize=8,
            color=c, ha="left", va="top", fontweight="bold")
for ax in axes[2]: ax.set_xlabel("dRA [arcsec]", fontsize=8, color=MUTED)
for ax in axes[:, 0]: ax.set_ylabel("Jy / pixel", fontsize=8, color=MUTED)
fig.suptitle("Cut through the unresolved knot (max over $\\pm$1 pixel in dDec) — "
             "matern loses half the peak; the adaptive and Gibbs priors recover it",
             fontsize=11, color=INK)
fig.savefig("/tmp/fig_profiles.png", dpi=145, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- figure 4
# Metrics. One panel per measure (never a shared axis across measures);
# dashed line = the ideal value; every point directly labelled.
specs = [("corr", "correlation with truth", None, "higher is better"),
         ("cmp_peak", "compact peak / truth", 1.0, "1.0 is exact"),
         ("cmp_flux", "compact flux / truth", 1.0, "1.0 is exact"),
         ("resid_cmp", "peak residual at knot [$\\sigma$]", None, "lower is better")]
ypos = np.arange(len(order))[::-1]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), constrained_layout=True, sharey=True)
for ax, (key, title, ideal, note) in zip(axes, specs):
    vals = [metrics[t][key] for t in order]
    if ideal is not None:
        ax.axvline(ideal, color=MUTED, lw=1.0, ls="--", zorder=1)
    for y, t, v in zip(ypos, order, vals):
        c = SERIES[family(t)]
        ax.plot([min(vals + ([ideal] if ideal else [])) * 0.0, v], [y, y],
                color=c, lw=2.0, alpha=0.35, zorder=2, solid_capstyle="round")
        ax.plot(v, y, "o", ms=9, color=c, mec="white", mew=1.6, zorder=3)
        ax.annotate(f"{v:.2f}", (v, y), textcoords="offset points",
                    xytext=(10, 0), va="center", fontsize=8, color=INK)
    ax.set_title(f"{title}\n{note}", fontsize=9.5, color=INK)
    style(ax); ax.grid(axis="x", color="#e8e8e4", lw=0.7); ax.set_axisbelow(True)
    ax.margins(x=0.22)
axes[0].set_yticks(ypos); axes[0].set_yticklabels(order, fontsize=9, color=INK)
fig.suptitle("Variant scorecard — all fitted to the same chi2 = N "
             "(one mock, one noise realisation)", fontsize=11, color=INK)
fig.savefig("/tmp/fig_metrics.png", dpi=145, facecolor="white")
plt.close(fig)
print("figures written")

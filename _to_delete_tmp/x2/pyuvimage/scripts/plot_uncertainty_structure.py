import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from pyuvimage import mock, fitting

D = np.load("/tmp/gibbs_unc.npz")
model, post, samp, mc = D["model"], D["post"], D["samp"], D["mc"]
p, fov, knot = float(D["pix"]), float(D["fov"]), D["knot"]
# matern, for contrast (stationary => must be flat)
uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(n_vis=600, mesh_n=32, sigma_jy=5e-4)
uv, d, nz = uvd.flattened()
mat = fitting.fit_dataset(fitting.make_dataset(uv, d, nz, geom), geom, reg_kind="matern",
                          prior={"coefficient":3e7,"scale":0.25,"nu":1.5},
                          positive_only=False).model_uncertainty

INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
half = fov/2; EXT=[half,-half,-half,half]
kd, kdd = -float(knot[1]), float(knot[0])
def style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values(): s.set_color(MUTED); s.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=2.5)
def rel(a):  # structure, independent of the absolute level
    return a/np.median(a[a>0])

fig, axes = plt.subplots(1, 5, figsize=(19, 4.3), constrained_layout=True)
im0 = axes[0].imshow(model, origin="upper", extent=EXT, cmap="magma",
                     norm=AsinhNorm(linear_width=0.05*model.max(), vmin=0, vmax=model.max()))
axes[0].set_title("model (gibbs, the default)", fontsize=9.5, color=INK)
cb=fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02); cb.set_label("Jy/pixel", fontsize=8, color=MUTED)
cb.ax.tick_params(colors=MUTED, labelsize=7)
panels = [(rel(post), "gibbs posterior 1$\\sigma$\n6x range across the map"),
          (rel(samp), "gibbs noise-only 1$\\sigma$"),
          (rel(mc), "gibbs Monte Carlo scatter\n30 realisations"),
          (rel(mat), "matern posterior 1$\\sigma$\nstationary prior = flat by construction")]
vmax = max(np.percentile(x[0], 99) for x in panels)
for ax, (img, title) in zip(axes[1:], panels):
    im = ax.imshow(img, origin="upper", extent=EXT, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9.5, color=INK)
    cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("$\\sigma$ / median $\\sigma$", fontsize=8, color=MUTED)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
for ax in axes:
    style(ax); ax.set_xticks([1,0,-1]); ax.set_yticks([-1,0,1])
    ax.plot(kd, kdd, "o", mfc="none", mec="#ff5ca8", mew=1.4, ms=13)
axes[0].set_xlabel("dRA [arcsec]", fontsize=8, color=MUTED)
axes[0].set_ylabel("dDec [arcsec]", fontsize=8, color=MUTED)
fig.suptitle("Uncertainty structure. The posterior covariance (F+H)$^{-1}$ contains no data, so it responds to the "
             "prior and the uv coverage — not to source brightness.\nA non-stationary prior (gibbs) therefore varies "
             "around the knot (pink circle); a stationary one (matern) is flat. MC/analytic = 0.995.",
             fontsize=10.5, color=INK)
fig.savefig("/tmp/fig_uncertainty2.png", dpi=145, facecolor="white")
print("post/median range: gibbs %.2f-%.2f   matern %.2f-%.2f"
      % (rel(post).min(), rel(post).max(), rel(mat).min(), rel(mat).max()))

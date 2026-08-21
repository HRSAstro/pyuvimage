import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import autogalaxy as ag
from pyuvimage import mock, fitting

uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(n_vis=600, mesh_n=32, sigma_jy=5e-4)
uv, d, nz = uvd.flattened()
PRIOR = {"coefficient": 3e7, "scale": 0.25, "nu": 1.5}
sf = fitting.fit_dataset(fitting.make_dataset(uv, d, nz, geom), geom,
                         reg_kind="matern", prior=PRIOR, positive_only=False)
model, post, samp = sf.model_image, sf.model_uncertainty, sf.model_uncertainty_sampling
truth_img = np.kron(truth, np.ones((2, 2))) / 4.0

mask_t = ag.Mask2D.all_false(shape_native=truth.shape, pixel_scales=geom.mesh_pixel_scale)
v_true = np.asarray(ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask_t)
                    .visibilities_from(image=ag.Array2D(values=truth, mask=mask_t)))
rng = np.random.default_rng(7); imgs = []
for _ in range(30):
    dk = v_true + rng.normal(0, nz.real) + 1j * rng.normal(0, nz.imag)
    imgs.append(fitting.fit_dataset(fitting.make_dataset(uv, dk, nz, geom), geom,
                reg_kind="matern", prior=PRIOR, positive_only=False).model_image)
imgs = np.array(imgs); mc = imgs.std(axis=0, ddof=1)
total = np.sqrt(((imgs - truth_img) ** 2).mean(axis=0))

INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
half = geom.fov_arcsec / 2; EXT = [half, -half, -half, half]
def style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values(): s.set_color(MUTED); s.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=2.5)

fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True)
unorm = AsinhNorm(linear_width=0.2 * post.max(), vmin=0, vmax=post.max())
panels = [(model, "model", "Jy/pixel", "magma",
           AsinhNorm(linear_width=0.05*model.max(), vmin=0, vmax=model.max())),
          (post, "posterior 1$\\sigma$\n(shipped as uncertainty.fits)", "Jy/pixel", "viridis", unorm),
          (samp, "noise-only 1$\\sigma$\n(uncertainty_noise.fits)", "Jy/pixel", "viridis", unorm),
          (mc, "Monte Carlo scatter\n30 noise realisations", "Jy/pixel", "viridis", unorm),
          (np.where(post > 0, model / post, 0), "model / posterior 1$\\sigma$", "", "cividis", None)]
for ax, (img, title, unit, cmap, nrm) in zip(axes, panels):
    im = ax.imshow(img, origin="upper", extent=EXT, cmap=cmap, norm=nrm)
    ax.set_title(title, fontsize=9.5, color=INK); style(ax)
    ax.set_xticks([1, 0, -1]); ax.set_yticks([-1, 0, 1])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(unit, fontsize=8, color=MUTED); cb.ax.tick_params(colors=MUTED, labelsize=7)
axes[0].set_xlabel("dRA [arcsec]", fontsize=8, color=MUTED)
axes[0].set_ylabel("dDec [arcsec]", fontsize=8, color=MUTED)
r1 = np.median(mc / np.maximum(samp, 1e-30)); r2 = np.median(total / np.maximum(post, 1e-30))
fig.suptitle(f"Per-pixel uncertainty.  Monte Carlo / noise-only prediction = {r1:.2f} "
             f"(formula verified);  total rms error from truth / posterior = {r2:.2f}. "
             "Panels 2-4 share one colour scale.", fontsize=10.5, color=INK)
fig.savefig("/tmp/fig_uncertainty.png", dpi=145, facecolor="white")
print("ratios", round(float(r1),3), round(float(r2),3))

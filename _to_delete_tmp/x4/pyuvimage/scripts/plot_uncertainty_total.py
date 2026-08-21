"""The single total uncertainty map, and what it is made of."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyuvimage import fitting, mock
from pyuvimage.fitting import _block_contrast

uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
    n_vis=600, mesh_n=32, compact_flux=0.012, compact_centre=(0.8, -0.7))
uv, d, nz = uvd.flattened()
ds = fitting.make_dataset(uv, d, nz, geom)
# matern with a fixed prior: fast, and the checkerboard is a property of the
# mesh/image grid pair, not of the kernel -- a stationary prior shows it most
# cleanly because nothing else varies across the map
sf = fitting.fit_dataset(ds, geom, reg_kind="matern",
    prior={"coefficient": 5e7, "scale": 0.25, "nu": 1.5}, positive_only=False)

model = sf.model_image
stat = sf.model_uncertainty
sys_ = sf.prior_systematic()
raw = np.hypot(stat, sys_)
total, terms = sf.model_uncertainty_total()
ovs = geom.oversample
print("checkerboard: raw %.1f%%  ->  delivered %.1f%%"
      % (100*_block_contrast(raw, ovs), 100*_block_contrast(total, ovs)))
print(terms)

ext = [geom.fov_arcsec/2, -geom.fov_arcsec/2, -geom.fov_arcsec/2, geom.fov_arcsec/2]
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
panels = [
    (model, "model", "inferno", "Jy/pixel"),
    (stat, "statistical 1$\\sigma$  (posterior)", "viridis", "Jy/pixel"),
    (sys_, "prior-strength systematic", "viridis", "Jy/pixel"),
    (raw, "total, before de-blocking", "viridis", "Jy/pixel"),
    (total, "total, delivered (uncertainty.fits)", "viridis", "Jy/pixel"),
    (np.where(total > 0, model/total, 0), "model / total 1$\\sigma$ (snr.fits)",
     "magma", ""),
]
for a, (img, title, cmap, unit) in zip(ax.ravel(), panels):
    im = a.imshow(img, origin="upper", extent=ext, cmap=cmap)
    a.set_title(title, fontsize=10)
    a.set_xlabel('dRA ["]'); a.set_ylabel('dDec ["]')
    fig.colorbar(im, ax=a, fraction=0.046, label=unit)

# zoom insets showing the checkerboard
for a, img, lab in [(ax[1,0], raw, "before"), (ax[1,1], total, "after")]:
    n = img.shape[0]; c = n//2
    sub = img[c-8:c+8, c-8:c+8]
    inset = a.inset_axes([0.62, 0.62, 0.36, 0.36])
    inset.imshow(sub, origin="upper", cmap="viridis")
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_title(f"{lab} ({100*_block_contrast(img, ovs):.0f}%)", fontsize=7,
                    color="w", pad=2)
    for sp in inset.spines.values():
        sp.set_color("w")

fig.suptitle("A single total uncertainty map. Statistical (posterior) and "
             "prior-strength systematic combined in quadrature,\\nwith the "
             "mesh/image checkerboard replaced by its upper envelope so "
             "significance maps are artefact-free.", fontsize=11)
fig.tight_layout()
fig.savefig("figures/uncertainty_total.png", dpi=130)
print("wrote figures/uncertainty_total.png")

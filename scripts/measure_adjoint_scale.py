"""Measure the factor separating our pynufft adjoint from the DFT adjoint.

autoarray states it as `4 * N_y * N_x`. This checks that against the DFT
rather than taking it on faith, because our transformer is a vendored
reimplementation and could have diverged.
"""

import numpy as np
import autogalaxy as ag

from pyuvimage import fitting

rng = np.random.default_rng(0)

for n_pix, n_vis, pscale in [(16, 300, 0.1), (24, 500, 0.05), (32, 400, 0.08)]:
    mask = ag.Mask2D.all_false(shape_native=(n_pix, n_pix), pixel_scales=pscale)
    pixel_rad = pscale * np.pi / (180 * 3600)
    nyq = 1.0 / (2.0 * pixel_rad)
    uv = rng.uniform(-0.4 * nyq, 0.4 * nyq, size=(n_vis, 2))
    vis = ag.Visibilities(
        visibilities=rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
    )

    dft = ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
    pn = fitting.pynufft_transformer_class()(
        uv_wavelengths=uv, real_space_mask=mask
    )

    a = np.asarray(dft.image_from(visibilities=vis).native)
    raw = np.asarray(pn.image_from(visibilities=vis).native)
    scaled = np.asarray(
        pn.image_from(visibilities=vis, use_adjoint_scaling=True).native
    )

    keep = np.abs(raw) > np.abs(raw).max() * 1e-3
    ratio = a[keep] / raw[keep]
    rel = np.abs(scaled - a).max() / np.abs(a).max()
    print(
        f"n_pix={n_pix:3d}  measured factor={np.median(ratio):10.2f}  "
        f"4*Ny*Nx={4 * n_pix * n_pix:6d}  "
        f"adjoint_scaling={pn.adjoint_scaling:8.0f}  "
        f"max|scaled-dft|/max|dft| = {rel:.3e}"
    )

"""Simulate an unlensed exponential source as dataprep-format FITS products."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def synthetic_uv_wavelengths(
    n_vis=200,
    uv_min=2.0e4,
    uv_max=7.0e5,
    seed=42,
):
    """
    Simple random UV coverage in wavelengths.

    Default ``uv_max`` ≈ 7e5 → Nyquist ≈ 0.15\" (usable with FOV ~5\").
    """
    rng = np.random.default_rng(seed)
    radii = np.sqrt(rng.uniform(uv_min**2, uv_max**2, size=n_vis))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_vis)
    u = radii * np.cos(angles)
    v = radii * np.sin(angles)
    return np.stack([u, v], axis=-1)


def load_or_make_uv(uv_path=None, **kwargs):
    if uv_path is not None:
        path = Path(uv_path)
        if path.is_file():
            from astropy.io import fits

            uv = np.asarray(fits.getdata(path), dtype=float)
            # Dataprep products are (n_chan, n_row, 2); simulator wants (n_row, 2).
            if uv.ndim == 3 and uv.shape[0] == 1 and uv.shape[-1] == 2:
                uv = uv[0]
            elif uv.ndim == 3 and uv.shape[-1] == 2:
                uv = uv.mean(axis=0)
            return uv
    return synthetic_uv_wavelengths(**kwargs)


def simulate_exponential_interferometer(
    *,
    uv_wavelengths,
    fov=5.0,
    n_pixels=64,
    intensity=1.0,
    effective_radius=0.6,
    axis_ratio=0.75,
    angle=30.0,
    centre=(0.0, 0.0),
    noise_sigma=0.05,
    exposure_time=300.0,
    noise_seed=1,
    transformer_class=None,
):
    """
    Simulate noisy visibilities of an exponential galaxy (no lensing).

    Returns ``(dataset, tracer, grid, truth_image)``.
    """
    import autolens as al

    if transformer_class is None:
        transformer_class = al.TransformerDFT

    pixel_scales = float(fov) / int(n_pixels)
    grid = al.Grid2D.uniform(
        shape_native=(n_pixels, n_pixels),
        pixel_scales=pixel_scales,
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.Exponential(
            centre=centre,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=axis_ratio, angle=angle
            ),
            intensity=intensity,
            effective_radius=effective_radius,
        ),
    )
    # Identity mass keeps Autolens Tracer API available for plots / sanity checks.
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.PowerLaw(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            einstein_radius=0.0,
            slope=2.0,
        ),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    truth_image = tracer.image_2d_from(grid=grid)

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.asarray(uv_wavelengths, dtype=float),
        exposure_time=exposure_time,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        transformer_class=transformer_class,
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    return dataset, tracer, grid, truth_image


def simulate_dual_component_interferometer(
    *,
    uv_wavelengths,
    fov=5.0,
    n_pixels=64,
    compact=None,
    extended=None,
    noise_sigma=0.05,
    exposure_time=300.0,
    noise_seed=1,
    transformer_class=None,
):
    """
    Simulate noisy visibilities of a bright compact + faint extended source.

    ``compact`` / ``extended`` are dicts of light-profile kwargs (defaults below).
    Returns ``(dataset, tracer, grid, truth_image)``.
    """
    import autolens as al

    if transformer_class is None:
        transformer_class = al.TransformerDFT

    compact_cfg = {
        "centre": (0.05, -0.05),
        "intensity": 2.0,
        "effective_radius": 0.08,
        "axis_ratio": 0.95,
        "angle": 0.0,
    }
    extended_cfg = {
        "centre": (-0.2, 0.15),
        "intensity": 0.12,
        "effective_radius": 0.75,
        "axis_ratio": 0.55,
        "angle": 55.0,
    }
    if compact:
        compact_cfg.update(compact)
    if extended:
        extended_cfg.update(extended)

    pixel_scales = float(fov) / int(n_pixels)
    grid = al.Grid2D.uniform(
        shape_native=(n_pixels, n_pixels),
        pixel_scales=pixel_scales,
    )

    def _exponential(**kwargs):
        return al.lp.Exponential(
            centre=tuple(kwargs["centre"]),
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=float(kwargs["axis_ratio"]),
                angle=float(kwargs["angle"]),
            ),
            intensity=float(kwargs["intensity"]),
            effective_radius=float(kwargs["effective_radius"]),
        )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        compact=_exponential(**compact_cfg),
        extended=_exponential(**extended_cfg),
    )
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.PowerLaw(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            einstein_radius=0.0,
            slope=2.0,
        ),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    truth_image = tracer.image_2d_from(grid=grid)

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=np.asarray(uv_wavelengths, dtype=float),
        exposure_time=exposure_time,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        transformer_class=transformer_class,
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    return dataset, tracer, grid, truth_image, compact_cfg, extended_cfg


def write_dataprep_products(
    output_dir,
    dataset,
    *,
    uid="mock_exponential",
    width="mfs",
    filename_suffix="_contsub",
    frequency_hz=230.0e9,
    n_channels=1,
):
    """
    Write LensKin/pyuvimage dataprep-shaped FITS products.

    Shapes:
      visibilities / sigma: ``(n_corr=2, n_chan, n_row, 2)``
      uv_wavelengths: ``(n_chan, n_row, 2)``
      frequencies: ``(n_chan,)``
    """
    from src.utils.io import write_exported_array

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.asarray(dataset.data)
    noise = np.asarray(dataset.noise_map)
    uv = np.asarray(dataset.uv_wavelengths, dtype=float)

    # Autolens Visibilities are complex (or structured); normalize to Re/Im.
    if np.iscomplexobj(data):
        vis_ri = np.stack([data.real, data.imag], axis=-1)
    elif data.ndim == 2 and data.shape[-1] == 2:
        vis_ri = data
    else:
        vis_ri = np.stack([np.asarray(data).real, np.asarray(data).imag], axis=-1)

    if np.iscomplexobj(noise):
        # Noise maps may be complex-typed but store real sigma in .real.
        sigma_1d = np.asarray(np.real(noise), dtype=float)
        if sigma_1d.ndim == 0:
            sigma_ri = np.full_like(vis_ri, float(sigma_1d))
        else:
            sigma_ri = np.stack([sigma_1d.reshape(-1), sigma_1d.reshape(-1)], axis=-1)
    elif noise.ndim == 2 and noise.shape[-1] == 2:
        sigma_ri = np.asarray(noise, dtype=float)
    else:
        sigma_1d = np.asarray(noise, dtype=float).reshape(-1)
        sigma_ri = np.stack([sigma_1d, sigma_1d], axis=-1)

    n_row = vis_ri.shape[0]
    # Tile identical continuum channel(s); duplicate XX/YY.
    vis_chan = np.broadcast_to(
        vis_ri[None, :, :], (n_channels, n_row, 2)
    ).copy()
    sigma_chan = np.broadcast_to(
        sigma_ri[None, :, :], (n_channels, n_row, 2)
    ).copy()
    uv_chan = np.broadcast_to(uv[None, :, :], (n_channels, n_row, 2)).copy()

    visibilities = np.stack([vis_chan, vis_chan], axis=0)  # (2, n_chan, n_row, 2)
    sigma = np.stack([sigma_chan, sigma_chan], axis=0)
    frequencies = np.full(n_channels, float(frequency_hz), dtype=float)

    stem = f"{{prefix}}_{uid}_width_{width}{filename_suffix}"
    paths = {}
    for prefix, array in (
        ("visibilities", visibilities),
        ("sigma_statwt", sigma),
        ("uv_wavelengths", uv_chan),
        ("frequencies", frequencies),
    ):
        base = output_dir / stem.format(prefix=prefix)
        paths[prefix] = write_exported_array(str(base), array)

    meta = {
        "uid": uid,
        "width": width,
        "filename_suffix": filename_suffix,
        "n_channels": n_channels,
        "n_vis": int(n_row),
        "frequency_hz": float(frequency_hz),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    meta_path = output_dir / "mock_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    return paths


def make_exponential_mock(
    output_dir="./data/mock_exponential",
    *,
    fov=5.0,
    n_pixels=64,
    uv_path=None,
    n_vis=200,
    noise_sigma=0.05,
    intensity=1.0,
    effective_radius=0.6,
    seed=1,
):
    """Generate mock dataprep products and a truth image FITS/PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    from src.utils.io import write_image_fits

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uv = load_or_make_uv(uv_path, n_vis=n_vis, seed=seed)
    dataset, tracer, grid, truth_image = simulate_exponential_interferometer(
        uv_wavelengths=uv,
        fov=fov,
        n_pixels=n_pixels,
        intensity=intensity,
        effective_radius=effective_radius,
        noise_sigma=noise_sigma,
        noise_seed=seed,
    )
    paths = write_dataprep_products(output_dir, dataset)

    truth = np.asarray(truth_image.native if hasattr(truth_image, "native") else truth_image)
    pixel_scale = float(fov) / int(n_pixels)
    truth_fits = write_image_fits(
        output_dir / "truth_image.fits",
        truth,
        header_cards={
            "FOV": float(fov),
            "PIXSCALE": pixel_scale,
            "NPIX": int(n_pixels),
            "PROFILE": "Exponential",
            "INTENS": float(intensity),
            "NOISESIG": float(noise_sigma),
        },
    )
    # Augment dataprep meta with simulation knobs used for this mock.
    meta_path = paths.get("meta")
    if meta_path is not None and Path(meta_path).is_file():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        meta.update(
            {
                "intensity": float(intensity),
                "effective_radius": float(effective_radius),
                "noise_sigma": float(noise_sigma),
                "fov": float(fov),
                "n_pixels": int(n_pixels),
                "seed": int(seed),
            }
        )
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["truth"] = truth_fits

    # Dirty image for a quick look (uses Autolens transformer on a mask).
    import autolens as al

    mask = al.Mask2D.circular(
        shape_native=(n_pixels, n_pixels),
        pixel_scales=fov / n_pixels,
        radius=fov / 2.0,
    )
    dirty_dataset = al.Interferometer(
        data=dataset.data,
        noise_map=dataset.noise_map,
        uv_wavelengths=dataset.uv_wavelengths,
        real_space_mask=mask,
        transformer_class=al.TransformerDFT,
        raise_error_dft_visibilities_limit=False,
    )
    dirty = np.asarray(dirty_dataset.dirty_image.native)
    if dirty.shape != truth.shape:
        raise ValueError(
            f"Mock dirty shape {dirty.shape} != truth shape {truth.shape}"
        )

    from src.deconv.plots import sky_extent_arcsec

    extent = sky_extent_arcsec(truth.shape, pixel_scale)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    im0 = axes[0].imshow(
        truth,
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    axes[0].set_title("Truth (Exponential)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(
        dirty,
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1].set_title("Dirty image")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("ΔRA [arcsec]")
        ax.set_ylabel("ΔDec [arcsec]")
    fig.suptitle(f"pixel scale={pixel_scale:.4g}\"  grid={n_pixels}×{n_pixels}", fontsize=10)
    fig.tight_layout()
    preview = output_dir / "mock_preview.png"
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    paths["preview"] = preview

    write_image_fits(
        output_dir / "dirty_image.fits",
        dirty,
        header_cards={
            "FOV": float(fov),
            "PIXSCALE": pixel_scale,
            "NPIX": int(n_pixels),
        },
    )
    return {
        "output_dir": output_dir,
        "paths": paths,
        "dataset": dataset,
        "tracer": tracer,
        "grid": grid,
        "truth_image": truth,
        "dirty_image": dirty,
        "fov": fov,
        "n_vis": int(np.asarray(uv).shape[0]),
    }


def make_dual_component_mock(
    output_dir="./data/mock_dual_component",
    *,
    fov=5.0,
    n_pixels=64,
    uv_path=None,
    n_vis=200,
    noise_sigma=0.05,
    compact=None,
    extended=None,
    seed=1,
    uid="mock_dual",
):
    """
    Generate mock dataprep products for a bright compact + faint extended source.

    Same UV/dataprep layout as ``make_exponential_mock``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.io import write_image_fits

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uv = load_or_make_uv(uv_path, n_vis=n_vis, seed=seed)
    (
        dataset,
        tracer,
        grid,
        truth_image,
        compact_cfg,
        extended_cfg,
    ) = simulate_dual_component_interferometer(
        uv_wavelengths=uv,
        fov=fov,
        n_pixels=n_pixels,
        compact=compact,
        extended=extended,
        noise_sigma=noise_sigma,
        noise_seed=seed,
    )
    paths = write_dataprep_products(output_dir, dataset, uid=uid)

    truth = np.asarray(
        truth_image.native if hasattr(truth_image, "native") else truth_image
    )
    pixel_scale = float(fov) / int(n_pixels)
    truth_fits = write_image_fits(
        output_dir / "truth_image.fits",
        truth,
        header_cards={
            "FOV": float(fov),
            "PIXSCALE": pixel_scale,
            "NPIX": int(n_pixels),
            "PROFILE": "DualExp",
            "C_INT": float(compact_cfg["intensity"]),
            "C_RE": float(compact_cfg["effective_radius"]),
            "E_INT": float(extended_cfg["intensity"]),
            "E_RE": float(extended_cfg["effective_radius"]),
            "NOISESIG": float(noise_sigma),
        },
    )
    meta_path = paths.get("meta")
    if meta_path is not None and Path(meta_path).is_file():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        meta.update(
            {
                "profile": "dual_exponential",
                "compact": compact_cfg,
                "extended": extended_cfg,
                "noise_sigma": float(noise_sigma),
                "fov": float(fov),
                "n_pixels": int(n_pixels),
                "seed": int(seed),
            }
        )
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["truth"] = truth_fits

    import autolens as al

    mask = al.Mask2D.circular(
        shape_native=(n_pixels, n_pixels),
        pixel_scales=fov / n_pixels,
        radius=fov / 2.0,
    )
    dirty_dataset = al.Interferometer(
        data=dataset.data,
        noise_map=dataset.noise_map,
        uv_wavelengths=dataset.uv_wavelengths,
        real_space_mask=mask,
        transformer_class=al.TransformerDFT,
        raise_error_dft_visibilities_limit=False,
    )
    dirty = np.asarray(dirty_dataset.dirty_image.native)

    from src.deconv.plots import sky_extent_arcsec

    extent = sky_extent_arcsec(truth.shape, pixel_scale)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    im0 = axes[0].imshow(
        truth,
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    axes[0].set_title("Truth (compact + extended)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(
        dirty,
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1].set_title("Dirty image")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("ΔRA [arcsec]")
        ax.set_ylabel("ΔDec [arcsec]")
    fig.suptitle(
        f"pixel scale={pixel_scale:.4g}\"  grid={n_pixels}×{n_pixels}",
        fontsize=10,
    )
    fig.tight_layout()
    preview = output_dir / "mock_preview.png"
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    paths["preview"] = preview

    write_image_fits(
        output_dir / "dirty_image.fits",
        dirty,
        header_cards={
            "FOV": float(fov),
            "PIXSCALE": pixel_scale,
            "NPIX": int(n_pixels),
        },
    )
    return {
        "output_dir": output_dir,
        "paths": paths,
        "dataset": dataset,
        "tracer": tracer,
        "grid": grid,
        "truth_image": truth,
        "dirty_image": dirty,
        "fov": fov,
        "n_vis": int(np.asarray(uv).shape[0]),
        "compact": compact_cfg,
        "extended": extended_cfg,
    }

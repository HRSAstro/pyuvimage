"""Identity-lens pixelization model for UV-plane deconvolution."""

from __future__ import annotations

import autofit as af
import autolens as al
import numpy as np

SOURCE_PATH = "('galaxies', 'source')"

_ADAPT_REG_TYPES = frozenset({"adapt", "adapt_split", "adapt_split_zeroth"})
_SPLIT_REG_TYPES = frozenset({"constant_split", "adapt_split", "adapt_split_zeroth"})
_KERNEL_REG_TYPES = frozenset({"gaussian_kernel", "matern_kernel"})


def identity_lens_galaxy(redshift=0.5):
    """Galaxy with θ_E = 0 PowerLaw mass (identity ray-trace)."""
    mass = af.Model(
        al.mp.PowerLaw,
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        einstein_radius=0.0,
        slope=2.0,
    )
    shear = af.Model(al.mp.ExternalShear, gamma_1=0.0, gamma_2=0.0)
    return af.Model(al.Galaxy, redshift=redshift, mass=mass, shear=shear)


def _scalar_prior_from_settings(reg_cfg, param_name, defaults):
    prior_type = reg_cfg.get("prior_type", "log_uniform")
    param_cfg = reg_cfg.get(param_name)
    if isinstance(param_cfg, dict):
        if param_cfg.get("prior_type") == "fixed" or "value" in param_cfg:
            return float(param_cfg.get("value", defaults["fixed"]))
        lower_limit = param_cfg.get("lower_limit", defaults["lower"])
        upper_limit = param_cfg.get("upper_limit", defaults["upper"])
        return af.LogUniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)

    if prior_type == "fixed":
        return float(reg_cfg.get(param_name, defaults["fixed"]))

    lower_limit = reg_cfg.get(
        f"{param_name}_lower_limit",
        reg_cfg.get("lower_limit", defaults["lower"]),
    )
    upper_limit = reg_cfg.get(
        f"{param_name}_upper_limit",
        reg_cfg.get("upper_limit", defaults["upper"]),
    )
    return af.LogUniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)


def _signal_scale_from_settings(reg_cfg, default=3.0):
    param_cfg = reg_cfg.get("signal_scale", default)
    if isinstance(param_cfg, dict):
        if param_cfg.get("prior_type") == "fixed" or "value" in param_cfg:
            return float(param_cfg.get("value", default))
        return af.UniformPrior(
            lower_limit=param_cfg.get("lower_limit", 0.0),
            upper_limit=param_cfg.get("upper_limit", 10.0),
        )
    if reg_cfg.get("prior_type") == "fixed":
        return float(param_cfg)
    if isinstance(param_cfg, (int, float)):
        return float(param_cfg)
    return float(default)


def _build_regularization_model(reg_cfg, reg_type, fixed_coefficient=None):
    if reg_type in {"constant", "constant_split"}:
        reg_cls = al.reg.Constant if reg_type == "constant" else al.reg.ConstantSplit
        regularization = af.Model(reg_cls)
        if fixed_coefficient is not None:
            regularization.coefficient = float(fixed_coefficient)
        else:
            regularization.coefficient = _scalar_prior_from_settings(
                reg_cfg,
                param_name="coefficient",
                defaults={
                    "fixed": reg_cfg.get("value", 1e5),
                    "lower": reg_cfg.get("lower_limit", 1e-2),
                    "upper": reg_cfg.get("upper_limit", 1e6),
                },
            )
        return regularization

    if reg_type in _ADAPT_REG_TYPES:
        reg_cls = {
            "adapt": al.reg.Adapt,
            "adapt_split": al.reg.AdaptSplit,
            "adapt_split_zeroth": al.reg.AdaptSplitZeroth,
        }[reg_type]
        regularization = af.Model(reg_cls)
        regularization.inner_coefficient = _scalar_prior_from_settings(
            reg_cfg,
            param_name="inner_coefficient",
            defaults={"fixed": 1.0, "lower": 1e-6, "upper": 1e6},
        )
        outer_ratio = reg_cfg.get("outer_inner_ratio")
        if outer_ratio is not None and isinstance(
            regularization.inner_coefficient, (int, float)
        ):
            regularization.outer_coefficient = float(outer_ratio) * float(
                regularization.inner_coefficient
            )
        else:
            regularization.outer_coefficient = _scalar_prior_from_settings(
                reg_cfg,
                param_name="outer_coefficient",
                defaults={"fixed": 100.0, "lower": 1e-6, "upper": 1e6},
            )
        regularization.signal_scale = _signal_scale_from_settings(reg_cfg, default=3.0)
        if reg_type == "adapt_split_zeroth":
            regularization.zeroth_coefficient = _scalar_prior_from_settings(
                reg_cfg,
                param_name="zeroth_coefficient",
                defaults={"fixed": 1.0, "lower": 1e-6, "upper": 1e6},
            )
            regularization.zeroth_signal_scale = _signal_scale_from_settings(
                reg_cfg, default=1.0
            )
            # overwrite signal_scale key name for zeroth
            zcfg = reg_cfg.get("zeroth_signal_scale", 1.0)
            if isinstance(zcfg, dict) and (
                zcfg.get("prior_type") == "fixed" or "value" in zcfg
            ):
                regularization.zeroth_signal_scale = float(zcfg.get("value", 1.0))
        return regularization

    if reg_type in _KERNEL_REG_TYPES:
        reg_cls = {
            "gaussian_kernel": al.reg.GaussianKernel,
            "matern_kernel": al.reg.MaternKernel,
        }[reg_type]
        regularization = af.Model(reg_cls)
        if fixed_coefficient is not None:
            regularization.coefficient = float(fixed_coefficient)
        else:
            regularization.coefficient = _scalar_prior_from_settings(
                reg_cfg,
                param_name="coefficient",
                defaults={
                    "fixed": reg_cfg.get("value", reg_cfg.get("coefficient", 100.0)),
                    "lower": reg_cfg.get("lower_limit", 1e-2),
                    "upper": reg_cfg.get("upper_limit", 1e6),
                },
            )
        regularization.scale = _kernel_scale_from_settings(reg_cfg, default=0.15)
        if reg_type == "matern_kernel":
            regularization.nu = float(reg_cfg.get("nu", 1.5))
        return regularization

    raise ValueError(f"Unsupported regularization type: {reg_type!r}")


def _kernel_scale_from_settings(reg_cfg, default=0.15):
    """Correlation length (arcsec) for Gaussian / Matérn kernel regularization."""
    scale_cfg = reg_cfg.get("scale", default)
    if isinstance(scale_cfg, dict):
        if scale_cfg.get("prior_type") == "fixed" or "value" in scale_cfg:
            return float(scale_cfg.get("value", default))
        return af.LogUniformPrior(
            lower_limit=float(scale_cfg.get("lower_limit", 1e-3)),
            upper_limit=float(scale_cfg.get("upper_limit", 10.0)),
        )
    if reg_cfg.get("prior_type") == "fixed" or isinstance(scale_cfg, (int, float)):
        return float(scale_cfg)
    return float(default)


def _image_plane_mesh_grid_from_settings(settings, mask_2d):
    """Overlay image-mesh vertices (+ circular edge points) for Delaunay."""
    mesh_shape = tuple(settings.get("mesh_shape", [32, 32]))
    edge_pixels = int(settings.get("delaunay_edge_pixels", 30))
    mask_radius = float(settings["mask_radius"])

    image_mesh = al.image_mesh.Overlay(shape=mesh_shape)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask_2d)
    return al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask_2d.mask_centre,
        radius=mask_radius + mask_2d.pixel_scale / 2.0,
        n_points=edge_pixels,
    )


def build_deconv_model(settings, *, fixed_coefficient=None, mask_2d=None):
    """
    Build autofit model: identity lens + pixelization.

    Rectangular meshes use ``Constant`` / ``Adapt``.
    ``rectangular_adapt_image`` keeps a fixed rectangular topology but
    brightness-weights pixels via the dirty-image adapt map.
    Delaunay meshes use ``ConstantSplit`` / ``AdaptSplit`` (+ optional zeroth).
    """
    mesh_type = settings.get("mesh_type", "rectangular_uniform")
    reg_cfg = settings.get("regularization", {})
    reg_type = reg_cfg.get("type", "constant")

    if reg_type in _SPLIT_REG_TYPES and mesh_type != "delaunay":
        raise ValueError(
            f"regularization.type={reg_type!r} requires mesh_type='delaunay'."
        )

    lens = identity_lens_galaxy(redshift=settings.get("redshift_lens", 0.5))

    if mesh_type == "delaunay":
        if mask_2d is None:
            raise ValueError("Delaunay mesh requires mask_2d to build the image mesh.")
        image_plane_mesh_grid = _image_plane_mesh_grid_from_settings(settings, mask_2d)
        edge_pixels = int(settings.get("delaunay_edge_pixels", 30))
        areas_factor = float(settings.get("delaunay_areas_factor", 0.5))
        mesh = af.Model(
            al.mesh.Delaunay,
            pixels=int(image_plane_mesh_grid.shape[0]),
            zeroed_pixels=edge_pixels,
            areas_factor=areas_factor,
        )
        # Stash grid for adapt_images builder (avoid recomputing inconsistently).
        settings["_image_plane_mesh_grid"] = image_plane_mesh_grid
    elif mesh_type == "rectangular_uniform":
        mesh_shape = tuple(settings["mesh_shape"])
        mesh = af.Model(al.mesh.RectangularUniform, shape=mesh_shape)
    elif mesh_type == "rectangular_adapt_image":
        mesh_shape = tuple(settings["mesh_shape"])
        mesh = af.Model(
            al.mesh.RectangularAdaptImage,
            shape=mesh_shape,
            weight_power=float(settings.get("weight_power", 1.0)),
            weight_floor=float(settings.get("weight_floor", 0.0)),
        )
    else:
        raise ValueError(
            f"Unsupported mesh_type: {mesh_type!r}. "
            "Choose 'rectangular_uniform', 'rectangular_adapt_image', or 'delaunay'."
        )

    regularization = _build_regularization_model(
        reg_cfg, reg_type, fixed_coefficient=fixed_coefficient
    )
    pixelization = af.Model(
        al.Pixelization, mesh=mesh, regularization=regularization
    )
    source = af.Model(
        al.Galaxy,
        redshift=settings.get("redshift_source", 1.0),
        pixelization=pixelization,
    )
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def reconstruction_mask_from_settings(settings):
    """
    Circular real-space mask for the inversion.

    Extends to ``mask_radius`` (science FOV/2 plus optional pad). Pixel scale
    is set by ``fov / science_n_pixels``.
    """
    mask_n_pixels = int(settings["mask_n_pixels"])
    pixel_scale = float(settings["mask_pixel_scale"])
    mask_radius = float(settings["mask_radius"])
    return al.Mask2D.circular(
        shape_native=(mask_n_pixels, mask_n_pixels),
        pixel_scales=pixel_scale,
        radius=mask_radius,
    )


def inversion_settings_from_settings(settings):
    """Build ``al.Settings`` from runner flags (positivity / edge / border)."""
    return al.Settings(
        use_positive_only_solver=bool(settings.get("use_positive_only_solver", True)),
        use_edge_zeroed_pixels=bool(settings.get("use_edge_zeroed_pixels", False)),
        use_border_relocator=bool(settings.get("use_border_relocator", False)),
    )


def coefficient_from_instance(instance):
    """Extract a summary regularization value from a fit instance."""
    reg = instance.galaxies.source.pixelization.regularization
    if hasattr(reg, "coefficient"):
        return float(reg.coefficient)
    if hasattr(reg, "inner_coefficient") and hasattr(reg, "outer_coefficient"):
        # Geometric mean as a single summary number for logs/titles.
        return float(
            np.sqrt(float(reg.inner_coefficient) * float(reg.outer_coefficient))
        )
    raise AttributeError("Could not extract a regularization coefficient summary.")


def adapt_images_for_dataset(settings, dataset, mask_2d=None):
    """
    Build ``AdaptImages`` when the mesh or regularization needs external maps.

    - Delaunay: image-plane mesh vertex grid
    - Adapt / AdaptSplit / rectangular_adapt_image: dirty-image brightness map
    """
    mesh_type = settings.get("mesh_type", "rectangular_uniform")
    reg_type = settings.get("regularization", {}).get("type", "constant")

    image_dict = {}
    mesh_dict = {}

    needs_dirty = reg_type in _ADAPT_REG_TYPES or mesh_type == "rectangular_adapt_image"
    if needs_dirty:
        dirty_image = dataset.dirty_image
        if dirty_image is None:
            dirty_image = dataset.masked_dirty_image
        image_dict[SOURCE_PATH] = dirty_image

    if mesh_type == "delaunay":
        grid = settings.get("_image_plane_mesh_grid")
        if grid is None:
            if mask_2d is None:
                mask_2d = dataset.mask
            grid = _image_plane_mesh_grid_from_settings(settings, mask_2d)
            settings["_image_plane_mesh_grid"] = grid
        mesh_dict[SOURCE_PATH] = grid

    if not image_dict and not mesh_dict:
        return None

    kwargs = {}
    if image_dict:
        kwargs["galaxy_name_image_dict"] = image_dict
    if mesh_dict:
        kwargs["galaxy_name_image_plane_mesh_grid_dict"] = mesh_dict
    return al.AdaptImages(**kwargs)

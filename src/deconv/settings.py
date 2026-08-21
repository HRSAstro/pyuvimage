"""Load and validate deconvolution runner settings."""

from __future__ import annotations

import json
from pathlib import Path

MODES = frozenset({"mfs", "cube"})
RECONSTRUCTORS = frozenset({"pixelization", "log_sky", "linear_sky", "auto"})
SKY_RECONSTRUCTORS = frozenset({"log_sky", "linear_sky", "auto"})
MESH_TYPES = frozenset(
    {"rectangular_uniform", "rectangular_adapt_image", "delaunay"}
)
REG_TYPES = frozenset(
    {
        "constant",
        "adapt",
        "constant_split",
        "adapt_split",
        "adapt_split_zeroth",
        "gaussian_kernel",
        "matern_kernel",
    }
)
_SPLIT_REG_TYPES = frozenset({"constant_split", "adapt_split", "adapt_split_zeroth"})
_KERNEL_REG_TYPES = frozenset({"gaussian_kernel", "matern_kernel"})
_RECT_MESH_TYPES = frozenset({"rectangular_uniform", "rectangular_adapt_image"})
CUBE_REG_MODES = frozenset({"from_mfs", "per_channel"})


def resolve_settings_path(path, repo_root=None):
    path = Path(path)
    if path.is_file():
        return path

    candidates = [path]
    if repo_root is not None:
        repo_root = Path(repo_root)
        candidates.extend(
            [
                repo_root / path,
                repo_root / "settings" / "runners" / path,
                repo_root / "settings" / "runners" / path.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Settings file not found: {path} (tried: {tried})")


def load_settings(path, repo_root=None):
    settings_path = resolve_settings_path(path=path, repo_root=repo_root)
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_settings(settings):
    """
    Validate runner settings. Mutates defaults in place where safe.

    Requires ``fov`` (arcsec). Sets defaults for ``mode``, mesh, regularization.
    """
    if "fov" not in settings:
        raise KeyError(
            "settings['fov'] is required (full field of view in arcsec)."
        )
    fov = float(settings["fov"])
    if fov <= 0.0:
        raise ValueError(f"fov must be positive; got {fov!r}")

    for key in ("config_path", "output_path", "data_directory"):
        if key not in settings:
            raise KeyError(f"settings['{key}'] is required.")

    if "data_patterns" not in settings:
        raise KeyError("settings['data_patterns'] is required.")

    mode = str(settings.get("mode", "mfs")).lower()
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode!r}. Choose from {sorted(MODES)}.")
    settings["mode"] = mode

    mesh_type = settings.get("mesh_type", "rectangular_uniform")
    if mesh_type not in MESH_TYPES:
        raise ValueError(
            f"Unsupported mesh_type: {mesh_type!r}. "
            f"Choose from {sorted(MESH_TYPES)}."
        )
    settings["mesh_type"] = mesh_type

    reg_cfg = settings.setdefault("regularization", {})
    if mesh_type == "delaunay":
        reg_type = reg_cfg.get("type", "adapt_split")
    else:
        reg_type = reg_cfg.get("type", "constant")
    if reg_type not in REG_TYPES:
        raise ValueError(
            f"Unsupported regularization.type: {reg_type!r}. "
            f"Choose from {sorted(REG_TYPES)}."
        )
    if reg_type in _SPLIT_REG_TYPES and mesh_type != "delaunay":
        raise ValueError(
            f"regularization.type={reg_type!r} requires mesh_type='delaunay'."
        )
    if mesh_type in _RECT_MESH_TYPES and reg_type in _SPLIT_REG_TYPES:
        raise ValueError(
            f"{mesh_type} cannot use *_split regularization; "
            "use 'constant', 'adapt', 'gaussian_kernel', or 'matern_kernel', "
            "or set mesh_type='delaunay'."
        )
    if reg_type in _KERNEL_REG_TYPES:
        reg_cfg.setdefault("scale", 0.15)
        if float(reg_cfg["scale"]) <= 0.0:
            raise ValueError(
                f"regularization.scale must be positive; got {reg_cfg['scale']!r}"
            )
        if reg_type == "matern_kernel":
            reg_cfg.setdefault("nu", 1.5)
            if float(reg_cfg["nu"]) <= 0.0:
                raise ValueError(
                    f"regularization.nu must be positive; got {reg_cfg['nu']!r}"
                )
    reg_cfg["type"] = reg_type
    reg_cfg.setdefault("prior_type", "log_uniform")
    settings.setdefault("delaunay_edge_pixels", 30)
    settings.setdefault("mask_pad_pixels", 0)
    settings.setdefault("use_edge_zeroed_pixels", False)
    settings.setdefault("use_border_relocator", False)
    if mesh_type == "rectangular_adapt_image":
        settings.setdefault("weight_power", 1.0)
        settings.setdefault("weight_floor", 0.0)

    cube_cfg = settings.setdefault("cube", {})
    cube_reg = cube_cfg.get("regularization", "from_mfs")
    if cube_reg not in CUBE_REG_MODES:
        raise ValueError(
            f"Unsupported cube.regularization: {cube_reg!r}. "
            f"Choose from {sorted(CUBE_REG_MODES)}."
        )
    cube_cfg["regularization"] = cube_reg

    reconstructor = str(settings.get("reconstructor", "pixelization")).lower()
    if reconstructor not in RECONSTRUCTORS:
        raise ValueError(
            f"Unsupported reconstructor: {reconstructor!r}. "
            f"Choose from {sorted(RECONSTRUCTORS)}."
        )
    settings["reconstructor"] = reconstructor
    if reconstructor in SKY_RECONSTRUCTORS:
        sky_auto = settings.setdefault("sky_auto", {})
        sky_auto.setdefault("snr_threshold", 100.0)
        if float(sky_auto["snr_threshold"]) <= 0.0:
            raise ValueError(
                f"sky_auto.snr_threshold must be > 0; got {sky_auto['snr_threshold']!r}"
            )

        # When auto, prepare defaults for both sky blocks (choice made at runtime).
        sky_keys = (
            ("log_sky", "linear_sky")
            if reconstructor == "auto"
            else (reconstructor,)
        )
        for sky_key in sky_keys:
            _validate_sky_block(
                settings.setdefault(sky_key, {}),
                sky_key=sky_key,
                default_smooth_auto=(reconstructor == "auto"),
            )

    settings.setdefault("source_pixel_scale", "nyquist")
    settings.setdefault("transformer", "dft")
    settings.setdefault("use_positive_only_solver", True)
    settings.setdefault("use_jax", True)
    settings.setdefault("redshift_lens", 0.5)
    settings.setdefault("redshift_source", 1.0)
    settings.setdefault("uids", ["dataset"])
    settings.setdefault("width", "native")

    search = settings.setdefault("search", {})
    search.setdefault("path_prefix", "deconv")
    search.setdefault("name", mode)
    search.setdefault("optimizer", "LBFGS")
    search.setdefault("maxiter", 200)
    search.setdefault(
        "figure_of_merit", "log_likelihood_with_regularization"
    )

    return settings


def _validate_sky_block(sky_cfg, *, sky_key, default_smooth_auto=False):
    """Validate / default one of ``log_sky`` or ``linear_sky`` config blocks."""
    if sky_key == "log_sky":
        sky_cfg.setdefault("i0", "auto")
    if default_smooth_auto:
        sky_cfg.setdefault("smooth", "auto")
    else:
        sky_cfg.setdefault("smooth", 1.0)

    smooth_cfg = sky_cfg["smooth"]
    optimize = bool(sky_cfg.get("optimize_smooth", False))
    if isinstance(smooth_cfg, str):
        if smooth_cfg.lower() != "auto":
            raise ValueError(
                f"{sky_key}.smooth must be a non-negative float or 'auto'; "
                f"got {smooth_cfg!r}"
            )
        optimize = True
        sky_cfg["optimize_smooth"] = True
    else:
        if float(smooth_cfg) < 0.0:
            raise ValueError(
                f"{sky_key}.smooth must be >= 0; got {smooth_cfg!r}"
            )
        sky_cfg.setdefault("optimize_smooth", False)
        optimize = bool(sky_cfg["optimize_smooth"])

    defaults_init = {"log_sky": 1.0e4, "linear_sky": 1.0e6}
    sky_cfg.setdefault("smooth_init", defaults_init[sky_key])
    sky_cfg.setdefault("smooth_bounds", [1.0e-2, 1.0e10])
    sky_cfg.setdefault("edge_prior_ratio", 100.0)
    sky_cfg.setdefault("maxiter", 200)
    sky_cfg.setdefault("edge_frac", 0.0)
    sky_cfg.setdefault("pixel_scale", "mask")

    if float(sky_cfg["smooth_init"]) <= 0.0:
        raise ValueError(
            f"{sky_key}.smooth_init must be > 0; got {sky_cfg['smooth_init']!r}"
        )
    bounds = sky_cfg["smooth_bounds"]
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        raise ValueError(
            f"{sky_key}.smooth_bounds must be [min, max]; got {bounds!r}"
        )
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (lo > 0.0 and hi > lo):
        raise ValueError(
            f"{sky_key}.smooth_bounds must satisfy 0 < min < max; got {bounds!r}"
        )
    sky_cfg["smooth_bounds"] = [lo, hi]
    if float(sky_cfg["edge_prior_ratio"]) < 0.0:
        raise ValueError(
            f"{sky_key}.edge_prior_ratio must be >= 0; "
            f"got {sky_cfg['edge_prior_ratio']!r}"
        )
    if int(sky_cfg["maxiter"]) < 1:
        raise ValueError(
            f"{sky_key}.maxiter must be >= 1; got {sky_cfg['maxiter']!r}"
        )
    sky_cfg.setdefault(
        "maxiter_trial",
        max(100, min(int(sky_cfg["maxiter"]), int(sky_cfg["maxiter"]) // 5 or 100)),
    )
    if int(sky_cfg["maxiter_trial"]) < 1:
        raise ValueError(
            f"{sky_key}.maxiter_trial must be >= 1; got {sky_cfg['maxiter_trial']!r}"
        )

    ps_cfg = sky_cfg["pixel_scale"]
    if isinstance(ps_cfg, str):
        if ps_cfg.lower() not in {"mask", "auto", "nyquist"}:
            raise ValueError(
                f"{sky_key}.pixel_scale must be 'mask', 'auto', 'nyquist', or "
                f"a positive float (arcsec); got {ps_cfg!r}"
            )
    else:
        if float(ps_cfg) <= 0.0:
            raise ValueError(
                f"{sky_key}.pixel_scale must be positive; got {ps_cfg!r}"
            )

    edge_frac = float(sky_cfg["edge_frac"])
    if edge_frac < 0.0 or edge_frac > 1.0:
        raise ValueError(
            f"{sky_key}.edge_frac must be in [0, 1]; got {edge_frac!r}"
        )
    if edge_frac > 0.0 and not optimize and not isinstance(smooth_cfg, str):
        # Fixed-smooth path: default absolute edge_prior if missing.
        sky_cfg.setdefault(
            "edge_prior", float(sky_cfg["edge_prior_ratio"]) * float(smooth_cfg)
        )
        if float(sky_cfg["edge_prior"]) < 0.0:
            raise ValueError(
                f"{sky_key}.edge_prior must be >= 0; got {sky_cfg['edge_prior']!r}"
            )
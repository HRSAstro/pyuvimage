"""Build AutoFit optimizers for regularization searches."""

import logging
import os

import autofit as af

logger = logging.getLogger(__name__)


def resolve_number_of_cores(search_cfg):
    env_cores = os.environ.get("PYUVIMAGE_CORES")
    if env_cores is not None and str(env_cores).strip() != "":
        cores = str(env_cores).strip().lower()
    else:
        cores = search_cfg.get("number_of_cores", "auto")
        if cores is not None:
            cores = str(cores).strip().lower()

    if cores in (None, "", "auto"):
        available = os.cpu_count() or 1
        return max(1, available - 1)
    return max(1, int(cores))


_OPTIMIZER_CLASSES = {
    "LBFGS": af.LBFGS,
    "BFGS": af.BFGS,
}


def build_optimizer_from_settings(search_cfg):
    """Build LBFGS / BFGS from a settings ``search`` block."""
    search_cfg = dict(search_cfg)
    optimizer_name = search_cfg.get("optimizer", "LBFGS").upper()
    try:
        optimizer_cls = _OPTIMIZER_CLASSES[optimizer_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Choose from {sorted(_OPTIMIZER_CLASSES)}."
        ) from exc

    number_of_cores = resolve_number_of_cores(search_cfg)
    kwargs = {}
    for key in (
        "maxiter",
        "maxfun",
        "ftol",
        "gtol",
        "eps",
        "maxls",
        "visualize",
        "iterations_per_quick_update",
        "iterations_per_full_update",
        "unique_tag",
    ):
        if key in search_cfg:
            kwargs[key] = search_cfg[key]

    logger.info(
        "%s optimization: number_of_cores=%s, kwargs=%s",
        optimizer_name,
        number_of_cores,
        kwargs or "(config defaults)",
    )

    return optimizer_cls(
        path_prefix=search_cfg["path_prefix"],
        name=search_cfg["name"],
        number_of_cores=number_of_cores,
        **kwargs,
    )

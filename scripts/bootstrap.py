import os
import sys
from pathlib import Path


def repo_root(from_file):
    for candidate in Path(from_file).resolve().parents:
        if (candidate / "src" / "deconv").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise RuntimeError("Could not locate pyuvimage repository root")


def repo_root_from_search_paths():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        candidate = candidate.resolve()
        if (candidate / "src" / "deconv").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate pyuvimage repository root. "
        "Change directory to the pyuvimage workspace before running."
    )


def ensure_import_path(from_file=None):
    if from_file is not None:
        try:
            root = repo_root(from_file)
        except RuntimeError:
            root = repo_root_from_search_paths()
    else:
        root = repo_root_from_search_paths()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def setup(from_file=None):
    root = ensure_import_path(from_file=from_file)
    os.chdir(root)
    from src.utils.jax_compat import ensure_numpy_jax_stub

    ensure_numpy_jax_stub()
    return root


def bootstrap_script(from_file=None):
    """
    Insert the repo on ``sys.path`` and ``chdir`` to it.

    Works when the script is run normally (``__file__`` set) or when CASA
    executes it via ``exec(open(...).read())`` (no ``__file__``).
    """
    return setup(from_file=from_file)

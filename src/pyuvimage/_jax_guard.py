"""Neutralise a JAX that is installed but cannot be imported.

The PyAuto stack decides whether JAX is available with
``importlib.util.find_spec("jax")`` -- which only asks whether the package is
*on disk*, and never imports it.  A broken JAX install therefore passes that
check, and the first code path that actually does ``import jax.numpy`` blows
up.  Worse, `autoarray.abstract_ndarray.__getitem__` guards its import with
``except ImportError``, so a failure of any other kind -- a jax/jaxlib version
mismatch raises ``AttributeError: partially initialized module 'jax' has no
attribute 'version'`` -- escapes as an unhandled exception from deep inside
array indexing, thousands of lines from anything the user did.

So: import JAX once, up front, before any of the PyAuto packages get a chance
to look for it.  If that fails, install an import hook that makes every later
``import jax`` raise a plain ``ImportError`` -- the case the libraries already
handle by falling back to NumPy.

The hook blocks at the *loader*, not the finder, and that detail matters.  The
obvious shortcut, ``sys.modules["jax"] = None``, breaks scipy: its
``array_api_compat`` helper asks ``sys.modules["jax"]`` for an attribute
whenever it type-checks an array, so a ``None`` there turns every
``scipy.linalg.block_diag`` call into ``'NoneType' object has no attribute
'Array'``.  Leaving ``sys.modules`` clean and failing during module execution
keeps both libraries on the paths they already support.

This does not fix JAX.  It turns a broken optional dependency into an absent
one, which is what it effectively is, and says so loudly enough to act on.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("pyuvimage")

# Set when JAX was neutralised.  The guard has to run at import time, which is
# before a CLI has configured logging, so the message would go straight to the
# NullHandler and nobody would ever see it.  Stash it and let the entry points
# report it once logging is up.
DISABLED_REASON: str | None = None
_REPORTED = False

MESSAGE = (
    "JAX is installed but cannot be imported (%s: %s).\n"
    "  Falling back to the NumPy path, which is fully supported -- expect "
    "fits to take longer on large datasets.\n"
    "  This is a broken JAX install, not a pyuvimage problem. To fix it, in "
    "the environment you run pyuvimage from:\n"
    "      python -c 'import jax; print(jax.__version__)'   # see the real error\n"
    "      pip uninstall -y jax jaxlib jax-metal\n"
    "      pip install -U 'jax[cpu]'                        # matched pair\n"
    "  A jax/jaxlib version mismatch is the usual cause; on Apple silicon the "
    "jax-metal plugin is another. Mixing conda-forge and pip installs of jax "
    "in one environment does it too.\n"
    "  If you would rather not use JAX at all, `pip uninstall jax jaxlib` is "
    "a clean answer."
)


class _BrokenJaxLoader:
    """A loader that refuses to build the module, with a useful message."""

    def __init__(self, reason: str):
        self.reason = reason

    def create_module(self, spec):
        raise ImportError(
            f"pyuvimage disabled this JAX install because it cannot be "
            f"imported ({self.reason}); running on the NumPy path instead",
            name=spec.name,
        )

    def exec_module(self, module):  # pragma: no cover - never reached
        raise ImportError(self.reason, name=getattr(module, "__name__", "jax"))


class _BlockJaxFinder:
    """Returns a spec for `jax*` whose loader always fails.

    Deliberately *returns* a spec rather than raising: the PyAuto stack calls
    ``importlib.util.find_spec("jax")`` to decide whether JAX exists, and an
    exception out of that would be a new crash rather than a fixed one.
    """

    def __init__(self, reason: str):
        self.reason = reason

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            from importlib.machinery import ModuleSpec

            return ModuleSpec(fullname, _BrokenJaxLoader(self.reason))
        return None


def disable_broken_jax() -> str | None:
    """Return a description of the failure if JAX was neutralised, else None."""
    if "jax" in sys.modules:
        return None  # already imported successfully by someone else

    try:
        import importlib.util

        if importlib.util.find_spec("jax") is None:
            return None  # genuinely not installed; nothing to do
    except Exception:  # a spec lookup that itself fails counts as broken
        pass

    try:
        import jax  # noqa: F401
    except BaseException as exc:  # noqa: BLE001 - anything at all disqualifies it
        reason = f"{type(exc).__name__}: {exc}"
        # drop whatever half-built submodules the failed import left behind
        for name in [k for k in sys.modules if k == "jax" or k.startswith("jax.")]:
            del sys.modules[name]
        sys.meta_path.insert(0, _BlockJaxFinder(reason))
        global DISABLED_REASON
        DISABLED_REASON = reason
        return reason
    return None


def report_if_disabled() -> None:
    """Log the warning once, from somewhere logging is actually configured."""
    global _REPORTED
    if DISABLED_REASON is None or _REPORTED:
        return
    _REPORTED = True
    kind, _, detail = DISABLED_REASON.partition(": ")
    logger.warning(MESSAGE, kind, detail)

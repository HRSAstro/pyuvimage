"""Helpers for hosts where jax/jaxlib cannot load (e.g. no AVX)."""

from __future__ import annotations

import sys
import types
import warnings

_USING_NUMPY_STUB = False
STUB_VERSION = 2  # bump when stub API changes; appears in RuntimeWarning


def using_numpy_jax_stub():
    return _USING_NUMPY_STUB


def jax_is_usable():
    """
    Return True if a working real ``jax`` / ``jaxlib`` can be imported.

    A NumPy-backed stub does **not** count as usable JAX.
    """
    if _USING_NUMPY_STUB:
        return False
    try:
        import jax  # noqa: F401
        import jax.numpy  # noqa: F401

        if getattr(jax, "_pyuvimage_numpy_stub", False):
            return False
        return True
    except Exception:
        return False


def _identity_decorator(fn=None, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


def _make_numpy_module(name, np):
    module = types.ModuleType(name)
    for attr in dir(np):
        if attr.startswith("__"):
            continue
        try:
            setattr(module, attr, getattr(np, attr))
        except Exception:
            pass
    module.ndarray = np.ndarray
    return module


def _install_submodule(full_name, module):
    sys.modules[full_name] = module
    parent_name, _, child = full_name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child, module)


def _purge_jax_modules():
    """Remove partial/broken jax imports left after a failed jaxlib load."""
    for name in list(sys.modules):
        if name == "jax" or name.startswith("jax.") or name == "jaxlib" or name.startswith(
            "jaxlib."
        ):
            del sys.modules[name]


def _stub_is_complete(jax_mod):
    return bool(
        getattr(jax_mod, "_pyuvimage_numpy_stub", False)
        and hasattr(jax_mod, "Array")
        and getattr(jax_mod, "_pyuvimage_stub_version", 0) >= STUB_VERSION
    )


def ensure_numpy_jax_stub():
    """
    If ``jax`` cannot load, install a NumPy-backed ``jax`` stub.

    Recent PyAutoArray / ``nufftax`` need ``from jax import Array`` at import
    time. A failed jaxlib import can leave a broken ``jax`` entry in
    ``sys.modules``; this helper purges that and installs a complete stub.
    """
    global _USING_NUMPY_STUB

    if "jax" in sys.modules and _stub_is_complete(sys.modules["jax"]):
        _USING_NUMPY_STUB = True
        return True

    reason = None
    try:
        import jax  # noqa: F401
        import jax.numpy  # noqa: F401

        if getattr(jax, "_pyuvimage_numpy_stub", False):
            if _stub_is_complete(jax):
                _USING_NUMPY_STUB = True
                return True
            reason = "incomplete pyuvimage jax stub (missing Array / outdated)"
        else:
            # Real jax imported successfully.
            return False
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"

    import numpy as np

    _purge_jax_modules()

    warnings.warn(
        "jax/jaxlib is unavailable ("
        f"{reason}). Installing pyuvimage jax-compat stub v{STUB_VERSION} "
        "(NumPy-backed, includes jax.Array) so PyAutoLens can import without "
        "JAX. Prefer installing a CPU-compatible jaxlib when possible.",
        RuntimeWarning,
        stacklevel=2,
    )

    jnp = _make_numpy_module("jax.numpy", np)

    jax = types.ModuleType("jax")
    jax.__file__ = __file__
    jax._pyuvimage_numpy_stub = True
    jax._pyuvimage_stub_version = STUB_VERSION
    jax.numpy = jnp
    # Required by nufftax: ``from jax import Array``
    jax.Array = np.ndarray
    jax.DeviceArray = np.ndarray
    jax.__all__ = [
        "Array",
        "DeviceArray",
        "numpy",
        "jit",
        "grad",
        "vmap",
        "pmap",
        "value_and_grad",
        "config",
        "lax",
        "random",
        "tree_util",
        "scipy",
        "typing",
    ]
    jax.jit = _identity_decorator
    jax.grad = _identity_decorator
    jax.vmap = _identity_decorator
    jax.pmap = _identity_decorator
    jax.value_and_grad = _identity_decorator
    jax.jacfwd = _identity_decorator
    jax.jacrev = _identity_decorator
    jax.hessian = _identity_decorator
    jax.checkpoint = _identity_decorator
    jax.remat = _identity_decorator
    jax.pure_callback = lambda cb, result_shape_dtypes, *args, **kwargs: cb(*args)
    jax.named_call = _identity_decorator
    jax.device_put = lambda x, device=None: np.asarray(x)
    jax.device_get = lambda x: np.asarray(x)
    jax.devices = lambda *args, **kwargs: ["cpu"]
    jax.local_devices = lambda *args, **kwargs: ["cpu"]
    jax.default_backend = lambda: "cpu"
    jax.shape = lambda x: np.shape(x)
    jax.typeof = lambda x: np.asarray(x).dtype
    jax.config = types.SimpleNamespace(
        update=lambda *args, **kwargs: None,
        read=lambda *args, **kwargs: None,
        values={},
    )

    lax = types.ModuleType("jax.lax")
    lax.stop_gradient = lambda x: x
    lax.cond = (
        lambda pred, true_fun, false_fun, *operands: true_fun(*operands)
        if pred
        else false_fun(*operands)
    )
    lax.scan = None
    lax.fori_loop = None
    lax.while_loop = None

    random = types.ModuleType("jax.random")
    random.PRNGKey = lambda seed: np.array([0, int(seed)], dtype=np.uint32)
    random.split = lambda key, num=2: [key for _ in range(num)]
    random.normal = lambda key, shape=(): np.random.normal(size=shape)
    random.uniform = lambda key, shape=(): np.random.uniform(size=shape)

    tree_util = types.ModuleType("jax.tree_util")
    tree_util.tree_map = lambda f, tree, *rest: f(tree)
    tree_util.tree_leaves = lambda tree: [tree]
    tree_util.tree_structure = lambda tree: None
    tree_util.register_pytree_node = lambda *args, **kwargs: None
    tree_util.register_pytree_node_class = _identity_decorator

    scipy = types.ModuleType("jax.scipy")
    scipy_numpy = _make_numpy_module("jax.scipy.numpy", np)

    typing_mod = types.ModuleType("jax.typing")
    typing_mod.ArrayLike = np.ndarray

    sys.modules["jax"] = jax
    _install_submodule("jax.numpy", jnp)
    _install_submodule("jax.lax", lax)
    _install_submodule("jax.random", random)
    _install_submodule("jax.tree_util", tree_util)
    _install_submodule("jax.scipy", scipy)
    _install_submodule("jax.scipy.numpy", scipy_numpy)
    _install_submodule("jax.typing", typing_mod)
    jax.lax = lax
    jax.random = random
    jax.tree_util = tree_util
    jax.scipy = scipy
    jax.typing = typing_mod

    sys.modules["jaxlib"] = types.ModuleType("jaxlib")
    _USING_NUMPY_STUB = True
    return True

"""pyuvimage: easy image reconstruction of radio interferometric data by
forward modelling in the uv-plane."""

import logging

logging.getLogger("pyuvimage").addHandler(logging.NullHandler())

# Must run before anything imports the PyAuto packages: they decide whether
# JAX exists by looking for it on disk, not by importing it, so a broken JAX
# install is invisible to them until it crashes a fit.
from ._jax_guard import disable_broken_jax, enable_double_precision  # noqa: E402

# Before the guard, because the guard imports JAX and the 64-bit setting has
# to be in the environment first. JAX defaults to float32; the sparse
# inversion's curvature matrix is built through JAX, and in single precision
# it cannot match the dense NumPy path.
enable_double_precision()
disable_broken_jax()

from .grids import ImageGeometry, nyquist_pixel_scale_arcsec, resolve_geometry  # noqa: E402
from .uvdata import UVData  # noqa: E402

__version__ = "0.1.0"


def __getattr__(name):
    """`run` and `RunResult` load on first use.

    `api` pulls in autogalaxy, scipy.signal/optimize and astropy -- 1.3-2 s
    measured -- and `pyuvimage import` / `pyuvimage convert` need none of it.
    The eager import here made every CLI command pay for the fit's imports.
    """
    if name in ("run", "RunResult"):
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'pyuvimage' has no attribute {name!r}")

__all__ = [
    "run",
    "RunResult",
    "UVData",
    "ImageGeometry",
    "resolve_geometry",
    "nyquist_pixel_scale_arcsec",
    "__version__",
]

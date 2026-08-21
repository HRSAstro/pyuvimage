"""pyuvimage: easy image reconstruction of radio interferometric data by
forward modelling in the uv-plane."""

import logging

logging.getLogger("pyuvimage").addHandler(logging.NullHandler())

from .api import RunResult, run  # noqa: E402
from .grids import ImageGeometry, nyquist_pixel_scale_arcsec, resolve_geometry  # noqa: E402
from .uvdata import UVData  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "run",
    "RunResult",
    "UVData",
    "ImageGeometry",
    "resolve_geometry",
    "nyquist_pixel_scale_arcsec",
    "__version__",
]

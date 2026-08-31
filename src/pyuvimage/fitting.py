"""Core forward-modelling machinery built on PyAutoGalaxy.

The model is a freeform source on a uniform rectangular mesh, fitted to the
visibilities by a regularised linear inversion (normal equations
``(F + lam*H) s = D``).  The regularisation strength ``lam`` is chosen by the
Morozov discrepancy principle by default -- the strongest smoothing that
still fits the data to the noise level (chi^2 ~= N) -- so the model can
neither overfit the noise nor be needlessly smooth.  Bayesian-evidence
maximisation is available as an alternative criterion.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

import autogalaxy as ag

from .beam import DirtyImager
from .grids import ImageGeometry

logger = logging.getLogger("pyuvimage")

# The DFT is exact but O(n_vis * n_pix); beyond this many visibilities the
# NUFFT (JAX-based) is required for sane runtimes.
DFT_MAX_VIS = 20_000
# The direct DFT builds n_image_pixels x n_vis float64 temporaries, so its cost
# is the *product*, not either factor.
# Found the hard way on the first well-formed real dataset: 5158 visibilities
# (well under DFT_MAX_VIS) on a 30" field at 0.19" resolution is 384400 image
# pixels, and autoarray asked the OS for 14.8 GB.
#
# The first threshold was 2e8, reasoned from a single array being ~1.6 GB.
# That reasoning was wrong: autoarray holds several such temporaries at once,
# so the peak is a few times one array. Two measured points bracket it, both
# on J0116 (5158 visibilities) in a 7 GB container:
#
#   --pixel-scale auto     100x100 grid   5.2e7 elements (0.41 GB)  ran fine
#   --pixel-scale nyquist  168x168 grid   1.46e8 elements (1.17 GB) OOM-killed
#
# So the limit is somewhere between, and 1e8 (~0.8 GB per array) splits them.
# Below it nothing changes; above it `auto` reaches for a NUFFT, and if none
# is installed it warns and uses the DFT anyway -- so erring low costs little.
DFT_MAX_PRODUCT = 1e8

REGULARIZATIONS = ("gibbs", "adaptive", "matern", "gaussian", "exponential", "constant")

# Kernel (Gaussian-process) source priors carry a correlation length; these
# are the schemes for which `scale` is a free hyperparameter.
KERNEL_REGULARIZATIONS = ("matern", "exponential", "gaussian", "adaptive", "gibbs")

# Priors that additionally carry a spatial envelope (see envelope.py).
ENVELOPE_REGULARIZATIONS = ("gaussian",)

# Priors that need a first-pass model to adapt to (two-stage fits).
ADAPTIVE_REGULARIZATIONS = ("adaptive", "gibbs")

# Matern smoothness. nu=0.5 is the (rough) exponential kernel, nu=1.5 is once
# differentiable, nu->inf is a Gaussian. PyAutoLabs fit this with a
# Uniform(0.5, 5.5) prior; we fix it by default and expose it as a knob.
DEFAULT_NU = 1.5

# Hyperparameter search bounds, in log10. PyAutoLabs' shipped priors are
# LogUniform(1e-6, 1e6) on both coefficient and scale; the scale is bounded
# below by the pixel scale and above by the field of view at fit time, since
# values outside that range are meaningless for the image.
LOG_COEFFICIENT_BOUNDS = (-6.0, 6.0)
# ...but the coefficient's meaningful magnitude depends on the data units, so
# the bracket is extended up to here when the shipped range is too narrow.
MAX_LOG_COEFFICIENT = 18.0

# chi^2 = N is not always reachable. Positivity raises chi^2, and a mesh has
# only so much freedom, so the *constrained* fit can floor above the target
# while the unconstrained solve reaches it (PJ0116 at 245 GHz: constrained
# floor chi^2/N = 1.024 against a target of 1.0). Bisecting towards a target
# that cannot be met is not a near miss -- every trial reads "still too high",
# so the search walks the coefficient down to its lower bound, switches the
# prior off and overfits. Instead, when the floor is within
# CHI2_UNREACHABLE_FACTOR of the target, aim just above the floor: the
# strongest prior that still fits essentially as well as this model can.
#
# "Essentially as well" has to be measured in units of how well chi^2 is
# determined, not as a fixed percentage. The standard error of chi^2/N is
# sqrt(2/N), so a flat 5% -- which is what this used to be -- means very
# different things at different dataset sizes:
#
#     PJ0116   N =  10,316   5% = 3.6 sigma   (a defensible knee)
#     Ruby     N = 296,954   5% =  19 sigma   (a thousand-fold over-smoothing)
#     9io9     N = 328,524   5% =  20 sigma
#
# On Ruby that let the discrepancy criterion pick a coefficient ~1000x too
# strong while chi^2/N still read 1.069, leaving the whole ring in the
# residual map at 60 sigma. CHI2_FLOOR_SIGMAS multiples of sqrt(2/N) instead:
# on both large datasets k=2 lands within 0.1% of where the independent
# `structure` criterion puts the coefficient (Ruby 1.0233 vs 1.0225,
# 9io9 1.0321 vs 1.0298), which is a strong sign it is the right scale.
#: Smallest relative change in the reconstruction, between regularisation
#: coefficients twelve decades apart, that counts as the prior having any
#: effect at all. Below this the non-negative solver is ignoring it. A working
#: prior changes the model by order 1 across that range -- on Ruby the
#: structure ratio runs 0.228 to 3.51 -- so 1% is far below anything healthy
#: and well above solver noise.
POSITIVITY_PRIOR_RESPONSE = 0.01

CHI2_FLOOR_SIGMAS = 2.0
# Fallback when the sample count is unknown, and a cap so that a very small
# dataset -- where sqrt(2/N) is large -- cannot be given unlimited slack.
CHI2_FLOOR_TOLERANCE = 0.05
CHI2_FLOOR_TOLERANCE_MAX = 0.10
CHI2_UNREACHABLE_FACTOR = 1.3
# How far the constrained fit's chi^2 may sit from the target before the
# coefficient is re-bisected with the constrained solver. Same argument as
# above -- a flat 3% is 11 sigma at N = 3x10^5 -- so it follows sqrt(2/N) too,
# with a small absolute floor so the gate does not chase numerical noise on a
# very large dataset.
CHI2_REBISECT_TOLERANCE = 0.03
CHI2_REBISECT_FLOOR = 0.005

# The residual-structure criterion drives the residual map's rms to exactly
# what white noise of the same total power would give.
STRUCTURE_TARGET = 1.0

CRITERIA = ("discrepancy", "structure", "evidence")

# `--criterion auto` picks between the two chi^2-free-of-charge options on the
# one thing that decides which of them works: how much data there is per model
# parameter.
#
# `structure` is the better criterion wherever it is calibrated -- it drives
# the residual *map* to white, which is what "the fit is done" actually means,
# and on all three real datasets at ratio 1.0 the residual lands at 3.9-5.0
# sigma. But it is only calibrated where the residual map really would be
# white at chi^2 = N, and on a weakly constrained fit it is not: the demo mock
# (400 data points, 144 mesh pixels) sits at ratio 0.49 with chi^2/N = 0.999,
# and driving that to 1 over-smooths to chi^2/N = 1.59.
#
# The two regimes separate cleanly on data per parameter:
#
#     demo mock     2.8:1   discrepancy (structure over-smooths)
#     PJ0116        4.1:1   either -- 3.9 sigma both ways
#     Ruby        439:1     structure (discrepancy 6.3 sigma at ratio 1.43)
#     9io9        486:1     structure
#
# so a threshold of 10 selects `structure` exactly where `discrepancy` fails
# and leaves the small and marginal cases on the faster criterion, where it
# costs nothing.
CRITERION_AUTO_DATA_PER_PARAMETER = 10.0


def resolve_criterion(
    criterion: str, n_data: int | None = None, n_mesh_pixels: int | None = None
) -> str:
    """Turn ``"auto"`` into a concrete criterion; pass anything else through.

    Resolved once, early, so that everything downstream -- including the
    point-source retune, which only fires under `discrepancy` -- sees a
    concrete choice rather than having to re-derive it.
    """
    if criterion != "auto":
        return criterion
    if not n_data or not n_mesh_pixels:
        return "discrepancy"
    per_parameter = float(n_data) / float(n_mesh_pixels)
    if per_parameter >= CRITERION_AUTO_DATA_PER_PARAMETER:
        logger.info(
            "criterion auto -> structure: %.0f data points per model pixel "
            "(%d / %d), comfortably enough for the residual map to be white "
            "at chi^2 = N, and chi^2 is a weak discriminant at this size. "
            "Pass --criterion discrepancy to override.",
            per_parameter, n_data, n_mesh_pixels,
        )
        return "structure"
    logger.info(
        "criterion auto -> discrepancy: only %.1f data points per model pixel "
        "(%d / %d). The structure ratio is not calibrated on a fit this "
        "weakly constrained -- it reads well below 1 even at chi^2 = N -- so "
        "chi^2 is the safer selector here. Pass --criterion structure to "
        "override.",
        per_parameter, n_data, n_mesh_pixels,
    )
    return "discrepancy"

# Brightness-adaptive regularisation: ratio of the smoothing strength in
# bright (inner) vs faint (outer) regions, and the brightness contrast scale.
ADAPT_FLOOR = 1e-2   # prior width in the faintest regions, relative to the peak
# How strongly the adaptive prior's width tracks brightness, w = b^power.
# 2.0 is the default: measured against p=1 on the extended+compact mock it
# gives the better extended model without overfitting, which is why `adaptive`
# is the default prior.
ADAPT_POWER = 2.0
GIBBS_ELL_FLOOR = 0.25  # shortest correlation length, as a fraction of the beam


def pynufft_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import pynufft  # noqa: F401

        return True
    except Exception:
        return False


_PYNUFFT_CLASS = None


def pynufft_transformer_class():
    """A pynufft-backed transformer that agrees with `TransformerDFT`.

    Standalone rather than a subclass of autoarray's
    ``TransformerNUFFTPyNUFFT``, for two reasons:

    1. **That class no longer exists upstream.** PyAutoArray PR #475
       (2026.8.23.1) deleted it outright in favour of the JAX-native nufftax
       backend. Subclassing it made autoarray's optional legacy class a hard
       import-time dependency of this whole package, which is how one
       `pip install -U autoarray` produced an AttributeError before pyuvimage
       could even import. Vendoring the ~60 lines (it is MIT-licensed) keeps
       the no-JAX NUFFT path alive on every autoarray version.

    2. **It had a half-pixel bug.** It built ``self.shift`` -- the phase ramp
       aligning the NUFFT grid convention with the DFT's -- and never applied
       it, so it sat half a pixel from `TransformerDFT` in both axes. Measured:
       the discrepancy grows linearly with baseline length exactly as a phase
       error must (9% of the visibility rms at the shortest baselines, 38% at
       the longest), and applying ``shift`` once recovers the DFT to 1e-5. A
       fit built entirely on it still converges -- the offset is shared by the
       mapping matrix and the model -- but the sky lands half a pixel from
       where every DFT-computed product puts it. The vendored copy applies the
       shift in `visibilities_from` and its conjugate in `image_from`, so the
       adjoint identity <Rx, y> == <x, R^T y> is preserved.

    Built lazily so that importing pyuvimage never requires pynufft.
    `tests/test_pynufft_transformer.py` pins the agreement with the DFT, and
    `test_the_jax_nufft_agrees_with_the_dft` (skipped without JAX) asks the
    same question of nufftax.
    """
    global _PYNUFFT_CLASS
    if _PYNUFFT_CLASS is not None:
        return _PYNUFFT_CLASS
    try:  # pragma: no cover - environment dependent
        from pynufft.linalg.nufft_cpu import NUFFT_cpu
    except Exception:
        return None

    from astropy import units

    import autoarray as aa

    class TransformerPyNUFFT(NUFFT_cpu):
        # After autoarray's removed TransformerNUFFTPyNUFFT (MIT licence),
        # with the half-pixel shift applied; see pynufft_transformer_class.

        def __init__(self, uv_wavelengths, real_space_mask, xp=np, **kwargs):
            super().__init__()
            self.uv_wavelengths = np.asarray(uv_wavelengths, dtype=float)
            self.real_space_mask = real_space_mask
            self.grid = aa.Grid2D.from_mask(mask=real_space_mask).in_radians

            # pynufft wants uv in radians per pixel, scaled to [-pi, pi) at
            # the image grid's Nyquist frequency, in (v, u) order
            pixel_rad = self.grid.pixel_scales[0] * units.arcsec.to(units.rad)
            nyquist = 1.0 / (2.0 * pixel_rad)
            om = np.array(
                [
                    self.uv_wavelengths[:, 1] / nyquist * np.pi,
                    self.uv_wavelengths[:, 0] / nyquist * np.pi,
                ]
            ).T
            shape = self.grid.shape_native
            self.plan(om=om, Nd=shape, Kd=(2 * shape[0], 2 * shape[1]), Jd=(6, 6))

            # the half-pixel phase ramp aligning pynufft's grid convention
            # with TransformerDFT's
            half_pix = self.grid.pixel_scales[0] / 2.0 * units.arcsec.to(units.rad)
            self.shift = np.exp(
                -2.0j * np.pi * half_pix
                * (self.uv_wavelengths[:, 1] + self.uv_wavelengths[:, 0])
            )

            # What `image_from(use_adjoint_scaling=True)` multiplies by, to put
            # pynufft's internally-normalised adjoint back on the plain
            # mathematical adjoint's scale -- the one TransformerDFT and the
            # nufftax TransformerNUFFT already use. Same expression as
            # autoarray's own `TransformerNUFFTPyNUFFT.adjoint_scaling`.
            self.adjoint_scaling = float(4 * shape[0] * shape[1])

        def visibilities_from(self, image, xp=np):
            vis = self.forward(np.asarray(image.native.array)[::-1, :])
            return ag.Visibilities(visibilities=vis * self.shift)

        def image_from(self, visibilities, use_adjoint_scaling=False, xp=np):
            """Adjoint transform, optionally on the common adjoint scale.

            `use_adjoint_scaling` is **load-bearing here and nowhere else**.
            pynufft's adjoint applies its own internal IFFT normalisation,
            which leaves the result a factor `4 * N_y * N_x` below the plain
            mathematical adjoint that `TransformerDFT` and the nufftax
            `TransformerNUFFT` both return. autoarray's
            `Interferometer.apply_sparse_operator` passes `use_adjoint_scaling
            =True` when it builds the w-tilde operator's dirty image, so that
            `D` is on the same scale as the kernel `W~` regardless of which
            transformer produced it.

            This signature used to end in `**kwargs`, which swallowed the
            argument silently. The result was not a crash but a plausible
            image: `D` came out `4 N_y N_x` too small while `F` did not, so
            the sparse reconstruction had the right morphology at ~1/10000 of
            the right amplitude, the residual map kept 99.99% of the source
            (231 sigma on Ruby), and -- because an `F` that large swamps any
            `H = lambda C^-1` in the search range -- chi^2 was identical to
            eleven significant figures across twelve orders of magnitude of
            `lambda`, so the hyperparameter search had nothing to bisect and
            ran to the ceiling. Never accept this argument into `**kwargs`.
            """
            shifted = np.asarray(visibilities) * np.conj(self.shift)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = np.real(self.adjoint(shifted))[::-1, :]
            if use_adjoint_scaling:
                image = image * self.adjoint_scaling
            return aa.Array2D(values=image, mask=self.real_space_mask)

        def transform_mapping_matrix(self, mapping_matrix, xp=np):
            out = np.zeros(
                (self.uv_wavelengths.shape[0], mapping_matrix.shape[1]),
                dtype=complex,
            )
            native_index = (
                self.real_space_mask.derive_indexes.native_for_slim.astype(int)
            )
            shape = self.grid.shape_native
            for j in range(mapping_matrix.shape[1]):
                image_2d = np.zeros(shape, dtype=mapping_matrix.dtype)
                image_2d[native_index[:, 0], native_index[:, 1]] = (
                    mapping_matrix[:, j]
                )
                out[:, j] = self.forward(image_2d[::-1, :]) * self.shift
            return out

    _PYNUFFT_CLASS = TransformerPyNUFFT
    return _PYNUFFT_CLASS


_CHUNKED_NUFFT_CLASS = None


def chunked_nufft_transformer_class():
    """`TransformerNUFFT` with the mapping-matrix transform split into blocks.

    Upstream passes every mesh pixel through one batched `nufft2d2`, and
    nufftax materialises a `n_mesh x n_vis x nspread^2` complex gather buffer
    for it (see `nufftax_gather_gb`). On any real dataset that is tens to
    hundreds of GB, and the process is killed with no traceback -- `Killed: 9`
    and nothing else. autoarray's own `chunk_size` argument caps exactly this
    buffer, but `transform_mapping_matrix` is the one path that ignores it.

    Splitting by mesh pixel is the safe axis: the columns are independent, so
    the blocks concatenate with no accumulation and no change to the result.
    The cost is one nufft2d2 call per block instead of one in total.

    Built lazily, and never used unless the whole-stack call would not fit --
    `nufftax_block_columns` returns the full width whenever it would.
    """
    global _CHUNKED_NUFFT_CLASS
    if _CHUNKED_NUFFT_CLASS is not None:
        return _CHUNKED_NUFFT_CLASS

    class TransformerNUFFTChunked(ag.TransformerNUFFT):
        def transform_mapping_matrix(self, mapping_matrix, xp=np):
            n_src = int(mapping_matrix.shape[1])
            block = nufftax_block_columns(
                int(self.total_visibilities), n_src, eps=self.eps
            )
            if block >= n_src:
                return super().transform_mapping_matrix(mapping_matrix, xp=xp)
            if not getattr(self, "_chunk_reported", False):
                self._chunk_reported = True
                logger.warning(
                    "the JAX NUFFT would need %.0f GB to transform this "
                    "mapping matrix in one batch, so it is being split into "
                    "%d blocks of %d mesh pixel(s) -- correct, but %d times "
                    "the NUFFT calls, on every trial of the hyperparameter "
                    "search. `--transformer pynufft` avoids the batching "
                    "entirely and is the faster path at this size.",
                    nufftax_gather_gb(
                        int(self.total_visibilities), n_src, self.eps
                    ),
                    -(-n_src // block), block, -(-n_src // block),
                )
            parts = [
                super(TransformerNUFFTChunked, self).transform_mapping_matrix(
                    mapping_matrix[:, j : j + block], xp=xp
                )
                for j in range(0, n_src, block)
            ]
            return xp.concatenate(parts, axis=1)

    _CHUNKED_NUFFT_CLASS = TransformerNUFFTChunked
    return _CHUNKED_NUFFT_CLASS


def _require_pynufft():
    cls = pynufft_transformer_class()
    if cls is None:
        raise RuntimeError(
            "--transformer pynufft needs the pynufft package: "
            "`pip install pynufft`."
        )
    return cls


# Peak resident memory of one inversion, per (visibility x mesh pixel).
#
# The inversion builds a transformed mapping matrix of n_vis x n_mesh complex
# entries -- 16 bytes each -- and holds roughly three of them at once (the
# matrix, its real/imaginary split, and the curvature product). Measured on
# Ruby (148,477 visibilities) on the NumPy/pynufft path:
#
#     mesh 16 (256 px)    3.8e7 elements    1.90 GB    50 B/element
#     mesh 24 (576 px)    8.6e7 elements    3.78 GB    44 B/element
#     mesh 32 (1024 px)   1.5e8 elements    OOM >7 GB  (~46 B/element)
#
# so ~44 B/element plus a fixed overhead. This is a floor, not a guarantee:
# JAX holds its own buffers and can need more.
BYTES_PER_MAPPING_ELEMENT = 44
MEMORY_BASE_GB = 0.4


def estimate_peak_memory_gb(n_vis: int, n_mesh_pixels: int) -> float:
    """Rough peak RSS for one inversion, in GB.

    Note what it scales with: **n_vis x n_mesh**, not the number of image
    pixels. That is why a small field over a large dataset can need far more
    memory than a large field over a small one -- Ruby at a 26x26 mesh needs
    ~8x what PJ0116 needs at 50x50, because Ruby has 29x the visibilities.
    """
    elements = float(n_vis) * float(n_mesh_pixels)
    return MEMORY_BASE_GB + elements * BYTES_PER_MAPPING_ELEMENT / 1e9


# --------------------------------------------------------------------------
# The nufftax (JAX) path is a different memory regime entirely
# --------------------------------------------------------------------------
#
# `TransformerNUFFT.transform_mapping_matrix` scatters every mesh pixel into
# its own image and passes the whole stack through one batched `nufft2d2`
# call. Inside nufftax the type-2 interpolation materialises its gather
# buffer in full (`core/spread.py::interp_2d_impl`):
#
#     fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, nspread, nspread)
#     c = jnp.sum(fw_gathered * weights_2d[None], axis=(-2, -1))
#
# so the peak is `n_mesh x n_vis x nspread^2` complex128, twice over -- the
# gather and the weighted product are separate arrays, and nufftax is not
# jitted, so nothing fuses them away. autoarray knows about this buffer (its
# `chunk_size` argument exists to cap it) but `transform_mapping_matrix`
# never uses `chunk_size`; only the plain forward/adjoint calls do.
#
# `nspread` follows the requested precision, and autoarray asks for
# `eps=1e-12` by default, which is the widest kernel nufftax will build.
NUFFTAX_DEFAULT_EPS = 1e-12
NUFFTAX_MAX_KERNEL_WIDTH = 16
# Fraction of available memory the gather buffer may occupy before the
# batched call is split (or another transformer preferred).
NUFFTAX_GATHER_BUDGET = 0.25


def nufftax_kernel_width(eps: float = NUFFTAX_DEFAULT_EPS) -> int:
    """`nspread` for a requested precision -- nufftax's own heuristic.

    From `nufftax.core.kernel.compute_kernel_params`: `ceil(log10(1/eps) + 1)`,
    rounded up to even and capped at 16. eps=1e-12 gives 14, so 196 grid taps
    per visibility per mesh pixel; eps=1e-6 gives 8, so 64.
    """
    width = int(math.ceil(math.log10(1.0 / float(eps)) + 1.0))
    width = max(2, min(width, NUFFTAX_MAX_KERNEL_WIDTH))
    return width + 1 if width % 2 else width


def nufftax_gather_gb(
    n_vis: int, n_mesh_pixels: int, eps: float = NUFFTAX_DEFAULT_EPS
) -> float:
    """Peak GB of the batched nufft2d2 gather buffer, in one call.

    This dwarfs everything else in the inversion. Ruby at a 20x20 mesh --
    148,477 visibilities, 400 mesh pixels, the fit whose mapping matrix is a
    harmless 0.5 GB -- needs 186 GB here.
    """
    width = nufftax_kernel_width(eps)
    elements = float(n_mesh_pixels) * float(n_vis) * width * width
    return 2.0 * elements * 16.0 / 1e9


def nufftax_block_columns(
    n_vis: int,
    n_mesh_pixels: int,
    eps: float = NUFFTAX_DEFAULT_EPS,
    available_gb: float | None = None,
) -> int:
    """Mesh pixels per batched nufft2d2 call that keep the gather buffer down.

    Returns `n_mesh_pixels` when the whole stack already fits, so the
    unchunked upstream call is used unchanged wherever it is affordable.
    """
    if available_gb is None:
        available_gb = available_memory_gb()
    if not available_gb or available_gb <= 0:
        return n_mesh_pixels
    per_column = nufftax_gather_gb(n_vis, 1, eps)
    if per_column <= 0:
        return n_mesh_pixels
    budget = NUFFTAX_GATHER_BUDGET * available_gb
    return int(max(1, min(n_mesh_pixels, math.floor(budget / per_column))))


def available_memory_gb() -> float | None:
    """Usable memory in GB, or None if it cannot be determined.

    Deliberately best-effort: a wrong number here must never stop a fit that
    would have worked.
    """
    try:  # pragma: no cover - platform dependent
        import psutil

        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        pass
    try:  # pragma: no cover - Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    try:  # pragma: no cover - macOS and other POSIX: total, not available
        import os

        return (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        )
    except Exception:
        return None


def transformer_memory_gb(transformer_cls, n_vis: int, n_mesh_pixels: int) -> float:
    """Extra GB the chosen transformer needs on top of the mapping matrix.

    Only the nufftax-backed `TransformerNUFFT` has one worth counting, and it
    is not a correction -- it is usually the whole bill. Chunking bounds it,
    so the figure follows the block size the transform will actually use.
    """
    if transformer_cls is None:
        return 0.0
    try:
        is_nufftax = issubclass(transformer_cls, ag.TransformerNUFFT)
    except TypeError:
        return 0.0
    if not is_nufftax:
        return 0.0
    chunked = _CHUNKED_NUFFT_CLASS is not None and issubclass(
        transformer_cls, _CHUNKED_NUFFT_CLASS
    )
    block = (
        nufftax_block_columns(n_vis, n_mesh_pixels)
        if chunked else n_mesh_pixels
    )
    return nufftax_gather_gb(n_vis, block)


def current_memory_gb() -> float:
    """Resident memory this process is already holding, in GB.

    It belongs in the budget. By the time a fit allocates, the interpreter,
    numpy, autoarray and -- if it is installed -- JAX with its own arena are
    already resident, and on one 8 GB laptop that was around a gigabyte before
    the first mapping matrix existed. Comparing the *allocation* against total
    available memory ignores it and reports a comfortable margin that is not
    there: a fit estimated at 2.1 GB against 6.9 GB available was killed.
    """
    try:  # pragma: no cover - platform dependent
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kB on Linux and bytes on macOS
        return r / 1e9 if r > 1e8 else r / 1e6
    except Exception:
        return 0.0


def check_memory(
    n_vis: int,
    n_mesh_pixels: int,
    transformer_cls=None,
    n_chan: int | None = None,
    prior_thin: int = 1,
    inversion: str = "dense",
    n_image_pixels: int | None = None,
    kernel_cached: bool = False,
) -> None:
    """Say what the fit will need, and warn before it is killed.

    An OOM kill gives the user nothing -- `Killed: 9` and no traceback -- so
    the useful moment to speak is before the allocation, while a smaller
    --mesh or --fov is still an option.

    ``transformer_cls`` matters more than everything else put together on the
    JAX path: the mapping matrix for Ruby at a 20x20 mesh is 0.5 GB, and the
    batched nufft2d2 that transforms it asks for 186 GB.

    ``n_chan`` reports the cube case honestly. Cube mode fits each channel
    separately, which is cheap, but it first runs one MFS fit over *every*
    channel's visibilities to fix the prior -- and that pass is `n_chan` times
    the size of any single channel. On Ruby CO(7-6) at a 27x27 mesh the
    per-channel fits need 2.9 GB each and the MFS pass needs 20.1 GB, so the
    cube looks unaffordable when only one step of it is.
    """
    held = current_memory_gb()
    if inversion == "sparse":
        # A different regime entirely, and the reason it exists: nothing here
        # scales with n_vis, so the dense estimate would refuse fits that are
        # comfortable. Report what will actually be allocated.
        if n_image_pixels is None:
            return
        chunk_k = sparse_chunk_k_for_budget(n_image_pixels)
        need = sparse_peak_memory_gb(
            n_image_pixels, n_mesh_pixels, chunk_k=chunk_k,
            kernel_cached=kernel_cached,
        ) + held
        have = available_memory_gb()
        logger.info(
            "sparse inversion: roughly %.1f GB (%.1f GB already resident), "
            "independent of the %d visibilities -- the kernel build streams "
            "them %d at a time onto a fixed %d-pixel grid",
            need, held, n_vis, chunk_k, n_image_pixels,
        )
        if n_chan and n_chan > 1:
            # The dense path's cube warning is about one pass over every
            # channel dwarfing the per-channel fits. There is no such pass
            # here: the estimate above is already per channel, and it is the
            # same for every channel because it depends on the image and the
            # mesh, not on how many visibilities the channel holds.
            logger.info(
                "  and the same for each of the %d channels: on the sparse "
                "path the cube costs no more memory than one channel, only "
                "more time. Each channel's kernel streams only its own "
                "visibilities, so the %d builds total one pass over the "
                "dataset.", n_chan, n_chan,
            )
        if have is not None and need > have:
            logger.warning(
                "estimated peak memory (%.1f GB) exceeds what looks "
                "available (%.1f GB). On the sparse path the ceiling is the "
                "model, not the data: reduce --mesh (F is n_mesh^2) or the "
                "image size (--fov, --pixel-scale).",
                need, have,
            )
        return

    def _need(nv):
        return (
            estimate_peak_memory_gb(nv, n_mesh_pixels)
            + transformer_memory_gb(transformer_cls, nv, n_mesh_pixels)
        )

    full = _need(n_vis) + held
    per_chan = (
        _need(max(n_vis // n_chan, 1)) + held
        if n_chan and n_chan > 1 else None
    )
    # What the run will actually peak at: in cube mode with a thinned prior
    # pass, nothing ever sees more than one channel's worth at a time.
    need = per_chan if (per_chan is not None and prior_thin > 1) else full
    have = available_memory_gb()
    if have is None:
        logger.info("this fit needs roughly %.1f GB of memory", need)
        return
    logger.info(
        "this fit needs roughly %.1f GB (%.1f GB already resident + %.1f GB "
        "to allocate); about %.1f GB is available",
        need, held, need - held, have,
    )
    if per_chan is not None:
        if prior_thin > 1:
            logger.info(
                "  one channel at a time, all the way through: the shared "
                "prior is fitted on a %d-fold thinned set too. A single pass "
                "over all %d channels would need %.1f GB (--cube-prior mfs).",
                prior_thin, n_chan, full,
            )
        else:
            logger.info(
                "  of which the per-channel fits need about %.1f GB each; "
                "the %.1f GB is the one MFS pass over all %d channels that "
                "fixes the prior", per_chan, full, n_chan,
            )
            if have is not None and full > have >= per_chan:
                logger.warning(
                    "the per-channel fits would fit (%.1f GB) but the MFS "
                    "pass that precedes them would not (%.1f GB against %.1f "
                    "GB available). --cube-prior channel fits the shared "
                    "prior on one channel's worth instead, which is the "
                    "default; a coarser --mesh or --pixel-scale shrinks both.",
                    per_chan, full, have,
                )
    if need > have:
        logger.warning(
            "estimated peak memory (%.1f GB) exceeds what looks available "
            "(%.1f GB), so this fit may be killed outright -- an OOM kill "
            "prints nothing useful. Memory scales as n_vis x n_mesh, so the "
            "levers are --mesh (or a coarser --pixel-scale) and --fov, not "
            "the image size. A mesh of %d per side would fit.",
            need, have, _mesh_that_fits(n_vis, have),
        )
    elif need > 0.8 * have:
        logger.info(
            "  that is most of the available memory; if it is killed, reduce "
            "--mesh or --fov"
        )


def _mesh_that_fits(n_vis: int, available_gb: float) -> int:
    """Largest square mesh whose estimate stays under 80% of `available_gb`."""
    budget = max(0.8 * available_gb - MEMORY_BASE_GB, 0.0) * 1e9
    n_pix = budget / (BYTES_PER_MAPPING_ELEMENT * max(n_vis, 1))
    return max(int(np.sqrt(max(n_pix, 1.0))), 1)


def jax_available() -> bool:
    """Whether the JAX-native NUFFT (`TransformerNUFFT`) can actually run.

    Both halves are required and they fail independently -- see
    `jax_path_diagnosis` for why that distinction is worth reporting.
    """
    return not jax_path_diagnosis()


def jax_path_diagnosis() -> str | None:
    """Why the JAX NUFFT is unavailable, or None if it is available.

    ``nufftax`` is what `TransformerNUFFT` is actually built on, and since
    PyAutoLens#702 it is **not** part of a default install: JAX itself moved
    into autonerves' base dependencies, while nufftax stayed behind in
    autoarray's ``optional`` extra. So the common case on a freshly built
    environment is a perfectly good JAX and no nufftax -- and the symptom is
    silence, because `_jax_guard` only speaks up when JAX is *broken*.
    Distinguish the two, or the fast path is missing for a reason nothing
    reports.

    The version floor is ours to care about too: nufftax <0.6.1 cannot
    differentiate a batched ``nufft2d2``, and ``transform_mapping_matrix`` --
    the call the inversion makes for every mesh pixel -- relies on it.
    """
    try:  # pragma: no cover - environment dependent
        import jax  # noqa: F401
    except Exception as e:
        return (
            f"JAX cannot be imported ({type(e).__name__}: {e}). See the "
            "startup warning from pyuvimage._jax_guard for how to repair it."
        )
    try:  # pragma: no cover - environment dependent
        import nufftax  # noqa: F401
    except Exception:
        return (
            "JAX works but `nufftax` is not installed, and that is what the "
            "JAX NUFFT is built on. It is deliberately not part of a default "
            "install (it lives in autoarray's `optional` extra), so a fresh "
            "environment usually lacks it. Fix with:\n"
            "      pip install 'nufftax>=0.6.1,<0.7.0'\n"
            "  The 0.6.1 floor matters here: earlier versions cannot "
            "differentiate a batched nufft2d2, which is the call the "
            "inversion makes for every mesh pixel."
        )
    return None


def _jax_nufft_class(n_vis: int, n_mesh_pixels: int | None):
    """`TransformerNUFFT`, chunked if its one-shot gather buffer will not fit.

    The chunked subclass is a no-op wherever the plain call is affordable, so
    this is safe to return unconditionally -- but only return the chunked one
    when we have the mesh size to judge with.
    """
    if not n_mesh_pixels:
        return ag.TransformerNUFFT
    if nufftax_block_columns(n_vis, n_mesh_pixels) >= n_mesh_pixels:
        return ag.TransformerNUFFT
    return chunked_nufft_transformer_class()


def resolve_transformer(
    n_vis: int,
    transformer: str = "auto",
    n_image_pixels: int | None = None,
    n_mesh_pixels: int | None = None,
):
    """Pick the Fourier transform implementation.

    Three quantities decide this, and each one kills a different backend:

    * ``n_image_pixels x n_vis`` -- the direct DFT's float64 temporary. A
      modest number of visibilities on a large field is just as fatal as a
      large number on a small one.
    * ``n_mesh_pixels x n_vis x nspread^2`` -- the JAX/nufftax gather buffer
      in the batched mapping-matrix transform (`nufftax_gather_gb`). This is
      by far the largest of the three and the one nothing upstream guards, so
      when it will not fit we prefer pynufft, whose mapping-matrix transform
      is a per-column loop that never allocates more than one column.
    * everything else -- `estimate_peak_memory_gb`.

    ``n_mesh_pixels`` is optional only so that older callers keep working; pass
    it whenever it is known, or the nufftax check cannot be made.
    """
    if not isinstance(transformer, str):
        return transformer  # already a class: pass it through untouched
    if transformer == "dft":
        return ag.TransformerDFT
    if transformer == "nufft":
        return _jax_nufft_class(n_vis, n_mesh_pixels)
    if transformer == "pynufft":
        return _require_pynufft()
    if transformer != "auto":
        raise ValueError(f"unknown transformer {transformer!r}")

    product = n_vis * (n_image_pixels or 1)
    too_big = n_vis > DFT_MAX_VIS or product > DFT_MAX_PRODUCT
    if not too_big:
        return ag.TransformerDFT
    if jax_available():
        gather = (
            nufftax_gather_gb(n_vis, n_mesh_pixels)
            if n_mesh_pixels else 0.0
        )
        have = available_memory_gb()
        fits = not have or gather <= NUFFTAX_GATHER_BUDGET * have
        if fits or not pynufft_available():
            return _jax_nufft_class(n_vis, n_mesh_pixels)
        logger.info(
            "the JAX NUFFT would need a %.0f GB gather buffer to transform "
            "the mapping matrix in one batch (%d mesh pixels x %d "
            "visibilities x a %d-tap kernel), against %.1f GB available: "
            "using the pynufft transformer instead, which loops over mesh "
            "pixels and never holds more than one at a time.",
            gather, n_mesh_pixels, n_vis,
            nufftax_kernel_width() ** 2, have,
        )
        return pynufft_transformer_class()
    if pynufft_available():
        # No JAX, but pynufft is a pure NumPy/SciPy NUFFT and is the
        # difference between "impossible" and "an hour": on 164k visibilities
        # over a 116x116 image the DFT cannot even allocate its 16.5 GB
        # temporary, while a pynufft transform takes 20 ms.
        logger.info(
            "%d visibilities on a %s-pixel image: using the pynufft "
            "transformer (no JAX installed).", n_vis, n_image_pixels,
        )
        return pynufft_transformer_class()
    if n_vis > DFT_MAX_VIS:
        why = f"{n_vis} visibilities"
    else:
        why = (
            f"{n_vis} visibilities on a {n_image_pixels}-pixel image "
            f"({product / 1e6:.0f}e6 DFT elements, ~{product * 8 / 1e9:.1f} GB "
            "per temporary)"
        )
    diagnosis = jax_path_diagnosis()
    if diagnosis:
        logger.info("the JAX NUFFT is unavailable: %s", diagnosis)
    warnings.warn(
        f"{why} with no NUFFT installed: falling back to the direct DFT, "
        "which will be slow and may run out of memory. `pip install pynufft` "
        "is the quickest fix and needs no JAX; pyuvimage[jax] is the fast "
        "path where JAX wheels exist. Reducing --fov also helps.",
        stacklevel=2,
    )
    return ag.TransformerDFT


# --------------------------------------------------------------------------
# The sparse (w-tilde) inversion: memory that does not scale with n_vis
# --------------------------------------------------------------------------
#
# The dense inversion builds an `n_vis x n_mesh` transformed mapping matrix
# and then throws it away, so peak memory scales with the number of
# visibilities: 21.6 GB for Ruby CO(7-6) at a 28x28 mesh, to produce an `F`
# that is 4.9 MB. CASA never does this -- tclean accumulates visibilities into
# a fixed-size uv grid and FFTs once, so its memory is set by the image.
#
# autoarray ships the equivalent for a regularised inversion. A
# translation-invariant kernel `W~` is accumulated over the visibilities in
# chunks -- shape (2 Ny, 2 Nx), sub-megabyte on every dataset we have -- and
# `F = A^T W~ A` is then assembled from sparse mapping triplets, a batch of
# source-pixel columns at a time, with no dense matrix anywhere.
#
# NOT yet verified against the dense path. An earlier version of this comment
# claimed "chi^2 = 305200.43 both ways, identical to eight significant
# figures, 0.3 s against 25.4 s". That measurement was worthless: the probe
# ran at a fixed coefficient of 1e8, above the top of the search range, which
# nulls the model on *both* paths -- two near-zero reconstructions compared
# and found equal. It is withdrawn. What has been measured since is a healthy
# sparse fit on Ruby (chi^2/N = 1.022, structure ratio 1.00, residual 4.3
# sigma), which agrees with the dense figure of record but was taken on
# different code. A like-for-like comparison in float64, at a coefficient
# that actually bites, is still owed.
SPARSE_CHUNK_K = 4096       # visibilities per chunk while building W~
SPARSE_BATCH_SIZE = 128     # source-pixel columns per batch while assembling F
SPARSE_KERNEL_SUFFIX = ".wtilde.npy"
SPARSE_CHUNK_K_MIN = 64
# Concurrent (Ny, Nx, chunk_k) float64 temporaries inside the kernel build.
# `accum_from_corner_np` evaluates
#     phase = dx[..., None] * ku[k0:k1] + dy[..., None] * kv[k0:k1]
#     acc  += np.sum(np.cos(phase) * w[k0:k1], axis=2)
# which is two temporaries for `phase` (the product, then the sum), `cos(phase)`
# while `phase` is still referenced, and the weighted product. Four is the
# honest ceiling; numpy frees some of them earlier.
SPARSE_KERNEL_BUILD_ARRAYS = 4
# Concurrent (batch, 2Ny, 2Nx) complex128 temporaries in `apply_operator`:
# F_pad, Fhat, Ghat, G_pad.
SPARSE_OPERATOR_BATCH_ARRAYS = 4


def sparse_kernel_build_gb(n_image_pixels: int, chunk_k: int = SPARSE_CHUNK_K) -> float:
    """Peak GB of the one-off W~ kernel build.

    This is the whole point of the sparse path, so read the scaling carefully:
    **n_image_pixels x chunk_k**. The visibility count is not in it. The build
    streams the data in chunks of `chunk_k` and accumulates into a fixed
    (2Ny, 2Nx) array, exactly as tclean streams visibilities onto a fixed uv
    grid -- so a dataset ten times larger costs ten times the *time* and not
    one byte more memory, and `chunk_k` trades the two against each other.
    """
    return (
        SPARSE_KERNEL_BUILD_ARRAYS
        * float(n_image_pixels) * float(chunk_k) * 8.0 / 1e9
    )


def sparse_peak_memory_gb(
    n_image_pixels: int,
    n_mesh_pixels: int,
    chunk_k: int = SPARSE_CHUNK_K,
    batch_size: int = SPARSE_BATCH_SIZE,
    kernel_cached: bool = False,
) -> float:
    """Rough peak RSS for one sparse inversion, in GB.

    Three terms, none of which involves n_vis:

    * the kernel build, `n_image x chunk_k` (skipped when a cached kernel is
      reused, which is why `--kernel-cache` is worth having);
    * the operator itself, a (2Ny, 2Nx) kernel and its FFT, plus the
      `batch_size` columns in flight while F is assembled;
    * F and the regularisation matrix, `n_mesh^2` each, plus the solver's copy.

    The last term is the one that eventually bites: the sparse path moves the
    ceiling off the data and onto the model, so it is `--mesh` that limits a
    sparse fit, not the size of the measurement set.
    """
    kernel = 0.0 if kernel_cached else sparse_kernel_build_gb(n_image_pixels, chunk_k)
    operator = (
        4.0 * float(n_image_pixels) * (8.0 + 16.0)              # W~ and Khat
        + SPARSE_OPERATOR_BATCH_ARRAYS
        * float(batch_size) * 4.0 * float(n_image_pixels) * 16.0  # padded FFTs
    ) / 1e9
    curvature = 3.0 * float(n_mesh_pixels) ** 2 * 8.0 / 1e9
    return MEMORY_BASE_GB + kernel + operator + curvature


def sparse_chunk_k_for_budget(
    n_image_pixels: int,
    available_gb: float | None = None,
    chunk_k: int = SPARSE_CHUNK_K,
    budget: float = 0.25,
) -> int:
    """Shrink `chunk_k` until the kernel build fits in `budget` of memory.

    A knob that tunes itself. `chunk_k` costs only time, so there is no reason
    to make the user discover it: pick the largest chunk that fits and say so
    in the log if it had to come down.
    """
    if available_gb is None:
        available_gb = available_memory_gb()
    if available_gb is None or available_gb <= 0:
        return int(chunk_k)
    allowance = budget * available_gb * 1e9 / (
        SPARSE_KERNEL_BUILD_ARRAYS * 8.0 * max(float(n_image_pixels), 1.0)
    )
    return int(max(SPARSE_CHUNK_K_MIN, min(float(chunk_k), math.floor(allowance))))


#: Visibility count at or above which `--inversion auto` prefers the sparse
#: w-tilde path. Below it the dense mapping matrix is small, already fast, and
#: the one path with a long track record; above it the dense build is what
#: costs the memory, and the sparse operator's cost stops depending on the
#: data at all.
SPARSE_AUTO_MIN_VISIBILITIES = 5000

#: Above this median re/im noise asymmetry, `auto` stays on the dense path:
#: the W~ reduction assumes sigma_re == sigma_im and degrades roughly in
#: proportion to the difference, and dense has no such assumption. Recentring
#: pools the two, so a recentred fit reads 0 here. Ruby unrecentred reads 9%.
SPARSE_AUTO_MAX_ASYMMETRY = 0.05


def resolve_inversion(
    inversion: str,
    n_vis: int,
    point_sources=False,
    transformer_cls=None,
    noise=None,
) -> str:
    """Turn `auto` into `dense` or `sparse`, and say why.

    `auto` prefers sparse on anything big enough for the dense mapping matrix
    to be the thing that costs -- but only when sparse can actually deliver
    the same answer. Every condition below is a case where dense is either
    faster, better conditioned, or the only one that works, and an explicit
    `--inversion sparse` still raises rather than falling back, because a user
    who asked for it by name wants to know it could not be given.
    """
    if inversion not in ("auto", "dense", "sparse"):
        raise ValueError(
            f"unknown inversion {inversion!r}: 'auto', 'dense' or 'sparse'"
        )
    if inversion != "auto":
        return inversion

    def _dense(why):
        logger.info("inversion auto -> dense: %s", why)
        return "dense"

    if n_vis < SPARSE_AUTO_MIN_VISIBILITIES:
        return _dense(
            f"only {n_vis} visibilities, below the {SPARSE_AUTO_MIN_VISIBILITIES} "
            "at which the w-tilde path starts to pay for its kernel build. "
            "Pass --inversion sparse to use it anyway"
        )
    if point_sources:
        return _dense(
            "point components were requested, and the sparse path cannot fit "
            "them yet (their cross-terms need the dense mapping matrix)"
        )
    reason = sparse_inversion_diagnosis()
    if reason is not None:
        return _dense(reason)
    if transformer_cls is not None:
        try:
            assert_adjoint_scale_consistent(transformer_cls)
        except Exception as e:
            return _dense(f"the transformer is not scale-consistent ({e})")
    if noise is not None:
        from .uvdata import reim_asymmetry

        asymmetry = reim_asymmetry(noise)
        if asymmetry > SPARSE_AUTO_MAX_ASYMMETRY:
            return _dense(
                f"sigma_re and sigma_im differ by {100 * asymmetry:.1f}%, and "
                "the w-tilde reduction assumes they are equal -- dense has no "
                "such assumption. Recentring (--image-centre) pools them"
            )

    logger.info(
        "inversion auto -> sparse: %d visibilities, above the %d at which the "
        "dense n_vis x n_mesh mapping matrix is the dominant cost. The "
        "w-tilde path's cost does not scale with the data at all. Pass "
        "--inversion dense to force the mapping-matrix path.",
        n_vis, SPARSE_AUTO_MIN_VISIBILITIES,
    )
    return "sparse"


def sparse_inversion_diagnosis() -> str | None:
    """Why the sparse inversion is unavailable, or None if it can run."""
    if not hasattr(ag.Interferometer, "apply_sparse_operator"):
        return (
            "this autoarray has no `Interferometer.apply_sparse_operator`; "
            "the sparse inversion needs a version that ships the w-tilde path."
        )
    diagnosis = jax_path_diagnosis()
    if diagnosis is not None and "nufftax" not in diagnosis:
        # nufftax is the NUFFT's problem, not the sparse operator's -- the
        # sparse path only needs jax.numpy
        return f"the sparse inversion needs JAX: {diagnosis}"
    try:  # pragma: no cover - environment dependent
        import jax.numpy  # noqa: F401
    except Exception as e:
        return (
            f"the sparse inversion needs JAX ({type(e).__name__}: {e}). "
            "`--inversion dense` is the fallback and needs nothing."
        )
    return None


def sparse_kernel_key(uv_wavelengths, noise, geometry) -> str:
    """A short hash identifying the (uv coverage, noise, geometry) a W~ kernel
    belongs to.

    Everything that changes the kernel is in it. The noise matters because the
    kernel is inverse-variance weighted -- and recentring the field pools the
    real and imaginary sigmas, so a recentred dataset needs its own kernel even
    though its uv coordinates are unchanged.
    """
    h = hashlib.blake2b(digest_size=8)
    for a in (np.ascontiguousarray(np.asarray(uv_wavelengths, dtype=np.float64)),
              np.ascontiguousarray(np.asarray(noise, dtype=np.complex128))):
        h.update(a.tobytes())
    h.update(
        f"{geometry.shape_native}|{geometry.pixel_scale!r}|"
        f"{geometry.mesh_shape}".encode()
    )
    return h.hexdigest()


def sparse_kernel_cache_path(cache_dir, key: str):
    """Where a W~ kernel for `key` lives, or None if caching is off."""
    if cache_dir is None:
        return None
    from pathlib import Path

    return Path(cache_dir) / f"pyuvimage-{key}{SPARSE_KERNEL_SUFFIX}"


def adjoint_image(transformer, visibilities):
    """`image_from`, on the plain mathematical adjoint's scale.

    `use_adjoint_scaling` is passed only when the transformer accepts it,
    because autoarray removed the argument from its own transformers in
    2026.8.29.1 -- `TransformerDFT.image_from()` there raises TypeError on it,
    and `apply_sparse_operator` no longer passes it either. That removal is
    consistent rather than careless: the flag was always a no-op for
    `TransformerDFT` and the nufftax `TransformerNUFFT`, whose adjoints are
    already the plain mathematical one.

    It is *not* a no-op for a pynufft-backed transformer, whose internal IFFT
    normalisation leaves its raw adjoint a factor `4 N_y N_x` low -- which is
    why ours still accepts it and why `assert_adjoint_scale_consistent`
    measures the result against the DFT rather than trusting any of this.
    """
    import inspect

    try:
        accepts = "use_adjoint_scaling" in inspect.signature(
            transformer.image_from).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        accepts = False
    if accepts:
        return transformer.image_from(
            visibilities=visibilities, use_adjoint_scaling=True)
    return transformer.image_from(visibilities=visibilities)


def assert_adjoint_scale_consistent(
    transformer_cls, n_pix: int = 16, tolerance: float = 1e-2
) -> None:
    """Check a transformer's adjoint is on the scale the w-tilde kernel assumes.

    `apply_sparse_operator` builds `D` from
    `transformer.image_from(..., use_adjoint_scaling=True)` while `W~` is
    accumulated directly from `1/sigma^2`. If the transformer ignores that
    argument, `D` and `F` end up on different scales and the reconstruction
    comes out uniformly too small -- with the right morphology, a plausible
    log-evidence, and no error anywhere. That is exactly what happened on
    Ruby: a 231 sigma residual and a chi^2 that did not move across twelve
    orders of magnitude of `lambda`.

    So check it, on a tiny synthetic problem where `TransformerDFT` is the
    reference. Milliseconds, and it makes the failure loud.

    `tolerance` is deliberately loose. A NUFFT is an approximation and its
    error grows as the grid shrinks -- on a 16x16 probe it is ~1e-5, on an 8x8
    one ~2e-3 -- while the failure this guards against is a factor of
    `4 * N_y * N_x`, i.e. a relative error of order 1. Four orders of magnitude
    separate the two, so there is no need to sit close to the noise floor.
    """
    mask = ag.Mask2D.all_false(shape_native=(n_pix, n_pix), pixel_scales=0.1)
    rng = np.random.default_rng(0)
    pixel_rad = 0.1 * math.pi / (180 * 3600)
    nyquist = 1.0 / (2.0 * pixel_rad)
    uv = rng.uniform(-0.4 * nyquist, 0.4 * nyquist, size=(64, 2))
    vis = ag.Visibilities(
        visibilities=rng.normal(size=64) + 1j * rng.normal(size=64)
    )

    reference = np.asarray(
        adjoint_image(
            ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask), vis
        ).native
    )
    got = np.asarray(
        adjoint_image(
            transformer_cls(uv_wavelengths=uv, real_space_mask=mask), vis
        ).native
    )

    scale = np.abs(reference).max()
    error = np.abs(got - reference).max() / scale
    if error < tolerance:
        return

    ratio = np.median(
        reference[np.abs(got) > 0] / got[np.abs(got) > 0]
    ) if np.any(np.abs(got) > 0) else float("nan")
    raise RuntimeError(
        f"{transformer_cls.__name__}.image_from does not honour "
        f"use_adjoint_scaling: its adjoint sits a factor ~{ratio:.4g} off "
        f"TransformerDFT's (relative error {error:.2e}). The sparse "
        "inversion builds its data vector through this call, so the "
        "reconstruction would come out uniformly mis-scaled with no other "
        "symptom. Fix the transformer, or use --inversion dense."
    )


def with_sparse_operator(
    dataset,
    uv_wavelengths,
    noise,
    geometry,
    cache_dir=None,
    chunk_k: int | None = None,
    batch_size: int = SPARSE_BATCH_SIZE,
):
    """Attach the w-tilde operator, building or reusing its kernel.

    The kernel depends only on the uv coverage, the noise and the geometry --
    never on the data values or the source prior -- so it is the one expensive
    invariant of a run and is cached on disk, the way CASA caches its
    convolution functions in `cfcache`.
    """
    reason = sparse_inversion_diagnosis()
    if reason is not None:
        raise RuntimeError(reason)
    assert_adjoint_scale_consistent(type(dataset.transformer))
    warn_on_reim_asymmetry(noise)
    warn_on_single_precision()

    key = sparse_kernel_key(uv_wavelengths, noise, geometry)
    path = sparse_kernel_cache_path(cache_dir, key)
    kernel = None
    if path is not None and path.exists():
        try:
            kernel = np.load(path)
            logger.info("reusing the cached w-tilde kernel %s", path.name)
        except Exception as e:  # a corrupt cache must never stop a fit
            logger.warning("could not read %s (%s); rebuilding", path, e)
            kernel = None

    if kernel is None:
        n_image_pixels = int(np.prod(geometry.shape_native))
        if chunk_k is None:
            chunk_k = sparse_chunk_k_for_budget(n_image_pixels)
            if chunk_k < SPARSE_CHUNK_K:
                logger.info(
                    "streaming %d visibilities at a time rather than %d to "
                    "keep the kernel build under %.1f GB",
                    chunk_k, SPARSE_CHUNK_K,
                    sparse_kernel_build_gb(n_image_pixels, chunk_k),
                )
        t = time.time()
        logger.info(
            "building the w-tilde kernel (one pass over %d visibilities; this "
            "is the only step whose cost scales with the data)",
            len(np.asarray(uv_wavelengths)),
        )
        kernel = np.asarray(dataset.psf_precision_operator_from(chunk_k=chunk_k))
        logger.info(
            "  kernel %s, %.2f MB, %.1f s", kernel.shape, kernel.nbytes / 1e6,
            time.time() - t,
        )
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, kernel)
                logger.info("  cached as %s; a re-fit of this field reuses it",
                            path.name)
            except Exception as e:
                logger.warning("could not cache the kernel at %s (%s)", path, e)

    sparse = dataset.apply_sparse_operator(
        nufft_precision_operator=kernel, batch_size=batch_size
    )
    return repair_sparse_dirty_image(sparse, dataset)


#: Median re/im noise asymmetry above which the sparse path is worth a word.
#: Ruby reads 9% unrecentred, which is estimator scatter rather than anything
#: physical, but it is still a real inconsistency in the linear system.
SPARSE_REIM_ASYMMETRY_WARN = 0.02


def _model_response(fit_fn, weak_coefficient, strong_coefficient):
    """Relative change in the reconstruction between two prior strengths.

    None when either fit cannot be made or the models are not comparable --
    an unknown response must never be read as "no response", or a solver
    failure would masquerade as the pathology and switch positivity off.
    """
    models = []
    for coefficient in (weak_coefficient, strong_coefficient):
        try:
            fit = fit_fn(coefficient)
            models.append(np.asarray(fit.inversion.reconstruction, dtype=float))
            del fit
        except Exception:
            return None
    weak, strong = models
    if weak.shape != strong.shape:
        return None
    scale = float(np.linalg.norm(weak))
    if scale <= 0:
        return None
    return float(np.linalg.norm(strong - weak)) / scale


def warn_on_single_precision() -> bool | None:
    """Say so if JAX is in 32-bit mode, which silently halves the solve.

    `pyuvimage/__init__.py` sets `JAX_ENABLE_X64` before JAX is imported, so
    this normally passes. It can still fail: a caller who imported JAX first
    gets whatever they configured, and by then the environment variable is
    too late.

    The consequence is not subtle once you know to look for it -- the sparse
    curvature matrix, its regularised copy and the solve all drop to float32,
    roughly seven decimal digits for a matrix whose condition number is not
    small -- but nothing announces it except three UserWarnings from deep
    inside autoarray about a requested float64 being "truncated to float32".
    """
    from ._jax_guard import jax_double_precision_active

    active = jax_double_precision_active()
    if active is False:
        logger.warning(
            "JAX is in 32-bit mode, so the sparse curvature matrix and its "
            "solve will be built in float32 -- about seven decimal digits, "
            "against the dense path's sixteen. Set JAX_ENABLE_X64=True "
            "before importing JAX (importing pyuvimage first does this for "
            "you), or expect the two paths to agree only to ~1e-7 at best."
        )
    return active


def warn_on_reim_asymmetry(noise) -> float:
    """Say so when sigma_re != sigma_im, which the w-tilde reduction assumes.

    The asymmetry is not symmetric in its effects, so to speak: `W~` is
    accumulated from `noise_map_real` **alone**

        nufft_precision_operator_from(noise_map_real=self.noise_map.array.real, ...)

    while the data vector's dirty image weights the two parts separately

        data.real * noise.real**-2 + 1j * data.imag * noise.imag**-2

    so `F` and `D` are built on different weightings whenever the sigmas
    differ, and the solution of `(F + H)s = D` is correspondingly off.
    autoarray confirms this degrades agreement with the dense path on the
    sparse route generally, not merely for particular blocks.

    `--image-centre` pools the two in quadrature (see
    `uvdata.shift_image_centre`), which is both the fix and, by the argument
    in `uvdata._report_reim_asymmetry`, the better noise estimate anyway --
    twice the sample size. So a recentred fit satisfies the assumption
    exactly, and most real fits are recentred. This warns about the rest.
    """
    from .uvdata import reim_asymmetry

    asymmetry = reim_asymmetry(noise)
    if asymmetry > SPARSE_REIM_ASYMMETRY_WARN:
        logger.warning(
            "sigma_re and sigma_im differ by %.1f%% (median), and the "
            "w-tilde reduction assumes they are equal: it builds the "
            "curvature matrix from sigma_re alone while the data vector "
            "weights the two separately, so the sparse and dense paths will "
            "not agree to better than about that. Recentring the field "
            "(--image-centre) pools them in quadrature, which removes the "
            "discrepancy and is the better noise estimate in any case; "
            "--inversion dense is unaffected either way.",
            100.0 * asymmetry,
        )
    return asymmetry


def scaled_dirty_image(dataset) -> np.ndarray:
    """The inverse-variance-weighted adjoint, on the w-tilde kernel's scale.

    Exactly what `apply_sparse_operator` computes internally -- reproduced here
    so we can check its answer rather than trust it.
    """
    from autoarray.structures.visibilities import Visibilities

    data = np.asarray(dataset.data)
    noise = np.asarray(dataset.noise_map)
    weighted = (
        data.real * noise.real ** -2.0 + 1j * data.imag * noise.imag ** -2.0
    )
    return np.asarray(
        adjoint_image(
            dataset.transformer, Visibilities(visibilities=weighted)
        ).array
    )


def repair_sparse_dirty_image(sparse_dataset, dataset, tolerance: float = 1e-6):
    """Put the operator's dirty image on the kernel's scale if it is not.

    `D = L^T dirty_image` while `W~` is accumulated straight from `1/sigma^2`,
    so the two are only comparable if the adjoint was scaled. autoarray builds
    that dirty image itself, via

        self.transformer.image_from(..., use_adjoint_scaling=True)

    and whether the flag is actually passed has varied between versions -- it
    is a no-op for `TransformerDFT` and the nufftax `TransformerNUFFT`, so a
    release that drops it looks harmless and breaks only the pynufft path.
    autoarray 2026.8.17.1 passes it; 2026.8.23.1, measured, does not.

    Trusting the caller here is what produced a 231 sigma fit whose chi^2 was
    identical to eleven significant figures across twelve orders of magnitude
    of `lambda`. So verify, repair, and say so.
    """
    import dataclasses

    operator = getattr(sparse_dataset, "sparse_operator", None)
    if operator is None or getattr(operator, "dirty_image", None) is None:
        return sparse_dataset

    want = scaled_dirty_image(dataset)
    got = np.asarray(operator.dirty_image)
    if got.shape != want.shape:
        logger.warning(
            "the sparse operator's dirty image has shape %s, expected %s -- "
            "leaving it alone, but the reconstruction's scale is unverified",
            got.shape, want.shape,
        )
        return sparse_dataset

    scale = max(np.abs(want).max(), 1e-300)
    if np.abs(got - want).max() <= tolerance * scale:
        return sparse_dataset

    ratio = np.abs(want).max() / max(np.abs(got).max(), 1e-300)
    logger.warning(
        "this autoarray builds the sparse operator's dirty image without "
        "adjoint scaling, leaving it a factor %.6g off the w-tilde kernel's "
        "scale. Correcting it. (Uncorrected, the reconstruction keeps its "
        "morphology at 1/%.0f of the right amplitude and chi^2 stops "
        "depending on the regularisation coefficient entirely.)",
        ratio, ratio,
    )
    # `InterferometerSparseOperator` is a frozen dataclass, so it is replaced
    # rather than mutated; `Interferometer` is not a dataclass and holds
    # `sparse_operator` as a plain attribute, so it is assigned.
    sparse_dataset.sparse_operator = dataclasses.replace(
        operator, dirty_image=want
    )
    return sparse_dataset


def make_dataset(
    uv_wavelengths: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    geometry: ImageGeometry,
    transformer: str = "auto",
    mask_shape: str = "square",
) -> ag.Interferometer:
    """Assemble the PyAutoGalaxy Interferometer dataset.

    ``mask_shape="square"`` (default) makes the reconstruction region the full
    square field. A circular mask leaves the mesh's corner pixels covering no
    image pixels at all, so no data constrains them and the prior alone sets
    their value -- which showed up as bright spurious blobs in the corners of
    the restored image, carrying ~29% of the source flux on one test mock.
    """
    if mask_shape == "square":
        mask = ag.Mask2D.all_false(
            shape_native=geometry.shape_native,
            pixel_scales=geometry.pixel_scale,
        )
    elif mask_shape == "circular":
        mask = ag.Mask2D.circular(
            shape_native=geometry.shape_native,
            pixel_scales=geometry.pixel_scale,
            radius=geometry.mask_radius * 1.001,  # keep boundary pixels
        )
    else:
        raise ValueError("mask_shape must be 'square' or 'circular'")
    cls = resolve_transformer(
        n_vis=len(data),
        transformer=transformer,
        n_image_pixels=int(np.prod(geometry.shape_native)),
    )
    return ag.Interferometer(
        data=ag.Visibilities(np.asarray(data, dtype=complex)),
        noise_map=ag.VisibilitiesNoiseMap(np.asarray(noise, dtype=complex)),
        uv_wavelengths=np.asarray(uv_wavelengths, dtype=float),
        real_space_mask=mask,
        transformer_class=cls,
        raise_error_dft_visibilities_limit=False,
    )


def make_regularization(
    kind: str,
    coefficient: float,
    scale: float | None = None,
    nu: float = DEFAULT_NU,
    envelope: dict | None = None,
):
    """Build the source prior (regularisation scheme).

    ``scale`` is a correlation length in **arcsec** -- the kernel schemes build
    their covariance from the mesh coordinates themselves, which are in the
    image plane's angular units.
    """
    kind = kind.lower()
    if kind == "constant":
        return ag.reg.Constant(coefficient=coefficient)
    if kind == "matern":
        from .envelope import CachedMaternKernel

        return CachedMaternKernel(
            coefficient=coefficient, scale=scale or 1.0, nu=nu
        )
    if kind == "gaussian":
        # Matern GP modulated by a Gaussian envelope on the prior width:
        # supplies the spatial information a stationary prior lacks, which
        # is what keeps dirty-beam sidelobes out of the model when the
        # visibilities are sparse.
        from .envelope import GaussianEnvelopeMatern

        env = envelope or {}
        return GaussianEnvelopeMatern(
            coefficient=coefficient,
            scale=scale or 1.0,
            nu=nu,
            envelope_fwhm=env.get("fwhm", 1.0),
            envelope_floor=env.get("floor", 1e-2),
            centre=env.get("centre", (0.0, 0.0)),
        )
    if kind == "exponential":
        return ag.reg.ExponentialKernel(
            coefficient=coefficient, scale=scale or 1.0
        )
    if kind == "gibbs":
        # Non-stationary: the correlation length shortens where the source is
        # bright, so an unresolved component is not asked to be beam-smooth.
        from .envelope import GibbsMatern

        env = envelope or {}
        return GibbsMatern(
            coefficient=coefficient,
            scale=scale or 1.0,
            brightness=env.get("brightness"),
            ell_floor=env.get("ell_floor", GIBBS_ELL_FLOOR),
            power=env.get("power", ADAPT_POWER),
        )
    if kind == "adaptive":
        # Brightness-adaptive: smooth less where a first-pass model says the
        # source is bright, more where it is faint.
        from .envelope import AdaptiveMatern

        env = envelope or {}
        return AdaptiveMatern(
            coefficient=coefficient,
            scale=scale or 1.0,
            nu=nu,
            brightness=env.get("brightness"),
            floor=env.get("floor", ADAPT_FLOOR),
            power=env.get("power", ADAPT_POWER),
        )
    raise ValueError(
        f"unknown regularization {kind!r}; options: {REGULARIZATIONS}"
    )


def galaxy_for(
    mesh_shape: tuple[int, int],
    reg_kind: str,
    coefficient: float,
    reg_scale: float | None = None,
    nu: float = DEFAULT_NU,
    envelope: dict | None = None,
) -> ag.Galaxy:
    pix = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=mesh_shape),
        regularization=make_regularization(
            reg_kind, coefficient, reg_scale, nu, envelope
        ),
    )
    # The redshift is inert without lensing; any value works.
    return ag.Galaxy(redshift=1.0, pixelization=pix)


def fit_at(
    dataset: ag.Interferometer,
    mesh_shape: tuple[int, int],
    reg_kind: str,
    coefficient: float,
    positive_only: bool = True,
    reg_scale: float | None = None,
    nu: float = DEFAULT_NU,
    envelope: dict | None = None,
    adapt_image=None,
) -> ag.FitInterferometer:
    galaxy = galaxy_for(mesh_shape, reg_kind, coefficient, reg_scale, nu, envelope)
    settings = ag.Settings(use_positive_only_solver=positive_only)
    kwargs = {}
    if adapt_image is not None:
        kwargs["adapt_images"] = ag.AdaptImages(
            galaxy_image_dict={galaxy: adapt_image}
        )
    return ag.FitInterferometer(
        dataset=dataset, galaxies=[galaxy], settings=settings, **kwargs
    )


def _safe_evidence(fit: ag.FitInterferometer) -> float:
    try:
        return float(fit.figure_of_merit)
    except Exception as e:  # LinAlgError etc.
        logger.info("    (evidence evaluation failed: %s: %s)", type(e).__name__, e)
        return -np.inf


@dataclass
class PriorScan:
    """Record of the source-prior hyperparameter optimisation."""

    criterion: str = ""
    reg_kind: str = ""
    n_data: int = 0
    free_parameters: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    #: chi^2 of the weakest prior tried, under the solver the delivered fit
    #: uses. With positivity on this is a floor the fit cannot go below, so
    #: the discrepancy target has to respect it (see `effective_chi2_target`).
    chi2_floor: float = float("nan")

    def record(
        self,
        params: dict,
        log_evidence: float,
        chi_squared: float,
        structure_ratio: float = float("nan"),
        positive: bool = False,
    ) -> None:
        trial = {
            **{k: float(v) for k, v in params.items()},
            "log_evidence": float(log_evidence),
            "chi_squared": float(chi_squared),
            "chi_squared_per_datum": (
                float(chi_squared) / self.n_data if self.n_data else float("nan")
            ),
        }
        if np.isfinite(structure_ratio):
            trial["structure_ratio"] = float(structure_ratio)
        if positive:
            trial["positive"] = True
        self.trials.append(trial)

    def as_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "regularization": self.reg_kind,
            "free_parameters": list(self.free_parameters),
            "n_data": int(self.n_data),
            "best": {k: float(v) for k, v in self.best.items()},
            "n_evaluations": len(self.trials),
            "trials": self.trials,
            **(
                {"chi_squared_floor": float(self.chi2_floor)}
                if np.isfinite(self.chi2_floor) else {}
            ),
        }


def chi2_floor_tolerance(n_data: int | None = None) -> float:
    """How far above the chi^2 floor still counts as "fits essentially as well".

    ``CHI2_FLOOR_SIGMAS`` standard errors of chi^2/N, which is sqrt(2/N) --
    so the allowance shrinks as the dataset grows, as it must. Falls back to
    ``CHI2_FLOOR_TOLERANCE`` when the sample count is unknown, and is capped
    at ``CHI2_FLOOR_TOLERANCE_MAX`` so a tiny dataset is not handed unlimited
    slack.
    """
    if not n_data or n_data <= 0:
        return CHI2_FLOOR_TOLERANCE
    return min(
        CHI2_FLOOR_SIGMAS * math.sqrt(2.0 / float(n_data)),
        CHI2_FLOOR_TOLERANCE_MAX,
    )


def chi2_rebisect_tolerance(n_data: int | None = None) -> float:
    """How far off target the constrained fit may sit before re-bisecting.

    Unlike `chi2_floor_tolerance`, this one is only ever allowed to get
    *tighter*: a loose gate here means shipping a model that was never
    re-bisected, so `CHI2_REBISECT_TOLERANCE` stays the ceiling and sqrt(2/N)
    pulls it down on large datasets (0.5% on Ruby against the old 3%).
    """
    if not n_data or n_data <= 0:
        return CHI2_REBISECT_TOLERANCE
    return min(
        CHI2_REBISECT_TOLERANCE,
        max(chi2_floor_tolerance(n_data), CHI2_REBISECT_FLOOR),
    )


def effective_chi2_target(
    target: float, floor: float, n_data: int | None = None
) -> float:
    """Raise an unreachable chi^2 target to just above the achievable floor.

    ``target`` and ``floor`` are both absolute chi^2 (not per datum). If the
    best chi^2 this model and solver can reach already exceeds the target,
    aiming at the target is worse than useless -- see CHI2_FLOOR_TOLERANCE.
    Aim a hair above the floor instead, which selects the strongest prior
    whose fit is indistinguishable from the best available.

    Pass ``n_data``: without it the allowance is a fixed 5%, which on a large
    dataset is tens of sigma of over-smoothing rather than a knee.
    """
    if np.isfinite(floor) and np.isfinite(target) and floor > target:
        return float(floor) * (1.0 + chi2_floor_tolerance(n_data))
    return float(target)


def structure_ratio(fit: ag.FitInterferometer, imager, n_data: int) -> float:
    """How much the residual *map* looks like noise, rather than like sky.

    ``chi^2`` constrains the residual's total power and nothing else. Two fits
    with the same chi^2 can leave a featureless residual or the whole source,
    and only the image plane can tell them apart: incoherent residuals average
    down as ``1/sqrt(N)``, coherent ones add in phase and land ``sqrt(N)``
    higher.

    So compare the residual map's measured rms against ``sqrt(chi^2/N)``, the
    rms it would have if the residual visibilities were white:

    - **~1** the leftover is noise
    - **>1** coherent structure the fit discarded
    - **<1** the residual is quieter than the noise it should contain, i.e. the
      model has absorbed noise

    On PJ0116 this separates a good fit (0.94) from an overfit one (0.73) while
    chi^2/N moves only 1.007 -> 1.036.
    """
    chi2 = _chi_squared(fit)
    if not np.isfinite(chi2) or chi2 <= 0 or not n_data:
        return float("nan")
    try:
        resid = np.asarray(fit.dataset.data) - np.asarray(fit.model_data)
        rms = imager.rms
        if not np.isfinite(rms) or rms <= 0:
            return float("nan")
        resid_map = np.asarray(imager.dirty_image(resid), dtype=float) / rms
        # the transformer returns zeros outside the real-space mask, which
        # would drag the rms down; measure only where the image is defined
        inside = imager.inside
        if getattr(inside, "shape", None) == resid_map.shape:
            resid_map = resid_map[inside]
        measured = float(np.nanstd(resid_map))
    except Exception as e:  # pragma: no cover - diagnostic only
        logger.debug("structure ratio failed: %s", e)
        return float("nan")
    expected = np.sqrt(chi2 / n_data)
    if not (np.isfinite(expected) and expected > 0 and np.isfinite(measured)):
        return float("nan")
    return measured / expected


def _chi_squared(fit: ag.FitInterferometer) -> float:
    try:
        return float(fit.inversion.fast_chi_squared)
    except Exception:
        return float("nan")


def optimise_prior(
    dataset: ag.Interferometer,
    geometry: ImageGeometry,
    reg_kind: str = "matern",
    criterion: str = "discrepancy",
    nu: float = DEFAULT_NU,
    fixed_scale: float | None = None,
    envelope: dict | None = None,
    optimise_envelope: bool = False,
    adapt_image=None,
    chi2_target: float = 1.0,
    max_evaluations: int = 60,
    positive_only: bool = False,
) -> tuple[dict, PriorScan]:
    """Optimise the source-prior hyperparameters.

    Follows the PyAutoLabs convention: the pixelized source's prior is a
    regularisation scheme whose hyperparameters -- the ``coefficient``, plus a
    correlation ``scale`` in arcsec for the kernel (Gaussian-process) schemes
    -- are free parameters chosen by maximising the Bayesian log evidence.
    The evidence balances goodness of fit against model complexity, so a
    single global smoothing strength is no longer imposed by hand.

    The search is a coarse log-spaced grid followed by Nelder-Mead
    refinement, always with the fast unconstrained solver (the evidence is
    defined for the linear solution); positivity is applied to the final fit.

    ``criterion="discrepancy"`` instead drives chi^2 to ``chi2_target * N``,
    which is more robust when the noise map is trustworthy but the evidence
    is poorly behaved (e.g. far fewer visibilities than model pixels).

    ``criterion="structure"`` drives the residual *map's* structure ratio to
    1 instead (see `structure_ratio`). chi^2 constrains only the residual's
    total power, and on real data it can be nearly flat in the coefficient --
    on PJ0116 chi^2/N moved by 0.0008 across two decades of smoothing while
    the residual went from white to visibly over-smoothed -- so the image
    plane is the more discriminating test of when to stop.

    ``positive_only`` says whether the delivered fit imposes positivity. The
    structure search then runs on that solver throughout, since positivity
    changes the residual map; the discrepancy search stays unconstrained (it
    is much faster and `fit_dataset` re-bisects afterwards) but its
    reachability probe uses it, because the constrained chi^2 floor is what
    the delivered fit actually has to live with.
    """
    mesh_shape = geometry.mesh_shape
    n_data = 2 * len(np.asarray(dataset.data))
    # a kernel prior whose correlation length is pinned has only one free
    # hyperparameter, exactly like the non-kernel schemes
    is_kernel_scheme = reg_kind in KERNEL_REGULARIZATIONS
    # The second free hyperparameter is either the kernel correlation length
    # or, when the correlation length is pinned to the beam and the user asks
    # for it, the Gaussian envelope's width.
    kernel = is_kernel_scheme and fixed_scale is None
    second = "scale" if kernel else ("envelope_fwhm" if optimise_envelope else None)
    two_d = second is not None
    free = ["coefficient"] + ([second] if two_d else [])
    scan = PriorScan(
        criterion=criterion, reg_kind=reg_kind, n_data=n_data,
        free_parameters=free,
    )

    # scale bounds: below the mesh pixel scale the kernel is meaningless,
    # above ~half the field it cannot represent structure.
    log_scale_bounds = (
        np.log10(geometry.mesh_pixel_scale),
        np.log10(max(geometry.fov_arcsec / 2.0, geometry.mesh_pixel_scale * 4)),
    )
    if second == "envelope_fwhm":
        # an envelope narrower than the beam is meaningless; wider than the
        # field cannot constrain anything
        beam_like = fixed_scale or geometry.mesh_pixel_scale
        log_scale_bounds = (
            np.log10(beam_like), np.log10(geometry.fov_arcsec / 2.0)
        )

    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")

    # The structure criterion is measured in the image plane, so it needs a
    # dirty imager; build it once and reuse it across every evaluation.
    imager = DirtyImager(dataset) if criterion == "structure" else None
    # positivity changes the residual map, so the structure search has to run
    # on the solver the delivered fit uses; the other criteria stay on the
    # fast unconstrained solve (the evidence is defined for it).
    search_positive = bool(positive_only) and criterion == "structure"

    def evaluate(
        log_params: np.ndarray, positive: bool | None = None
    ) -> tuple[float, float, float]:
        positive = search_positive if positive is None else bool(positive)
        coefficient = 10.0 ** float(log_params[0])
        scale = 10.0 ** float(log_params[1]) if kernel else fixed_scale
        env = envelope
        if second == "envelope_fwhm":
            env = {**(envelope or {}), "fwhm": 10.0 ** float(log_params[1])}
        try:
            fit = fit_at(
                dataset, mesh_shape, reg_kind, coefficient,
                positive_only=positive, reg_scale=scale, nu=nu,
                envelope=env, adapt_image=adapt_image,
            )
            ev = _safe_evidence(fit)
            chi2 = _chi_squared(fit)
            ratio = (
                structure_ratio(fit, imager, n_data)
                if imager is not None else float("nan")
            )
        except Exception as e:
            logger.debug("prior evaluation failed: %s", e)
            return -np.inf, float("nan"), float("nan")
        params = {"coefficient": coefficient}
        if scale is not None:
            params["scale"] = scale
        if second == "envelope_fwhm":
            params["envelope_fwhm"] = env["fwhm"]
        scan.record(params, ev, chi2, ratio, positive=positive)
        logger.info(
            "  coefficient=%.4g%s  log_evidence=%.6g  chi2/N=%.4g%s",
            coefficient,
            f"  scale={scale:.4g}\"" if kernel else "",
            ev, chi2 / n_data if n_data else np.nan,
            f"  structure={ratio:.3g}" if np.isfinite(ratio) else "",
        )
        return ev, chi2, ratio

    def score(log_params: np.ndarray) -> float:
        """Higher is better."""
        for i, (lo, hi) in enumerate(bounds):
            if not (lo <= log_params[i] <= hi):
                return -np.inf
        ev, chi2, ratio = evaluate(log_params)
        if criterion == "evidence":
            return ev
        # break ties (kernel schemes) by evidence
        tie = ev / (1.0 + abs(ev)) if np.isfinite(ev) else 0.0
        if criterion == "structure":
            if not np.isfinite(ratio) or ratio <= 0:
                return -np.inf
            return -np.log10(ratio / STRUCTURE_TARGET) ** 2 + 1e-3 * tie
        if not np.isfinite(chi2) or chi2 <= 0:
            return -np.inf
        # drive chi^2 to the target
        miss = np.log10(chi2 / (chi2_target * n_data)) ** 2
        return -miss + 1e-3 * tie

    bounds = [LOG_COEFFICIENT_BOUNDS] + ([log_scale_bounds] if two_d else [])

    if criterion in ("discrepancy", "structure"):
        # Both criteria drive a single measure up to a target as the prior
        # strengthens -- chi^2 to chi2_target * N, or the residual structure
        # ratio to 1 -- so in both cases the coefficient can be bisected.
        # For kernel priors the correlation scale is the remaining freedom:
        # among the priors that all hit the target, take the one the evidence
        # prefers.
        structure = criterion == "structure"
        target = STRUCTURE_TARGET if structure else chi2_target * n_data
        unit = "structure ratio" if structure else "chi^2/N"
        per = 1.0 if structure else n_data

        def measure(log_params: np.ndarray) -> tuple[float, float]:
            """(the quantity being driven to `target`, the log evidence)."""
            ev, chi2, ratio = evaluate(log_params)
            return (ratio if structure else chi2), ev

        # If even the weakest prior overshoots the target, bisection has no
        # solution: every trial reads "still too high", so the search drives
        # the coefficient to its floor -- silently switching regularisation
        # off and returning a noisy, overfitted model. Detect that first,
        # with the solver the delivered fit uses: positivity raises chi^2, so
        # probing unconstrained hides a constrained floor above the target,
        # which is exactly how PJ0116 ended up with no effective prior.
        # the probe vector must match the number of free hyperparameters
        probe = [LOG_COEFFICIENT_BOUNDS[0]]
        if kernel:
            probe.append(float(np.mean(log_scale_bounds)))
        _, chi2_weakest, ratio_weakest = evaluate(
            np.array(probe), positive=positive_only
        )
        if not structure:
            scan.chi2_floor = chi2_weakest
        floor = ratio_weakest if structure else chi2_weakest
        hopeless = target * (1.0 if structure else CHI2_UNREACHABLE_FACTOR)

        if np.isfinite(floor) and floor > hopeless:
            logger.warning(
                "%s = %.3g with essentially no regularisation, above the "
                "target of %.3g: the model cannot reproduce this data however "
                "little it is smoothed. Common causes: the source has real "
                "structure finer than the model pixel scale, the noise map "
                "underestimates the noise, or the field of view is too small. "
                "Falling back to maximum-evidence selection so the prior is "
                "not switched off.",
                unit, floor / per, target / per,
            )
            best, ev_scan = optimise_prior(
                dataset, geometry, reg_kind=reg_kind, criterion="evidence",
                nu=nu, fixed_scale=fixed_scale, envelope=envelope,
                adapt_image=adapt_image, max_evaluations=max_evaluations,
            )
            scan.trials.extend(ev_scan.trials)
            scan.criterion = f"{criterion}->evidence (unreachable target)"
            scan.best = best
            return best, scan

        if np.isfinite(floor) and floor > target:
            # A near miss: the target is out of reach but only just, so aim
            # at the knee instead of chasing something impossible.
            target = effective_chi2_target(target, floor, n_data)
            logger.info(
                "chi^2/N cannot go below %.4g with this model%s, short of the "
                "%.4g asked for; aiming for %.4g instead -- %.3g sigma above "
                "the floor, where one sigma of chi^2/N is sqrt(2/N) = %.4g.",
                floor / per, " under positivity" if positive_only else "",
                chi2_target, target / per, CHI2_FLOOR_SIGMAS,
                np.sqrt(2.0 / n_data) if n_data else float("nan"),
            )

        # A bisection that runs to the top of the bracket has not chosen a
        # coefficient, it has run out of room. Recorded here so the structure
        # criterion can hand back to chi^2 rather than deliver the bound.
        saturated: list[bool] = []

        def coefficient_for_target(log_scale: float | None) -> tuple[float, float]:
            lo, hi = LOG_COEFFICIENT_BOUNDS
            p = (lambda c: np.array([c, log_scale])) if two_d else (
                lambda c: np.array([c])
            )
            # The coefficient's natural scale depends on the data's units and
            # signal-to-noise, so the shipped LogUniform(1e-6, 1e6) range is
            # not always wide enough: extend the bracket until the measure
            # brackets the target.
            value_hi, ev_hi = measure(p(hi))
            while (
                np.isfinite(value_hi) and value_hi < target
                and hi < MAX_LOG_COEFFICIENT
            ):
                hi = min(hi + 3.0, MAX_LOG_COEFFICIENT)
                value_hi, ev_hi = measure(p(hi))
            if np.isfinite(value_hi) and value_hi < target:
                saturated.append(True)
                logger.warning(
                    "even the strongest prior tried (coefficient 1e%g) leaves "
                    "%s at %.3g, below the target of %.3g: the model has far "
                    "more freedom than the data constrain, so its faint "
                    "structure is set by the prior.",
                    hi, unit, value_hi / per, target / per,
                )
                # rank scales by evidence, like every other return here
                return hi, ev_hi
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                value, _ = measure(p(mid))
                if not np.isfinite(value) or value < target:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.02:
                    break
            best_c = 0.5 * (lo + hi)
            _, ev = measure(p(best_c))
            return best_c, ev

        if not two_d:
            c, _ = coefficient_for_target(None)
            best_log = np.array([c])
        else:
            scales = np.linspace(log_scale_bounds[0], log_scale_bounds[1], 4)
            results = []
            for ls in scales:
                c, ev = coefficient_for_target(ls)
                results.append((ev if np.isfinite(ev) else -np.inf, c, ls))
                logger.info(
                    "  scale=%.4g\" -> coefficient=%.4g (chi2 target), "
                    "log_evidence=%.6g", 10**ls, 10**c, ev,
                )
            best_ev, best_c, best_ls = max(results, key=lambda t: t[0])
            # one refinement pass around the best scale
            step = (log_scale_bounds[1] - log_scale_bounds[0]) / 6.0
            for ls in (best_ls - step, best_ls + step):
                if not (log_scale_bounds[0] <= ls <= log_scale_bounds[1]):
                    continue
                c, ev = coefficient_for_target(ls)
                if np.isfinite(ev) and ev > best_ev:
                    best_ev, best_c, best_ls = ev, c, ls
            best_log = np.array([best_c, best_ls])

        if structure and saturated:
            # The structure ratio never reached 1 however hard the prior was
            # pushed, so the model's faint structure is prior-set and the
            # ratio is not calibrated here -- the small-mock regime described
            # in design-notes.md, and the reason `--criterion auto` exists.
            # Delivering the bracket's ceiling would be the worst of both, so
            # hand back to chi^2, which does have a reachable target.
            logger.warning(
                "the structure criterion could not reach a ratio of %.3g at "
                "any coefficient in range, so it has nothing to select on: "
                "this fit is too weakly constrained for the residual map to "
                "be white at chi^2 = N. Falling back to --criterion "
                "discrepancy.", target,
            )
            best, chi_scan = optimise_prior(
                dataset, geometry, reg_kind=reg_kind, criterion="discrepancy",
                positive_only=positive_only, nu=nu, fixed_scale=fixed_scale,
                envelope=envelope, adapt_image=adapt_image,
                max_evaluations=max_evaluations, chi2_target=chi2_target,
            )
            scan.trials.extend(chi_scan.trials)
            scan.chi2_floor = chi_scan.chi2_floor
            scan.criterion = "structure->discrepancy (ratio unreachable)"
            scan.best = best
            return best, scan
    else:
        # ---- coarse grid then local refinement, maximising the evidence
        coarse_coeff = np.linspace(
            LOG_COEFFICIENT_BOUNDS[0], LOG_COEFFICIENT_BOUNDS[1], 7
        )
        if two_d:
            coarse_scale = np.linspace(log_scale_bounds[0], log_scale_bounds[1], 3)
            grid = [np.array([c, sc]) for sc in coarse_scale for c in coarse_coeff]
        else:
            grid = [np.array([c]) for c in coarse_coeff]
        scores = [score(p) for p in grid]
        if not np.any(np.isfinite(scores)):
            raise RuntimeError(
                "the inversion failed for every source prior tried; check the "
                "noise map, the uv coordinates and the field of view"
            )
        x0 = grid[int(np.nanargmax(scores))]

        from scipy.optimize import minimize

        remaining = max(10, max_evaluations - len(scan.trials))
        res = minimize(
            lambda p: -score(p), x0, method="Nelder-Mead",
            options={
                "maxfev": remaining, "xatol": 0.01, "fatol": 1e-3,
                "initial_simplex": (
                    np.vstack([x0, x0 + np.eye(len(x0))[0] * 0.5]
                              + ([x0 + np.eye(len(x0))[1] * 0.3] if two_d else []))
                ),
            },
        )
        best_log = res.x if np.isfinite(res.fun) else x0

    best = {"coefficient": 10.0 ** float(best_log[0])}
    if kernel:
        best["scale"] = 10.0 ** float(best_log[1])
    else:
        if fixed_scale is not None:
            best["scale"] = float(fixed_scale)
        if second == "envelope_fwhm":
            best["envelope_fwhm"] = 10.0 ** float(best_log[1])
    if is_kernel_scheme:
        best["nu"] = float(nu)
    scan.best = best
    logger.info(
        "source prior optimised (%s, %d evaluations): %s",
        criterion, len(scan.trials),
        ", ".join(f"{k}={v:.4g}" for k, v in best.items()),
    )
    return best, scan


def _block_contrast(image: np.ndarray, oversample: int) -> float:
    """Peak-to-peak spread *within* an oversample x oversample block, relative
    to the block mean -- i.e. how strong the checkerboard is."""
    if oversample < 2:
        return 0.0
    a = np.asarray(image, dtype=float)
    n = (a.shape[0] // oversample) * oversample
    blocks = a[:n, :n].reshape(
        n // oversample, oversample, n // oversample, oversample)
    hi = blocks.max(axis=(1, 3))
    lo = blocks.min(axis=(1, 3))
    mean = blocks.mean(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmedian(np.where(mean > 0, (hi - lo) / mean, np.nan)))


def _deblock(image: np.ndarray, oversample: int) -> np.ndarray:
    """Remove the mesh/image checkerboard, keeping the upper envelope.

    A grey dilation over one block replaces every pixel with the largest
    uncertainty in its neighbourhood (killing the low interpolated pixels),
    then a box mean of the same size smooths the result back to a continuous
    field.  Deliberately conservative: the map may slightly over-state the
    error on interpolated pixels, which is the safe direction for a
    significance map.
    """
    if oversample < 2:
        return image
    from scipy.ndimage import maximum_filter, uniform_filter

    a = np.asarray(image, dtype=float)
    finite = np.isfinite(a)
    filled = np.where(finite, a, 0.0)
    envelope = maximum_filter(filled, size=oversample, mode="nearest")
    smoothed = uniform_filter(envelope, size=oversample, mode="nearest")
    return np.where(finite, smoothed, np.nan)


@dataclass
class SingleFit:
    """Everything downstream products need from one fit."""

    fit: ag.FitInterferometer
    geometry: ImageGeometry
    prior: dict
    scan: PriorScan | None = None
    #: whether the *delivered* fit imposed positivity. Not the same as what
    #: the caller asked for: the non-negative solver is disabled mid-fit when
    #: it is caught ignoring the prior, and `fit_parameters.json` was
    #: reporting the request rather than what actually ran.
    positive_only: bool = True

    @property
    def coefficient(self) -> float:
        return float(self.prior["coefficient"])

    @property
    def model_mesh_image(self) -> np.ndarray:
        """Reconstruction on the source mesh, 2D [Jy / mesh pixel].

        The inversion solves for surface brightness per *image* pixel (each
        mesh cell's value is replicated over oversample^2 image pixels by the
        mapper), so the per-mesh-pixel flux carries that factor.
        """
        recon = np.asarray(self.fit.inversion.reconstruction)
        k2 = (
            self.geometry.shape_native[0] // self.geometry.mesh_shape[0]
        ) ** 2
        return recon.reshape(self.geometry.mesh_shape) * k2

    @property
    def model_image(self) -> np.ndarray:
        """The model image on the product grid, exactly as the fit formed it.

        Built from each linear object's mapping matrix, which is what the
        inversion actually used.  The rectangular mesh mapper *interpolates*
        between mesh pixel centres rather than block-replicating them, so
        reshaping the reconstruction and repeating each value over its block
        is not the model: it differs by up to ~45% per pixel.
        """
        inv = self.fit.inversion
        slim = None
        for obj, values in inv.reconstruction_dict.items():
            contrib = np.asarray(obj.mapping_matrix) @ np.asarray(values)
            slim = contrib if slim is None else slim + contrib
        return np.asarray(
            ag.Array2D(
                values=slim, mask=self.fit.dataset.real_space_mask
            ).native
        )

    @property
    def posterior_covariance(self) -> np.ndarray:
        """Posterior covariance of the reconstruction, (F + H)^-1.

        For a linear inversion with Gaussian noise and a Gaussian (GP) prior
        the posterior is Gaussian with this covariance, where F is the
        curvature A^T C_d^-1 A and H the regularisation matrix.

        NOTE this is conditional on the prior *and its fitted
        hyperparameters*, and on the noise map being correct.  It does not
        include the systematic error from the prior itself being wrong, which
        on a poorly-sampled field is usually the larger effect.
        """
        return np.linalg.inv(np.asarray(self.fit.inversion.curvature_reg_matrix))

    @property
    def sampling_covariance(self) -> np.ndarray:
        """Covariance of the *estimator*: (F+H)^-1 F (F+H)^-1.

        The MAP solution is s = (F+H)^-1 D with D = A^T C_d^-1 d, so over
        noise realisations Var(D) = F and the estimator scatters with this
        covariance -- smaller than the posterior, because regularisation
        shrinks the solution.  Verified against a 30-realisation Monte Carlo
        to 0.4%.
        """
        cov = self.posterior_covariance
        return cov @ np.asarray(self.fit.inversion.curvature_matrix) @ cov

    def _propagate(self, cov: np.ndarray) -> np.ndarray:
        """sqrt(diag(M C M^T)) on the image grid, for a parameter covariance."""
        var = None
        for obj, _ in self.fit.inversion.reconstruction_dict.items():
            M = np.asarray(obj.mapping_matrix)
            sl = self._parameter_slice(obj)
            block = cov[sl, :][:, sl]
            contrib = np.einsum("ij,jk,ik->i", M, block, M, optimize=True)
            var = contrib if var is None else var + contrib
        return np.asarray(
            ag.Array2D(
                values=np.sqrt(np.clip(var, 0.0, None)),
                mask=self.fit.dataset.real_space_mask,
            ).native
        )

    @property
    def model_uncertainty_sampling(self) -> np.ndarray:
        """1-sigma scatter of the model image over noise realisations.

        How much each pixel would jitter if the source were re-observed with
        the same array and the same prior.  It does NOT include the smoothing
        bias, which is typically the larger term (~3x this on our test mock).
        """
        return self._propagate(self.sampling_covariance)

    @property
    def model_uncertainty(self) -> np.ndarray:
        """Per-pixel 1-sigma uncertainty of the model image [Jy/pixel].

        The mesh mapper interpolates, so an image pixel is a weighted sum of
        several mesh pixels whose errors are strongly correlated (over roughly
        the prior's correlation length).  The uncertainty is therefore
        propagated properly as sqrt(diag(M C M^T)) rather than by copying
        per-mesh-pixel errors onto the image grid, which would misstate it.

        This is the Bayesian posterior width.  Measured against a Monte Carlo
        on our test mock it is 1.25x the *total* rms error from the truth, so
        it is a reasonable (slightly conservative) error bar -- but see
        `model_uncertainty_sampling` for the purely random part, and note that
        the smoothing bias, not the noise, dominates the difference.
        """
        return self._propagate(self.posterior_covariance)

    def model_image_at_scale(self, factor: float) -> np.ndarray:
        """The model image with the regularisation strength scaled by `factor`.

        H = coefficient x C^-1 scales linearly with the coefficient, so this
        needs no refit: re-solve (F + factor*H) s = D and re-map.  Returns
        None if the rescaled system is not positive definite (weakening the
        prior far enough makes F alone singular wherever the mesh has pixels
        the uv coverage does not constrain).
        """
        inv = self.fit.inversion
        F = np.asarray(inv.curvature_matrix)
        H = np.asarray(inv.regularization_matrix)
        D = np.asarray(inv.data_vector)
        try:
            values = np.linalg.solve(F + float(factor) * H, D)
        except np.linalg.LinAlgError:
            return None
        slim = None
        for obj, _ in inv.reconstruction_dict.items():
            M = np.asarray(obj.mapping_matrix)
            sl = self._parameter_slice(obj)
            contrib = M @ values[sl]
            slim = contrib if slim is None else slim + contrib
        return np.asarray(
            ag.Array2D(
                values=slim, mask=self.fit.dataset.real_space_mask
            ).native
        )

    def prior_systematic(self, spread_dex: float = 0.5) -> np.ndarray:
        """Per-pixel systematic from the choice of regularisation strength.

        The statistical error `model_uncertainty` is conditional on one prior
        *and* one strength for it.  How far the model moves when that strength
        is varied over the range these data allow is a measurable systematic,
        and on compact features it is the larger term.  Same construction as
        the point-source flux systematic, which turned pulls of up to 24 sigma
        into pulls under 3.

        This does not capture the prior *family* being wrong -- nothing
        cheap does.
        """
        base = self.model_image
        worst = np.zeros_like(base)
        for factor in (10.0**spread_dex, 10.0**-spread_dex):
            alt = self.model_image_at_scale(factor)
            if alt is None:
                continue
            worst = np.maximum(worst, np.abs(alt - base))
        return worst

    def model_uncertainty_total(
        self, spread_dex: float = 0.5, deblock: bool = True
    ) -> tuple[np.ndarray, dict]:
        """Single total 1-sigma map on the model image [Jy/pixel].

        sqrt(statistical^2 + systematic^2), with the mesh/image checkerboard
        removed.  Returns (map, terms) where `terms` holds the two components
        and the median of each, for the header and the parameter record.

        **The checkerboard.** Products live on a grid `oversample` times finer
        than the model mesh, and the mapper interpolates: a pixel sitting on a
        mesh node inherits one mesh pixel's variance, while a pixel between
        nodes is a weighted average of several and has a genuinely smaller
        one.  Both numbers are correct, but the alternating pattern is an
        artefact of the two grids, not of the sky, and it lands straight in
        any significance map the user makes.  `deblock` replaces it with its
        upper envelope (see `_deblock`), which is the conservative choice:
        an over-stated error never manufactures a detection.
        """
        stat = self.model_uncertainty
        sys_ = self.prior_systematic(spread_dex)
        total = np.hypot(stat, sys_)
        raw_total = total
        if deblock:
            total = _deblock(total, self.geometry.oversample)
        terms = {
            "statistical_median": float(np.nanmedian(stat)),
            "systematic_median": float(np.nanmedian(sys_)),
            "total_median": float(np.nanmedian(total)),
            "checkerboard_amplitude": float(_block_contrast(
                raw_total, self.geometry.oversample)),
            "systematic_spread_dex": float(spread_dex),
            "deblocked": bool(deblock),
        }
        return total, terms

    def _parameter_slice(self, obj):
        """Index range of one linear object within the stacked parameter vector."""
        start = 0
        for other, values in self.fit.inversion.reconstruction_dict.items():
            n = np.asarray(values).size
            if other is obj:
                return slice(start, start + n)
            start += n
        raise KeyError("linear object not found in the inversion")

    def aperture_uncertainty(self, region: np.ndarray) -> float:
        """1-sigma uncertainty on the summed flux inside a region [Jy].

        Per-pixel errors must NOT be added in quadrature: the posterior
        covariance is strongly correlated over roughly the prior's correlation
        length.  Quadrature is then wrong in either direction depending on the
        correlation structure -- on our test mock it *overstates* a compact
        aperture's error by ~1.4x, because interpolated pixels are
        anticorrelated with their neighbours.  This computes w^T (M C M^T) w
        properly, with w the region's indicator on the image grid.

        The result covers the random error only.  The smoothing bias is
        typically larger: on the same mock a knot's aperture flux came out
        0.0115 +/- 0.0001 against a true 0.0129, an 11% offset at ~12 sigma.

        Parameters
        ----------
        region
            Boolean array on the product grid selecting the aperture.
        """
        region = np.asarray(region, dtype=bool)
        if region.shape != self.geometry.shape_native:
            raise ValueError(
                f"region shape {region.shape} != image grid "
                f"{self.geometry.shape_native}"
            )
        mask = self.fit.dataset.real_space_mask
        w_native = region.astype(float)
        w = np.asarray(ag.Array2D(values=w_native, mask=mask).slim)
        cov = self.posterior_covariance
        var = 0.0
        for obj, _ in self.fit.inversion.reconstruction_dict.items():
            M = np.asarray(obj.mapping_matrix)
            sl = self._parameter_slice(obj)
            v = M.T @ w                      # aperture weights in mesh space
            var += float(v @ cov[sl, :][:, sl] @ v)
        return float(np.sqrt(max(var, 0.0)))

    @property
    def model_visibilities(self) -> np.ndarray:
        return np.asarray(self.fit.model_data)

    @property
    def residual_visibilities(self) -> np.ndarray:
        return np.asarray(self.fit.dataset.data) - self.model_visibilities

    @property
    def log_evidence(self) -> float:
        return _safe_evidence(self.fit)

    @property
    def chi_squared(self) -> float:
        return _chi_squared(self.fit)


def fit_dataset(
    dataset: ag.Interferometer,
    geometry: ImageGeometry,
    reg_kind: str = "matern",
    prior: dict | None = None,
    positive_only: bool = True,
    enforce_positive: bool = False,
    criterion: str = "discrepancy",
    nu: float = DEFAULT_NU,
    fixed_scale: float | None = None,
    envelope: dict | None = None,
    optimise_envelope: bool = False,
    chi2_target: float = 1.0,
    adapt_image=None,
    warn_on_chi2: bool = True,
) -> SingleFit:
    """Fit one dataset, optimising the source prior unless `prior` is given.

    `warn_on_chi2=False` suppresses the "does not reproduce the data" warning
    for a fit that is an intermediate stage rather than the answer. The demo
    is the case: its 4 mJy point source is one no mesh can hold, so every
    mesh-only pass sits at chi^2/N ~ 2.9 and told the user three times that
    the products should not be trusted -- before the point fit that brings it
    to 1.000. The caller that knows more is the one that should speak.
    """
    """Fit one Interferometer dataset (one channel, or the MFS stack).

    ``prior`` fixes the source-prior hyperparameters (keys ``coefficient``
    and, for kernel schemes, ``scale`` in arcsec); if ``None`` they are
    optimised.  ``reg_kind="adapt"`` runs two stages: a first pass with
    constant regularisation whose model becomes the brightness map that the
    second pass's adaptive regularisation follows.
    """
    mesh_shape = geometry.mesh_shape

    envelope = dict(envelope or {})
    if reg_kind in ADAPTIVE_REGULARIZATIONS and envelope.get("brightness") is None:
        logger.info("%s prior: first pass (plain Matern)...", reg_kind)
        first = fit_dataset(
            dataset, geometry, reg_kind="matern", prior=None,
            positive_only=positive_only, criterion=criterion, nu=nu,
            fixed_scale=fixed_scale, chi2_target=chi2_target,
            enforce_positive=enforce_positive,
        )
        envelope["brightness"] = np.clip(first.model_mesh_image.ravel(), 0.0, None)
        # Drop the first-pass fit before the second one allocates. It holds an
        # `ag.FitInterferometer`, and so the transformed mapping matrix --
        # n_vis x n_mesh complex, several GB on a real dataset -- while all
        # that is needed from it is the brightness array just copied out.
        # Keeping it alive roughly doubles the peak, which is what OOM-killed
        # 9io9 twice at the exact moment the second pass started, on a fit
        # whose single-inversion estimate (5.3 GB) fitted comfortably.
        del first
        gc.collect()
        logger.info(
            "%s prior: second pass (tracks the first-pass model)", reg_kind,
        )

    # ---- positivity sanity check -------------------------------------
    # The non-negative solver is not always reliable: on some datasets it
    # returns a solution that ignores the prior entirely (identical chi^2 for
    # coefficients spanning eight orders of magnitude) and fits far worse than
    # the unconstrained solve. Silently shipping that would make every prior
    # look identical, so check once and fall back if it happens.
    if positive_only and prior is None:
        probe_scale = fixed_scale if reg_kind in KERNEL_REGULARIZATIONS else None
        probe_kwargs = dict(
            reg_scale=probe_scale, nu=nu, envelope=envelope,
            adapt_image=adapt_image,
        )

        def _probe(coefficient, positive):
            try:
                return _chi_squared(fit_at(
                    dataset, mesh_shape, reg_kind, coefficient,
                    positive_only=positive, **probe_kwargs))
            except Exception:
                return np.nan

        def _probe_model(coefficient, positive):
            """The reconstruction itself, which is what the prior acts on."""
            try:
                fit = fit_at(
                    dataset, mesh_shape, reg_kind, coefficient,
                    positive_only=positive, **probe_kwargs)
                model = np.asarray(fit.inversion.reconstruction, dtype=float)
                del fit  # the fit holds the transformed mapping matrix
                return model
            except Exception:
                return None

        free = _probe(1.0, False)
        constrained = _probe(1.0, True)
        n_vis = len(np.asarray(dataset.data))
        reason = None
        if (
            np.isfinite(free) and np.isfinite(constrained)
            and constrained > max(2.0 * free, free + 2 * n_vis)
        ):
            reason = (
                f"it fits far worse than the unconstrained solve "
                f"(chi^2 {constrained:.4g} vs {free:.4g})"
            )
        else:
            # Second, independent symptom: the solver *ignores* the prior.
            # Compare two strengths twelve decades apart -- but compare the
            # **reconstruction**, not chi^2.
            #
            # This used to test chi^2, and that was wrong. chi^2 being
            # insensitive to the coefficient is a normal property of
            # well-constrained data, not a symptom: on Ruby (439 data points
            # per model pixel) chi^2 moves 0.4% across twelve decades while
            # the model changes out of all recognition -- the structure ratio
            # runs 0.228 to 3.51 over the same range. The old test fired on
            # every such dataset and silently disabled positivity on a fit
            # where the prior was working perfectly well.
            #
            # The claim in the message is about the prior's effect on the
            # solution, so measure that. A solver that is genuinely ignoring
            # the prior returns the same model at both ends; a working one
            # cannot.
            weak_m = _probe_model(1e-3, True)
            strong_m = _probe_model(1e9, True)
            if (
                weak_m is not None and strong_m is not None
                and weak_m.shape == strong_m.shape
            ):
                scale = float(np.linalg.norm(weak_m))
                change = (
                    float(np.linalg.norm(strong_m - weak_m)) / scale
                    if scale > 0 else 0.0
                )
                if change < POSITIVITY_PRIOR_RESPONSE:
                    reason = (
                        f"the reconstruction changes by only {100 * change:.2g}% "
                        f"between regularisation strengths twelve decades "
                        f"apart, so it is ignoring the prior entirely"
                    )
        if reason is not None and enforce_positive:
            logger.warning(
                "the non-negative solver looks unreliable on this data: %s. "
                "Keeping positivity anyway because enforce_positive was "
                "requested -- the model will be non-negative, but the prior "
                "may have little effect on it.", reason,
            )
        elif reason is not None:
            logger.warning(
                "the non-negative solver is unreliable on this data: %s. "
                "Disabling positivity for this fit; the model may contain "
                "small negative values. Pass enforce_positive=True "
                "(--enforce-positive) to keep it regardless.", reason,
            )
            positive_only = False

    scan = None
    if prior is None:
        prior, scan = optimise_prior(
            dataset, geometry, reg_kind=reg_kind, criterion=criterion,
            nu=nu, fixed_scale=fixed_scale, envelope=envelope,
            optimise_envelope=optimise_envelope, adapt_image=adapt_image,
            chi2_target=chi2_target, positive_only=positive_only,
        )
    if "envelope_fwhm" in prior:
        envelope = {**envelope, "fwhm": float(prior["envelope_fwhm"])}
    prior = dict(prior)
    if reg_kind in KERNEL_REGULARIZATIONS:
        prior.setdefault("nu", nu)

    def _fit(coefficient: float) -> ag.FitInterferometer:
        return fit_at(
            dataset, mesh_shape, reg_kind, coefficient,
            positive_only=positive_only, reg_scale=prior.get("scale"),
            nu=prior.get("nu", nu), envelope=envelope, adapt_image=adapt_image,
        )

    fit = _fit(prior["coefficient"])

    # The hyperparameter search uses the fast unconstrained solver, but the
    # final fit may impose positivity, which raises chi^2. When that shifts
    # the fit off the noise level, re-bisect the coefficient with the solver
    # actually in use so the delivered model really does fit to the noise.
    if (
        scan is not None and criterion == "discrepancy" and positive_only
        # a search that gave up on chi^2 and fell back to the evidence must
        # not then have its answer re-bisected against the target it gave up on
        and "->evidence" not in scan.criterion
    ):
        n_data = 2 * len(np.asarray(dataset.data))
        # The search probed the weakest prior on this same solver, so we know
        # what chi^2 the constrained fit can actually reach. If that floor is
        # above the target, bisecting towards the target walks the
        # coefficient to its lower bound and switches the prior off; aim just
        # above the floor instead.
        target = effective_chi2_target(
            chi2_target * n_data, scan.chi2_floor, n_data
        )
        if target > chi2_target * n_data:
            logger.info(
                "the constrained fit cannot go below chi2/N = %.4g, so the "
                "coefficient is chosen against %.4g rather than %.4g.",
                scan.chi2_floor / n_data, target / n_data, chi2_target,
            )
        chi2 = _chi_squared(fit)
        # A few per cent, not the 50%% this used to allow: chi^2 is nearly
        # flat in the coefficient near the floor, so a loose gate lets a
        # badly over- or under-smoothed model through as "close enough".
        if not np.isfinite(chi2) or abs(
            np.log10(max(chi2, 1e-30) / target)
        ) > np.log10(1.0 + chi2_rebisect_tolerance(n_data)):
            logger.info(
                "positivity moved chi2/N to %.4g; re-optimising the "
                "coefficient with the constrained solver...",
                chi2 / n_data if np.isfinite(chi2) else np.nan,
            )
            # Bisect using only constrained evaluations, tracked separately
            # from the (unconstrained) hyperparameter search trials.
            tried: list[tuple[float, float]] = [(prior["coefficient"], chi2)]
            lo, hi = LOG_COEFFICIENT_BOUNDS[0], np.log10(prior["coefficient"])
            while chi2 < target and hi < MAX_LOG_COEFFICIENT:
                hi = min(hi + 3.0, MAX_LOG_COEFFICIENT)
                trial = _fit(10.0**hi)
                chi2 = _chi_squared(trial)
                tried.append((10.0**hi, chi2))
                logger.info(
                    "  coefficient=%.4g (positive)  chi2/N=%.4g",
                    10.0**hi, chi2 / n_data if np.isfinite(chi2) else np.nan,
                )
            for _ in range(9):
                mid = 0.5 * (lo + hi)
                trial = _fit(10.0**mid)
                c = _chi_squared(trial)
                tried.append((10.0**mid, c))
                scan.record(
                    {**prior, "coefficient": 10.0**mid},
                    _safe_evidence(trial), c,
                )
                logger.info(
                    "  coefficient=%.4g (positive)  chi2/N=%.4g",
                    10.0**mid, c / n_data if np.isfinite(c) else np.nan,
                )
                if not np.isfinite(c) or c < target:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.02:
                    break
            usable = [
                (co, c) for co, c in tried if np.isfinite(c) and c > 0
            ]
            if usable:
                prior["coefficient"] = float(
                    min(usable, key=lambda t: abs(np.log10(t[1] / target)))[0]
                )
            else:
                prior["coefficient"] = 10.0 ** (0.5 * (lo + hi))
            fit = _fit(prior["coefficient"])
            logger.info(
                "  chosen coefficient=%.4g (chi2/N=%.4g)",
                prior["coefficient"], _chi_squared(fit) / n_data,
            )
            scan.best = dict(prior)

            # Second, stronger check on the non-negative solver.  The probe
            # above compares it with the unconstrained solve at a single
            # coefficient, and that misses the case where it merely *ignores*
            # the prior: on one dataset every trial spanning many decades of
            # coefficient returned the same answer, which no real prior can do.
            #
            # As with the earlier probe, judge the **reconstruction** and not
            # chi^2. A flat chi^2 across the bisection is normal on
            # well-constrained data -- Ruby moves 0.4% across twelve decades
            # -- and testing it here disabled positivity on a fit whose prior
            # was working, with the structure ratio running 0.228 to 3.51 over
            # the same range.
            # tried[0] is the coefficient the *unconstrained* search chose, so
            # it sits apart from the rest; judge on the constrained trials.
            finite = [
                (co, c) for co, c in tried[1:] if np.isfinite(c) and c > 0
            ]
            if len(finite) >= 3 and not enforce_positive:
                decades = np.ptp(np.log10([co for co, _ in finite]))
                lo_co = min(co for co, _ in finite)
                hi_co = max(co for co, _ in finite)
                change = _model_response(_fit, lo_co, hi_co)
                if (
                    decades > 3.0 and change is not None
                    and change < POSITIVITY_PRIOR_RESPONSE
                ):
                    logger.warning(
                        "the non-negative solver's reconstruction changes by "
                        "only %.2g%% across %.0f decades of regularisation "
                        "strength: it is ignoring the prior. Disabling "
                        "positivity for this fit; the model may contain "
                        "small negative values.",
                        100 * change, decades,
                    )
                    positive_only = False
                    prior, scan = optimise_prior(
                        dataset, geometry, reg_kind=reg_kind,
                        criterion=criterion, nu=nu, fixed_scale=fixed_scale,
                        envelope=envelope, optimise_envelope=optimise_envelope,
                        adapt_image=adapt_image, chi2_target=chi2_target,
                    )
                    if reg_kind in KERNEL_REGULARIZATIONS:
                        prior.setdefault("nu", nu)
                    fit = _fit(prior["coefficient"])

    chi2_final = _chi_squared(fit)
    n_data_final = 2 * len(np.asarray(dataset.data))
    if np.isfinite(chi2_final) and warn_on_chi2:
        ratio = chi2_final / (chi2_target * n_data_final)
        if ratio > CHI2_UNREACHABLE_FACTOR:
            logger.warning(
                "chi^2/N = %.3g against a target of %.3g: the model does not "
                "reproduce the data and its products should not be trusted. "
                "Usual causes are emission outside --fov, a mesh too coarse "
                "for the S/N, or an underestimated noise map.",
                chi2_final / n_data_final, chi2_target,
            )

    return SingleFit(
        fit=fit, geometry=geometry, prior=prior, scan=scan,
        positive_only=bool(positive_only),
    )

"""The sparse (w-tilde) inversion: guards, kernel identity, memory model.

`--inversion sparse` replaces the dense n_vis x n_mesh mapping matrix with the
w-tilde formalism: one streaming pass over the visibilities accumulates a
translation-invariant kernel of shape (2Ny, 2Nx), and the curvature matrix is
then assembled from the mapper's sparse triplets by FFT convolution against
that kernel. Measured against the dense path on Ruby 200 GHz continuum
(fov 3, mesh 16, matern, coefficient 1e8):

    chi^2      305200.43 both ways -- identical to eight significant figures
    flux       agreeing to seven figures
    time       0.3 s sparse against 25.4 s dense
    kernel     0.10 MB, against a 21.6 GB dense mapping matrix on Ruby CO

It is not an approximation. What it is, is narrower: it needs JAX, the
curvature matrix comes from one mapper so point sources are silently omitted,
and every channel of a cube would need its own kernel.

These tests cover the parts that run without JAX -- the guards, the cache key,
the memory model. The end-to-end equivalence above was measured on hardware
that has JAX; nothing here re-measures it.
"""

import numpy as np
import pytest

from pyuvimage import fitting


# --------------------------------------------------------------------------
# The kernel cache key
# --------------------------------------------------------------------------
class _Geom:
    """Enough of an ImageGeometry for the key to be computed."""

    def __init__(self, shape=(64, 64), pixel_scale=0.05, mesh_shape=(16, 16)):
        self.shape_native = shape
        self.pixel_scale = pixel_scale
        self.mesh_shape = mesh_shape


def _uv(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1e5, size=(n, 2))


def _noise(n=500, seed=0):
    rng = np.random.default_rng(seed + 1)
    return (rng.uniform(0.5, 1.5, n) + 1j * rng.uniform(0.5, 1.5, n))


def test_kernel_key_is_stable():
    """The same inputs must give the same key, or the cache never hits."""
    uv, n, g = _uv(), _noise(), _Geom()
    assert fitting.sparse_kernel_key(uv, n, g) == fitting.sparse_kernel_key(uv, n, g)


def test_kernel_key_tracks_uv_coverage():
    uv, n, g = _uv(), _noise(), _Geom()
    other = uv.copy()
    other[0, 0] += 1.0
    assert fitting.sparse_kernel_key(uv, n, g) != fitting.sparse_kernel_key(other, n, g)


def test_kernel_key_tracks_noise():
    """The kernel is inverse-variance weighted, so the noise is part of it.

    This one matters in practice: recentring the field pools the real and
    imaginary sigmas without touching a single uv coordinate, so a key built
    on uv alone would hand a recentred fit the wrong kernel and it would look
    perfectly healthy.
    """
    uv, n, g = _uv(), _noise(), _Geom()
    other = n.copy()
    other[0] *= 1.01
    assert fitting.sparse_kernel_key(uv, n, g) != fitting.sparse_kernel_key(uv, other, g)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (128, 128)},
        {"pixel_scale": 0.04},
        {"mesh_shape": (20, 20)},
    ],
)
def test_kernel_key_tracks_geometry(kwargs):
    uv, n = _uv(), _noise()
    a = fitting.sparse_kernel_key(uv, n, _Geom())
    b = fitting.sparse_kernel_key(uv, n, _Geom(**kwargs))
    assert a != b


def test_kernel_key_is_short():
    """A filename component, not a checksum to read aloud."""
    key = fitting.sparse_kernel_key(_uv(), _noise(), _Geom())
    assert len(key) == 16 and all(c in "0123456789abcdef" for c in key)


def test_cache_path_is_none_when_caching_is_off():
    assert fitting.sparse_kernel_cache_path(None, "abc") is None


def test_cache_path_carries_the_key_and_suffix(tmp_path):
    path = fitting.sparse_kernel_cache_path(tmp_path, "deadbeefdeadbeef")
    assert path.parent == tmp_path
    assert "deadbeefdeadbeef" in path.name
    assert path.name.endswith(fitting.SPARSE_KERNEL_SUFFIX)


# --------------------------------------------------------------------------
# The memory model
# --------------------------------------------------------------------------
def test_kernel_build_does_not_scale_with_visibilities():
    """The claim the whole feature rests on.

    `sparse_kernel_build_gb` has no n_vis argument at all -- that is the point,
    and this test exists so that a future signature change has to face it.
    """
    import inspect

    params = inspect.signature(fitting.sparse_kernel_build_gb).parameters
    assert "n_vis" not in params
    assert set(params) == {"n_image_pixels", "chunk_k"}


def test_kernel_build_scales_with_image_and_chunk():
    base = fitting.sparse_kernel_build_gb(4096, chunk_k=1024)
    assert fitting.sparse_kernel_build_gb(8192, chunk_k=1024) == pytest.approx(2 * base)
    assert fitting.sparse_kernel_build_gb(4096, chunk_k=2048) == pytest.approx(2 * base)


def test_kernel_build_matches_the_measured_peak():
    """Ruby continuum, fov 3: a 64x64 image at chunk_k 4096 peaked at 1.73 GB
    resident, of which roughly a gigabyte was the interpreter, numpy, autoarray
    and JAX's arena. The build term should be a few hundred megabytes, not tens.
    """
    gb = fitting.sparse_kernel_build_gb(64 * 64, chunk_k=4096)
    assert 0.3 < gb < 0.8


def test_a_cached_kernel_removes_the_build_term():
    args = dict(n_image_pixels=4096, n_mesh_pixels=256)
    cold = fitting.sparse_peak_memory_gb(**args, kernel_cached=False)
    warm = fitting.sparse_peak_memory_gb(**args, kernel_cached=True)
    assert warm < cold
    assert cold - warm == pytest.approx(fitting.sparse_kernel_build_gb(4096))


def test_the_sparse_ceiling_is_the_mesh_not_the_data():
    """F is n_mesh^2, so the curvature term goes as the fourth power of the
    mesh per side. That is the practical advice that replaces "reduce --fov":
    on the sparse path the data is free and the model is not.
    """
    kw = dict(n_image_pixels=4096, kernel_cached=True)
    a = fitting.sparse_peak_memory_gb(n_mesh_pixels=256, **kw)
    b = fitting.sparse_peak_memory_gb(n_mesh_pixels=512, **kw)
    c = fitting.sparse_peak_memory_gb(n_mesh_pixels=1024, **kw)
    # everything but the curvature term is common, so the differences isolate it
    assert (c - b) / (b - a) == pytest.approx(4.0, rel=1e-6)


def test_a_large_mesh_dominates_a_sparse_fit():
    """Where the crossover actually is, so the warning text is honest.

    At a 64x64 mesh the curvature term passes 0.4 GB and becomes the largest
    single allocation in a sparse fit; below that the fixed costs -- the padded
    FFT batch above all -- dominate and the mesh barely registers.
    """
    fixed = fitting.sparse_peak_memory_gb(4096, 1, kernel_cached=True)
    at_64 = fitting.sparse_peak_memory_gb(4096, 64 * 64, kernel_cached=True)
    assert at_64 - fixed > 0.4
    at_16 = fitting.sparse_peak_memory_gb(4096, 16 * 16, kernel_cached=True)
    assert at_16 - fixed < 0.01


def test_chunk_k_shrinks_to_fit_a_small_budget():
    chunk = fitting.sparse_chunk_k_for_budget(64 * 64, available_gb=0.5)
    assert chunk < fitting.SPARSE_CHUNK_K
    assert fitting.sparse_kernel_build_gb(64 * 64, chunk) <= 0.25 * 0.5 + 1e-9


def test_chunk_k_never_falls_below_the_floor():
    """A chunk of one visibility would be correct and unusably slow."""
    chunk = fitting.sparse_chunk_k_for_budget(1024 * 1024, available_gb=0.001)
    assert chunk == fitting.SPARSE_CHUNK_K_MIN


def test_chunk_k_is_capped_at_the_default_on_a_large_machine():
    chunk = fitting.sparse_chunk_k_for_budget(64 * 64, available_gb=1024.0)
    assert chunk == fitting.SPARSE_CHUNK_K


def test_chunk_k_falls_back_when_memory_is_unknown():
    assert fitting.sparse_chunk_k_for_budget(4096, available_gb=None) in (
        fitting.SPARSE_CHUNK_K,
        fitting.sparse_chunk_k_for_budget(4096),
    )


def test_check_memory_reports_the_sparse_figure(caplog):
    """The dense estimate would refuse a fit the sparse path finds comfortable.

    150k visibilities against a 256-pixel mesh is ~1.7 GB of dense mapping
    matrix and essentially nothing sparse; if `check_memory` reported the dense
    number under --inversion sparse it would be telling the user to shrink the
    fit that was chosen precisely so they would not have to.
    """
    import logging

    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        fitting.check_memory(
            150_000, 256, transformer_cls=None,
            inversion="sparse", n_image_pixels=64 * 64,
        )
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "sparse inversion" in text
    assert "independent of the 150000 visibilities" in text


def test_check_memory_is_silent_without_an_image_size():
    """No image size, no sparse estimate -- and certainly not a dense one."""
    fitting.check_memory(150_000, 256, inversion="sparse", n_image_pixels=None)


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------
def test_diagnosis_returns_a_string_or_none():
    reason = fitting.sparse_inversion_diagnosis()
    assert reason is None or isinstance(reason, str)


def test_diagnosis_names_what_is_missing():
    """If it is unavailable, the message must say which half is missing.

    A bare "sparse inversion unavailable" sends the user to the wrong place:
    an autoarray too old to have the w-tilde path and a missing JAX need
    different fixes.
    """
    reason = fitting.sparse_inversion_diagnosis()
    if reason is not None:
        assert "JAX" in reason or "autoarray" in reason


def test_with_sparse_operator_raises_the_diagnosis(monkeypatch):
    monkeypatch.setattr(
        fitting, "sparse_inversion_diagnosis", lambda: "no JAX here"
    )
    with pytest.raises(RuntimeError, match="no JAX here"):
        fitting.with_sparse_operator(object(), _uv(), _noise(), _Geom())


def _run_kwargs(tmp_path, n_chan=1, **over):
    """A minimal on-disk dataset, only ever read as far as the guards.

    Every guard fires before a single visibility is transformed, which is the
    point of them -- an OOM kill or a wrong-but-plausible image is no way to
    learn that a combination is unsupported.
    """
    rng = np.random.default_rng(0)
    n_vis = 64
    path = tmp_path / "d.npz"
    data = rng.normal(size=(n_chan, n_vis)) + 1j * rng.normal(size=(n_chan, n_vis))
    np.savez(
        path,
        uvw=rng.normal(0, 100.0, (n_vis, 3)),
        frequencies=np.linspace(100e9, 101e9, n_chan),
        data_re=data.real, data_im=data.imag,
        noise_re=np.full((n_chan, n_vis), 0.1),
        noise_im=np.full((n_chan, n_vis), 0.1),
    )
    kw = dict(dataset=str(path), fov=1.0, out=str(tmp_path / "out"),
              inversion="sparse")
    kw.update(over)
    return kw


def test_unknown_inversion_is_rejected(tmp_path):
    from pyuvimage import api

    with pytest.raises(ValueError, match="unknown inversion"):
        api.run(**_run_kwargs(tmp_path, inversion="wtilde"))


def test_sparse_refuses_point_sources(tmp_path):
    """Refuse rather than silently give up the benefit.

    Our point components are not autoarray linear objects -- they are a
    bordered system built on top of the framework's inversion, and it needs
    `inversion.operated_mapping_matrix` to form the mesh/point cross-terms.
    `InversionInterferometerSparse` inherits that property unchanged, so
    touching it triggers the dense n_vis x n_mesh build the w-tilde path
    exists to avoid. The fit would be correct and would allocate exactly what
    the user asked to escape, so the honest move is an error at the top of the
    run naming which of the two to give up.
    """
    from pyuvimage import api

    with pytest.raises(ValueError, match="cannot yet fit point sources"):
        api.run(**_run_kwargs(tmp_path, point_sources=True))


def test_sparse_accepts_cube_mode(tmp_path, caplog):
    """Cube mode is supported, and the guard that refused it is gone.

    Each channel's uv coordinates are the same baselines in metres scaled by
    its own frequency, so each needs its own kernel. That is cheap rather than
    expensive: a channel's build streams only that channel's visibilities, so
    n_chan builds over n_vis/n_chan each total one pass over the dataset --
    the same work as the single MFS kernel.

    This only checks that the guard no longer fires and that the cost is
    explained; the fit itself needs JAX, which this environment lacks.
    """
    import logging

    from pyuvimage import api

    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            api.run(**_run_kwargs(tmp_path, n_chan=4, mode="cube"))
    # whatever stops it, it must not be the old mfs-only refusal
    assert "mfs-only" not in str(excinfo.value)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "one w-tilde kernel per channel" in text


def test_the_per_channel_kernels_key_apart(tmp_path):
    """The cache needs no channel index: scaling uv by the channel frequency
    changes the hash on its own. If it did not, every channel would silently
    reuse the first channel's kernel."""
    rng = np.random.default_rng(0)
    uvw = rng.normal(0, 100.0, (256, 2))
    noise = np.full(256, 0.1 + 0.1j)
    g = _Geom()
    C = 299792458.0
    keys = {
        fitting.sparse_kernel_key(uvw * (f / C), noise, g)
        for f in (2.30e11, 2.31e11, 2.32e11)
    }
    assert len(keys) == 3


def test_inversion_is_recorded_in_the_parameter_file():
    """Whoever reads fit_parameters.json a year from now needs to know which
    algebra produced the image."""
    import inspect

    src = inspect.getsource(__import__("pyuvimage.api", fromlist=["api"]))
    assert '"inversion": inversion' in src


# --------------------------------------------------------------------------
# The operator's dirty image must be on the kernel's scale
# --------------------------------------------------------------------------
#
# `D = L^T dirty_image` while `W~` is accumulated straight from `1/sigma^2`,
# so the two are comparable only if the adjoint was scaled. autoarray builds
# that dirty image itself, via
#
#     self.transformer.image_from(..., use_adjoint_scaling=True)
#
# and whether the flag is passed has varied between releases: 2026.8.17.1
# passes it, 2026.8.23.1 (measured on Ruby: 8.898382e+02 against a correct
# 9.624490e+06, a factor of exactly 4*52*52) does not. It is a no-op for
# TransformerDFT and the nufftax TransformerNUFFT, so dropping it looks
# harmless and breaks only the pynufft path -- which is the path `auto`
# chooses for every dataset large enough to want the sparse inversion.
#
# pyuvimage therefore verifies and repairs rather than trusting the caller.

def _fake_operator(dirty_image):
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class Op:
        dirty_image: object

    return Op(dirty_image=dirty_image)


class _FakeDataset:
    def __init__(self, want, sparse_operator=None):
        self._want = want
        self.sparse_operator = sparse_operator


def _patched(monkeypatch, want):
    monkeypatch.setattr(fitting, "scaled_dirty_image", lambda ds: want)


def test_a_correctly_scaled_operator_is_left_alone(monkeypatch):
    want = np.arange(16.0)
    _patched(monkeypatch, want)
    op = _fake_operator(want.copy())
    ds = _FakeDataset(want, sparse_operator=op)
    out = fitting.repair_sparse_dirty_image(ds, _FakeDataset(want))
    assert out.sparse_operator is op, "an operator already on scale was rebuilt"


def test_an_unscaled_operator_is_repaired(monkeypatch):
    """The 10816x case, in miniature."""
    want = np.arange(1.0, 17.0)
    _patched(monkeypatch, want)
    ds = _FakeDataset(want, sparse_operator=_fake_operator(want / 10816.0))
    out = fitting.repair_sparse_dirty_image(ds, _FakeDataset(want))
    np.testing.assert_allclose(out.sparse_operator.dirty_image, want)


def test_the_repair_says_so(monkeypatch, caplog):
    """Silence here would hide an upstream regression from the next person."""
    import logging

    want = np.arange(1.0, 17.0)
    _patched(monkeypatch, want)
    ds = _FakeDataset(want, sparse_operator=_fake_operator(want / 10816.0))
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        fitting.repair_sparse_dirty_image(ds, _FakeDataset(want))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "adjoint scaling" in text
    assert "10816" in text


def test_a_shape_mismatch_is_reported_not_forced(monkeypatch, caplog):
    """Repairing across a shape change would be guessing, not fixing."""
    import logging

    want = np.arange(16.0)
    _patched(monkeypatch, want)
    ds = _FakeDataset(want, sparse_operator=_fake_operator(np.arange(9.0)))
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        out = fitting.repair_sparse_dirty_image(ds, _FakeDataset(want))
    assert out.sparse_operator.dirty_image.shape == (9,)
    assert "unverified" in "\n".join(r.getMessage() for r in caplog.records)


def test_no_operator_is_not_an_error(monkeypatch):
    want = np.arange(16.0)
    _patched(monkeypatch, want)
    ds = _FakeDataset(want, sparse_operator=None)
    assert fitting.repair_sparse_dirty_image(ds, _FakeDataset(want)) is ds


# --------------------------------------------------------------------------
# The w-tilde reduction assumes sigma_re == sigma_im
# --------------------------------------------------------------------------
#
# Confirmed by autoarray: unequal real and imaginary sigmas degrade agreement
# between the sparse and dense paths generally, not just for particular
# blocks. The asymmetry enters twice and differently -- `W~` is accumulated
# from `noise_map_real` alone, while the data vector's dirty image weights the
# two parts separately -- so `F` and `D` end up on different weightings.
#
# `--image-centre` pools them in quadrature, which removes the discrepancy and
# is independently the better noise estimate (twice the sample size), so a
# recentred fit satisfies the assumption exactly. These tests cover the rest.

def test_equal_sigmas_pass_quietly(caplog):
    import logging

    noise = np.full(64, 0.1 + 0.1j)
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        assert fitting.warn_on_reim_asymmetry(noise) == pytest.approx(0.0)
    assert caplog.records == []


def test_unequal_sigmas_are_reported_with_their_size(caplog):
    """The number belongs in the message: it bounds how well the two paths can
    be expected to agree.

    This warning is now for direct callers of `with_sparse_operator` -- scripts
    and tests. `api.run` pools the noise before the kernel is built, so a fit
    through the CLI cannot reach the kernel build with unequal sigmas, and the
    message names pooling rather than `--image-centre`, which used to be the
    only route to it.

    The line it warns above is `uvdata.REIM_ASYMMETRY_WARN`, the same one
    `api.run` draws when it pools -- there used to be a second, 2% line here,
    so the same dataset was told its noise was fine by one message and
    inconsistent by the next. Ruby unrecentred reads 9%: estimator scatter,
    below the shared line, and no longer a warning.
    """
    import logging

    from pyuvimage import uvdata

    assert fitting.SPARSE_REIM_ASYMMETRY_WARN == uvdata.REIM_ASYMMETRY_WARN

    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        ruby = fitting.warn_on_reim_asymmetry(np.full(64, 0.10 + 0.11j))
    assert ruby == pytest.approx(0.0952, rel=1e-2)
    assert caplog.records == [], "9% is scatter, not a warning"

    noise = np.full(64, 0.10 + 0.14j)
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        asym = fitting.warn_on_reim_asymmetry(noise)
    assert asym > uvdata.REIM_ASYMMETRY_WARN
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"{100 * asym:.1f}%" in text
    assert "pooling" in text.lower()
    assert "--inversion dense" in text


def test_a_recentred_dataset_satisfies_the_assumption_exactly():
    """Recentring is the fix, so check it actually equalises them."""
    from pyuvimage import uvdata

    rng = np.random.default_rng(0)
    n_vis = 128
    uvd = uvdata.UVData(
        uvw=rng.normal(0, 100.0, (n_vis, 3)),
        frequencies=np.array([2.3e11]),
        data=rng.normal(size=(1, n_vis)) + 1j * rng.normal(size=(1, n_vis)),
        noise=np.full((1, n_vis), 0.10 + 0.11j),
        meta={},
    )
    assert uvdata.reim_asymmetry(uvd.noise) > 0.09
    shifted = uvdata.shift_image_centre(uvd, (0.5, -0.5))
    assert uvdata.reim_asymmetry(shifted.noise) == pytest.approx(0.0, abs=1e-12)


def test_the_asymmetry_measure_survives_unusable_noise():
    from pyuvimage import uvdata

    assert uvdata.reim_asymmetry(np.zeros(8, dtype=complex)) == 0.0
    assert uvdata.reim_asymmetry(np.full(8, np.nan + 1j * np.nan)) == 0.0


# --------------------------------------------------------------------------
# JAX must be in 64-bit mode
# --------------------------------------------------------------------------
#
# JAX defaults to float32. The PyAuto stack knows this and ships
# `autonerves.jax_wrapper` (imported as `autoconf.jax_wrapper` at the top of
# the autolens_workspace scripts, before every other import) which sets
# JAX_ENABLE_X64=True because double precision "is required for most
# scientific computing applications". pyuvimage never imported it.
#
# Measured on Ruby with autoarray 2026.8.29.1: three UserWarnings from
# inversion_interferometer_util saying a requested float64 was "truncated to
# dtype float32" -- for `vals`, for `C0`, and for `F` itself. The curvature
# matrix, its regularised copy and the solve were all single precision.

def test_importing_pyuvimage_asks_jax_for_float64():
    """The environment variable is the mechanism, and it has to be set before
    `import jax` -- `jax.config.update` afterwards does not retrofit arrays
    that already exist."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    out = subprocess.run(
        [sys.executable, "-c",
         "import os; import pyuvimage; print(os.environ['JAX_ENABLE_X64'])"],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().lower() == "true"


def test_the_setting_happens_before_the_guard_imports_jax():
    """Ordering is the whole fix, and it is one line away from breaking.

    `disable_broken_jax()` does `import jax` at package-import time -- that is
    its entire purpose, to get in ahead of the PyAuto packages' on-disk check.
    But JAX reads JAX_ENABLE_X64 when it is imported, so our own guard was
    what pinned it to 32-bit: `autonerves.jax_wrapper` sets the variable when
    autogalaxy loads it, which is after we have already imported JAX, and by
    then it changes nothing.

    Swapping these two calls back would restore the bug and no test that
    merely checks the final value of the variable would notice, because
    autonerves sets it to True either way.
    """
    import inspect

    import pyuvimage

    src = inspect.getsource(pyuvimage)
    assert src.index("enable_double_precision()") < src.index(
        "disable_broken_jax()"
    ), "the 64-bit setting must be in the environment before JAX is imported"


def test_single_precision_is_reported(monkeypatch, caplog):
    import logging

    from pyuvimage import _jax_guard

    monkeypatch.setattr(_jax_guard, "jax_double_precision_active", lambda: False)
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        assert fitting.warn_on_single_precision() is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "32-bit" in text and "JAX_ENABLE_X64" in text


def test_double_precision_passes_quietly(monkeypatch, caplog):
    import logging

    from pyuvimage import _jax_guard

    monkeypatch.setattr(_jax_guard, "jax_double_precision_active", lambda: True)
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        assert fitting.warn_on_single_precision() is True
    assert caplog.records == []


def test_absent_jax_is_not_a_warning(monkeypatch, caplog):
    """No JAX means no sparse path at all; a precision warning would be noise."""
    import logging

    from pyuvimage import _jax_guard

    monkeypatch.setattr(_jax_guard, "jax_double_precision_active", lambda: None)
    with caplog.at_level(logging.WARNING, logger="pyuvimage.fitting"):
        assert fitting.warn_on_single_precision() is None
    assert caplog.records == []


# --------------------------------------------------------------------------
# `--inversion auto`
# --------------------------------------------------------------------------
#
# Sparse above 5000 visibilities, dense below -- but only where sparse can
# deliver the same answer. Every fallback below is a case where dense is
# faster, better conditioned, or the only one that works. An explicit
# `--inversion sparse` still raises rather than falling back: a user who asked
# for it by name wants to know it could not be given.

def test_auto_takes_dense_on_a_small_dataset(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        got = fitting.resolve_inversion("auto", n_vis=4999)
    assert got == "dense"
    assert "below the 5000" in "\n".join(
        r.getMessage() for r in caplog.records
    )


def test_auto_takes_sparse_above_the_threshold(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        got = fitting.resolve_inversion("auto", n_vis=5000)
    assert got == "sparse"
    assert "does not scale with the data" in "\n".join(
        r.getMessage() for r in caplog.records
    )


def test_the_threshold_is_inclusive_at_5000(monkeypatch):
    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    assert fitting.resolve_inversion("auto", n_vis=5000) == "sparse"
    assert fitting.resolve_inversion("auto", n_vis=4999) == "dense"


def test_auto_avoids_sparse_when_point_sources_are_wanted(monkeypatch):
    """Sparse cannot fit them yet, and auto must not turn that into an error."""
    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    assert fitting.resolve_inversion(
        "auto", n_vis=10**6, point_sources=True
    ) == "dense"


def test_auto_falls_back_when_sparse_is_unavailable(monkeypatch, caplog):
    """No JAX must degrade, not crash -- and must say which half is missing."""
    import logging

    monkeypatch.setattr(
        fitting, "sparse_inversion_diagnosis", lambda: "the sparse inversion needs JAX"
    )
    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        got = fitting.resolve_inversion("auto", n_vis=10**6)
    assert got == "dense"
    assert "needs JAX" in "\n".join(r.getMessage() for r in caplog.records)


def test_the_sigmas_never_decide_the_path(monkeypatch):
    """A re/im asymmetry is a warning now, not a fallback.

    Two thresholds were tried -- 5%, then 25% -- and real data arrived above
    each: Ruby reads 9.1% unrecentred, 9io9 15.6%. Each move silently changed
    which path a fit took, and a wrong guess costs a large dataset the dense
    mapping matrix, tens of GB and hours, for a difference the user is better
    placed to judge. `api.run` pools either way and says how far apart they
    were; nothing here refuses.
    """
    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    for sigma_im in (0.10, 0.109, 0.117, 0.20, 0.40):
        noise = np.full(64, 0.10 + 1j * sigma_im)
        assert fitting.resolve_inversion(
            "auto", n_vis=10**6, noise=noise) == "sparse", (
            f"sigma_im={sigma_im} sent auto to dense; the asymmetry is meant "
            "to change only what is logged"
        )


def test_auto_falls_back_on_a_scale_inconsistent_transformer(monkeypatch):
    """The condition that produced the 231 sigma fit. Under `auto` it is a
    reason to use dense, not to stop."""
    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)

    def bad(_cls):
        raise RuntimeError("does not honour use_adjoint_scaling")

    monkeypatch.setattr(fitting, "assert_adjoint_scale_consistent", bad)
    assert fitting.resolve_inversion(
        "auto", n_vis=10**6, transformer_cls=object) == "dense"


@pytest.mark.parametrize("choice", ["dense", "sparse"])
def test_an_explicit_choice_is_returned_untouched(choice, monkeypatch):
    """No silent downgrade of something the user named -- the guards in
    api.run raise instead, so the reason reaches them."""
    monkeypatch.setattr(
        fitting, "sparse_inversion_diagnosis", lambda: "no JAX"
    )
    assert fitting.resolve_inversion(
        choice, n_vis=1, point_sources=True) == choice


def test_an_unknown_inversion_is_still_rejected():
    with pytest.raises(ValueError, match="unknown inversion"):
        fitting.resolve_inversion("wtilde", n_vis=10)


# --------------------------------------------------------------------------
# The gather-buffer veto is a dense-path cost
# --------------------------------------------------------------------------
#
# `resolve_transformer` rejects the JAX NUFFT when its batched
# `transform_mapping_matrix` gather buffer (n_mesh x n_vis x nspread^2) will
# not fit. On the sparse path that call never happens: the data vector uses the
# plain mapping matrix, the curvature matrix comes from sparse triplets, and
# the transformer is asked only for one dirty image and one forward transform
# of the reconstructed image per likelihood call.
#
# Measured on 9io9 (164,262 visibilities, 1444 mesh pixels): the mapping-matrix
# buffer is 1488 GB and the single-image one is 1.03 GB. Vetoing on the former
# picked pynufft -- whose adjoint then needs the 4*N_y*N_x repair -- for a
# build that was never going to run.

def test_the_veto_still_applies_on_the_dense_path(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(fitting, "jax_available", lambda: True)
    monkeypatch.setattr(fitting, "pynufft_available", lambda: True)
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 12.1)
    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        cls = fitting.resolve_transformer(
            n_vis=164262, transformer="auto",
            n_image_pixels=116 * 116, n_mesh_pixels=1444, inversion="dense",
        )
    assert cls is fitting.pynufft_transformer_class()
    assert "gather buffer" in "\n".join(r.getMessage() for r in caplog.records)


def test_the_veto_does_not_apply_on_the_sparse_path(monkeypatch, caplog):
    """9io9's numbers exactly: 1488 GB for a build that never happens."""
    import logging

    monkeypatch.setattr(fitting, "jax_available", lambda: True)
    monkeypatch.setattr(fitting, "pynufft_available", lambda: True)
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 12.1)
    sentinel = object()
    monkeypatch.setattr(fitting, "_jax_nufft_class", lambda *a, **k: sentinel)
    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        cls = fitting.resolve_transformer(
            n_vis=164262, transformer="auto",
            n_image_pixels=116 * 116, n_mesh_pixels=1444, inversion="sparse",
        )
    assert cls is sentinel, "the sparse path was vetoed on a cost it never pays"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "one image at a time" in text


def test_the_single_image_buffer_is_what_sparse_actually_pays():
    """The arithmetic the change rests on, so a kernel-width change is caught."""
    mapping = fitting.nufftax_gather_gb(164262, 1444)
    single = fitting.nufftax_gather_gb(164262, 1)
    assert mapping > 1000
    assert single < 2
    assert mapping / single == pytest.approx(1444, rel=1e-6)


def test_an_explicit_transformer_is_unaffected_by_the_inversion():
    """`--transformer pynufft` means pynufft whatever the inversion."""
    for inversion in ("dense", "sparse"):
        assert fitting.resolve_transformer(
            n_vis=10**6, transformer="dft", inversion=inversion
        ) is __import__("autogalaxy").TransformerDFT


# --------------------------------------------------------------------------
# Unequal sigmas: pool rather than refuse
# --------------------------------------------------------------------------
#
# The W~ reduction assumes sigma_re == sigma_im. Thermal noise has them equal
# by construction, so a measured difference is scatter in the estimator, and
# pooling in quadrature is the better estimate of both -- twice the sample
# size, total variance unchanged. `--image-centre` already does this on every
# recentred fit.
#
# Refusing sparse on any asymmetry made the fast path depend on whether the
# user happened to recentre, which has nothing to do with the physics.
# Measured unrecentred: Ruby 9.1%, 9io9 15.6% -- both scatter, both pushed onto
# the dense path by the old 5% threshold. `pyuvimage fit 9io9.npz --fov 8`
# could not reach the sparse inversion at all.
#
# The threshold is gone entirely now, not just moved. Two were tried and real
# data arrived above each, and every move silently changed which path a fit
# took; the asymmetry decides the log *level* and nothing else. The tests
# below are on `describe_pooling`, which returns that level and the wording,
# so they run without JAX -- an end-to-end check of a warning would skip on
# every machine without it, which is exactly where a message that stopped
# appearing would go unnoticed.

def test_ordinary_estimator_scatter_is_reported_quietly(monkeypatch):
    """9io9's 15.6% and Ruby's 9.1% are scatter: pool, mention it, move on."""
    import logging

    from pyuvimage.uvdata import describe_pooling, reim_asymmetry

    for sigma_im in (0.109, 0.117):  # ~9% and ~16% asymmetry against 0.10
        noise = np.full(64, 0.10 + 1j * sigma_im)
        asymmetry = reim_asymmetry(noise)
        assert 0.05 < asymmetry < 0.25, (
            "this fixture is meant to sit in the scatter regime"
        )
        level, message = describe_pooling(asymmetry)
        assert level == logging.INFO
        assert "scatter in the noise estimator" in message


def test_a_difference_too_large_to_be_scatter_warns_but_still_fits(monkeypatch):
    """The change this test exists for: warn, do not fall back.

    Above the line, pooling weights the real and imaginary parts equally when
    the noise map says they should not be -- a real caveat on the result, and
    one the user can act on. It is not a reason to spend the dense path's
    memory on their behalf without asking, so the message has to carry both
    halves: what pooling did, and which flag avoids it.
    """
    import logging

    from pyuvimage.uvdata import describe_pooling

    level, message = describe_pooling(0.67)
    assert level == logging.WARNING
    assert "--inversion dense" in message
    assert "chi^2 statistics stay valid" in message

    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    assert fitting.resolve_inversion(
        "auto", n_vis=10**6, noise=np.full(64, 0.10 + 0.20j)) == "sparse"


def test_equal_sigmas_say_nothing_at_all():
    """Nothing to pool, so no line in the log about pooling."""
    from pyuvimage.uvdata import describe_pooling

    level, message = describe_pooling(0.0)
    assert level is None and message == ""


def test_the_warning_line_is_the_one_uvdata_already_draws():
    """One definition of "more than scatter", used by both messages."""
    import logging

    from pyuvimage import uvdata

    just_below = uvdata.REIM_ASYMMETRY_WARN - 1e-6
    just_above = uvdata.REIM_ASYMMETRY_WARN + 1e-6
    assert uvdata.describe_pooling(just_below)[0] == logging.INFO
    assert uvdata.describe_pooling(just_above)[0] == logging.WARNING


def test_pooling_preserves_total_variance():
    """The property that keeps chi^2 statistics untouched."""
    from pyuvimage.uvdata import pooled_noise

    noise = np.full(64, 0.10 + 0.13j)
    pooled = pooled_noise(noise)
    before = noise.real**2 + noise.imag**2
    after = pooled.real**2 + pooled.imag**2
    np.testing.assert_allclose(after, before)
    np.testing.assert_allclose(pooled.real, pooled.imag)

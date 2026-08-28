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


def test_sparse_refuses_cube_mode(tmp_path):
    """Each channel has its own uv coverage in wavelengths, so each needs its
    own kernel. Not wired up, so say so rather than silently reusing one."""
    from pyuvimage import api

    with pytest.raises(ValueError, match="mfs-only"):
        api.run(**_run_kwargs(tmp_path, n_chan=4, mode="cube"))


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

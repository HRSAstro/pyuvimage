"""The pynufft transformer must agree with the DFT.

`autoarray.TransformerNUFFTPyNUFFT` computes the half-pixel phase ramp that
aligns its grid convention with the DFT's -- `self.shift` -- in `__init__`, and
then never applies it. The two transformers in the same library therefore place
the image grid half a pixel apart in both axes. Measured on the demo mock, the
disagreement grows linearly with baseline length exactly as a phase error must:
9% of the visibility rms at the shortest baselines, 38% at the longest.

`TransformerPyNUFFT` applies it. These tests pin that down, because the failure
is silent: a fit built entirely on the uncorrected transformer still converges,
it just reconstructs the sky half a pixel from where every DFT-computed product
puts it.
"""

import numpy as np
import pytest

import autogalaxy as ag

from pyuvimage import fitting, mock
from pyuvimage.fitting import (
    pynufft_available,
    pynufft_transformer_class,
    resolve_transformer,
)

pynufft_only = pytest.mark.skipif(
    not pynufft_available(),
    reason="pynufft, or autoarray's TransformerNUFFTPyNUFFT, is unavailable",
)
TransformerPyNUFFT = pynufft_transformer_class()


@pytest.fixture(scope="module")
def problem():
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=400, mesh_n=16, seed=3)
    uv, data, noise = uvd.flattened()
    mask = ag.Mask2D.all_false(
        shape_native=geom.shape_native, pixel_scales=geom.pixel_scale
    )
    rng = np.random.default_rng(0)
    image = ag.Array2D(values=rng.normal(size=int(np.sum(~mask))), mask=mask)
    return uv, data, noise, geom, mask, image


# --- transformer selection (no pynufft needed) -----------------------------

@pynufft_only
def test_pynufft_can_be_asked_for_by_name():
    assert resolve_transformer(10, transformer="pynufft") is TransformerPyNUFFT


def test_asking_for_a_missing_pynufft_says_why(monkeypatch):
    """PyAutoArray PR #475 (2026.8.23.1) deleted TransformerNUFFTPyNUFFT
    outright, and an earlier pyuvimage subclassed it at import time -- so one
    `pip install -U autoarray` produced an AttributeError before pyuvimage
    could even import. The transformer is vendored now, so the only missing
    piece can be the pynufft package itself, and that is a clear message at
    the point of use, never an import failure."""
    monkeypatch.setattr(fitting, "pynufft_transformer_class", lambda: None)
    with pytest.raises(RuntimeError, match="pip install pynufft"):
        resolve_transformer(10, transformer="pynufft")


def test_auto_survives_pynufft_being_missing(monkeypatch):
    monkeypatch.setattr(fitting, "pynufft_available", lambda: False)
    monkeypatch.setattr(fitting, "jax_available", lambda: False)
    with pytest.warns(UserWarning):
        got = resolve_transformer(164_262, n_image_pixels=116 * 116)
    assert got is ag.TransformerDFT


def test_the_vendored_class_does_not_depend_on_the_deleted_upstream_one():
    """The regression that motivated vendoring: the transformer must build
    from pynufft directly, whether or not this autoarray still ships
    TransformerNUFFTPyNUFFT."""
    if not fitting.pynufft_available():
        pytest.skip("pynufft is not installed")
    cls = fitting.pynufft_transformer_class()
    upstream = getattr(ag, "TransformerNUFFTPyNUFFT", None)
    assert cls is not None
    assert upstream is None or not issubclass(cls, upstream)


def test_an_unknown_transformer_is_still_rejected():
    with pytest.raises(ValueError, match="unknown transformer"):
        resolve_transformer(10, transformer="nufftt")


def test_auto_keeps_the_dft_for_small_problems():
    assert resolve_transformer(100, n_image_pixels=100) is ag.TransformerDFT


@pynufft_only
def test_auto_reaches_for_pynufft_when_the_dft_cannot_cope(monkeypatch):
    """164k visibilities on a 116x116 image is 16.5 GB of DFT temporary --
    numpy raises MemoryError before it computes anything."""
    monkeypatch.setattr(fitting, "jax_available", lambda: False)
    got = resolve_transformer(164_262, n_image_pixels=116 * 116)
    assert got is TransformerPyNUFFT


# --- the correction itself -------------------------------------------------

@pynufft_only
def test_visibilities_match_the_dft(problem):
    uv, _, _, _, mask, image = problem
    ref = np.asarray(
        ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    got = np.asarray(
        TransformerPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    # gridding accuracy, not machine precision: pynufft's default plan is a
    # (6,6) kernel at 2x oversampling
    assert np.max(np.abs(got - ref)) / np.std(np.abs(ref)) < 1e-3


def _skip_without_upstream_legacy_class():
    """Two tests demonstrate the *upstream* class's half-pixel bug by running
    it uncorrected. PyAutoArray PR #475 (2026.8.23.1) deleted that class, so
    on newer installs the demonstration has nothing to run against -- the
    vendored transformer's own correctness tests still run everywhere."""
    if getattr(ag, "TransformerNUFFTPyNUFFT", None) is None:
        pytest.skip("autoarray no longer ships TransformerNUFFTPyNUFFT")


@pynufft_only
def test_the_uncorrected_transformer_really_is_off(problem):
    """Guard against the correction being quietly removed as a no-op -- and
    against upstream fixing it, which would make the wrapper double-shift.

    Doubly conditional: it needs pynufft installed *and* an autoarray old
    enough to still ship the legacy class. `@pynufft_only` used to sit on the
    helper below rather than on the tests, where a mark does nothing, so
    without pynufft this failed instead of skipping."""
    _skip_without_upstream_legacy_class()
    uv, _, _, _, mask, image = problem
    ref = np.asarray(
        ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    raw = np.asarray(
        ag.TransformerNUFFTPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    assert np.max(np.abs(raw - ref)) / np.std(np.abs(ref)) > 0.1


@pynufft_only
def test_the_correction_is_a_pure_phase(problem):
    _skip_without_upstream_legacy_class()
    """A half-pixel shift moves no power, so amplitudes were never wrong --
    which is why this survived: every amplitude-based check passes."""
    uv, _, _, _, mask, image = problem
    tr = TransformerPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
    raw = np.asarray(
        ag.TransformerNUFFTPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    got = np.asarray(tr.visibilities_from(image=image))
    assert np.allclose(np.abs(raw), np.abs(got))


@pynufft_only
def test_forward_and_adjoint_stay_adjoint(problem):
    """<Rx, y> == <x, R^T y>, the load-bearing assumption of the inversion.
    The adjoint carries conj(shift), so a one-sided fix would break this."""
    uv, data, _, _, mask, image = problem
    tr = TransformerPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
    rng = np.random.default_rng(1)
    y = rng.normal(size=len(data)) + 1j * rng.normal(size=len(data))
    Rx = np.asarray(tr.visibilities_from(image=image))
    Rty = np.asarray(tr.image_from(visibilities=ag.Visibilities(y)))
    lhs = float(np.real(np.vdot(Rx, y)))
    rhs = float(np.dot(np.asarray(image), Rty))
    # upstream returns the adjoint unscaled by `adjoint_scaling`; the constant
    # cancels everywhere it is used, so only its constancy is asserted here
    assert lhs / rhs == pytest.approx(4.0 * np.prod(mask.shape), rel=1e-4)


@pynufft_only
def test_a_fit_agrees_with_the_dft_end_to_end(problem):
    """The claim that actually matters: same data, same prior, two
    transformers, same reconstruction."""
    uv, data, noise, geom, _, _ = problem
    prior = {"coefficient": 1e5, "scale": 0.3, "nu": 1.5}
    out = {}
    for name in ("dft", "pynufft"):
        ds = fitting.make_dataset(uv, data, noise, geom, transformer=name)
        out[name] = fitting.fit_dataset(
            ds, geom, reg_kind="matern", prior=prior, positive_only=False
        )
    a, b = out["dft"], out["pynufft"]
    assert b.chi_squared == pytest.approx(a.chi_squared, rel=1e-3)
    am, bm = np.asarray(a.model_mesh_image), np.asarray(b.model_mesh_image)
    assert np.max(np.abs(bm - am)) / np.max(np.abs(am)) < 5e-3


# --- the same question, asked of the JAX backend ---------------------------

@pytest.mark.skipif(
    not fitting.jax_available(), reason="JAX/nufftax is not installed"
)
def test_the_jax_nufft_agrees_with_the_dft(problem):
    """Does `nufftax` share the pynufft backend's half-pixel convention?

    Unanswered when this was written: no container available here can run JAX,
    and the question matters -- if it does, every JAX-path fit reconstructs
    the sky half a pixel from where the DFT puts it, silently, because a
    self-consistent fit still converges and the amplitudes are untouched.

    This test answers it on any machine that has JAX. If it fails with an
    error growing linearly in baseline length, nufftax needs the same
    correction `pynufft_transformer_class` applies.
    """
    uv, _, _, _, mask, image = problem
    ref = np.asarray(
        ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    got = np.asarray(
        ag.TransformerNUFFT(uv_wavelengths=uv, real_space_mask=mask)
        .visibilities_from(image=image)
    )
    err = np.abs(got - ref) / np.std(np.abs(ref))
    b = np.hypot(uv[:, 0], uv[:, 1])
    long_half = err[b > np.median(b)].mean()
    short_half = err[b <= np.median(b)].mean()
    assert np.max(err) < 1e-3, (
        f"nufftax disagrees with the DFT: mean relative error "
        f"{short_half:.3g} on short baselines and {long_half:.3g} on long "
        f"ones. Growing with baseline length means a phase offset -- almost "
        f"certainly the same half-pixel shift the pynufft backend has."
    )


# --- why the JAX path is missing, when it is -------------------------------

def test_a_missing_nufftax_is_diagnosed_separately(monkeypatch):
    """The quiet failure this exists to catch: nufftax is not part of a
    default install (it sits in autoarray's `optional` extra), so a freshly
    built environment routinely has a perfect JAX and no fast path -- and
    `_jax_guard` says nothing, because JAX is not broken."""
    import builtins

    real_import = builtins.__import__

    def no_nufftax(name, *args, **kwargs):
        if name == "nufftax":
            raise ImportError("no nufftax")
        if name == "jax":
            return object()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_nufftax)
    diagnosis = fitting.jax_path_diagnosis()
    assert diagnosis is not None
    assert "nufftax" in diagnosis
    assert "0.6.1" in diagnosis, "the version floor is the part that bites"
    assert fitting.jax_available() is False


def test_a_broken_jax_is_diagnosed_as_a_different_thing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken_jax(name, *args, **kwargs):
        if name == "jax":
            raise AttributeError("partially initialized module 'jax'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_jax)
    diagnosis = fitting.jax_path_diagnosis()
    assert diagnosis is not None and "cannot be imported" in diagnosis
    assert "nufftax" not in diagnosis


# --- the adjoint scale ------------------------------------------------------
#
# pynufft's adjoint applies its own internal IFFT normalisation, leaving it a
# factor `4 * N_y * N_x` below the plain mathematical adjoint that both
# `TransformerDFT` and the nufftax `TransformerNUFFT` return. autoarray's
# `Interferometer.apply_sparse_operator` passes `use_adjoint_scaling=True` when
# it builds the w-tilde operator's dirty image, precisely so `D` lands on the
# same scale as the kernel `W~`.
#
# Our vendored `image_from` used to end in `**kwargs`, which swallowed that
# argument without a word. The consequence was not a crash: the sparse
# reconstruction had the right morphology at ~1/10000 of the right amplitude,
# the residual map kept 99.99% of the source (231 sigma on Ruby), and chi^2 was
# identical to eleven significant figures across twelve orders of magnitude of
# `lambda` -- because an `F` that large swamps any `H` in the search range --
# so the hyperparameter search had nothing to bisect and ran to its ceiling.
#
# Every one of these tests exists to make that failure loud.

def _adjoint_problem(n_pix=16, n_vis=300, pscale=0.1, seed=0):
    rng = np.random.default_rng(seed)
    mask = ag.Mask2D.all_false(shape_native=(n_pix, n_pix), pixel_scales=pscale)
    pixel_rad = pscale * np.pi / (180 * 3600)
    nyquist = 1.0 / (2.0 * pixel_rad)
    uv = rng.uniform(-0.4 * nyquist, 0.4 * nyquist, size=(n_vis, 2))
    vis = ag.Visibilities(
        visibilities=rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
    )
    return mask, uv, vis


@pynufft_only
def test_image_from_accepts_use_adjoint_scaling_by_name():
    """Not via **kwargs. The signature is the thing that broke."""
    import inspect

    params = inspect.signature(TransformerPyNUFFT.image_from).parameters
    assert "use_adjoint_scaling" in params
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), "a **kwargs catch-all is how this argument got silently dropped before"


@pynufft_only
@pytest.mark.parametrize("n_pix", [16, 24, 32])
def test_the_scaled_adjoint_matches_the_dft(n_pix):
    """The invariant the sparse inversion's data vector depends on."""
    mask, uv, vis = _adjoint_problem(n_pix=n_pix)
    reference = np.asarray(
        ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
        .image_from(visibilities=vis).native
    )
    scaled = np.asarray(
        TransformerPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
        .image_from(visibilities=vis, use_adjoint_scaling=True).native
    )
    error = np.abs(scaled - reference).max() / np.abs(reference).max()
    assert error < 1e-3, f"relative error {error:.2e}"


@pynufft_only
def test_the_factor_is_four_ny_nx():
    """Measured, not assumed -- our transformer is a reimplementation."""
    mask, uv, vis = _adjoint_problem(n_pix=24)
    pn = TransformerPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
    raw = np.asarray(pn.image_from(visibilities=vis).native)
    reference = np.asarray(
        ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
        .image_from(visibilities=vis).native
    )
    keep = np.abs(raw) > np.abs(raw).max() * 1e-3
    measured = np.median(reference[keep] / raw[keep])
    assert measured == pytest.approx(4 * 24 * 24, rel=1e-3)
    assert pn.adjoint_scaling == pytest.approx(4 * 24 * 24)


@pynufft_only
def test_the_default_is_still_the_unscaled_adjoint():
    """Everything else in pyuvimage reads the dirty image unscaled; only the
    sparse operator asks for the common scale. Changing the default would move
    every dirty image and beam product at once."""
    mask, uv, vis = _adjoint_problem()
    pn = TransformerPyNUFFT(uv_wavelengths=uv, real_space_mask=mask)
    a = np.asarray(pn.image_from(visibilities=vis).native)
    b = np.asarray(pn.image_from(visibilities=vis, use_adjoint_scaling=False).native)
    np.testing.assert_allclose(a, b)


@pynufft_only
def test_the_preflight_guard_passes_a_correct_transformer():
    fitting.assert_adjoint_scale_consistent(TransformerPyNUFFT)


@pynufft_only
def test_the_preflight_guard_catches_a_swallowed_argument():
    """Reintroduce the exact bug and confirm the guard names it."""

    class Swallowing(TransformerPyNUFFT):
        def image_from(self, visibilities, xp=np, **kwargs):
            return super().image_from(visibilities, use_adjoint_scaling=False)

    with pytest.raises(RuntimeError, match="does not honour use_adjoint_scaling"):
        fitting.assert_adjoint_scale_consistent(Swallowing)


def test_the_preflight_guard_passes_the_dft():
    """A no-op for the DFT, and it must stay a no-op rather than an error."""
    fitting.assert_adjoint_scale_consistent(ag.TransformerDFT)

"""The DFT's cost is n_image_pixels * n_vis, not either factor alone.

Found on the first well-formed real dataset: 5158 visibilities -- well under
DFT_MAX_VIS -- on a 30" field at 0.19" resolution is 384400 image pixels, and
autoarray asked the OS for 14.8 GB before anything else could go wrong.
"""

import warnings

import pytest

from pyuvimage import fitting
from pyuvimage.fitting import DFT_MAX_PRODUCT, DFT_MAX_VIS, resolve_transformer


def _name(cls) -> str:
    return cls.__name__


def test_small_problem_uses_the_dft():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cls = resolve_transformer(1000, n_image_pixels=4096)
    assert _name(cls) == "TransformerDFT"


def test_explicit_choices_ignore_both_limits():
    huge = int(DFT_MAX_PRODUCT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _name(resolve_transformer(huge, "dft", huge)) == "TransformerDFT"
        assert _name(resolve_transformer(1, "nufft", 1)) == "TransformerNUFFT"


def test_unknown_transformer_rejected():
    with pytest.raises(ValueError, match="unknown transformer"):
        resolve_transformer(10, "fft")


def _expected_fallback() -> str:
    """What `auto` should reach for once the DFT is too expensive.

    Three tiers, in order: nufftax (JAX-native), pynufft (pure NumPy), and
    finally the DFT with a warning, because a slow answer beats none.
    """
    if fitting.jax_available():
        return "TransformerNUFFT"
    if fitting.pynufft_available():
        return "TransformerPyNUFFT"
    return "TransformerDFT"


def test_many_visibilities_leaves_the_dft():
    """The pre-existing n_vis limit still applies on its own."""
    n_vis = DFT_MAX_VIS + 1
    expected = _expected_fallback()
    warns = expected == "TransformerDFT"
    with pytest.warns(UserWarning) if warns else _null():
        cls = resolve_transformer(n_vis, n_image_pixels=1)
    assert _name(cls) == expected


def test_big_image_leaves_the_dft_even_with_few_visibilities():
    """The regression: n_vis alone would have said 'DFT, fine'."""
    n_vis = 5158
    n_pix = 384_400
    assert n_vis <= DFT_MAX_VIS                     # the old test passed
    assert n_vis * n_pix > DFT_MAX_PRODUCT          # the new one does not

    expected = _expected_fallback()
    if expected != "TransformerDFT":
        cls = resolve_transformer(n_vis, n_image_pixels=n_pix)
        assert _name(cls) == expected
    else:
        with pytest.warns(UserWarning) as record:
            cls = resolve_transformer(n_vis, n_image_pixels=n_pix)
        assert _name(cls) == "TransformerDFT"
        message = str(record[0].message)
        # the warning has to name the field of view as the thing to change,
        # otherwise the user has no idea what to do about it
        assert "fov" in message
        assert "GB" in message
        # ...and the one-line fix that needs no JAX
        assert "pynufft" in message


def test_pynufft_is_preferred_over_the_dft_when_jax_is_missing(monkeypatch):
    """The tier that matters on a machine without JAX wheels: 164k
    visibilities over a 116x116 image is 16.5 GB of DFT temporary and 20 ms of
    NUFFT, so falling back to the DFT there is not a fallback, it is a
    failure."""
    if not fitting.pynufft_available():
        pytest.skip("pynufft is not installed")
    monkeypatch.setattr(fitting, "jax_available", lambda: False)
    cls = resolve_transformer(164_262, n_image_pixels=116 * 116)
    assert _name(cls) == "TransformerPyNUFFT"


def test_the_dft_is_still_the_last_resort(monkeypatch):
    monkeypatch.setattr(fitting, "jax_available", lambda: False)
    monkeypatch.setattr(fitting, "pynufft_available", lambda: False)
    with pytest.warns(UserWarning, match="pynufft"):
        cls = resolve_transformer(164_262, n_image_pixels=116 * 116)
    assert _name(cls) == "TransformerDFT"


def test_geometry_pixel_count_reaches_the_chooser(monkeypatch):
    """make_dataset must pass the image size through, not just len(data)."""
    import numpy as np

    from pyuvimage.grids import resolve_geometry

    seen = {}
    real = fitting.resolve_transformer

    def spy(n_vis, transformer="auto", n_image_pixels=None):
        seen["n_vis"] = n_vis
        seen["n_image_pixels"] = n_image_pixels
        return real(n_vis, "dft", n_image_pixels)

    monkeypatch.setattr(fitting, "resolve_transformer", spy)

    n = 64
    geometry = resolve_geometry(fov_arcsec=1.0, max_baseline_wavelengths=2e5)
    rng = np.random.default_rng(0)
    uv = rng.normal(0, 1e5, (n, 2))
    data = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    noise = np.full(n, 1 + 1j)
    fitting.make_dataset(uv, data, noise, geometry)

    assert seen["n_vis"] == n
    assert seen["n_image_pixels"] == int(np.prod(geometry.shape_native))
    assert seen["n_image_pixels"] > 1


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_the_nyquist_case_that_oomed_is_now_caught():
    """J0116 at --pixel-scale nyquist: 5,158 visibilities on a 168x168 image.

    1.46e8 elements, which the original 2e8 threshold passed as affordable.
    It was OOM-killed in a 7 GB container -- autoarray holds several of those
    1.17 GB temporaries at once. The threshold was measured from a single
    array's size rather than the peak.
    """
    n_vis, n_pix = 5158, 168 * 168
    assert n_vis * n_pix > DFT_MAX_PRODUCT, (
        "the nyquist case must now exceed the threshold"
    )
    expected = _expected_fallback()
    if expected == "TransformerDFT":
        with pytest.warns(UserWarning):
            cls = resolve_transformer(n_vis, n_image_pixels=n_pix)
    else:
        cls = resolve_transformer(n_vis, n_image_pixels=n_pix)
    assert _name(cls) == expected


def test_the_ordinary_real_data_case_still_uses_the_dft():
    """J0116 at the auto pixel scale -- 5,158 visibilities on 100x100 -- ran
    fine on the DFT for the whole project. Lowering the threshold must not
    drag it onto a NUFFT it does not need."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _name(resolve_transformer(5158, n_image_pixels=100 * 100)) == (
            "TransformerDFT"
        )

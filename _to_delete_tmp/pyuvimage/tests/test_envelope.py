"""The Gaussian envelope source prior (see pyuvimage/envelope.py)."""

import numpy as np
import pytest

from pyuvimage.envelope import (
    SIGMA_TO_FWHM,
    GaussianEnvelopeMatern,
    estimate_envelope,
    peak_offset_arcsec,
)


def _blob(shape, row, col, sigma_pix, amp=1.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    return amp * np.exp(-0.5 * (((yy - row) ** 2 + (xx - col) ** 2) / sigma_pix**2))


def test_peak_offset_conventions():
    """Native row 0 is +y (North); column 0 is -x. An asymmetric position
    catches any transposition or sign flip."""
    pix = 0.1
    img = _blob((61, 61), row=20, col=40, sigma_pix=2.0)
    y, x = peak_offset_arcsec(img, pix)
    # row 20 is 10 rows above centre (30) -> +1.0"; col 40 is 10 right -> +1.0"
    assert y == pytest.approx(+1.0)
    assert x == pytest.approx(+1.0)

    img2 = _blob((61, 61), row=45, col=15, sigma_pix=2.0)
    y2, x2 = peak_offset_arcsec(img2, pix)
    assert y2 == pytest.approx(-1.5)
    assert x2 == pytest.approx(-1.5)


def test_envelope_weights_shape():
    reg = GaussianEnvelopeMatern(
        coefficient=1.0, scale=0.2, envelope_fwhm=1.0, envelope_floor=1e-2,
        centre=(0.5, -0.25),
    )
    pts = np.array([[0.5, -0.25], [0.5 + 0.5, -0.25], [8.0, 8.0]])
    w = reg.envelope_weights(pts)
    assert w[0] == pytest.approx(1.0)          # peak at the centre
    # at half the FWHM from the centre the Gaussian part is at half height
    assert w[1] == pytest.approx(1e-2 + 0.99 * 0.5, rel=1e-3)
    assert w[2] == pytest.approx(1e-2, abs=1e-6)  # far away -> the floor


def test_envelope_rejects_bad_parameters():
    with pytest.raises(ValueError):
        GaussianEnvelopeMatern(envelope_floor=0.0)
    with pytest.raises(ValueError):
        GaussianEnvelopeMatern(envelope_fwhm=-1.0)


def test_estimate_envelope_finds_offset_source():
    pix = 0.05
    beam = 0.2
    img = _blob((121, 121), row=40, col=80, sigma_pix=4.0, amp=1.0)
    centre, fwhm = estimate_envelope(
        img, pixel_scale=pix, rms=1e-3, beam_fwhm=beam
    )
    assert centre[0] == pytest.approx(+1.0, abs=pix)   # 20 rows above centre
    assert centre[1] == pytest.approx(+1.0, abs=pix)
    # generous but not absurd: at least a few beams, of order the blob size
    assert fwhm >= 3 * beam
    assert fwhm == pytest.approx(4.0 * pix * SIGMA_TO_FWHM, rel=0.5)


def test_estimate_envelope_floors_on_faint_data():
    """With nothing significant in the map the envelope falls back to a
    minimum size rather than collapsing onto a noise spike."""
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1e-3, (61, 61))
    _, fwhm = estimate_envelope(img, pixel_scale=0.05, rms=1e-3, beam_fwhm=0.2)
    assert fwhm >= 3 * 0.2


def test_adaptive_weights_track_brightness():
    from pyuvimage.envelope import AdaptiveMatern

    brightness = np.array([1.0, 0.5, 0.0, 0.25])
    reg = AdaptiveMatern(
        coefficient=1.0, scale=0.2, brightness=brightness, floor=0.01, power=1.0
    )
    w = reg.adaptive_weights()
    assert w[0] == pytest.approx(1.0)      # brightest -> widest prior
    assert w[2] == pytest.approx(0.01)     # blank -> the floor
    assert w[1] > w[3] > w[2]              # monotonic in brightness


def test_adaptive_power_zero_is_uniform():
    from pyuvimage.envelope import AdaptiveMatern

    reg = AdaptiveMatern(
        coefficient=1.0, scale=0.2, brightness=np.array([1.0, 0.1, 0.5]),
        floor=0.01, power=0.0,
    )
    assert np.allclose(reg.adaptive_weights(), 1.0)


def test_adaptive_requires_brightness():
    from pyuvimage.envelope import AdaptiveMatern

    with pytest.raises(ValueError):
        AdaptiveMatern(coefficient=1.0, scale=0.2)


def test_covariance_cache_reuses_across_coefficients():
    """The kernel covariance depends on the mesh/scale, not the coefficient --
    caching it is what makes the hyperparameter search affordable."""
    from pyuvimage.envelope import (
        cached_inverse_covariance,
        clear_covariance_cache,
    )

    clear_covariance_cache()
    pts = np.stack(np.mgrid[0:12, 0:12].reshape(2, -1).astype(float), axis=1) * 0.1
    a = cached_inverse_covariance(pts, scale=0.3, nu=1.5)
    b = cached_inverse_covariance(pts, scale=0.3, nu=1.5)
    assert a is b                      # same object -> served from the cache
    c = cached_inverse_covariance(pts, scale=0.6, nu=1.5)
    assert c is not a                  # a different scale must recompute
    clear_covariance_cache()

import json

import numpy as np
import pytest

from pyuvimage.uvdata import V_SIGN, C_M_S, UVData


def _example(n_chan=2, n_vis=50, seed=0):
    rng = np.random.default_rng(seed)
    return UVData(
        uvw=rng.normal(0, 100.0, (n_vis, 3)),
        frequencies=np.linspace(100e9, 101e9, n_chan),
        data=rng.normal(size=(n_chan, n_vis)) + 1j * rng.normal(size=(n_chan, n_vis)),
        noise=np.full((n_chan, n_vis), 0.1 + 0.1j),
        meta={"telescope": "test", "dish_diameter_m": 12.0},
    )


def test_roundtrip_directory(tmp_path):
    uvd = _example()
    uvd.write(tmp_path / "ds", overwrite=True)
    back = UVData.read(tmp_path / "ds")
    np.testing.assert_allclose(back.uvw, uvd.uvw)
    np.testing.assert_allclose(back.data, uvd.data)
    np.testing.assert_allclose(back.noise, uvd.noise)
    assert back.meta["telescope"] == "test"


def test_roundtrip_npz(tmp_path):
    uvd = _example()
    np.savez(
        tmp_path / "x.npz",
        uvw=uvd.uvw, frequencies=uvd.frequencies,
        data_re=uvd.data.real, data_im=uvd.data.imag,
        noise_re=uvd.noise.real, noise_im=uvd.noise.imag,
        flags=np.zeros(uvd.data.shape, dtype=np.uint8),
        meta=json.dumps(uvd.meta),
    )
    back = UVData.read(tmp_path / "x.npz")
    np.testing.assert_allclose(back.data, uvd.data)
    assert back.flags is None


def test_uv_wavelengths_scaling():
    """u scales; v scales *and changes sign*.

    The stored `uvw` is the measurement set's, and its v has the opposite
    sign to the one the imaging grid wants -- measured against CASA, see
    `uvdata.V_SIGN`. Negating it in the accessor is what puts every image,
    the FITS WCS and the beam position angle into the true sky frame."""
    uvd = _example()
    uv0 = uvd.uv_wavelengths(0)
    scaled = uvd.uvw[:, :2] * uvd.frequencies[0] / C_M_S
    np.testing.assert_allclose(uv0[:, 0], scaled[:, 0])
    np.testing.assert_allclose(uv0[:, 1], -scaled[:, 1])
    assert V_SIGN == -1.0


def test_flatten_respects_flags():
    uvd = _example()
    flags = np.zeros(uvd.data.shape, dtype=bool)
    flags[0, :10] = True
    uvd.flags = flags
    uv, d, n = uvd.flattened()
    assert len(d) == uvd.n_chan * uvd.n_vis - 10


def test_validate_rejects_bad_noise():
    uvd = _example()
    uvd.noise[0, 0] = 0.0
    with pytest.raises(ValueError):
        uvd.validate()


def _legacy_arrays(hands, n_vis=20, seed=1):
    """Build legacy (n_corr, n_chan, n_vis, 2) arrays from complex hands."""
    n_corr = len(hands)
    legacy_vis = np.zeros((n_corr, 1, n_vis, 2))
    for i, h in enumerate(hands):
        legacy_vis[i, 0, :, 0] = h.real
        legacy_vis[i, 0, :, 1] = h.imag
    sigma = np.full((n_corr, 1, n_vis, 2), 0.1)
    uv = np.random.default_rng(seed).normal(0, 1e5, (1, n_vis, 2))
    return legacy_vis, sigma, uv


def test_legacy_conversion_forms_stokes_i():
    n_vis = 20
    rng = np.random.default_rng(1)
    xx = rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
    yy = rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
    legacy_vis, sigma, uv = _legacy_arrays([xx, yy], n_vis)
    uvd = UVData.from_legacy(legacy_vis, sigma, uv, [230e9])
    np.testing.assert_allclose(uvd.data[0], 0.5 * (xx + yy))
    # two *independent* hands: sigma_I = sigma / sqrt(2)
    np.testing.assert_allclose(uvd.noise.real, 0.1 / np.sqrt(2))
    # legacy files stored the measurement set's own (u, v), so the accessor
    # applies the same v sign it applies to everything else
    got = uvd.uv_wavelengths(0)
    np.testing.assert_allclose(got[:, 0], uv[0][:, 0])
    np.testing.assert_allclose(got[:, 1], V_SIGN * uv[0][:, 1])


def test_legacy_duplicated_hands_do_not_reduce_noise():
    """Regression: some exports duplicate a single correlation.  Averaging
    identical hands cannot beat down the noise, so sigma must not be divided
    by sqrt(n_corr) -- doing so underestimates the noise and makes the fit
    overfit (absorbing dirty-beam sidelobes into the model)."""
    n_vis = 20
    rng = np.random.default_rng(2)
    hand = rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
    legacy_vis, sigma, uv = _legacy_arrays([hand, hand], n_vis)
    with pytest.warns(UserWarning, match="duplicated hands"):
        uvd = UVData.from_legacy(legacy_vis, sigma, uv, [230e9])
    np.testing.assert_allclose(uvd.data[0], hand)
    np.testing.assert_allclose(uvd.noise.real, 0.1)

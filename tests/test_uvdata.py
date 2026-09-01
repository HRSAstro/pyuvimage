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


# ------------------------------------------------------------- select / meta

def _with_ingredients(n_chan=3, n_vis=40, seed=0):
    rng = np.random.default_rng(seed)
    return UVData(
        uvw=rng.normal(0, 100.0, (n_vis, 3)),
        frequencies=np.linspace(100e9, 101e9, n_chan),
        data=rng.normal(size=(n_chan, n_vis)) + 1j * rng.normal(size=(n_chan, n_vis)),
        noise=np.full((n_chan, n_vis), 0.1 + 0.1j),
        flags=rng.random((n_chan, n_vis)) < 0.1,
        meta={"telescope": "test"},
        antenna1=rng.integers(0, 5, n_vis),
        antenna2=rng.integers(5, 10, n_vis),
        time=np.arange(n_vis, dtype=float),
        weight_sigma=rng.random((n_chan, n_vis)) * (1 + 1j),
    )


def test_select_keeps_the_noise_reestimation_ingredients():
    """A channel pulled out of a cube lost antenna1/antenna2/time and
    weight_sigma, so `recompute_noise` refused it as an old export."""
    uvd = _with_ingredients()
    one = uvd.select(channel=1)
    assert one.n_chan == 1 and one.can_reestimate_noise
    np.testing.assert_array_equal(one.antenna1, uvd.antenna1)
    np.testing.assert_array_equal(one.antenna2, uvd.antenna2)
    np.testing.assert_array_equal(one.time, uvd.time)
    # per-channel arrays are sliced along the channel axis, per-row ones shared
    np.testing.assert_array_equal(one.weight_sigma, uvd.weight_sigma[1:2])
    np.testing.assert_array_equal(one.flags, uvd.flags[1:2])
    np.testing.assert_array_equal(one.frequencies, uvd.frequencies[1:2])
    assert one.meta == uvd.meta and one.meta is not uvd.meta
    # and one without them is still fine
    bare = _example().select(channel=0)
    assert bare.weight_sigma is None and not bare.can_reestimate_noise


def test_recompute_noise_on_multi_spw_records_the_mode_at_the_top_level():
    """`fit_parameters.json` reads the top-level meta, which kept the *old*
    mode while every window's meta carried the new one."""
    from pyuvimage.uvdata import MultiSpwUVData, recompute_noise

    def spw(seed):
        rng = np.random.default_rng(seed)
        n_ant, n_time = 8, 10
        a1, a2 = np.triu_indices(n_ant, k=1)
        ant1, ant2 = np.tile(a1, n_time), np.tile(a2, n_time)
        time = np.repeat(np.arange(n_time) * 6.0, a1.size)
        n = ant1.size
        return UVData(
            uvw=rng.normal(0, 100.0, (n, 3)),
            frequencies=np.array([100e9 + seed * 1e9]),
            data=rng.normal(0, 0.01, (1, n)) + 1j * rng.normal(0, 0.01, (1, n)),
            noise=np.full((1, n), 0.5 + 0.5j),
            meta={"noise_estimate": "difference", "noise_chunk_seconds": 600.0},
            antenna1=ant1, antenna2=ant2, time=time,
            weight_sigma=np.full((1, n), 3.0 + 3.0j),
        )

    multi = MultiSpwUVData(spws=[spw(1), spw(2)])
    assert multi.meta["noise_estimate"] == "difference"

    scaled = recompute_noise(multi, "scaled")
    assert scaled.meta["noise_estimate"] == "scaled"
    assert scaled.meta["noise_chunk_seconds"] is None
    assert all(s.meta["noise_estimate"] == "scaled" for s in scaled.spws)

    chunked = recompute_noise(multi, "difference", chunk_seconds=30.0)
    assert chunked.meta["noise_estimate"] == "difference"
    assert chunked.meta["noise_chunk_seconds"] == 30.0
    assert all(s.meta["noise_chunk_seconds"] == 30.0 for s in chunked.spws)
    # the single-spw path agrees with itself, and leaves no stale chunk width
    one = recompute_noise(spw(3), "hybrid")
    assert one.meta["noise_estimate"] == "hybrid"
    assert one.meta["noise_chunk_seconds"] is None


# ------------------------------------------------------ baseline percentiles

def test_multi_spw_baseline_percentile_uses_each_windows_own_frequency():
    """Every window's lengths in wavelengths at *its own* maximum frequency,
    as `max_baseline_wavelengths` already did. Scaling all of them by the
    global maximum put a 100 GHz window's baselines where a 200 GHz window's
    are, and sized the mesh off baselines that were never observed."""
    from pyuvimage.uvdata import MultiSpwUVData

    rng = np.random.default_rng(0)
    uvw = rng.normal(0, 100.0, (60, 3))

    def spw(freq_hz):
        return UVData(
            uvw=uvw, frequencies=np.array([freq_hz]),
            data=np.zeros((1, 60), dtype=complex), noise=np.ones((1, 60)) * (1 + 1j),
        )

    lo, hi = spw(100e9), spw(200e9)
    multi = MultiSpwUVData(spws=[lo, hi])
    got = multi.baseline_percentile_wavelengths(95.0)
    per_spw = np.concatenate([
        np.hypot(uvw[:, 0], uvw[:, 1]) * (f / C_M_S) for f in (100e9, 200e9)
    ])
    assert got == pytest.approx(np.percentile(per_spw, 95.0))
    # the old answer scaled everything by 200 GHz and came out higher
    wrong = np.percentile(
        np.concatenate([np.hypot(uvw[:, 0], uvw[:, 1]) * (200e9 / C_M_S)] * 2), 95.0
    )
    assert got < wrong
    # consistent with the maximum, which was already per-spw
    assert multi.max_baseline_wavelengths == pytest.approx(
        max(lo.max_baseline_wavelengths, hi.max_baseline_wavelengths)
    )
    assert multi.baseline_percentile_wavelengths(100.0) == pytest.approx(
        multi.max_baseline_wavelengths
    )


def test_fractional_bandwidth_is_one_definition_for_both_layouts():
    from pyuvimage.uvdata import MultiSpwUVData

    a = _example(n_chan=2)
    a.frequencies = np.array([100e9, 101e9])
    b = _example(n_chan=2)
    b.frequencies = np.array([110e9, 112e9])
    multi = MultiSpwUVData(spws=[a, b])
    assert multi.fractional_bandwidth == pytest.approx((112e9 - 100e9) / 106e9)
    assert a.fractional_bandwidth == pytest.approx(1e9 / 100.5e9)
    assert _example(n_chan=1).fractional_bandwidth == 0.0


# ------------------------------------------------------------- flattened

def test_flattened_is_the_per_channel_loop_bit_for_bit():
    """The order is channel-major and relied on downstream; the values are
    each one multiplication, so nothing may move even in the last bit."""
    uvd = _with_ingredients(n_chan=4, n_vis=50)
    uv_l, d_l, n_l = [], [], []
    for c in range(uvd.n_chan):
        keep = ~uvd.flags[c]
        uv_l.append(uvd.uv_wavelengths(c)[keep])
        d_l.append(uvd.data[c][keep])
        n_l.append(uvd.noise[c][keep])
    want = (np.concatenate(uv_l), np.concatenate(d_l), np.concatenate(n_l))
    for x, y in zip(uvd.flattened(), want):
        assert np.array_equal(x, y)
    # without flags the same, and nothing is a view into the dataset
    uvd.flags = None
    uv, d, n = uvd.flattened()
    assert d.shape == (uvd.n_chan * uvd.n_vis,)
    assert np.array_equal(uv[:uvd.n_vis], uvd.uv_wavelengths(0))
    assert not np.shares_memory(d, uvd.data) and not np.shares_memory(n, uvd.noise)


# ------------------------------------------------------- shift_image_centre

def test_shift_image_centre_is_the_per_channel_ramp():
    """One broadcast phase for all channels must equal the channel loop."""
    from pyuvimage.uvdata import ARCSEC_RAD, pooled_noise, shift_image_centre

    uvd = _with_ingredients(n_chan=4, n_vis=50)
    y0, x0 = 1.3, -0.7
    out = shift_image_centre(uvd, (y0, x0))
    for c in range(uvd.n_chan):
        uv = uvd.uv_wavelengths(c)
        phase = np.exp(2j * np.pi * (uv[:, 0] * x0 + uv[:, 1] * y0) * ARCSEC_RAD)
        np.testing.assert_allclose(out.data[c], uvd.data[c] * phase, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(out.noise, pooled_noise(uvd.noise), rtol=1e-12)
    # the input is untouched and the ingredients travel
    assert not np.shares_memory(out.data, uvd.data)
    assert out.can_reestimate_noise and out.weight_sigma is not None


def test_a_zero_shift_is_a_no_op_for_multi_spw_too():
    """It recursed first and pooled every window's noise for a shift of
    nothing, which changes the noise map -- and with it chi^2 -- on a run that
    asked for the phase centre."""
    from pyuvimage.uvdata import MultiSpwUVData, shift_image_centre

    a, b = _with_ingredients(seed=1), _with_ingredients(seed=2)
    a.noise = a.noise * 1.0 + 0.05j              # lopsided, so pooling would show
    multi = MultiSpwUVData(spws=[a, b])
    assert shift_image_centre(multi, (0.0, 0.0)) is multi
    moved = shift_image_centre(multi, (0.5, 0.0))
    assert moved is not multi and moved.meta["image_centre_offset_arcsec"] == [0.5, 0.0]


def test_the_asymmetry_report_is_the_same_number_as_reim_asymmetry(caplog):
    """`_report_reim_asymmetry` re-implemented `reim_asymmetry`; now it calls
    it, and on a large map it reads a stride of the cells. The reported number
    is identical below the stride threshold and within scatter above it."""
    import logging

    from pyuvimage import uvdata as U

    rng = np.random.default_rng(0)
    small = (1.0 + 0.1 * rng.random(5000)) + 1j * (1.0 + 0.1 * rng.random(5000))
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        U._report_reim_asymmetry(small)
    reported = float(caplog.records[-1].getMessage().split("differ by ")[1].split("%")[0])
    assert reported == pytest.approx(100 * U.reim_asymmetry(small), abs=0.05)

    caplog.clear()
    n = 3 * U.REIM_DIAGNOSTIC_SAMPLES
    big = (1.0 + 0.1 * rng.random(n)) + 1j * (1.0 + 0.1 * rng.random(n))
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        U._report_reim_asymmetry(big)
    reported = float(caplog.records[-1].getMessage().split("differ by ")[1].split("%")[0])
    assert reported == pytest.approx(100 * U.reim_asymmetry(big), abs=0.2)

    # nothing usable: silent, as before, rather than "0.0%"
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        U._report_reim_asymmetry(np.zeros(10, dtype=complex))
    assert not caplog.records
    assert U.reim_asymmetry(np.zeros(10, dtype=complex)) == 0.0


def test_read_dataset_opens_an_npz_once(tmp_path, monkeypatch):
    """It peeked at the keys, closed the file, and reopened it to read."""
    import json

    from pyuvimage import uvdata as U

    uvd = _example()
    np.savez(
        tmp_path / "x.npz",
        uvw=uvd.uvw, frequencies=uvd.frequencies,
        data_re=uvd.data.real, data_im=uvd.data.imag,
        noise_re=uvd.noise.real, noise_im=uvd.noise.imag,
        meta=json.dumps(uvd.meta),
    )
    opens = []
    real_load = np.load

    def counting(*a, **k):
        opens.append(a[0])
        return real_load(*a, **k)

    monkeypatch.setattr(U.np, "load", counting)
    back = U.read_dataset(tmp_path / "x.npz")
    assert len(opens) == 1
    np.testing.assert_allclose(back.data, uvd.data)
    # the multi-spw layout too
    payload = {"n_spw": np.array(1), "meta": json.dumps({"per_spw_meta": [uvd.meta]})}
    for k in ("uvw", "frequencies"):
        payload["spw000_" + k] = getattr(uvd, k)
    payload.update(spw000_data_re=uvd.data.real, spw000_data_im=uvd.data.imag,
                   spw000_noise_re=uvd.noise.real, spw000_noise_im=uvd.noise.imag)
    np.savez(tmp_path / "m.npz", **payload)
    opens.clear()
    back = U.read_dataset(tmp_path / "m.npz")
    assert len(opens) == 1 and back.n_spw == 1

"""Several spectral windows, imaged together by MFS.

Spectral windows are ragged: different channel counts *and* different row
counts, which is why they are a list of `UVData` rather than another axis on
one array.
"""

import json

import numpy as np
import pytest
from astropy.io import fits

import pyuvimage
from pyuvimage import mock
from pyuvimage.uvdata import C_M_S, MultiSpwUVData, UVData, read_dataset


@pytest.fixture(scope="module")
def multi():
    return mock.make_multi_spw_dataset(n_vis=150, mesh_n=16)


def test_the_mock_is_genuinely_ragged(multi):
    """If every spw had the same shape the test would prove nothing."""
    m, _, _ = multi
    assert m.n_spw == 3
    assert len({s.n_chan for s in m.spws}) > 1, "channel counts are all equal"
    assert len({s.n_vis for s in m.spws}) > 1, "row counts are all equal"


def test_counts_and_frequencies(multi):
    m, _, _ = multi
    assert m.n_chan == sum(s.n_chan for s in m.spws)
    assert m.n_vis == sum(s.n_vis for s in m.spws)
    assert m.n_samples == sum(s.n_chan * s.n_vis for s in m.spws)
    f = m.frequencies
    assert np.all(np.diff(f) > 0), "frequencies must come back sorted"
    assert f.size == m.n_chan
    assert np.min(f) <= m.central_frequency <= np.max(f)


def test_flattened_carries_every_sample_at_its_own_frequency(multi):
    """Each sample's uv must be computed at its own channel frequency.

    This is the whole reason MFS across spws is exact rather than an
    approximation: nothing is averaged, so there is no bandwidth smearing
    introduced by the combination itself.
    """
    m, _, _ = multi
    uv, d, n = m.flattened()
    assert uv.shape == (m.n_samples, 2)
    assert d.shape == n.shape == (m.n_samples,)

    # the longest baseline must correspond to the highest frequency present
    expected = max(
        float(np.max(np.hypot(*(s.uvw[:, :2] * (f / C_M_S)).T)))
        for s in m.spws for f in s.frequencies
    )
    assert np.max(np.hypot(uv[:, 0], uv[:, 1])) == pytest.approx(expected)


def test_splitting_a_dataset_into_spws_changes_nothing():
    """The strongest check available: same samples, two containers, one answer.

    A single multi-channel dataset and the same data split across three
    spectral windows must give a bit-identical image -- if they differ, the
    multi-spw path is doing something to the data that the single-spw path
    is not.
    """
    uvw = mock.random_uv_coverage(
        300, (16 / (2 * 3.0 / 206265.0)) * C_M_S / 230e9, 230e9, seed=3)
    from pyuvimage.grids import resolve_geometry

    geom = resolve_geometry(
        3.0,
        max_baseline_wavelengths=float(
            np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * 230e9 / C_M_S),
        mesh_shape=(16, 16))
    truth = mock.exponential_image(
        geom.mesh_shape, geom.mesh_pixel_scale, flux_jy=0.05, r_eff_arcsec=0.4)
    freqs = 230e9 + 2e8 * np.arange(6)
    one = mock.simulate(
        truth, geom.mesh_pixel_scale, uvw, freqs, sigma_jy=3e-4, seed=9,
        meta={"telescope": "mock", "dish_diameter_m": 12.0,
              "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0})
    split = mock.split_into_spws(one, n_spw=3)
    assert split.n_spw == 3 and split.n_samples == one.n_samples

    a, b = one.flattened(), split.flattened()
    for x, y in zip(a, b):
        assert np.array_equal(x, y)

    ra = pyuvimage.run(one, fov=3.0, mesh_shape=(16, 16),
                       uncertainty_map=False, write=False)
    rb = pyuvimage.run(split, fov=3.0, mesh_shape=(16, 16),
                       uncertainty_map=False, write=False)
    pa, pb = ra.products[0], rb.products[0]
    assert pa.chi_squared == pytest.approx(pb.chi_squared, rel=1e-12)
    assert np.array_equal(np.nan_to_num(pa.model_image),
                          np.nan_to_num(pb.model_image))


def test_round_trip_on_disk(multi, tmp_path):
    m, _, _ = multi
    m.write(tmp_path / "ds")
    assert sorted(p.name for p in (tmp_path / "ds").iterdir()) == [
        "meta.json", "spw000", "spw001", "spw002"]
    back = read_dataset(tmp_path / "ds")
    assert isinstance(back, MultiSpwUVData)
    assert back.n_spw == m.n_spw and back.n_samples == m.n_samples
    for x, y in zip(m.flattened(), back.flattened()):
        assert np.array_equal(x, y)


def test_single_spw_datasets_still_read_as_before(multi, tmp_path):
    """Datasets written before multi-spw support must keep working."""
    m, _, _ = multi
    m.spws[0].write(tmp_path / "one")
    back = read_dataset(tmp_path / "one")
    assert isinstance(back, UVData)
    assert back.n_spw == 1 and back.spws == [back]


def test_mfs_run_and_parameter_record(multi, tmp_path):
    m, _, _ = multi
    res = pyuvimage.run(m, fov=3.0, mesh_shape=(16, 16), out=tmp_path / "o",
                        uncertainty_map=False)
    data = res.parameters["data"]
    assert data["n_spw"] == 3
    assert data["n_samples"] == m.n_samples
    assert data["fractional_bandwidth"] > 0
    assert res.parameters["fit_quality"]["n_data"] == 2 * m.n_samples
    assert (tmp_path / "o" / "model.fits").exists()


def test_cube_channels_are_ordered_by_frequency_across_spws(multi, tmp_path):
    """A cube spanning spws still needs a monotonic frequency axis."""
    m, _, _ = multi
    order = m._channel_index()
    freqs = [m.spws[i].frequencies[c] for i, c in order]
    assert np.all(np.diff(freqs) > 0)
    # select() must return the right single channel
    first = m.select(channel=0)
    assert float(first.frequencies[0]) == pytest.approx(min(freqs))


def test_irregular_cube_frequency_axis_is_not_silently_linear(tmp_path):
    """Channels from disjoint spws are not evenly spaced.

    CRVAL3 + n*CDELT3 would put later planes at frequencies that were never
    observed, so the true values have to be recorded and the mismatch stated.
    """
    m, _, _ = mock.make_multi_spw_dataset(
        n_vis=80, mesh_n=12, spw_frequencies_hz=(230e9, 236e9),
        n_chan_per_spw=(2, 2))
    pyuvimage.run(m, fov=3.0, mesh_shape=(12, 12), mode="cube",
                  out=tmp_path / "c", uncertainty_map=False)
    h = fits.getheader(tmp_path / "c" / "model.fits")
    assert h["FREQIRR"] is True
    got = [h[f"FRQ{i:04d}"] for i in range(4)]
    assert got == pytest.approx([230e9, 230.2e9, 236e9, 236.2e9])
    side = json.loads((tmp_path / "c" / "frequencies.json").read_text())
    assert side["evenly_spaced"] is False
    assert side["frequencies_hz"] == pytest.approx(got)


def test_wide_band_warns_that_mfs_assumes_a_flat_spectrum(caplog, tmp_path):
    """MFS fits one frequency-independent image; a wide band must say so."""
    import logging

    m, _, _ = mock.make_multi_spw_dataset(
        n_vis=80, mesh_n=12, spw_frequencies_hz=(100e9, 200e9),
        n_chan_per_spw=(1, 1))
    assert m.fractional_bandwidth > 0.5
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        pyuvimage.run(m, fov=3.0, mesh_shape=(12, 12), write=False,
                      uncertainty_map=False)
    assert any("fractional bandwidth" in r.message for r in caplog.records)


def test_spw_argument_parsing():
    from pyuvimage.cli import _parse_spw

    assert _parse_spw("0") == 0
    assert _parse_spw("all") == "all"
    assert _parse_spw("0,2") == [0, 2]
    assert _parse_spw("0-3") == [0, 1, 2, 3]
    assert _parse_spw("0-1,4") == [0, 1, 4]


def test_multi_spw_npz_round_trip(multi, tmp_path):
    """The CASA export path: several windows side by side in one .npz.

    Windows are stored under `spw000_*` keys rather than stacked, because they
    are ragged. Built here exactly as `casa_export.export` writes it, so the
    reader is tested without needing CASA.
    """
    m, _, _ = multi
    payload = {"n_spw": np.array(m.n_spw)}
    metas = []
    for i, s in enumerate(m.spws):
        pre = f"spw{i:03d}_"
        payload[pre + "uvw"] = s.uvw
        payload[pre + "frequencies"] = s.frequencies
        payload[pre + "data_re"] = s.data.real
        payload[pre + "data_im"] = s.data.imag
        payload[pre + "noise_re"] = s.noise.real
        payload[pre + "noise_im"] = s.noise.imag
        metas.append(dict(s.meta))
    meta = dict(m.meta)
    meta["per_spw_meta"] = metas
    payload["meta"] = json.dumps(meta)
    f = tmp_path / "export.npz"
    np.savez_compressed(f, **payload)

    back = read_dataset(f)
    assert isinstance(back, MultiSpwUVData)
    assert back.n_spw == m.n_spw
    for x, y in zip(m.flattened(), back.flattened()):
        assert np.array_equal(x, y)


def test_single_spw_npz_still_reads_as_uvdata(multi, tmp_path):
    m, _, _ = multi
    s = m.spws[0]
    f = tmp_path / "one.npz"
    np.savez_compressed(
        f, uvw=s.uvw, frequencies=s.frequencies,
        data_re=s.data.real, data_im=s.data.imag,
        noise_re=s.noise.real, noise_im=s.noise.imag,
        meta=json.dumps(dict(s.meta)))
    back = read_dataset(f)
    assert isinstance(back, UVData)
    assert back.n_chan == s.n_chan and back.n_vis == s.n_vis

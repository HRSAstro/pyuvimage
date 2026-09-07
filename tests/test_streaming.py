"""The streaming load: one pass over the data, nothing per visibility kept.

Everything the sparse fit consumes is a sum over visibilities -- the w-tilde
kernel, the dirty images, d^T N^-1 d, the noise normalisation -- so it can be
accumulated chunk by chunk from a file that is never read whole. These tests
pin that the streamed quantities are the in-memory ones (bitwise for the
readers, ~1e-15 for the sums) on a deliberately ragged multi-spw mock with
flags, and with a chunk small enough that channels are split by row.

The fit itself on the streamed terms needs autoarray's sparse operator,
which needs JAX; those tests skip without it and are the ones to run on a
machine that has it (`-k streamed_fit`).
"""

import json

import numpy as np
import pytest

import autogalaxy as ag

from pyuvimage import fitting, mock
from pyuvimage import streaming as stm
from pyuvimage.beam import DirtyImager
from pyuvimage.uvdata import with_pooled_noise


CHUNK = 300     # below every window's row count: splits rows within a channel


@pytest.fixture(scope="module")
def ragged():
    uvd, truth, geom = mock.make_multi_spw_dataset(
        n_vis=500, mesh_n=10, fov_arcsec=3.0, seed=4)
    rng = np.random.default_rng(1)
    for spw in uvd.spws:
        spw.flags = rng.random(spw.data.shape) < 0.05
        spw.flags[:, 3] = True                  # a row flagged in every channel
    # sigma_re != sigma_im, so pooling has something to do
    for spw in uvd.spws:
        spw.noise = spw.noise.real + 1j * spw.noise.real * 1.1
    return uvd, truth, geom


@pytest.fixture(scope="module")
def flat(ragged):
    """The in-memory reference, on *pooled* noise: the sparse path pools when
    sigma_re and sigma_im disagree, and the products only compare like with
    like once it has (both sides then weight by the one sigma)."""
    uvd, _, geom = ragged
    uv, d, n = with_pooled_noise(uvd).flattened()
    ds = fitting.make_dataset(uv, d, n, geom, ag.TransformerDFT, mask_shape="square")
    return uv, d, n, ds


@pytest.fixture(scope="module")
def npz_path(ragged, tmp_path_factory):
    uvd, _, _ = ragged
    p = tmp_path_factory.mktemp("stream") / "mock.npz"
    payload = {"n_spw": np.array(len(uvd.spws)), "meta": json.dumps({"dish_diameter_m": 12.0})}
    for i, spw in enumerate(uvd.spws):
        pre = f"spw{i:03d}_"
        payload.update({
            pre + "uvw": spw.uvw, pre + "frequencies": spw.frequencies,
            pre + "data_re": spw.data.real, pre + "data_im": spw.data.imag,
            pre + "noise_re": spw.noise.real, pre + "noise_im": spw.noise.imag,
            pre + "flags": spw.flags,
        })
    np.savez_compressed(p, **payload)      # deflated, as casa_export writes
    return p


@pytest.fixture(scope="module")
def fits_dir(ragged, tmp_path_factory):
    uvd, _, _ = ragged
    return uvd.write(tmp_path_factory.mktemp("stream") / "mockdir")


def _cat(chunks):
    cs = list(chunks)
    return (np.concatenate([c.uv for c in cs]), np.concatenate([c.data for c in cs]),
            np.concatenate([c.noise for c in cs]), len(cs))


# --- readers ------------------------------------------------------------------

@pytest.mark.parametrize("source_kind", ["memory", "npz", "fits_dir"])
def test_every_reader_reproduces_flattened_bit_for_bit(request, ragged, flat, source_kind):
    """Same samples, same order, same bits -- across window, channel and row
    boundaries -- from an in-memory object, a deflated .npz streamed through
    zipfile, and a memory-mapped FITS directory."""
    uvd, _, _ = ragged
    uv, d, n, _ = flat
    source = {"memory": uvd, "npz": request.getfixturevalue("npz_path") if source_kind == "npz" else None,
              "fits_dir": request.getfixturevalue("fits_dir") if source_kind == "fits_dir" else None}[source_kind]
    u2, d2, n2, k = _cat(stm.iter_chunks(source, CHUNK))
    assert k > len(uvd.spws), "the chunking never split anything"
    uv0, d0, n0 = uvd.flattened()              # the readers stream the file as written
    assert np.array_equal(u2, uv0) and np.array_equal(d2, d0) and np.array_equal(n2, n0)
    assert np.array_equal(u2, uv) and np.array_equal(d2, d)


def test_the_npz_reader_never_materialises_a_member(npz_path, monkeypatch):
    """`np.load` on an .npz reads members whole; the reader must not call it."""
    import numpy

    def boom(*a, **k):
        raise AssertionError("np.load was called on the npz")
    monkeypatch.setattr(numpy, "load", boom)
    total = sum(len(c) for c in stm.iter_npz_chunks(npz_path, CHUNK))
    assert total > 0


# --- the accumulated terms -----------------------------------------------------

@pytest.fixture(scope="module")
def terms(ragged, flat):
    uvd, _, geom = ragged
    _, _, _, ds = flat
    return stm.accumulate_sparse_terms(
        stm.iter_uvdata_chunks(uvd, CHUNK), geom, ds.real_space_mask, ag.TransformerDFT,
        pool_noise=True)


def test_the_kernel_is_autoarrays_own_over_the_whole_dataset(ragged, flat, terms):
    from autoarray.inversion.inversion.interferometer import (
        inversion_interferometer_util as su)
    uv, d, n, ds = flat
    _, _, geom = ragged
    whole = su.nufft_precision_operator_from(
        noise_map_real=n.real, uv_wavelengths=uv,
        shape_masked_pixels_2d=tuple(geom.shape_native),
        grid_radians_2d=ds.real_space_mask.derive_grid.all_false.in_radians.native.array,
        chunk_k=2048)
    np.testing.assert_allclose(terms.kernel, whole, rtol=1e-12, atol=1e-12 * np.abs(whole).max())
    assert terms.n_vis == len(d)


def test_the_dirty_image_is_the_operators(ragged, flat, terms):
    """D = M^T dirty needs exactly `scaled_dirty_image`'s array."""
    _, _, geom = ragged
    _, _, _, ds = flat
    want = fitting.scaled_dirty_image(ds).reshape(geom.shape_native)
    np.testing.assert_allclose(terms.dirty_image, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_the_products_side_matches_dirty_imager(flat, terms):
    _, d, _, ds = flat
    im = DirtyImager(ds)
    np.testing.assert_allclose(terms.beam_raw / terms.sum_weights, im.dirty_beam, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(terms.data_dirty_raw / terms.sum_weights,
                               im.dirty_image(np.asarray(ds.data)), rtol=1e-12,
                               atol=1e-12 * np.abs(im.dirty_image(np.asarray(ds.data))).max())
    assert terms.sum_weights == pytest.approx(np.sum(im.weights), rel=1e-12)
    assert terms.rms == pytest.approx(im.rms, rel=1e-12)


def test_the_chi2_constants_match_the_linear_system(ragged, flat, terms):
    _, _, geom = ragged
    _, _, _, ds = flat
    system = fitting.build_linear_system(ds, geom.mesh_shape)
    assert terms.data_term == pytest.approx(system.data_term, rel=1e-12)
    assert terms.noise_normalization == pytest.approx(system.noise_normalization, rel=1e-12)


def test_the_kernel_images_a_model_without_visibilities(ragged, flat, terms):
    """The w-tilde identity the products rest on: the dirty image of the
    model's visibilities is W~ applied to the model image."""
    _, _, geom = ragged
    _, _, _, ds = flat
    model = np.zeros(geom.shape_native)
    model[9, 11] = 1.0
    model[5, 4] = 0.3
    vis = np.asarray(ds.transformer.visibilities_from(
        ag.Array2D(values=model, mask=ds.real_space_mask)))
    via_vis = DirtyImager(ds).dirty_image(vis)
    kimg = stm.KernelDirtyImager(terms, ds.real_space_mask)
    np.testing.assert_allclose(kimg.dirty_image_of_model(model), via_vis,
                               rtol=1e-10, atol=1e-12 * np.abs(via_vis).max())
    assert kimg.inside.all() and kimg.inside.shape == tuple(geom.shape_native)


def test_pooling_per_chunk_is_pooling_the_dataset(ragged, flat):
    uvd, _, geom = ragged
    _, _, _, ds = flat
    pooled = with_pooled_noise(uvd)
    a = stm.accumulate_sparse_terms(stm.iter_uvdata_chunks(uvd, CHUNK), geom,
                                    ds.real_space_mask, ag.TransformerDFT, pool_noise=True)
    b = stm.accumulate_sparse_terms(stm.iter_uvdata_chunks(pooled, CHUNK), geom,
                                    ds.real_space_mask, ag.TransformerDFT, pool_noise=False)
    np.testing.assert_allclose(a.kernel, b.kernel, rtol=1e-12, atol=1e-12 * np.abs(b.kernel).max())
    assert a.data_term == pytest.approx(b.data_term, rel=1e-12)
    assert a.noise_normalization == pytest.approx(b.noise_normalization, rel=1e-12)
    # the asymmetry was measured on the originals, not the pooled sigmas
    assert a.reim_asymmetry_sum > 0 and b.reim_asymmetry_sum == pytest.approx(0.0, abs=1e-9)


# --- the header ---------------------------------------------------------------

@pytest.mark.parametrize("source_kind", ["memory", "npz", "fits_dir"])
def test_the_header_reproduces_the_dataset_object(request, ragged, source_kind):
    uvd, _, _ = ragged
    source = {"memory": uvd, "npz": request.getfixturevalue("npz_path") if source_kind == "npz" else None,
              "fits_dir": request.getfixturevalue("fits_dir") if source_kind == "fits_dir" else None}[source_kind]
    h = stm.scan_header(source, CHUNK)
    assert h.n_spw == uvd.n_spw and h.n_vis == uvd.n_vis and h.n_chan == uvd.n_chan
    assert h.n_samples == uvd.n_samples
    assert np.array_equal(h.frequencies, uvd.frequencies)
    assert h.central_frequency == pytest.approx(uvd.central_frequency, rel=1e-12)
    assert h.fractional_bandwidth == pytest.approx(uvd.fractional_bandwidth, rel=1e-12)
    assert h.max_baseline_wavelengths == uvd.max_baseline_wavelengths
    for p in (50.0, 95.0):
        assert h.baseline_percentile_wavelengths(p) == uvd.baseline_percentile_wavelengths(p)
    assert len(h.noise) > 0


# --- cache --------------------------------------------------------------------

def test_terms_round_trip_and_the_cache_is_reused(ragged, flat, terms, tmp_path, caplog):
    import logging

    uvd, _, geom = ragged
    _, _, _, ds = flat
    key = stm.terms_key(uvd, geom)
    path = stm.terms_cache_path(tmp_path, key)
    terms.save(path, key)
    back, stored = stm.SparseTerms.load(path)
    assert stored == key
    for name in ("kernel", "dirty_image", "beam_raw", "data_dirty_raw", "row_lengths"):
        assert np.array_equal(getattr(back, name), getattr(terms, name))
    assert back.n_vis == terms.n_vis and back.data_term == terms.data_term
    assert back.baseline_percentile(95.0) == terms.baseline_percentile(95.0)

    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        again = stm.sparse_terms_for(uvd, geom, ds.real_space_mask, ag.TransformerDFT,
                                     cache_dir=tmp_path)
    assert "no data read this run" in caplog.text
    assert np.array_equal(again.kernel, terms.kernel)


def test_reload_streams_again_and_replaces_the_cache(ragged, flat, terms, tmp_path, caplog):
    """`--reload`: the cache is keyed on path, size and mtime, which a file
    rewritten in place can defeat, so there has to be a way to say "read it
    again". A poisoned cache entry must be replaced by the fresh terms."""
    import logging

    uvd, _, geom = ragged
    _, _, _, ds = flat
    key = stm.terms_key(uvd, geom, pool_noise=True)   # the fixture's terms are pooled
    path = stm.terms_cache_path(tmp_path, key)
    from dataclasses import replace

    poisoned = replace(terms, kernel=0.0 * terms.kernel)
    poisoned.save(path, key)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        fresh = stm.sparse_terms_for(uvd, geom, ds.real_space_mask, ag.TransformerDFT,
                                     cache_dir=tmp_path, pool_noise=True, reuse_cache=False,
                                     chunk_k=CHUNK)
    assert "no data read this run" not in caplog.text
    assert "streaming the visibilities once" in caplog.text
    assert np.array_equal(fresh.kernel, terms.kernel)
    back, _ = stm.SparseTerms.load(path)
    assert np.array_equal(back.kernel, terms.kernel), "the cache entry was replaced"


def test_the_key_tracks_geometry_pooling_and_source(ragged, npz_path):
    uvd, _, geom = ragged
    from dataclasses import replace
    k = stm.terms_key(uvd, geom)
    assert k != stm.terms_key(uvd, geom, pool_noise=True)
    assert k != stm.terms_key(uvd, replace(geom, pixel_scale=geom.pixel_scale * 1.01))
    assert k != stm.terms_key(npz_path, geom), "a file and an object are different sources"
    assert stm.terms_key(npz_path, geom) == stm.terms_key(npz_path, geom), "and stable"


# --- the fit on a stub: what the streaming path refuses without JAX --------------

def test_n_data_reads_the_terms_not_the_stub(terms):
    class Stub:
        data = np.zeros(8)
    setattr(Stub, fitting.STREAMED_TERMS_ATTR, terms)
    assert fitting.n_data_of(Stub()) == 2 * terms.n_vis != 16
    assert fitting.streamed_terms_of(Stub()) is terms
    assert fitting.streamed_terms_of(object()) is None


@pytest.mark.parametrize("kwargs, exc", [
    (dict(point_sources=True), NotImplementedError),
    (dict(inversion="dense"), ValueError),
    (dict(mode="slices"), ValueError),
    (dict(mode="cube", cube_prior="median"), ValueError),
])
def test_run_streamed_refuses_what_it_cannot_do_yet(ragged, kwargs, exc):
    from pyuvimage import api
    uvd, _, _ = ragged
    with pytest.raises(exc):
        api.run_streamed(uvd, fov=3.0, write=False, **kwargs)


# --- streaming="auto": the default, and when it holds back ---------------------

def test_auto_streams_a_file_on_the_mfs_sparse_path(npz_path, monkeypatch, caplog):
    """The default. A dataset on disk, MFS, sparse available, no points, no
    recentring: stream, and hand back the header so it is not scanned twice."""
    import logging
    from pyuvimage import api

    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    monkeypatch.setattr(fitting, "SPARSE_AUTO_MIN_VISIBILITIES", 1)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        stream, header = api.resolve_streaming("auto", str(npz_path))
    assert stream is True
    assert header is not None and header.n_samples > 0
    assert "streaming auto -> streamed" in caplog.text
    # cube mode and recentring stream too, now that both are wired
    for kwargs in (dict(mode="cube"), dict(image_centre="auto"), dict(image_centre=(1.0, 0.5))):
        assert api.resolve_streaming("auto", str(npz_path), **kwargs)[0] is True


@pytest.mark.parametrize("kwargs, why", [
    (dict(point_sources=True), "point components"),
    (dict(inversion="dense"), "dense"),
])
def test_auto_holds_the_data_where_streaming_is_not_supported(
    npz_path, monkeypatch, caplog, kwargs, why
):
    """Under `auto` an unsupported combination is a reason to run in memory,
    logged -- not a refusal. Refusing is what naming `--streaming` buys."""
    import logging
    from pyuvimage import api

    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        stream, header = api.resolve_streaming("auto", str(npz_path), **kwargs)
    assert stream is False and header is None
    assert "streaming auto -> in memory" in caplog.text
    assert why in caplog.text


def test_auto_holds_an_in_memory_dataset_and_a_small_file(ragged, npz_path, monkeypatch, caplog):
    import logging
    from pyuvimage import api

    uvd, _, _ = ragged
    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        assert api.resolve_streaming("auto", uvd) == (False, None)
        assert "already in memory" in caplog.text
        # the same visibility threshold `resolve_inversion` applies under auto
        monkeypatch.setattr(fitting, "SPARSE_AUTO_MIN_VISIBILITIES", 10**9)
        assert api.resolve_streaming("auto", str(npz_path)) == (False, None)
        assert "below the" in caplog.text
        # ...unless sparse was asked for by name
        stream, _ = api.resolve_streaming("auto", str(npz_path), inversion="sparse")
        assert stream is True


def test_auto_holds_without_jax(npz_path, monkeypatch):
    from pyuvimage import api

    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: "no JAX here")
    assert api.resolve_streaming("auto", str(npz_path)) == (False, None)


def test_explicit_streaming_is_the_users_word(ragged):
    from pyuvimage import api

    uvd, _, _ = ragged
    assert api.resolve_streaming(True, uvd, mode="cube") == (True, None)
    assert api.resolve_streaming(False, "some/file.npz") == (False, None)
    with pytest.raises(ValueError, match="unknown streaming"):
        api.resolve_streaming("maybe", uvd)
    with pytest.raises(ValueError, match="unknown inversion"):
        api.resolve_streaming("auto", "some/file.npz", inversion="wtilde")


def test_the_cli_defaults_to_auto_and_has_both_switches(monkeypatch):
    """`--streaming` and `--no-streaming` are the two words; nothing said
    means `auto`. `--reload` rides along as a plain flag."""
    from pyuvimage import api, cli

    seen = []
    monkeypatch.setattr(api, "run", lambda *a, **k: seen.append(k))
    base = ["fit", "d.npz", "--fov", "1"]
    for extra, streaming, reload in (
        ([], "auto", False),
        (["--streaming"], True, False),
        (["--no-streaming"], False, False),
        (["--reload"], "auto", True),
        (["--streaming", "--reload"], True, True),
    ):
        cli.main(base + extra)
        got = seen[-1]["streaming"]
        assert got is streaming if isinstance(streaming, bool) else got == "auto"
        assert seen[-1]["reload"] is reload


# --- recentring, chunk by chunk ---------------------------------------------------

def test_recentred_chunks_are_shift_image_centre_chunk_by_chunk(ragged):
    """`streaming.recentred` is `uvdata.shift_image_centre` applied per chunk:
    the same phase ramp, the same pooled noise, the uv untouched."""
    from pyuvimage.uvdata import shift_image_centre

    uvd, _, _ = ragged
    centre = (0.7, -0.4)                        # grid (y, x) arcsec
    reference = list(stm.iter_uvdata_chunks(shift_image_centre(uvd, centre), CHUNK))
    streamed = list(stm.recentred(stm.iter_uvdata_chunks(uvd, CHUNK), centre))
    assert len(reference) == len(streamed) > 1
    for r, s in zip(reference, streamed):
        assert np.array_equal(r.uv, s.uv)
        np.testing.assert_allclose(s.data, r.data, rtol=1e-12, atol=1e-12 * np.abs(r.data).max())
        np.testing.assert_allclose(s.noise, r.noise, rtol=1e-14)
    # a zero shift is a no-op, noise included -- exactly as in memory
    plain = list(stm.recentred(stm.iter_uvdata_chunks(uvd, CHUNK), (0.0, 0.0)))
    for a, b in zip(plain, stm.iter_uvdata_chunks(uvd, CHUNK)):
        assert np.array_equal(a.data, b.data) and np.array_equal(a.noise, b.noise)


def test_recentred_terms_are_the_terms_of_the_recentred_dataset(ragged, tmp_path):
    from pyuvimage.uvdata import shift_image_centre

    uvd, _, geom = ragged
    centre = (0.7, -0.4)
    mask = fitting.make_mask(geom, "square")
    direct = stm.accumulate_sparse_terms(
        stm.iter_uvdata_chunks(shift_image_centre(uvd, centre), CHUNK), geom, mask,
        ag.TransformerDFT, pool_noise=True)
    streamed = stm.sparse_terms_for(uvd, geom, mask, ag.TransformerDFT, chunk_k=CHUNK,
                                    pool_noise=True, centre=centre, cache_dir=tmp_path)
    np.testing.assert_allclose(streamed.kernel, direct.kernel, rtol=1e-12)
    np.testing.assert_allclose(streamed.dirty_image, direct.dirty_image, rtol=1e-10,
                               atol=1e-12 * np.abs(direct.dirty_image).max())
    assert streamed.data_term == pytest.approx(direct.data_term, rel=1e-12)
    # the centre is in the key: the phase-centre terms are a different entry
    assert stm.terms_key(uvd, geom, pool_noise=True, centre=centre) != \
        stm.terms_key(uvd, geom, pool_noise=True)
    assert stm.terms_key(uvd, geom, pool_noise=True, centre=(0.0, 0.0)) == \
        stm.terms_key(uvd, geom, pool_noise=True)


def test_the_streamed_wide_field_image_is_the_direct_one(ragged, flat, npz_path, tmp_path, caplog):
    """`--image-centre auto` on a stream images the whole file by the same
    direct summation `beam.wide_field_dirty_image` uses, and caches it."""
    import logging
    from pyuvimage import beam as beam_mod

    uvd, _, _ = ragged
    uv, d, n = uvd.flattened()
    ref, rms_ref = beam_mod.wide_field_dirty_image(uv, d, n, 6.0, n_pixels=16)
    img, rms = stm.wide_field_image_for(str(npz_path), 6.0, 16, chunk_k=CHUNK, cache_dir=tmp_path)
    np.testing.assert_allclose(img, ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())
    assert rms == pytest.approx(rms_ref, rel=1e-12)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        again, _ = stm.wide_field_image_for(str(npz_path), 6.0, 16, chunk_k=CHUNK, cache_dir=tmp_path)
    assert "reusing the cached wide-field image" in caplog.text
    assert np.array_equal(again, img)


def test_the_streamed_centre_decision_matches_the_in_memory_one(ragged, npz_path, tmp_path):
    from pyuvimage import api

    uvd, _, _ = ragged
    header = stm.scan_header(str(npz_path), CHUNK)
    # explicit: the same grid offset `_recentre` would shift by
    assert api._streamed_centre(str(npz_path), header, (1.0, 0.5), 3.0, None, CHUNK,
                                tmp_path, True) == pytest.approx(api._explicit_centre((1.0, 0.5)))
    assert api._streamed_centre(str(npz_path), header, "0,0", 3.0, None, CHUNK, tmp_path, True) is None
    assert api._streamed_centre(str(npz_path), header, "centre", 3.0, None, CHUNK, tmp_path, True) is None
    # the header of a recentred stream carries the offset for the WCS
    shifted = header.with_centre(0.7, -0.4)
    assert shifted.meta["image_centre_offset_arcsec"] == [0.7, -0.4]
    assert "image_centre_offset_arcsec" not in header.meta


# --- cube mode: one channel at a time ---------------------------------------------

def test_per_channel_chunks_carry_their_channel_and_never_span_two(ragged, npz_path):
    uvd, _, _ = ragged
    for source in (uvd, str(npz_path)):
        chunks = list(stm.iter_chunks(source, CHUNK, per_channel=True))
        assert all(ch.channel is not None for ch in chunks)
        # concatenated, still `flattened()` bit for bit
        uv0, d0, n0 = uvd.flattened()
        assert np.array_equal(np.concatenate([c.uv for c in chunks]), uv0)
        assert np.array_equal(np.concatenate([c.data for c in chunks]), d0)
        # every (spw, channel) of the header appears, in header order
        seen = list(dict.fromkeys((c.spw, c.channel) for c in chunks))
        assert sorted(seen) == sorted((i, c) for i, s in enumerate(uvd.spws)
                                      for c in range(s.n_chan))
    # the header's channel order is the in-memory one
    header = stm.scan_header(str(npz_path), CHUNK)
    assert header.channel_index() == uvd._channel_index()


def test_channel_terms_are_each_channels_own_and_sum_to_the_mfs_terms(ragged, terms):
    """One pass yields every channel's terms; each equals the terms of that
    channel alone, and their sum is the MFS terms (every field is a sum over
    visibilities)."""
    uvd, _, geom = ragged
    mask = fitting.make_mask(geom, "square")
    per_channel, mfs = stm.accumulate_channel_terms(
        stm.iter_uvdata_chunks(uvd, CHUNK, per_channel=True), geom, mask,
        ag.TransformerDFT, pool_noise=True)
    assert set(per_channel) == {(i, c) for i, s in enumerate(uvd.spws) for c in range(s.n_chan)}
    for c, (spw_i, chan_i) in enumerate(uvd._channel_index()):
        one = uvd.select(channel=c)
        alone = stm.accumulate_sparse_terms(stm.iter_uvdata_chunks(one, CHUNK), geom, mask,
                                            ag.TransformerDFT, pool_noise=True)
        got = per_channel[(spw_i, chan_i)]
        assert got.n_vis == alone.n_vis > 0
        np.testing.assert_allclose(got.kernel, alone.kernel, rtol=1e-12)
        np.testing.assert_allclose(got.dirty_image, alone.dirty_image, rtol=1e-10,
                                   atol=1e-12 * np.abs(alone.dirty_image).max())
        assert got.data_term == pytest.approx(alone.data_term, rel=1e-12)
    np.testing.assert_allclose(mfs.kernel, terms.kernel, rtol=1e-12)
    np.testing.assert_allclose(mfs.dirty_image, terms.dirty_image, rtol=1e-10,
                               atol=1e-12 * np.abs(terms.dirty_image).max())
    assert mfs.n_vis == terms.n_vis
    assert mfs.data_term == pytest.approx(terms.data_term, rel=1e-12)
    # every channel's terms carry the whole dataset's row bookkeeping
    assert np.array_equal(mfs.row_lengths, terms.row_lengths)
    for t in per_channel.values():
        assert np.array_equal(t.row_lengths, terms.row_lengths)
        assert np.array_equal(t.row_keep, terms.row_keep)


def test_thinned_prior_terms_hold_about_one_channels_worth(ragged, terms):
    uvd, _, geom = ragged
    mask = fitting.make_mask(geom, "square")
    n_chan = uvd.n_chan
    _, thinned = stm.accumulate_channel_terms(
        stm.iter_uvdata_chunks(uvd, CHUNK, per_channel=True), geom, mask,
        ag.TransformerDFT, pool_noise=True, thin=n_chan)
    expected = terms.n_vis / n_chan
    assert abs(thinned.n_vis - expected) < 4 * np.sqrt(expected)
    assert 0 < thinned.data_term < terms.data_term
    assert thinned.kernel[0, 0] < terms.kernel[0, 0]
    # deterministic: the same draw on a second pass
    _, again = stm.accumulate_channel_terms(
        stm.iter_uvdata_chunks(uvd, CHUNK, per_channel=True), geom, mask,
        ag.TransformerDFT, pool_noise=True, thin=n_chan)
    assert again.n_vis == thinned.n_vis and np.array_equal(again.kernel, thinned.kernel)


def test_channel_terms_are_cached_per_channel(ragged, npz_path, tmp_path, caplog):
    import logging

    uvd, _, geom = ragged
    mask = fitting.make_mask(geom, "square")
    header = stm.scan_header(str(npz_path), CHUNK)
    channels = header.channel_index()
    first, prior = stm.channel_terms_for(str(npz_path), geom, mask, ag.TransformerDFT, channels,
                                         cache_dir=tmp_path, chunk_k=CHUNK, pool_noise=True,
                                         thin=len(channels))
    assert len(list(tmp_path.glob("terms_*"))) == len(channels) + 1
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        again, prior2 = stm.channel_terms_for(str(npz_path), geom, mask, ag.TransformerDFT, channels,
                                              cache_dir=tmp_path, chunk_k=CHUNK, pool_noise=True,
                                              thin=len(channels))
    assert "no data read this run" in caplog.text
    for key in channels:
        assert np.array_equal(again[key].kernel, first[key].kernel)
    assert np.array_equal(prior2.kernel, prior.kernel)
    # --reload streams again
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        stm.channel_terms_for(str(npz_path), geom, mask, ag.TransformerDFT, channels,
                              cache_dir=tmp_path, chunk_k=CHUNK, pool_noise=True,
                              thin=len(channels), reuse_cache=False)
    assert "no data read this run" not in caplog.text


# --- the run_streamed flow, with the JAX-only pieces faked -----------------------

class _ZeroFit:
    """A `SingleFit` stand-in: the zero model on a stub's terms.

    Everything `run_streamed` and `_products_for` read from a fit -- and
    nothing that needs the sparse operator. chi^2 of the zero model is the
    data term, so the numbers stay honest."""

    def __init__(self, stub, geometry, prior):
        terms = fitting.streamed_terms_of(stub)
        self.model_image = np.zeros(tuple(geometry.shape_native))
        self.model_mesh_image = np.zeros(tuple(geometry.mesh_shape))
        self.chi_squared = float(terms.data_term)
        self.log_evidence = -0.5 * self.chi_squared
        self.prior = dict(prior or {"coefficient": 1.0, "scale": 0.5, "nu": 1.5})
        self.coefficient = float(self.prior["coefficient"])
        self.scan = None
        self.positive_only = True
        self.points = []


class _Stub:
    pass


@pytest.fixture
def faked_sparse_fit(monkeypatch):
    """Route the sparse-only steps of `run_streamed` through fakes so the rest
    of the flow -- header, centre, per-channel terms, channel loop, record,
    products -- runs where JAX is absent."""
    from pyuvimage import api

    monkeypatch.setattr(fitting, "sparse_inversion_diagnosis", lambda: None)

    def stub_from(terms, geometry, mask, batch_size=128):
        st = _Stub()
        setattr(st, fitting.STREAMED_TERMS_ATTR, terms)
        st.terms, st.mask = terms, mask
        return st

    monkeypatch.setattr(stm, "stub_dataset_from_terms", stub_from)
    monkeypatch.setattr(fitting, "imager_for", lambda ds: stm.KernelDirtyImager(ds.terms, ds.mask))
    fits = []

    def fake_fit_dataset(dataset, geometry, prior=None, **kwargs):
        f = _ZeroFit(dataset, geometry, prior)
        fits.append((f, {"prior": prior, **kwargs}))
        return f

    monkeypatch.setattr(fitting, "fit_dataset", fake_fit_dataset)
    return fits


def test_run_streamed_cube_flow(ragged, npz_path, tmp_path, faked_sparse_fit):
    """Cube mode end to end on the streaming path: one plane per channel in
    header order, each fitted on its own terms with the frozen prior, the
    record and the products written."""
    from pyuvimage import api

    uvd, _, _ = ragged
    res = api.run(str(npz_path), fov=3.0, mode="cube", out=tmp_path / "cube",
                  reg="matern", coefficient=10.0, reg_scale=0.5, pb_correction=False,
                  uncertainty_map=False, chunk_k=CHUNK, kernel_cache=str(tmp_path / "cache"),
                  streaming=True)
    assert len(res.products) == uvd.n_chan
    # the MFS/prior fit, then one per channel with the frozen prior
    fits = faked_sparse_fit
    assert len(fits) == 1 + uvd.n_chan
    assert all(kw["prior"] == fits[0][0].prior for _, kw in fits[1:])
    assert res.parameters["streaming"]["n_visibilities_streamed"] == uvd.n_samples
    assert res.parameters["source_prior"]["prior_fitted_on_one_visibility_in"] == uvd.n_chan
    assert len(res.parameters["fit_quality"]["channel_chi2_per_datum"]) == uvd.n_chan
    assert (tmp_path / "cube" / "model.fits").exists()
    from astropy.io import fits as afits
    assert afits.getdata(tmp_path / "cube" / "model.fits").shape[0] == uvd.n_chan
    # cached per channel (+ the thinned prior): a re-run reads nothing
    assert len(list((tmp_path / "cache").glob("terms_*"))) == uvd.n_chan + 1


def test_run_streamed_recentred_flow(ragged, npz_path, tmp_path, faked_sparse_fit, caplog):
    import logging
    from pyuvimage import api

    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        res = api.run(str(npz_path), fov=3.0, out=tmp_path / "rc", reg="matern",
                      coefficient=10.0, reg_scale=0.5, pb_correction=False,
                      uncertainty_map=False, chunk_k=CHUNK, image_centre=(0.6, -0.3),
                      kernel_cache=str(tmp_path / "cache"), streaming=True)
    assert "recentring the reconstruction on x +0.600" in caplog.text
    assert res.uvdata.meta["image_centre_offset_arcsec"] == pytest.approx(
        list(api._explicit_centre((0.6, -0.3))))
    assert res.parameters["streaming"]["image_centre_grid_arcsec"] == pytest.approx(
        list(api._explicit_centre((0.6, -0.3))))
    assert res.parameters["streaming"]["noise_pooled"] is True
    # and "auto" images the stream once, then decides
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        api.run(str(npz_path), fov=3.0, out=tmp_path / "auto", reg="matern",
                coefficient=10.0, reg_scale=0.5, pb_correction=False,
                uncertainty_map=False, chunk_k=CHUNK, image_centre="auto",
                kernel_cache=str(tmp_path / "cache"), streaming=True)
    assert "imaging a" in caplog.text and "brightest peak" in caplog.text


# --- JAX-only: the fit itself ----------------------------------------------------

jax_needed = pytest.mark.skipif(
    fitting.sparse_inversion_diagnosis() is not None,
    reason="the sparse operator needs JAX",
)


@jax_needed
def test_streamed_fit_matches_the_in_memory_sparse_fit(ragged, flat, tmp_path):
    """The same F, D and constants, so the same chi^2, reconstruction and
    products -- from a stub that holds eight visibilities."""
    from pyuvimage import api

    uvd, _, geom = ragged
    uv, d, n, ds = flat
    reference = api.run(uvd, fov=3.0, out=tmp_path / "mem", reg="matern",
                        criterion="discrepancy", inversion="sparse",
                        pb_correction=False, write=False)
    streamed = api.run(uvd, fov=3.0, out=tmp_path / "str", reg="matern",
                       criterion="discrepancy", inversion="sparse",
                       pb_correction=False, write=False, streaming=True, chunk_k=CHUNK)
    r, s = reference.products[0], streamed.products[0]
    assert s.chi_squared == pytest.approx(r.chi_squared, rel=1e-8)
    np.testing.assert_allclose(s.model_image, r.model_image, rtol=1e-8,
                               atol=1e-10 * np.abs(r.model_image).max())
    np.testing.assert_allclose(s.residual_sigma, r.residual_sigma, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(s.dirty_image, r.dirty_image, rtol=1e-10,
                               atol=1e-12 * np.abs(r.dirty_image).max())
    assert streamed.parameters["fit_quality"]["n_data"] == 2 * len(d)
    assert streamed.parameters["streaming"]["n_visibilities_streamed"] == len(d)


@jax_needed
def test_streamed_cube_matches_the_in_memory_sparse_cube(ragged, tmp_path):
    """Every channel from its own streamed terms, the shared prior from the
    summed ones (`cube_prior="mfs"`, so both paths fit it on all the data)."""
    from pyuvimage import api

    uvd, _, geom = ragged
    kw = dict(fov=3.0, mode="cube", cube_prior="mfs", reg="matern", criterion="discrepancy",
              inversion="sparse", pb_correction=False, write=False, uncertainty_map=False)
    reference = api.run(uvd, out=tmp_path / "mem", **kw)
    streamed = api.run(uvd, out=tmp_path / "str", streaming=True, chunk_k=CHUNK, **kw)
    assert len(streamed.products) == len(reference.products) == uvd.n_chan
    for r, s in zip(reference.products, streamed.products):
        assert s.chi_squared == pytest.approx(r.chi_squared, rel=1e-8)
        np.testing.assert_allclose(s.model_image, r.model_image, rtol=1e-8,
                                   atol=1e-10 * np.abs(r.model_image).max())
        np.testing.assert_allclose(s.residual_sigma, r.residual_sigma, rtol=1e-8, atol=1e-8)
    assert streamed.parameters["streaming"]["n_visibilities_streamed"] == uvd.n_samples


@jax_needed
def test_streamed_recentred_fit_matches_the_in_memory_one(ragged, tmp_path):
    from pyuvimage import api

    uvd, _, geom = ragged
    kw = dict(fov=3.0, reg="matern", criterion="discrepancy", inversion="sparse",
              pb_correction=False, write=False, uncertainty_map=False, image_centre=(0.6, -0.3))
    reference = api.run(uvd, out=tmp_path / "mem", **kw)
    streamed = api.run(uvd, out=tmp_path / "str", streaming=True, chunk_k=CHUNK, **kw)
    r, s = reference.products[0], streamed.products[0]
    assert s.chi_squared == pytest.approx(r.chi_squared, rel=1e-8)
    np.testing.assert_allclose(s.model_image, r.model_image, rtol=1e-8,
                               atol=1e-10 * np.abs(r.model_image).max())
    assert streamed.uvdata.meta["image_centre_offset_arcsec"] == \
        pytest.approx(reference.uvdata.meta["image_centre_offset_arcsec"])

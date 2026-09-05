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
    (dict(mode="cube"), NotImplementedError),
    (dict(point_sources=True), NotImplementedError),
    (dict(image_centre="auto"), NotImplementedError),
    (dict(inversion="dense"), ValueError),
])
def test_run_streamed_refuses_what_it_cannot_do_yet(ragged, kwargs, exc):
    from pyuvimage import api
    uvd, _, _ = ragged
    with pytest.raises(exc):
        api.run_streamed(uvd, fov=3.0, write=False, **kwargs)


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

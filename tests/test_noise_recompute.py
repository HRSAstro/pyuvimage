"""The choice of noise estimator must not be frozen at import time.

It was, until the first real dataset arrived carrying a noise map that measured
the source rather than the noise -- with nothing in the file to rebuild it
from, the only repair was to go back to the measurement set. Exports now carry
`antenna1`, `antenna2`, `time` and the weight-derived relative sigma, so any
estimator can be applied afterwards, and `pyuvimage convert --noise` stores the
result once instead of every run paying for it.
"""

import numpy as np
import pytest

from pyuvimage.uvdata import (
    NOISE_MODES,
    MultiSpwUVData,
    UVData,
    read_dataset,
    recompute_noise,
)

N_ANT, N_TIME, N_CHAN = 16, 8, 2


def _dataset(seed=0, with_weights=True, with_rows=True, wrong_noise=0.09):
    rng = np.random.default_rng(seed)
    a1i, a2i = np.triu_indices(N_ANT, k=1)
    n_base = a1i.size
    ant1 = np.tile(a1i, N_TIME)
    ant2 = np.tile(a2i, N_TIME)
    time = np.repeat(np.arange(N_TIME) * 6.0, n_base)
    n_vis = ant1.size

    sigma_true = 0.004 * np.repeat(rng.uniform(1.0, 2.0, n_base), 1)
    sigma_true = np.tile(sigma_true, N_TIME)[None, :] * np.ones((N_CHAN, 1))
    sky = np.tile(rng.normal(0.0, 0.05, n_base), N_TIME)[None, :]
    data = (
        sky
        + rng.normal(0, 1, sigma_true.shape) * sigma_true
        + 1j * rng.normal(0, 1, sigma_true.shape) * sigma_true
    )
    return UVData(
        uvw=rng.normal(0, 200, (n_vis, 3)),
        frequencies=np.linspace(1e11, 1.01e11, N_CHAN),
        data=data,
        # deliberately wrong, as a mis-scaled weight column would give
        noise=np.full((N_CHAN, n_vis), wrong_noise * (1 + 1j)),
        antenna1=ant1 if with_rows else None,
        antenna2=ant2 if with_rows else None,
        time=time if with_rows else None,
        weight_sigma=(
            np.full((N_CHAN, n_vis), 7.0 * (1 + 1j)) * sigma_true
            if with_weights else None
        ),
        meta={"noise_estimate": "difference"},
    ), sigma_true


def test_every_mode_recovers_the_real_level():
    uvd, truth = _dataset()
    for mode in ("difference", "chunked", "hybrid", "scaled"):
        out = recompute_noise(uvd, mode)
        assert np.median(out.noise.real) == pytest.approx(
            np.median(truth), rel=0.25
        ), mode
        assert out.meta["noise_estimate"] == mode


def test_keep_is_a_no_op():
    uvd, _ = _dataset()
    assert recompute_noise(uvd, "keep") is uvd


def test_the_input_is_not_modified():
    uvd, _ = _dataset()
    before = uvd.noise.copy()
    recompute_noise(uvd, "difference")
    assert np.array_equal(uvd.noise, before)


def test_the_ingredients_are_carried_forward():
    """So a second re-estimate does not have to go back to the MS either."""
    uvd, _ = _dataset()
    once = recompute_noise(uvd, "difference")
    assert once.can_reestimate_noise
    twice = recompute_noise(once, "hybrid")
    assert np.all(np.isfinite(twice.noise.real))


def test_a_dataset_without_row_metadata_says_so_clearly():
    uvd, _ = _dataset(with_rows=False)
    assert not uvd.can_reestimate_noise
    with pytest.raises(ValueError, match="antenna1/antenna2/time"):
        recompute_noise(uvd, "difference")


def test_weight_modes_refuse_rather_than_guess_without_weights():
    uvd, _ = _dataset(with_weights=False)
    for mode in ("hybrid", "scaled"):
        with pytest.raises(ValueError, match="weight column"):
            recompute_noise(uvd, mode)
    # the data-only modes are unaffected
    assert np.all(recompute_noise(uvd, "difference").noise.real > 0)


def test_unknown_mode_is_rejected():
    uvd, _ = _dataset()
    with pytest.raises(ValueError, match="unknown noise mode"):
        recompute_noise(uvd, "statwt")


def test_flagged_samples_do_not_steer_the_estimate():
    uvd, truth = _dataset()
    flags = np.zeros(uvd.data.shape, dtype=bool)
    flags[0, ::5] = True
    corrupted = uvd.data.copy()
    corrupted[0, ::5] = 1e6                       # nonsense, but flagged
    bad = UVData(
        uvw=uvd.uvw, frequencies=uvd.frequencies, data=corrupted,
        noise=uvd.noise, flags=flags, antenna1=uvd.antenna1,
        antenna2=uvd.antenna2, time=uvd.time, weight_sigma=uvd.weight_sigma,
    )
    out = recompute_noise(bad, "difference")
    assert np.median(out.noise.real) == pytest.approx(np.median(truth), rel=0.3)


def test_it_round_trips_through_a_dataset_directory(tmp_path):
    """The point of caching: write once, and later runs just read it."""
    uvd, truth = _dataset()
    fixed = recompute_noise(uvd, "chunked")
    fixed.write(tmp_path / "ds")

    back = read_dataset(tmp_path / "ds")
    assert np.allclose(back.noise, fixed.noise)
    assert back.meta["noise_estimate"] == "chunked"
    # and the ingredients survived, so the mode can still be changed later
    assert back.can_reestimate_noise
    assert back.weight_sigma is not None


def test_multi_spw_recomputes_every_window(tmp_path):
    a, _ = _dataset(seed=1)
    b, _ = _dataset(seed=2)
    multi = MultiSpwUVData(spws=[a, b])
    out = recompute_noise(multi, "difference")
    assert out.n_spw == 2
    for s in out.spws:
        assert np.all(s.noise.real > 0)
        assert s.meta["noise_estimate"] == "difference"


def test_the_mode_list_is_what_the_cli_offers():
    assert NOISE_MODES[0] == "keep"
    assert set(NOISE_MODES) == {"keep", "difference", "chunked", "hybrid", "scaled"}

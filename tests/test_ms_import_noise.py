"""End-to-end MS import with a fake casacore table.

`_import_from_open_ms` takes its `main` table and `table` factory as arguments,
so the whole import path -- Stokes I formation, WEIGHT_SPECTRUM, the three
noise modes, the weight consistency check -- runs without python-casacore
installed. Before this, none of that code was exercised by any test; it was
only ever run by hand against a real MS.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from pyuvimage.ms_import import _import_from_open_ms

C = 299792458.0
XX, YY = 9, 12

N_ANT = 12
N_TIME = 3          # 2 differences per baseline: below MIN_DIFFS on purpose
N_CHAN = 4
TRUE_SIGMA = 0.004
WEIGHT_SCALE = 7.0  # the pipeline's weights are off by this factor


class _Table:
    """Minimal stand-in for a casacore table."""

    def __init__(self, cols, nrows=None):
        self._cols = cols
        self._nrows = nrows if nrows is not None else len(next(iter(cols.values())))

    def query(self, _expr):
        return self          # the fake data is already the selection

    def nrows(self):
        return self._nrows

    def colnames(self):
        return list(self._cols)

    def getcol(self, name):
        if name not in self._cols:
            raise RuntimeError(f"no column {name}")
        return self._cols[name]

    def getcell(self, name, row):
        return self._cols[name][int(row)]

    def close(self):
        pass


def _make_ms(with_weight_spectrum=True, antenna_tsys=True, seed=0,
             n_chan=N_CHAN, n_time=N_TIME, flag_edges=False,
             sporadic_flags=0.0, source_jy=0.02):
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(N_ANT, k=1)
    n_base = a1.size
    ant1 = np.tile(a1, n_time)
    ant2 = np.tile(a2, n_time)
    time = np.repeat(np.arange(float(n_time)) * 30.0, n_base)
    n_row = ant1.size

    # per-antenna sensitivity -> a real, physically motivated weight shape
    if antenna_tsys:
        tsys = rng.uniform(1.0, 2.5, N_ANT)
        scale = np.sqrt(tsys[ant1] * tsys[ant2])
    else:
        scale = np.ones(n_row)
    # band edges are noisier, as WEIGHT_SPECTRUM would record
    chan_scale = np.ones(n_chan)
    if n_chan > 2:
        chan_scale[0] = chan_scale[-1] = 2.0

    sigma_true = TRUE_SIGMA * scale[:, None] * chan_scale[None, :]  # (row, chan)

    # A baseline keeps its uv position and drifts slowly with time, as earth
    # rotation would. If each integration got a fresh random uvw the source
    # would change between them and the time-difference estimator -- which
    # assumes the sky is the same in consecutive integrations -- would measure
    # the source instead of the noise.
    base_uvw = rng.normal(0.0, 200.0, (n_base, 3))
    drift = np.arange(n_time)[:, None, None] * rng.normal(0.0, 0.5, (1, n_base, 3))
    uvw = (base_uvw[None, :, :] + drift).reshape(n_row, 3)

    source = source_jy * np.exp(-((uvw[:, 0] ** 2 + uvw[:, 1] ** 2) / (2 * 150.0**2)))
    # XX and YY carry *different* weights: the variance of the weighted average
    # is then 1/sum(w), which differs from sum(sigma^2)/n^2
    per_corr = np.stack([sigma_true, 1.6 * sigma_true], axis=-1) * np.sqrt(2.0)
    data = (
        source[:, None, None]
        + rng.normal(0, 1, (n_row, n_chan, 2)) * per_corr
        + 1j * rng.normal(0, 1, (n_row, n_chan, 2)) * per_corr
    )

    flag = np.zeros((n_row, n_chan, 2), dtype=bool)
    if flag_edges and n_chan > 2:           # band edges flagged in every row
        flag[:, 0, :] = flag[:, -1, :] = True
    if sporadic_flags > 0:                  # isolated cells, both hands
        flag |= (rng.random((n_row, n_chan)) < sporadic_flags)[:, :, None]
    cols = {
        "DATA": data,
        "FLAG": flag,
        "UVW": uvw,
        "WEIGHT": (1.0 / (WEIGHT_SCALE * per_corr[:, 0, :]) ** 2),
        "ANTENNA1": ant1,
        "ANTENNA2": ant2,
        "TIME": time,
    }
    if with_weight_spectrum:
        cols["WEIGHT_SPECTRUM"] = 1.0 / (WEIGHT_SCALE * per_corr) ** 2

    main = _Table(cols, nrows=n_row)

    freqs = 100e9 + np.arange(n_chan) * 1e9
    subtables = {
        "SPECTRAL_WINDOW": _Table({"CHAN_FREQ": [freqs]}),
        "DATA_DESCRIPTION": _Table(
            {"SPECTRAL_WINDOW_ID": np.array([0]), "POLARIZATION_ID": np.array([0])}
        ),
        "POLARIZATION": _Table({"CORR_TYPE": [np.array([XX, YY])]}),
        "FIELD": _Table({"PHASE_DIR": [np.array([0.3, -0.4])], "NAME": ["target"]}),
        "ANTENNA": _Table({"DISH_DIAMETER": np.full(N_ANT, 12.0)}),
        "OBSERVATION": _Table({"TELESCOPE_NAME": ["ALMA"]}),
    }

    def table_factory(path, **_kw):
        return subtables[str(path).rsplit("/", 1)[-1]]

    # sigma of the weighted Stokes I average = 1/sqrt(sum w)
    w = 1.0 / per_corr**2
    sigma_i = 1.0 / np.sqrt(w.sum(axis=2))          # (row, chan)
    return main, table_factory, sigma_i.T           # -> (chan, row)


def _run(noise_estimate, **kw):
    main, table_factory, sigma_i = _make_ms(**kw)
    uvd = _import_from_open_ms(
        Path("/fake.ms"), main, table_factory, "auto", 0, 0, noise_estimate
    )
    return uvd, sigma_i


def test_difference_recovers_the_level_but_loses_the_shape():
    """PJ0116's regime exactly: one channel per spw, three integrations.

    `sigma_from_time_differences` pools channels, so with several channels a
    baseline can clear MIN_DIFFS even from three integrations. With a single
    channel -- which is what a 2-spw-by-1-channel continuum MS gives -- every
    baseline has two differences, falls below the threshold, and takes the
    pooled value. All per-baseline structure is lost.
    """
    uvd, sigma_i = _run("difference", n_chan=1)
    assert np.median(uvd.noise.real) == pytest.approx(np.median(sigma_i), rel=0.2)
    assert np.allclose(uvd.noise.real, uvd.noise.real.flat[0])


def test_scaled_keeps_the_shape_where_difference_cannot():
    """Same single-channel regime, but the weight shape survives."""
    uvd, sigma_i = _run("scaled", n_chan=1)
    assert uvd.noise.real.std() > 0
    assert np.corrcoef(uvd.noise.real.ravel(), sigma_i.ravel())[0, 1] > 0.99


def test_scaled_keeps_the_shape_and_fixes_the_level():
    uvd, sigma_i = _run("scaled")
    ratio = uvd.noise.real / sigma_i
    assert np.median(ratio) == pytest.approx(1.0, rel=0.2)
    # the per-antenna and per-channel structure survived
    assert uvd.noise.real.std() > 0
    assert np.corrcoef(uvd.noise.real.ravel(), sigma_i.ravel())[0, 1] > 0.99


def test_scaled_beats_difference_per_cell():
    diff, sigma_i = _run("difference")
    scaled, _ = _run("scaled")
    err_diff = np.median(np.abs(diff.noise.real / sigma_i - 1.0))
    err_scaled = np.median(np.abs(scaled.noise.real / sigma_i - 1.0))
    assert err_scaled < 0.5 * err_diff


def test_sigma_mode_reproduces_the_columns_wrong_scale_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="pyuvimage"):
        uvd, sigma_i = _run("sigma")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("--noise sigma" in w for w in warnings)
    # it faithfully reports what the column claims -- which is WEIGHT_SCALE off
    assert np.median(uvd.noise.real / sigma_i) == pytest.approx(WEIGHT_SCALE, rel=0.1)


def test_the_consistency_ratio_is_always_reported(caplog):
    for mode in ("difference", "scaled", "sigma"):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="pyuvimage"):
            _run(mode)
        lines = [r.getMessage() for r in caplog.records]
        check = [ln for ln in lines if "noise check" in ln]
        assert check, f"no consistency line for --noise {mode}"
        ratio = float(check[0].rsplit("ratio ", 1)[1].rstrip(")"))
        assert ratio == pytest.approx(WEIGHT_SCALE, rel=0.15)


def test_missing_weight_spectrum_is_not_fatal():
    uvd, sigma_i = _run("scaled", with_weight_spectrum=False)
    assert np.all(np.isfinite(uvd.noise.real)) and np.all(uvd.noise.real > 0)
    # without per-channel weights the band-edge structure is gone, by definition
    assert uvd.noise.real[0].mean() == pytest.approx(uvd.noise.real[1].mean(), rel=0.05)


def test_stokes_i_variance_uses_the_weights_actually_used():
    """Var of a *weighted* average is 1/sum(w), not sum(sigma^2)/n^2.

    With unequal per-hand weights the two differ, and the old code returned the
    variance of an estimator it had not formed.
    """
    uvd, sigma_i = _run("sigma")
    main, _, _ = _make_ms()
    w = main.getcol("WEIGHT_SPECTRUM")
    naive = np.sqrt((1.0 / w).sum(axis=2) / w.shape[2] ** 2).T
    correct = (1.0 / np.sqrt(w.sum(axis=2))).T
    assert not np.allclose(naive, correct)
    assert np.allclose(uvd.noise.real, correct)


# ------------------------------------------------ flags and the noise estimate

def _old_estimate_on_zero_filled(uvd):
    """What the import used to feed the estimators: flagged cells as 0.0."""
    from pyuvimage.noise import sigma_from_time_differences

    zero_filled = np.where(uvd.flags, 0.0, uvd.data)
    return sigma_from_time_differences(
        zero_filled, uvd.antenna1, uvd.antenna2, uvd.time
    )


def test_fully_flagged_edge_channels_do_not_lower_the_noise():
    """Flagged cells are stored as 0.0 and must be invisible to the estimator.

    A band edge flagged in every row differences zero against zero, and those
    exact zeros pull the pooled sigma down. Fed the zero-filled array, as the
    import used to, this harness gives 0.70x the truth (the review measured
    0.69x); masked to NaN it gives the truth.
    """
    uvd, sigma_i = _run("difference", flag_edges=True, n_time=12)
    ok = ~uvd.flags
    assert np.all(uvd.flags[0]) and np.all(uvd.flags[-1])
    truth = np.median(sigma_i[ok])
    assert np.median(uvd.noise.real[ok]) == pytest.approx(truth, rel=0.1)
    # the failure mode, stated
    old = _old_estimate_on_zero_filled(uvd)
    assert np.median(old.real[ok]) < 0.8 * truth


def test_sporadic_flags_on_a_bright_source_do_not_inflate_the_noise():
    """The other face of the same bug: a cell flagged in one integration but
    not the next differences the *source* against zero. With a bright source
    and 10% sporadic flags the zero-filled estimate is several times the truth
    (3.2x here, 27x in the review's case); masked, it is the truth."""
    uvd, sigma_i = _run("difference", sporadic_flags=0.10, source_jy=0.5, n_time=12)
    ok = ~uvd.flags
    truth = np.median(sigma_i[ok])
    assert np.median(uvd.noise.real[ok]) == pytest.approx(truth, rel=0.2)
    old = _old_estimate_on_zero_filled(uvd)
    assert np.median(old.real[ok]) > 2.0 * truth


@pytest.mark.parametrize("mode", ["difference", "hybrid", "scaled"])
def test_every_mode_sees_flags_as_nan(mode):
    """All the estimators and diagnostics run on the masked copy, not just
    the default mode."""
    uvd, sigma_i = _run(mode, sporadic_flags=0.10, source_jy=0.5, n_time=12)
    ok = ~uvd.flags
    assert np.median(uvd.noise.real[ok] / sigma_i[ok]) == pytest.approx(1.0, rel=0.25)
    assert np.all(np.isfinite(uvd.noise.real)) and np.all(uvd.noise.real > 0)


# ------------------------------------------- the re-estimation ingredients

def test_an_imported_dataset_can_have_its_noise_re_estimated():
    """`recompute_noise` refused imported datasets as "written by an older
    export": the importer did not pass antenna1/antenna2/time/weight_sigma
    through, only casa_export.py did. Now both store the same ingredients."""
    from pyuvimage.uvdata import recompute_noise

    uvd, sigma_i = _run("difference")
    assert uvd.can_reestimate_noise
    assert uvd.antenna1.dtype == np.int64 and uvd.time.dtype == np.float64
    assert uvd.weight_sigma is not None and uvd.weight_sigma.shape == uvd.data.shape
    # the relative sigma the weights imply, real == imag as casa_export stores it
    assert np.allclose(uvd.weight_sigma.real, uvd.weight_sigma.imag)
    assert np.median(uvd.weight_sigma.real / sigma_i) == pytest.approx(WEIGHT_SCALE, rel=0.1)

    for mode in ("difference", "hybrid", "scaled"):
        again = recompute_noise(uvd, mode)
        assert again.meta["noise_estimate"] == mode
        assert np.median(again.noise.real / sigma_i) == pytest.approx(1.0, rel=0.25)


def test_the_ingredients_survive_a_round_trip_to_disk(tmp_path):
    from pyuvimage.uvdata import read_dataset

    uvd, _ = _run("difference")
    uvd.write(tmp_path / "ds")
    back = read_dataset(tmp_path / "ds")
    assert back.can_reestimate_noise and back.weight_sigma is not None
    np.testing.assert_array_equal(back.antenna1, uvd.antenna1)
    np.testing.assert_allclose(back.weight_sigma, uvd.weight_sigma, equal_nan=True)


def test_the_whole_track_estimate_is_computed_once(monkeypatch):
    """The import used to run `sigma_from_time_differences` three times over
    on the same data -- once itself, once inside the chunked estimator, once
    inside the baseline-length diagnostic."""
    from pyuvimage import noise as noise_mod

    calls = []
    real = noise_mod.sigma_from_time_differences

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(noise_mod, "sigma_from_time_differences", counting)
    _run("hybrid")
    assert len(calls) == 1

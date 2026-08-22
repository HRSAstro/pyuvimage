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
             n_chan=N_CHAN, n_time=N_TIME):
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

    source = 0.02 * np.exp(-((uvw[:, 0] ** 2 + uvw[:, 1] ** 2) / (2 * 150.0**2)))
    # XX and YY carry *different* weights: the variance of the weighted average
    # is then 1/sum(w), which differs from sum(sigma^2)/n^2
    per_corr = np.stack([sigma_true, 1.6 * sigma_true], axis=-1) * np.sqrt(2.0)
    data = (
        source[:, None, None]
        + rng.normal(0, 1, (n_row, n_chan, 2)) * per_corr
        + 1j * rng.normal(0, 1, (n_row, n_chan, 2)) * per_corr
    )

    cols = {
        "DATA": data,
        "FLAG": np.zeros((n_row, n_chan, 2), dtype=bool),
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

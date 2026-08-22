"""The residual-structure ratio: does the leftover look like noise?

chi^2 constrains the residual's *total power*. It cannot tell a white residual
from a coherent one of the same power, and in the image plane those look
nothing alike: incoherent residuals average down as 1/sqrt(N), coherent ones
add in phase and land sqrt(N) higher.

On PJ0116 at 245 GHz a noise map inflated 1.4x by an export bug gave
chi^2/N = 1.0076 -- an apparently perfect fit -- with the entire Einstein ring
sitting in the residual map at 26.8 sigma. Structure ratio 4.3. With the noise
corrected: chi^2/N 1.0073, residual 3.7 sigma, ratio 0.93.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pyuvimage.api import STRUCTURE_RATIO_WARN, _report_dynamic_range


def _products(residual_sigma, chi_squared, peak=1.0, rms=0.01):
    return SimpleNamespace(
        reconvolved=np.array([[peak]]),
        residual_sigma=np.asarray(residual_sigma, dtype=float),
        rms=rms,
        chi_squared=float(chi_squared),
    )


def _lines(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def test_white_residual_reports_a_ratio_near_one(caplog):
    rng = np.random.default_rng(0)
    n_data = 10_000
    resid = rng.normal(0.0, 1.0, (64, 64))          # white, unit variance
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _report_dynamic_range(_products(resid, chi_squared=n_data), n_data)

    info = " ".join(_lines(caplog, logging.INFO))
    assert "structure ratio" in info
    ratio = float(info.split("structure ratio")[1].split()[0])
    assert ratio == pytest.approx(1.0, abs=0.1)
    assert not _lines(caplog, logging.WARNING)


def test_coherent_residual_is_flagged(caplog):
    """Same chi^2, but the residual is a smooth arc instead of noise."""
    n_data = 10_000
    y, x = np.mgrid[0:64, 0:64]
    r = np.hypot(y - 32, x - 32)
    # scaled so the map rms is ~4 sigma, as the real PJ0116 failure was
    resid = np.exp(-((r - 20.0) ** 2) / (2 * 2.0**2))
    resid *= 4.28 / resid.std()                                # a ring
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _report_dynamic_range(_products(resid, chi_squared=n_data), n_data)

    warnings = _lines(caplog, logging.WARNING)
    assert warnings, "a structured residual must warn"
    assert "structure ratio" in warnings[0]
    assert "residual.fits" in warnings[0]
    # and it must name the cause we actually hit in practice
    assert "noise map" in warnings[0]


def test_identical_chi2_opposite_verdicts(caplog):
    """The whole point: chi^2 cannot separate these, the ratio can."""
    rng = np.random.default_rng(1)
    n_data = 10_000
    y, x = np.mgrid[0:64, 0:64]
    r = np.hypot(y - 32, x - 32)

    ring = np.exp(-((r - 20.0) ** 2) / (2 * 2.0**2))
    ring *= 4.28 / ring.std()

    verdicts = {}
    for tag, resid in (("white", rng.normal(0.0, 1.0, (64, 64))), ("ring", ring)):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="pyuvimage"):
            # SAME chi^2 for both -- this is what the criterion sees
            _report_dynamic_range(_products(resid, chi_squared=n_data), n_data)
        verdicts[tag] = bool(_lines(caplog, logging.WARNING))

    assert verdicts == {"white": False, "ring": True}


def test_no_n_data_means_no_ratio(caplog):
    """Cube mode and older call sites must not crash or invent a number."""
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        _report_dynamic_range(_products(np.zeros((8, 8)), chi_squared=1.0))
    assert "structure ratio" not in " ".join(_lines(caplog, logging.INFO))


def test_threshold_leaves_room_for_scatter():
    assert 1.0 < STRUCTURE_RATIO_WARN < 2.5

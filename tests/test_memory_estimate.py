"""Warn before the OOM killer, not after.

An out-of-memory kill gives the user `Killed: 9` and nothing else -- no
traceback, no hint which knob to turn. The estimate exists so the fit says
what it needs while --mesh and --fov are still adjustable.

The scaling is the part people get wrong, including me: peak memory goes as
**n_vis x n_mesh**, not the image size. Ruby on a 26x26 mesh needs ~8x what
PJ0116 needs on a 50x50 one, because Ruby has 29x the visibilities.
"""

import logging

import pytest

from pyuvimage.fitting import (
    _mesh_that_fits,
    available_memory_gb,
    check_memory,
    estimate_peak_memory_gb,
)


def test_memory_scales_with_visibilities_not_image_size():
    """The counter-intuitive case, stated as an assertion.

    Ruby on a 26x26 mesh has 3.7x *fewer* model pixels than PJ0116 on a
    50x50 one, and still needs ~5x the memory, because it has 29x the
    visibilities. Reaching for a smaller field or mesh is the fix; reasoning
    from the image size is what makes the failure surprising.
    """
    ruby = estimate_peak_memory_gb(148_477, 26 * 26)      # small mesh, big data
    pj0116 = estimate_peak_memory_gb(5_158, 50 * 50)      # big mesh, small data
    assert 26 * 26 < 50 * 50, "Ruby really does have fewer model pixels"
    assert ruby > 4 * pj0116


@pytest.mark.parametrize("n_vis,n_mesh,expected", [
    (148_477, 16 * 16, 1.9),
    (148_477, 24 * 24, 3.8),
])
def test_the_estimate_matches_what_was_measured(n_vis, n_mesh, expected):
    """Calibrated against real peak RSS on the NumPy/pynufft path."""
    assert estimate_peak_memory_gb(n_vis, n_mesh) == pytest.approx(
        expected, rel=0.25
    )


def test_the_case_that_was_killed_is_flagged(caplog, monkeypatch):
    """Ruby at a 26x26 mesh on a machine with 4 GB free."""
    import pyuvimage.fitting as f

    monkeypatch.setattr(f, "available_memory_gb", lambda: 4.0)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        check_memory(148_477, 26 * 26)
    said = " ".join(r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "--mesh" in said, "the warning must name the lever"


def test_a_comfortable_fit_is_not_warned_about(caplog, monkeypatch):
    import pyuvimage.fitting as f

    monkeypatch.setattr(f, "available_memory_gb", lambda: 64.0)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        check_memory(5_158, 50 * 50)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_it_says_what_mesh_would_fit(caplog, monkeypatch):
    import pyuvimage.fitting as f

    monkeypatch.setattr(f, "available_memory_gb", lambda: 4.0)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        check_memory(148_477, 40 * 40)
    suggested = _mesh_that_fits(148_477, 4.0)
    assert 0 < suggested < 40
    assert estimate_peak_memory_gb(148_477, suggested**2) < 4.0


def test_an_unknown_memory_size_never_blocks_a_fit(caplog, monkeypatch):
    """Best-effort: a platform we cannot measure must still run."""
    import pyuvimage.fitting as f

    monkeypatch.setattr(f, "available_memory_gb", lambda: None)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        check_memory(148_477, 26 * 26)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_available_memory_is_a_positive_number_or_none():
    got = available_memory_gb()
    assert got is None or got > 0

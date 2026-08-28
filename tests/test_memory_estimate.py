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


# --- the adaptive two-pass peak -------------------------------------------
#
# `--reg adaptive` OOM-killed 9io9 twice at the exact moment its second pass
# began, on a fit whose single-inversion estimate (5.3 GB) fitted comfortably
# in 7.5 GB. The cause was not the estimate: `fit_dataset` held the first
# pass's `SingleFit` -- and so an `ag.FitInterferometer`, and so the
# transformed mapping matrix, n_vis x n_mesh complex -- for the whole of the
# second pass, while the only thing needed from it was the brightness array
# already copied out. Two mapping matrices alive at once is double the peak,
# and no memory estimate of a single inversion can predict that.


def test_the_first_adaptive_pass_is_released_before_the_second_allocates():
    """A weakref to the first pass's fit must be dead by the time the second
    pass starts, or the peak is twice what was reported."""
    import gc
    import weakref

    import pyuvimage
    from pyuvimage import fitting, mock

    uvd, _, geometry, _ = mock.make_demo_dataset(n_vis=400, mesh_n=8, seed=5)
    uv, d, n = uvd.flattened()
    dataset = fitting.make_dataset(uv, d, n, geometry, transformer="dft")

    real = fitting.fit_dataset
    seen: list = []
    alive_at_second_pass: list = []

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        if not seen:                      # the inner (first-pass) call
            seen.append(weakref.ref(out.fit))
        return out

    class Watch(logging.Handler):
        def emit(self, record):
            if "second pass" in record.getMessage() and seen:
                gc.collect()
                alive_at_second_pass.append(seen[0]() is not None)

    logger = logging.getLogger("pyuvimage")
    watch = Watch()
    logger.addHandler(watch)
    old = fitting.fit_dataset
    fitting.fit_dataset = spy
    try:
        spy(dataset, geometry, reg_kind="adaptive",
            prior={"coefficient": 1e4, "scale": 0.5}, positive_only=False)
    finally:
        fitting.fit_dataset = old
        logger.removeHandler(watch)

    assert alive_at_second_pass, "the second pass never started"
    assert alive_at_second_pass[0] is False, (
        "the first-pass fit was still alive when the second pass began: its "
        "transformed mapping matrix doubles the peak memory"
    )


# --- cube mode: the MFS pass is the expensive step -------------------------
#
# Cube mode fits each channel separately, which is cheap, but it first runs
# one MFS fit over *every* channel's visibilities to fix the prior -- and that
# pass is n_chan times the size of any single channel. Ruby CO(7-6) at a 27x27
# mesh: 2.9 GB per channel, 20.1 GB for the MFS pass. Reporting one number
# makes the whole cube look unaffordable when only one step of it is.


def test_cube_reports_the_per_channel_cost_separately(caplog):
    from pyuvimage.fitting import check_memory

    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        check_memory(613_512, 729, None, n_chan=8)
    assert "per-channel fits need about" in caplog.text
    assert "MFS pass over all 8 channels" in caplog.text


def test_cube_says_so_when_only_the_mfs_pass_is_unaffordable(monkeypatch, caplog):
    from pyuvimage import fitting

    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 6.4)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        fitting.check_memory(613_512, 729, None, n_chan=8)
    assert "per-channel fits would fit" in caplog.text
    # and it still names a mesh that would let the MFS pass through
    assert "per side" in caplog.text


def test_mfs_mode_reports_one_number_as_before(caplog):
    from pyuvimage.fitting import check_memory

    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        check_memory(613_512, 729, None)
    assert "per-channel" not in caplog.text


def test_the_cube_numbers_are_the_ones_that_were_measured():
    """Ruby CO(7-6), 613,512 samples over 8 channels, 27x27 mesh."""
    from pyuvimage.fitting import estimate_peak_memory_gb

    assert estimate_peak_memory_gb(613_512, 729) == pytest.approx(20.1, abs=0.3)
    assert estimate_peak_memory_gb(613_512 // 8, 729) == pytest.approx(2.9, abs=0.2)


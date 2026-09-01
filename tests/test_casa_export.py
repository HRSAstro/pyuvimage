"""The CASA export script, tested as far as it goes without casatools.

`_export_one` needs a live table tool, but the two things that went wrong in
this file did not: the spw argument parser dropped the inside of every range,
and the noise loop was a hand copy of `pyuvimage.noise` that could drift.
"""

import numpy as np
import pytest

from pyuvimage import casa_export
from pyuvimage.noise import (
    MIN_DIFFS,
    adjacent_pairs,
    baseline_sigma_from_pairs,
    sigma_from_time_differences,
)


@pytest.mark.parametrize("text,expected", [
    ("0", [0]),
    ("0,2", [0, 2]),
    ("0-3", [0, 1, 2, 3]),           # was [0, 3]: the range collapsed to its ends
    ("0-1,4", [0, 1, 4]),
    (" 2 , 0 ", [0, 2]),
    ("3-5,4", [3, 4, 5]),
])
def test_ranges_are_expanded_not_truncated(text, expected):
    assert casa_export.parse_spw_text(text) == expected
    # and `_resolve_spws` -- the one `export()` calls -- agrees, without
    # touching casatools for anything but "all"
    assert casa_export._resolve_spws("/no/such.ms", text) == expected


def test_the_command_line_and_the_api_share_one_parser():
    """`__main__` had its own correct parser while `_resolve_spws` had the
    broken one; a caller of `export(spw="0-3")` got two windows out of four."""
    assert casa_export._resolve_spws("x.ms", [3, 1, 1]) == [1, 3]
    assert casa_export._resolve_spws("x.ms", 2) == [2]
    with pytest.raises(ValueError):
        casa_export.parse_spw_text(",")


def test_the_export_noise_map_is_the_package_estimator():
    """casa_export now builds its noise from `baseline_sigma_from_pairs`; the
    fill-from-the-pool step it keeps must reproduce `sigma_from_time_differences`
    exactly, flags and all."""
    rng = np.random.default_rng(0)
    n_ant, n_time, n_chan = 10, 6, 3
    a1, a2 = np.triu_indices(n_ant, k=1)
    ant1, ant2 = np.tile(a1, n_time), np.tile(a2, n_time)
    time = np.repeat(np.arange(n_time) * 6.0, a1.size)
    n = ant1.size
    vis = (rng.normal(0, 0.004, (n_chan, n)) + 1j * rng.normal(0, 0.004, (n_chan, n))
           + 0.05 * np.cos(ant1 + 3 * ant2)[None, :])
    flags = rng.random((n_chan, n)) < 0.1
    vis[flags] = 0.0                            # as the export writes them

    # the script's own steps
    usable_cell = ~flags & np.isfinite(vis.real) & np.isfinite(vis.imag)
    pairs = adjacent_pairs(ant1, ant2, time)
    est = baseline_sigma_from_pairs(np.where(usable_cell, vis, np.nan), pairs)
    per_row = (est.sigma_re + 1j * est.sigma_im)[pairs.row_baseline]
    sigma = np.broadcast_to(per_row[None, :], vis.shape).copy()
    assert est.pool_count >= MIN_DIFFS
    bad = ~np.isfinite(sigma.real) | (sigma.real <= 0)
    sigma[bad] = est.pool_re + 1j * est.pool_im

    want = sigma_from_time_differences(np.where(flags, np.nan, vis), ant1, ant2, time)
    np.testing.assert_allclose(sigma, want, rtol=1e-12)
    # flagged cells were zeros in `vis`; they must not have been differenced
    assert np.all(np.isfinite(want.real))
    assert np.median(want.real) == pytest.approx(0.004, rel=0.25)


def test_the_script_imports_noise_without_the_rest_of_the_package():
    """It runs under CASA's python, so `noise.py` must stay numpy-only."""
    import importlib.util
    import pathlib

    src = pathlib.Path(casa_export.__file__).with_name("noise.py").read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.startswith(("import ", "from ")) and "__future__" not in ln]
    assert imports == ["import logging", "from typing import NamedTuple", "import numpy as np"]
    # and it loads standalone, by path, the way the script's fallback does
    spec = importlib.util.spec_from_file_location(
        "noise_alone", pathlib.Path(casa_export.__file__).with_name("noise.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "adjacent_pairs") and hasattr(mod, "baseline_sigma_from_pairs")

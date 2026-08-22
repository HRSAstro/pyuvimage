"""Reading MS subtable columns whose shape varies from row to row.

`POLARIZATION.CORR_TYPE`, `SPECTRAL_WINDOW.CHAN_FREQ` and `FIELD.PHASE_DIR`
are array columns whose cells need not share a shape. `getcol` has to return
one rectangular array for the whole column, so on a real MS with two
polarisation setups, or spectral windows with different channel counts, it
fails with

    Table DataManager error: Internal error:
    StManIndArray::get/put shapes not conforming

which is what a real ALMA dataset produced. These tests use a stub table that
behaves the same way, so the fix is checked without needing CASA installed.
"""

import numpy as np
import pytest

from pyuvimage.casa_export import _cell as casa_cell
from pyuvimage.ms_import import _cell as casacore_cell


class RaggedTable:
    """A subtable whose column cells have different shapes.

    `getcol` raises exactly as casacore does; `getcell` works. Mirrors a real
    MS with, say, a 2-correlation and a 4-correlation polarisation setup, or
    spectral windows of 128 and 3840 channels.
    """

    def __init__(self, cells):
        self.cells = [np.asarray(c) for c in cells]
        self.getcol_calls = 0

    def getcell(self, column, row):
        return self.cells[int(row)]

    def getcol(self, column):
        self.getcol_calls += 1
        shapes = {c.shape for c in self.cells}
        if len(shapes) > 1:
            raise RuntimeError(
                "Table DataManager error: Internal error: "
                "StManIndArray::get/put shapes not conforming"
            )
        return np.stack(self.cells)


class FixedShapeTable(RaggedTable):
    """The easy case: every cell the same shape, so getcol would also work."""


@pytest.mark.parametrize("cell", [casa_cell, casacore_cell])
def test_ragged_column_is_read_row_by_row(cell):
    """The regression: a mixed-correlation POLARIZATION table."""
    tab = RaggedTable([np.array([9, 12]), np.array([9, 10, 11, 12])])
    assert np.array_equal(cell(tab, "CORR_TYPE", 0), [9, 12])
    assert np.array_equal(cell(tab, "CORR_TYPE", 1), [9, 10, 11, 12])
    assert tab.getcol_calls == 0, "getcell must be tried first"


@pytest.mark.parametrize("cell", [casa_cell, casacore_cell])
def test_ragged_spectral_windows(cell):
    """Spectral windows with different channel counts -- the multi-spw case."""
    tab = RaggedTable([
        np.linspace(230e9, 232e9, 128),
        np.linspace(240e9, 244e9, 3840),
    ])
    assert cell(tab, "CHAN_FREQ", 0).size == 128
    assert cell(tab, "CHAN_FREQ", 1).size == 3840
    assert float(cell(tab, "CHAN_FREQ", 1)[0]) == pytest.approx(240e9)


@pytest.mark.parametrize("cell", [casa_cell, casacore_cell])
def test_getcol_on_a_ragged_column_really_does_fail(cell):
    """Guard the guard: if getcol worked, these tests would prove nothing."""
    tab = RaggedTable([np.array([9, 12]), np.array([9, 10, 11, 12])])
    with pytest.raises(RuntimeError, match="not conforming"):
        tab.getcol("CORR_TYPE")


@pytest.mark.parametrize("cell", [casa_cell, casacore_cell])
def test_fixed_shape_columns_still_work(cell):
    tab = FixedShapeTable([np.array([9, 12]), np.array([5, 8])])
    assert np.array_equal(cell(tab, "CORR_TYPE", 1), [5, 8])


@pytest.mark.parametrize("cell", [casa_cell, casacore_cell])
def test_phase_dir_gives_ra_then_dec(cell):
    """PHASE_DIR cells are (npoly, 2); ravel()[:2] must be (ra, dec)."""
    tab = RaggedTable([np.array([[1.234, -0.567]])])
    got = np.asarray(cell(tab, "PHASE_DIR", 0)).ravel()
    assert got[0] == pytest.approx(1.234)
    assert got[1] == pytest.approx(-0.567)


class NoGetcellTable:
    """A binding too old to have getcell: the fallback path must still work."""

    def __init__(self, cells, row_axis_last):
        self.cells = [np.asarray(c) for c in cells]
        self.row_axis_last = row_axis_last

    def getcol(self, column):
        stacked = np.stack(self.cells)
        return np.moveaxis(stacked, 0, -1) if self.row_axis_last else stacked


def test_casa_fallback_handles_row_axis_last():
    """casatools' getcol puts the row axis last."""
    tab = NoGetcellTable([np.array([9, 12]), np.array([5, 8])], row_axis_last=True)
    assert np.array_equal(casa_cell(tab, "CORR_TYPE", 1), [5, 8])


def test_casacore_fallback_handles_row_axis_first():
    """python-casacore's getcol puts the row axis first."""
    tab = NoGetcellTable([np.array([9, 12]), np.array([5, 8])], row_axis_last=False)
    assert np.array_equal(casacore_cell(tab, "CORR_TYPE", 1), [5, 8])

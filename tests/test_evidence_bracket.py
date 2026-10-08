"""The evidence search follows the coefficient wherever the data's units put it."""
import numpy as np

from pyuvimage import fitting, mock


def _best_coefficient(scale_units: float) -> float:
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=300, mesh_n=10, seed=4)
    uv, d, n = uvd.flattened()
    # Rescaling data and noise together changes nothing physical, but
    # multiplies F, and so the evidence-optimal coefficient, by 1/scale^2.
    ds = fitting.make_dataset(uv, d * scale_units, n * scale_units, geom, transformer="dft")
    prior, scan = fitting.optimise_prior(
        ds, geom, reg_kind="matern", criterion="evidence", fixed_scale=0.3,
        positive_only=False)
    return float(prior["coefficient"])


def test_the_evidence_peak_is_found_outside_the_shipped_bracket():
    base = _best_coefficient(1.0)
    shifted = _best_coefficient(1e-4)          # peak moves up by 8 decades
    assert np.log10(shifted) > fitting.LOG_COEFFICIENT_BOUNDS[1]
    np.testing.assert_allclose(np.log10(shifted / base), 8.0, atol=0.2)

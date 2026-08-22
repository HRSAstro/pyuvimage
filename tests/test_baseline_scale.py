"""The auto mesh scale must follow the baselines the data actually populates.

`0.5 / b_max` is the information limit, but a real array reaches b_max with a
handful of samples. PJ0116 at 245 GHz: median baseline 213 klambda, maximum
1054 klambda, naturally weighted beam 0.54". Sizing the mesh off the maximum
asked for ~30x more pixels than the data constrain, which took minutes per
likelihood evaluation and left the extra pixels set by the prior.
"""

import numpy as np
import pytest

from pyuvimage.api import BASELINE_PERCENTILE
from pyuvimage.grids import nyquist_pixel_scale_arcsec, resolve_geometry
from pyuvimage.uvdata import UVData

C = 299792458.0


def _dataset(lengths_m: np.ndarray, freq_hz: float = 245e9, flags=None) -> UVData:
    n = lengths_m.size
    angle = np.linspace(0.0, np.pi, n, endpoint=False)
    uvw = np.zeros((n, 3))
    uvw[:, 0] = lengths_m * np.cos(angle)
    uvw[:, 1] = lengths_m * np.sin(angle)
    return UVData(
        uvw=uvw,
        frequencies=np.array([freq_hz]),
        data=np.zeros((1, n), dtype=complex),
        noise=np.ones((1, n), dtype=complex) * (1 + 1j),
        flags=flags,
    )


def test_uniform_coverage_is_essentially_unchanged():
    """Every mock in this project: the percentile must not move the mesh."""
    rng = np.random.default_rng(0)
    # uniformly filled uv disc -> p95 radius is sqrt(0.95) of the maximum
    r = np.sqrt(rng.uniform(0, 1, 20_000)) * 500.0
    u = _dataset(r)
    ratio = u.max_baseline_wavelengths / u.baseline_percentile_wavelengths(
        BASELINE_PERCENTILE
    )
    assert ratio < 1.1, f"uniform coverage moved the scale by {ratio:.3f}x"


def test_sparse_long_tail_coarsens_the_mesh():
    """The real case: a few very long baselines must not size the mesh."""
    bulk = np.linspace(15.0, 350.0, 900)
    tail = np.linspace(400.0, 1300.0, 50)      # 5% of samples, 4x longer
    u = _dataset(np.concatenate([bulk, tail]))

    b_max = u.max_baseline_wavelengths
    b_eff = u.baseline_percentile_wavelengths(BASELINE_PERCENTILE)
    assert b_max > 2.5 * b_eff

    auto = resolve_geometry(
        fov_arcsec=8.0,
        max_baseline_wavelengths=b_max,
        effective_baseline_wavelengths=b_eff,
    )
    strict = resolve_geometry(
        fov_arcsec=8.0,
        max_baseline_wavelengths=b_max,
        pixel_scale="nyquist",
        effective_baseline_wavelengths=b_eff,
    )
    assert auto.mesh_pixel_scale > 2.0 * strict.mesh_pixel_scale
    # and the saving is quadratic in pixel count -- the point of the change
    assert np.prod(strict.mesh_shape) > 6 * np.prod(auto.mesh_shape)
    # nyquist_pixel_scale reports the information limit either way
    assert auto.nyquist_pixel_scale == pytest.approx(
        nyquist_pixel_scale_arcsec(b_max)
    )


def test_explicit_nyquist_still_uses_the_longest_baseline():
    u = _dataset(np.concatenate([np.full(950, 100.0), np.full(50, 1000.0)]))
    b_max = u.max_baseline_wavelengths
    g = resolve_geometry(
        fov_arcsec=4.0,
        max_baseline_wavelengths=b_max,
        pixel_scale="nyquist",
        effective_baseline_wavelengths=u.baseline_percentile_wavelengths(95.0),
    )
    n = int(np.ceil(4.0 / nyquist_pixel_scale_arcsec(b_max)))
    assert g.mesh_shape[0] in (n, n + 1)


def test_omitting_the_effective_baseline_reproduces_the_old_behaviour():
    g_old = resolve_geometry(fov_arcsec=4.0, max_baseline_wavelengths=5e5)
    g_strict = resolve_geometry(
        fov_arcsec=4.0, max_baseline_wavelengths=5e5, pixel_scale="nyquist"
    )
    assert g_old.mesh_shape == g_strict.mesh_shape
    assert g_old.pixel_scale == g_strict.pixel_scale


def test_flagged_samples_do_not_set_the_scale():
    """A fully flagged long baseline contributes no information."""
    lengths = np.concatenate([np.linspace(15.0, 300.0, 990), np.full(10, 5000.0)])
    flags = np.zeros((1, lengths.size), dtype=bool)
    flags[0, -10:] = True
    u = _dataset(lengths, flags=flags)
    assert u.baseline_percentile_wavelengths(99.9) < 0.5 * u.max_baseline_wavelengths


def test_multi_spw_pools_every_spw():
    from pyuvimage.uvdata import MultiSpwUVData

    a = _dataset(np.linspace(15.0, 300.0, 500), freq_hz=243e9)
    b = _dataset(np.linspace(15.0, 900.0, 500), freq_hz=245e9)
    m = MultiSpwUVData(spws=[a, b])
    pooled = m.baseline_percentile_wavelengths(95.0)
    assert a.baseline_percentile_wavelengths(95.0) < pooled
    assert pooled < m.max_baseline_wavelengths

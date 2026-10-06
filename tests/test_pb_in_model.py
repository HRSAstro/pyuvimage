"""The primary beam folded into the forward model (`--pb-in-model`)."""
import numpy as np
import pytest

from pyuvimage import fitting, mock, primary_beam
from pyuvimage.envelope import PrimaryBeamPrior


@pytest.fixture(scope="module")
def demo():
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=200, mesh_n=10, seed=5)
    uv, d, n = uvd.flattened()
    ds = fitting.make_dataset(uv, d, n, geom, transformer="dft")
    return ds, geom, fitting.build_linear_system(ds, geom.mesh_shape)


@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.3, -0.5)])
def test_the_pb_centre_follows_the_products_convention(offset):
    """The prior's PB, evaluated on autoarray grid coordinates with centre
    (-y0, -x0), is the PB `pb.fits` writes (`primary_beam_map`, whose
    orientation the WCS test pins down)."""
    import autoarray as aa
    from pyuvimage.api import _pb_prior_envelope
    from pyuvimage.grids import ImageGeometry

    geom = ImageGeometry(fov_arcsec=3.0, pixel_scale=0.15, shape_native=(20, 20),
                         mesh_shape=(10, 10), mask_radius=1.5, nyquist_pixel_scale=0.3)
    freq, dish = 230e9, 0.3            # a tiny dish: a PB a few arcsec across
    env = _pb_prior_envelope(None, geom, freq, dish, 1.13, offset)
    reg = fitting.make_regularization("matern", 1.0, 0.3, envelope=env)
    grid = aa.Grid2D.uniform(shape_native=geom.shape_native, pixel_scales=geom.pixel_scale)
    reg.floor = 0.0
    got = reg.pb_at(np.asarray(grid.native).reshape(-1, 2))
    want = primary_beam.primary_beam_map(geom.shape_native, geom.pixel_scale, freq, dish,
                                         1.13, offset).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)


def test_it_is_the_pb_folded_into_f(demo):
    """(F + P^-1 H P^-1) a = D  <=>  (P F P + H) t = P D,  a = P t."""
    _, geom, system = demo
    n = system.F.shape[0]
    inner = fitting.make_regularization("matern", 1.0, 0.3)
    wrapped = fitting.make_regularization(
        "matern", 1.0, 0.3, envelope={"primary_beam": {"fwhm": 2.0, "centre": (0.4, -0.2)}})
    assert isinstance(wrapped, PrimaryBeamPrior)
    H = system.regularization_matrix(inner)
    Hw = system.regularization_matrix(wrapped)
    pb = wrapped.pb
    assert pb.min() < 0.5 and pb.size == n
    a = np.linalg.solve(system.F + Hw, system.D)
    P = np.diag(pb)
    t = np.linalg.solve(P @ system.F @ P + H, P @ system.D)
    np.testing.assert_allclose(a, pb * t, rtol=0, atol=1e-6 * np.abs(a).max())
    assert wrapped.coefficient == inner.coefficient


def test_the_floor_and_the_true_sky_brightness():
    from types import SimpleNamespace

    yx = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    mesh = SimpleNamespace(source_plane_mesh_grid=SimpleNamespace(array=yx))
    fwhm = 2.0                                     # pb(1") = 0.5
    reg = fitting.make_regularization(
        "adaptive", 1.0, 0.3, envelope={
            "primary_beam": {"fwhm": fwhm}, "brightness": np.ones(3),
            "floor": 0.01, "power": 1.0})
    reg.regularization_matrix_from(mesh)
    np.testing.assert_allclose(reg.pb, [1.0, 0.5, fitting.PB_PRIOR_FLOOR])
    # an adaptive prior's first-pass map is apparent; the true-sky prior
    # follows the true sky
    np.testing.assert_allclose(reg.inner.brightness, [0.1, 0.2, 1.0])


def test_without_a_pb_nothing_changes():
    reg = fitting.make_regularization("matern", 1.0, 0.3, envelope={"floor": 0.1})
    assert not isinstance(reg, PrimaryBeamPrior)

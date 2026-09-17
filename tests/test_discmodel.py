"""Uniform disc + polar-grid surface-brightness perturbations."""

import numpy as np
import pytest

from pyuvimage.discmodel import (
    PolarGrid,
    UniformDisc,
    disc_shape_visibilities,
    guess_radius_arcsec,
    make_disc_dataset,
    make_rdor_mock,
    run_disc,
)
def test_polar_cells_tile_the_disc():
    grid = PolarGrid(radius=0.035, n_rings=5, az_oversample=1.0)
    assert np.isclose(grid.areas.sum(), np.pi * 0.035**2)
    assert grid.n_cells == sum(grid.cells_per_ring)
    assert all(n >= 1 for n in grid.cells_per_ring)
    # outer rings have more azimuthal cells
    assert grid.cells_per_ring[-1] > grid.cells_per_ring[0]


def test_uniform_polar_grid_matches_airy():
    """A polar grid filled with the disc's uniform SB reproduces the Airy V."""
    disc = UniformDisc(flux_jy=0.4, radius_arcsec=0.030, d_ra=0.002, d_dec=-0.001)
    uvd, _, _ = make_disc_dataset(
        n_vis=600, disc=disc, spots=[], sigma_jy=1e-12,
        resolution_elements=3.5, seed=3,
    )
    uv, _, _ = uvd.flattened()
    grid = PolarGrid(
        radius=disc.radius_arcsec, n_rings=5, centre=disc.centre_grid,
    )
    cols = grid.visibility_columns(uv)
    from_grid = cols @ np.full(grid.n_cells, disc.surface_brightness)
    analytic = disc.visibilities(uv)
    # midpoint quadrature of the cells; the docstring promises this round trip
    assert np.allclose(from_grid, analytic, rtol=3e-3, atol=1e-3 * np.max(np.abs(analytic)))


def test_disc_shape_at_zero_spacing_is_one():
    uv = np.array([[0.0, 0.0], [1.0, 0.0]])
    v = disc_shape_visibilities(uv, 0.03, 0.0, 0.0)
    assert v[0] == pytest.approx(1.0)


def test_guess_radius_from_first_null():
    disc = UniformDisc(flux_jy=0.35, radius_arcsec=0.032)
    uvd, _, _ = make_disc_dataset(
        n_vis=2500, disc=disc, spots=[], sigma_jy=1e-4,
        resolution_elements=6.0, seed=1,
    )
    uv, data, _ = uvd.flattened()
    guessed = guess_radius_arcsec(uv, data)
    assert guessed == pytest.approx(disc.radius_arcsec, rel=0.15)


def test_fit_disc_recovers_flux_radius_centre():
    true = UniformDisc(0.35, 0.030, d_ra=0.002, d_dec=-0.0015)
    uvd, _, _ = make_disc_dataset(
        n_vis=1200, disc=true, spots=[], sigma_jy=4e-4,
        resolution_elements=4.0, seed=4,
    )
    uv, data, noise = uvd.flattened()
    from pyuvimage.discmodel import DiscPerturbationFit

    start = UniformDisc(0.5, 0.040, 0.0, 0.0)
    fitted = DiscPerturbationFit(uv, data, noise, start).fit_disc()
    assert fitted.flux_jy == pytest.approx(true.flux_jy, rel=0.05)
    assert fitted.radius_arcsec == pytest.approx(true.radius_arcsec, rel=0.05)
    assert fitted.d_ra == pytest.approx(true.d_ra, abs=0.001)
    assert fitted.d_dec == pytest.approx(true.d_dec, abs=0.001)


def test_linear_solve_recovers_a_planted_cell():
    """The mapping matrix plus the linear solve must invert a single-cell bump."""
    from pyuvimage.discmodel import DiscPerturbationFit
    from pyuvimage.mock import random_uv_coverage, uv_of
    from pyuvimage.uvdata import C_M_S

    disc = UniformDisc(0.40, 0.035)
    freq = 330e9
    b_max = 5.0 * (C_M_S / freq) / (2.0 * disc.radius_arcsec * np.pi / (180 * 3600))
    uvw = random_uv_coverage(800, b_max, freq, seed=8)
    uv = uv_of(uvw, freq)
    grid = PolarGrid(radius=disc.radius_arcsec, n_rings=4, centre=disc.centre_grid)
    cols = grid.visibility_columns(uv)
    true_s = np.zeros(grid.n_cells)
    k = grid.n_cells // 2
    true_s[k] = 0.2 * disc.surface_brightness
    vis = disc.visibilities(uv) + cols @ true_s
    noise = np.full(vis.shape, 5e-4 + 1j * 5e-4)
    fit = DiscPerturbationFit(uv, vis, noise, disc)
    fit._build_grid(disc, n_rings=4, az_oversample=1.0, scale=None, nu=1.5,
                    reg="matern")
    s, chi2, _ = fit._solve(disc, coefficient=1e-8)
    assert np.argmax(np.abs(s)) == k
    assert s[k] == pytest.approx(true_s[k], rel=0.15)
    assert chi2 / fit.n_data < 0.05


def test_perturbations_find_a_hot_spot(tmp_path):
    disc = UniformDisc(0.40, 0.035)
    spot = (0.030, 0.015, 0.010, 0.006)   # flux, dRA, dDec, sigma
    uvd, _, _ = make_disc_dataset(
        n_vis=1500, disc=disc, spots=[spot], sigma_jy=3e-4,
        resolution_elements=5.0, seed=7,
    )
    result = run_disc(
        uvd, disc=disc, n_rings=4, coefficient=1e-4,
        n_iter=1, refit_disc=False, out=tmp_path / "disc", write=True,
    )
    # a compact Gaussian is not a polar cell, so chi^2/N will not hit 1;
    # the perturbations must still mop up most of the disc-only residual
    # and peak on the injected spot
    assert result.chi_squared < 0.1 * result.disc_chi_squared

    i = int(np.argmax(result.perturbations))
    y, x = result.grid.cell_points[i]
    d_ra, d_dec = -x, y
    assert np.hypot(d_ra - spot[1], d_dec - spot[2]) < 0.02

    for name in (
        "model.fits", "perturbation.fits", "significance.fits",
        "fit_parameters.json", "perturbations.npz", "summary.png",
    ):
        assert (tmp_path / "disc" / name).exists(), name


def test_render_paints_the_right_cell():
    grid = PolarGrid(radius=1.0, n_rings=2, az_oversample=1.0)
    values = np.arange(grid.n_cells, dtype=float)
    img = grid.render(values, (81, 81), pixel_scale=2.0 / 80)
    # pixel at the first-ring mid-radius, PA=0 (North / +y)
    r_mid = 0.5 * (0.5 + 1.0)   # second ring
    # centre pixel is (40, 40); +y is decreasing row
    cy = 40
    row = int(round(cy - (0.25 / (2.0 / 80))))   # inner-ring mid, PA=0
    col = 40
    painted = img[row, col]
    # inner ring, first azimuthal cell (theta ~ 0)
    assert painted == values[0]


def test_rdor_mock_smoke(tmp_path):
    uvd, disc, spots = make_rdor_mock(n_vis=800, sigma_jy=6e-4, seed=2)
    result = run_disc(
        uvd, disc=disc, n_rings=4, coefficient="chi2", chi2_target=1.2,
        n_iter=1, out=tmp_path / "rdor",
        truth_disc=disc, truth_spots=spots,
    )
    assert result.grid.n_cells > 10
    assert result.disc.radius_arcsec == pytest.approx(0.035, rel=0.1)
    assert len(spots) == 2
    assert result.regularization == "matern"
    assert (tmp_path / "rdor" / "truth_perturbation.fits").exists()


def test_reg_constant_and_exponential_run(tmp_path):
    uvd, disc, spots = make_rdor_mock(n_vis=600, sigma_jy=8e-4, seed=3)
    for reg in ("constant", "exponential", "gaussian"):
        result = run_disc(
            uvd, disc=disc, n_rings=3, reg=reg, coefficient=1.0,
            n_iter=1, refit_disc=False, write=False,
        )
        assert result.regularization == ("exponential" if reg == "exponential" else reg)
        if reg == "exponential":
            assert result.nu == 0.5
        assert result.chi_squared < result.disc_chi_squared

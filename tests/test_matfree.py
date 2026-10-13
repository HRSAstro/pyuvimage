"""The matrix-free solve for large meshes (`matfree`).

Every piece is checked against the dense algebra on a mesh small enough to
form F: the sparse mapping, F and D through the w-tilde FFT, the SPDE prior's
covariance, the preconditioned CG solution and the evidence -- then the path
end to end through `run`, with the switch-over lowered so a small mock takes it.
"""
import numpy as np
import pytest

import autogalaxy as ag

from pyuvimage import fitting, matfree, mock
from pyuvimage import streaming as stm


@pytest.fixture(scope="module")
def small():
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=800, mesh_n=16, fov_arcsec=3.0, seed=2)
    mask = fitting.make_mask(geom, "square")
    terms = stm.sparse_terms_for(uvd, geom, mask, ag.TransformerDFT, pool_noise=True)
    stub = stm.stub_dataset_from_terms(terms, geom, mask)
    fit = fitting.fit_at(stub, geom.mesh_shape, "constant", 1.0, positive_only=False)
    mapper = fit.inversion.linear_obj_list[0]
    data = matfree.DataOperator(terms, matfree.sparse_mapping(mapper),
                                ~np.asarray(mask.native, bool))
    order, shape, pix = matfree.mesh_grid_order(np.asarray(mapper.source_plane_mesh_grid.array))
    return dict(uvd=uvd, geom=geom, terms=terms, fit=fit, mapper=mapper, data=data,
                order=order, shape=shape, pix=pix,
                F=np.asarray(fitting.system_for(fit).F), D=np.asarray(fitting.system_for(fit).D))


def test_sparse_mapping_is_the_mapping_matrix(small):
    np.testing.assert_array_equal(matfree.sparse_mapping(small["mapper"]).toarray(),
                                  np.asarray(small["mapper"].mapping_matrix))


def test_wtilde_fft_gives_the_dense_F_and_D(small):
    data, F = small["data"], small["F"]
    v = np.random.default_rng(0).normal(size=F.shape[0])
    np.testing.assert_allclose(data.F(v), F @ v, rtol=0, atol=1e-12 * np.abs(F @ v).max())
    np.testing.assert_allclose(data.D, small["D"], rtol=0, atol=1e-12 * np.abs(small["D"]).max())
    s = np.linalg.solve(F + 1e7 * np.eye(F.shape[0]), small["D"])
    assert data.chi_squared(s) == pytest.approx(fitting.system_for(small["fit"]).chi_squared(s),
                                                rel=1e-10)


def test_spde_prior_has_unit_variance_and_matern_shape(small):
    """C = (c L^2)^-1 with unit variance at the centre and the Matern(nu=1)
    correlation c(r) = kappa r K1(kappa r) away from the edges."""
    from scipy.special import kv

    prior = matfree.SparseMaternPrior(small["order"], small["shape"], small["pix"], 0.3)
    C = np.linalg.inv(prior.Qw.toarray())
    my, mx = small["shape"]
    centre = int(np.flatnonzero(prior.order == (my // 2) * mx + mx // 2)[0])
    assert C[centre, centre] == pytest.approx(1.0, rel=1e-8)
    kappa = np.sqrt(2.0) / 0.3
    for d in (1, 2, 3):
        j = int(np.flatnonzero(prior.order == (my // 2) * mx + mx // 2 + d)[0])
        r = d * small["pix"]
        # the discrete operator approximates the continuum kernel to ~10%
        assert C[centre, j] == pytest.approx(kappa * r * kv(1, kappa * r), rel=0.15)
    # and the weighted form is W K W
    w = np.linspace(0.1, 1.0, prior.n)
    pw = matfree.SparseMaternPrior(small["order"], small["shape"], small["pix"], 0.3, weights=w)
    np.testing.assert_allclose(np.linalg.inv(pw.Qw.toarray()), w[:, None] * C * w[None, :],
                               rtol=1e-6, atol=1e-10)
    assert pw.log_det_Cinv == pytest.approx(np.linalg.slogdet(pw.Qw.toarray())[1], rel=1e-8)


@pytest.mark.parametrize("weighted", [False, True])
def test_preconditioned_cg_and_evidence_match_dense(small, weighted):
    data, F = small["data"], small["F"]
    w = None
    if weighted:
        rng = np.random.default_rng(3)
        w = 0.01 + 0.99 * rng.random(F.shape[0])
    prior = matfree.SparseMaternPrior(small["order"], small["shape"], small["pix"], 0.3, weights=w)
    system = matfree.MatrixFreeSystem(data, prior)
    for lam in (1e7, 1e9):
        sol = system.solve(lam)
        H = lam * prior.Qw.toarray()
        s = np.linalg.solve(F + H, data.D)
        np.testing.assert_allclose(sol.s, s, rtol=0, atol=1e-4 * np.abs(s).max())
        assert sol.prior_term == pytest.approx(s @ H @ s, rel=1e-4)
        lnz = -0.5 * (data.chi_squared(s) + s @ H @ s
                      + np.linalg.slogdet(F + H)[1] - np.linalg.slogdet(H)[1]
                      + data.noise_normalization)
        # stochastic Lanczos quadrature: ~0.2% of ln Z on this mock
        assert system.log_evidence(sol) == pytest.approx(lnz, rel=3e-3)


def test_run_takes_the_matrix_free_path_above_the_switch(small, tmp_path, monkeypatch, caplog):
    import logging

    from pyuvimage import api

    monkeypatch.setattr(matfree, "MATRIX_FREE_MIN_MESH_PIXELS", 100)
    with caplog.at_level(logging.INFO, logger="pyuvimage"):
        res = api.run(small["uvd"], fov=3.0, mesh_shape=(16, 16), out=tmp_path / "mf",
                      reg="adaptive", streaming=True, pb_correction=False,
                      uncertainty="statistical")
    assert "matrix-free solve" in caplog.text
    assert "not available on the matrix-free path" in caplog.text     # uncertainty
    p = res.products[0]
    assert np.isfinite(p.model_image).all() and p.uncertainty is None
    assert res.parameters["fit_quality"]["chi_squared"] / (2 * small["uvd"].n_samples) < 1.5
    assert res.scan["solver"] == "matrix-free"
    assert (tmp_path / "mf" / "model.fits").exists()
    with pytest.raises(NotImplementedError, match="point components"):
        api.run(small["uvd"], fov=3.0, mesh_shape=(16, 16), out=tmp_path / "p",
                streaming=True, point_sources=True, write=False)


@pytest.mark.parametrize("kind", ["density", "radial"])
def test_fourier_preconditioners_agree_with_the_sparse_one(small, kind):
    """The uv-density preconditioners change the route, not the answer;
    weighted priors keep the sparse P, which follows the weights."""
    data = small["data"]
    plain = matfree.SparseMaternPrior(small["order"], small["shape"], small["pix"], 0.3)
    w = 0.01 + 0.99 * np.random.default_rng(4).random(data.n_mesh)
    weighted = matfree.SparseMaternPrior(small["order"], small["shape"], small["pix"], 0.3,
                                         weights=w)
    assert matfree.MatrixFreeSystem(data, plain).preconditioner == "density"
    assert matfree.MatrixFreeSystem(data, weighted).preconditioner == "sparse"
    radial = matfree.MatrixFreeSystem(data, plain, preconditioner=kind)
    sparse = matfree.MatrixFreeSystem(data, plain, preconditioner="sparse")
    for lam in (1e7, 1e9):
        a, b = radial.solve(lam), sparse.solve(lam)
        np.testing.assert_allclose(a.s, b.s, rtol=0, atol=1e-4 * np.abs(b.s).max())   # CG tolerance
    # the measured density is non-negative and the prior symbol matches C^-1
    rho, q = radial._spectra()
    assert (rho >= 0).all()
    my, mx = small["shape"]
    e = np.zeros((my, mx)); e[my // 2, mx // 2] = 1.0
    col = radial._to_grid(plain.Qw @ radial._from_grid(e))
    stencil = np.zeros(q.shape)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            stencil[dy, dx] = col[my // 2 + dy, mx // 2 + dx]
    np.testing.assert_allclose(np.real(np.fft.fft2(stencil)), q, rtol=1e-10, atol=1e-10 * q.max())


def test_pb_in_model_reaches_the_matrix_free_prior(small, tmp_path, monkeypatch):
    """--pb-in-model puts the prior on the true sky: on this path, prior
    weights times the PB. It used to be dropped silently."""
    from pyuvimage import api

    monkeypatch.setattr(matfree, "MATRIX_FREE_MIN_MESH_PIXELS", 100)
    seen = []
    orig = matfree.SparseMaternPrior.__init__

    def spy(self, order, mesh_shape, pixel_scale, scale, weights=None):
        seen.append(None if weights is None else np.asarray(weights).copy())
        orig(self, order, mesh_shape, pixel_scale, scale, weights=weights)

    monkeypatch.setattr(matfree.SparseMaternPrior, "__init__", spy)
    api.run(small["uvd"], fov=3.0, mesh_shape=(16, 16), out=tmp_path / "pb", reg="matern",
            streaming=True, pb_correction=False, pb_in_model=True, dish_diameter=12.0,
            write=False)
    assert seen and all(w is not None for w in seen)
    w = seen[-1]
    assert w.max() == pytest.approx(1.0, abs=0.01) and w.min() < w.max()

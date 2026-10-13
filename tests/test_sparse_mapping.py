"""The mesh -> image mapping is kept sparse (`fitting.mesh_mapping`).

autoarray's `mapping_matrix` is dense, (n_image, n_mesh): 1.3 GB at a 48^2
mesh on a 384^2 image, and the sparse inversion's data vector built it twice.
These pin that the sparse replacements give the same numbers.
"""
import numpy as np
import autogalaxy as ag

from pyuvimage import fitting, mock
from pyuvimage import streaming as stm


def _fit():
    uvd, _, geom, _ = mock.make_demo_dataset(n_vis=600, mesh_n=12, fov_arcsec=3.0, seed=4)
    mask = fitting.make_mask(geom, "square")
    terms = stm.sparse_terms_for(uvd, geom, mask, ag.TransformerDFT, pool_noise=True)
    stub = stm.stub_dataset_from_terms(terms, geom, mask)
    return terms, fitting.fit_at(stub, geom.mesh_shape, "matern", 1e8, positive_only=False,
                                 reg_scale=0.3)


def test_sparse_mapping_equals_the_dense_one_and_is_cached():
    _, fit = _fit()
    mapper = fit.inversion.linear_obj_list[0]
    M = fitting.mesh_mapping(mapper)
    np.testing.assert_array_equal(M.toarray(), np.asarray(mapper.mapping_matrix))
    assert fitting.mesh_mapping(mapper) is M


def test_sparse_data_vector_is_the_dense_product():
    terms, fit = _fit()
    mapper = fit.inversion.linear_obj_list[0]
    dense = np.asarray(mapper.mapping_matrix).T @ np.asarray(
        fit.inversion.dataset.sparse_operator.dirty_image)
    np.testing.assert_allclose(np.asarray(fit.inversion.data_vector), dense, rtol=1e-12, atol=0)


def test_diag_quadratic_matches_einsum():
    _, fit = _fit()
    mapper = fit.inversion.linear_obj_list[0]
    M = fitting.mesh_mapping(mapper)
    n = M.shape[1]
    A = np.random.default_rng(0).normal(size=(n, n))
    C = A @ A.T
    D = np.asarray(mapper.mapping_matrix)
    np.testing.assert_allclose(fitting.diag_quadratic(M, C),
                               np.einsum("ij,jk,ik->i", D, C, D), rtol=1e-10)

"""Unit tests for sky SNR routing and LLWR helpers."""

import numpy as np

from src.deconv.sky_reg import (
    choose_sky_reconstructor,
    llwr_from_hamiltonian,
    llwr_from_terms,
    log_evidence_approx_from_terms,
    noise_normalization_from_sigma,
)


def test_choose_sky_reconstructor_threshold():
    assert choose_sky_reconstructor(100.0, 50.0) == "log_sky"
    assert choose_sky_reconstructor(50.0, 50.0) == "log_sky"
    assert choose_sky_reconstructor(49.9, 50.0) == "linear_sky"


def test_llwr_matches_negative_hamiltonian():
    chi2 = 10.0
    smooth_term = 2.0
    edge_term = 0.5
    smooth = 3.0
    edge_prior = 4.0
    noise_norm = 1.0
    H = 0.5 * chi2 + smooth * smooth_term + edge_prior * edge_term
    llwr = llwr_from_terms(chi2, smooth_term, edge_term, smooth, edge_prior, noise_norm)
    assert np.isclose(llwr, -H - 0.5 * noise_norm)
    assert np.isclose(llwr, llwr_from_hamiltonian(H, noise_norm))


def test_evidence_approx_penalizes_tiny_smooth():
    kwargs = dict(
        chi2=1.0,
        smooth_term=1.0,
        edge_term=0.0,
        edge_prior=0.0,
        n_free=10000,
        n_data=400,
        noise_norm=0.0,
    )
    low = log_evidence_approx_from_terms(smooth=1e-6, **kwargs)
    mid = log_evidence_approx_from_terms(smooth=1e3, **kwargs)
    # Bare LLWR prefers low smooth; Occam term should reverse that for equal chi2.
    assert mid > low


def test_noise_normalization_matches_autolens_form():
    sigma = np.array([0.1, 0.2, 0.3])
    n = noise_normalization_from_sigma(sigma)
    expected = 2.0 * np.sum(np.log(2.0 * np.pi * sigma * sigma))
    assert np.isclose(n, expected)

"""The batched nufft2d2 gather buffer, which is what actually runs out of memory.

Ruby at a 20x20 mesh was killed on a machine that had just been told the fit
needed 3.0 GB and had 6.4 GB. Both halves of that sentence were true and the
estimate was still wrong by two orders of magnitude, because it costed the
mapping matrix (0.5 GB) and not the thing that transforms it: one batched
`nufft2d2` over all 400 mesh pixels, whose gather buffer in nufftax is
`n_mesh x n_vis x nspread^2` complex128 -- 186 GB, materialised, unfused.

These tests pin the arithmetic and the choice it drives, so that a fit which
cannot fit is diverted or split rather than killed.
"""

import logging

import numpy as np
import pytest

import autogalaxy as ag

from pyuvimage import fitting


# ---------------------------------------------------------------- the kernel

@pytest.mark.parametrize(
    "eps, width",
    [
        (1e-12, 14),   # autoarray's default: the widest kernel in practice
        (1e-9, 10),
        (1e-6, 8),
        (1e-1, 2),     # floor
        (1e-30, 16),   # cap
    ],
)
def test_kernel_width_follows_nufftax_own_heuristic(eps, width):
    """`ceil(log10(1/eps) + 1)`, rounded up to even, clamped to [2, 16]."""
    assert fitting.nufftax_kernel_width(eps) == width


def test_the_kernel_width_is_always_even():
    """nufftax rounds up for symmetry, and nspread^2 is what we are costing."""
    for eps in (1e-3, 1e-5, 1e-7, 1e-11):
        assert fitting.nufftax_kernel_width(eps) % 2 == 0


# ------------------------------------------------------------- the buffer

def test_the_gather_buffer_dwarfs_the_mapping_matrix_it_transforms():
    """The regression, in the one case that was actually killed."""
    n_vis, n_mesh = 148_477, 400
    matrix = fitting.estimate_peak_memory_gb(n_vis, n_mesh)
    gather = fitting.nufftax_gather_gb(n_vis, n_mesh)
    assert matrix < 4.0            # what the old estimate reported: 3.0 GB
    assert gather > 100.0          # what it actually needed
    assert gather / matrix > 50


def test_the_buffer_scales_with_every_one_of_its_three_factors():
    base = fitting.nufftax_gather_gb(10_000, 100)
    assert fitting.nufftax_gather_gb(20_000, 100) == pytest.approx(2 * base)
    assert fitting.nufftax_gather_gb(10_000, 200) == pytest.approx(2 * base)
    # width enters squared: 14 -> 8 taps is (8/14)^2
    coarse = fitting.nufftax_gather_gb(10_000, 100, eps=1e-6)
    assert coarse / base == pytest.approx((8 / 14) ** 2)


# -------------------------------------------------------------- the blocking

def test_no_blocking_when_the_whole_stack_already_fits():
    """Small problems must keep using the single upstream call unchanged."""
    n = fitting.nufftax_block_columns(500, 40, available_gb=64.0)
    assert n == 40


def test_blocking_never_returns_zero_columns():
    """A block of zero would be an infinite loop; one column is the floor even
    when a single column does not fit either."""
    assert fitting.nufftax_block_columns(10**7, 4000, available_gb=0.5) == 1


def test_blocking_respects_the_budget():
    n_vis, n_mesh, have = 148_477, 400, 32.0
    block = fitting.nufftax_block_columns(n_vis, n_mesh, available_gb=have)
    assert 1 <= block <= n_mesh
    assert (
        fitting.nufftax_gather_gb(n_vis, block)
        <= fitting.NUFFTAX_GATHER_BUDGET * have + 1e-9
    )


def test_unknown_memory_does_not_block(monkeypatch):
    """A memory figure we cannot obtain must never stop a fit that would have
    worked -- same rule as `available_memory_gb` itself."""
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: None)
    assert fitting.nufftax_block_columns(10**6, 900) == 900


# ------------------------------------------------------- what gets chosen

@pytest.fixture
def jax_and_pynufft(monkeypatch):
    """Both backends installed, so the choice is made on memory alone."""
    sentinel = type("TransformerPyNUFFTStub", (), {})
    monkeypatch.setattr(fitting, "jax_available", lambda: True)
    monkeypatch.setattr(fitting, "pynufft_available", lambda: True)
    monkeypatch.setattr(fitting, "pynufft_transformer_class", lambda: sentinel)
    return sentinel


def test_pynufft_is_preferred_when_the_batched_gather_will_not_fit(
    jax_and_pynufft, monkeypatch, caplog
):
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 6.4)
    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        cls = fitting.resolve_transformer(
            n_vis=148_477, n_image_pixels=1600, n_mesh_pixels=400
        )
    assert cls is jax_and_pynufft
    assert "gather buffer" in caplog.text


def test_the_jax_nufft_is_kept_when_it_does_fit(jax_and_pynufft, monkeypatch):
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 2000.0)
    cls = fitting.resolve_transformer(
        n_vis=148_477, n_image_pixels=1600, n_mesh_pixels=400
    )
    assert cls is ag.TransformerNUFFT


def test_without_a_mesh_size_the_old_behaviour_is_unchanged(
    jax_and_pynufft, monkeypatch
):
    """Callers that cannot say how big the mesh is get what they always got."""
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 6.4)
    cls = fitting.resolve_transformer(n_vis=148_477, n_image_pixels=1600)
    assert cls is ag.TransformerNUFFT


def test_forcing_nufft_splits_rather_than_dies(monkeypatch):
    """`--transformer nufft` is a deliberate choice, so it is honoured -- but
    with the mapping-matrix transform chunked, not killed."""
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 6.4)
    cls = fitting.resolve_transformer(
        n_vis=148_477, transformer="nufft", n_image_pixels=1600,
        n_mesh_pixels=400,
    )
    assert issubclass(cls, ag.TransformerNUFFT)
    assert cls is not ag.TransformerNUFFT
    assert hasattr(cls, "transform_mapping_matrix")


def test_an_already_resolved_class_passes_straight_through():
    assert fitting.resolve_transformer(
        n_vis=10, transformer=ag.TransformerDFT
    ) is ag.TransformerDFT


# ------------------------------------------------------- what gets reported

def test_the_reported_figure_includes_the_transformer(monkeypatch, caplog):
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 6.4)
    with caplog.at_level(logging.INFO, logger="pyuvimage.fitting"):
        fitting.check_memory(148_477, 400, ag.TransformerNUFFT)
    assert "may be killed outright" in caplog.text


def test_a_transformer_with_no_buffer_of_its_own_adds_nothing():
    for cls in (None, ag.TransformerDFT):
        assert fitting.transformer_memory_gb(cls, 148_477, 400) == 0.0


def test_chunking_is_costed_at_the_block_size_not_the_whole_stack(monkeypatch):
    monkeypatch.setattr(fitting, "available_memory_gb", lambda: 6.4)
    chunked = fitting.chunked_nufft_transformer_class()
    whole = fitting.transformer_memory_gb(ag.TransformerNUFFT, 148_477, 400)
    split = fitting.transformer_memory_gb(chunked, 148_477, 400)
    assert split < whole / 100

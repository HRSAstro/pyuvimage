# Large meshes: the matrix-free solve

[← back to the README](../README.md)

Above **100 × 100 mesh pixels** (`matfree.MATRIX_FREE_MIN_MESH_PIXELS`) the
streamed path stops forming the `n_mesh × n_mesh` matrix F + H and solves
without it, after Powell et al. (2021, MNRAS 501, 515).

**Why.** The dense path stores F + H (8 n_mesh² bytes: 11 GB at 192 × 192)
and factorises it (~n_mesh³/3 flops: 2×10¹³ at 192 × 192) at every trial,
and the non-negative solver refactorises many times per solve. On NGC 4274
(`--fov 35`, an automatic 192 × 192 mesh, 2.6 M visibilities) the first
non-negative solve alone extrapolated to days on 2 cores. The matrix-free fit
of the same data takes **78 s on 2 cores with a 0.9 GB peak**.

**How.**

| piece | dense path | matrix-free path |
|---|---|---|
| data term F s | F = Mᵀ W̃ M formed | Mᵀ W̃ (M s): the w-tilde convolution, one FFT pair on the padded image grid; M the mapper's sparse interpolation matrix |
| prior | Matérn ν = 1.5, C⁻¹ dense | Matérn ν = 1 in its sparse-precision (SPDE) form, C⁻¹ = W⁻¹ c L² W⁻¹, L = κ² − ∇² (Lindgren et al. 2011) |
| solve | Cholesky / NNLS | conjugate gradients; preconditioner below |
| ln det H | Cholesky | exact, from the sparse factor of L |
| ln det(F + H) | Cholesky | ln det P exactly + ln det(P⁻¹A) by stochastic Lanczos quadrature (10 fixed probes, ≤40 steps) |
| criteria | all | `structure`, `discrepancy`, `evidence`; the adaptive prior's two passes as on the dense path |

**The preconditioner.** Two, chosen by the prior:

- *Unweighted prior* (plain Matérn, exponential, the adaptive prior's first
  pass): the **uv weight density**. On the mesh, F is nearly a convolution by
  the dirty beam, so in Fourier space it multiplies by the gridded uv weight
  density ρ(k); the SPDE precision is a convolution too. ρ is measured once
  (the beam at every lag, two matrix-vector products), smoothed by one uv
  cell, and (ρ + λ q)⁻¹ is applied by FFT. No factorisation is needed.
- *Weighted prior* (the adaptive second pass, the envelope): Powell et al.'s
  sparse **P = κ₀ MᵀM + λC⁻¹** (W̃ replaced by its peak κ₀), factorised once
  per λ. It follows the per-pixel weights exactly, which a Fourier
  preconditioner cannot: with adaptive weights it needs ~3× fewer
  iterations.

Measured (iterations per solve):

| case | sparse P | uv density | ring average ("radial") |
|---|---|---|---|
| mock, 96², λ = 1e8 / 1e9 | 790 / 310 | 572 / 200 | 572 / 214 |
| NGC 4274, 192², λ = 1e10 / 1e11 | 185 / 62 | 275 / 60 | 502 / 99 |
| NGC 4274, whole fit | 78 s | 79 s | 90 s |

The density preconditioner is also ~2× cheaper per iteration. The ring
average (density by baseline length alone) loses on NGC 4274 because its
coverage is elongated, not round; a Gaussian radial model does worse still.
`matfree.UNWEIGHTED_PRECONDITIONER` selects `"density"` (default),
`"radial"` or `"sparse"`. The evidence always uses the sparse P, whose
determinant is exact.

**What sets the cost.** The iterations depend on how strong the chosen prior
is relative to the data: roughly λ^-0.4 at fixed data. A bright,
high-S/N source picks a weak prior, and then the uneven uv coverage sets the
conditioning; a faint one picks a strong prior, which dominates most modes
and makes the system easy. The prior-to-data ratio per pixel, λ/κ₀, is ~300
for NGC 4274 (23–270 iterations per solve at 192 × 192) and ~0.03 for the
benchmark mock (thousands at 192 × 192, `scripts/bench_solvers.py`). More
visibilities per uv cell help at a given ratio.

F and D agree with the dense path to 1e-16, and the solve and evidence agree
with a dense solve of the same system to 1e-4 and ~0.2%
(`tests/test_matfree.py`). Powell et al. stopped at ln det P, which they
report is 1–20% off; the Lanczos correction removes most of that.

**What differs from the dense path.**

- **The prior's smoothness is ν = 1, not 1.5.** ν = 1.5 has no sparse
  precision. The correlation length is still the beam, and the variance is
  normalised to 1 at the mesh centre, so the coefficient means roughly the
  same thing on both paths. The SPDE's edges are reflecting: the prior
  variance rises within about one correlation length of the mesh edge (to
  ~2× at a corner), where the field of view should hold no emission.
- **No positivity.** The solve is unconstrained, so the model can dip
  slightly negative.
- **Not available:** point components (refused), and the uncertainty
  products (`--uncertainty` is set to none with a warning), which need
  diag((F + H)⁻¹). `--reg gibbs` is refused; `matern`, `exponential`
  (treated as ν = 1 too), `gaussian` and `adaptive` are supported.
- **In-memory runs** (not streamed) do not switch; they warn that a mesh
  this size will be slow and suggest `--streaming` or a coarser `--mesh`.

Cube mode streams as before; each channel's fit at the frozen prior is one
preconditioned CG solve.

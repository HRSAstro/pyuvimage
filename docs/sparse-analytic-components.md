# Point components on the sparse path

**Status:** unblocked upstream, not yet wired up here.

An issue was drafted for PyAutoArray about this and never filed — they had
already implemented it. This note keeps what we need in order to use it.

## What was missing

`InterferometerSparseOperator` provided only `curvature_matrix_diag_from`,
against the imaging operator's three methods (`..._diag_from`,
`..._off_diag_from`, `..._off_diag_func_list_from`), and
`interferometer/sparse.py` had no dispatch on model contents — the string
`AbstractLinearObjFuncList` did not appear in the file. So a `Mapper` plus any
analytic linear object had no route through the sparse path, and our only way
to the cross-terms was `operated_mapping_matrix`, which builds the dense
`[n_vis, n_mesh]` matrix the w-tilde formalism exists to avoid (21.6 GB on
Ruby CO(7-6), against a 0.10 MB kernel).

## What upstream shipped

`InterferometerSparseOperator` gained the mapper–mapper off-diagonal,
mapper–function and function–function block methods, and
`InversionInterferometerSparse.curvature_matrix` now assembles the full matrix
for any mix of linear function lists and mappers. It matches the dense
`InversionInterferometerMapping` path to ~1e-16 on their test cases. Expected
on PyPI 29 Aug 2026.

## The one thing to get right when we wire it up

**The linear-function columns go in un-weighted.**

The interferometer operator is `W̃ = Re(Fᴴ W F)` — it already carries the
inverse-variance weighting. The imaging operator is split as `Hᵀ N⁻¹ H`, so
its `curvature_weights` argument expects `(H B) / noise²` pre-divided. Porting
the imaging call pattern across would double-weight everything.

This is a live trap for us specifically, because our own bordered system does
pre-weight. `pointsource.PointExtendedSystem._curv` is

```python
def _curv(self, P, Q):
    return P.real.T @ (self.w_re[:, None] * Q.real) + P.imag.T @ (
        self.w_im[:, None] * Q.imag
    )
```

with `w_re = 1/sigma_re**2`. Handing those same weighted columns to the sparse
operator would apply `1/sigma²` twice. On the operator path, `P` goes in raw.

## Its other precondition

The `W̃` reduction assumes `sigma_re == sigma_im`, and unequal sigmas degrade
sparse-vs-dense agreement generally, not just for these new blocks. `W̃` is
accumulated from `noise_map_real` alone while the data vector weights the real
and imaginary parts separately, so `F` and `D` end up on different weightings.

`api.run` pools the two in quadrature before every sparse fit
(`uvdata.with_pooled_noise`), which removes the discrepancy and is
independently the better estimate — twice the sample size, total variance
unchanged — so the assumption is satisfied exactly by the time the kernel is
built. `--image-centre` does the same on every recentred fit, for the same
reason: a phase rotation mixes the real and imaginary parts.

How far apart they were changes only what is said. Below
`uvdata.REIM_ASYMMETRY_WARN` (25%) it is scatter in the estimator and the
pooling is logged; above it the difference may be real, pooling then weights
the two parts equally when the noise map says otherwise, and
`uvdata.describe_pooling` returns a warning naming `--inversion dense` as the
path that makes no such assumption. It is deliberately not a fallback:
refusing sparse sends a large dataset onto the dense mapping matrix — tens of
GB and hours — over a judgement the user is better placed to make, and the two
thresholds tried before this (5%, then 25%) each had real data arrive above
them. Measured unrecentred: Ruby 9%, 9io9 15.6%.

## What has to change here

1. Drop the `point_sources` guard in `api.run` once the release is in.
2. Decide whether our bordered system (`PointExtendedSystem`) stays as the
   dense-path implementation with the operator used only for `--inversion
   sparse`, or whether the framework's own assembly replaces it on both. The
   bordered system does things the framework's does not — the Schur-complement
   scan over trial positions, the amplitude covariance, the regularisation
   retune — so it is not a straight swap.
3. Whichever way, remove the pre-weighting on any column handed to the
   operator.
4. Re-check `--inversion auto`: on the sparse path `transform_mapping_matrix`
   is never called, so the nufftax gather buffer that makes `auto` reject the
   JAX NUFFT does not arise, and upstream pairs the sparse operator with
   `TransformerDFT` or `TransformerNUFFT`, never pynufft.

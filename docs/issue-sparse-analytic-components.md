# Sparse interferometer inversion cannot combine linear function lists with a Mapper (the imaging one can)

**Repo:** PyAutoArray
**Version checked:** 2026.8.17.1 (also present in 2026.8.23.1)

## Summary

On the **imaging** side, `InversionImagingSparse` supports an analytic linear
component (an `AbstractLinearObjFuncList`, e.g. linear light profiles) fitted
simultaneously with a pixelized `Mapper`. On the **interferometer** side,
`InversionInterferometerSparse` does not. There is no equivalent code path,
and we believe a model combining the two cannot currently be fitted.

We would like to fit an analytic component alongside the pixelization while
keeping the w-tilde formalism's memory behaviour, and at the moment the only
way to get the cross-terms is `operated_mapping_matrix`, which builds the
dense `[n_vis, n_mesh]` matrix the sparse path exists to avoid (21.6 GB on one
of our datasets, against a 0.10 MB kernel).

## The asymmetry

`ImagingSparseOperator` provides:

| method | purpose |
|---|---|
| `curvature_matrix_diag_from` | mapper block |
| `curvature_matrix_off_diag_from` | mapper × mapper |
| `curvature_matrix_off_diag_func_list_from` | **mapper × linear function list** |

`InterferometerSparseOperator` provides only `curvature_matrix_diag_from`.

Correspondingly, `imaging/sparse.py` dispatches on what the model contains:

```python
if self.has(cls=AbstractLinearObjFuncList):
    curvature_matrix = self._curvature_matrix_func_list_and_mapper
elif self.total(cls=Mapper) == 1:
    curvature_matrix = self._curvature_matrix_x1_mapper
else:
    curvature_matrix = self._curvature_matrix_multi_mapper
```

while `interferometer/sparse.py` has no such branch — the string
`AbstractLinearObjFuncList` does not appear in the file at all. Its
`curvature_matrix` returns `curvature_matrix_diag`, which is built from a
single mapper:

```python
mapper = self.cls_list_from(cls=Mapper)[0]
...
return self.dataset.sparse_operator.curvature_matrix_diag_from(
    rows=rows, cols=cols, vals=vals, S=mapper.params
)
```

## Expected failure mode

We have read the code but not run this case (the sparse operator needs JAX and
our test environment has none), so this part is inference rather than a
traceback:

* `data_vector` is `mapping_matrix.T @ dirty_image`, and
  `AbstractInversion.mapping_matrix` is an `hstack` over **all** linear
  objects — so `D` has length `S_total`.
* `curvature_matrix` is `(S_mapper, S_mapper)`.
* `regularization_matrix` is `block_diag` over **all** linear objects — so
  `(S_total, S_total)`.
* `curvature_reg_matrix` is `xp.add(curvature_matrix, regularization_matrix)`.

With any non-mapper linear object present, `S_total > S_mapper` and that
addition should raise a broadcasting error. A shape error is at least loud; we
mention it mainly because it suggests the combination was never intended to be
reachable rather than that it silently misbehaves.

## What we are asking for

The interferometer equivalents of the two missing operator methods —
principally `curvature_matrix_off_diag_func_list_from` — plus the dispatch in
`InversionInterferometerSparse.curvature_matrix` that uses them.

We think the terms are all available without any dense matrix. The
w-tilde kernel `W̃(Δy, Δx) = Σ_k w_k cos(Δx·ku_k + Δy·kv_k)` is a continuous
function of the offset, and the mapper × analytic cross-term is
`B_j = Σ_i L_ij · W̃(y_i − y_p, x_i − x_p)`, i.e. the same accumulation
`nufft_precision_operator_via_np_from` already performs, evaluated from the
analytic component's position rather than from the grid corners. The self- and
data-terms are direct sums over visibilities. So the cost should be one extra
streaming pass, with the same `n_image_pixels × chunk_k` memory and no
dependence on `n_vis`.

Two caveats on that sketch: it assumes equal real and imaginary weights, which
the existing kernel build already assumes (it takes `noise_map_real` only) but
which is not documented at the call site; and we have not implemented it. If
we have missed an existing route to this, we would much rather be told.

## Context

pyuvimage — an ALMA imager built on autogalaxy with the lens equation off. We
fit analytic point components alongside a rectangular mesh and would like to
keep them when using `apply_sparse_operator`.

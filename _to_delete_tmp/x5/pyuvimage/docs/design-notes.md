# Design notes and things that bite

[← back to the README](../README.md)

## Why the model mesh is coarser than the product grid

Every product is written on one grid at one pixel scale, but the model *mesh*
is deliberately coarser than it (`oversample`, default 2). That is not
cosmetic. If the mesh spans the same grid the residual dirty image is computed
on, the normal equations force

    A^T W r = H s

so the residual map stops measuring the data misfit and becomes the prior's
pull — collapsing towards zero exactly where the prior is weak. A blank
residual then looks like a perfect fit when it is really a fit with nothing
holding it back. Keeping the product grid finer than the mesh leaves residual
power the model cannot absorb, so the map is diagnostic again. There is a
regression test for this.

Note the model image is *not* the mesh values repeated over blocks: the
rectangular mesh mapper interpolates between mesh pixel centres, so the model
is smooth. `model.fits` is built from each linear object's mapping matrix,
which reproduces the fitted visibilities to ~1e-15; block-replicating the
reconstruction instead differs by up to 45% per pixel.

## Reading the residual map

Two separate things get confused here, and the demo showed both.

**A residual quoted in sigma means nothing on its own.** A regularised model
under-fits a bright compact peak by some *fraction* of that peak, and a
fraction of a very bright peak is a lot of sigma. The demo's source peaks at
~1100 sigma, so a 4 sigma residual is an 0.4% error. Every run prints this:

```
peak brightness 0.00632 Jy/beam = 1095 sigma; largest residual 4.0 sigma
  = 0.37% of the peak (dynamic range 274:1)
```

and `summary.png` states it in the residual panel's title. CLEAN behaves the
same way.

**But a residual that does not scale with the source is a different problem.**
Run the same source at 0.05, 0.005 and 0.0005 Jy and look at the peak residual
*within one beam of the centre*:

| source flux | peak brightness | `gibbs` | `adaptive` (default) |
|---|---|---|---|
| 0.05 Jy | 1095 sigma | 10.0 sigma | **0.8 sigma** |
| 0.005 Jy | 108 sigma | 9.3 sigma | **1.1 sigma** |
| 0.0005 Jy | 9 sigma | 4.6 sigma | **1.3 sigma** |

A pure dynamic-range effect would fall by 10x with the source. The `gibbs`
central residual does not: it is nearly flat at ~9-10 sigma across a factor of
10 in brightness, so it is not a dynamic-range floor at all — it is the prior
suppressing the centre. `adaptive` removes it. This is the measurement behind
making `adaptive` the default.

Under `adaptive` the demo's residual is 4.0 sigma (0.37% of the peak) and the
central beam contributes 0.8 sigma of it, so what is left really is
dynamic-range-limited.

If you do see a compact residual that refuses to scale down, the two things
worth trying are `--reg adaptive` (if you moved off it) and `--point-sources`,
which takes sub-beam flux out of the mesh entirely.

## Known issue: the mask edge

On the mocks the largest residual is usually at the circular mask boundary
(~5 sigma), not at the source: edge mesh pixels are poorly constrained and
absorb flux. Judge a fit from the interior, and treat edge features with
suspicion. `Settings(use_edge_zeroed_pixels=True)` upstream is the likely fix
and is not yet wired in.

## A caveat on under-constrained fits

If the model has more pixels than data points, the fit is under-constrained:
faint structure is set by the prior rather than the data, and the residual map
can sit below the noise even at `chi^2/N = 1` (a few visibilities carry the
whole `chi^2` budget while the rest are overfitted). pyuvimage warns when this
happens. It is not a concern for real datasets, where visibilities greatly
outnumber pixels, but it does mean sparse or heavily-averaged data should be
read with care — check `residual.fits` and `prior_scan.json`.

In that regime the two criteria also pull apart, and which is preferable is
data-dependent: `discrepancy` refuses to fit below the noise (so sidelobes are
not absorbed) but can over-smooth a bright compact core to get there, while
`evidence` preserves the core but, with far more pixels than data, may drive
chi^2 towards zero. If a fit looks over-smoothed at the peak, try
`--criterion evidence`; if it looks like it contains beam sidelobes, stay on
`discrepancy`. With visibilities outnumbering pixels the two agree.

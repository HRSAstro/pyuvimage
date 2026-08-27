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

## When chi^2 = N cannot be reached

The default criterion picks the strongest smoothing that still fits to the
noise level, `chi^2 = N`. That assumes `chi^2 = N` is reachable. On PJ0116 at
245 GHz it is not: with positivity on, the constrained fit floors at
`chi^2/N = 1.024` no matter how weak the prior is, while the *unconstrained*
solve used by the hyperparameter search gets to 1.000.

Chasing a target that cannot be met is not a near miss. Every bisection trial
reads "still too high", so the search walks the coefficient all the way down
to its lower bound, switches the prior off, and delivers the noisiest model
available. That is what produced this, and it is not subtle in the image:

| | coefficient | chi^2/N | structure ratio | flux | residual |
|---|---|---|---|---|---|
| chasing chi^2 = N | 7.2e4 | 1.036 | **0.73** | 55.6 mJy | 3.9 sigma |
| aiming at the floor | 3.6e7 | 1.079 | **0.98** | 53.4 mJy | 3.8 sigma |
| `--criterion structure` | 3.6e7 | 1.083 | **1.00** | 53.5 mJy | 3.9 sigma |

Three decades of coefficient, and `chi^2/N` moves by 0.004. It is nearly flat
in exactly the region where the answer changes, which is why it could not
choose between them; the residual *map* separates them cleanly.

Two things follow, both implemented:

**The floor is measured on the solver that will deliver the fit.** The
reachability probe used to run unconstrained, which hid the constrained
floor entirely. When the floor turns out to be above the target but within
`CHI2_UNREACHABLE_FACTOR` (1.3x) of it, the target is raised to
`floor x (1 + CHI2_FLOOR_TOLERANCE)` — the strongest prior that still fits
essentially as well as this model and solver can. A floor *far* above the
target is a different failure (the model genuinely cannot reproduce the data)
and still falls back to maximum evidence.

**`--criterion structure` selects on the residual map instead.** It bisects
the coefficient until the residual map's rms equals what white noise of the
same total power would give (see "Reading the residual map"). Because
positivity changes the residual map, that search runs on the constrained
solver throughout, which makes it ~30% slower than the default. On PJ0116 the
two criteria agree to within the bisection resolution — which is the reason
`discrepancy` stays the default and `structure` is the second opinion.

**Do not use `--criterion structure` on a weakly constrained fit.** The
structure ratio is only calibrated when the residual map really would be white
at `chi^2 = N`, and on a small mock it is not: the demo dataset (400 data
points, 144 mesh pixels) sits at ratio **0.49** with `chi^2/N = 0.999`, for the
reason in "A caveat on under-constrained fits" below. Driving that ratio to 1
then over-smooths to `chi^2/N = 1.59`, which the run flags as a model that does
not reproduce the data. Use it where the data outnumber the model comfortably
and `chi^2` has gone flat — which is the real-data case it was built for.

## Large datasets: what actually runs out

PJ0116 has 5,158 visibility samples. Two later ALMA datasets have 164,262 and
148,477 — about 30× — and they hit two separate walls.

**The transform.** The direct DFT allocates an `n_pixels × n_vis` float64
temporary, so its cost is the *product*. 164k visibilities over a 116×116
image is 16.5 GB and numpy raises `MemoryError` before computing anything. A
NUFFT does the same transform in 20 ms. There are two backends:

- `nufftax`, JAX-native, what `TransformerNUFFT` uses. Fast and
  differentiable, but it needs a working JAX, which is exactly what
  `_jax_guard.py` exists to survive the absence of.
- `pynufft`, pure NumPy/SciPy. `pip install pynufft` and nothing else.
  `--transformer pynufft` selects it and `auto` reaches for it when JAX is
  missing. The transformer is **vendored** into `fitting.py` (from
  autoarray's MIT-licensed `TransformerNUFFTPyNUFFT`, with the half-pixel fix
  below): PyAutoArray PR #475 deleted that class outright in 2026.8.23.1, and
  an earlier pyuvimage subclassed it at import time — so one
  `pip install -U autoarray` broke the package before it could import.
  Vendoring the ~60 lines keeps the no-JAX NUFFT path alive on every
  autoarray version; it is built lazily, so importing pyuvimage never
  requires pynufft either.

**A half-pixel bug in the pynufft path.** `autoarray.TransformerNUFFTPyNUFFT`
builds `self.shift` — the phase ramp that aligns its grid convention with the
DFT's — in `__init__` and never applies it, so the two transformers in the
same library place the image grid half a pixel apart in both axes. Measured on
the demo mock, the disagreement grows linearly with baseline length exactly as
a phase error must: 9% of the visibility rms at the shortest baselines, 38% at
the longest, and multiplying by `shift` once recovers the DFT to 1e-5.

A fit built entirely on the uncorrected transformer still converges — the
mapping matrix and the model visibilities share the offset — it just puts the
sky half a pixel from where every DFT-computed product puts it. The amplitudes
are untouched, so nothing amplitude-based catches it.
`pynufft_transformer_class()` applies the shift;
`tests/test_pynufft_transformer.py` pins it in both directions, including a
guard that fails if upstream fixes it (which would make the wrapper
double-shift). **Whether `nufftax` shares the convention is still unanswered
here** — no container available could run JAX — so the same file carries
`test_the_jax_nufft_agrees_with_the_dft`, which skips without JAX and settles
it on any machine that has it.

**Memory, which the NUFFT does not fix.** The inversion builds an
`n_vis × n_mesh` matrix whatever the transformer, and this autoarray has no
w-tilde path. Measured on Ruby (148k visibilities), it works out at about
`n_vis × n_mesh × 44 bytes`:

| mesh at 8" | per evaluation | peak RSS |
|---|---|---|
| 16 (0.50"/px) | 12 s | 1.9 GB |
| 24 (0.33"/px) | 25 s | 3.8 GB |
| 32 (0.25"/px) | ~44 s | ~6.7 GB |
| 70 (0.11"/px, Nyquist) | ~210 s | ~32 GB |

So the field of view is the expensive parameter, quadratically. Which is why
`--image-centre` matters: both of these sources sit 3–4″ off the phase centre,
so reaching them from the centre forced an 8″ field. Recentred, Ruby's ring
needs 3″ — the same source at full Nyquist resolution for 4.4 GB instead of
32 GB. The recentring is an exact phase ramp, `V' = V exp(2πi(u·x₀ + v·y₀))`,
the same operation as CASA's `phaseshift` minus the w-term, which is
negligible over a few arcsec. `CRVAL` moves with the grid, so the astrometry
is unchanged — that is the one way this could do real damage silently, and
`tests/test_image_centre.py` covers it.

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

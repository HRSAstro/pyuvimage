# Uncertainty maps, in full

[← back to the README](../README.md)

One map, `uncertainty.fits`, in Jy/pixel: the best total 1σ per pixel the fit
can estimate, so that `model.fits / uncertainty.fits` (written for you as
`snr.fits`) is directly usable as a significance map.

**What goes into it.** Two terms, added in quadrature, with the medians of
each written to the FITS header so you can see the split without recomputing
anything:

| term | header key | what it is | how it is obtained |
|---|---|---|---|
| statistical | `ERRSTAT` | how well the data pin this pixel down, given the prior | `sqrt(diag(M C M^T))` with `C = (F+H)^-1`, the closed-form posterior covariance |
| prior systematic | `ERRSYS` | how much the answer depends on *how strongly* you smoothed | how far the pixel moves when the regularisation strength is varied over the range χ² cannot distinguish |
| | `ERRWLO`, `ERRWHI` | (record that range, in dex either side of the fitted strength) | |
| | `ERRWMEA` | (`T` if the range was measured, `F` if a fixed window was asked for) | |
| | `ERRDEBL` | (records that the checkerboard was removed) | |

Rule of thumb from the mocks: the statistical term dominates in smooth
extended emission, the systematic dominates on compact features — which is
exactly where the prior is doing the most work and where a purely statistical
error bar would mislead you. On the demo the medians are 1.7e-6 and 1.0e-6
Jy/pixel respectively.

In detail:

**Statistical.** The inversion is linear with a Gaussian prior, so the
posterior covariance is closed-form, `C = (F + H)^-1`, propagated to the image
grid as `sqrt(diag(M C M^T))` — not by copying per-mesh-pixel errors across,
since the mapper interpolates and neighbouring mesh errors are correlated. The
noise-only part of this, `(F+H)^-1 F (F+H)^-1`, was verified at **0.996**
(matern) and **0.995** (gibbs) against 30-realisation Monte Carlos.

**Prior systematic.** `C` contains no data and is conditional on one prior at
one strength, which makes it optimistic: a regularised model is smoothed, so
it is biased, and on the extended+compact mock the smoothing bias is ~2.8x the
random scatter. The systematic term measures how far each pixel moves when the
regularisation strength is varied — the same construction used for
point-source fluxes, where it turned pulls of up to 24σ into pulls under 3. It
concentrates where it should: around compact features, where the prior is
doing the most work.

**How far to vary it is measured, not assumed** (`SingleFit.chi2_admissible_dex`).
The window is the set of strengths the data cannot tell apart: those whose χ²
is within one σ(χ²) = √(2N) of the fitted strength's, found by walking outward
in half decades. It is floored at ±0.5 dex — the walk cannot resolve a reach
finer than one step, and this is the fixed window the method used before — and
capped at ±6 dex, by which point the model no longer resembles the data.

A fixed window assumes the admissible range of strengths is the same whatever
the data, and it is not. On a weakly constrained fit the prior takes over, χ²
stops responding to the strength, and the model can be orders of magnitude
away with no χ² penalty — while the *statistical* term shrinks, because a
strong prior shrinks the posterior variance. The quoted error then falls as
the data get worse, which is exactly backwards. Measured on the demo mock down
a 60× range in noise, comparing the quoted 1σ against the actual rms error
(coverage = actual / quoted, >1 means the map under-states the error):

| peak S/N | measured window | coverage, fixed ±0.5 dex | coverage, measured |
|---|---|---|---|
| 300 | ±0.5 | 0.73 | 0.73 |
| 132 | ±0.5 | 0.71 | 0.71 |
| 58 | ±0.5 | 0.67 | 0.67 |
| 26 | ±1.5 | 0.72 | 0.59 |
| 11 | −5.0/+1.5 | **1.91** | 0.53 |
| 5 | ±6.0 | **9.64** | 1.87 |

Because of the floor the two are identical wherever the fit is well
constrained; the window only ever opens. The cost is the walk: 5 extra solves
of the n_mesh system on a well-constrained fit, at most 25 on one χ² barely
responds to — no transforms and no refit. Passing
`model_uncertainty_total(0.5)` restores a fixed window if you want the old
number.

**What it still misses.** It does not cover the prior *family* being wrong,
nor calibration or deconvolution error, and it does not cover the smoothing
bias *at* the fitted strength: on the extended+compact mock, where a resolved
knot sits on a smooth disc, the total under-states the rms error on source
pixels by ~4× at high S/N whichever window is used, because moving λ within
the admissible range does not move the model but the model is biased anyway.
Treat the map as a floor on the error near compact structure, not a complete
account of it. Nothing cheap fixes this.

![uncertainty](../figures/uncertainty_total.png)

**The checkerboard is removed.** Products live on a grid `oversample`× finer
than the model mesh, and the mapper interpolates: a pixel on a mesh node
inherits one mesh pixel's variance, a pixel between nodes is a weighted average
of several and has a genuinely smaller one. Both numbers are right, but the
alternating pattern is an artefact of the two grids and it lands straight in
any significance map — measured at **55%** peak-to-peak within a block on the
test mock. The delivered map replaces it with its upper envelope (a block
maximum, then a block mean), bringing it to **11%**. This is deliberately the
conservative direction: an over-stated error never manufactures a detection.
`ERRDEBL` records that it was done.

**Why the map looks the way it does.** With the prior held fixed the
statistical term is *identical* for completely different datasets (verified to
exactly zero difference) and **cannot respond to how bright the source is**.
Its structure comes from the uv coverage and noise (through `F`), the prior
(through `H`), and the mask edges. A **stationary** prior (`matern`,
`exponential`) makes both translation-invariant and the term is flat by
construction — a featureless matern map is the correct answer, not a bug. A
**non-stationary** prior (`gibbs`, `adaptive`, `gaussian`) varies: on the
extended+compact mock the gibbs map peaks 5x its median at the unresolved knot,
with a 6x range across the field, and the Monte Carlo reproduces that
structure. The knot has the *larger* error bar because the prior is
deliberately weakest there. The systematic term, by contrast, does respond to
the source — it is a difference of two fits.

**Do not add per-pixel errors in quadrature.** The covariance is strongly
correlated over the prior's correlation length. Use
`SingleFit.aperture_uncertainty(region)`, which evaluates `w^T (M C M^T) w`
properly. On our mock quadrature *overstates* a compact aperture's error by
~1.4x.

Everything above is conditional on the noise map being right and on the fitted
hyperparameters; it does not include the uncertainty in those.

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
| prior systematic | `ERRSYS` | how much the answer depends on *how strongly* you smoothed | how far the pixel moves when the regularisation strength is varied over ±0.5 dex |
| | `ERRSPRD` | (records the ±dex used) | |
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
regularisation strength is varied over ±0.5 dex — the same construction used
for point-source fluxes, where it turned pulls of up to 24σ into pulls under
3. It concentrates where it should: around compact features, where the prior
is doing the most work.

It does **not** cover the prior *family* being wrong, nor calibration or
deconvolution error. Nothing cheap does.

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

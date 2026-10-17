# Uncertainty maps, in full

[← back to the README](../README.md)

**Choosing what is computed (`--uncertainty`).** Off by default (`none`).

| mode | `uncertainty.fits` is | extra product | cost on REBELS-25 (fit ~90 s) |
|---|---|---|---|
| `none` | not written (nor `snr.fits`) | — | — |
| `statistical` | the posterior width at the fitted prior strength | — | +11 s |
| `bayesian` | the posterior marginalised over the prior strength, weighted by the evidence | `model_bayesian.fits`, the posterior mean those errors belong to | +61 s |

The header key `ERRTYPE` says which. `statistical` is conditional on the
prior strength the criterion chose and under-states the error at low S/N;
`bayesian` is the one to use when the error bars matter (see "How far each
can be trusted" below). A third mode, statistical + a "prior systematic",
existed until Oct 2026 and was removed; why is at the end.

**`uncertainty.fits` is not the denominator of a significance map.** On a
model sampled finer than the beam neighbouring pixels are strongly
anticorrelated: each one is poorly determined while their sum is not. On
REBELS-25 (0.03" pixels, 0.24" beam) `model / uncertainty` peaked at ~3 on a
14.5σ source, and on a 40σ mock disc at 12.6.

**`snr.fits` is S/N at the restoring beam's resolution** (`SingleFit.beam_snr`):
the model convolved with the restoring beam — the restored image without its
residuals, in Jy/beam — over the 1σ of *that smoothed model*, propagated from
the full posterior covariance, sqrt(diag(K M C Mᵀ Kᵀ)) (`smoothed_std`), never
by smoothing the per-pixel error map. In `bayesian` mode both the numerator
and the covariance are the marginal posterior's (`model_bayesian.fits` and its
covariance). With point components the points are added as the restoring beam
at their fitted positions, and the error uses the joint covariance of mesh
and amplitudes, whose cross-term is negative (a point and the mesh beneath it
trade flux); in `bayesian` mode that joint covariance is the marginal one,
with the amplitudes in the mixture (see "Point sources" below). The header
carries the beam it refers to.

**For a number to quote, use a region.** Every run reports one: `source_flux`
in `fit_parameters.json`, and a line in the log — the flux inside the region
where the model convolved with the restoring beam exceeds 3× the rms, with its
error from the full posterior covariance (`SingleFit.aperture_uncertainty`).
In `bayesian` mode the record adds `flux_bayesian_jy` and
`flux_error_bayesian_jy`, the marginal posterior mean's own sum and error
(`aperture_bayesian`), and `snr` is theirs.

**Header keys.**

| key | what it records |
|---|---|
| `ERRTYPE` | `statistical` or `bayesian` |
| `ERRSTAT` | median 1σ [Jy/pixel] |
| `ERRDEBL` | `T`: the mesh/image checkerboard was replaced by its envelope |
| `BAYLAM`, `BAYLSD` | `bayesian` only: posterior mean and sd of log₁₀ λ |

## Statistical

The inversion is linear with a Gaussian prior, so the posterior covariance is
closed-form, `C = (F + H)^-1`, propagated to the image grid as
`sqrt(diag(M C M^T))` — not by copying per-mesh-pixel errors across, since the
mapper interpolates and neighbouring mesh errors are correlated. The
noise-only part of this, `(F+H)^-1 F (F+H)^-1`, was verified at **0.996**
(matern) and **0.995** (gibbs) against 30-realisation Monte Carlos.

It is conditional on one prior at one strength, which makes it optimistic: a
regularised model is smoothed, so it is biased, and on the extended+compact
mock the smoothing bias is ~2.8x the random scatter. Worse, the term *falls*
as the data get worse: a weaker dataset gets a stronger prior, and a strong
prior shrinks the posterior variance.

## Bayesian

With the prior's shape held at the fitted one (for `adaptive`, its brightness
map) and only the strength λ free, a flat prior on log λ gives
p(log λ | d) ∝ Z(λ), the evidence. The image posterior is then a mixture of
the conditional Gaussians, whose mean and covariance are

    mean = Σ_j w_j ŝ_j
    cov  = Σ_j w_j [ C_j + (ŝ_j − mean)(ŝ_j − mean)ᵀ ]      (law of total variance)

with ŝ_j = (F+H_j)⁻¹ D, C_j = (F+H_j)⁻¹ and w_j = Z(λ_j) / Σ_k Z(λ_k) on a
grid in log λ (`SingleFit.bayesian_posterior`). One dimension needs no
sampler: a coarse pass (0.5 dex steps, walking further out while the peak
sits at an edge) locates the evidence peak, a fine pass (0.1 dex) resolves
it, and grid points with w < 10⁻⁶ are dropped. Each grid point is one
unconstrained solve and one evidence; each point kept costs one inverse.

It is **unconstrained throughout**: the evidence and the Gaussian posterior
are exact only without positivity, so `model_bayesian.fits` can dip slightly
below zero. It is centred on the evidence-preferred λ, not the criterion's,
so it differs from `model.fits`: on the demo mocks the evidence sits 1–1.4 dex
weaker than `structure` (a sharper, lumpier map); on REBELS-25, 0.5 dex
stronger. On bright sources the two agree (REBELS-25 source flux 1.20 vs
1.17 mJy).

## Posterior samples

`--posterior-samples N` writes `posterior_samples.npz`: N draws from the
posterior of every channel, from the same covariance as `uncertainty.fits`
(the marginal one under `--uncertainty bayesian`, the fitted-strength one
otherwise). Each draw is the delivered model plus a correlated offset
`M delta`, `delta ~ N(0, C)`, so the draws' mean is `model.fits` and their
scatter per pixel is the undeblocked `uncertainty.fits`. Under `bayesian`
they are still centred on `model.fits` (the positive, delivered model, which
the moments are taken of), with the marginal covariance, which includes the
spread between the means at different strengths.

Anything computed from the model can be computed from each draw, and the
scatter over draws is its error, with every correlation in `C` carried
through: between neighbouring pixels (anticorrelated across the
interpolation), and over the prior's correlation length. A sum over a region
reproduces `aperture_uncertainty` (to 5% with 4000 draws on the mocks),
where the per-pixel errors in quadrature do not.

This is what says how much structure finer than the beam the data support.
A forward model is not limited to the beam the way a CLEAN image is: where
the S/N is high the data constrain the model on smaller scales, and the
draws agree on that structure; where it is low they spread as far as the
prior allows. `snr.fits` answers the question at the beam's resolution; the
draws answer it at any resolution, for any derived quantity.

```python
from astropy.io import fits
from pyuvimage.posterior import PosteriorSamples

post = PosteriorSamples("pyuvimage_out/posterior_samples.npz")
model = fits.getdata("pyuvimage_out/model.fits")
values = [my_statistic(post.draw(k, model)) for k in range(post.n_samples)]
```

`scripts/moment_maps.py` uses them for moment-map errors: every draw goes
through the same mask and the same moments, and half the 16-84% range over
draws is written as `<out>_mom0_err.fits`, `_mom1_err.fits` and
`_mom2_err.fits`, with `<out>_mom0_snr.fits`.

**Caveats.** The draws are Gaussian, so positivity is not imposed: where the
model sits at zero they scatter either side of it, and a moment 1 or 2 whose
weights nearly cancel is unstable (hence the percentile range rather than the
standard deviation). Errors from a fixed mask are conditional on it. Analytic
point components are not sampled (the draws are skipped for fits with
points), and the matrix-free solve has no covariance to draw from.

**Size.** Per channel, `N x n_mesh x 4` bytes of draws plus ~12 bytes per
image pixel per mesh neighbour for the map onto the image grid: for a
37-channel cube with 9216-point meshes on a 384² image and N = 100, about
330 MB. Drawing them costs one Cholesky factorisation of `C` per channel.

## Point sources

Point components are solved together with the mesh, with no prior on their
amplitudes. `point_sources.json` always carries each point's statistical
error (`flux_error_jy`, at the delivered prior strength, the mesh
marginalised out). With `--uncertainty bayesian` the amplitudes go into the
evidence-weighted mixture with the mesh: at each λ the joint system is
solved, its evidence is

    ln Z = −½ [χ² + sᵀHs + ln det(F+H) + ln det S − ln det H] + const

(S = C − Bᵀ(F+H)⁻¹B, the amplitudes' curvature with the mesh marginalised
out; the flat amplitude prior adds only a constant), and the mixture is taken
over (s, a) together. Each point then gets `flux_bayesian_jy` ±
`flux_error_bayesian_jy`; `model_bayesian.fits` carries the points at those
fluxes; `uncertainty.fits` is the extended model's marginal error with the
amplitudes marginalised out; `snr.fits` uses the joint marginal covariance.
On the demo S/N series (one 4 mJy point beside a disc) the Bayesian pulls are
0.2–1.0 against 1.4–1.7 statistical; where a point sits on extended emission
both under-state the error badly (docs/point-sources.md).

## How far each can be trusted

On the demo S/N series (`scripts/marginal_lambda_mock.py`,
`figures/bayesian_vs_statistical.pdf`), scored on source pixels against the
truth:

| peak S/N | within 1σ, statistical | within 1σ, Bayesian | flux, statistical | flux, Bayesian |
|---|---|---|---|---|
| 299 | 0.58 | 0.85 | 1.00 | 1.00 |
| 131 | 0.42 | 0.82 | 0.98 | 1.00 |
| 58 | 0.29 | 0.75 | 0.92 | 0.98 |
| 26 | 0.22 | 0.77 | 0.81 | 0.96 |
| 11 | 0.07 | 0.68 | 0.60 | 0.87 |
| 5 | 0.00 | 0.65 | 0.27 | 0.72 |

0.68 is calibrated. The statistical error is the posterior width at the
`structure` strength; most of the Bayesian gain comes from *where* the
evidence puts λ, not from the spread over λ (the posterior on log₁₀ λ is only
0.07–0.28 dex wide and adds ≤1% to the variance). On REBELS-25 the two agree:
source-flux S/N 15.0 statistical, 15.2 Bayesian.

Neither covers the prior *family* being wrong, nor calibration or
deconvolution error.

## The checkerboard is removed

Products live on a grid `oversample`× finer than the model mesh, and the
mapper interpolates: a pixel on a mesh node inherits one mesh pixel's
variance, a pixel between nodes is a weighted average of several and has a
genuinely smaller one. Both numbers are right, but the alternating pattern is
an artefact of the two grids and it lands straight in any significance map —
measured at **55%** peak-to-peak within a block on the test mock. The
delivered map replaces it with its upper envelope (a block maximum, then a
block mean), bringing it to **11%**. This is deliberately the conservative
direction: an over-stated error never manufactures a detection. `ERRDEBL`
records that it was done.

## Why the map looks the way it does

With the prior held fixed the statistical term is *identical* for completely
different datasets (verified to exactly zero difference) and **cannot respond
to how bright the source is**. Its structure comes from the uv coverage and
noise (through `F`), the prior (through `H`), and the mask edges. A
**stationary** prior (`matern`, `exponential`) makes both
translation-invariant and the term is flat by construction — a featureless
matern map is the correct answer, not a bug. A **non-stationary** prior
(`gibbs`, `adaptive`, `gaussian`) varies: on the extended+compact mock the
gibbs map peaks 5x its median at the unresolved knot, with a 6x range across
the field, and the Monte Carlo reproduces that structure. The knot has the
*larger* error bar because the prior is deliberately weakest there. The
Bayesian map does respond to the data, through the evidence weights.

**Do not add per-pixel errors in quadrature.** The covariance is strongly
correlated over the prior's correlation length. Use
`SingleFit.aperture_uncertainty(region)`, which evaluates `w^T (M C M^T) w`
properly. On our mock quadrature *overstates* a compact aperture's error by
~1.4x.

Everything above is conditional on the noise map being right.

## Why there is no prior-systematic mode

Until Oct 2026 a third mode, `systematic`, added to the statistical term the
largest change of the model across a window of "admissible" prior strengths,
walked in half decades from the fitted one. It was removed because:

- **Its size was set by the window rule, and the rule was a choice.** χ²
  within √(2N) for a `discrepancy` fit; the structure ratio within its noise
  scatter for a `structure` fit. Neither is a posterior on the strength. On
  REBELS-25 the χ² rule gave ±6 dex and the structure rule ±0.5 dex on the
  same fit, and source-flux S/N of ~1 and 14.5.
- **The structure window did not open at low S/N**, because the ratio barely
  responds to the strength when the data are weak; the quoted error fell ~10×
  short at 5σ. Widening the tolerance to 5× fixed the mock but tripled the
  error on REBELS-25 (S/N 14.5 → 5.4), through the next point.
- **Positivity inflated it at weak priors**: non-negative solves keep the
  positive noise excursions and drop the negative ones, so flux was added from
  noise (up to 65% of the source flux on REBELS-25).
- **It could not see the smoothing bias at the fitted strength**, which is
  what dominates at low S/N.
- **The Bayesian mode covers the truth better at every S/N tested**
  (fraction within 1σ 0.00–0.75 for statistical + systematic, against
  0.65–0.85 Bayesian, on the demo series) and is defined without a tuning
  rule.

For comparison with other codes: PyAutoArray's `reconstruction_noise_map` is
√diag((F+H)⁻¹) at a fixed coefficient (our `statistical`); in
PyAutoGalaxy/PyAutoLens the coefficient is a sampled model parameter, so
quantities computed per sample are marginalised over it. Suyu et al. (2006)
treat the strength as a delta function and note that the resulting errors
understate the error against the truth. The EHT (2019, Paper IV) quote the
spread over a "top set" of regulariser weights validated on synthetic data,
explicitly not a posterior — the closest analogue of the removed mode.

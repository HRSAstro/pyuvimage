# pyuvimage

Easy image reconstruction of radio interferometric data by **forward
modelling in the uv-plane**. A lightweight alternative to CLEAN for users who
want a regularised maximum-likelihood image with honest residuals, without
being an interferometry expert and without heavy compute.

Built on [PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy)'s
pixelized-source inversion (developed for gravitational lens modelling, used
here with the lens equation switched off): the sky is a freeform image on a
cartesian grid, solved by a linear inversion under a Gaussian-process source
prior (PyAutoLabs' Matérn kernel) whose hyperparameters are optimised
automatically — so the model fits to the noise level rather than through it,
with no knobs to tune.

## Install

```bash
pip install -e .            # core (numpy backend)
pip install -e ".[ms]"      # + python-casacore, to read measurement sets
pip install -e ".[jax]"     # + JAX/nufftax: strongly recommended above ~10^4 vis
```

Python ≥ 3.12 is required by current PyAutoGalaxy releases (3.11 works with
`version: python_version_check: False` in a local `config/general.yaml`).

## Use

```bash
# one-off: convert the measurement set (calibrated data, one field/spw)
pyuvimage import obs.ms mydata/

# reconstruct — fov must cover ALL the emission in the field
pyuvimage fit mydata/ --fov 3.0
```

No CASA? Export with the bundled script instead, then fit the `.npz` directly:

```bash
casa --nologger --nogui -c src/pyuvimage/casa_export.py obs.ms mydata.npz
pyuvimage fit mydata.npz --fov 3.0
```

Or from Python:

```python
import pyuvimage
result = pyuvimage.run("mydata/", fov=3.0, mode="cube")  # or "mfs" (default)
```

Try it with no data at all: `pyuvimage demo`.

### Outputs (all FITS, with WCS from the MS phase centre)

| file | unit | content |
|---|---|---|
| `model.fits` | Jy/pixel | the reconstructed sky (apparent, i.e. PB-attenuated) |
| `model_pbcor.fits` | Jy/pixel | primary-beam-corrected model |
| `clean.fits` | Jy/beam | model ⊗ fitted Gaussian beam + residuals (CLEAN-like) |
| `dirty_image.fits` | Jy/beam | naturally weighted dirty image of the data |
| `dirty_model.fits` | Jy/beam | dirty image of the model visibilities |
| `residual.fits` | σ | (data − model) dirty image / rms (rms in header `RMS`) |
| `pb.fits` | — | primary beam (Gaussian, FWHM ≈ 1.13 λ/D) |
| `uncertainty.fits` | Jy/pixel | per-pixel 1σ posterior uncertainty |
| `uncertainty_noise.fits` | Jy/pixel | per-pixel 1σ from noise alone (no bias) |
| `snr.fits` | — | model / posterior 1σ |

plus `point_sources.json` when point components are fitted (positions,
fluxes, 1 sigma errors), `summary.png` (dirty / model / clean / residual, each with a colour bar;
the residual panel states its peak and rms in sigma), `prior_scan.json` (every hyperparameter trial with its
evidence and chi^2) and `fit_parameters.json` (every parameter of the run).

### Uncertainty maps — and what they do not cover

The inversion is linear with a Gaussian prior, so the posterior covariance of
the reconstruction is available in closed form, `C = (F + H)^-1`. Two maps are
written, both propagated to the image grid as `sqrt(diag(M C M^T))` — not by
copying per-mesh-pixel errors across, since the mapper interpolates and
neighbouring mesh errors are correlated:

| map | what it answers | verified |
|---|---|---|
| `uncertainty.fits` | posterior width: how well is this pixel determined, given the data *and* the prior | vs the total rms error from truth: 1.25x (matern), 0.61x (gibbs) |
| `uncertainty_noise.fits` | `(F+H)^-1 F (F+H)^-1`: how much would this pixel jitter if the source were re-observed | **0.996** (matern) and **0.995** (gibbs) against 30-realisation Monte Carlos |

**Why the maps look the way they do.** `C = (F+H)^-1` contains no data. With
the prior held fixed the uncertainty map is therefore *identical* for
completely different datasets (verified to exactly zero difference) and
**cannot respond to how bright the source is**. Its structure comes from three
places only: the uv coverage and noise (through `F`), the prior (through `H`),
and the mask edges. So:

- a **stationary** prior (`matern`, `exponential`) makes `F` and `H` both
  translation-invariant, and the map is *flat by construction* — a featureless
  matern uncertainty map is the correct answer, not a bug (interior scatter
  measured at 12%, all of it edges and the node/interpolated checkerboard);
- a **non-stationary** prior (`gibbs`, `adaptive`, `gaussian`) varies: on the
  extended+compact mock the gibbs map peaks 5x its median at the unresolved
  knot, with a 6x range across the field, and the Monte Carlo reproduces that
  structure. The knot has the *larger* error bar because the prior is
  deliberately weakest there.

Dynamic-range effects that astronomers often expect near bright sources —
calibration error, deconvolution error — are systematics outside this linear
model and appear in none of these maps.

Three more things to know before quoting these numbers.

**Bias dominates, and neither map includes it.** A regularised model is
smoothed, so it is biased. On the extended+compact mock the smoothing bias is
~2.8x the random scatter, and a compact aperture came out at
`0.0115 +/- 0.0001` against a true `0.0129` — an 11% offset, some 12 sigma from
the error bar. How well the posterior stands in for the total error is
prior-dependent: for `matern` it is 1.25x the true rms error (mildly
conservative), but for the default `gibbs` it *understates* it by ~1.6x. Treat
these maps as the random-error term, not an error budget.

**Do not add per-pixel errors in quadrature.** The covariance is strongly
correlated over the prior's correlation length. Use
`SingleFit.aperture_uncertainty(region)`, which evaluates `w^T (M C M^T) w`
properly. On our mock quadrature *overstates* a compact aperture's error by
~1.4x.

**The maps show a checkerboard.** Image pixels that land on mesh nodes carry
the full mesh variance; pixels between nodes are interpolations of correlated
neighbours and carry less. This is a real property of the estimator — the
Monte Carlo reproduces it — not a plotting artefact.

Everything above is conditional on the noise map being right and on the fitted
hyperparameters; it does not include the uncertainty in those.

### Parameters used in the model fit

Every run writes `fit_parameters.json` recording all of the below, plus
`prior_scan.json` with every hyperparameter trial. Defaults in **bold**.

**Source prior** (the prior on the pixelized source; PyAutoLabs' shipped
default for a pixelized source is the Matern kernel, so it is ours too)

| Parameter | Default | Meaning |
|---|---|---|
| `--reg` | **matern** | `matern`/`exponential`: Gaussian-process prior, regularisation matrix `H = coefficient x C^-1` with `C` a Matern/exponential covariance between mesh pixels. `gaussian`: the same, modulated by a Gaussian envelope on the prior width — adds spatial information, **recommended when visibilities are sparse**. `adaptive`: two-stage, prior width follows a first-pass model — **recommended when a bright compact core sits in fainter emission**. `constant`: nearest-neighbour gradient (rank-deficient — its evidence is ill-behaved). |
| `--envelope-fwhm` | **auto** | For `--reg gaussian`: FWHM [arcsec] of the envelope. `auto` sizes it from the extent of significant emission in the dirty image (at least 3 beams, at most `fov/2`); `optimise` fits it as a free hyperparameter alongside the coefficient. |
| `--envelope-centre` | **auto** | `auto` places the envelope at the **dirty-image peak** — not the phase centre, since the source need not sit there — or `centre`, or `"dy,dx"` in arcsec. |
| `--envelope-floor` | **0.01** | Prior width far from the envelope relative to its peak. Smaller suppresses distant structure more strongly. |
| `--lambda` (coefficient) | **auto** | Prior strength. Optimised; searched over `LogUniform(1e-6, 1e6)`, matching PyAutoLabs' shipped prior. |
| `--scale` | **auto** = beam | Correlation length **in arcsec**. `auto` sets it to the synthesised beam size `sqrt(bmaj x bmin)` — structure finer than the beam is not constrained by the data. `optimise` fits it instead. |
| `--nu` | **1.5** | Matern smoothness (0.5 = exponential, higher = smoother). Fixed by default; PyAutoLabs fit it with `Uniform(0.5, 5.5)`. |
| `--criterion` | **discrepancy** | How the coefficient is chosen. `discrepancy`: the strongest smoothing that still fits to the noise level, `chi^2 = chi2_target x N`. `evidence`: maximise the Bayesian evidence (PyAutoLabs' choice; prefer it when visibilities outnumber pixels). |
| `--chi2-target` | **1.0** | Target `chi^2/N` for the discrepancy criterion. |

**Geometry** (all derived from `--fov`, the one required input)

| Parameter | Default | Meaning |
|---|---|---|
| `--fov` | *required* | Full field of view in arcsec. Must cover all emission. |
| `--pixel-scale` | **auto** | Pixel scale of *every* product: `auto` = half-Nyquist (`0.25/b_max`, ~4 pixels per beam, the usual imaging convention), `nyquist` = `0.5/b_max` (~2 per beam, ~4x cheaper), or a value in arcsec. |
| `--mesh` | derived | Mesh pixels per side; overrides `--pixel-scale`. |
| (oversample) | 1 | Image grid / model mesh ratio. The default of 1 means the model is reconstructed on the same grid the images use, so every product shares one pixel scale. |
| `mask_shape` | **square** | Reconstruction region. A circular mask leaves the mesh's corner pixels covering no image pixels, so no data constrains them and the prior alone sets their value — worth ~29% of the source flux in spurious corner blobs on one test. |

**Solver / data**

| Parameter | Default | Meaning |
|---|---|---|
| `--no-positive` | off (positivity **on**) | The inversion solves `(F + H)s = D`; positivity uses a non-negative solver. The hyperparameter search always uses the fast unconstrained solve, then the coefficient is re-bisected with the constrained solver so the delivered model really does fit to the noise. |
| `--transformer` | **auto** | `dft` below 20k visibilities, else `nufft` (JAX). |
| `--mode` | **mfs** | `mfs` fits all channels jointly to one image; `cube` fits each channel with the prior frozen from the MFS fit. |
| `--noise` (import) | **difference** | Noise from pairwise time-differenced visibilities; `sigma` trusts the MS column instead. |
| `--dish-diameter`, `--no-pb` | from MS | Gaussian primary beam, FWHM = `1.13 lambda/D`. |

### Sparse visibilities: the envelope prior

The Matern prior is *stationary* — it asks that the sky be smooth on beam
scales, but says nothing about **where** the flux is. With few visibilities
that is not enough: a dirty-beam sidelobe far from the source is as acceptable
to it as the source itself, so sidelobe structure leaks into the model.

`--reg gaussian` supplies the missing spatial information in the simplest
form: the prior standard deviation follows a 2D Gaussian, centred on the dirty
image's peak and sized from the emission's extent, falling to a small floor
outside. The prior mean stays zero everywhere, so nothing is imposed on the
flux — pixels far from the source are simply pulled towards zero unless the
data insist otherwise.

On a deliberately sparse test (200 visibilities, 1024 model pixels):

| prior | correlation with truth | model flux vs truth | flux beyond 1.2" (truth 0.15) |
|---|---|---|---|
| `matern` | 0.946 | +20% | 0.32 |
| `gaussian` (envelope = beam) | 0.945 | +16% | 0.30 |
| `gaussian` (envelope = auto) | **0.991** | **−7%** | **0.10** |

Note the envelope wants to be a few beams across, not one: at beam width the
optimiser simply weakens the coefficient to compensate. It assumes the
emission is reasonably compact around one peak — for a wide or multi-component
field, widen it or stay on `matern`.

### A bright compact core: the adaptive prior

A single global correlation length has to compromise between a bright compact
core and faint extended emission, and the core loses — which shows up as a
strong residual right at the peak. Making the mesh finer does *not* fix this
(the core is unresolved, so the limit is the prior, not the pixel size) and
costs a lot:

| mesh (multi-component mock) | chi^2/N | central residual | time |
|---|---|---|---|
| 32 | 1.38 | 4.4 sigma | 9 s |
| 48 | 1.23 | 6.2 sigma | 72 s |
| 64 | 1.18 | 7.1 sigma | 336 s |

`--reg adaptive` is the fix, and is the pixelized-source analogue of the
adaptive treatment PyAutoLens uses for foreground lens light. (Its
`over_sample_size_pixelization` machinery does not apply here: it is fixed to
1 for interferometer datasets upstream, and would be a no-op anyway because
our model mesh and image grid are aligned.) Instead the *prior* is allowed to
vary: a first pass with the plain Matern prior gives a brightness map, and the
second pass sets the prior width per pixel to
`floor + (1 - floor) * (b_i / max(b))^power`, so the core is smoothed less and
the faint outskirts more. On the multi-component mock the central residual
falls from **4.4 sigma to 1.4 sigma**, correlation with truth rises 0.9936 ->
0.9972 and the flux error halves, for about twice the run time.

### Choosing a prior

Measured on two deliberately different mocks: a single exponential with very
sparse coverage (200 visibilities), and a multi-component source (bright
compact core + faint offset disc + small offset knot, 600 visibilities).

Sparse single exponential:

| prior | corr | flux ratio | flux beyond 1.2" (truth 0.15) |
|---|---|---|---|
| `matern` | 0.971 | 1.36 | 0.28 |
| `gaussian` (auto width) | 0.999 | 1.02 | 0.12 |
| `gaussian` (`--envelope-fwhm optimise`) | **0.9994** | **1.00** | 0.11 |

Multi-component source (all four priors, same grid, residual map diagnostic):

| prior | chi^2/N | corr | flux ratio | resid rms | central resid | core / disc / knot |
|---|---|---|---|---|---|---|
| `matern` | 1.00 | **0.896** | 1.12 | 0.63 sigma | 2.2 sigma | 1.04 / 1.02 / 1.05 |
| `gaussian` (auto) | 1.28 | 0.887 | **1.05** | 0.87 sigma | 0.3 sigma | 1.04 / 0.99 / 1.03 |
| `gaussian` (optimise) | 1.27 | 0.884 | 1.06 | 0.83 sigma | 0.9 sigma | 1.04 / 0.98 / 1.04 |
| `adaptive` | 1.16 | **0.896** | 1.07 | 0.75 sigma | 1.4 sigma | 1.04 / 1.01 / 1.05 |

**The envelope's large advantage does not generalise.** It is worth a lot on
the sparse single-component mock and is roughly neutral on the complex source
— slightly worse in morphology, slightly better in total flux. All four priors
recover the three components to within ~5%, including the faint offset knot,
so none of them is suppressing real structure. Treat `gaussian` as a tool for
sparse coverage rather than a general default, and `matern` (the default) or
`adaptive` as the general choice. The correlation figures are not comparable
between the two tables: they are measured against truth on the finer product
grid, where the block-replicated model saturates around 0.9.

### Why the model mesh is coarser than the product grid

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

### Extended source + unresolved off-centre knot

The test that separates the priors. An extended exponential (r_eff 0.7") plus
an unresolved compact source offset by ~1", 600 visibilities, mesh 32:

| prior | chi^2/N | corr | compact flux | extended flux | **compact peak** | resid at knot |
|---|---|---|---|---|---|---|
| `matern` | 0.99 | 0.659 | 0.93 | 1.05 | **0.59** | 5.5 sigma |
| `gaussian` | 1.00 | 0.834 | 1.30 | 0.96 | 0.99 | 3.2 sigma |
| `adaptive` | 1.00 | **0.969** | **1.03** | **1.01** | 1.42 | 3.9 sigma |

One global correlation length cannot serve both components: `matern` smooths
to suit the extended emission and recovers only **48%** of the knot's peak,
leaving a 5.7 sigma dipole residual on it.

Variants measured on this mock, all fitted to the same chi^2 = N so the
comparison is at equal goodness of fit (model images assembled exactly, via
the mapping matrices):

| variant | corr | compact flux | extended flux | compact peak | resid at knot |
|---|---|---|---|---|---|
| `matern` | 0.640 | 0.90 | 1.04 | 0.48 | 5.7 sigma |
| `adaptive` (power 1) | 0.886 | 0.98 | 1.00 | **1.04** | 3.9 sigma |
| `adaptive` (power 2) | 0.892 | 0.98 | 1.00 | 1.09 | 3.6 sigma |
| `adaptive` x2 iterations | 0.866 | 0.95 | 1.01 | 0.94 | 4.7 sigma |
| hybrid: mesh + linear Gaussian | 0.752 | 0.98 | 1.01 | 1.72 | 6.0 sigma |
| hybrid + `adaptive` | **0.899** | 0.98 | 1.00 | 1.07 | 5.0 sigma |
| Gibbs (non-stationary length) | 0.868 | 0.99 | 0.98 | 1.11 | **2.4 sigma** |
| Gibbs + amplitude adaptation | 0.877 | 0.97 | 1.00 | 1.07 | 3.1 sigma |

`adaptive` recovers both components' fluxes to within 3% and the knot's peak
to 4%. A non-stationary *correlation length* (Gibbs kernel, short where the
source is bright) is the only variant that materially reduces the residual
**at** the knot — 2.4 sigma against 3.6-6.0 for everything else — which is the
direct symptom of mis-fitting compact emission. Adding an explicit linear
point component helps only in combination with `adaptive`, and costs a
position that must be found first.

These are one mock and one noise realisation; differences of ~0.01 in
correlation are not meaningful.

### True point sources: an analytic delta component

A genuine point source is the one thing a pixel grid cannot represent. Its
visibilities are exact and closed-form,

    V(u, v) = A exp(-2i pi (x u + y v))

so the sensible thing is not to put it on the grid at all. `--point-sources`
adds analytic delta components whose amplitudes are solved **in the same
linear system** as the mesh (Schur complement on the augmented normal
equations); only the position is non-linear, and it is refined by a lattice
scan followed by Nelder-Mead. This is opt-in, and off by default.

Why it is worth doing: on the test data a nearest-pixel delta half a pixel
off-centre misrepresents the source at chi^2/N = 31.5, and the best *gridded*
Gaussian still leaves ~1.9 — an error at or above the noise, for a source the
model is meant to describe perfectly.

```bash
pyuvimage fit mydata/ --fov 3.0 --point-sources          # auto-detect
pyuvimage fit mydata/ --fov 3.0 --point 0.70,0.80        # you supply it
```

A supplied position is kept and refined. Auto-detected candidates are taken
from the residual peak and must survive two cuts before being accepted, plus
the significance cut itself:

| guard | what it prevents |
|---|---|
| minimum separation 0.75 x beam | several deltas stacking inside one beam and splitting one feature between them |
| **unresolved test** | a delta being recruited to absorb a *resolved* feature |
| significance > 5 sigma (default) | fitting noise |

The unresolved test is the important one. A Gaussian also has an analytic
visibility, so the candidate can be refitted with its width free: an
unresolved source gains nothing, a resolved one gains a lot. Without it, a
plain exponential disc — no point source at all — yields **five** spurious
"detections" at 9-14 sigma, all within 0.2" of the centre, carrying 5.3% of
the flux. They are absorbing the disc's central cusp, which the smoothed mesh
cannot render. With it, the same data yields none:

```
candidate at dRA 0.045", dDec 0.041" rejected: resolved
    (a 0.161" sigma Gaussian fits better by delta chi2 = 147.5)
```

whereas a real knot passes with nothing to gain from widening:

```
point source accepted at dRA 0.702", dDec 0.796": 0.01204 Jy
    (26.3 sigma, unresolved: widening gains only delta chi2 = 0.0)
```

Measured on the extended + knot mock (600 visibilities, mesh 32, `gibbs`;
truth: 0.040 Jy disc + 0.012 Jy knot at dRA 0.700", dDec 0.800"):

| | chi^2/N | peak residual | knot flux | position error |
|---|---|---|---|---|
| mesh only | 1.00 | 2.42 sigma | — (smeared into the mesh) | — |
| mesh + point | 0.61 | **0.48 sigma** | 0.01204 +- 0.00046 (26.3 sigma) | 0.004" |
| control: disc only, auto-detect | 1.14 | 11.1 sigma | no point accepted | — |

![point sources](figures/point_sources.png)

**On the chi^2/N = 0.61 — a real trade-off, not a bug.** The regularisation
strength was chosen by the discrepancy principle with the knot's flux forced
through the mesh. Once a point carries it, the mesh has freedom it no longer
needs, and the combined fit lands below the target: chi^2/N = 0.61 is the
signature of a mesh now fitting noise. `--point-retune` re-imposes chi^2 = N
by stiffening the prior (here by 8e6, coefficient 6.2e3 -> 5.1e10). Neither
answer is cleanly better:

| | extended model | knot flux (truth 0.01200) | peak residual |
|---|---|---|---|
| mesh only, no point | striped by beam sidelobes at the +-5e-5 level, half the disc's peak | knot smeared into the mesh | 2.42 sigma |
| default (point, no retune) | mottled at +-1e-4, i.e. fitting noise | 0.01204 +- 0.00046 (26 sigma) | **0.48 sigma** |
| `--point-retune` | smooth and disc-like, but 22% low at the centre | 0.01180 +- 0.000033 (358 sigma) | 5.67 sigma |

Compare panels 2-4 of the figure against the truth in panel 1. By eye the
retuned model is much the closest, and the striping in the mesh-only panel is
a reminder of what a prior tuned around an unmodelled compact source costs.
But the retuned error bar is conditional on that much stiffer prior and
carries no prior-induced bias term: the flux is 2% low, which is 7 sigma by
its own quoted error, and the disc centre is under-fitted at 5.7 sigma. The
default keeps the honest error bar and the unbiased flux; the retune keeps
the better-looking disc. The retune is off by default because an under-stated
uncertainty is the more damaging failure mode, and because the low chi^2/N is
reported in the log and in `fit_parameters.json` rather than engineered away.
On real data, try both.

Point components appear in `point_sources.json` with positions, fluxes and
1 sigma errors; in `model.fits` as flux dropped into the nearest pixel (the
grid cannot hold a sub-pixel delta, so that file is flux-correct but
positionally quantised); in `clean.fits` placed analytically at the fitted
sub-pixel position; and in the header as `NPOINTS` and `PTFLUX`. The mesh
uncertainty map is marginalised over the point amplitudes,
`Cov = M^-1 + (M^-1 B) S^-1 (M^-1 B)^T` — ignoring the second term would
understate the error wherever a point competes with the mesh for flux.

**Limits.** The amplitude covariance is conditional on the prior, so a point
sitting on bright extended emission has an error bar that is only as good as
the prior's description of that emission. Detection has no look-elsewhere
correction: 5 sigma is per trial position, not per map. And the resolved-vs-
unresolved threshold (delta chi^2 = 9) was set on mocks, not on real data.

### Known issue: the mask edge

On the mocks the largest residual is usually at the circular mask boundary
(~5 sigma), not at the source: edge mesh pixels are poorly constrained and
absorb flux. Judge a fit from the interior, and treat edge features with
suspicion. `Settings(use_edge_zeroed_pixels=True)` upstream is the likely fix
and is not yet wired in.

### Run time

The dense kernel covariance is built once per (mesh, scale, envelope) and
cached across the coefficient search, so the hyperparameter search is cheap;
the non-negative solver then dominates. Rough CPU (2 cores, NumPy backend,
no JAX) figures for a single-channel fit:

| model pixels | prior | time |
|---|---|---|
| 1024 (32x32) | matern | ~10 s |
| 1024 | adaptive (two-stage) | ~20 s |
| 2304 (48x48) | matern | ~70 s |
| 4096 (64x64) | matern | ~5 min |
| 4096 | adaptive | > 10 min |

The default `--pixel-scale auto` (half-Nyquist, ~4 pixels per beam) gives a
properly sampled restoring beam and unblocky images, but four times the pixels
of `--pixel-scale nyquist`. For a quick look, or on a small machine, use
`--pixel-scale nyquist`; for real work install `pyuvimage[jax]`, which is
where this stack is designed to run.

### A caveat on under-constrained fits

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

## How it works

1. **Import** forms Stokes I from the parallel hands (respecting flags) and
   estimates per-visibility noise from pairwise time-differenced
   visibilities on each baseline (σ = std(diff)/√2) — no reliance on the MS
   weights being correct.
2. **MFS** fits all channels jointly, each with its exact uv coordinates in
   wavelengths (no channel-averaging approximation). **Cube** mode fits each
   channel with the regularisation strength frozen from an MFS fit.
3. The **inversion** solves (F + H) s = D on a rectangular source mesh
   (default Nyquist sampling of the longest baseline) with a
   non-negativity constraint. The source prior's hyperparameters are
   optimised first with the fast unconstrained solver, then the coefficient
   is re-bisected with the constrained solver so the delivered model fits to
   the noise level rather than through it.
4. **Products**: the restoring beam is a Gaussian fitted to the dirty beam
   and evaluated centred on the grid (no sub-pixel shift); dirty images are
   naturally weighted and normalised so the dirty beam peaks at 1 (→
   Jy/beam); the image rms is analytic (1/√(Σw)), verified against a
   Monte-Carlo estimate.

## Caveats (v1)

- Stokes I only; w-term neglected (small-field approximation); single
  field/spw per dataset.
- Total flux is only constrained down to the shortest measured baseline:
  emission resolved out by the array cannot be recovered (same as CLEAN).
- Correlations are assumed independent unless byte-identical; duplicated
  hands are detected so the Stokes I noise is not underestimated.
- If `chi²/N` is reported ≫ 1, the model cannot represent the data — usually
  the `--fov` is too small for the emission, or the pixel scale too coarse.

## Development

`python -m pytest tests/` — includes end-to-end regression tests on
simulated data (adjoint consistency, flux conservation, WCS, restore
centring, noise estimator).

# pyuvimage

Easy image reconstruction of radio interferometric data by **forward modelling
in the uv-plane**. A lightweight alternative to CLEAN for people who want a
regularised maximum-likelihood image with honest residuals and honest error
bars, without being an interferometry expert and without heavy compute.

Built on [PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy)'s
pixelized-source inversion (developed for gravitational lens modelling, used
here with the lens equation switched off): the sky is a freeform image on a
cartesian grid, solved by a linear inversion under a Gaussian-process source
prior whose hyperparameters are optimised automatically — so the model fits to
the noise level rather than through it, with nothing to tune by hand.

## Install

```bash
pip install -e .            # core (numpy backend)
pip install -e ".[ms]"      # + python-casacore, to read measurement sets
pip install -e ".[jax]"     # + JAX/nufftax: strongly recommended above ~10^4 vis
                            #   (if JAX misbehaves, see Troubleshooting below —
                            #    the NumPy path is fully supported)
```

Python ≥ 3.12 is required by current PyAutoGalaxy releases (3.11 works with
`version: python_version_check: False` in a local `config/general.yaml`).

## Use

```bash
# one-off: convert the measurement set (calibrated data, one field)
pyuvimage import obs.ms mydata/            # spw 0
pyuvimage import obs.ms mydata/ --spw all  # or every spectral window

# reconstruct — fov must cover ALL the emission in the field
pyuvimage fit mydata/ --fov 3.0
```

No python-casacore? Export the target data from CASA with the bundled script,
then fit the `.npz` directly:

```bash
# field 0, spw 0; spw may also be a list (0,2), a range (0-3), or "all"
casa --nologger --nogui -c src/pyuvimage/casa_export.py obs.ms mydata.npz 0 all
pyuvimage fit mydata.npz --fov 3.0
```

Or from Python:

```python
import pyuvimage
result = pyuvimage.run("mydata/", fov=3.0, mode="cube")  # or "mfs" (default)
```

Try it with no data at all: `pyuvimage demo`.

## Outputs

All FITS, all on one grid at one pixel scale, WCS from the MS phase centre.

| file | unit | content |
|---|---|---|
| `model.fits` | Jy/pixel | the reconstructed sky (apparent, i.e. PB-attenuated) |
| `model_pbcor.fits` | Jy/pixel | primary-beam-corrected model |
| `model_reconvolved.fits` | Jy/beam | model ⊗ fitted Gaussian beam + residuals — the CLEAN-image analogue |
| `model_reconvolved_pbcor.fits` | Jy/beam | primary-beam-corrected version of it |
| `residual.fits` | σ | (data − model) dirty image / rms (rms in header `RMS`) |
| `uncertainty.fits` | Jy/pixel | total 1σ per pixel — see below |
| `snr.fits` | — | `model / uncertainty`, ready to use as a significance map |
| `dirty_image.fits` | Jy/beam | naturally weighted dirty image of the data |
| `dirty_model.fits` | Jy/beam | dirty image of the model visibilities |
| `pb.fits` | — | primary beam (Gaussian, FWHM ≈ 1.13 λ/D) |

Plus `summary.png` (dirty / model / reconvolved / residual / uncertainty, each
with a colour bar), `fit_parameters.json` (every parameter of the run),
`prior_scan.json` (every hyperparameter trial), and `point_sources.json` when
point components are fitted.

![summary.png for PJ0116 at 245 GHz](figures/pj0116_summary.png)

A real one: PJ0116 at 245 GHz, an ALMA Band 6 continuum observation of a
lensed source (2 spectral windows x 1 channel, 5158 visibility samples,
`--fov 8`). The arc and its counter-image are reconstructed at χ²/N = 1.007,
the residual peaks at 3.7σ — 5% of the source peak, a dynamic range of 20:1 —
and the residual map is featureless, which is what you want to see. The model
panel is in Jy/pixel and the reconvolved panel in Jy/beam, which is why they
look so different: the model is the sky at the mesh scale, not smoothed by the
beam.

## The settings worth knowing about

`--fov` is the only required input; everything else has a default that is
meant to be left alone. These are the ones worth reaching for:

| flag | when |
|---|---|
| `--reg gibbs` | a single bright compact feature you want as sharp as possible (see [docs/priors.md](docs/priors.md)) |
| `--reg gaussian` | very sparse visibilities, where a stationary prior lets sidelobes leak into the model |
| `--point-sources` | there is a genuine point source in the field — no pixel grid can represent one |
| `--pixel-scale nyquist` | you want the longest baselines sampled — finer mesh, several times slower, and only worth it if the long baselines are well populated |
| `--criterion evidence` | the fit looks over-smoothed at a bright peak *and* you have fewer visibilities than model pixels |
| `--mode cube` | per-channel images instead of one MFS image |

Full reference: [docs/parameters.md](docs/parameters.md).

## The uncertainty map

`uncertainty.fits` is the total 1σ per pixel, so `snr.fits` (written for you)
is directly usable as a significance map. Two terms, added in quadrature, with
the median of each in the FITS header:

| term | header key | what it answers |
|---|---|---|
| statistical | `ERRSTAT` | how well the data pin this pixel down, given the prior |
| prior systematic | `ERRSYS` | how much the answer depends on *how strongly* you smoothed |

The statistical term is the closed-form posterior width `sqrt(diag(M C M^T))`
with `C = (F+H)^-1`; its noise-only part was verified against Monte Carlo at
0.996. That term alone is optimistic — a regularised model is smoothed, hence
biased — so the systematic term measures how far each pixel moves when the
regularisation strength is varied over ±0.5 dex. The statistical term tends to
dominate in smooth extended emission and the systematic on compact features,
which is exactly where a purely statistical error bar would mislead you.

Neither covers the prior *family* being wrong, nor calibration or
deconvolution error.

Two things to know before quoting numbers from it:

- **Do not add per-pixel errors in quadrature.** They are correlated over the
  prior's correlation length. Use `SingleFit.aperture_uncertainty(region)`,
  which evaluates `w^T (M C M^T) w` properly; quadrature overstated a compact
  aperture's error by ~1.4x on our mocks.
- The mesh/image **checkerboard is removed** — replaced by its upper envelope,
  conservatively, since an over-stated error never manufactures a detection.
  `ERRDEBL` records that it was done.

![uncertainty](figures/uncertainty_total.png)

Details and validation: [docs/uncertainty.md](docs/uncertainty.md).

## True point sources

A genuine point source is the one thing a pixel grid cannot represent: on our
test data a nearest-pixel delta half a pixel off-centre misrepresents it at
chi^2/N = 31.5. `--point-sources` instead adds analytic delta components whose
amplitudes are solved **in the same linear system** as the mesh.

```bash
pyuvimage fit mydata/ --fov 3.0 --point-sources     # auto-detect
pyuvimage fit mydata/ --fov 3.0 --point 0.70,0.80   # you supply the position
```

This is opt-in and deliberately conservative. Detection is a matched filter
over the whole field, not a residual peak-finder, and a candidate must clear a
significance cut *and* be positive, at least 0.75 beams from any other
candidate, and genuinely unresolved (it is refitted as a Gaussian with free
width and rejected if that fits materially better). Across the generalisation
suite this gave **zero false positives**. Fluxes and 1σ errors land in
`point_sources.json`.

Fitting is refused outright if the pixelized model has not converged to its
chi^2 target, because the residual is then model error rather than sky.

![point sources](figures/point_sources.png)

Details: [docs/point-sources.md](docs/point-sources.md).

## Several spectral windows

`--spw` takes one window (`0`), a list or range (`0,2`, `0-3`), or `all`.
Multiple windows are imported into one dataset and imaged together by
multifrequency synthesis as a single image:

```bash
pyuvimage import obs.ms mydata/ --spw all
pyuvimage fit mydata/ --fov 3.0
```

Nothing is averaged or resampled to make them fit together. Every visibility
already carries its own (u, v) computed at its own channel frequency, so
combining windows is the same operation the single-window path has always
performed across channels — splitting a dataset into spectral windows and
imaging it gives a bit-identical image, which is a regression test.

Windows keep their own channels, their own rows and their own noise estimate,
because in a measurement set all three differ between them. On disk the
dataset becomes `spw000/`, `spw001/`, ...; single-window datasets written
before this keep working unchanged.

**MFS fits one frequency-independent image.** That is mild within one window
and can be strong across several: at fractional bandwidth *B*, a source with
spectral index α is mis-modelled by roughly |α|·*B* across the band — about
40% for α = −0.7 over a 2:1 frequency range. pyuvimage has no Taylor-term
expansion (CLEAN's `mtmfs`), so it warns above 20% fractional bandwidth and
leaves the judgement to you. The fit will still reach its chi^2 target by
absorbing the spectral structure into the image; read the result as a
band-averaged sky, and image the windows separately if you need spectra.

`--mode cube` also works across windows: channels are ordered by frequency and
fitted independently. Their spacing is then irregular, which a linear FITS
frequency axis cannot express, so the true per-plane frequencies are written to
the header as `FRQ0000...` and to `frequencies.json`, and `FREQIRR` marks that
`CDELT3` is only indicative.

## Troubleshooting

**`AttributeError: partially initialized module 'jax' has no attribute
'version'`** (or any other crash mentioning jax). Your JAX install is broken,
not pyuvimage. The PyAuto libraries decide whether JAX exists by looking for it
on disk rather than importing it, so a broken install passes that check and
then fails deep inside a fit.

pyuvimage now detects this at startup, falls back to the NumPy path, and tells
you. To fix JAX itself, in the environment you run pyuvimage from:

```bash
python -c 'import jax; print(jax.__version__)'   # see the real error
pip uninstall -y jax jaxlib jax-metal
pip install -U 'jax[cpu]'                        # a matched jax/jaxlib pair
```

A jax/jaxlib version mismatch is the usual cause; on Apple silicon the
`jax-metal` plugin is another; mixing conda-forge and pip installs of jax in one
environment does it too. If you would rather not use JAX, `pip uninstall jax
jaxlib` is a clean answer — the NumPy path is fully supported and everything in
these docs was measured on it.

## When not to trust a fit

- **A structure ratio above ~1.5.** Every run prints one:

  ```
  residual map rms 4.28 sigma against 1.00 expected for white noise
  at chi2/N = 1.008: structure ratio 4.28
  ```

  If the residual visibilities were noise, the residual dirty image would have
  rms `sqrt(chi^2/N)` in sigma, because incoherent residuals average down as
  `1/sqrt(N)`. Coherent residuals — sky the model failed to reproduce — add in
  phase instead and land `sqrt(N)` higher. So the ratio says whether what is
  left over is *noise or signal*, which `chi^2` cannot: **`chi^2` constrains
  the residual's total power, not its structure**, and a fit can sit exactly on
  `chi^2/N = 1` with the whole source still in `residual.fits`.

  This is the single most useful number the tool prints. It caught a noise map
  inflated 1.4x by a bug in the export, on a fit whose `chi^2/N` read 1.0076.
  The usual cause is a noise map that **over**estimates the noise, which stops
  the fit early; also check `--fov` and whether the source has structure finer
  than the model pixel scale.

- **`chi^2/N` well above 1.** The model cannot represent the data. The run
  says so loudly and refuses to fit point sources.
- **Sparsely sampled uv plane.** In cases where the uv plane is very sparsely
  sampled, faint structure is then set by the prior rather than the data.
  pyuvimage gives a warning in this case.
- **The mask edge.** On mocks the largest residual is often at the mask
  boundary rather than the source: edge mesh pixels are poorly constrained and
  absorb flux. Judge a fit from the interior.
- **A "the non-negative solver is unreliable" warning.** Positivity has been
  disabled for that fit and the model may contain small negative values.

## Run time

Rough CPU figures (2 cores, NumPy backend, no JAX) for a single-channel fit:

| model pixels | time |
|---|---|
| 1024 (32x32) | ~10-20 s |
| 2304 (48x48) | ~70 s |
| 4096 (64x64) | ~5 min |

Point detection adds ~1 s per candidate, and the retune iteration 10-60 s.
Cost is set by the mesh, which `--pixel-scale auto` sizes from the baseline
95% of samples fall within. `--pixel-scale nyquist` sizes it from the *longest*
baseline instead: on an array with a sparse long-baseline tail that can be
several times finer and tens of times slower. For real work install
`pyuvimage[jax]`, which is where this stack is designed to run.

**Averaging the data down first is usually the bigger win.** The fit cost
scales with the number of visibilities, and a modern dataset carries far more
channels and time samples than a small field needs. Averaging in frequency and
in time before `pyuvimage import` costs nothing scientifically *up to the point
where smearing sets in* — and that point depends on how far your emission sits
from the phase centre, so it is worth working out rather than guessing:

- channel averaging is limited by **bandwidth smearing** (chromatic
  aberration), which radially blurs sources in proportion to their distance
  from the phase centre:
  [NRAO note with the formulae](https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/BandwidthSmearing.pdf)
- time averaging is limited by **time-average smearing**, which blurs
  tangentially: [Hitchhiker's Guide to the VLA, on choosing an averaging
  time](https://www.cv.nrao.edu/vla/hhg2vla/node12.html) — it also gives the
  useful rule that time smearing should be kept a little *below* the
  chromatic-aberration loss you have already accepted

Both effects are worst at the field edge, so size them for the most distant
emission you care about, not for the phase centre. Averaging past those limits
resolves out real flux, and neither pyuvimage nor CLEAN can put it back.

## Noise and the MS weights

**The noise is recomputed on every import.** `SIGMA` is nominally
`1/√(2Δν Δt)` — an absolute number — but that only holds if the calibration
put it on an absolute scale. In practice a pipeline sets weights *proportional*
to the true inverse variance without being equal to it, and every `split`,
`mstransform` or averaging step rescales them again. See the
[CASA memo on data weights](https://casa.nrao.edu/Memos/CASA-data-weights.pdf).

This matters more than it sounds. Everything downstream scales with σ: `χ²`,
the discrepancy criterion that chooses how much to smooth, and every
uncertainty in `uncertainty.fits`. A weight column off by 40% stops the fit
early and leaves real emission in the residual — which is exactly what
happened on the first dataset this was tried on.

Every import prints the comparison, whichever mode you use:

```
noise check: MS weights imply median sigma 5.111e-03 Jy,
time-differenced visibilities give 3.696e-03 Jy (ratio 1.383)
```

| `--noise` | what it does |
|---|---|
| **`difference`** (default) | Times-differences each baseline; ignores the weight column entirely. One σ per baseline, pooled over the whole track. |
| `chunked` | The same, but within blocks of the track (`--noise-chunk`, default 600 s), so σ can change as the target rises and sets. Uses no weights. A chunk as long as the track reduces to `difference`. |
| `hybrid` | Per-baseline level from the differences, **time profile** from the weights. Reduces to `difference` exactly when the weights carry no time dependence, so it cannot do worse. |
| `scaled` | Whole shape from `WEIGHT` / `WEIGHT_SPECTRUM`, scale from the differences. |
| `sigma` | Trusts the column as an absolute level. Warns, because it usually isn't one. |

![which effect is visible to what](figures/noise_axes.png)

**The two sources of information are blind on different axes**, which is why
there is more than one mode.

The weight column is *radiometric*: Tsys, bandwidth, integration time, flagged
fraction. It therefore knows exactly what happens along the **time** axis —
your target rises and sets, airmass and Tsys follow, and the column tracks it —
and it knows per-antenna receiver temperature and a heterogeneous array. What
it cannot know is anything downstream of the radiometry: **decorrelation**,
which grows with baseline length because the atmosphere loses coherence faster
over longer separations, or an antenna whose calibration is simply worse than
its Tsys suggests. Two baselines with the same Tsys look identical to it.

Differencing is the reverse. It measures whatever really made the data scatter,
decorrelation included, so the **baseline** axis is honest — but
`sigma_from_time_differences` pools each baseline's whole track into one
number, the quadratic mean of σ(t), so it has no time resolution at all. On a
30→70° track that hands the noisiest data ~2.3× more weight than it deserves.

With both effects present neither wins: on a mock carrying 1.9× of elevation
variation *and* 3.1× of baseline-dependent decorrelation, the median error in σ
was 16.7% for `difference` and 17.0% for `scaled`. Taking each axis from the
source that can see it gave **10.2%**.

**You do not have to guess which regime your data is in — the import measures
it.** Two checks run on every import and print what they find:

```
noise vs baseline length: measured/claimed is 0.16 on the shortest
quartile and 0.23 on the longest (ratio 1.47)

WARNING: the long baselines are 1.47x noisier than the weight column
claims, relative to the short ones -- decorrelation or calibration
quality, which the weights cannot see because they are purely
radiometric. --noise scaled would discard that.
```

```
WARNING: the noise level changes by at least 1.30x over the track
(thirds: 6.92, 5.32, 6.92 mJy) -- elevation and Tsys, most likely.
--noise difference gives every integration on a baseline the same
sigma, so the noisiest data ends up over-weighted.
```

If the first fires, the weight column is missing baseline structure — stay off
`scaled`. If the second fires, `difference` is missing the time dependence —
`chunked` or `hybrid` recovers it. If neither fires, the default is fine.

### When the noise is estimated, and how to change it later

The estimate is made **once, when the data leaves the measurement set**, and
stored in the dataset — `pyuvimage fit` never recomputes it, so a fit costs the
same however the noise was derived.

That used to mean the choice was frozen at import. It is not any more: exports
carry `antenna1`, `antenna2`, `time` and the weight column's relative sigma, so
any estimator can be applied afterwards without going back to the MS.

```bash
pyuvimage convert export.npz mydata/ --noise chunked
```

recomputes and **stores the result**, so no later run pays for it again. Use
`--noise keep` (the default) to leave the map the export already wrote.

Which matters, because the two paths in do not offer the same thing:

| | `pyuvimage import` (needs python-casacore) | `casa_export.py` (runs inside CASA) |
|---|---|---|
| modes available at that step | all of them | `difference` only |
| stores what is needed to change later | yes | yes |

So on the CASA-script route you export once and pick the estimator at
`convert` time. A dataset written by an older export has no antenna or time
columns and will say so rather than guessing.

### Choosing between `chunked` and `hybrid`

Both fix the time axis; they differ in what they need. `chunked` measures it
from the data, so it sees Tsys *and* phase together and needs no weight column
at all — but it has to spend differences on it, and a σ from `n` differences
carries about `1/√(2n)`. `hybrid` reads the time profile off the weights for
free, but the weights are radiometric, and since decorrelation is driven by the
same airmass as Tsys they get the direction of the time dependence right and
the amplitude wrong.

So it comes down to how many integrations land in a chunk — and **how much
target time there is to divide up in the first place**. A typical ALMA
execution runs 1–1.5 h *including* calibrator visits, which take 30–40% of it,
so the target gets roughly 45–60 minutes. That is what the chunks have to
split, and it is why the default is 600 s rather than something longer: it
leaves 5–6 chunks of the track.

Median error in σ on a simulated 75-minute execution with interleaved
calibrators (48 min on source), carrying elevation-driven Tsys *and*
decorrelation correlated with it:

| chunk | 300 s | 450 s | **600 s** | 900 s | 1200 s | 1800 s | (`difference`) |
|---|---|---|---|---|---|---|---|
| 6 s integrations | 7.9% | 7.1% | **7.1%** | 8.1% | 8.2% | 10.7% | 22.4% |
| 30 s integrations | 15.5% | 14.6% | **13.9%** | 12.7% | 12.2% | 14.2% | 22.1% |
| 60 s integrations | 13.5% | 14.4% | **16.8%** | 18.0% | 17.3% | 18.3% | 26.0% |

Too short and each σ comes from too few differences; too long and there is no
time resolution left — by 1800 s it is most of the way back to `difference`.

Two practical notes. Differences that span a **calibrator gap** are excluded
automatically: over a 90 s gap the earth turns a 1.5 km baseline through
~8 kλ, which changes the visibility of anything larger than an arcsecond, so
such a difference would measure the source rather than the noise. And heavily
averaged data — a continuum MS with a handful of timestamps — cannot be chunked
at all; `chunked` detects that and falls back to `difference`.

![difference vs scaled](figures/noise_methods.png)

**Both `difference` and `scaled` difference in time; neither takes a plain std
of the visibilities.** That matters, because a plain std measures the *source*
as well as the noise — on a source 12× the noise it comes out ~650% high,
which would over-smooth every fit. Differencing consecutive integrations on one
baseline cancels the sky exactly, since it is the same sky both times.

The two differ only in **how σ is allowed to vary** across the dataset.
Estimating a baseline's own σ from `n` differences has a fractional error of
about `1/√(2n)` — ±32% from five differences — and that noise in the noise map
randomly up- and down-weights baselines. The weight column supplies the same
shape essentially noise-free, at the cost of assuming its ratios are right.
Below `MIN_DIFFS` differences per baseline, `difference` has no shape at all
and every baseline takes the same pooled number.

`WEIGHT_SPECTRUM` is used when present, for both the Stokes I average and the
noise shape; without it every channel in a row shares one weight, which is
wrong at the edges of any real spw.

## How it works

1. **Import** forms Stokes I from the parallel hands (respecting flags) and
   **recomputes the per-visibility noise from the data**, by differencing
   visibilities adjacent in time on the same baseline (σ = std(diff)/√2). The
   MS weight column is never trusted for scale — see *Noise and the MS weights*
   below.
2. **MFS** fits all channels jointly, each with its exact uv coordinates in
   wavelengths (no channel-averaging approximation). **Cube** mode fits each
   channel with the regularisation frozen from an MFS fit.
3. The **inversion** solves `(F + H)s = D` on a rectangular source mesh, with
   the prior's hyperparameters optimised first by the fast unconstrained
   solver, then the coefficient re-bisected with the constrained solver so the
   delivered model fits to the noise level rather than through it.
4. **Products**: the restoring beam is a Gaussian fitted to the dirty beam;
   dirty images are naturally weighted and normalised so the dirty beam peaks
   at 1 (→ Jy/beam); the image rms is analytic (`1/√(Σw)`), verified against
   Monte Carlo.

## Caveats

- Only Stokes I is currently supported.
- Only a single field is currently supported; several spectral windows can be imaged together (see below).
- The w-term is neglected (small-field approximation), so very large fields are
  not supported.
- Total flux cannot be resolved beyond the maximum recovery scale of the data
  set, which is determined by the baseline length distribution: emission
  resolved out by the array cannot be recovered (same as CLEAN).

## Further reading

| doc | what is in it |
|---|---|
| [docs/priors.md](docs/priors.md) | what each prior is, and how they compare across three mocks |
| [docs/uncertainty.md](docs/uncertainty.md) | the uncertainty map in full, with the Monte Carlo validation |
| [docs/point-sources.md](docs/point-sources.md) | how delta components are solved, and the detection guards |
| [docs/generalisation.md](docs/generalisation.md) | the generalisation suite: a crowded field, four arrays, three noise levels |
| [docs/design-notes.md](docs/design-notes.md) | the grid trap, reading the residual map, under-constrained fits, and other things that bite |

## Development

`python -m pytest tests/` — 58 tests, including end-to-end regressions on
simulated data (adjoint consistency, flux conservation, WCS, restore centring,
the noise estimator, and one test per bug listed in the docs above).

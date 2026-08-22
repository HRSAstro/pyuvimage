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
| **`difference`** (default) | Times-differences each baseline; ignores the weight column entirely. Falls back to one pooled value for baselines with too few integrations to measure their own σ. |
| `scaled` | Keeps the **shape** of `WEIGHT` / `WEIGHT_SPECTRUM` — which does carry real Tsys, band-edge and atmospheric structure — and takes only the **scale** from the same time differences. Better than `difference` when baselines have few integrations, since the per-baseline and per-channel structure survives. It assumes the column's shape is right; if you don't believe that, use `difference`. |
| `sigma` | Trusts the column as an absolute level. Warns, because it usually isn't one. |

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

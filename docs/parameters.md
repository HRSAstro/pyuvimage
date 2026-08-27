# Full parameter reference

[← back to the README](../README.md)

Every run writes `fit_parameters.json` recording all of the below, plus
`prior_scan.json` with every hyperparameter trial. Defaults in **bold**.

**Source prior** (the prior on the pixelized source; PyAutoLabs' shipped
default for a pixelized source is the Matern kernel, so it is ours too)

| Parameter | Default | Meaning |
|---|---|---|
| `--reg` | **adaptive** | `adaptive`: two-stage, the prior's *amplitude* follows a first-pass model as `b^power` — the default, best extended model and no central artefact. `gibbs`: non-stationary, the prior's *correlation length* shortens where the source is bright — sharpest on a single compact feature, but leaves a central residual on extended sources. `matern`/`exponential`: stationary Gaussian-process prior, `H = coefficient x C^-1` with `C` a Matern/exponential covariance between mesh pixels. `gaussian`: matern modulated by a Gaussian envelope on the prior width — **recommended when visibilities are sparse**. `constant`: nearest-neighbour gradient (rank-deficient — its evidence is ill-behaved). |
| `--adapt-power` | **2** | Exponent in the `adaptive`/`gibbs` brightness weighting. |
| `--envelope-fwhm` | **auto** | For `--reg gaussian`: FWHM [arcsec] of the envelope. `auto` sizes it from the extent of significant emission in the dirty image (at least 3 beams, at most `fov/2`); `optimise` fits it as a free hyperparameter alongside the coefficient. |
| `--envelope-centre` | **auto** | `auto` places the envelope at the **dirty-image peak** — not the phase centre, since the source need not sit there — or `centre`, or `"dy,dx"` in arcsec. |
| `--envelope-floor` | **0.01** | Prior width far from the envelope relative to its peak. Smaller suppresses distant structure more strongly. |
| `--lambda` | **auto** | Prior strength (the regularisation coefficient). Optimised; searched over `LogUniform(1e-6, 1e6)`, matching PyAutoLabs' shipped prior. |
| `--scale` | **auto** = beam | Correlation length **in arcsec**. `auto` sets it to the synthesised beam size `sqrt(bmaj x bmin)` — structure finer than the beam is not constrained by the data. `optimise` fits it instead. |
| `--nu` | **1.5** | Matern smoothness (0.5 = exponential, higher = smoother). Fixed by default; PyAutoLabs fit it with `Uniform(0.5, 5.5)`. |
| `--criterion` | **discrepancy** | How the coefficient is chosen. `discrepancy`: the strongest smoothing that still fits to the noise level, `chi^2 = chi2_target x N`. `structure`: the smoothing that makes the residual *map* look like noise (structure ratio 1) — more discriminating when `chi^2` is nearly flat in the coefficient, which happens on real data. Only meaningful on a well-constrained fit; on small mocks the ratio reads ~0.5 at `chi^2 = N` and this criterion over-smooths (see design-notes.md). `evidence`: maximise the Bayesian evidence (PyAutoLabs' choice; prefer it when visibilities outnumber pixels). |
| `--chi2-target` | **1.0** | Target `chi^2/N` for the discrepancy criterion. |

**Geometry** (all derived from `--fov`, the one required input)

| Parameter | Default | Meaning |
|---|---|---|
| `--fov` | *required* | Full field of view in arcsec. Must cover all emission. |
| `--pixel-scale` | **auto** | Model-mesh scale. `auto` = `0.5/b_95`, Nyquist of the baseline 95% of unflagged samples fall within — what the bulk of the data actually supports. `nyquist` = `0.5/b_max`, Nyquist of the *longest* baseline: finer, several times slower, and more mesh than a sparse long-baseline tail can constrain. `fine` = half that again. Or a value in arcsec. Products are written on a grid `oversample` times finer, so they are always finer than the mesh. |
| `--noise-chunk` (on `pyuvimage import` and `convert`) | **600** | How finely `--noise difference` resolves the noise in time, in seconds. Blocks with too few integrations fall back to one sigma per baseline automatically; `0` forces that everywhere. See [noise.md](noise.md). |
| `--mesh` | derived | Mesh pixels per side; overrides `--pixel-scale`. |
| `--image-centre` | **centre** | Where to centre the reconstruction. `centre` = the phase centre. `auto` = the brightest peak in a wide-field dirty image. Or `x,y` in arcsec from the phase centre in **image axes** — +x right and +y up, as you read it off `summary.png`. RA increases leftward, so `x = -dRA`; products are written in dRA/dDec. Same convention as `--point`. A negative x needs the `=` form: `--image-centre="-2.3,0.3"`. Cost goes as `--fov` squared, so recentring on a source a few arcsec off the phase centre is far cheaper than growing the field to reach it — 8″ → 3″ on one real dataset was 32 GB → 4.4 GB *at finer resolution*. The visibilities are rotated by an exact phase ramp and `CRVAL` follows, so the astrometry is unchanged. |
| `--transformer` | **auto** | `auto` picks the direct DFT while it is affordable, then a NUFFT — `pynufft` on any dataset where the JAX one would not fit in memory (which is most of them; see below), otherwise nufftax. `dft`, `nufft`, `pynufft` force one. The DFT allocates `n_pixels × n_vis`, so on 164k visibilities over a 116×116 image it needs 16.5 GB and a NUFFT needs 20 ms — `pip install pynufft` is the fix and needs no JAX. |
| (oversample) | **2** | Image grid / model mesh ratio. Products are written on a grid twice as fine as the model mesh. It must be >1: see "the grid trap" in `design-notes.md`. |
| `mask_shape` | **square** | Reconstruction region. A circular mask leaves the mesh's corner pixels covering no image pixels, so no data constrains them and the prior alone sets their value — worth ~29% of the source flux in spurious corner blobs on one test. |

**Point sources** (all opt-in; nothing is added unless asked for)

| Parameter | Default | Meaning |
|---|---|---|
| `--point-sources` | off | Fit analytic delta components, auto-detecting their positions. |
| `--point x,y` | — | Fit a point at this offset in arcsec from the phase centre, in **image axes** (+x right, +y up — same as `--image-centre`; `x = -dRA`). The position is refined. Repeatable; implies `--point-sources` and disables auto-detection. A negative x needs the `=` form: `--point="-1.2,0.4"`. Fitted positions are *reported* in dRA/dDec, matching the FITS WCS. |
| `--point-significance` | **5** | Keep auto-detected points above this significance. |
| `--max-points` | **5** | Most auto-detected components to keep. |
| `--no-point-retune` | off (retune **on**) | Keep the mesh-only regularisation instead of re-imposing `chi^2 = N` with the points present. |

**Uncertainty**

| Parameter | Default | Meaning |
|---|---|---|
| `--no-uncertainty` | off (map **on**) | Skip `uncertainty.fits` and `snr.fits`. |

**Solver / data**

| Parameter | Default | Meaning |
|---|---|---|
| `--no-positive` | off (positivity **on**) | The inversion solves `(F + H)s = D`; positivity uses a non-negative solver. The hyperparameter search always uses the fast unconstrained solve, then the coefficient is re-bisected with the constrained solver so the delivered model really does fit to the noise. |
| `--enforce-positive` | off | Keep positivity even when the solver looks unreliable. By default pyuvimage probes the non-negative solver and **silently falls back to the unconstrained solve** if it is ignoring the prior or fitting far worse — see below. Use this when a strictly non-negative model matters more than the best image. |
| `--mode` | **mfs** | `mfs` fits all channels jointly to one image; `cube` fits each channel with the prior frozen from the MFS fit. |
| `--spw` (on `pyuvimage import`) | **0** | Spectral window(s): one DATA_DESC_ID, a comma-separated list or range (`0,2`, `0-3`), or `all`. Several are imaged together by MFS. |
| `--noise` (on `pyuvimage import` and `convert`) | **difference** | How the per-visibility noise is set. MS weights are relative, not absolute, so the scale is always recomputed from the data. `difference`: from the visibilities alone. `hybrid`: adds the weight column's time profile. `scaled`: whole shape from the weights. `sigma`: trust `SIGMA` as absolute, and warn. On `convert`, `keep` (the default there) leaves the stored map alone. Full discussion in [noise.md](noise.md). |
| `--dish-diameter`, `--no-pb` | from MS | Gaussian primary beam, FWHM = `1.13 lambda/D`. |


## When positivity is turned off for you

`positive_only` defaults on, but pyuvimage will disable it mid-fit if the
non-negative solver misbehaves. This happens more often than you would expect
— it fired on both Ruby and 9io9 — so it is worth knowing the three checks and
how to see which solver actually ran.

**Where it happens.** All three are in `fitting.fit_dataset`, and all three
only apply when the coefficient is being optimised (`--lambda auto`). A fixed
coefficient skips them entirely.

| # | when | the test | why |
|---|---|---|---|
| 1 | before the hyperparameter search | constrained vs unconstrained `chi^2` at coefficient 1: disable if `chi2_pos > max(2 x chi2_free, chi2_free + 2 n_vis)` | the solver is simply failing on this data |
| 2 | same place, if #1 passes | `chi^2` at coefficient `1e-3` vs `1e9`, twelve decades apart: disable if they agree to within 1% | the solver is *ignoring the prior*. No real prior gives the same `chi^2` at both ends. On one mock every coefficient from `1e-6` to `5.9` returned `chi^2/N = 1.159` to four figures |
| 3 | after the constrained re-bisection | the bisection's own trials: disable if `chi^2` spreads by <1% across more than 3 decades of coefficient | the same symptom as #2, caught on the real trials rather than a probe |

Check #2 is the one that fires on real data. It is why a Ruby fit that reports
`--reg matern --criterion structure` is in fact running **unconstrained**.

**How to tell what ran.** The fit logs *"the non-negative solver is unreliable
on this data: …"* at WARNING, and `fit_parameters.json` records the solver that
actually delivered the model under `solver.positive_only` — not what you asked
for. (Until 26 Aug it recorded the request, which was a lie on exactly these
fits.)

**How to override.** `--enforce-positive` keeps positivity through all three
checks and downgrades them to a warning. The model is then guaranteed
non-negative, but the prior may have little effect on it — which is the
trade-off the automatic fallback exists to avoid.


## Why `auto` usually picks pynufft over JAX

The JAX NUFFT is the faster transform and the worse *transformer* for this
job, because of one line in how the mapping matrix is transformed.

`TransformerNUFFT.transform_mapping_matrix` scatters every mesh pixel into its
own image and passes the whole stack through a single batched `nufft2d2`.
Inside nufftax, the type-2 interpolation materialises its gather buffer in
full (`core/spread.py`):

```python
fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, nspread, nspread)
c = jnp.sum(fw_gathered * weights_2d[None, :, :, :], axis=(-2, -1))
```

That array is `n_mesh × n_vis × nspread²` complex128, and nufftax is not
jitted, so nothing fuses it away — the weighted product is a second array the
same size. `nspread` is the kernel width for the requested precision, and
autoarray asks for `eps=1e-12`, the widest nufftax will build: **14**, so 196
taps per visibility per mesh pixel.

| | mapping matrix | nufftax gather buffer |
|---|---|---|
| Ruby, 20×20 mesh, 148,477 vis | 0.5 GB | **186 GB** |
| 9io9, 50×50 mesh, 328,524 vis | 7 GB | **2,600 GB** |

autoarray knows about this buffer — its `chunk_size` argument exists to cap it,
and its own docstring says chunking is *required* above a few million
visibilities — but `transform_mapping_matrix` is the one path that never
consults `chunk_size`.

pynufft has no equivalent: its mapping-matrix transform is a loop over mesh
pixels that never holds more than one column. So `--transformer auto` compares
the JAX buffer against available memory and takes pynufft when it will not
fit, saying so in the log. `--transformer nufft` still forces JAX, but the
mapping-matrix transform is then split into blocks small enough to survive —
correct, and considerably slower, since it multiplies the NUFFT calls on
*every* trial of the hyperparameter search.

None of this affects the DFT path, `--transformer pynufft`, or JAX's use
anywhere else in the fit.

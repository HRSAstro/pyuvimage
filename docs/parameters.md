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
| `--criterion` | **discrepancy** | How the coefficient is chosen. `discrepancy`: the strongest smoothing that still fits to the noise level, `chi^2 = chi2_target x N`. `evidence`: maximise the Bayesian evidence (PyAutoLabs' choice; prefer it when visibilities outnumber pixels). |
| `--chi2-target` | **1.0** | Target `chi^2/N` for the discrepancy criterion. |

**Geometry** (all derived from `--fov`, the one required input)

| Parameter | Default | Meaning |
|---|---|---|
| `--fov` | *required* | Full field of view in arcsec. Must cover all emission. |
| `--pixel-scale` | **auto** | Model-mesh scale. `auto` = `0.5/b_95`, Nyquist of the baseline 95% of unflagged samples fall within — what the bulk of the data actually supports. `nyquist` = `0.5/b_max`, Nyquist of the *longest* baseline: finer, several times slower, and more mesh than a sparse long-baseline tail can constrain. `fine` = half that again. Or a value in arcsec. Products are written on a grid `--oversample` times finer, so they are always finer than the mesh. |
| `--noise-chunk` (on `pyuvimage import` and `convert`) | **600** | How finely `--noise difference` resolves the noise in time, in seconds. Blocks with too few integrations fall back to one sigma per baseline automatically; `0` forces that everywhere. See [noise.md](noise.md). |
| `--mesh` | derived | Mesh pixels per side; overrides `--pixel-scale`. |
| (oversample) | **2** | Image grid / model mesh ratio. Products are written on a grid twice as fine as the model mesh. It must be >1: see "the grid trap" in `design-notes.md`. |
| `mask_shape` | **square** | Reconstruction region. A circular mask leaves the mesh's corner pixels covering no image pixels, so no data constrains them and the prior alone sets their value — worth ~29% of the source flux in spurious corner blobs on one test. |

**Point sources** (all opt-in; nothing is added unless asked for)

| Parameter | Default | Meaning |
|---|---|---|
| `--point-sources` | off | Fit analytic delta components, auto-detecting their positions. |
| `--point dRA,dDec` | — | Fit a point at this offset in arcsec; the position is refined. Repeatable; implies `--point-sources` and disables auto-detection. |
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
| `--transformer` | **auto** | `dft` below 20k visibilities, else `nufft` (JAX). |
| `--mode` | **mfs** | `mfs` fits all channels jointly to one image; `cube` fits each channel with the prior frozen from the MFS fit. |
| `--spw` (on `pyuvimage import`) | **0** | Spectral window(s): one DATA_DESC_ID, a comma-separated list or range (`0,2`, `0-3`), or `all`. Several are imaged together by MFS. |
| `--noise` (on `pyuvimage import` and `convert`) | **difference** | How the per-visibility noise is set. MS weights are relative, not absolute, so the scale is always recomputed from the data. `difference`: from the visibilities alone. `hybrid`: adds the weight column's time profile. `scaled`: whole shape from the weights. `sigma`: trust `SIGMA` as absolute, and warn. On `convert`, `keep` (the default there) leaves the stored map alone. Full discussion in [noise.md](noise.md). |
| `--dish-diameter`, `--no-pb` | from MS | Gaussian primary beam, FWHM = `1.13 lambda/D`. |

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
| `--criterion` | **auto** | How the coefficient is chosen. `auto` picks between `discrepancy` and `structure` on data per model pixel — `structure` above 10 data points per mesh pixel, `discrepancy` below — and logs which it took and why. The rest force one. `discrepancy`: the strongest smoothing that still fits to the noise level, `chi^2 = chi2_target x N`. `structure`: the smoothing that makes the residual *map* look like noise (structure ratio 1) — more discriminating when `chi^2` is nearly flat in the coefficient, which happens on real data. Only meaningful on a well-constrained fit; on small mocks the ratio reads ~0.5 at `chi^2 = N` and this criterion over-smooths (see design-notes.md). `evidence`: maximise the Bayesian evidence (PyAutoLabs' choice; prefer it when visibilities outnumber pixels). |
| `--chi2-target` | **1.0** | Target `chi^2/N` for the discrepancy criterion. If the model cannot reach it, the search aims just above the achievable floor instead — by `2·sqrt(2/N)`, the standard error of `chi^2/N`, not a fixed percentage. See design-notes.md. |

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
| `--enforce-positive` | off | Keep positivity even when the solver looks unreliable. **Note:** positivity applies to the mesh-only solve. With `--point-sources`, the bordered system is eliminated by an unconstrained Cholesky solve, so the delivered mesh may contain small negative values whatever this is set to (measured on the demo: 0 negative mesh pixels without points, 78 carrying 0.59% of the flux with them). The point amplitudes are unaffected. pyuvimage warns when this applies. By default pyuvimage probes the non-negative solver and **silently falls back to the unconstrained solve** if it is ignoring the prior or fitting far worse — see below. Use this when a strictly non-negative model matters more than the best image. |
| `--inversion` | **dense** | How the curvature matrix `F` is built. `dense` forms the `n_vis x n_mesh` mapping matrix — the allocation that limits every large dataset. `sparse` uses the w-tilde formalism: one streaming pass over the visibilities builds a small translation-invariant kernel, and `F` is then assembled from it by FFT. Exact, not an approximation — identical `chi^2` to eight significant figures on Ruby, and ~85x faster. Needs JAX; MFS only; cannot be combined with `--point-sources`. See below. |
| `--kernel-cache` | beside the output | `--inversion sparse` only: where w-tilde kernels are kept. The kernel depends on the uv coverage, the noise and the geometry and nothing else, so re-fitting the same field with different regularisation reuses it. |
| `--mode` | **mfs** | `mfs` fits all channels jointly to one image; `cube` fits each channel with a shared prior — see `--cube-prior`. |
| `--cube-prior` | **channel** | Cube mode only: what the shared prior is fitted on. `channel` uses a random 1-in-`n_chan` subset of the visibilities — the same amount of data each channel fit will have, and `n_chan` times cheaper. `mfs` uses every channel's visibilities at once, which is the single step that makes a cube run out of memory (Ruby CO(7-6): 2.9 GB per channel against 20.1 GB for that one pass). See below. |
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


## Why `auto` never picks the JAX NUFFT

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
fit, saying so in the log.

**In practice that is always.** The two thresholds do not overlap, and not by a
little:

| | condition |
|---|---|
| the DFT gives up when | `n_vis × n_image_pixels > 1e8`, i.e. `n_vis × n_mesh > 2.5e7` at `oversample 2` |
| the JAX gather fits when | `n_vis × n_mesh ≤ 1.9e5` on a 4.7 GB machine |

A factor of 133 apart. Raising the memory does not help much either, since the
gap is fixed: **a window opens only on a machine with ~627 GB of RAM**, and
even then only for meshes of 8–16 pixels a side. A brute-force sweep over
`(mesh, n_vis)` at 4.7, 16 and 64 GB finds no combination where `auto` picks
it.

So the JAX NUFFT is reachable only through an explicit `--transformer nufft`,
where the mapping-matrix transform is split into blocks small enough to
survive — correct, and considerably slower than pynufft, since it multiplies
the NUFFT calls on *every* trial of the hyperparameter search. The memory
check stays because it is the right general mechanism: if upstream ever makes
`transform_mapping_matrix` honour `chunk_size`, the JAX path becomes reachable
again with no change here.

**None of this makes JAX pointless.** It is required by autoarray's w-tilde
(`InversionInterferometerSparse`) path, which sidesteps the mapping matrix
altogether and is where the real gain is — see the CASA comparison in the
project notes. JAX's value to us is the inversion, not the transform.

None of this affects the DFT path, `--transformer pynufft`, or JAX's use
anywhere else in the fit.


## What the cube's shared prior is fitted on

Cube mode fits each channel separately and gives them all the same prior, so
the planes are mutually consistent. The question is what that prior should be
fitted on — and until 27 Aug the answer was "one MFS fit over every channel's
visibilities", which is both the most expensive step in the run and the wrong
size.

**Wrong size** because the prior is *applied* to single channels. A
coefficient chosen so that all `n_chan` channels together fit to the noise is
not the coefficient a single channel wants. Measured on Ruby 200 GHz
continuum, fitting the full set against a random 1/8 of it:

| criterion | full | 1/8, raw | 1/8, x8 | ratio to full |
|---|---|---|---|---|
| `structure` | 1.883e8 | 5.008e8 | 4.007e9 | **21.3** |
| `discrepancy` | 2.920e8 | 3.991e7 | 3.193e8 | **1.09** |

The `discrepancy` row is the algebra working. `(F + lambda C^-1)s = D` has `F`
and `D` both sums over visibilities, so thinning by `f` and scaling `lambda`
by `f` reproduces the identical model — and `chi^2/N` is invariant under that,
so the criterion agrees. 9% is the scatter from the thinned uv coverage.

The `structure` row is why the coefficient must **not** be scaled back. A
whiter residual map is not a scale-invariant target: a fit with an eighth of
the data genuinely wants a much stronger prior, and the criterion says so — 21
times stronger than the scaling law would predict. Since `--criterion auto`
takes `structure` on any well-constrained fit, scaling back would be wrong
where it matters most.

So `--cube-prior channel` (the default) fits the prior on a random
1-in-`n_chan` subset drawn across all channels — a dataset the size of the
ones the prior will be applied to — and uses the coefficient as fitted. The
subset is drawn from every channel rather than taken as one channel so that
the sky it sees is the band average, which matters for a line cube where a
single channel may be nearly empty.

`--cube-prior mfs` restores the old behaviour.

**Costs**, on Ruby CO(7-6) at `--fov 3` (613,512 samples, 8 channels, 27x27
mesh): 2.9 GB for the prior pass and each channel fit, against 20.1 GB for the
MFS pass. `pyuvimage fit` reports both figures before allocating anything.


## The sparse (w-tilde) inversion

`--inversion sparse` changes how `F = M^T N^-1 M` is computed, and nothing
else. The image, the criterion, the regularisation and the products are all
identical — on Ruby 200 GHz continuum (fov 3, mesh 16, matern, coefficient
1e8) the two paths give `chi^2 = 305200.43` and total fluxes agreeing to seven
significant figures. What changes is what it costs:

| | dense | sparse |
|---|---|---|
| Ruby continuum, fov 3, mesh 16 | 25.4 s | **0.3 s** |
| largest allocation | `n_vis x n_mesh` mapping matrix | `n_image x chunk_k` streaming buffer |
| Ruby CO(7-6) that allocation | 21.6 GB | 0.10 MB kernel |
| scales with the number of visibilities | yes | **no** |

The last row is the whole point, and it is the lesson taken from CASA's
`tclean`: stream the data onto a fixed-size grid rather than holding a matrix
whose size is the data. A dataset ten times larger costs ten times the *time*
in the kernel build and not one byte more memory. `chunk_k` — how many
visibilities are accumulated at once — trades the two against each other and
is chosen automatically to keep the build inside a quarter of available
memory, so there is no knob to find.

The kernel is the one expensive invariant of a run, and it depends only on the
uv coverage, the noise and the geometry — never on the data values or the
source prior. So it is cached on disk (`pyuvimage-<key>.wtilde.npy`, beside
the output unless `--kernel-cache` says otherwise) and a re-fit of the same
field with different regularisation skips straight past it. This is CASA's
`cfcache` idea, doing the same job.

The noise is in the cache key, not just the uv coordinates, and that matters
in practice: `--image-centre` recentres the field by an exact phase ramp,
which leaves every uv coordinate untouched while pooling the real and
imaginary sigmas. A key built on uv alone would hand the recentred fit a
stale kernel, and it would look perfectly healthy.

**Where the ceiling moves to.** The sparse path takes the data out of the
memory bill and leaves the model in it: `F` is `n_mesh^2`, so the limit on a
sparse fit is `--mesh`, not the size of the measurement set. Below about a
64x64 mesh the curvature matrix is not even the largest allocation — the
padded FFT batch is. Above it, `--mesh` is the only lever that matters.

**A scale trap, now guarded.** `apply_sparse_operator` builds the data vector
through `transformer.image_from(..., use_adjoint_scaling=True)`, while `W̃` is
accumulated straight from `1/σ²`. pynufft's adjoint carries its own internal
IFFT normalisation, so without that argument it comes back a factor
`4·N_y·N_x` low — and `D` and `F` end up on different scales. The symptom is
nasty precisely because it isn't an error: the reconstruction keeps the right
morphology at a tiny fraction of the right amplitude, the residual map retains
almost all of the source, and `χ²` stops depending on the coefficient at all,
so the hyperparameter search has nothing to bisect and runs to its ceiling.
pyuvimage now checks the transformer against `TransformerDFT` on a small
synthetic problem before every sparse fit and refuses to continue if they
disagree.

**It assumes σ_re = σ_im.** The `W̃` kernel is accumulated from the real
part's sigma alone, while the data vector weights the real and imaginary parts
separately — so where the two differ, `F` and `D` are built on different
weightings and the sparse and dense paths will not agree to better than about
that difference. `--image-centre` pools the sigmas in quadrature, so a
recentred fit satisfies the assumption exactly (and gets the better noise
estimate into the bargain); pyuvimage warns before a sparse fit when they
differ by more than 2%.

**Two things it will not do.**

* **It needs JAX.** The kernel build is pure NumPy, but the operator itself
  (`Khat`, and the FFT convolution that applies it) is `jax.numpy`. There is
  no fallback; `--inversion dense` needs nothing.
* **It refuses `--point-sources`.** Not for correctness — for cost. Point
  components are solved as a bordered system, and its cross-terms between the
  mesh columns and the point columns need `operated_mapping_matrix`: the dense
  `n_vis × n_mesh` build that the w-tilde path exists to avoid (21.6 GB on
  Ruby CO(7-6)). Asking for both would give up the entire benefit and most
  likely be OOM-killed, so pyuvimage raises and asks which one to drop.
* ~~It is MFS only.~~ **Cube mode works.** Each channel's uv coordinates are
  the same baselines in metres scaled by its own frequency, so each channel
  needs its own kernel — but a channel's build streams only that channel's
  visibilities, so `n_chan` builds over `n_vis/n_chan` each come to one pass
  over the dataset: the same total work as the single MFS kernel. Memory is
  per-channel and identical to a single-channel fit, because it depends on the
  image and the mesh rather than on how many visibilities a channel holds. The
  kernel cache keys on uv and noise, so the channels separate on their own.

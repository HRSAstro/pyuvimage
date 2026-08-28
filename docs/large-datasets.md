# Running the large datasets

Written for 9io9 at 135 GHz (164,262 unflagged visibilities, 4 spw) and Ruby
at 200 GHz (148,477, 2 spw). Both are ~30× larger than PJ0116, and both need
two things PJ0116 did not: a NUFFT, and a field of view that is not being
stretched to reach an off-centre source.

## 1. Get a NUFFT working

Two backends. **At these dataset sizes you want `pynufft`, not the JAX one** —
which is the opposite of what the speed comparison suggests, so it is worth
saying why before the installation notes.

The JAX transform itself is faster. But `TransformerNUFFT.transform_mapping_matrix`
pushes every mesh pixel through one batched `nufft2d2`, and nufftax
materialises that call's gather buffer in full: `n_mesh × n_vis × nspread²`
complex128, with `nspread = 14` at autoarray's default `eps=1e-12`. For Ruby
at a 20×20 mesh that is **186 GB** — to transform a mapping matrix that is
itself 0.5 GB. The process is killed with no traceback (`Killed: 9`), which is
what it looks like from the outside. pynufft's mapping-matrix transform is a
loop over mesh pixels and never holds more than one column.

`--transformer auto` now checks this before choosing and takes pynufft when
the JAX buffer will not fit, saying so in the log; `--transformer nufft`
forces JAX and splits the transform into blocks instead of dying. The full
arithmetic is in [parameters.md](parameters.md#why-auto-usually-picks-pynufft-over-jax).

So: install pynufft, and treat JAX as the thing that accelerates the rest of
the fit.

**`pynufft`** — `pip install pynufft`. Pure NumPy/SciPy, no JAX involved,
works under Rosetta. `--transformer pynufft` forces it.

This works on every autoarray version: PyAutoArray PR #475 (2026.8.23.1)
deleted the upstream `TransformerNUFFTPyNUFFT` class, so pyuvimage vendors its
own copy (corrected — see below) that talks to `pynufft` directly.

**`nufftax` (JAX-native)** — `--transformer nufft`. It needs two separate
things, and they fail independently.

**`nufftax` itself is not part of a default install.** Since PyAutoLens#702
JAX moved into autonerves' base dependencies while nufftax stayed behind in
autoarray's `optional` extra, so a freshly built environment usually has JAX
and no nufftax — and nothing says so, because the startup guard only speaks up
when JAX is *broken*. Install it explicitly:

```
pip install 'nufftax>=0.6.1,<0.7.0'
```

The floor matters for us specifically: nufftax <0.6.1 cannot differentiate a
batched `nufft2d2`, and `transform_mapping_matrix` — the call the inversion
makes for every mesh pixel — relies on it. `pyuvimage fit` now reports which
half is missing when it falls back.

**And JAX itself has to import**, which is not a given:

- On **Apple silicon, an x86 Python is fatal**. Every x86 `jaxlib` wheel is
  built with AVX, Rosetta does not provide it, and the error reads *"This
  version of jaxlib was built using AVX instructions, which your CPU and/or
  operating system do not support."* No `pip install -U jax` fixes it — the
  environment itself has to be an **arm64** build. Check with
  `python -c "import platform; print(platform.machine())"`; you want `arm64`,
  not `x86_64`.
- A jax/jaxlib version mismatch, the `jax-metal` plugin, or mixing conda-forge
  and pip installs of jax in one environment will each break it differently.
  `pyuvimage` survives all of these — `_jax_guard.py` catches the broken
  import and falls back to NumPy — but you lose the fast path.

**Does `nufftax` have the half-pixel bug too?** The pynufft transformer builds
a half-pixel phase ramp and then never applies it, so it sits half a pixel
from the DFT in both axes; pyuvimage corrects that. Whether `nufftax` shares
the convention was untestable here — no container available could run JAX — so
the check ships as a test instead. On a machine with JAX,

```
pytest tests/test_pynufft_transformer.py -q
```

runs `test_the_jax_nufft_agrees_with_the_dft`, which skips without JAX and
otherwise compares the two transformers directly. If it fails with an error
that grows with baseline length, `nufftax` needs the same correction, and
every JAX-path fit so far has been reconstructing the sky half a pixel off.
That is worth knowing before trusting a position.

## 2. Find the source before choosing `--fov`

Both of these sit several arcsec off the phase centre, which is easy to miss
and expensive to get wrong:

Measured from a 16″ dirty image at 0.1″/pixel. **These are image `x, y` —
+x right and +y up, as `--image-centre` and `--point` take them**, with the sky
pair alongside; RA increases leftward, so `x = −dRA`.

| | brightest peak | where the emission is |
|---|---|---|
| 9io9 135 GHz | 0.840 mJy/beam (84σ) at **x +1.75, y +2.25** (dRA −1.75, dDec +2.25) | an arc through (x +1.75, y +2.25), (+2.75, +0.15) and (+2.55, +1.15), centred near **(x +2.3, y +0.4)** — plus the counter-image, 51σ at **(x −1.95, y −0.65)** (dRA +1.95, dDec −0.65). Together they span x −2.25 to +3.05 and y −1.85 to +2.85 above 20σ, so a field holding *both* is centred near **(x +0.4, y +0.5)**. Below ~15σ there are features 6–8″ out; on a field this dominated by the arc, treat those as possible sidelobes until a fit says otherwise. |
| Ruby 200 GHz | 4.89 mJy/beam (216σ) at **x −2.05, y +2.85** (dRA +2.05, dDec +2.85) | a ring ~1.5″ across centred on that peak |

`--image-centre auto` does this for you: it makes a wide-field dirty image
(4× the requested field, capped at the primary beam) and recentres on the
brightest peak. That is the right answer for Ruby and the wrong one for 9io9,
where the peak is the arc's brightest *knot* and the counter-image is 4″ away
on the other side of the phase centre — so 9io9 wants the centre set by hand.

## 3. Pick a field, knowing what it costs

Memory is the binding constraint, not speed. The inversion builds an
`n_vis × n_mesh` matrix whatever the transformer, at roughly

```
peak RSS  ~  n_vis  ×  n_mesh_pixels  ×  44 bytes
```

That figure is for the DFT and pynufft paths. On the JAX path it is not the
binding term at all — see section 1 — and `pyuvimage fit` now reports the
estimate for whichever transformer it has actually chosen, before allocating
anything.

Measured on Ruby (148k visibilities) at 8″:

| mesh | pixel scale | per evaluation | peak RSS |
|---|---|---|---|
| 16 | 0.50″ | 12 s | 1.9 GB |
| 24 | 0.33″ | 25 s | 3.8 GB |
| 32 | 0.25″ | ~44 s | ~6.7 GB |
| 70 | 0.11″ (Nyquist) | ~210 s | ~32 GB |

A full fit is roughly 30–40 evaluations for the default `adaptive` prior (two
passes), plus the uncertainty map.

Because `n_mesh` goes as `fov²`, recentring is the single biggest lever:

| | field | mesh at Nyquist | peak RSS |
|---|---|---|---|
| Ruby from the phase centre | 8″ | 70 | ~32 GB |
| Ruby recentred on the ring | 3″ | 26 | **~4.4 GB** |

Same data, same resolution, seven times less memory.

### Or take the data out of the bill entirely

Everything above is about making `n_vis × n_mesh` small enough to hold.
`--inversion sparse` (experimental) removes that matrix altogether: one
streaming pass over the visibilities builds a small kernel, and the memory
then depends on the image size and the mesh but **not** on the number of
visibilities. Ruby continuum at fov 3, mesh 16: 0.3 s and ~1.7 GB against
25.4 s dense, for an identical `chi^2`. It needs JAX, is MFS only, and cannot
be combined with `--point-sources` — see
[parameters.md](parameters.md#the-sparse-w-tilde-inversion). Where it applies,
it is the answer to this whole section.

### Averaging the data down

**Averaging the data down first is usually the bigger win.** A modern dataset
carries far more channels and time samples than a small field needs, and
averaging before `pyuvimage import` costs nothing scientifically *up to the
point where smearing sets in*: channel averaging is limited by [bandwidth
smearing](https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/BandwidthSmearing.pdf)
(radial), time averaging by [time-average
smearing](https://www.cv.nrao.edu/vla/hhg2vla/node12.html) (tangential). Both
are worst at the field edge, so size them for the most distant emission you
care about. Past those limits, averaging resolves out real flux and neither
pyuvimage nor CLEAN can put it back.

## 4. The commands

`--image-centre` and `--point` take **image x, y** — +x right, +y up, as you
read them off `summary.png`. A leading minus needs the `=` form, or argparse
reads it as a flag.

Ruby — the ring is compact, so recentred it fits comfortably:

```
pyuvimage fit Ruby_200GHz_cont.npz \
    --fov 3 --image-centre auto --transformer pynufft \
    --out ruby_out
```

`auto` lands on (x −2.05, y +2.85); `--image-centre="-2.05,2.85"` is the same
thing written out.

9io9 — the arc is ~4″ long and the counter-image sits 4.7″ away on the other
side of the phase centre, so `auto` is not the right call here: it would
centre on the arc's brightest knot. Two reasonable choices:

```
# the arc alone, centred on it, at full resolution (~9 GB at Nyquist,
# ~5 GB at --mesh 26)
pyuvimage fit 9io9_135GHz_cont.npz \
    --fov 5 --image-centre "2.3,0.4" --transformer pynufft --out 9io9_arc

# arc and counter-image together, centred between them, coarser
pyuvimage fit 9io9_135GHz_cont.npz \
    --fov 7 --image-centre "0.4,0.5" --mesh 32 \
    --transformer pynufft --out 9io9_wide
```

Which is better science depends on the question. The counter-image is part of
the same lensed source, so a lens model wants both; if you only want the arc's
own structure, fitting it alone buys resolution everywhere.

## 5. What to look at first

1. **The structure ratio** the run prints. Near 1 means the residual map is
   noise. Above ~1.5 the fit left real emission behind; below ~0.85 it
   absorbed noise.
2. **Whether `chi^2/N` could reach its target.** If the log says *"the
   constrained fit cannot go below chi2/N = …, so the coefficient is chosen
   against … rather than 1"*, this dataset has the same floor PJ0116 had, and
   the coefficient was chosen at the knee instead. That is working as
   intended — see `design-notes.md`.
3. **`residual.fits`**, always, before trusting a flux.

## Notes and caveats

- 9io9 spans 74 hours — several execution blocks concatenated, 50 timestamps
  at ~91 s. The noise estimator's calibrator-gap guard (3 × the median time
  step) keeps it from differencing across the joins.
- The exported noise now agrees with a fresh `difference` estimate to 3.6%
  (9io9) and 1.6% (Ruby), so the export-side pooling bug is genuinely fixed.
  On PJ0116 that same comparison was off by 1.41×.
- There is no w-tilde path in this autoarray, which is what would remove the
  `n_vis × n_mesh` matrix entirely and make field of view cheap again. That is
  the real fix if these dataset sizes become routine.

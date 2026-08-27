# Running the large datasets

Written for 9io9 at 135 GHz (164,262 unflagged visibilities, 4 spw) and Ruby
at 200 GHz (148,477, 2 spw). Both are ~30× larger than PJ0116, and both need
two things PJ0116 did not: a NUFFT, and a field of view that is not being
stretched to reach an off-centre source.

## 1. Get a NUFFT working

Two backends, and which one you can have depends on the environment.

**`nufftax` (JAX-native)** is the default and the faster of the two —
`--transformer nufft`. It needs two separate things, and they fail
independently.

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

**`pynufft`** is the pure NumPy/SciPy alternative: `pip install pynufft`, no
JAX involved, works under Rosetta. `--transformer auto` picks it up once the
DFT stops being affordable; `--transformer pynufft` forces it.

This works on every autoarray version: PyAutoArray PR #475 (2026.8.23.1)
deleted the upstream `TransformerNUFFTPyNUFFT` class, so pyuvimage vendors its
own copy (corrected — see below) that talks to `pynufft` directly.

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

Measured from a 16″ dirty image, in arcsec from the phase centre
(+RA East, +Dec North):

| | brightest peak | where the emission is |
|---|---|---|
| 9io9 135 GHz | 0.96 mJy/beam (96σ) | an arc running from (dRA −2.7, dDec +0.1) through (−1.7, +2.3) to (−2.5, −1.5) — ~4″ long, centred near **(−2.3, +0.3)** — plus a compact 58σ companion at **(+1.96, −0.62)**. A handful of 12–14σ features sit 6–8″ out; at that level, on a field this dominated by the arc, treat them as possible sidelobes until a proper fit says otherwise. |
| Ruby 200 GHz | 6.46 mJy/beam (285σ) | a ring ~1.5″ across centred at **(+2.06, +2.81)** |

`--image-centre auto` does this for you: it makes a wide-field dirty image
(4× the requested field, capped at the primary beam), reports the brightest
peak in dRA/dDec, and recentres there. `--image-centre "dRA,dDec"` sets it by
hand — which is the better choice for 9io9, because `auto` finds the arc's
brightest *knot* at (−1.71, +2.29) rather than the arc's centre.

## 3. Pick a field, knowing what it costs

Memory is the binding constraint, not speed. The inversion builds an
`n_vis × n_mesh` matrix whatever the transformer, at roughly

```
peak RSS  ~  n_vis  ×  n_mesh_pixels  ×  44 bytes
```

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

## 4. The commands

Ruby — the ring is compact, so recentred it fits comfortably:

```
pyuvimage fit Ruby_200GHz_cont.npz \
    --fov 3 --image-centre auto --transformer pynufft \
    --out ruby_out
```

9io9 — the arc is ~4″ long and the companion sits 4.7″ away on the other side
of the phase centre, so `auto` is not the right call here: it would centre on
the arc's brightest knot. Two reasonable choices:

```
# the arc alone, centred on it, at full resolution (~9 GB at Nyquist,
# ~5 GB at --mesh 26)
pyuvimage fit 9io9_135GHz_cont.npz \
    --fov 5 --image-centre "-2.3,0.3" --transformer pynufft --out 9io9_arc

# arc and companion together, centred between them, coarser
pyuvimage fit 9io9_135GHz_cont.npz \
    --fov 7 --image-centre "-0.4,0.0" --mesh 32 \
    --transformer pynufft --out 9io9_wide
```

Fitting the arc alone is the better science: the companion is 4″ away and
nothing in the model couples them, so including it only costs resolution
everywhere.

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

# Source priors: what they are and how they compare

[← back to the README](../README.md)

## Sparse visibilities: the envelope prior

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

## A bright compact core: the adaptive prior

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

## Choosing a prior

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
sparse coverage rather than a general default. The correlation figures are not
comparable between the two tables: they are measured against truth on the
finer product grid, where the block-replicated model saturates around 0.9.

**Why `adaptive` with power 2 is the default.** It gives the best extended
model of the variants tested without overfitting, and — the decisive
measurement — it removes a central residual that `gibbs` leaves behind and
that does *not* scale with source brightness (see "Reading the residual
map" in `design-notes.md`).
`gibbs` shortens the prior's correlation *length* where the source is bright,
which strengthens the penalty there and suppresses the peak; `adaptive`
loosens the prior's *amplitude* instead. On a single bright compact peak
`gibbs` is still the sharper choice, so it remains one flag away
(`--reg gibbs`). `--adapt-power` changes the exponent.

## Extended source + unresolved off-centre knot

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
direct symptom of mis-fitting compact emission.

That last line is why `gibbs` was briefly the default. It is still the right
choice for a single bright compact feature, but it buys that sharpness by
strengthening the prior at the source centre, and on an extended source that
leaves a central residual which does not scale with brightness (see "Reading
the residual map" in `design-notes.md`). `adaptive` with power 2 does not, so it is the default;
`--reg gibbs` is one flag away.

These are one mock and one noise realisation; differences of ~0.01 in
correlation are not meaningful.

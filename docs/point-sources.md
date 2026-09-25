# Analytic point-source components, in full

[← back to the README](../README.md)

A genuine point source is the one thing a pixel grid cannot represent. Its
visibilities are exact and closed-form,

    V(u, v) = A exp(-2i pi (x u + y v))

so the sensible thing is not to put it on the grid at all. `--point-sources`
adds analytic delta components whose amplitudes are solved **in the same
linear system** as the mesh (Schur complement on the augmented normal
equations); only the position is non-linear, and it is refined by a lattice
scan followed by Nelder-Mead. Point fitting itself is opt-in and off
by default; when it is on, the regularisation retune below is on with it.

Why it is worth doing: on the test data a nearest-pixel delta half a pixel
off-centre misrepresents the source at chi^2/N = 31.5, and the best *gridded*
Gaussian still leaves ~1.9 — an error at or above the noise, for a source the
model is meant to describe perfectly.

```bash
pyuvimage fit mydata/ --fov 3.0 --point-sources          # auto-detect
pyuvimage fit mydata/ --fov 3.0 --point 0.70,0.80        # you supply it
```

A supplied position is kept and refined.

**Detection is a matched filter, not a peak finder.** The obvious approach —
take the brightest pixel of the residual dirty image — fails, and it took a
written-products run to expose how badly: the mesh fit has by construction
been driven to chi^2 = N and has already absorbed much of the compact source,
so what is left in the residual is sidelobe structure. On one mock that gave
five "sources" spread over half an arcsecond, four of them with *negative*
flux, and the real 0.012 Jy knot missed entirely. Instead, every trial
position on the product grid is asked the right question — how far would the
fit improve if a point were added *here*, with the mesh free to re-adjust —
which the Schur elimination answers in closed form for one extra column:

    a_j = r_j / s_j,   Var(a_j) = 1/s_j,   s_j = C_jj - b_j^T M^-1 b_j

That is one BLAS call per chunk over the whole field, and it already accounts
for the mesh's ability to mimic a point. Candidates then have to survive:

| guard | what it prevents |
|---|---|
| positive amplitude only | a delta patching a residual trough; a negative "source" is not sky |
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
| mesh + point | 1.00 | 5.67 sigma | 0.01180 +- 0.00024 | 0.004" |
| control: disc only, auto-detect | 1.14 | 11.1 sigma | no point accepted | — |

![point sources](../figures/point_sources.png)

**Re-tuning the regularisation (on by default; `--no-point-retune` disables it).** The
strength is chosen by the discrepancy principle with the compact flux forced
through the mesh. Once a point carries it, the mesh has freedom it no longer
needs and the combined fit lands below the target — chi^2/N = 0.61 on this
mock, the signature of a mesh now fitting noise. The retune re-imposes
chi^2 = N by stiffening the prior (here by 8e6, coefficient 6.2e3 -> 5.1e10),
which is the same regime the *disc-only* control independently optimises to.

**It re-tunes on the criterion the search used.** Until Sep 2026 the retune
ran only under `discrepancy`. On a large dataset `auto` picks `structure`
(chi^2 is flat there: 0.99995-1.00076 over twelve decades of the coefficient
on one 1.8e7-datum field), so nothing re-tuned, and the delivered prior was
the one the *mesh-only* search chose -- tuned to let the mesh chase the
point's residual. With the point carrying that flux the mesh then fits noise:
structure ratio 0.40 on the real field, 0.14 on a mock of it. Worse, the
split between the point and the mesh under it is set by a prior too weak to
set anything: on a small mock a true 12 mJy point came back at **-86 mJy**,
the mesh carrying the difference. Under `structure` the retune now brings the
structure ratio of the combined (mesh + point) residual to 1 -- on that mock
by a factor of ~8700, recovering 12.5 mJy. A coefficient fixed with
`--lambda` is the user's and is not re-tuned (nor, now, re-optimised on the
adaptive refit, which it silently was).

*Known limitation.* The retune scales one global strength until the *whole*
residual map reads as noise, and that can be met by moving the misfit rather
than removing it. On a mock of a 10 mJy point on diffuse emission with two
narrow streamers: under `adaptive` the retuned fit leaves a +-5 sigma ring
around the point and the flux comes out 8.5 +- 1.4 mJy; under `matern` the
point is exact (9.97 +- 0.07 mJy) but the streamers, narrower than the beam,
sit in the residual at +-4 sigma. Before the retune both fits were overfit
(structure ratio 0.16). Look at `residual.fits` around the point.

**Positions you supply are checked too.** They are kept whatever their
significance, but a component that comes back *negative* is dropped with a
warning, and one that a Gaussian fits better (the same unresolved test
auto-detection applies) is flagged: its flux is then split between the point
and the mesh by the prior, not the data. On mocks a negative component at a
user position is the signature of a resolved compact source -- the mesh
describes it and overshoots slightly at its centre -- and it stays negative at
every prior strength. See `claude/central-point-mock-reproduction.md`.

| | extended model | knot flux (truth 0.01200) | peak residual |
|---|---|---|---|
| mesh only, no point | striped by beam sidelobes at +-5e-5, half the disc's peak | knot smeared into the mesh | 2.42 sigma |
| point, `--no-point-retune` | mottled at +-1e-4, i.e. fitting noise | 0.01204 +- 0.00046 | **0.48 sigma** |
| point + retune (**default**) | smooth and disc-like | 0.01180 +- 0.00024 | 5.67 sigma |

Compare panels 2-4 of the figure against the truth in panel 1: the retuned
model is much the closest, and the striping in the mesh-only panel is what a
prior tuned around an unmodelled compact source costs.

The retune's own cost is that the point's *statistical* error is conditional
on the stiffer prior and is far too small on its own — 3.3e-5 here, which
would make a 2%-low flux a 7 sigma discrepancy. So the quoted `flux_error` is
not the statistical error alone: it adds, in quadrature, how far the amplitude
moves when the regularisation strength is varied over the range these data
allow. Both terms are written separately to `point_sources.json`
(`flux_error_stat_jy`, `flux_error_sys_jy`). Across the generalisation tests
below that turns pulls of up to 24 sigma into pulls under 3. Detection
significance still uses the statistical error alone — a scale uncertainty
should not make a real source look marginal.

Point components appear in `point_sources.json` with positions, fluxes and
1 sigma errors; in `model.fits` as flux dropped into the nearest pixel (the
grid cannot hold a sub-pixel delta, so that file is flux-correct but
positionally quantised); in `model_reconvolved.fits` placed analytically at the fitted
sub-pixel position; and in the header as `NPOINTS` and `PTFLUX`. The mesh
uncertainty map is marginalised over the point amplitudes,
`Cov = M^-1 + (M^-1 B) S^-1 (M^-1 B)^T` — ignoring the second term would
understate the error wherever a point competes with the mesh for flux.

**On the sparse (w-tilde) inversion.** The bordered system needs two things
from the mesh: the cross-terms `B = A^T W P` between the mesh columns and the
point columns, and `A s` for the model visibilities. Both used to be read off
`inversion.operated_mapping_matrix` — the dense `n_vis x n_mesh` build — which
is why `--point-sources` forced `--inversion dense` (21.6 GB on Ruby CO(7-6)
against a 0.10 MB kernel: asking for both gave up the whole point of sparse).

Neither actually needs the matrix. `A = F M`, the real-space mapping matrix
followed by the Fourier transform, so

    A^T W P = M^T Re(F^H (w_re Re(P) + i w_im Im(P)))
    A s     = F (M s)

— the first is the *dirty image of the point column* projected onto the mesh,
one adjoint transform per column; the second is one forward transform of one
image. `pointsource.SparseMesh` does exactly that, and `pointsource.DenseMesh`
keeps the old stacked-matrix route for the dense path. They agree to 5e-16 on
the cross-terms and 7e-14 on the forward direction.

The amplitudes then differ between the two paths at ~5e-7 relative, which is
*not* the backends disagreeing: `F + H` has a condition number around 1e10 (the
mesh has pixels the uv coverage barely constrains), so a 1e-16 difference
anywhere reaches the amplitudes at ~1e-6. Perturbing the dense cross-terms
randomly by the same 5e-16 moves them roughly ten times *further*, which is
what `test_the_difference_is_conditioning_not_error` asserts.

The detector needed one thing more. It scores a trial point at every pixel of
the image grid, and one adjoint transform per trial is the wrong price —
measured at 1e5 visibilities, 1.3 s x 2304 lattice positions is 50 minutes.
But a trial position on a pixel centre is a delta on the grid, so

    b_j = A^T W P_j = M^T F^H W F e_j = M^T W~ e_j

and the whole lattice at once is `(W~ M)^T`: the kernel applied to the mesh's
few hundred mapping columns rather than to the lattice's few thousand, in one
batched FFT. 2400x faster, agreeing with the exact route to 1.6e-14. It
inherits the kernel's `sigma_re == sigma_im` assumption — which the sparse
inversion's own `F` already makes — and a lattice that is not on the grid
declines it and falls back. It is only the detector: accepted positions are
refined and solved through the exact per-column route.

**Streaming is the part still held back**, and for a different reason: a point
column is an analytic function of uv, and all three of its terms are sums over
samples the streamed pass has already discarded. The accumulated terms hold
those sums only on the image grid, and a point's whole reason for existing is
that it is not on the grid. So `--point-sources` runs the sparse inversion in
memory: ~136 B per visibility for the data, rather than the `n_vis x n_mesh`
matrix it used to need.

**Limits.** The amplitude covariance is conditional on the prior, so a point
sitting on bright extended emission has an error bar that is only as good as
the prior's description of that emission. Detection has no look-elsewhere
correction: 5 sigma is per trial position, not per map. And the resolved-vs-
unresolved threshold (delta chi^2 = 9) was set on mocks, not on real data.

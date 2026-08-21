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

**Limits.** The amplitude covariance is conditional on the prior, so a point
sitting on bright extended emission has an error bar that is only as good as
the prior's description of that emission. Detection has no look-elsewhere
correction: 5 sigma is per trial position, not per map. And the resolved-vs-
unresolved threshold (delta chi^2 = 9) was set on mocks, not on real data.

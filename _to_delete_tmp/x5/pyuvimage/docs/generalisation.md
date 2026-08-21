# Generalisation tests

[← back to the README](../README.md)

Everything else in these docs was tuned on one mock: an exponential disc with
one knot. `scripts/generalisation_tests.py` runs three harder families and scores them
identically; `scripts/generalisation_figures.py` draws each fit in the same
layout as `summary.png`. In all of them the point sources are injected
**analytically** at sub-pixel positions, so nothing in the truth sits on the
model's grid: a point placed on the truth image is a source the pixelized
model can already represent, and recovering it would prove nothing.

The field: 68 mJy of extended emission in three components (a 0.8" disc, a
bright 0.25" blob, a faint 0.5" ellipse) plus four true points from 1.5 to
12 mJy, one isolated, one on the disc, one buried in the bright blob, one
faint and isolated.

![generalisation](../figures/generalisation.png)

| test | chi^2/N | total flux | points found | false |
|---|---|---|---|---|
| crowded field | 1.001 | x0.999 | 4/4 | 0 |
| coarse beam, b_max 300 m (beam 1.13") | 1.000 | x1.000 | 1/4 * | 0 |
| nominal, b_max 800 m (beam 0.48") | 1.000 | x1.000 | 3/4 | 0 |
| fine beam, b_max 2.5 km (beam 0.16") | 1.001 | x0.995 | 4/4 | 0 |
| wide field, 8" instead of 4" | 1.001 | x0.998 | 4/4 | 0 |
| high S/N (sigma 3e-5) | 1.001 | x1.000 | 4/4 | 0 |
| nominal S/N (sigma 3e-4) | 1.001 | x0.999 | 4/4 | 0 |
| low S/N (sigma 1.5e-3) | 1.000 | x0.998 | 2/4 | 0 |
| **fov 2" for a 3" field** | **350** | x0.914 | 0/4 | fit refused |

`*` not a failure: with a 1.13" beam and 0.2" mesh pixels the mesh represents
an unresolved source perfectly well, so a delta component buys little and the
detector mostly declines. Total flux is still exact — the point flux simply
lives in the mesh.

**Every converged case sits at chi^2/N = 1.000-1.001 with total flux recovered
to 0.5% or better, and there are no false positives anywhere.** Positions come
out to 0-33 mas against beams of 155-1130 mas. Detection is conservative by
construction: 26 of the 28 recoverable points were found, and the ones missed
are the 3 mJy point buried in the bright 0.25" blob and the faintest point at
low S/N.

**Where it is still weak.** 22 of the 26 detections are within 3 sigma of
truth; four are not. The 3 mJy point sitting on the bright blob is the
recurring problem — recovered at x0.52 in the crowded field, missed outright
in three others: a point on top of bright compact extended emission is
genuinely degenerate with it, and the prior is what breaks the tie. The other
outliers are at the highest S/N (pulls of 7-8 on fluxes 2% off), where the
systematic term stops keeping up; see the uncertainty section.

Below, the same nine fits drawn out — dirty image, model, reconvolved model,
residual and total uncertainty, with true points marked in green and fitted
ones circled in cyan. The last row is the deliberate out-of-field failure.

![generalisation summaries](../figures/generalisation_summaries_all.png)

Six real defects came out of this, all fixed:

1. **The retune crashed** (`LinAlgError`) whenever it weakened the prior far
   enough that the curvature matrix alone went singular — which happens as
   soon as the mesh is comparable in size to the data. It now stops at that
   boundary and says so.
2. **Positions were refined under the pre-retune prior and never re-polished.**
   A stiffer mesh moves where the best point position is, so amplitudes were
   read off stale positions: at high S/N that was a 20% flux error with 30 mas
   offsets. Strength, positions and the significance cut now iterate together.
3. **Components survived on a pre-retune significance.** The retune changes
   the errors, so the cut is applied again afterwards — that is what removed
   the last spurious detection at high S/N, which fell from 20 sigma to 3. A
   component whose amplitude has gone negative is dropped for the same reason.
4. **The non-negative solver silently ignored the prior** on the coarse-beam
   data, returning chi^2/N = 1.159 to four figures for every strength from
   1e-6 to 5.9. The existing check compared it with the unconstrained solve at
   one strength and missed this; it now also compares two strengths twelve
   decades apart, which no real prior can match.
5. **The adaptive prior was loosened by the very points it was fitting.** It
   follows a first-pass brightness map, and that map has the point smeared
   into it — so the prior ended up weakest exactly where the point sat and the
   mesh underneath soaked up its flux. The 3 mJy point on the bright blob came
   back at 56% of its flux, the 12 mJy point at 94% with a -7.5 sigma pull.
   Once the points are known the map is rebuilt from the extended model alone
   and the fit repeated: 110% and 100%, and the false positives disappeared.
6. **Point flux errors were statistical only**, giving pulls to 24 sigma. A
   prior-strength systematic is now added in quadrature.

**What still fails, loudly.** A field of view that does not contain the
emission is unrecoverable: chi^2/N runs to 350, and the point fitter, given a
residual that is model error rather than sky, previously returned an 11.5 Jy
"source" at 76 sigma in a 0.09 Jy field. Point fitting is now refused outright
when the pixelized model has not converged to its target, and the run warns
that no product should be trusted. Fix `--fov` first.

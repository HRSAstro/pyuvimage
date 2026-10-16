# Adaptive meshes and emission masks

[← back to the README](../README.md)

The default mesh is a rectangle of `--mesh` × `--mesh` pixels spread evenly over
the field. Two options spend the mesh where the emission is instead, on a
Delaunay triangulation (autoarray's `Delaunay` mesh, linear interpolation
inside each triangle). Both run on the w-tilde (sparse) path at any size: the
matrix-free solve needs the rectangle's grid.

## `--mesh-type hilbert`: points that follow the brightness

After PyAutoLens's Hilbert image-mesh. A quick first pass (plain Matérn,
unconstrained, on a coarse rectangle of ¼ the nominal mesh per side, between 16
and 48) gives an *adapt image*; the image is read along a Hilbert
space-filling curve and `--mesh-points` points are drawn with density

    max((adapt⁺ / peak)^power, floor)

(`--hilbert-power`, `--hilbert-floor`). `adapt⁺` is the adapt image with its
**negatives clipped**: PyAutoLens weights by |adapt|, and an interferometric
first pass has negative bowls and sidelobes on which |adapt| put dense points.
A ring of points just outside the field keeps every image pixel inside the
triangulation. `--mesh` still sets the image grid and the nominal resolution.

**When it pays.** On a compact three-component source in a 6″ field, 1064
Hilbert points matched a 64² (4096-pixel) rectangle — better on the bright
core, equal on a faint knot — in 12 s against 155 s, and the n_mesh² matrices
are 16× smaller. Emission that fills much of the field gains less: NGC 4274
needed ~4000 points to match its 192² matrix-free model. Below ~1000 points
faint components are undersampled and show triangular facets.

### Cubes: `--cube-mesh`

A cube shares the MFS fit's prior across its channels. With
`--cube-mesh channel` — the default (`auto`) for a Hilbert mesh — each channel
also gets its **own** Hilbert mesh, drawn from a quick first pass on that
channel's data, so the points follow where that channel is bright: a line that
moves across the source (a rotating disc) is sampled where it is in each
channel, not spread over everywhere it is in any. The prior's strength stays
the MFS fit's; an adaptive prior's brightness map is each channel's own (see
below). The mask, if any, applies to every
channel. Cost: one coarse first pass per channel (seconds to ~15 s each).
`--cube-mesh shared` fits every channel on the MFS mesh.

On a mock where a compact source moves across three channels, 60 points per
channel at a fixed prior: rms error 0.054–0.074 of the peak per channel with
per-channel meshes, against 0.065–0.095 with the shared one.

**Caution — the shared prior on a moving line.** The prior's strength is
still chosen on the MFS data, which assumes one sky for every channel. Where
the emission moves, no single image reproduces all channels (on that mock the
MFS fit sat at χ²/N ≈ 1400, its chosen prior was far too strong, and the
channels lost half their flux). For such cubes, fix `--coefficient` from a
channel fit until the prior can be chosen per channel.

**The adaptive brightness map is per channel.** It used to be frozen from the
MFS fit along with the coefficient. On a line cube that is badly wrong: the
MFS model is the emission summed over velocity, normalised to its brightest
knot, and with weights w = b² nearly all of one channel's emission sits at the
map's 1% floor — a prior ~10³ times stiffer there than the MFS fit was tuned
for. On NGC 4274 CO(2–1) (37 channels, mask, Hilbert) every line channel was
left with 20–45σ residuals tracing the emission. On one channel (3000 points,
the same coefficient): MFS map 31σ peak residual, structure ratio 3.3; its
own map 8–11σ, 1.8. Each channel now runs its own first pass (plain Matérn)
at the frozen prior — one extra solve, no search — and uses that model as its
map. The rest of that residual is the frozen coefficient: the MFS fit of a
line cube cannot reach χ²/N = 1 (below), so its strength comes from the
evidence and is several times too strong for a channel (discrepancy on the
channel alone: 1.4×10⁶ against 7.4×10⁶, 7σ).

`--cube-prior per-channel` lets each channel choose its own strength. On three
NGC 4274 line channels (3000 points, mask) with the default criterion
(structure, now that `auto` counts the Hilbert points: 47 data per point):
strengths 1.5–4×10⁴, residual rms 1.01σ inside the mask, peaks 4–5σ, against
1.8–1.9σ and 7–10σ at the shared 7.4×10⁶. The weaker prior also puts some
faint emission across the mask (model flux 1.5 Jy against 1.0; flux in the
region above 3σ 1.1–1.16 Jy against 0.9). `--criterion discrepancy` is erratic
per channel here (133 to 1.4×10⁶): under positivity the χ² floor sits within
1% of the target, so χ² barely constrains the strength.

## `--mask`: emission only where you allow it

A mask says where the sky may hold emission. The mesh's free points are placed
only inside it, and a ring of points around it is **held at zero** (autoarray's
`zeroed_pixels`, enforced by the non-negative solver — so a mask implies
positivity). With `--mesh-type rectangular` the free points are the rectangle's
own pixel centres inside the mask; with `hilbert` they are drawn inside it.

**Why.** A bright extended source whose integrated flux is far above its peak
brightness turns the dirty beam's faint large-scale sidelobes — the few
shortest baselines' plane waves — into strong bands across the dirty image.
On NGC 4274 (0.65 Jy against an 8.7 mJy/beam peak) they are ~40σ, and
continue far outside the primary beam. Without short spacings the data cannot
tell them from real broad emission, so an unmasked positive fit puts about half
its flux into bands across the field; an unconstrained one adds negative
bowls. With a mask round the galaxy (11.5% of the field) the model holds
0.69 Jy, all in the galaxy, and the bands are explained as its sidelobes.

**Reading the residual.** What the mask excludes shows up in the residual —
which is the check that the mask is not too tight. On NGC 4274 the masked
residual is white across the galaxy, with a few compact >5σ blobs outside it
(real emission the mask leaves out: widen it there) and ±3σ broad structure
(the large-scale emission the data barely constrain).

**Choosing the strength with a mask.** The `structure` criterion asks the
residual map to look like white noise. Outside a mask it cannot: it holds what
the model is forbidden to fit, so over the whole field the target was
unreachable at any strength and the search fell back to the evidence — on NGC
4274 a weaker prior that overfitted inside the mask (residual 0.82 of white
there). With a mask the ratio is measured inside it, dilated by one beam; the
target is then reached (0.99 on NGC 4274), and the run reports the residual
outside the mask separately.

**χ² above target at every strength.** On NGC 4274 every MFS fit — masked or
not, constrained or not — sits at χ²/N = 1.355, the same as with the prior
switched off. That excess is in the data, not in the fit; the warning says so
when the delivered χ² is within 1% of the weakest prior's, and suggests
`--chi2-target` at that floor. On NGC 4274 it is **the line**: the dataset is
CO(2–1), one MFS image cannot reproduce emission that changes from channel
to channel, and the line-free channels of the cube fit sit at χ²/N =
0.98–1.00 (so the noise map is right). An MFS fit of line data should be read
as a moment-0 sketch, not a fit to the noise.

**Making a mask.**

- A **CASA region file** (`.crtf`) from the CASA viewer or CARTA — ellipses,
  boxes, polygons, in sky or pixel coordinates; regions prefixed `-` are cut
  out. DS9 `.reg` files work too. Needs `pip install regions`.
- **Draw one**: `pyuvimage mask dirty_image.fits --out mask.fits` opens
  autogalaxy's painter over any FITS image (a fit's `dirty_image.fits`, a CASA
  image): `1` paints emission, `2` erases, `=`/`-` resize the brush, `z` undoes,
  Esc saves. `--start region.crtf` refines an existing mask.
- Any **FITS mask** with a celestial WCS (non-zero = emission), resampled onto
  the fit grid by WCS.

Masks are placed through the fit grid's WCS, so they follow `--image-centre`.

**Not yet:** point components with a mask or Hilbert mesh, and masks on the
matrix-free path (a mask always moves the fit onto the w-tilde path).

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

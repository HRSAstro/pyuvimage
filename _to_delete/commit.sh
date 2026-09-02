cd $HOME/mnt/Work/pyuvimage
unlock() { for l in .git/*.lock; do [ -e "$l" ] && mv "$l" _to_delete/$(basename $l).$RANDOM 2>/dev/null; done; }
G="git -c user.name=HRSAstro -c user.email=hrstacey@icloud.com"
T="Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V4AtTWc5c27eTcL5eRWWt8"
c() { unlock; git add "$@" 2>&1 | grep -v unlink; unlock; $G commit -q -F /tmp/msg.txt 2>&1 | grep -v unlink; unlock; git log --oneline -1; }

cat > /tmp/msg.txt <<M
beam: half-pixel restore shift, BPA sign, mirrored point beam, dirty-image scale

Four coupled fixes in beam.py and one north-up expression for every Gaussian:

* gaussian_kernel was evaluated about the geometric centre (n-1)/2 while
  fftconvolve(mode="same") centres on index (n-1)//2. They differ on even
  grids, which resolve_geometry always produces, so every restored extended
  image was shifted (+0.5, +0.5) px relative to the residual and the points.
* fit_beam rotated in row-down coordinates and returned the sky PA negated;
  BPA in every header was wrong in sign. No test checked it and the comment
  claiming CASA agreement was untrue.
* restore_points was north-up, gaussian_kernel row-down: points were painted
  with the beam mirrored relative to the extended emission.
* DirtyImager normalised by the sampled beam peak (0.92 sum(w) on the mock,
  half a pixel off on an even grid) while rms assumed sum(w): dirty images
  were 5-9% too bright next to the noise, and the structure ratio read
  1.06-1.09 for a white residual. Normalise by sum(w).

Tests: PA east of north at seven angles and both major-axis branches, the
kernel regenerated from a fit matching the beam on an even grid,
restore_points == gaussian_kernel, centroid-exact restore on even and odd
grids, and the weight-sum normalisation against the analytic rms.

$T
M
c src/pyuvimage/beam.py tests/test_beam.py

cat > /tmp/msg.txt <<M
data layer: flags honoured on import, fields carried, one pair helper, vectorised noise

Bugs: ms_import estimated the noise on flagged cells written as 0.0 (fully
flagged edge channels alone gave 0.7x truth; sporadic flags with a bright
source 3x) -- flagged cells are NaN before every estimator now, as
casa_export and recompute_noise already did. import_ms returns
antenna1/antenna2/time/weight_sigma so --noise scaled/hybrid work on imported
data. recompute_noise on a multi-spw dataset updates the top-level
noise_estimate that fit_parameters.json records. UVData.select keeps the
re-estimation fields. casa_export._resolve_spws("0-3") expands the range.
noise_time_variation gets the gap guard the other estimators had.
baseline_percentile_wavelengths is per-spw like max_baseline_wavelengths.

Redundancy: the group-by-baseline / sort-by-time / difference / gap-guard
block existed five times in noise.py and once in casa_export; one
adjacent_pairs + grouped_std pair replaces them all, pinned to a naive loop
at 1e-12 on ragged, gapped, NaN-laden data. ms_import computes the
whole-track estimate once instead of two or three times.

Performance (2e5 rows, identical results): sigma_in_time_chunks 4.4x at 8
channels, 2.4x at 64; flattened and shift_image_centre ~2x. np.take on a
non-leading axis is ~7x faster than fancy indexing there, and a 2-D boolean
mask on a 3-D array ~15x slower than a flat take.

$T
M
c src/pyuvimage/uvdata.py src/pyuvimage/noise.py src/pyuvimage/ms_import.py src/pyuvimage/casa_export.py tests/test_casa_export.py tests/test_ms_import_noise.py tests/test_noise_vectorised.py tests/test_uvdata.py

cat > /tmp/msg.txt <<M
pointsource: stacked real A, memoised columns, budgeted scan chunks

A.real.T @ ... materialised a strided copy of the complex mapping matrix on
every call of _curv/_dvec -- 1.3 s vs 0.14 s at 1e5 visibilities x 576 mesh
pixels -- and solve() runs ~600 times per two-point auto-detection. The
system now stores [A.real; A.imag] contiguously (memory-neutral) and every
quadratic form is one real GEMM. Trial columns for the scan are built by
broadcasting as stacked real cos/sin; per-lattice terms are memoised so later
detection iterations compute only the accepted-point rows; per-position
column terms are cached across retune and refine, where only h_scale or one
column changes; the scan chunk is sized from a memory budget instead of a
fixed 1024 (1.6 GB at 1e5 visibilities, which OOM-killed the old code).

End-to-end on the crowded mock: auto-detect 6.9 s -> 1.1 s, two user
positions 13.4 s -> 0.7 s; positions and fluxes agree to 1e-8, chi^2 to
1e-10. Also: chi_squared_with_width goes through solve(); _best_candidate
uses <= radius so a rejected candidate is not re-picked forever at radius 0;
fit_point_sources uses the dirty_imager it accepted and ignored.

$T
M
c src/pyuvimage/pointsource.py tests/test_pointsource_fast_path.py

cat > /tmp/msg.txt <<M
fitting: build the linear system once per fit, not once per trial

The coefficient search rebuilt A_t = transform_mapping_matrix(A), F and D on
every trial although none depends on the coefficient (or on the correlation
scale). Profile on the mock: 43.6 s, 28 FitInterferometer builds, 30 s in
the transform, 82 data_vector and 106 curvature_matrix recomputations
(autoarray exposes them as plain properties) -- and 0.3 s in the solves.

LinearSystem reads F and D off one framework inversion (the same objects on
the dense mapping class and the sparse w-tilde class, so the sparse path is
covered without touching operated_mapping_matrix, which there would trigger
the dense build the path exists to avoid), solves each trial with the
framework's own public solvers, and reproduces fast_chi_squared and
log_evidence_from op for op. The delivered fit is one real FitInterferometer
at the chosen hyperparameters. Held to the pre-change results on six
configurations: coefficient, chi^2, evidence and reconstruction all bitwise
identical; 19x-95x faster (evid_2d 215 s -> 3.7 s, gauss_env 340 s -> 3.6 s).

Also: _chi_squared, _safe_evidence, structure_ratio and
SingleFit.model_visibilities no longer rebuild F/D or transform the mapping
matrix a second time; SingleFit products cached; the redundant refits after
the bisection removed; --reg gaussian --envelope-fwhm optimise no longer
IndexErrors (the probe vector matched the number of free parameters only for
kernel priors) and the fallbacks forward optimise_envelope; the
structure->discrepancy fallback now gets the constrained re-bisection; the
positivity probe runs at the chosen coefficient; prior_systematic uses the
positive solver when the fit did; current_memory_gb reads RSS not the
high-water mark; the W~ kernel build asks autoarray for use_jax=True with a
logged NumPy fallback (to be timed on Ruby). envelope: the three custom
regularisations get log-det/regularisation-term overrides from their own
covariance, so the slogdet setting could not silently use the unweighted
Matern.

$T
M
c src/pyuvimage/fitting.py src/pyuvimage/envelope.py tests/test_criterion_selection.py tests/test_linear_system.py tests/test_sparse_inversion.py

cat > /tmp/msg.txt <<M
api/products/cli: points carried through cubes, honest records, lazy import

Cube mode fitted the points only in the MFS pass, fitted the channels with
none, and stapled the MFS points onto plane 0 -- a point sat in every
channel's residual (chi^2/N 8-10 against an MFS of 1.0) while model.fits
plane 0 carried its flux. The MFS pass now decides the positions and every
channel fits its own amplitude there; the retune factor the point fit
applied is folded into the frozen prior; point_sources.json lists the planes.

Channel fits receive chi2_target and the positivity the MFS guard actually
ran with; fit_quality records the n_data the chi^2 was measured on (it
divided a thinned-MFS chi^2 by the full count, 0.25 while the channels sat
at 8) plus per-channel chi^2/N; solver.transformer records the class that
ran rather than "auto"; criterion auto and the under-constrained warning
judge the data each fit sees under cube_prior=channel. --envelope-fwhm
optimise is honoured with a fixed coefficient. One DirtyImager per run
instead of four. --envelope-centre takes image "x,y" like every other
position flag. products: a channel whose uncertainty failed writes NaN
planes instead of crashing the whole write; the MFS frequency is OBSFREQ,
not RESTFRQ; FREQIRR fits in a card. grids rejects a non-square mesh
instead of silently fitting a square one. pyuvimage imports api lazily
(1.3-2 s saved on import/convert).

$T
M
c src/pyuvimage/api.py src/pyuvimage/products.py src/pyuvimage/cli.py src/pyuvimage/grids.py src/pyuvimage/__init__.py tests/test_cube_prior.py docs/parameters.md
unlock
git status --short | grep -v _to_delete
git log --oneline -7

"""Mock datasets for demos and regression tests.

The simulator uses the same Fourier transform implementation as the fitting
code, so mocks are convention-consistent by construction.
"""

from __future__ import annotations

import numpy as np

import autogalaxy as ag

from .grids import resolve_geometry
from .uvdata import C_M_S, V_SIGN, UVData


def random_uv_coverage(
    n_vis: int,
    max_baseline_m: float,
    frequency_hz: float,
    seed: int = 0,
) -> np.ndarray:
    """Plausible centrally condensed uv coverage; returns uvw in metres."""
    rng = np.random.default_rng(seed)
    r = max_baseline_m * rng.beta(1.5, 2.5, n_vis)
    phi = rng.uniform(0, 2 * np.pi, n_vis)
    uvw = np.zeros((n_vis, 3))
    uvw[:, 0] = r * np.cos(phi)
    uvw[:, 1] = r * np.sin(phi)
    return uvw


def exponential_image(
    shape: tuple[int, int],
    pixel_scale: float,
    flux_jy: float = 1.0,
    r_eff_arcsec: float = 0.3,
    centre_arcsec: tuple[float, float] = (0.0, 0.0),
    axis_ratio: float = 1.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Exponential disc, total flux flux_jy, in native (row 0 = +y) orientation."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x = (xx - cx) * pixel_scale - centre_arcsec[1]
    y = (cy - yy) * pixel_scale - centre_arcsec[0]  # native: row 0 = +y
    th = np.radians(angle_deg)
    xr = x * np.cos(th) + y * np.sin(th)
    yr = -x * np.sin(th) + y * np.cos(th)
    r = np.hypot(xr, yr / axis_ratio)
    img = np.exp(-1.6783 * r / r_eff_arcsec)
    return img / img.sum() * flux_jy


def uv_of(uvw_m: np.ndarray, frequency_hz: float) -> np.ndarray:
    """(u, v) in wavelengths, in the same sky frame `UVData.uv_wavelengths`
    uses -- `V_SIGN` and all.

    Mocks have to be built in the frame the imager reads, or a round trip
    closes on a mirrored sky and every astrometry test passes while the real
    data comes out flipped in declination. That is exactly what happened
    before 28 Aug; see `uvdata.V_SIGN`.
    """
    uv = np.asarray(uvw_m)[:, :2] * (float(frequency_hz) / C_M_S)
    return np.column_stack((uv[:, 0], V_SIGN * uv[:, 1]))


def simulate(
    image_native: np.ndarray,
    pixel_scale: float,
    uvw_m: np.ndarray,
    frequencies_hz: np.ndarray,
    sigma_jy: float = 1e-4,
    seed: int = 1,
    meta: dict | None = None,
) -> UVData:
    """Forward-model an image to noisy visibilities with the fitting stack's
    own transformer (DFT)."""
    rng = np.random.default_rng(seed)
    frequencies_hz = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    ny, nx = image_native.shape
    mask = ag.Mask2D.all_false(
        shape_native=(ny, nx), pixel_scales=pixel_scale
    )
    img = ag.Array2D(values=np.asarray(image_native, dtype=float), mask=mask)

    n_vis = uvw_m.shape[0]
    n_chan = len(frequencies_hz)
    data = np.zeros((n_chan, n_vis), dtype=complex)
    for c, f in enumerate(frequencies_hz):
        uv = uv_of(uvw_m, f)
        transformer = ag.TransformerDFT(uv_wavelengths=uv, real_space_mask=mask)
        vis = np.asarray(transformer.visibilities_from(image=img))
        noise = rng.normal(0, sigma_jy, n_vis) + 1j * rng.normal(
            0, sigma_jy, n_vis
        )
        data[c] = vis + noise
    noise_map = np.full((n_chan, n_vis), sigma_jy + 1j * sigma_jy)
    return UVData(
        uvw=uvw_m,
        frequencies=frequencies_hz,
        data=data,
        noise=noise_map,
        meta=dict(meta or {}),
    )


def multi_component_image(
    shape: tuple[int, int],
    pixel_scale: float,
    components=None,
) -> np.ndarray:
    """A deliberately awkward source: several components of different size,
    brightness and shape, used to check that priors are not tuned to one
    simple blob.

    Each component is (flux_jy, r_eff_arcsec, (y, x) arcsec, axis_ratio,
    angle_deg).  The default is a bright compact core, a faint extended disc
    offset from it, and a small secondary knot further out.
    """
    if components is None:
        components = [
            (0.030, 0.10, (0.0, 0.0), 1.0, 0.0),      # bright compact core
            (0.015, 0.60, (0.25, -0.30), 0.5, 30.0),  # faint extended disc
            (0.005, 0.15, (-0.9, 0.8), 1.0, 0.0),     # small offset knot
        ]
    img = np.zeros(shape, dtype=float)
    for flux, r_eff, centre, q, angle in components:
        img += exponential_image(
            shape, pixel_scale, flux_jy=flux, r_eff_arcsec=r_eff,
            centre_arcsec=centre, axis_ratio=q, angle_deg=angle,
        )
    return img


def make_multi_component_dataset(
    n_vis: int = 600,
    frequency_hz: float = 230e9,
    fov_arcsec: float = 3.0,
    sigma_jy: float = 1e-4,
    seed: int = 21,
    mesh_n: int = 48,
):
    """Multi-component source + uv coverage. Returns (uvdata, truth, geometry)."""
    fov_rad = fov_arcsec / 206265.0
    uv_max = mesh_n / (2.0 * fov_rad)
    max_b = uv_max * C_M_S / frequency_hz
    uvw = random_uv_coverage(n_vis, max_b, frequency_hz, seed=seed)
    geom = resolve_geometry(
        fov_arcsec,
        max_baseline_wavelengths=float(
            np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * frequency_hz / C_M_S
        ),
        mesh_shape=(mesh_n, mesh_n),
    )
    truth = multi_component_image(geom.shape_native, geom.pixel_scale)
    uvd = simulate(
        truth, geom.pixel_scale, uvw, np.array([frequency_hz]),
        sigma_jy=sigma_jy, seed=seed + 1,
        meta={"telescope": "mock", "dish_diameter_m": 12.0,
              "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0},
    )
    return uvd, truth, geom


def make_extended_plus_compact_dataset(
    n_vis: int = 600,
    frequency_hz: float = 230e9,
    fov_arcsec: float = 3.0,
    sigma_jy: float = 5e-4,
    seed: int = 31,
    mesh_n: int = 32,
    extended_flux: float = 0.040,
    extended_r_eff: float = 0.70,
    compact_flux: float = 0.012,
    compact_r_eff: float = 0.03,
    compact_centre: tuple[float, float] = (0.8, -0.7),
    truth_on_mesh: bool = True,
):
    """An extended exponential plus an unresolved off-centre compact source.

    This is the discriminating test for brightness-adaptive priors: one global
    smoothing length cannot serve both components at once.  Smoothing suited
    to the extended emission smears the knot; smoothing suited to the knot
    leaves the extended source noisy.

    Returns (uvdata, truth, geometry, components) where `components` records
    each source's (flux, centre) for per-component scoring.
    """
    fov_rad = fov_arcsec / 206265.0
    uv_max = mesh_n / (2.0 * fov_rad)
    uvw = random_uv_coverage(
        n_vis, uv_max * C_M_S / frequency_hz, frequency_hz, seed=seed
    )
    geom = resolve_geometry(
        fov_arcsec,
        max_baseline_wavelengths=float(
            np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * frequency_hz / C_M_S
        ),
        mesh_shape=(mesh_n, mesh_n),
    )
    # Build the truth on the model mesh by default. If it is built on the
    # finer product grid instead it contains structure the model provably
    # cannot represent, so at high S/N chi^2/N floors well above 1 and the
    # comparison measures pixelisation error rather than the priors.
    tshape, tpix = (
        (geom.mesh_shape, geom.mesh_pixel_scale) if truth_on_mesh
        else (geom.shape_native, geom.pixel_scale)
    )
    extended = exponential_image(
        tshape, tpix, flux_jy=extended_flux, r_eff_arcsec=extended_r_eff,
    )
    compact = exponential_image(
        tshape, tpix, flux_jy=compact_flux, r_eff_arcsec=compact_r_eff,
        centre_arcsec=compact_centre,
    )
    truth = extended + compact
    uvd = simulate(
        truth, tpix, uvw, np.array([frequency_hz]),
        sigma_jy=sigma_jy, seed=seed + 1,
        meta={"telescope": "mock", "dish_diameter_m": 12.0,
              "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0},
    )
    components = {
        "extended": {"flux": extended_flux, "centre": (0.0, 0.0),
                     "r_eff": extended_r_eff},
        "compact": {"flux": compact_flux, "centre": compact_centre,
                    "r_eff": compact_r_eff},
    }
    return uvd, truth, geom, components


def make_demo_dataset(
    n_vis: int = 1200,
    frequency_hz: float = 230e9,
    fov_arcsec: float = 3.0,
    sigma_jy: float = 2e-4,
    seed: int = 0,
    mesh_n: int = 32,
    point_flux_jy: float = 0.0,
    point_centre: tuple[float, float] = (0.85, -0.65),
):
    """The self-contained demo: an extended disc plus one true point source.

    Deliberately well-posed: 1200 visibilities is 2400 data points, against
    the 576 model pixels the fit resolves to at `--fov 3` (a 24x24 mesh; the
    truth here is built on a finer 32x32 grid, which is why the disc is not
    exactly representable and the fit lands at chi^2/N = 1.001 rather than
    below it). The original demo used 300 visibilities, i.e. 1.7 model pixels
    per datum -- the under-constrained regime where the generalisation tests
    found spurious point detections. A first run of the tool should not be in
    that regime.

    Note the residual map still reads a structure ratio near 0.56. That is the
    small-mock artefact documented in design-notes.md, not overfitting: with
    1200 visibilities behind a 48x48 image grid the map has far more pixels
    than independent measurements, so its rms is set by the beam's correlated
    structure rather than by chi^2/N. More visibilities make it *worse*, not
    better -- measured 0.56 at 2400 and 0.14 at 3600 -- so it is not a knob to
    turn here.

    `point_flux_jy` is off by default so the mock stays a plain extended
    source for regression tests; the CLI demo turns it on.

    The truth image is built on the **mesh** grid, like every other mock here:
    built on the finer product grid it contains structure the model provably
    cannot represent, which muddles any comparison against it. The point
    source is added analytically, so the demo also exercises
    ``--point-sources`` on something the pixel grid genuinely cannot hold.

    Returns (uvdata, truth, geometry, components); `truth` is the extended
    emission only, on the mesh grid.
    """
    from .pointsource import point_visibilities, sky_to_grid

    fov_rad = fov_arcsec / 206265.0
    uv_max = mesh_n / (2.0 * fov_rad)  # wavelengths
    max_b = uv_max * C_M_S / frequency_hz
    uvw = random_uv_coverage(n_vis, max_baseline_m=max_b,
                             frequency_hz=frequency_hz, seed=seed)
    geom = resolve_geometry(
        fov_arcsec,
        max_baseline_wavelengths=float(
            np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * frequency_hz / C_M_S
        ),
        mesh_shape=(mesh_n, mesh_n),
    )
    truth = exponential_image(
        geom.mesh_shape, geom.mesh_pixel_scale, flux_jy=0.05,
        r_eff_arcsec=fov_arcsec / 8,
    )
    uvd = simulate(
        truth, geom.mesh_pixel_scale, uvw, np.array([frequency_hz]),
        sigma_jy=sigma_jy, seed=seed + 1,
        meta={
            "telescope": "mock", "dish_diameter_m": 12.0,
            "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0,
        },
    )
    if point_flux_jy:
        uv = uv_of(uvw, frequency_hz)
        y, x = sky_to_grid(*point_centre)
        uvd.data[0] += point_flux_jy * point_visibilities(uv, y, x)
    components = {
        "extended": [{"flux": 0.05, "r_eff": fov_arcsec / 8,
                      "centre": (0.0, 0.0)}],
        "points": ([{"flux": point_flux_jy, "centre": point_centre}]
                   if point_flux_jy else []),
    }
    return uvd, truth, geom, components


def make_field_dataset(
    n_vis: int = 800,
    frequency_hz: float = 230e9,
    fov_arcsec: float = 4.0,
    sigma_jy: float = 3e-4,
    seed: int = 77,
    mesh_n: int = 32,
    extended=None,
    points=None,
    max_baseline_m: float | None = None,
    truth_on_mesh: bool = True,
):
    """A general test field: several extended components plus *true* points.

    The point sources are injected **analytically**, with their exact
    closed-form visibilities at sub-pixel positions -- not as bright pixels on
    the truth grid.  That distinction is the whole test: a point placed on the
    grid is a source the pixelized model can already represent, so recovering
    it would prove nothing about the delta-function machinery.

    `extended` is a list of (flux_jy, r_eff_arcsec, (dRA, dDec), axis_ratio,
    angle_deg); `points` is a list of (flux_jy, (dRA, dDec)).  Both are given
    in sky offsets, positive dRA east, matching what the fitter reports.

    Returns (uvdata, truth_image, geometry, components).  Note the truth image
    contains only the extended emission -- the points have no grid
    representation, which is the point.
    """
    from .pointsource import point_visibilities, sky_to_grid

    if extended is None:
        extended = [(0.040, 0.70, (0.0, 0.0), 1.0, 0.0)]
    if points is None:
        points = []

    fov_rad = fov_arcsec / 206265.0
    if max_baseline_m is None:
        max_baseline_m = (mesh_n / (2.0 * fov_rad)) * C_M_S / frequency_hz
    uvw = random_uv_coverage(n_vis, max_baseline_m, frequency_hz, seed=seed)
    geom = resolve_geometry(
        fov_arcsec,
        max_baseline_wavelengths=float(
            np.max(np.hypot(uvw[:, 0], uvw[:, 1])) * frequency_hz / C_M_S
        ),
        mesh_shape=(mesh_n, mesh_n),
    )
    tshape, tpix = (
        (geom.mesh_shape, geom.mesh_pixel_scale) if truth_on_mesh
        else (geom.shape_native, geom.pixel_scale)
    )
    truth = np.zeros(tshape, dtype=float)
    for flux, r_eff, (d_ra, d_dec), q, angle in extended:
        truth += exponential_image(
            tshape, tpix, flux_jy=flux, r_eff_arcsec=r_eff,
            centre_arcsec=(d_dec, -d_ra), axis_ratio=q, angle_deg=angle,
        )

    uvd = simulate(
        truth, tpix, uvw, np.array([frequency_hz]),
        sigma_jy=sigma_jy, seed=seed + 1,
        meta={"telescope": "mock", "dish_diameter_m": 12.0,
              "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0},
    )
    # add the analytic points on top of the already-noisy visibilities
    uv = uv_of(uvw, frequency_hz)
    for flux, (d_ra, d_dec) in points:
        y, x = sky_to_grid(d_ra, d_dec)
        uvd.data[0] += flux * point_visibilities(uv, y, x)

    components = {
        "extended": [
            {"flux": f, "r_eff": r, "centre": c} for f, r, c, _, _ in extended
        ],
        "points": [{"flux": f, "centre": c} for f, c in points],
    }
    return uvd, truth, geom, components


def split_into_spws(uvd, n_spw: int = 2):
    """Split one dataset's channels into `n_spw` spectral windows.

    Used to test that imaging several spws together reproduces imaging them as
    one block: the samples are identical, only the container differs, so the
    two fits must agree to numerical precision.
    """
    from .uvdata import MultiSpwUVData, UVData

    edges = np.linspace(0, uvd.n_chan, n_spw + 1).astype(int)
    spws = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        spws.append(UVData(
            uvw=uvd.uvw.copy(),
            frequencies=uvd.frequencies[lo:hi].copy(),
            data=uvd.data[lo:hi].copy(),
            noise=uvd.noise[lo:hi].copy(),
            flags=None if uvd.flags is None else uvd.flags[lo:hi].copy(),
            meta=dict(uvd.meta),
        ))
    return MultiSpwUVData(spws=spws, meta=dict(uvd.meta))


def make_multi_spw_dataset(
    n_vis: int = 400,
    spw_frequencies_hz=(230e9, 232e9, 234e9),
    n_chan_per_spw=(2, 3, 2),
    channel_width_hz: float = 2e8,
    fov_arcsec: float = 3.0,
    sigma_jy: float = 3e-4,
    seed: int = 5,
    mesh_n: int = 24,
    flux_jy: float = 0.05,
    point_flux_jy: float = 0.0,
    point_centre: tuple[float, float] = (0.8, -0.6),
):
    """Several spectral windows of the same sky, deliberately ragged.

    Different channel counts *and* different rows per spw, which is what a real
    measurement set looks like and what a rectangular array cannot hold. The
    sky is frequency-independent, so MFS across the whole set is exact and any
    disagreement is a bug rather than a spectral-index effect.

    Returns (MultiSpwUVData, truth, geometry).
    """
    from .pointsource import point_visibilities, sky_to_grid
    from .uvdata import MultiSpwUVData

    fov_rad = fov_arcsec / 206265.0
    ref = float(np.max(spw_frequencies_hz))
    max_b = (mesh_n / (2.0 * fov_rad)) * C_M_S / ref
    geom = resolve_geometry(
        fov_arcsec,
        max_baseline_wavelengths=max_b * ref / C_M_S,
        mesh_shape=(mesh_n, mesh_n),
    )
    truth = exponential_image(
        geom.mesh_shape, geom.mesh_pixel_scale, flux_jy=flux_jy,
        r_eff_arcsec=fov_arcsec / 8,
    )
    spws = []
    for i, (f0, n_chan) in enumerate(zip(spw_frequencies_hz, n_chan_per_spw)):
        # each spw gets its own rows, as in a real MS
        uvw = random_uv_coverage(n_vis + 37 * i, max_b, f0, seed=seed + i)
        freqs = f0 + channel_width_hz * np.arange(n_chan)
        uvd = simulate(
            truth, geom.mesh_pixel_scale, uvw, freqs,
            sigma_jy=sigma_jy, seed=seed + 100 + i,
            meta={"telescope": "mock", "dish_diameter_m": 12.0,
                  "phase_centre_ra_deg": 150.0, "phase_centre_dec_deg": 2.0,
                  "spw": i},
        )
        if point_flux_jy:
            y, x = sky_to_grid(*point_centre)
            for c, f in enumerate(freqs):
                uv = uv_of(uvw, f)
                uvd.data[c] += point_flux_jy * point_visibilities(uv, y, x)
        spws.append(uvd)
    return MultiSpwUVData(spws=spws), truth, geom

"""Point components on the streamed path: the bordered system without visibilities.

`pointsource.AugmentedSystem` needs three things per point column P_p (a
unit point at p, optionally widened to a Gaussian of sigma s):

    B_p  = A^T W P_p      the cross-term with the mesh
    C_pq = P_p^T W P_q    the point-point Gram
    Dp_p = P_p^T W d      the data term

All three are sums over visibilities, and all three are values of two
image-plane functions that the streaming pass can accumulate once:

    K(D)  = sum_v w cos(2 pi (u Dx + v Dy))         the w-tilde kernel at lag D
    Dd(x) = Re sum_v c exp(+2 pi i (u x + v y))     the data's dirty image,
                                                    c = d_re/s_re^2 + i d_im/s_im^2

    B_p  = M^T k_p,  k_p[i] = K(x_i - p)    (the dirty image of the column,
                                             projected onto the mesh)
    C_pq = K(p - q)
    Dp_p = Dd(p)

A Gaussian width multiplies each visibility by exp(-2 pi^2 s^2 |u|^2), which
in the image plane is a convolution with a unit Gaussian G_s: K -> K * G_s
for B and Dd -> Dd * G_s for Dp, and K * G_{sqrt(s1^2 + s2^2)} for a pair.

Positions are continuous, so both functions are held on grids `oversample`
times finer than the image (`SparseTerms.kernel_fine` / `dirty_fine`,
accumulated by `streaming.TermsAccumulator`) and read off with a quintic
spline. At q = 8 that is 2e-9 of K(0) (`claude/teresa-test-2.md`), which on
Teresa's 1.8e7 chi^2 is far below the delta chi^2 of 1 the refinement and the
unresolved test resolve.

The w-tilde reduction assumes sigma_re == sigma_im, exactly as the streamed
inversion's own F does; the streaming pass pools the noise when they differ.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
from scipy import ndimage
from scipy.linalg import cho_factor, cho_solve

from .pointsource import (
    COLUMN_CACHE_BYTES,
    AugmentedSystem,
    SparseMesh,
)

logger = logging.getLogger(__name__)

#: the spline order the grids are read with; 5 is what the 2e-9 was measured at
SPLINE_ORDER = 5
#: the boundary mode for both the prefilter and the lookups (they must agree)
SPLINE_MODE = "mirror"
#: Gaussian-widened grids kept at once (each the size of `kernel_fine`)
WIDTH_CACHE = 6
#: lattice columns per block in `scan`
SCAN_BLOCK = 2048


def _gaussian_smooth(array: np.ndarray, sigma_px: float) -> np.ndarray:
    """Circular convolution with a unit-integral Gaussian of `sigma_px` pixels.

    Done in Fourier space rather than with `ndimage.gaussian_filter`, whose
    4-sigma truncation leaves a 3e-4 error in the taper -- larger than the
    interpolation by five decades. The FFT form is the exact sampled Gaussian
    (aliasing at exp(-2 pi^2 sigma_px^2), nothing for sigma_px >~ 1), and
    the grids are band-limited far below their Nyquist, so this is the
    continuous convolution. It wraps at the edges, which only reaches lags
    within a few sigma of the grid's own edge.
    """
    fy = np.fft.fftfreq(array.shape[0])[:, None]
    fx = np.fft.fftfreq(array.shape[1])[None, :]
    taper = np.exp(-2.0 * np.pi**2 * sigma_px**2 * (fy**2 + fx**2))
    return np.fft.ifft2(np.fft.fft2(array) * taper).real


class PointGrids:
    """K(dy, dx) and Dd(y, x) anywhere in the field, from `SparseTerms`.

    Coordinates are the grid's (y, x) arcsec of `pointsource` (``sky_to_grid``).
    The kernel grid is the fftshifted `kernel_fine`, zero lag at
    ``[Ny q, Nx q]``; autoarray's builder puts lag (dy, dx) = (+i h, -j h) at
    index (i, j), so a lag is at row ``Ny q + dy/h``, column ``Nx q - dx/h``.
    The dirty grid is centred (`streaming.type1_image`): (y, x) at row
    ``Ny q / 2 - y/h``, column ``Nx q / 2 + x/h``.
    """

    def __init__(self, terms):
        q = int(getattr(terms, "oversample", 0) or 0)
        if not q or terms.kernel_fine is None or terms.dirty_fine is None:
            raise ValueError(
                "these streamed terms carry no oversampled grids; accumulate "
                "them with point_oversample > 0 (`streaming.sparse_terms_for`)")
        self.q = q
        self.h = float(terms.pixel_scale) / q
        self._kernel = np.asarray(terms.kernel_fine, dtype=float)   # wraparound
        self._dirty = np.asarray(terms.dirty_fine, dtype=float)     # centred
        self.k0 = (self._kernel.shape[0] // 2, self._kernel.shape[1] // 2)
        self.d0 = (self._dirty.shape[0] // 2, self._dirty.shape[1] // 2)
        self._coeffs: OrderedDict = OrderedDict()

    def _spline(self, kind: str, sigma: float) -> np.ndarray:
        key = (kind, float(sigma))
        hit = self._coeffs.get(key)
        if hit is not None:
            self._coeffs.move_to_end(key)
            return hit
        if kind == "K":
            arr = self._kernel
            if sigma > 0:
                arr = _gaussian_smooth(arr, sigma / self.h)
            arr = np.fft.fftshift(arr)
        else:
            arr = self._dirty
            if sigma > 0:
                arr = _gaussian_smooth(arr, sigma / self.h)
        coeffs = ndimage.spline_filter(arr, order=SPLINE_ORDER, mode=SPLINE_MODE)
        self._coeffs[key] = coeffs
        # the two unwidened grids are always wanted; only widths are evicted
        while len(self._coeffs) > WIDTH_CACHE + 2:
            for k in self._coeffs:
                if k[1] > 0:
                    del self._coeffs[k]
                    break
            else:
                break
        return coeffs

    def _read(self, coeffs, rows, cols) -> np.ndarray:
        shape = np.shape(rows)
        out = ndimage.map_coordinates(
            coeffs, [np.ravel(rows), np.ravel(cols)], order=SPLINE_ORDER,
            mode=SPLINE_MODE, prefilter=False)
        return out.reshape(shape)

    def kernel(self, dy, dx, sigma: float = 0.0) -> np.ndarray:
        """K * G_sigma at lags (dy, dx) arcsec; ``sigma`` = 0 is K itself."""
        dy = np.asarray(dy, dtype=float)
        dx = np.asarray(dx, dtype=float)
        return self._read(self._spline("K", sigma),
                          self.k0[0] + dy / self.h, self.k0[1] - dx / self.h)

    def dirty(self, y, x, sigma: float = 0.0) -> np.ndarray:
        """Dd * G_sigma at grid positions (y, x) arcsec."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        return self._read(self._spline("D", sigma),
                          self.d0[0] - y / self.h, self.d0[1] + x / self.h)


class StreamedMesh(SparseMesh):
    """`SparseMesh` over the stub dataset: `cross_on_grid` only.

    The stub's eight visibilities are not the data, so the per-column routes
    that transform through its transformer would be silently wrong; they
    raise instead. `StreamedAugmentedSystem` never calls them.
    """

    def __init__(self, inversion, stub, n_vis: int):
        super().__init__(inversion, stub)
        self.n_vis = int(n_vis)

    def cross(self, Pw):
        raise NotImplementedError(
            "the streamed bordered system has no visibilities to transform; "
            "its cross-terms come from the kernel (`StreamedAugmentedSystem`)")

    def forward(self, mesh_values):
        raise NotImplementedError(
            "the streamed path keeps no visibilities, so there are no model "
            "visibilities to form; the residual is formed in the image plane "
            "(`StreamedAugmentedSystem.residual_dirty_image`)")


class StreamedAugmentedSystem(AugmentedSystem):
    """`AugmentedSystem` from streamed terms: B, C, Dp read off the grids.

    F, H and D are the stub inversion's, which the w-tilde operator built from
    the same terms; chi2_const is d^T N^-1 d from the streaming pass. Every
    solve, scan, refinement and retune in `pointsource` then runs unchanged
    -- they only ever ask for these terms, never for a visibility.
    """

    def __init__(self, inversion, stub, terms, imager=None):
        self.inversion = inversion
        self.terms = terms
        self.grids = PointGrids(terms)
        self.imager = imager
        self.mesh = StreamedMesh(inversion, stub, terms.n_vis)
        self.mapping_matrix = self.mesh.mapping_matrix
        self.n_vis = int(terms.n_vis)
        self.F = np.asarray(inversion.curvature_matrix)
        self.D = np.asarray(inversion.data_vector)
        self.H = np.asarray(inversion.regularization_matrix)
        self.n_mesh = self.F.shape[0]
        self.n_data = 2 * self.n_vis
        self.h_scale = 1.0
        self._cho = cho_factor(self.F + self.H, lower=True, check_finite=False)
        self._MinvD = None
        self.chi2_const = float(terms.data_term)
        self._columns: OrderedDict = OrderedDict()
        self._column_cache_max = int(
            min(4096, max(8, COLUMN_CACHE_BYTES // (8 * (self.n_mesh + 2)))))
        self._lattice = None
        # image-pixel centres in the mapping matrix's (slim) order
        grid = np.asarray(self.mesh.mask.derive_grid.unmasked).reshape(-1, 2)
        if grid.shape[0] != self.mapping_matrix.shape[0]:
            raise ValueError(
                "the mask's grid does not match the mapping matrix's rows")
        self._pix_y = grid[:, 0].copy()
        self._pix_x = grid[:, 1].copy()

    # ---- the visibility-space members, which do not exist here -----------
    @property
    def w_stack(self):
        raise AttributeError("the streamed system holds no visibilities")

    d_stack = wd_stack = d_re = d_im = uv = w_stack

    def _stacked_columns(self, ys, xs, sigmas=None):
        raise NotImplementedError("the streamed system forms no visibility columns")

    def columns_for(self, positions, sigma_arcsec=0.0):
        raise NotImplementedError("the streamed system forms no visibility columns")

    # ---- the column terms, from the grids --------------------------------
    def _column_terms(self, positions, sigmas=None):
        """(None, B, Dp): as the base class, with no visibility column P."""
        positions = list(positions)
        k = len(positions)
        if sigmas is None:
            sigmas = [0.0] * k
        keys = [(float(y), float(x), float(s))
                for (y, x), s in zip(positions, sigmas)]
        for key in keys:
            if key in self._columns:
                continue
            y, x, s = key
            k_p = self.grids.kernel(self._pix_y - y, self._pix_x - x, s)
            self._columns[key] = (
                self.mapping_matrix.T @ k_p,
                float(self.grids.dirty(y, x, s)),
            )
            while len(self._columns) > self._column_cache_max:
                self._columns.popitem(last=False)
        B = np.empty((self.n_mesh, k))
        Dp = np.empty(k)
        for j, key in enumerate(keys):
            self._columns.move_to_end(key)
            B[:, j], Dp[j] = self._columns[key]
        return None, B, Dp

    def _point_gram(self, positions, sigmas, P=None) -> np.ndarray:
        """C_ij = (K * G_sqrt(si^2 + sj^2))(p_i - p_j)."""
        positions = list(positions)
        k = len(positions)
        s = np.zeros(k) if sigmas is None else np.asarray(sigmas, dtype=float)
        ys = np.array([p[0] for p in positions], dtype=float)
        xs = np.array([p[1] for p in positions], dtype=float)
        C = np.empty((k, k))
        widths = np.sqrt(s[:, None] ** 2 + s[None, :] ** 2)
        for w in np.unique(widths):
            sel = widths == w
            i, j = np.nonzero(sel)
            C[i, j] = self.grids.kernel(ys[i] - ys[j], xs[i] - xs[j], float(w))
        return 0.5 * (C + C.T)

    def scan(self, positions, ys, xs, chunk: int | None = None):
        """`AugmentedSystem.scan` with the lattice terms read off the grids.

        Per trial column: b from `cross_on_grid` (or the grid kernel off the
        image grid), c = K(0) = sum w, dp = Dd at the position, and the
        accepted points' rows K(lattice - p).
        """
        ys = np.asarray(ys, dtype=float).ravel()
        xs = np.asarray(xs, dtype=float).ravel()
        if positions:
            _, B0, Dp0 = self._column_terms(positions)
            M = np.block([
                [self.F + self.h_scale * self.H, B0],
                [B0.T, self._point_gram(positions, None)],
            ])
            D_eff = np.concatenate([self.D, Dp0])
        else:
            M = self.F + self.h_scale * self.H
            D_eff = self.D
        M = M + np.eye(M.shape[0]) * 1e-12 * max(np.trace(M), 1e-30)
        cho = cho_factor(M, lower=True, check_finite=False)
        MinvD = cho_solve(cho, D_eff, check_finite=False)

        b_all = self.mesh.cross_on_grid(ys, xs)
        dp_all = self.grids.dirty(ys, xs)
        c0 = float(self.grids.kernel(0.0, 0.0))
        block = max(1, int(chunk or SCAN_BLOCK))
        amp = np.empty(ys.size)
        sig = np.empty(ys.size)
        for lo in range(0, ys.size, block):
            hi = min(lo + block, ys.size)
            if b_all is not None:
                b = b_all[:, lo:hi]
            else:
                b = self._column_terms(list(zip(ys[lo:hi], xs[lo:hi])))[1]
            if positions:
                rows = np.vstack([
                    self.grids.kernel(ys[lo:hi] - py, xs[lo:hi] - px)
                    for (py, px) in positions
                ])
                B = np.vstack([b, rows])
            else:
                B = b
            r = dp_all[lo:hi] - B.T @ MinvD
            MinvB = cho_solve(cho, B, check_finite=False)
            sch = np.maximum(c0 - np.einsum("ij,ij->j", B, MinvB), 1e-30)
            amp[lo:hi] = r / sch
            sig[lo:hi] = np.abs(r) / np.sqrt(sch)
        return amp, sig

    # ---- the image-plane residual ----------------------------------------
    def point_dirty_image(self, positions, amplitudes, sigmas=None) -> np.ndarray:
        """Sum a_j (K * G_sj)(x - p_j) on the native image grid, unnormalised
        (divide by sum w for Jy/beam, as `KernelDirtyImager` does)."""
        ny, nx = self.terms.shape_native
        pix = float(self.terms.pixel_scale)
        iy, ix = np.mgrid[0:ny, 0:nx]
        gy = ((ny - 1) / 2.0 - iy) * pix
        gx = (ix - (nx - 1) / 2.0) * pix
        out = np.zeros((ny, nx))
        if sigmas is None:
            sigmas = [0.0] * len(positions)
        for (py, px), a, s in zip(positions, amplitudes, sigmas):
            out += float(a) * self.grids.kernel(gy - py, gx - px, float(s))
        return out

    def residual_dirty_image(self, mesh_values, positions, amplitudes,
                             imager=None) -> np.ndarray:
        """dirty(data) - dirty(mesh model) - dirty(points) [Jy/beam]."""
        imager = imager if imager is not None else self.imager
        if imager is None:
            raise ValueError("the streamed residual needs the kernel imager")
        import autoarray as aa

        slim = self.mapping_matrix @ np.asarray(mesh_values)
        model = np.asarray(aa.Array2D(values=slim, mask=self.mesh.mask).native)
        out = imager.dirty_image_of_data() - imager.dirty_image_of_model(model)
        if len(positions):
            out = out - self.point_dirty_image(positions, amplitudes) / imager._norm
        return out

    def model_visibilities(self, mesh_values, positions, amplitudes):
        return self.mesh.forward(mesh_values)   # raises, with the reason

"""Matrix-free solve for large meshes (after Powell et al. 2021).

The dense path forms F + H, an N_mesh x N_mesh matrix, and factorises it:
O(N_mesh^2) memory and O(N_mesh^3) time -- 11 GB per copy and ~2e13 flops per
factorisation at a 192 x 192 mesh, and the non-negative solver needs many.
Above `MATRIX_FREE_MIN_MESH_PIXELS` (a 100 x 100 mesh) this module is used
instead, and nothing N_mesh x N_mesh is ever formed:

* **Data term.** F s = M^T (W~ (M s)): the w-tilde convolution, one FFT pair
  on the (2 Ny, 2 Nx) padded image grid, with M the mapper's own sparse
  interpolation matrix. D = M^T (dirty image of d / sigma^2). Both equal the
  dense path's F and D to ~1e-16.

* **Prior.** The Matern prior in its sparse-precision (SPDE) form,
  `SparseMaternPrior`: nu = 1 rather than the dense path's 1.5, which has no
  sparse form.

* **Solve.** Conjugate gradients on (F + lambda C^-1) s = D. Two
  preconditioners (`MatrixFreeSystem`): for an unweighted prior, the uv
  weight density -- F and C^-1 both made translation invariant and inverted
  by FFT, with the data term's spectrum lightly smoothed
  (`UNWEIGHTED_PRECONDITIONER`); for a weighted prior (adaptive, envelope),
  Powell et al.'s sparse P = kappa_0 M^T M + lambda C^-1 (W~ replaced by its
  diagonal), factorised once per lambda.

* **Evidence.** ln det H exactly (sparse); ln det(F + H) = ln det P (exact)
  + ln det(P^-1 A) by stochastic Lanczos quadrature with fixed probes.

Not available on this path (yet): positivity (the solve is unconstrained),
point components, and the uncertainty products, which need
diag((F + H)^-1).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("pyuvimage")

#: mesh pixels above which `run` / `run_streamed` take this path (100 x 100)
MATRIX_FREE_MIN_MESH_PIXELS = 100 * 100

#: relative residual at which the Krylov solve stops
CG_RTOL = 1e-8
CG_MAXITER = 5000

#: stochastic Lanczos quadrature: probe vectors and Lanczos steps per probe
SLQ_PROBES = 10
SLQ_STEPS = 40
SLQ_SEED = 20221

#: preconditioner for unweighted priors (`MatrixFreeSystem`): "density"
#: (the uv weight density, lightly smoothed), "radial" (its ring average) or
#: "sparse" (Powell et al.'s P, as for weighted priors)
UNWEIGHTED_PRECONDITIONER = "density"

#: smoothing of the uv density for the "density" preconditioner, in cells
#: of the padded mesh's uv grid
DENSITY_SMOOTHING_CELLS = 1.0


def memory_gb(n_mesh: int, n_image: int) -> float:
    """Rough peak of the matrix-free fit [GB]: a handful of padded-image FFT
    buffers (4 n_image each), the kernel's spectrum, and the sparse factor of
    the preconditioner, whose fill grows like n_mesh^1.5 for a 2-D grid."""
    fft = 12 * 4 * n_image * 8
    factor = 40 * n_mesh ** 1.5 * 8 / 2 ** 0.5
    return (fft + factor + 50 * n_mesh * 8) / 1e9


# ---------------------------------------------------------------------------
# geometry: the mapper's sparse M and the mesh's grid order
# ---------------------------------------------------------------------------

def sparse_mapping(mapper) -> sp.csr_matrix:
    """The mapper's interpolation matrix M (image slim x mesh), sparse.

    Built from the per-sub-pixel indexes and weights autoarray keeps, never
    from `mapping_matrix`, which is dense (43 GB at a 192 x 192 mesh on its
    384 x 384 image). Equal to `mapping_matrix` to the bit on the mocks.
    """
    idx = np.asarray(mapper.pix_indexes_for_sub_slim_index)
    wts = np.asarray(mapper.pix_weights_for_sub_slim_index, dtype=float)
    sizes = np.asarray(mapper.pix_sizes_for_sub_slim_index)
    sub2slim = np.asarray(mapper.slim_index_for_sub_slim_index)
    n_slim = int(sub2slim.max()) + 1
    n_mesh = int(mapper.pixels)
    counts = np.bincount(sub2slim, minlength=n_slim).astype(float)
    keep = np.arange(idx.shape[1])[None, :] < sizes[:, None]
    rows = np.broadcast_to(sub2slim[:, None], idx.shape)[keep]
    cols = idx[keep]
    vals = (wts / counts[sub2slim][:, None])[keep]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_slim, n_mesh))


def mesh_grid_order(points: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float]:
    """(grid index of each mesh pixel, mesh shape, pixel scale) for a regular mesh.

    Mesh pixel i sits at row r_i, column c_i of the (my, mx) grid; the prior's
    FFTs work on that grid and `to_grid` / `from_grid` move between the two.
    """
    pts = np.asarray(points, dtype=float)
    ys = np.unique(np.round(pts[:, 0], 9))
    pix = float(np.median(np.diff(ys))) if ys.size > 1 else 1.0
    rows = np.rint((pts[:, 0].max() - pts[:, 0]) / pix).astype(int)
    cols = np.rint((pts[:, 1] - pts[:, 1].min()) / pix).astype(int)
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    if shape[0] * shape[1] != pts.shape[0]:
        raise ValueError("the matrix-free path needs a regular rectangular mesh")
    return rows * shape[1] + cols, shape, pix


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class DataOperator:
    """F s = M^T W~ M s and D = M^T (dirty image of d / sigma^2), from the
    streamed terms' kernel; nothing per visibility."""

    def __init__(self, terms, M: sp.csr_matrix, inside: np.ndarray):
        self.M = M.tocsr()
        self.MT = self.M.T.tocsr()
        self.inside = np.asarray(inside, dtype=bool)
        self.shape = self.inside.shape
        ny, nx = self.shape
        self.kernel_ft = np.fft.rfft2(np.asarray(terms.kernel, dtype=float))
        # W~'s diagonal: the kernel at zero lag (its [0, 0] element)
        self.kernel_peak = float(np.asarray(terms.kernel, dtype=float)[0, 0])
        self._pad = (2 * ny, 2 * nx)
        self.D = self.MT @ np.asarray(terms.dirty_image, dtype=float)[self.inside]
        self.data_term = float(terms.data_term)
        self.noise_normalization = float(terms.noise_normalization)
        self.n_data = 2 * int(terms.n_vis)
        self.n_mesh = M.shape[1]

    def image(self, s: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape)
        out[self.inside] = self.M @ s
        return out

    def wtilde(self, image: np.ndarray) -> np.ndarray:
        ny, nx = self.shape
        pad = np.zeros(self._pad)
        pad[:ny, :nx] = image
        return np.fft.irfft2(np.fft.rfft2(pad) * self.kernel_ft, s=self._pad)[:ny, :nx]

    def F(self, s: np.ndarray) -> np.ndarray:
        return self.MT @ self.wtilde(self.image(s))[self.inside]

    def chi_squared(self, s: np.ndarray) -> float:
        return float(self.data_term - 2.0 * s @ self.D + s @ self.F(s))


class SparseMaternPrior:
    """The Matern prior as a sparse precision (Lindgren, Rue & Lindstrom 2011).

    A Gaussian field with Matern covariance of smoothness nu in two
    dimensions solves (kappa^2 - Laplacian)^(alpha/2) x = white noise with
    alpha = nu + 1, kappa = sqrt(2 nu) / scale. For integer alpha its
    precision is a sparse matrix. Here alpha = 2 (nu = 1):

        K^-1 = c L^2,   L = kappa^2 I - Laplacian_h   (5-point, reflecting edges)

        C^-1 = W^-1 K^-1 W^-1,   H = lambda C^-1

    with c set so the variance at the mesh centre is 1, as K(0) = 1 for the
    dense kernel, and W the per-pixel weights of the envelope / adaptive
    priors (C = W K W, as `envelope.py`). The dense path's nu = 1.5 has no
    sparse form; nu = 1 is the nearest. The reflecting edges raise the
    variance within ~1 correlation length of the mesh edge (to ~2x at a
    corner), where the field of view should hold no emission anyway.

    ``R`` is a square root, C^-1 = R^T R (R = sqrt(c) L W^-1), which the
    evidence's probe vectors are drawn with.
    """

    alpha = 2
    nu_effective = 1.0

    def __init__(self, order, mesh_shape, pixel_scale, scale, weights=None):
        from scipy.sparse.linalg import splu

        self.order = np.asarray(order)
        self.mesh_shape = tuple(mesh_shape)
        my, mx = self.mesh_shape
        n = my * mx
        h = float(pixel_scale)
        kappa = np.sqrt(2.0 * self.nu_effective) / float(scale)
        # grid Laplacian with reflecting (Neumann) edges, in grid order
        idx = np.arange(n).reshape(my, mx)
        rows, cols = [], []
        for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
            rows += [a.ravel(), b.ravel()]
            cols += [b.ravel(), a.ravel()]
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        adj = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
        degree = np.asarray(adj.sum(axis=1)).ravel()
        lap = (adj - sp.diags(degree)) / h ** 2
        L_grid = (kappa ** 2 * sp.identity(n) - lap).tocsc()
        # unit variance at the centre: c = e_c^T L^-2 e_c
        lu = splu(L_grid)
        e = np.zeros(n)
        e[(my // 2) * mx + mx // 2] = 1.0
        x = lu.solve(e)
        c = float(x @ x)
        self.log_det_L = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
        # mesh order: P maps grid order to mesh order (mesh i <- grid order[i])
        perm = sp.csr_matrix((np.ones(n), (np.arange(n), self.order)), shape=(n, n))
        L_mesh = (perm @ L_grid @ perm.T).tocsr()
        self.w = np.ones(n) if weights is None else np.asarray(weights, float).ravel()
        if self.w.size != n:
            raise ValueError("prior weights do not match the mesh")
        winv = sp.diags(1.0 / self.w)
        self.R = (np.sqrt(c) * L_mesh @ winv).tocsr()          # C^-1 = R^T R
        self.Qw = (self.R.T @ self.R).tocsr()                  # C^-1
        self.n = n
        self.c = c
        self.h = h
        self.kappa = kappa
        self.weighted = weights is not None
        # ln det C^-1 = n ln c + 2 ln det L - 2 sum ln w
        self.log_det_Cinv = float(n * np.log(c) + 2.0 * self.log_det_L
                                  - 2.0 * np.sum(np.log(self.w)))


@dataclass
class Solution:
    coefficient: float
    s: np.ndarray
    chi_squared: float
    prior_term: float          # s^T H s
    iterations: int


class MatrixFreeSystem:
    """(F + lambda C^-1) s = D for one prior shape and any strength lambda.

    F is applied through the w-tilde FFT; C^-1 is sparse. Conjugate gradients
    with one of two preconditioners:

    * **uv density** (unweighted prior: plain Matern, exponential, the
      adaptive prior's first pass). On the mesh, F is close to a convolution
      by the dirty beam, and the SPDE precision is a convolution away from
      the edges. Both are diagonal in Fourier space, F with the gridded uv
      weight density rho(k). rho is measured once (two matrix-vector
      products: the beam at every lag), smoothed by
      `DENSITY_SMOOTHING_CELLS`, and (rho + lambda q)^-1 applied by FFT on
      the 2x zero-padded mesh. Unsmoothed, the embedded rho is not positive
      and fails. The ring average ("radial": density by baseline length
      alone) does as well on round coverage but ~2.5x worse on NGC 4274,
      whose coverage is elongated.
    * **Sparse P** (weighted prior: adaptive second pass, envelope). Powell
      et al. (2021): P = kappa_0 M^T M + lambda C^-1, i.e. W~ replaced by its
      peak kappa_0, sparse and factorised once per lambda. It follows the
      weights exactly, which a Fourier preconditioner cannot; with the
      adaptive weights it needs ~3x fewer iterations than a Fourier one.

    The evidence always uses the sparse P (`log_det_A`), whose determinant
    is exact.
    """

    def __init__(self, data: DataOperator, prior: SparseMaternPrior,
                 preconditioner: str | None = None):
        self.data, self.prior = data, prior
        self.kappa0 = float(np.asarray(data.kernel_peak))
        self.MtM = (data.MT @ data.M).tocsc()
        self._factor = {}
        self._last = None
        if preconditioner is None:
            preconditioner = "sparse" if prior.weighted else UNWEIGHTED_PRECONDITIONER
        if preconditioner not in ("density", "radial", "sparse"):
            raise ValueError(f"unknown preconditioner {preconditioner!r}")
        self.preconditioner = preconditioner
        self._rho = None

    def _P(self, lam):
        from scipy.sparse.linalg import splu

        key = float(lam)
        if key not in self._factor:
            P = (self.kappa0 * self.MtM + lam * self.prior.Qw).tocsc()
            self._factor = {key: (P, splu(P))}            # keep one: they are large
        return self._factor[key]

    # -- the Fourier (uv density) preconditioners -----------------------
    def _to_grid(self, v):
        my, mx = self.prior.mesh_shape
        g = np.zeros(my * mx)
        g[self.prior.order] = v
        return g.reshape(my, mx)

    def _from_grid(self, g):
        return np.asarray(g).reshape(-1)[self.prior.order]

    def _spectra(self):
        """rho (smoothed, or ring-averaged) and q on the (2 my, 2 mx) padded
        mesh grid's frequencies."""
        if self._rho is None:
            my, mx = self.prior.mesh_shape
            Ny, Nx = 2 * my, 2 * mx

            def column(y, x):
                e = np.zeros((my, mx))
                e[y, x] = 1.0
                return self._to_grid(self.data.F(self._from_grid(e)))

            # the mesh-space beam at every lag: the corner columns give
            # (dy >= 0, dx >= 0) and (dy >= 0, dx <= 0); point symmetry the rest
            c00, c0m = column(0, 0), column(0, mx - 1)
            K = np.zeros((Ny, Nx))
            dy, dx = np.meshgrid(np.arange(my), np.arange(mx), indexing="ij")
            K[dy, dx] = c00
            K[-dy, -dx] = c00
            K[dy, -dx] = c0m[:, ::-1]
            K[-dy, dx] = c0m[:, ::-1]
            rho = np.real(np.fft.fft2(K))
            fy = np.fft.fftfreq(Ny)[:, None]
            fx = np.fft.fftfreq(Nx)[None, :]
            if self.preconditioner == "radial":
                # rings of equal baseline length, one padded-grid cell wide
                ring = np.rint(np.hypot(fy, fx) * max(Ny, Nx)).astype(int)
                prof = (np.bincount(ring.ravel(), weights=rho.ravel())
                        / np.maximum(np.bincount(ring.ravel()), 1))
                rho = prof[ring]
            else:
                from scipy.ndimage import gaussian_filter

                rho = gaussian_filter(rho, DENSITY_SMOOTHING_CELLS, mode="wrap")
            self._rho = np.clip(rho, 0.0, None)
            p = self.prior
            lap = 4.0 / p.h ** 2 * (np.sin(np.pi * fy) ** 2 + np.sin(np.pi * fx) ** 2)
            self._q = p.c * (p.kappa ** 2 + lap) ** 2
        return self._rho, self._q

    def _fourier(self, lam):
        from scipy.sparse.linalg import LinearOperator

        rho, q = self._spectra()
        den = rho + lam * q
        my, mx = self.prior.mesh_shape
        shape = den.shape

        def apply(v):
            g = np.zeros(shape)
            g[:my, :mx] = self._to_grid(v)
            return self._from_grid(np.fft.irfft2(np.fft.rfft2(g) / den[:, : shape[1] // 2 + 1],
                                                 s=shape)[:my, :mx])

        n = self.data.n_mesh
        return LinearOperator((n, n), matvec=apply, dtype=float)

    def solve(self, coefficient: float, x0=None) -> Solution:
        from scipy.sparse.linalg import LinearOperator, cg

        lam = float(coefficient)
        n = self.data.n_mesh
        Qw = self.prior.Qw
        count = [0]

        def matvec(v):
            count[0] += 1
            return self.data.F(v) + lam * (Qw @ v)

        A = LinearOperator((n, n), matvec=matvec, dtype=float)
        if self.preconditioner in ("density", "radial"):
            M = self._fourier(lam)
        else:
            _, lu = self._P(lam)
            M = LinearOperator((n, n), matvec=lu.solve, dtype=float)
        x0 = self._last if x0 is None else x0
        s, info = cg(A, self.data.D, x0=x0, rtol=CG_RTOL, maxiter=CG_MAXITER, M=M)
        if info > 0:
            logger.warning("matrix-free CG did not converge in %d iterations at "
                           "coefficient %.4g", info, lam)
        self._last = s
        return Solution(lam, s, self.data.chi_squared(s), lam * float(s @ (Qw @ s)),
                        count[0])

    # -- evidence -------------------------------------------------------
    def log_det_A(self, lam: float) -> float:
        """ln det(F + lambda C^-1) = ln det P + ln det(P^-1 A).

        ln det P is exact from the sparse factorisation (Powell et al. stop
        there; that is their approximation, 1-20% off). The remainder is
        estimated by stochastic Lanczos quadrature on P^-1 A, which the
        preconditioner makes well conditioned, so a few tens of Lanczos steps
        suffice. Probe vectors v ~ N(0, P^-1) are drawn as P^-1 B^T xi with
        B = [sqrt(kappa_0) M; sqrt(lambda) R], P = B^T B, and the Lanczos runs
        in the P inner product; the probes come from a fixed seed, so the
        estimate is a deterministic, smooth function of lambda.
        """
        P, lu = self._P(lam)
        log_det_P = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
        n = self.data.n_mesh
        rng = np.random.default_rng(SLQ_SEED)
        Qw = self.prior.Qw
        est = []
        for _ in range(SLQ_PROBES):
            xi_m = rng.standard_normal(self.data.M.shape[0])
            xi_r = rng.standard_normal(n)
            v = lu.solve(np.sqrt(self.kappa0) * (self.data.MT @ xi_m)
                         + np.sqrt(lam) * (self.prior.R.T @ xi_r))
            # Lanczos on T = P^-1 A, P-orthonormal basis
            Pv = P @ v
            norm2 = float(v @ Pv)
            q = v / np.sqrt(norm2)
            Pq = Pv / np.sqrt(norm2)
            q_prev, b = np.zeros(n), 0.0
            alpha, beta, Q, PQ = [], [], [], []
            for j in range(SLQ_STEPS):
                Q.append(q)
                PQ.append(Pq)
                Aq = self.data.F(q) + lam * (Qw @ q)            # = P (T q)
                a = float(q @ Aq)
                r = lu.solve(Aq) - a * q - b * q_prev           # T q - ...
                for qq, pq in zip(Q, PQ):                       # reorthogonalise (P)
                    r -= float(pq @ r) * qq
                Pr = P @ r
                b_new = float(np.sqrt(max(r @ Pr, 0.0)))
                alpha.append(a)
                if b_new < 1e-10 * abs(a) or j == SLQ_STEPS - 1:
                    break
                beta.append(b_new)
                q_prev, q, Pq = q, r / b_new, Pr / b_new
                b = b_new
            k = len(alpha)
            T = np.diag(alpha) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
            theta, U = np.linalg.eigh(T)
            est.append(norm2 * float(np.sum(U[0] ** 2 * np.log(np.clip(theta, 1e-300, None)))))
        return log_det_P + float(np.mean(est))

    def log_evidence(self, sol: Solution) -> float:
        lam = sol.coefficient
        log_det_H = self.data.n_mesh * np.log(lam) + self.prior.log_det_Cinv
        return -0.5 * (sol.chi_squared + sol.prior_term + self.log_det_A(lam)
                       - log_det_H + self.data.noise_normalization)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

@dataclass
class _Scan:
    criterion: str
    trials: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"criterion": self.criterion, "solver": "matrix-free",
                "trials": self.trials}


@dataclass
class MatrixFreeFit:
    """What `api._products_for` and the parameter record need from a fit.

    Duck-types the parts of `fitting.SingleFit` they read; the uncertainty
    and S/N methods are deliberately absent (see the module docstring).
    """

    geometry: object
    prior: dict
    scan: _Scan | None
    model_image: np.ndarray
    model_mesh_values: np.ndarray
    chi_squared: float
    log_evidence: float
    positive_only: bool = False
    points: list = field(default_factory=list)

    @property
    def coefficient(self) -> float:
        return float(self.prior["coefficient"])

    @property
    def model_mesh_image(self) -> np.ndarray:
        """As `SingleFit.model_mesh_image`: [Jy / mesh pixel], mesh order."""
        g = self.geometry
        k2 = (g.shape_native[0] // g.mesh_shape[0]) ** 2
        return np.asarray(self.model_mesh_values).reshape(g.mesh_shape) * k2


def _structure_ratio(data: DataOperator, imager, s, chi2):
    from .fitting import _structure_ratio_from_map

    model = data.image(s)
    resid = imager.dirty_image_of_data() - imager.dirty_image_of_model(model)
    return float(_structure_ratio_from_map(resid, chi2, imager, data.n_data))


def _search(system: MatrixFreeSystem, criterion: str, imager, chi2_target: float,
            scan: _Scan, label: str) -> Solution:
    """The strength by `criterion`: a coarse pass in whole decades from the
    strong end (warm starts run downhill in lambda), then bisection to 0.1 dex
    (structure, discrepancy) or a golden refinement of the evidence."""
    data = system.data

    def metric(sol):
        if criterion == "structure":
            return _structure_ratio(data, imager, sol.s, sol.chi_squared) - 1.0
        if criterion == "discrepancy":
            return sol.chi_squared / data.n_data - chi2_target
        return system.log_evidence(sol)

    cache = {}

    def at(logc):
        key = round(float(logc), 4)
        if key not in cache:
            sol = system.solve(10.0 ** key)
            m = metric(sol)
            cache[key] = (sol, m)
            scan.trials.append({"pass": label, "coefficient": float(10.0 ** key),
                                "metric": float(m), "chi2_per_datum": sol.chi_squared / data.n_data,
                                "matvecs": sol.iterations})
            logger.info("  coefficient=%.4g  %s=%.4g  chi2/N=%.4f  (%d matrix-vector products)",
                        10.0 ** key, "log_evidence" if criterion == "evidence" else
                        f"{criterion}-target", m, sol.chi_squared / data.n_data, sol.iterations)
        return cache[key]

    # walk down from the strong end in whole decades, stopping at the first
    # crossing (structure, discrepancy) or once the evidence has turned over:
    # the weak-prior solves are the expensive ones, and are only made if needed
    grid, vals = [], []
    for g in np.arange(12.0, -0.5, -1.0):
        grid.append(g)
        vals.append(at(g)[1])
        if criterion == "evidence":
            if len(vals) >= 3 and vals[-1] < vals[-2] < max(vals):
                break
        elif len(vals) >= 2 and np.sign(vals[-1]) != np.sign(vals[-2]):
            break
    grid = np.asarray(grid)
    if criterion == "evidence":
        i = int(np.argmax(vals))
        lo, hi = grid[min(i + 1, len(grid) - 1)], grid[max(i - 1, 0)]
        gr = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        while b - a > 0.05:
            c, d = b - gr * (b - a), a + gr * (b - a)
            if at(c)[1] > at(d)[1]:
                b = d
            else:
                a = c
        best = max(cache, key=lambda k: cache[k][1])
        return cache[best][0]
    # structure ratio and chi^2 both rise with lambda: bisect the crossing
    if np.sign(vals[-1]) == np.sign(vals[0]):
        k = int(np.argmin(np.abs(vals)))
        logger.warning("matrix-free search: the %s target is not reached in "
                       "1e0..1e12; using the closest strength", criterion)
        return at(grid[k])[0]
    hi, lo = grid[-2], grid[-1]                     # metric(hi) > 0 > metric(lo)
    while hi - lo > 0.1:
        mid = 0.5 * (hi + lo)
        if at(mid)[1] > 0:
            hi = mid
        else:
            lo = mid
    return at(hi if abs(at(hi)[1]) < abs(at(lo)[1]) else lo)[0]


def fit_matrix_free(terms, geometry, mask, stub, imager, *, reg_kind: str, criterion: str,
                    scale: float, nu: float, envelope: dict | None,
                    coefficient: float | None = None, chi2_target: float = 1.0) -> MatrixFreeFit:
    """The matrix-free counterpart of `fitting.fit_dataset` on streamed terms.

    ``coefficient`` fixes the strength (a cube channel's frozen prior);
    otherwise it is chosen by ``criterion``. ``reg_kind`` "matern", "gaussian"
    (envelope) or "adaptive" (two passes: plain Matern, then weights from
    that model, as `fitting.fit_dataset` does).
    """
    from . import fitting

    if reg_kind not in ("matern", "gaussian", "adaptive", "exponential"):
        raise NotImplementedError(
            f"--reg {reg_kind} is not available on the matrix-free path (meshes "
            f"above {MATRIX_FREE_MIN_MESH_PIXELS} pixels); use matern, gaussian "
            "or adaptive, or a coarser --mesh")
    if reg_kind == "exponential":
        nu = 0.5
    started = time.perf_counter()
    fit = fitting.fit_at(stub, geometry.mesh_shape, "constant", 1.0, positive_only=False)
    mapper = fit.inversion.linear_obj_list[0]
    M = sparse_mapping(mapper)
    inside = ~np.asarray(mask.native if hasattr(mask, "native") else mask, dtype=bool)
    data = DataOperator(terms, M, inside)
    order, mshape, mpix = mesh_grid_order(np.asarray(mapper.source_plane_mesh_grid.array))
    pts = np.asarray(mapper.source_plane_mesh_grid.array)
    logger.info("matrix-free solve: %d mesh pixels, operators ready in %.1f s "
                "(no %d x %d matrix is formed)", data.n_mesh,
                time.perf_counter() - started, data.n_mesh, data.n_mesh)

    def weights_for(env, brightness=None):
        if reg_kind == "gaussian" and env:
            sigma = env["fwhm"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            c = env.get("centre", (0.0, 0.0))
            g = np.exp(-0.5 * ((pts[:, 0] - c[0]) ** 2 + (pts[:, 1] - c[1]) ** 2) / sigma ** 2)
            fl = env.get("floor", 1e-2)
            return fl + (1.0 - fl) * g
        if reg_kind == "adaptive" and brightness is not None:
            b = np.clip(np.asarray(brightness, float).ravel(), 0.0, None)
            b = b / b.max() if b.max() > 0 else b
            fl, pw = env.get("floor", 1e-2), env.get("power", 1.0)
            return fl + (1.0 - fl) * b ** pw
        return None

    scan = None if coefficient is not None else _Scan(criterion=criterion)

    def one_pass(weights, label):
        prior = SparseMaternPrior(order, mshape, mpix, scale, weights=weights)
        system = MatrixFreeSystem(data, prior)
        if coefficient is not None:
            return system, system.solve(float(coefficient))
        logger.info("matrix-free %s (%s)...", label, criterion)
        return system, _search(system, criterion, imager, chi2_target, scan, label)

    brightness = (envelope or {}).get("brightness") if reg_kind == "adaptive" else None
    if reg_kind == "adaptive" and brightness is None:
        _, first = one_pass(None, "adaptive first pass (plain Matern)")
        brightness = first.s
    system, sol = one_pass(weights_for(envelope, brightness),
                           "fit" if reg_kind != "adaptive" else "adaptive second pass")
    try:
        lnz = system.log_evidence(sol)
    except Exception as e:                        # pragma: no cover - diagnostic
        logger.debug("matrix-free evidence failed: %s", e)
        lnz = float("nan")
    model = data.image(sol.s)
    logger.info("matrix-free fit: coefficient %.4g, chi2/N %.4f, %.0f s in all",
                sol.coefficient, sol.chi_squared / data.n_data,
                time.perf_counter() - started)
    prior = {"coefficient": float(sol.coefficient), "scale": float(scale), "nu": float(nu)}
    out = MatrixFreeFit(geometry=geometry, prior=prior, scan=scan, model_image=model,
                        model_mesh_values=sol.s, chi_squared=sol.chi_squared,
                        log_evidence=lnz)
    return out

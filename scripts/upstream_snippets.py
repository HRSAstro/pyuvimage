"""Self-contained snippets for PyAutoArray issues: matrix-free inversion and
the primary beam. Plain NumPy/SciPy; names follow autoarray where they exist.

Checked against pyuvimage's own implementation by `__main__` below.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, splu


# ---------------------------------------------------------------------------
# 1. Sparse mapping matrix M (image slim x mesh) from the mapper
# ---------------------------------------------------------------------------
def sparse_mapping_from(mapper) -> sp.csr_matrix:
    """Equal to `mapper.mapping_matrix` to the bit, without forming it."""
    idx = np.asarray(mapper.pix_indexes_for_sub_slim_index)            # (n_sub, k)
    wts = np.asarray(mapper.pix_weights_for_sub_slim_index, float)     # (n_sub, k)
    sizes = np.asarray(mapper.pix_sizes_for_sub_slim_index)            # (n_sub,)
    sub2slim = np.asarray(mapper.slim_index_for_sub_slim_index)        # (n_sub,)
    n_slim = int(sub2slim.max()) + 1
    counts = np.bincount(sub2slim, minlength=n_slim).astype(float)     # sub-pixels per pixel
    keep = np.arange(idx.shape[1])[None, :] < sizes[:, None]
    rows = np.broadcast_to(sub2slim[:, None], idx.shape)[keep]
    cols = idx[keep]
    vals = (wts / counts[sub2slim][:, None])[keep]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_slim, int(mapper.pixels)))


# ---------------------------------------------------------------------------
# 2. Data term as an operator: F s = M^T W~ (M s), D = M^T d~
# ---------------------------------------------------------------------------
class DataOperator:
    """`kernel` is the w-tilde preload (`SparseTerms.nufft_precision_operator`,
    shape (2Ny, 2Nx), FFT wraparound order); `dirty_image` is d~ = Re F^H W d
    on the native grid; `inside` the unmasked pixels (native bool)."""

    def __init__(self, kernel, dirty_image, M, inside):
        self.M, self.MT = M.tocsr(), M.T.tocsr()
        self.inside = np.asarray(inside, bool)
        self.kernel_ft = np.fft.rfft2(np.asarray(kernel, float))
        self.kappa0 = float(np.asarray(kernel)[0, 0])          # W~'s diagonal
        self.D = self.MT @ np.asarray(dirty_image, float)[self.inside]

    def wtilde(self, image):                                     # W~ * image, by FFT
        ny, nx = image.shape
        pad = np.zeros((2 * ny, 2 * nx))
        pad[:ny, :nx] = image
        return np.fft.irfft2(np.fft.rfft2(pad) * self.kernel_ft, s=pad.shape)[:ny, :nx]

    def F(self, s):
        image = np.zeros(self.inside.shape)
        image[self.inside] = self.M @ s
        return self.MT @ self.wtilde(image)[self.inside]


# ---------------------------------------------------------------------------
# 3. Matern prior with a sparse precision (SPDE, Lindgren, Rue & Lindstrom 2011)
# ---------------------------------------------------------------------------
def spde_matern_precision(mesh_shape, pixel_scale, scale, weights=None, nu=1.0):
    """C^-1 = W^-1 c L^2 W^-1 on a regular (my, mx) mesh in row-major order,
    L = kappa^2 - Laplacian (5-point, reflecting edges), kappa = sqrt(2 nu)/scale,
    c so the variance at the mesh centre is 1 (K(0) = 1 as for the dense kernel).
    Alpha = nu + 1 = 2 here; integer alpha only (nu = 1, 2, ... in 2-D).
    Returns (C^-1, R, ln det C^-1) with C^-1 = R^T R (R draws the evidence probes)."""
    assert nu == 1.0, "alpha = 2 shown; higher integer alpha is c L^alpha"
    my, mx = mesh_shape
    n = my * mx
    kappa = np.sqrt(2.0 * nu) / scale
    idx = np.arange(n).reshape(my, mx)
    rows, cols = [], []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
    adj = sp.csr_matrix((np.ones(sum(r.size for r in rows)),
                         (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    lap = (adj - sp.diags(np.asarray(adj.sum(axis=1)).ravel())) / pixel_scale ** 2
    L = (kappa ** 2 * sp.identity(n) - lap).tocsc()
    lu = splu(L, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options=dict(SymmetricMode=True))
    e = np.zeros(n)
    e[(my // 2) * mx + mx // 2] = 1.0
    c = float(lu.solve(e) @ lu.solve(e))                         # e^T L^-2 e
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    R = (np.sqrt(c) * L @ sp.diags(1.0 / w)).tocsr()
    log_det = n * np.log(c) + 2 * np.sum(np.log(np.abs(lu.U.diagonal()))) - 2 * np.sum(np.log(w))
    return (R.T @ R).tocsr(), R, float(log_det)


# ---------------------------------------------------------------------------
# 4. Solve (F + lam C^-1) s = D by CG, Powell et al. (2021) preconditioner
# ---------------------------------------------------------------------------
def powell_preconditioner(data, Cinv, lam):
    """P = kappa0 M^T M + lam C^-1: W~ replaced by its peak. Sparse, SPD,
    factorised once per lam; also gives ln det P exactly for the evidence."""
    P = (data.kappa0 * (data.MT @ data.M) + lam * Cinv).tocsc()
    return P, splu(P, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                   options=dict(SymmetricMode=True))


def solve(data, Cinv, lam, x0=None, rtol=1e-8):
    n = data.D.size
    A = LinearOperator((n, n), matvec=lambda v: data.F(v) + lam * (Cinv @ v))
    _, lu = powell_preconditioner(data, Cinv, lam)
    s, info = cg(A, data.D, x0=x0, rtol=rtol, maxiter=5000,
                 M=LinearOperator((n, n), matvec=lu.solve))
    return s


# ---------------------------------------------------------------------------
# 5. ln det(F + lam C^-1) = ln det P + ln det(P^-1 A), the second by
#    stochastic Lanczos quadrature (Ubaru, Chen & Saad 2017) in the P inner product
# ---------------------------------------------------------------------------
def log_det_A(data, Cinv, R, lam, n_probes=10, n_steps=40, seed=20221):
    P, lu = powell_preconditioner(data, Cinv, lam)
    log_det_P = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
    n = data.D.size
    rng = np.random.default_rng(seed)                 # fixed: smooth in lam
    est = []
    for _ in range(n_probes):
        # v ~ N(0, P^-1): P = B^T B with B = [sqrt(kappa0) M; sqrt(lam) R]
        v = lu.solve(np.sqrt(data.kappa0) * (data.MT @ rng.standard_normal(data.M.shape[0]))
                     + np.sqrt(lam) * (R.T @ rng.standard_normal(n)))
        Pv = P @ v
        norm2 = float(v @ Pv)
        q, Pq = v / np.sqrt(norm2), Pv / np.sqrt(norm2)
        q_prev, b, alpha, beta, Q, PQ = np.zeros(n), 0.0, [], [], [], []
        for j in range(n_steps):                      # Lanczos on P^-1 A, P-orthonormal
            Q.append(q); PQ.append(Pq)
            Aq = data.F(q) + lam * (Cinv @ q)
            a = float(q @ Aq)
            r = lu.solve(Aq) - a * q - b * q_prev
            for qq, pq in zip(Q, PQ):                 # full reorthogonalisation
                r -= float(pq @ r) * qq
            Pr = P @ r
            b_new = float(np.sqrt(max(r @ Pr, 0.0)))
            alpha.append(a)
            if b_new < 1e-10 * abs(a) or j == n_steps - 1:
                break
            beta.append(b_new)
            q_prev, q, Pq, b = q, r / b_new, Pr / b_new, b_new
        k = len(alpha)
        T = np.diag(alpha) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
        theta, U = np.linalg.eigh(T)
        est.append(norm2 * float(np.sum(U[0] ** 2 * np.log(np.clip(theta, 1e-300, None)))))
    return log_det_P + float(np.mean(est))


def log_evidence(data, Cinv, R, log_det_Cinv, lam, s, data_term, noise_normalization):
    """The usual five terms; chi^2 = d^T N^-1 d - 2 s^T D + s^T F s."""
    chi2 = data_term - 2 * s @ data.D + s @ data.F(s)
    prior = lam * s @ (Cinv @ s)
    log_det_H = s.size * np.log(lam) + log_det_Cinv
    return -0.5 * (chi2 + prior + log_det_A(data, Cinv, R, lam) - log_det_H
                   + noise_normalization)


# ---------------------------------------------------------------------------
# 6. Positivity: primal-dual interior point (Mehrotra), matrix-vector products
#    only; then the settled support solved exactly (exact zeros). Condensed:
#    pyuvimage's `matfree.MatrixFreeSystem.solve_positive` / `_polish` add the
#    support-settling test and active-set corrections.
# ---------------------------------------------------------------------------
def solve_positive(data, Cinv, lam, max_steps=60, tol=1e-9):
    n, b = data.D.size, data.D
    Av = lambda v: data.F(v) + lam * (Cinv @ v)
    P, _ = powell_preconditioner(data, Cinv, lam)
    diag = data.kappa0 * (data.MT @ data.M).diagonal() + lam * Cinv.diagonal()
    x0 = np.clip(b / diag, 0, None)                              # one Jacobi step
    x = x0 + 1e-2 * x0.max()
    g = Av(x) - b
    z = np.maximum(g, 0) + 1e-2 * np.abs(g).max()
    step = lambda v, dv: min(1.0, np.min(-v[dv < 0] / dv[dv < 0])) if (dv < 0).any() else 1.0
    for _ in range(max_steps):
        rd, mu = g - z, x @ z / n                                # KKT: A x - D = z, x z = 0
        gap, res = n * mu / abs(x @ b), np.linalg.norm(rd) / np.linalg.norm(b)
        if gap < tol and res < tol:
            break
        dg = z / x
        lu = splu((P + sp.diags(dg)).tocsc())                    # P plus the barrier diagonal
        Mi = LinearOperator((n, n), matvec=lu.solve)
        op = LinearOperator((n, n), matvec=lambda v: Av(v) + dg * v)
        rtol = float(np.clip(0.1 * min(gap, res + gap), 1e-8, 1e-2))
        dx, _ = cg(op, -rd - z, rtol=rtol, M=Mi)                 # predictor
        dz = -z - dg * dx
        sigma = (((x + step(x, dx) * dx) @ (z + step(z, dz) * dz)) / n / mu) ** 3
        corr = (sigma * mu - dx * dz) / x                        # centring + corrector
        dx, _ = cg(op, -rd - z + corr, x0=dx, rtol=rtol, M=Mi)
        dz = -z + corr - dg * dx
        x, z = x + 0.99 * step(x, dx) * dx, z + 0.99 * step(z, dz) * dz
        g = Av(x) - b
    # exact solve on the support (held where z_i > d_i x_i); repeat with the
    # negatives held if any come out negative (primal-dual active set)
    free = z <= diag * x
    idx = np.flatnonzero(free)
    sub = LinearOperator((idx.size, idx.size),
                         matvec=lambda v: Av(np.bincount(idx, v, minlength=n))[idx])
    s_free, _ = cg(sub, b[idx], x0=x[idx], rtol=1e-8,
                   M=LinearOperator((idx.size, idx.size),
                                    matvec=splu(P[idx][:, idx].tocsc()).solve))
    s = np.zeros(n)
    s[idx] = s_free
    return np.clip(s, 0, None)


# ---------------------------------------------------------------------------
# 7. Primary beam
# ---------------------------------------------------------------------------
def primary_beam_map(shape, pixel_scale, frequency_hz, dish_diameter_m,
                     pb_factor=1.13, image_centre_offset_arcsec=None):
    """Gaussian PB, FWHM = pb_factor lambda / D, peak 1 at the *phase centre*
    (the pointing), not the grid centre: after recentring the image by
    (y0, x0) the peak sits at row cy + y0/pix, col cx - x0/pix (row 0 north,
    columns toward +x = west)."""
    fwhm = pb_factor * (299792458.0 / frequency_hz) / dish_diameter_m * 206264.806
    sig = fwhm / (2 * np.sqrt(2 * np.log(2))) / pixel_scale
    ny, nx = shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    if image_centre_offset_arcsec:
        cy += image_centre_offset_arcsec[0] / pixel_scale
        cx -= image_centre_offset_arcsec[1] / pixel_scale
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sig ** 2)


def pb_correct(image, pb, cutoff=0.1):
    """image / PB, NaN below the cutoff (as CASA's pbcor)."""
    out = np.full(image.shape, np.nan)
    good = pb >= cutoff
    out[good] = image[good] / pb[good]
    return out


class PrimaryBeamRegularization:
    """A prior on the *true* sky t for a model of the apparent sky a = P t.

    Putting the PB in the response (A' = A P, solve for t) gives F' = P F P,
    D' = P D; rewritten for a it is (F + P^-1 H P^-1) a = D. So the PB enters
    only the regularization: H -> P^-1 H P^-1, with ln det H gaining
    -2 sum ln pb. F, D, the solvers and positivity are unchanged, the model
    stays apparent (pbcor = model / pb), and the evidence is the true-sky
    prior's. The PB is evaluated at the mesh pixel centres, floored so pixels
    at pb -> 0 are not pinned by a 1/pb^2 weight.

    Matrix-free: the same thing is the prior weights times the PB, W -> W P,
    in `spde_matern_precision(weights=w * pb)`.
    """

    def __init__(self, inner, fwhm, centre=(0.0, 0.0), floor=0.1):
        self.inner, self.fwhm, self.centre, self.floor = inner, fwhm, centre, floor

    def pb_at(self, yx):
        sig = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
        r2 = (yx[:, 0] - self.centre[0]) ** 2 + (yx[:, 1] - self.centre[1]) ** 2
        return np.clip(np.exp(-0.5 * r2 / sig ** 2), self.floor, 1.0)

    def regularization_matrix_from(self, linear_obj, xp=np):
        r = 1.0 / self.pb_at(np.asarray(linear_obj.source_plane_mesh_grid.array))
        return self.inner.regularization_matrix_from(linear_obj, xp=xp) * r[:, None] * r[None, :]

    def log_det_regularization_matrix_term_from(self, linear_obj, xp=np):
        pb = self.pb_at(np.asarray(linear_obj.source_plane_mesh_grid.array))
        return self.inner.log_det_regularization_matrix_term_from(linear_obj, xp=xp) \
            - 2.0 * np.sum(np.log(pb))

    def regularization_term_from(self, linear_obj, reconstruction, xp=np):
        pb = self.pb_at(np.asarray(linear_obj.source_plane_mesh_grid.array))
        return self.inner.regularization_term_from(linear_obj, reconstruction / pb, xp=xp)


if __name__ == "__main__":       # checks against pyuvimage on a small mock
    import os
    import sys

    os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from scipy.optimize import nnls

    import test_matfree as T
    from pyuvimage import matfree

    small = T.small.__wrapped__()
    terms, mapper, Fd = small["terms"], small["mapper"], small["F"]
    M = sparse_mapping_from(mapper)
    assert (M.toarray() == np.asarray(mapper.mapping_matrix)).all()
    data = DataOperator(terms.kernel, terms.dirty_image, M, small["data"].inside)
    v = np.random.default_rng(0).normal(size=M.shape[1])
    print("F    vs dense:", np.abs(data.F(v) - Fd @ v).max() / np.abs(Fd @ v).max())
    order, shape, pix = small["order"], small["shape"], small["pix"]
    prior = matfree.SparseMaternPrior(order, shape, pix, 0.3)
    Cinv_mesh, R_mesh, ldc = prior.Qw, prior.R, prior.log_det_Cinv
    # the snippet works in grid (row-major) order; pyuvimage's mesh order is a permutation
    Cinv_g, R_g, ldc_g = spde_matern_precision(shape, pix, 0.3)
    perm = sp.csr_matrix((np.ones(order.size), (np.arange(order.size), order)))
    print("C^-1 vs pyuvimage:", abs(perm @ Cinv_g @ perm.T - Cinv_mesh).max() / abs(Cinv_mesh).max(),
          " ln det:", ldc_g - ldc)
    for lam in (1e7, 1e9):
        s = solve(data, Cinv_mesh, lam)
        ref = np.linalg.solve(Fd + lam * Cinv_mesh.toarray(), data.D)
        A = Fd + lam * Cinv_mesh.toarray()
        ld = log_det_A(data, Cinv_mesh, R_mesh, lam)
        print(f"lam {lam:.0e}: CG vs dense {np.abs(s - ref).max() / np.abs(ref).max():.1e}, "
              f"ln det A rel err {(ld - np.linalg.slogdet(A)[1]) / np.linalg.slogdet(A)[1]:.1e}")
        L = np.linalg.cholesky(A)
        nn, _ = nnls(L.T, np.linalg.solve(L, data.D), maxiter=50 * A.shape[0])
        sp_ = solve_positive(data, Cinv_mesh, lam)
        print(f"          positive vs NNLS {np.abs(sp_ - nn).max() / nn.max():.1e}, "
              f"support mismatch {int(((sp_ == 0) != (nn == 0)).sum())}")

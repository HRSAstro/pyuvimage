import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
import autogalaxy as ag
from pyuvimage import mock, fitting
AS2RAD = np.pi/(180*3600)

def run_case(label, compact_flux):
    uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(
        n_vis=600, mesh_n=32, sigma_jy=5e-4, compact_flux=compact_flux)
    uv, d, nz = uvd.flattened(); N=2*len(d)
    ds = fitting.make_dataset(uv, d, nz, geom)
    sf = fitting.fit_dataset(ds, geom, reg_kind="matern",
                             prior={"coefficient":3e7,"scale":0.25,"nu":1.5}, positive_only=False)
    inv = sf.fit.inversion
    Aop = np.asarray(inv.operated_mapping_matrix)
    wr, wi = 1/np.asarray(nz).real**2, 1/np.asarray(nz).imag**2
    dr, di = np.asarray(ds.data).real, np.asarray(ds.data).imag
    curv = lambda A,B: A.real.T@(wr[:,None]*B.real) + A.imag.T@(wi[:,None]*B.imag)
    dvec = lambda A: A.real.T@(wr*dr) + A.imag.T@(wi*di)
    H = np.asarray(inv.regularization_matrix); F = curv(Aop,Aop); D = dvec(Aop)
    y0, x0 = comps["compact"]["centre"]
    ph = -2*np.pi*(x0*AS2RAD*uv[:,0] + y0*AS2RAD*uv[:,1])
    P = (np.cos(ph)+1j*np.sin(ph))[:,None]
    B = curv(Aop,P); C = curv(P,P); Dp = dvec(P)
    K = np.block([[F+H, B],[B.T, C]])
    th = np.linalg.solve(K, np.concatenate([D, Dp]))
    Cov = np.linalg.inv(K)
    a, sig_a = th[-1], np.sqrt(Cov[-1,-1])
    const = np.sum(dr**2*wr + di**2*wi)
    Faug = np.block([[F,B],[B.T,C]])
    chi2_pt = const + th@Faug@th - 2*th@np.concatenate([D,Dp])
    s0 = np.linalg.solve(F+H, D); chi2_0 = const + s0@F@s0 - 2*s0@D
    print("%-22s point flux = %9.3g +- %.3g  (%5.1f sigma)   dchi2 = %8.1f   truth = %g"
          % (label, a, sig_a, a/sig_a, chi2_0-chi2_pt, compact_flux), flush=True)

run_case("with a real knot", 0.012)
run_case("faint knot", 0.002)
run_case("NO knot (extended only)", 0.0)

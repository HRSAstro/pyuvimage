import numpy as np, logging, time, json
logging.basicConfig(level=logging.WARNING)
import autogalaxy as ag
from autoarray.inversion.regularization.matern_kernel import MaternKernel, inv_via_cholesky, apply_jitter
from pyuvimage import mock, fitting, beam as bm

uvd, truth, geom, comps = mock.make_extended_plus_compact_dataset(n_vis=600, mesh_n=32, sigma_jy=5e-4)
uv, d, nz = uvd.flattened(); N = 2*len(d)
ds = fitting.make_dataset(uv, d, nz, geom)
imager = bm.DirtyImager(ds); rms = imager.rms
bfit = bm.fit_beam(imager.dirty_beam, geom.pixel_scale)
BEAM = float(np.sqrt(bfit.bmaj_arcsec*bfit.bmin_arcsec))
dirty = np.asarray(imager.dirty_image(np.asarray(ds.data)))
ny = dirty.shape[0]
py, px = np.unravel_index(np.nanargmax(dirty), dirty.shape)
peak_yx = (((ny-1)/2 - py)*geom.pixel_scale, (px-(ny-1)/2)*geom.pixel_scale)

# truth + apertures on the product grid
k = geom.shape_native[0]//truth.shape[0]
truth_img = np.kron(truth, np.ones((k,k)))/k**2
n = truth_img.shape[0]; p = geom.pixel_scale
yy, xx = np.mgrid[0:n,0:n]
def rmap(cy,cx): return np.hypot(((n-1)/2-yy)*p-cy, (xx-(n-1)/2)*p-cx)
r_c = rmap(*comps["compact"]["centre"]); r_e = rmap(0.,0.)
c_ap = r_c < 0.30; e_ap = (r_e < 1.2) & ~c_ap
print(f"SETUP beam={BEAM:.3f}\" dirty_peak={peak_yx} true_knot={comps['compact']['centre']} "
      f"truth: cmp={truth_img[c_ap].sum():.5f} ext={truth_img[e_ap].sum():.5f} "
      f"cmp_peak={truth_img[c_ap].max():.6f}", flush=True)

class GibbsKernel(MaternKernel):
    """Non-stationary kernel: correlation length varies with brightness."""
    def __init__(self, coefficient=1.0, ell=None, weights=None):
        super().__init__(coefficient=coefficient, scale=1.0, nu=1.5, jitter_relative=True)
        self.ell = np.asarray(ell, float); self.w = None if weights is None else np.asarray(weights,float)
    def regularization_matrix_from(self, linear_obj, xp=np):
        pts = np.asarray(linear_obj.source_plane_mesh_grid.array)
        l2 = self.ell**2
        s = l2[:,None] + l2[None,:]
        d2 = np.sum(pts*pts,1)[:,None] + np.sum(pts*pts,1)[None,:] - 2*(pts@pts.T)
        C = (2.0*self.ell[:,None]*self.ell[None,:]/s) * np.exp(-np.maximum(d2,0)/s)
        if self.w is not None: C = C*(self.w[:,None]*self.w[None,:])
        return self.coefficient * inv_via_cholesky(apply_jitter(C, 1e-8, True))

def fit_coeff(make_gal, target=N, lo=-2.0, hi=16.0, steps=13):
    """Bisect the coefficient so chi2 == target (fair comparison point)."""
    best=None
    for _ in range(steps):
        mid=0.5*(lo+hi)
        f=ag.FitInterferometer(dataset=ds, galaxies=[make_gal(10**mid)],
                               settings=ag.Settings(use_positive_only_solver=False))
        try: c=float(f.inversion.fast_chi_squared)
        except Exception: c=np.nan
        if best is None or (np.isfinite(c) and abs(c-target)<abs(best[1]-target)): best=(f,c,10**mid)
        if not np.isfinite(c) or c<target: lo=mid
        else: hi=mid
        if hi-lo<0.02: break
    return best

def pix_for(reg): return ag.Pixelization(mesh=ag.mesh.RectangularUniform(shape=(32,32)), regularization=reg)

STORE = {}
GRID = ag.Grid2D.uniform(shape_native=geom.shape_native, pixel_scales=geom.pixel_scale)

def model_image_from(f, profile=None):
    """Exact model image via each linear object's mapping matrix."""
    slim = None
    for obj, vals in f.inversion.reconstruction_dict.items():
        contrib = np.asarray(obj.mapping_matrix) @ np.asarray(vals)
        slim = contrib if slim is None else slim + contrib
    return np.asarray(ag.Array2D(values=slim, mask=ds.real_space_mask).native)

def metrics(tag, res, t0, profile=None):
    f, chi, coef = res
    mi = model_image_from(f, profile)
    # self-check: does the assembled image reproduce the fitted visibilities?
    chk = np.asarray(ds.transformer.visibilities_from(
        image=ag.Array2D(values=mi, mask=ds.real_space_mask)))
    agree = float(np.max(np.abs(chk - np.asarray(f.model_data)))
                  / max(np.max(np.abs(np.asarray(f.model_data))), 1e-30))
    mv = np.asarray(f.model_data)
    rs = np.asarray(imager.dirty_image(np.asarray(ds.data)-mv))/rms
    out = dict(chi2n=round(chi/N,3),
        corr=round(float(np.corrcoef(mi.ravel(), truth_img.ravel())[0,1]),4),
        cmp_flux=round(float(mi[c_ap].sum()/truth_img[c_ap].sum()),3),
        ext_flux=round(float(mi[e_ap].sum()/truth_img[e_ap].sum()),3),
        cmp_peak=round(float(mi[c_ap].max()/truth_img[c_ap].max()),3),
        resid_cmp=round(float(np.nanmax(np.abs(rs[c_ap]))),1),
        resid_ext=round(float(np.nanmax(np.abs(rs[e_ap]))),1),
        resid_rms=round(float(np.nanstd(rs)),2), selfchk=float(f"{agree:.1e}"),
        t=round(time.time()-t0))
    print("R", tag, json.dumps(out), flush=True)
    STORE[tag] = dict(model=mi, resid=rs, metrics=out)
    return f

def brightness_from(f):
    for obj, vals in f.inversion.reconstruction_dict.items():
        if type(obj).__name__ == "Mapper":
            return np.clip(np.asarray(vals), 0, None)
    raise RuntimeError("no mesh in reconstruction")

# 1. matern baseline
t0=time.time(); base = metrics("matern", fit_coeff(
    lambda c: ag.Galaxy(redshift=1., pixelization=pix_for(fitting.make_regularization("matern",c,BEAM,1.5)))), t0)
b1 = brightness_from(base)

# 2-4. adaptive, varying power
from pyuvimage.envelope import AdaptiveMatern
for power in (0.5, 1.0, 2.0):
    t0=time.time(); f=metrics(f"adaptive p={power}", fit_coeff(
        lambda c,pw=power: ag.Galaxy(redshift=1., pixelization=pix_for(
            AdaptiveMatern(coefficient=c, scale=BEAM, nu=1.5, brightness=b1, power=pw)))), t0)
    if power==1.0: adapt1=f

# 5. adaptive, second iteration
b2 = brightness_from(adapt1)
t0=time.time(); metrics("adaptive x2", fit_coeff(
    lambda c: ag.Galaxy(redshift=1., pixelization=pix_for(
        AdaptiveMatern(coefficient=c, scale=BEAM, nu=1.5, brightness=b2, power=1.0)))), t0)

# 6-7. hybrid: mesh + one linear Gaussian at the dirty-image peak
for tag,regf in (("hybrid matern", lambda c: fitting.make_regularization("matern",c,BEAM,1.5)),
                 ("hybrid adaptive", lambda c: AdaptiveMatern(coefficient=c, scale=BEAM, nu=1.5, brightness=b1))):
    knot_prof = ag.lp_linear.Gaussian(centre=peak_yx, sigma=0.03)
    t0=time.time(); metrics(tag, fit_coeff(
        lambda c,rf=regf: ag.Galaxy(redshift=1., pixelization=pix_for(rf(c)),
            knot=ag.lp_linear.Gaussian(centre=peak_yx, sigma=0.03))), t0,
        profile=knot_prof)

# 8-9. Gibbs non-stationary length scale
bn = b1/max(b1.max(),1e-30)
for tag, wts in (("gibbs", None), ("gibbs+amp", 1e-2+(1-1e-2)*bn)):
    ell = 0.25*BEAM + (BEAM-0.25*BEAM)*(1.0-bn)      # short where bright
    t0=time.time(); metrics(tag, fit_coeff(
        lambda c,e=ell,w=wts: ag.Galaxy(redshift=1., pixelization=pix_for(
            GibbsKernel(coefficient=c, ell=e, weights=w)))), t0)


np.savez_compressed("/tmp/variants.npz",
    truth=truth_img, pixel_scale=geom.pixel_scale, fov=geom.fov_arcsec,
    knot=np.array(comps["compact"]["centre"]), beam=BEAM,
    order=np.array(list(STORE.keys()), dtype=object),
    models=np.array([STORE[t]["model"] for t in STORE]),
    resids=np.array([STORE[t]["resid"] for t in STORE]),
    metrics=json.dumps({t: STORE[t]["metrics"] for t in STORE}),
    allow_pickle=True)
print("SAVED", len(STORE), "variants", flush=True)

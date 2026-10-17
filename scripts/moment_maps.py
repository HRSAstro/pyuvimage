#!/usr/bin/env python
"""Moment maps (0, 1, 2 and peak) from a pyuvimage model cube.

Each channel is masked to where the model is significant, so the noise the
fit absorbed into the model stays out of the moments. The mask comes from,
in order of preference (--mask-from auto):

  snr       snr.fits beside the cube (written with --uncertainty statistical
            or bayesian): the model at the restoring beam's resolution over
            its own 1-sigma, per channel and per pixel. Keep snr >= nsigma.
  residual  model_reconvolved.fits against each channel's own noise, the
            robust (MAD) rms of residual.fits x its RMS header [Jy/beam].
            Keep reconvolved >= nsigma x sigma_channel. Per channel, so a
            channel with more noise or phase errors gets a higher cut.
  empty     model_reconvolved.fits against the rms of empty channel(s)
            (--rms-channels, the first by default), one cut for all.
  self      the cube itself against the rms of the empty channel(s).

Thresholding the model pixel by pixel ('self') is a poor mask: its noise is
sparse positive spikes, 12-16 x the rms in an empty NGC 4274 channel. At the
beam's resolution the absorbed noise reaches ~7.6 x the residual noise, so the
default cut is 5: on NGC 4274 that keeps 96-99% of the empty channels out and
~87% of the line's model flux in. The table printed per channel shows what
each channel kept; check the empty ones and adjust --nsigma.

    python moment_maps.py pyuvimage_out_cube/model.fits --restfreq 230.538e9
    python moment_maps.py model.fits --restfreq 230.538e9 --mask-from residual \\
        --nsigma 4 --vsys 930 --out ngc4274

Error maps. If posterior_samples.npz is beside the cube (`pyuvimage fit
--posterior-samples N`), every draw from the posterior goes through the same
mask and the same moments, and the scatter over draws is written as
<out>_mom0_err.fits, _mom1_err.fits and _mom2_err.fits (half the 16-84
percentile range, robust to the odd draw whose weights nearly cancel), plus
<out>_mom0_snr.fits. These carry every pixel-to-pixel and channel-to-channel
correlation of the posterior: where the S/N is high they show how much
structure finer than the beam the data support. The mask is held fixed, so
the errors are conditional on it; the draws are Gaussian, so near zero they
go negative and the moment-1 and -2 errors grow at the faint edges.

Writes <out>_mom0.fits, _mom1.fits, _mom2.fits, _peak.fits, _mask.fits and
<out>_moments.png. mom0 is in the cube's unit x km/s (Jy/pixel km/s for
model.fits: summed over pixels it is the flux in Jy km/s); mom1 and mom2 in
km/s; peak in the cube's unit. Velocities are radio, v = c (1 - f/f_rest),
minus --vsys.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

C_KMS = 299792.458


def parse_channels(text: str, n_chan: int) -> list[int]:
    """'0', '0,1,2', '0-3' or '-1' (counting from the end)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        out.extend(range(int(m[1]), int(m[2]) + 1) if m else [int(part)])
    return sorted({c % n_chan for c in out})


def load_cube(path):
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None)
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
    while data.ndim > 3:              # drop a degenerate Stokes axis
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a cube, got shape {data.shape}")
    return data, header


def frequencies(header, n_chan):
    """Plane frequencies [Hz] from the spectral WCS axis."""
    spec = WCS(header).sub(["spectral"])
    if spec.naxis != 1 or not str(spec.wcs.ctype[0]).upper().startswith("FREQ"):
        raise ValueError("the cube needs a FREQ spectral axis")
    f = np.asarray(spec.pixel_to_world_values(np.arange(n_chan)), float).ravel()
    unit = str(spec.wcs.cunit[0]) or "Hz"
    return f * {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(unit, 1.0)


def rms_of(cube, channels):
    x = cube[channels]
    x = x[np.isfinite(x)]
    x = x[np.abs(x) != 0]
    return float(np.sqrt(np.mean(x ** 2)))


def robust_std(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def beside(cube_path, name):
    p = Path(cube_path).with_name(name)
    return p if p.exists() else None


def build_mask(args, cube, n_chan, error):
    """(keep, description, per-channel cut or None, unit) for --mask-from."""
    mode = args.mask_from
    if mode == "auto":
        if beside(args.cube, "snr.fits"):
            mode = "snr"
        elif beside(args.cube, "residual.fits") and beside(args.cube, "model_reconvolved.fits"):
            mode = "residual"
        elif beside(args.cube, "model_reconvolved.fits"):
            mode = "empty"
        else:
            mode = "self"

    def need(name):
        p = beside(args.cube, name)
        if p is None:
            error(f"--mask-from {mode} needs {name} beside {args.cube}")
        c, h = load_cube(p)
        if c.shape != cube.shape:
            error(f"{name} has shape {c.shape}, the cube {cube.shape}")
        return c, h

    n = args.nsigma
    if mode == "snr":
        snr, _ = need("snr.fits")
        if not np.isfinite(snr).any():
            error("snr.fits is all NaN: no uncertainty was computed")
        keep = np.nan_to_num(snr) >= n
        return keep, f"snr.fits >= {n:g} (S/N at the restoring beam, per channel)", None, ""
    if mode == "residual":
        rec, rh = need("model_reconvolved.fits")
        res, hh = need("residual.fits")
        rms = float(hh.get("RMS", rh.get("RMS", 0.0)))
        if not rms:
            error("residual.fits has no RMS header to put it in Jy/beam")
        sig = np.array([rms * robust_std(res[c]) for c in range(n_chan)])
        keep = np.nan_to_num(rec) >= n * sig[:, None, None]
        return (keep, f"model_reconvolved.fits >= {n:g} x each channel's residual "
                f"noise ({sig.min():.3g}-{sig.max():.3g} Jy/beam)", n * sig, "Jy/beam")
    if mode in ("empty", "self"):
        if mode == "empty":
            src, sh = need("model_reconvolved.fits")
            name, unit = "model_reconvolved.fits", sh.get("BUNIT", "")
        else:
            src, name, unit = cube, Path(args.cube).name, ""
        chans = parse_channels(args.rms_channels, n_chan)
        rms = rms_of(src, chans)
        if rms <= 0:
            error(f"channel(s) {chans} of {name} are all zero: pick another empty "
                  "channel with --rms-channels")
        keep = np.nan_to_num(src) >= n * rms
        return (keep, f"{name} >= {n:g} x rms of channel(s) {chans} ({rms:.4g} {unit})",
                np.full(n_chan, n * rms), unit)
    error(f"unknown --mask-from {mode!r}")


def moments(cube, vel, keep):
    """Moments of `cube` over the voxels in `keep`."""
    order = np.argsort(vel)                      # integrate in increasing v
    cube, vel, keep = cube[order], vel[order], keep[order]
    dv = np.abs(np.gradient(vel))
    m = np.where(keep, np.nan_to_num(cube), 0.0)
    v = vel[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mom0 = np.sum(m * dv[:, None, None], axis=0)
        w = np.sum(m, axis=0)
        mom1 = np.sum(m * v, axis=0) / w
        mom2 = np.sqrt(np.clip(np.sum(m * (v - mom1) ** 2, axis=0) / w, 0, None))
    peak = np.max(m, axis=0)
    empty = ~keep.any(axis=0) | (w <= 0)
    for a in (mom0, mom1, mom2, peak):
        a[empty] = np.nan
    return mom0, mom1, mom2, peak


def moment_errors(samples_path, cube, vel, keep):
    """(mom0, mom1, mom2) error maps -- half the 16-84 percentile range over
    the posterior draws -- and the number of draws, with the mask fixed."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pyuvimage.posterior import PosteriorSamples

    post = PosteriorSamples(samples_path)
    n_chan = cube.shape[0]
    if post.n_planes != n_chan or post.shape != cube.shape[1:]:
        raise ValueError(f"{samples_path} is for {post.n_planes} planes of "
                         f"{post.shape}, the cube {cube.shape}")
    support = keep.any(axis=0).ravel()
    idx = np.flatnonzero(support)
    n = post.n_samples
    order = np.argsort(vel)
    dv = np.empty_like(vel)
    dv[order] = np.abs(np.gradient(vel[order]))
    S0 = np.zeros((n, idx.size))
    W = np.zeros_like(S0)
    S1 = np.zeros_like(S0)
    S2 = np.zeros_like(S0)
    for c in range(n_chan):
        kc = keep[c].ravel()[idx]
        if not kc.any():
            continue
        m = np.nan_to_num(cube[c]).ravel()[idx][None, :] + \
            post.offsets(c).reshape(n, -1)[:, idx]
        m = np.where(kc[None, :], m, 0.0)
        S0 += m * dv[c]
        W += m
        S1 += m * vel[c]
        S2 += m * vel[c] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mom1 = np.where(W > 0, S1 / W, np.nan)
        mom2 = np.sqrt(np.clip(np.where(W > 0, S2 / W, np.nan) - mom1 ** 2, 0, None))

    def spread(a):
        import warnings

        with warnings.catch_warnings():         # pixels with no valid draw stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(a, [16, 84], axis=0)
        out = np.full(cube.shape[1] * cube.shape[2], np.nan)
        out[idx] = 0.5 * (hi - lo)
        return out.reshape(cube.shape[1:])

    return spread(S0), spread(mom1), spread(mom2), n, post.mode


def plane_header(header, bunit, comment):
    h = WCS(header).celestial.to_header()
    for key in ("BMAJ", "BMIN", "BPA", "TELESCOP", "OBJECT"):
        if key in header:
            h[key] = header[key]
    h["BUNIT"] = bunit
    h["COMMENT"] = comment
    return h


def quicklook(path, maps, header, unit, title, errors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mom0, mom1, mom2, peak = maps
    ny, nx = mom0.shape
    sc = abs(WCS(header).celestial.wcs.cdelt[1]) * 3600.0
    ext = [nx / 2 * sc, -nx / 2 * sc, -ny / 2 * sc, ny / 2 * sc]
    good = np.isfinite(mom0)
    n_rows = 2 if errors is not None else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(17, 4.4 * n_rows), constrained_layout=True,
                             squeeze=False)
    if good.any():
        vc = np.nanmedian(mom1)
        vr = np.nanpercentile(np.abs(mom1 - vc), 99)
        s2 = np.nanpercentile(mom2, 99)
    else:
        vc, vr, s2 = 0.0, 1.0, 1.0
    panels = (("moment 0", mom0, "inferno", None, None, f"{unit} km/s"),
              ("moment 1", mom1, "RdBu_r", vc - vr, vc + vr, "km/s"),
              ("moment 2", mom2, "viridis", 0, s2, "km/s"),
              ("peak", peak, "inferno", None, None, unit))
    if errors is not None:
        e0, e1, e2 = errors
        with np.errstate(invalid="ignore", divide="ignore"):
            snr0 = np.where(e0 > 0, mom0 / e0, np.nan)
        panels = panels + (
            ("moment 0 / its error", snr0, "viridis", 0, np.nanpercentile(snr0, 99)
             if np.isfinite(snr0).any() else 1, "S/N"),
            ("moment 1 error", e1, "magma", 0, np.nanpercentile(e1, 95)
             if np.isfinite(e1).any() else 1, "km/s"),
            ("moment 2 error", e2, "magma", 0, np.nanpercentile(e2, 95)
             if np.isfinite(e2).any() else 1, "km/s"),
            ("moment 0 error", e0, "magma", None, None, f"{unit} km/s"))
    for ax, (t, arr, cmap, lo, hi, cu) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi, extent=ext)
        fig.colorbar(im, ax=ax, shrink=0.85, label=cu)
        ax.set_title(t)
        ax.set_xlabel("dRA [arcsec]")
        if good.any():
            rows, cols = np.nonzero(good)
            pad = 10
            ax.set_xlim((nx / 2 - cols.min() + pad) * sc, (nx / 2 - cols.max() - pad) * sc)
            ax.set_ylim((rows.min() - pad - ny / 2) * sc, (rows.max() + pad - ny / 2) * sc)
    for r in range(n_rows):
        axes[r, 0].set_ylabel("dDec [arcsec]")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Moment maps from a pyuvimage model cube, each channel masked "
                    "to where the model is significant (see the module docstring).")
    p.add_argument("cube", help="the cube to take moments of (e.g. model.fits)")
    p.add_argument("--restfreq", type=float, default=None,
                   help="rest frequency [Hz]; default RESTFRQ/RESTFREQ from the "
                        "header (CO(2-1): 230.538e9)")
    p.add_argument("--vsys", type=float, default=0.0,
                   help="systemic velocity [km/s] subtracted from the velocities")
    p.add_argument("--rms-channels", default="0",
                   help="--mask-from empty/self: the empty channel(s) for the rms: "
                        "'0' (default), '0,1', '0-2', '-1' for the last")
    p.add_argument("--nsigma", type=float, default=5.0,
                   help="the cut, in sigma (default 5; see the module docstring)")
    p.add_argument("--mask-from", default="auto",
                   choices=["auto", "snr", "residual", "empty", "self"],
                   help="auto (default): snr.fits if present, else the residual's "
                        "per-channel noise, else an empty channel's rms")
    p.add_argument("--min-channels", type=int, default=1,
                   help="drop pixels above the threshold in fewer than this many "
                        "channels in total (default 1: off)")
    p.add_argument("--out", default=None,
                   help="output prefix (default: the cube's path without .fits)")
    p.add_argument("--samples", default="auto",
                   help="posterior draws for the error maps: 'auto' (default: "
                        "posterior_samples.npz beside the cube, if any), a path, or 'none'")
    p.add_argument("--no-png", action="store_true", help="skip the quick-look PNG")
    args = p.parse_args(argv)

    cube, header = load_cube(args.cube)
    n_chan = cube.shape[0]
    unit = header.get("BUNIT", "")

    restfreq = args.restfreq or header.get("RESTFRQ") or header.get("RESTFREQ")
    if not restfreq:
        p.error("no rest frequency in the header: pass --restfreq (Hz)")
    vel = C_KMS * (1.0 - frequencies(header, n_chan) / float(restfreq)) - args.vsys

    keep, how, cut, cut_unit = build_mask(args, cube, n_chan, p.error)
    if args.min_channels > 1:
        keep &= (keep.sum(axis=0) >= args.min_channels)[None]
    print(f"mask: {how}")
    print(f"velocities {vel.min():.1f} to {vel.max():.1f} km/s "
          f"(rest {float(restfreq) / 1e9:.7g} GHz, vsys {args.vsys:g} km/s)")
    print(f"{'chan':>5} {'v [km/s]':>9}" + (f" {'cut':>10}" if cut is not None else "")
          + f" {'pixels':>7}  {'flux kept':>10} {'of':>10}  ({unit})")
    for c in range(n_chan):
        pos = np.clip(np.nan_to_num(cube[c]), 0, None)
        print(f"{c:5d} {vel[c]:9.1f}" + (f" {cut[c]:10.3g}" if cut is not None else "")
              + f" {int(keep[c].sum()):7d}  {pos[keep[c]].sum():10.4g} {pos.sum():10.4g}")

    maps = moments(cube, vel, keep)
    out = args.out or str(Path(args.cube).with_suffix(""))
    comment = f"{Path(args.cube).name}, mask: {how}"[:300]
    for name, arr, bunit in zip(("mom0", "mom1", "mom2", "peak"), maps,
                                (f"{unit} km/s".strip(), "km/s", "km/s", unit)):
        fits.writeto(f"{out}_{name}.fits", arr.astype(np.float32),
                     plane_header(header, bunit, comment), overwrite=True)
        print(f"wrote {out}_{name}.fits")
    errors = None
    spath = (beside(args.cube, "posterior_samples.npz") if args.samples == "auto"
             else None if args.samples == "none" else Path(args.samples))
    if spath is not None:
        e0, e1, e2, n_draws, emode = moment_errors(spath, cube, vel, keep)
        errors = (e0, e1, e2)
        ecomment = (f"half the 16-84% range over {n_draws} posterior draws "
                    f"({emode}), mask fixed")
        for name, arr, bunit in (("mom0_err", e0, f"{unit} km/s".strip()),
                                 ("mom1_err", e1, "km/s"), ("mom2_err", e2, "km/s")):
            fits.writeto(f"{out}_{name}.fits", arr.astype(np.float32),
                         plane_header(header, bunit, ecomment), overwrite=True)
            print(f"wrote {out}_{name}.fits")
        with np.errstate(invalid="ignore", divide="ignore"):
            snr0 = np.where(e0 > 0, maps[0] / e0, np.nan)
        fits.writeto(f"{out}_mom0_snr.fits", snr0.astype(np.float32),
                     plane_header(header, "", ecomment), overwrite=True)
        print(f"wrote {out}_mom0_snr.fits ({n_draws} draws; median moment-0 S/N "
              f"{np.nanmedian(snr0):.1f}, median moment-1 error {np.nanmedian(e1):.1f} km/s)")
    mh = header.copy()
    mh["BUNIT"] = ""
    mh["COMMENT"] = comment
    fits.writeto(f"{out}_mask.fits", keep.astype(np.int16), mh, overwrite=True)
    print(f"wrote {out}_mask.fits")
    if not args.no_png:
        quicklook(f"{out}_moments.png", maps, header, unit,
                  f"{Path(args.cube).name}, mask: {how}", errors)
        print(f"wrote {out}_moments.png")


if __name__ == "__main__":
    main()

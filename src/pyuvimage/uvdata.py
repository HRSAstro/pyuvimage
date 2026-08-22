"""UVData: the on-disk and in-memory representation of calibrated visibility data.

A pyuvimage dataset is a directory containing:

    uvw.fits          (n_vis, 3)  float64   baseline coordinates [metres]
    frequencies.fits  (n_chan,)   float64   channel frequencies [Hz]
    data.fits         (n_chan, n_vis, 2)    Stokes I visibilities, re/im [Jy]
    noise.fits        (n_chan, n_vis, 2)    per-visibility RMS, re/im [Jy]
    flags.fits        (n_chan, n_vis)  uint8   1 = flagged (optional; absent = none)
    meta.json         phase centre, dish diameter, provenance, ...

Storing uvw in metres (rather than wavelengths) keeps the file small and lets
each channel use its exact uv coordinates: u_lambda = u_m * nu / c.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

C_M_S = 299792458.0

_FILES = {
    "uvw": "uvw.fits",
    "frequencies": "frequencies.fits",
    "data": "data.fits",
    "noise": "noise.fits",
    "flags": "flags.fits",
}


@dataclass
class UVData:
    """Calibrated Stokes-I visibilities for a single field / spectral window."""

    uvw: np.ndarray  # (n_vis, 3) metres
    frequencies: np.ndarray  # (n_chan,) Hz
    data: np.ndarray  # (n_chan, n_vis) complex Jy
    noise: np.ndarray  # (n_chan, n_vis) complex (sigma_re + 1j sigma_im) Jy
    flags: np.ndarray | None = None  # (n_chan, n_vis) bool
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic
    @property
    def n_vis(self) -> int:
        return self.data.shape[1]

    @property
    def n_chan(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        """Unflagged (channel, row) samples -- the real size of the fit.

        Not `n_vis * n_chan`: flags remove samples, and for multi-spw data the
        product is meaningless because the array is ragged.
        """
        if self.flags is None:
            return int(self.n_chan * self.n_vis)
        return int(np.count_nonzero(~self.flags))

    @property
    def spws(self) -> list["UVData"]:
        """A single-spw dataset, presented as a one-element list."""
        return [self]

    @property
    def n_spw(self) -> int:
        return 1

    @property
    def central_frequency(self) -> float:
        return float(np.mean(self.frequencies))

    @property
    def fractional_bandwidth(self) -> float:
        """(nu_max - nu_min) / nu_centre. Zero for a single channel."""
        f = np.asarray(self.frequencies, dtype=float)
        lo, hi = float(np.min(f)), float(np.max(f))
        mid = 0.5 * (lo + hi)
        return (hi - lo) / mid if mid > 0 else 0.0

    def uv_wavelengths(self, channel: int) -> np.ndarray:
        """(n_vis, 2) u,v in wavelengths for one channel."""
        scale = self.frequencies[channel] / C_M_S
        return self.uvw[:, :2] * scale

    @property
    def max_baseline_wavelengths(self) -> float:
        uv = self.uvw[:, :2] * (np.max(self.frequencies) / C_M_S)
        return float(np.max(np.hypot(uv[:, 0], uv[:, 1])))

    def baseline_percentile_wavelengths(self, percentile: float = 95.0) -> float:
        """Baseline length in wavelengths below which `percentile`% of samples lie.

        Real arrays have a long, sparse tail: on the first ALMA dataset this ran
        on, the median baseline was 213 klambda and the maximum 1054 klambda, so
        the naturally weighted beam was 0.54" while `0.5 / b_max` implied 0.097".
        Sizing the mesh off the maximum then gives it ~30x more pixels than the
        data constrain -- slow, and prior-dominated where it is not constrained.
        Flagged samples are excluded, since they contribute no information.
        """
        scale = np.max(self.frequencies) / C_M_S
        lengths = np.hypot(self.uvw[:, 0], self.uvw[:, 1]) * scale
        if self.flags is not None:
            keep = ~np.all(self.flags, axis=0)
            if keep.any():
                lengths = lengths[keep]
        return float(np.percentile(lengths, percentile))

    def validate(self) -> None:
        n_chan, n_vis = self.data.shape
        if self.uvw.shape != (n_vis, 3):
            raise ValueError(f"uvw shape {self.uvw.shape} != ({n_vis}, 3)")
        if self.frequencies.shape != (n_chan,):
            raise ValueError(
                f"frequencies shape {self.frequencies.shape} != ({n_chan},)"
            )
        if self.noise.shape != self.data.shape:
            raise ValueError("noise shape must match data shape")
        if self.flags is not None and self.flags.shape != self.data.shape:
            raise ValueError("flags shape must match data shape")
        if not np.all(np.isfinite(self.uvw)):
            raise ValueError("non-finite uvw coordinates")
        sig = np.abs(self.noise.real) + np.abs(self.noise.imag)
        if np.any(~np.isfinite(sig)) or np.any(sig <= 0):
            good = self.flags is None or not np.all(
                self.flags[~np.isfinite(sig) | (sig <= 0)]
            )
            if good:
                raise ValueError(
                    "noise map contains non-positive or non-finite values on "
                    "unflagged visibilities"
                )

    # ------------------------------------------------------------- selection
    def select(self, channel: int | None = None) -> "UVData":
        """Return a copy restricted to a single channel (still 3D with n_chan=1)."""
        if channel is None:
            return self
        sl = slice(channel, channel + 1)
        return UVData(
            uvw=self.uvw,
            frequencies=self.frequencies[sl],
            data=self.data[sl],
            noise=self.noise[sl],
            flags=None if self.flags is None else self.flags[sl],
            meta=dict(self.meta),
        )

    def flattened(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All unflagged (channel, vis) samples flattened for a joint (MFS) fit.

        Every channel keeps its own uv coordinates in wavelengths -- no
        channel averaging, so there is no bandwidth-smearing approximation.

        Returns (uv_wavelengths (n, 2), data (n,) complex, noise (n,) complex).
        """
        uv_list, d_list, n_list = [], [], []
        for c in range(self.n_chan):
            keep = (
                np.ones(self.n_vis, dtype=bool)
                if self.flags is None
                else ~self.flags[c]
            )
            uv_list.append(self.uv_wavelengths(c)[keep])
            d_list.append(self.data[c][keep])
            n_list.append(self.noise[c][keep])
        return (
            np.concatenate(uv_list, axis=0),
            np.concatenate(d_list, axis=0),
            np.concatenate(n_list, axis=0),
        )

    # ------------------------------------------------------------------- I/O
    def write(self, path: str | Path, overwrite: bool = False) -> Path:
        self.validate()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        def _write(name: str, array: np.ndarray) -> None:
            fits.writeto(path / _FILES[name], array, overwrite=overwrite)

        _write("uvw", np.asarray(self.uvw, dtype=np.float64))
        _write("frequencies", np.asarray(self.frequencies, dtype=np.float64))
        _write("data", _complex_to_reim(self.data))
        _write("noise", _complex_to_reim(self.noise))
        if self.flags is not None and np.any(self.flags):
            _write("flags", self.flags.astype(np.uint8))
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return path

    @classmethod
    def read(cls, path: str | Path) -> "UVData":
        path = Path(path)
        if path.is_file() and path.suffix == ".npz":
            return cls._read_npz(path)
        if not (path / _FILES["data"]).exists():
            legacy = _find_legacy(path)
            if legacy is not None:
                return legacy
            raise FileNotFoundError(
                f"{path} is not a pyuvimage dataset (no data.fits) and does not "
                "match the legacy pyuvimage_dev export layout."
            )
        uvw = _getdata(path / _FILES["uvw"])
        frequencies = np.atleast_1d(_getdata(path / _FILES["frequencies"]))
        data = _reim_to_complex(_getdata(path / _FILES["data"]))
        noise = _reim_to_complex(_getdata(path / _FILES["noise"]))
        flags_path = path / _FILES["flags"]
        flags = _getdata(flags_path).astype(bool) if flags_path.exists() else None
        meta_path = path / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        obj = cls(
            uvw=uvw, frequencies=frequencies, data=data, noise=noise,
            flags=flags, meta=meta,
        )
        obj.validate()
        return obj

    @classmethod
    def _read_npz(cls, path: Path) -> "UVData":
        """Read the single-file export produced by casa_export.py."""
        z = np.load(path, allow_pickle=False)
        if "n_spw" in z:
            raise ValueError(
                f"{path} holds several spectral windows; read it with "
                "pyuvimage.uvdata.read_dataset()"
            )
        flags = z["flags"].astype(bool) if "flags" in z else None
        if flags is not None and not np.any(flags):
            flags = None
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        obj = cls(
            uvw=np.asarray(z["uvw"], dtype=float),
            frequencies=np.atleast_1d(np.asarray(z["frequencies"], dtype=float)),
            data=np.asarray(z["data_re"]) + 1j * np.asarray(z["data_im"]),
            noise=np.asarray(z["noise_re"]) + 1j * np.asarray(z["noise_im"]),
            flags=flags,
            meta=meta,
        )
        obj.validate()
        return obj

    # ------------------------------------------------- legacy prototype data
    @classmethod
    def from_legacy(
        cls,
        visibilities: np.ndarray,
        sigma: np.ndarray,
        uv_wavelengths: np.ndarray,
        frequencies: np.ndarray,
        meta: dict | None = None,
    ) -> "UVData":
        """Build from the pyuvimage_dev export arrays.

        Legacy shapes: visibilities/sigma (n_corr, n_chan, n_vis, 2) with the
        last axis re/im; uv_wavelengths (n_chan, n_vis, 2); frequencies
        (n_chan,).  Parallel-hand correlations are averaged to Stokes I.
        """
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        vis = np.asarray(visibilities, dtype=float)
        sig = np.asarray(sigma, dtype=float)
        uvw_l = np.asarray(uv_wavelengths, dtype=float)
        if vis.ndim == 3:  # single correlation squeezed out
            vis = vis[None]
            sig = sig[None]
        if uvw_l.ndim == 2:
            uvw_l = uvw_l[None]
        n_corr, n_chan, n_vis, _ = vis.shape

        vis_c = vis[..., 0] + 1j * vis[..., 1]  # (n_corr, n_chan, n_vis)
        sig_c = sig[..., 0] + 1j * sig[..., 1]

        # Stokes I = mean of parallel hands.  Averaging only beats down the
        # noise if the hands are independent: some exports (and mocks)
        # duplicate a single correlation, in which case averaging changes
        # nothing and dividing sigma by sqrt(n_corr) would underestimate the
        # noise -- which drives the fit to overfit.  Count distinct hands.
        n_independent = _count_independent_hands(vis_c)
        if n_independent < n_corr:
            import warnings

            warnings.warn(
                f"{n_corr} correlations supplied but only {n_independent} are "
                "distinct (duplicated hands): treating them as "
                f"{n_independent} independent measurement(s) when scaling the "
                "noise.",
                stacklevel=2,
            )
        data = vis_c.mean(axis=0)
        noise_re = np.sqrt(np.sum(sig_c.real**2, axis=0)) / n_corr
        noise_im = np.sqrt(np.sum(sig_c.imag**2, axis=0)) / n_corr
        # correct back for the hands that carry no independent information
        boost = np.sqrt(n_corr / n_independent)
        noise = (noise_re + 1j * noise_im) * boost

        # Recover uvw in metres from channel-0 wavelengths.
        scale = C_M_S / frequencies[0]
        uvw = np.zeros((n_vis, 3))
        uvw[:, :2] = uvw_l[0] * scale
        return cls(
            uvw=uvw, frequencies=frequencies, data=data, noise=noise,
            meta=dict(meta or {}),
        )


# ---------------------------------------------------------------- helpers
def _count_independent_hands(vis_c: np.ndarray) -> int:
    """Number of distinct correlations in a (n_corr, ...) visibility array.

    Duplicated hands carry no extra information, so they must not be counted
    when scaling the Stokes I noise.
    """
    n_corr = vis_c.shape[0]
    distinct = []
    for i in range(n_corr):
        if not any(np.allclose(vis_c[i], vis_c[j]) for j in distinct):
            distinct.append(i)
    return max(1, len(distinct))


def _complex_to_reim(a: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape + (2,), dtype=np.float64)
    out[..., 0] = a.real
    out[..., 1] = a.imag
    return out


def _reim_to_complex(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a[..., 0]) + 1j * np.ascontiguousarray(a[..., 1])


def _getdata(path: Path) -> np.ndarray:
    return np.asarray(fits.getdata(path), dtype=np.float64)


def _find_legacy(path: Path) -> UVData | None:
    """Detect a pyuvimage_dev-style export directory and load it."""
    vis_files = sorted(path.glob("visibilities_*.fits"))
    if not vis_files:
        return None
    vis_path = vis_files[0]
    stem = vis_path.name[len("visibilities_"):]

    def _sibling(prefix: str) -> Path | None:
        cands = [path / f"{prefix}_{stem}"] + sorted(path.glob(f"{prefix}_*.fits"))
        for c in cands:
            if c.exists():
                return c
        return None

    sigma_path = _sibling("sigma_statwt") or _sibling("sigma")
    uv_path = _sibling("uv_wavelengths")
    freq_path = _sibling("frequencies")
    if sigma_path is None or uv_path is None or freq_path is None:
        return None
    meta = {"legacy_source": str(path)}
    meta_json = path / "mock_meta.json"
    if meta_json.exists():
        meta["mock_meta"] = json.loads(meta_json.read_text())
    return UVData.from_legacy(
        visibilities=fits.getdata(vis_path),
        sigma=fits.getdata(sigma_path),
        uv_wavelengths=fits.getdata(uv_path),
        frequencies=fits.getdata(freq_path),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Several spectral windows, imaged together
# ---------------------------------------------------------------------------


@dataclass
class MultiSpwUVData:
    """Several spectral windows of the same field, imaged as one.

    Spectral windows are *ragged*: each has its own channel frequencies, and
    in a measurement set each also has its own rows, so channel counts and row
    counts both differ between them.  There is no rectangular array that holds
    them, which is why this is a list of `UVData` rather than another axis on
    one.

    That costs nothing for MFS.  The fit never sees a channel axis: it works
    from `flattened()`, a flat list of (u, v) in wavelengths with one data and
    one noise value each, and every sample already carries its own uv
    coordinates computed at its own frequency.  Concatenating across spectral
    windows is therefore exactly the same operation as concatenating across
    channels, which the single-spw path has always done.

    **MFS assumes the sky does not change across the combined band.** That is a
    mild assumption inside one spectral window and can be a strong one across
    several: over a wide fractional bandwidth a spectral index of a few tenths
    moves flux by more than the noise.  pyuvimage fits one frequency-independent
    image and has no Taylor-term expansion (CLEAN's `mtmfs`), so it warns when
    the fractional bandwidth is large and leaves the judgement to you.
    """

    spws: list[UVData]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.spws:
            raise ValueError("a multi-spw dataset needs at least one spw")
        if not self.meta:
            self.meta = dict(self.spws[0].meta)

    # ------------------------------------------------------------------ basic
    @property
    def n_spw(self) -> int:
        return len(self.spws)

    @property
    def n_vis(self) -> int:
        """Rows summed over spectral windows (they need not be equal)."""
        return int(sum(s.n_vis for s in self.spws))

    @property
    def n_chan(self) -> int:
        """Channels summed over spectral windows."""
        return int(sum(s.n_chan for s in self.spws))

    @property
    def n_samples(self) -> int:
        return int(sum(s.n_samples for s in self.spws))

    @property
    def frequencies(self) -> np.ndarray:
        """Every channel frequency, ascending, across all spectral windows."""
        return np.sort(np.concatenate([s.frequencies for s in self.spws]))

    @property
    def central_frequency(self) -> float:
        """Sample-weighted mean frequency.

        Weighted, not the midpoint of the span: an spw with more unflagged
        samples contributes more sensitivity, and this is the frequency the
        primary beam is evaluated at.
        """
        total = num = 0.0
        for s in self.spws:
            w = float(s.n_samples) / max(s.n_chan, 1)
            num += w * float(np.sum(s.frequencies))
            total += w * s.n_chan
        return float(num / total) if total else float(np.mean(self.frequencies))

    @property
    def fractional_bandwidth(self) -> float:
        """(nu_max - nu_min) / nu_centre over the combined band."""
        f = self.frequencies
        lo, hi = float(np.min(f)), float(np.max(f))
        mid = 0.5 * (lo + hi)
        return (hi - lo) / mid if mid > 0 else 0.0

    @property
    def max_baseline_wavelengths(self) -> float:
        return float(max(s.max_baseline_wavelengths for s in self.spws))

    def baseline_percentile_wavelengths(self, percentile: float = 95.0) -> float:
        """As `UVData.baseline_percentile_wavelengths`, pooled over all spws."""
        scale = max(np.max(s.frequencies) for s in self.spws) / C_M_S
        parts = []
        for s in self.spws:
            lengths = np.hypot(s.uvw[:, 0], s.uvw[:, 1]) * scale
            if s.flags is not None:
                keep = ~np.all(s.flags, axis=0)
                if keep.any():
                    lengths = lengths[keep]
            parts.append(lengths)
        return float(np.percentile(np.concatenate(parts), percentile))

    @property
    def noise(self) -> np.ndarray:
        """All noise values, flattened -- for summary statistics only."""
        return np.concatenate([np.asarray(s.noise).ravel() for s in self.spws])

    def validate(self) -> None:
        for i, s in enumerate(self.spws):
            try:
                s.validate()
            except ValueError as e:
                raise ValueError(f"spw {i}: {e}") from e

    # ------------------------------------------------------------- selection
    def _channel_index(self) -> list[tuple[int, int]]:
        """Global channel order -> (spw, channel within it), by frequency.

        Cube mode fits channels independently, so a global channel axis across
        spectral windows is well defined even when they are disjoint in
        frequency -- it just has to be sorted, or the output cube's frequency
        axis would not be monotonic.
        """
        pairs = [
            (float(f), i, c)
            for i, s in enumerate(self.spws)
            for c, f in enumerate(s.frequencies)
        ]
        pairs.sort()
        return [(i, c) for _, i, c in pairs]

    def select(self, channel: int | None = None) -> "UVData | MultiSpwUVData":
        if channel is None:
            return self
        spw_i, chan_i = self._channel_index()[channel]
        return self.spws[spw_i].select(channel=chan_i)

    def flattened(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every unflagged sample from every spw, for one joint (MFS) fit."""
        parts = [s.flattened() for s in self.spws]
        return (
            np.concatenate([p[0] for p in parts], axis=0),
            np.concatenate([p[1] for p in parts], axis=0),
            np.concatenate([p[2] for p in parts], axis=0),
        )

    # ------------------------------------------------------------------- I/O
    def write(self, path: str | Path, overwrite: bool = False) -> Path:
        """Write as `spw000/`, `spw001/`, ... plus a top-level meta.json."""
        self.validate()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(self.spws):
            s.write(path / f"spw{i:03d}", overwrite=overwrite)
        meta = dict(self.meta)
        meta["n_spw"] = self.n_spw
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def read(cls, path: str | Path) -> "MultiSpwUVData":
        path = Path(path)
        dirs = sorted(d for d in path.glob("spw*") if d.is_dir())
        if not dirs:
            raise FileNotFoundError(f"{path} holds no spw* subdirectories")
        spws = [UVData.read(d) for d in dirs]
        meta_path = path / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        obj = cls(spws=spws, meta=meta)
        obj.validate()
        return obj


def _read_multi_npz(path: Path) -> "MultiSpwUVData":
    """Read a multi-spw .npz written by casa_export.py.

    Windows are stored side by side under `spw000_*`, `spw001_*` keys rather
    than stacked, because they are ragged: different channel counts and
    different row counts.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    per_spw = meta.get("per_spw_meta") or []
    spws = []
    for i in range(int(z["n_spw"])):
        pre = f"spw{i:03d}_"
        flags = z[pre + "flags"].astype(bool) if pre + "flags" in z else None
        if flags is not None and not np.any(flags):
            flags = None
        spws.append(UVData(
            uvw=np.asarray(z[pre + "uvw"], dtype=float),
            frequencies=np.atleast_1d(
                np.asarray(z[pre + "frequencies"], dtype=float)),
            data=z[pre + "data_re"] + 1j * z[pre + "data_im"],
            noise=z[pre + "noise_re"] + 1j * z[pre + "noise_im"],
            flags=flags,
            meta=dict(per_spw[i]) if i < len(per_spw) else dict(meta),
        ))
    obj = MultiSpwUVData(spws=spws, meta=meta)
    obj.validate()
    return obj


def read_dataset(path: str | Path) -> "UVData | MultiSpwUVData":
    """Read any dataset layout: one spectral window, or several.

    A multi-spw dataset is a directory of `spw000/`, `spw001/`, ... or a single
    `.npz` carrying an `n_spw` key; a single-spw one has `data.fits` at the top
    level, or is a plain `.npz`.  Datasets written before multi-spw support
    keep working untouched.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            multi = "n_spw" in z
        return _read_multi_npz(path) if multi else UVData.read(path)
    if path.is_dir() and not (path / _FILES["data"]).exists():
        if any(d.is_dir() for d in path.glob("spw*")):
            return MultiSpwUVData.read(path)
    return UVData.read(path)

"""Stream the visibilities once; hold only what the fit actually uses.

The sparse (w-tilde) inversion never looks at a visibility after F and D
exist: F = M^T W~ M comes from the kernel, D from one adjoint transform of
the data, and chi^2 = s^T F s - 2 s^T D + d^T N^-1 d needs one more scalar.
Every quantity the search, the uncertainty map and even the residual dirty
image consume is image- or mesh-sized. Yet the load path -- `UVData` reading
the whole file, `flattened()` copying it, `make_dataset` handing all of it to
autoarray -- held ~136 bytes per visibility resident for the whole run,
which on a 202-million visibility MFS cube was 27 GB on a model of 324
pixels (`claude/sparse-path-visibility-cost.md`).

This module is the alternative: read the file in channel blocks, fold each
block into the terms below, discard it. Nothing here scales with the number
of visibilities except time.

    W~                sum_k w_k cos(dx ku_k + dy kv_k)    kernel, (2Ny, 2Nx)
    dirty image       adjoint of d / sigma^2              D = M^T . this
    dirty beam        adjoint of w                        products
    sum(w)            scalar                              beam normalisation
    d^T N^-1 d        scalar                              chi^2
    sum log 2 pi s^2  scalar                              evidence
    row |uv| lengths  one per row, not per sample        mesh scale (b_95, b_max)

All of them are sums over visibilities, so the chunking is exact up to
floating-point summation order.

The terms depend on the data, so their cache entry is keyed on the file's
identity (path, size, mtime) and the geometry rather than on a hash of the
arrays -- hashing 202M values *is* a pass over the data, which is the thing
a cached re-fit exists to skip. This is the same trade CASA's `cfcache`
makes.

The streamed terms feed a `LinearSystem` directly (`fitting.py`); nothing
here needs JAX, and the kernel accumulation is autoarray's own NumPy backend
called once per chunk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .uvdata import C_M_S, V_SIGN, UVData, MultiSpwUVData, pooled_noise

logger = logging.getLogger("pyuvimage")

#: Visibilities per chunk. A chunk's cost is the per-chunk transformer (a DFT
#: is n_image x chunk_k complex, 85 MB at 1296 pixels) and the kernel
#: accumulation's `chunk_k x n_image` phase array; both are freed before the
#: next chunk.
STREAM_CHUNK_K = 4096

@dataclass
class VisibilityChunk:
    """One block of unflagged samples: uv in wavelengths, data and sigma.

    `row_lengths` rides on the first chunk of each spectral window: every row's
    baseline length at that window's maximum frequency, which is what
    `UVData.max_baseline_wavelengths` and `baseline_percentile_wavelengths`
    are defined on (per *row*, not per sample, and the maximum over all rows
    flagged or not). `row_flag_all` is, per chunk, which rows were flagged in
    every channel of the block; the accumulator ANDs it across a window so the
    percentile can exclude rows flagged everywhere, exactly as the in-memory
    definition does. Both are n_rows long -- kilobytes.
    """

    uv: np.ndarray        # (n, 2) float64, v already carries V_SIGN
    data: np.ndarray      # (n,) complex128
    noise: np.ndarray     # (n,) complex128, sigma_re + 1j sigma_im
    spw: int = 0
    row_lengths: np.ndarray | None = None
    row_flag_all: np.ndarray | None = None
    rows: tuple[int, int] | None = None     # the row slice this chunk covers
    #: the channel within `spw` when the block holds exactly one channel
    #: (`per_channel=True` readers); None for a block spanning several
    channel: int | None = None

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def thinned(self, keep: np.ndarray) -> "VisibilityChunk":
        """The same chunk on a subset of its samples; row bookkeeping intact."""
        return VisibilityChunk(
            self.uv[keep], self.data[keep], self.noise[keep], spw=self.spw,
            row_lengths=self.row_lengths, row_flag_all=self.row_flag_all,
            rows=self.rows, channel=self.channel,
        )


# --------------------------------------------------------------------------
# Sources: in-memory UVData, .npz from casa_export, FITS directory
# --------------------------------------------------------------------------

def _block_to_chunk(
    uvw: np.ndarray, freqs: np.ndarray, data: np.ndarray, noise: np.ndarray,
    flags: np.ndarray | None, rows: tuple[int, int], spw: int,
    channel: int | None = None,
) -> VisibilityChunk:
    """The unflagged samples of a (n_block_chan, r1 - r0) block, channel-major.

    The same arithmetic as `UVData.flattened()` -- the (n_rows, 2) metres
    broadcast against the (n_chan,) scale, v negated -- so the samples come
    out bit-identical to the in-memory path. Order: within a channel block,
    channel-major; blocks are yielded in channel order, and row sub-blocks
    of one channel in row order, so the concatenation of every chunk is the
    `flattened()` array exactly as long as a block never spans more than one
    channel when it is split by rows (`_blocks` guarantees that).
    """
    r0, r1 = rows
    scale = np.asarray(freqs, dtype=float) / C_M_S
    uv = uvw[None, r0:r1, :2] * scale[:, None, None]
    uv[..., 1] *= V_SIGN
    uv = uv.reshape(-1, 2)
    d = np.asarray(data, dtype=complex)[:, r0:r1].reshape(-1)
    n = np.asarray(noise, dtype=complex)[:, r0:r1].reshape(-1)
    flag_all = None
    if flags is not None:
        f = np.asarray(flags, dtype=bool)[:, r0:r1]
        flag_all = np.all(f, axis=0)
        keep = np.flatnonzero(~f.reshape(-1))
        uv, d, n = np.take(uv, keep, axis=0), np.take(d, keep), np.take(n, keep)
    return VisibilityChunk(uv, d, n, spw=spw, row_flag_all=flag_all, rows=(r0, r1),
                           channel=channel)


def _blocks(n_chan: int, n_rows: int, chunk_k: int, per_channel: bool = False,
            ) -> Iterator[tuple[int, int, int, int]]:
    """(c0, c1, r0, r1) blocks holding about `chunk_k` samples each.

    Whole channels when a channel fits (several per block if they are
    narrow); one channel at a time in row slices when it does not -- a
    single ALMA channel can be 50k rows, and a DFT over that many at 1296
    pixels is a gigabyte. Splitting rows within a channel keeps the order of
    `flattened()`: channel-major, rows ascending. `per_channel` never puts
    two channels in one block, so cube mode can accumulate one channel's
    terms at a time; narrow channels then cost one small block each.
    """
    n_rows = max(n_rows, 1)
    if n_rows <= chunk_k:
        per_block = 1 if per_channel else max(1, chunk_k // n_rows)
        for c0 in range(0, n_chan, per_block):
            yield c0, min(n_chan, c0 + per_block), 0, n_rows
        return
    for c in range(n_chan):
        for r0 in range(0, n_rows, chunk_k):
            yield c, c + 1, r0, min(n_rows, r0 + chunk_k)


def _row_lengths(uvw: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Every row's |uv| at the window's maximum frequency -- `UVData`'s definition."""
    scale = float(np.max(freqs)) / C_M_S
    return np.hypot(uvw[:, 0], uvw[:, 1]) * scale


def _with_row_lengths(chunks: Iterator[VisibilityChunk], uvw, freqs) -> Iterator[VisibilityChunk]:
    """Attach the per-row baseline lengths to a window's first chunk."""
    first = True
    for ch in chunks:
        if first:
            ch.row_lengths = _row_lengths(uvw, freqs)
            first = False
        yield ch


def iter_uvdata_chunks(uvd: UVData | MultiSpwUVData, chunk_k: int = STREAM_CHUNK_K,
                       per_channel: bool = False) -> Iterator[VisibilityChunk]:
    """Chunks of an in-memory dataset, in `flattened()` order.

    Exists so the streamed accumulation can be checked against the in-memory
    fit on the same object, and so callers holding a small dataset can use one
    code path. It does not save memory -- the dataset is already resident.
    """
    for i, spw in enumerate(uvd.spws):
        n_chan, n_rows = spw.data.shape
        uvw = np.asarray(spw.uvw, dtype=float)

        def gen():
            for c0, c1, r0, r1 in _blocks(n_chan, n_rows, chunk_k, per_channel):
                yield _block_to_chunk(
                    uvw, spw.frequencies[c0:c1], spw.data[c0:c1], spw.noise[c0:c1],
                    None if spw.flags is None else spw.flags[c0:c1], (r0, r1), i,
                    channel=c0 if c1 == c0 + 1 else None,
                )
        yield from _with_row_lengths(gen(), uvw, spw.frequencies)


#: a Fortran-ordered member larger than this is held whole with a warning:
#: it cannot be streamed in channel blocks, since its file order is rows
#: outermost
FORTRAN_MEMBER_WARN_BYTES = 256 * 1024 ** 2


class _NpyMemberStream:
    """Sequential row-block reader for one .npy member inside a .npz.

    `np.load` on an .npz materialises each member whole -- the zip container
    defeats memory-mapping, and casa_export writes with `savez_compressed`, so
    the members are deflated. What a deflated stream does allow is sequential
    reading, and a C-order (n_chan, n_rows) array is exactly rows-of-channels
    in file order. So: parse the .npy header, then read `n_block x n_rows x
    itemsize` bytes at a time.

    A **Fortran-ordered** member is stored the other way round -- every
    channel of row 0, then row 1 -- so a channel block is not contiguous in
    the file and the member has to be read whole. casatools returns its
    columns column-major, and a `flags` array derived from them by reductions
    stayed that way in the first real exports (`casa_export` now writes C
    order, but the files already on disk do not change). Flags are a byte per
    sample, so holding them is cheap; a Fortran-ordered *data* member above
    `FORTRAN_MEMBER_WARN_BYTES` is still read, with a warning naming the fix.
    """

    def __init__(self, zf: zipfile.ZipFile, name: str):
        self._f = zf.open(name, "r")
        version = np.lib.format.read_magic(self._f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(self._f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(self._f)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self._row_bytes = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize
        self._rows_read = 0
        self._whole: np.ndarray | None = None
        if fortran and len(self.shape) >= 2:
            nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
            if nbytes > FORTRAN_MEMBER_WARN_BYTES:
                logger.warning(
                    "%s is Fortran-ordered, so it cannot be streamed in channel "
                    "blocks and is held whole (%.0f MB). Re-exporting the file "
                    "with the current casa_export writes it in C order.",
                    name, nbytes / 1e6,
                )
            raw = self._read_exactly(nbytes)
            # file order is the transposed C array; `.T` gives the declared
            # shape as an F-ordered view, no copy
            self._whole = np.frombuffer(raw, dtype=self.dtype).reshape(self.shape[::-1]).T

    def _read_exactly(self, want: int) -> bytearray:
        buf = bytearray(want)
        view = memoryview(buf)
        got = 0
        while got < want:                       # zip streams may return short reads
            k = self._f.readinto(view[got:])
            if not k:
                raise EOFError("unexpected end of npz member")
            got += k
        return buf

    def read_rows(self, n: int) -> np.ndarray:
        n = int(min(n, self.shape[0] - self._rows_read))
        r0 = self._rows_read
        self._rows_read += n
        if self._whole is not None:
            return self._whole[r0:r0 + n]
        buf = self._read_exactly(n * self._row_bytes)
        return np.frombuffer(buf, dtype=self.dtype).reshape((n,) + self.shape[1:])

    def read_all(self) -> np.ndarray:
        if not self.shape:                      # 0-d member, e.g. n_spw or meta
            buf = self._f.read(self.dtype.itemsize)
            return np.frombuffer(buf, dtype=self.dtype).reshape(())
        return self.read_rows(self.shape[0])

    def close(self) -> None:
        self._f.close()


def iter_npz_chunks(path: str | Path, chunk_k: int = STREAM_CHUNK_K,
                    per_channel: bool = False) -> Iterator[VisibilityChunk]:
    """Chunks of a casa_export .npz, single- or multi-spw, never whole.

    Per spectral window the five (n_chan, n_rows) members -- data_re, data_im,
    noise_re, noise_im and flags if present -- are opened as parallel streams
    and advanced together one channel block at a time. uvw and frequencies
    are read whole; they are per-row and per-channel, not per-sample, and
    together are under a megabyte on any dataset.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        def arr(name):
            s = _NpyMemberStream(zf, name + ".npy")
            try:
                return s.read_all()
            finally:
                s.close()

        n_spw = int(arr("n_spw")) if "n_spw.npy" in names else 1
        prefixes = [f"spw{i:03d}_" for i in range(n_spw)] if "n_spw.npy" in names else [""]
        for pre in prefixes:
            uvw = np.asarray(arr(pre + "uvw"), dtype=float)
            freqs = np.atleast_1d(np.asarray(arr(pre + "frequencies"), dtype=float))
            streams = {k: _NpyMemberStream(zf, f"{pre}{k}.npy")
                       for k in ("data_re", "data_im", "noise_re", "noise_im")}
            flags = (_NpyMemberStream(zf, f"{pre}flags.npy")
                     if f"{pre}flags.npy" in names else None)
            try:
                n_chan, n_rows = streams["data_re"].shape

                def gen(spw_index=len(prefixes) and prefixes.index(pre)):
                    held = None          # (c0, c1, data, noise, flags) read once
                    for c0, c1, r0, r1 in _blocks(n_chan, n_rows, chunk_k, per_channel):
                        if held is None or held[0] != c0:
                            k = c1 - c0
                            blk = {name: s.read_rows(k) for name, s in streams.items()}
                            fl = flags.read_rows(k) if flags is not None else None
                            held = (c0, c1, blk["data_re"] + 1j * blk["data_im"],
                                    blk["noise_re"] + 1j * blk["noise_im"], fl)
                        yield _block_to_chunk(uvw, freqs[c0:c1], held[2], held[3],
                                              held[4], (r0, r1), spw_index,
                                              channel=c0 if c1 == c0 + 1 else None)
                yield from _with_row_lengths(gen(), uvw, freqs)
            finally:
                for s in streams.values():
                    s.close()
                if flags is not None:
                    flags.close()


def iter_fits_dir_chunks(path: str | Path, chunk_k: int = STREAM_CHUNK_K,
                         per_channel: bool = False) -> Iterator[VisibilityChunk]:
    """Chunks of a `pyuvimage import` directory (FITS per array), memory-mapped.

    `data.fits` and `noise.fits` are (n_chan, n_rows, 2) float64 -- re/im as a
    trailing axis -- so a channel block is a contiguous slice of the mmap and
    nothing is read that is not used.
    """
    from astropy.io import fits

    from .uvdata import _FILES

    path = Path(path)
    spw_dirs = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("spw")) \
        if not (path / _FILES["data"]).exists() else [path]
    for d in spw_dirs:
        uvw = np.asarray(fits.getdata(d / _FILES["uvw"]), dtype=float)
        freqs = np.atleast_1d(np.asarray(fits.getdata(d / _FILES["frequencies"]), dtype=float))
        with fits.open(d / _FILES["data"], memmap=True) as hd, \
             fits.open(d / _FILES["noise"], memmap=True) as hn:
            data_mm, noise_mm = hd[0].data, hn[0].data
            flags_path = d / _FILES["flags"]
            hf = fits.open(flags_path, memmap=True) if flags_path.exists() else None
            try:
                n_chan, n_rows = data_mm.shape[:2]

                def gen(spw_index=spw_dirs.index(d)):
                    for c0, c1, r0, r1 in _blocks(n_chan, n_rows, chunk_k, per_channel):
                        dblk = np.asarray(data_mm[c0:c1, r0:r1])
                        nblk = np.asarray(noise_mm[c0:c1, r0:r1])
                        fl = (np.asarray(hf[0].data[c0:c1, r0:r1]).astype(bool)
                              if hf is not None else None)
                        # the block is already row-sliced, so pass the local rows
                        ch = _block_to_chunk(
                            uvw[r0:r1], freqs[c0:c1],
                            dblk[..., 0] + 1j * dblk[..., 1],
                            nblk[..., 0] + 1j * nblk[..., 1], fl,
                            (0, r1 - r0), spw_index,
                            channel=c0 if c1 == c0 + 1 else None,
                        )
                        ch.rows = (r0, r1)
                        yield ch
                yield from _with_row_lengths(gen(), uvw, freqs)
            finally:
                if hf is not None:
                    hf.close()


def iter_chunks(source, chunk_k: int = STREAM_CHUNK_K, per_channel: bool = False,
                ) -> Iterator[VisibilityChunk]:
    """Dispatch on what `source` is: a dataset object, an .npz, a directory.

    `per_channel=True` never lets a block span two channels, and stamps each
    chunk with its channel (`VisibilityChunk.channel`) -- what cube mode
    groups on.
    """
    if isinstance(source, (UVData, MultiSpwUVData)):
        return iter_uvdata_chunks(source, chunk_k, per_channel)
    p = Path(source)
    if p.is_file() and p.suffix == ".npz":
        return iter_npz_chunks(p, chunk_k, per_channel)
    if p.is_dir():
        return iter_fits_dir_chunks(p, chunk_k, per_channel)
    raise FileNotFoundError(f"{source} is neither a dataset object, an .npz nor a directory")


def recentred(chunks: Iterable[VisibilityChunk], centre_arcsec: tuple[float, float],
              ) -> Iterator[VisibilityChunk]:
    """The chunks with the reconstruction centre moved to ``(y0, x0)`` arcsec.

    `uvdata.shift_image_centre`, chunk by chunk: the phase ramp
    ``V' = V exp(+2 pi i (u x0 + v y0))`` with (u, v) in wavelengths -- the
    chunk's `uv` already carries V_SIGN and the channel's frequency, so the
    ramp is one complex exponential per sample and needs nothing from outside
    the chunk -- and the noise pooled in quadrature, because a phase rotation
    mixes the real and imaginary parts (see `pooled_noise`). A zero offset
    passes the chunks through untouched, noise included, exactly as the
    in-memory function does.
    """
    from .uvdata import ARCSEC_RAD

    y0, x0 = float(centre_arcsec[0]), float(centre_arcsec[1])
    if y0 == 0.0 and x0 == 0.0:
        yield from chunks
        return
    for ch in chunks:
        if len(ch):
            angle = (2.0 * np.pi * ARCSEC_RAD) * (ch.uv[:, 0] * x0 + ch.uv[:, 1] * y0)
            ch.data = ch.data * np.exp(1j * angle)
            ch.noise = pooled_noise(ch.noise)
        yield ch


# --------------------------------------------------------------------------
# What one pass over the data produces
# --------------------------------------------------------------------------

@dataclass
class SparseTerms:
    """Everything the sparse fit needs from the data, and nothing per-visibility.

    `kernel` is autoarray's w-tilde preload, (2Ny, 2Nx) in FFT-wrapped lag
    order (index 0 = zero lag), weights 1/sigma_re^2 -- the same array
    `psf_precision_operator_from` returns. `dirty_image` is the adjoint of
    d_re/sigma_re^2 + i d_im/sigma_im^2 on the mathematical adjoint's scale,
    which is what `apply_sparse_operator` stores and what D = M^T dirty needs.
    `beam_raw` and `data_dirty_raw` use the kernel's weight too, 1/sigma_re^2,
    so that every image on this path shares one weighting; with pooled noise
    that is also `DirtyImager`'s weight and the products are identical to the
    in-memory path's.
    """

    kernel: np.ndarray                 # (2Ny, 2Nx) float64
    dirty_image: np.ndarray            # (Ny, Nx) native, adjoint of d/sigma^2
    beam_raw: np.ndarray               # (Ny, Nx) native, adjoint of w (DirtyImager weights)
    data_dirty_raw: np.ndarray         # (Ny, Nx) native, adjoint of w d
    sum_weights: float                 # sum of DirtyImager weights
    data_term: float                   # d^T N^-1 d
    noise_normalization: float         # sum log(2 pi sigma_re^2) + sum log(2 pi sigma_im^2)
    n_vis: int
    row_lengths: np.ndarray            # every row of every window, |uv| at that window's f_max
    row_keep: np.ndarray               # rows not flagged in every channel (the percentile's population)
    reim_asymmetry_sum: float = 0.0    # sum |s_re - s_im| / mean(s), for the report
    shape_native: tuple[int, int] = (0, 0)
    pixel_scale: float = 0.0
    seconds: float = 0.0

    @property
    def max_baseline(self) -> float:
        """`UVData.max_baseline_wavelengths`: over all rows, flagged or not."""
        return float(np.max(self.row_lengths)) if self.row_lengths.size else 0.0

    def baseline_percentile(self, percentile: float) -> float:
        """`UVData.baseline_percentile_wavelengths`, on the same population."""
        lengths = self.row_lengths[self.row_keep] if self.row_keep.any() else self.row_lengths
        return float(np.percentile(lengths, percentile)) if lengths.size else 0.0

    @property
    def rms(self) -> float:
        """Image-plane rms of the naturally weighted dirty image, 1/sqrt(sum w)."""
        return float(1.0 / np.sqrt(self.sum_weights))

    @classmethod
    def combine(cls, parts: list["SparseTerms"]) -> "SparseTerms":
        """The terms of the union of the parts' samples: every field is a sum
        over visibilities, so the MFS terms of a cube are its channels' added
        up. Row bookkeeping is taken from the first part (they share it)."""
        if not parts:
            raise ValueError("nothing to combine")
        first = parts[0]
        return cls(
            kernel=sum(p.kernel for p in parts),
            dirty_image=sum(p.dirty_image for p in parts),
            beam_raw=sum(p.beam_raw for p in parts),
            data_dirty_raw=sum(p.data_dirty_raw for p in parts),
            sum_weights=float(sum(p.sum_weights for p in parts)),
            data_term=float(sum(p.data_term for p in parts)),
            noise_normalization=float(sum(p.noise_normalization for p in parts)),
            n_vis=int(sum(p.n_vis for p in parts)),
            row_lengths=first.row_lengths, row_keep=first.row_keep,
            reim_asymmetry_sum=float(sum(p.reim_asymmetry_sum for p in parts)),
            shape_native=first.shape_native, pixel_scale=first.pixel_scale,
            seconds=float(sum(p.seconds for p in parts)),
        )

    # -- persistence -------------------------------------------------------
    def save(self, path: Path, key: str) -> None:
        np.savez(
            path, kernel=self.kernel, dirty_image=self.dirty_image,
            beam_raw=self.beam_raw, data_dirty_raw=self.data_dirty_raw,
            scalars=np.array([self.sum_weights, self.data_term,
                              self.noise_normalization, self.n_vis,
                              self.reim_asymmetry_sum, self.seconds], dtype=float),
            row_lengths=self.row_lengths, row_keep=self.row_keep,
            shape_native=np.array(self.shape_native), pixel_scale=self.pixel_scale,
            key=np.array(key),
        )

    @classmethod
    def load(cls, path: Path) -> tuple["SparseTerms", str]:
        with np.load(path, allow_pickle=False) as z:
            s = z["scalars"]
            terms = cls(
                kernel=z["kernel"], dirty_image=z["dirty_image"],
                beam_raw=z["beam_raw"], data_dirty_raw=z["data_dirty_raw"],
                sum_weights=float(s[0]), data_term=float(s[1]),
                noise_normalization=float(s[2]), n_vis=int(s[3]),
                reim_asymmetry_sum=float(s[4]), seconds=float(s[5]),
                row_lengths=z["row_lengths"], row_keep=z["row_keep"].astype(bool),
                shape_native=tuple(int(v) for v in z["shape_native"]),
                pixel_scale=float(z["pixel_scale"]),
            )
            return terms, str(z["key"])


class RowTracker:
    """Per-row bookkeeping for the mesh scale, shared by every accumulator
    fed from one stream.

    `row_lengths` rides on each window's first chunk and `row_flag_all` on
    every chunk; the "flagged in every channel" exclusion is an AND across all
    of a window's blocks. Cube mode feeds one accumulator per channel from a
    single stream, and none of them sees every block -- so the tracker is
    shared and each `finish()` reads its state at that moment.
    """

    def __init__(self):
        self.row_lengths: dict[int, np.ndarray] = {}
        self.row_all_flagged: dict[int, np.ndarray] = {}

    def observe(self, ch: VisibilityChunk) -> None:
        if ch.row_lengths is not None:
            self.row_lengths[ch.spw] = np.asarray(ch.row_lengths, dtype=float)
            # "flagged in every channel" until some block shows otherwise
            self.row_all_flagged[ch.spw] = np.ones(len(ch.row_lengths), dtype=bool)
        if ch.rows is not None and ch.spw in self.row_all_flagged:
            r0, r1 = ch.rows
            acc = self.row_all_flagged[ch.spw]
            if ch.row_flag_all is None:          # no flags in this block at all
                acc[r0:r1] = False
            else:                                 # AND across blocks
                acc[r0:r1] &= np.asarray(ch.row_flag_all, dtype=bool)

    def lengths_and_keep(self) -> tuple[np.ndarray, np.ndarray]:
        spws = sorted(self.row_lengths)
        lengths = (np.concatenate([self.row_lengths[k] for k in spws])
                   if spws else np.zeros(0))
        keep = (np.concatenate([~self.row_all_flagged[k] for k in spws])
                if spws else np.zeros(0, dtype=bool))
        return lengths, keep


class TermsAccumulator:
    """Folds chunks into `SparseTerms`, one `add` at a time.

    Per chunk a transformer is built over that chunk's uv (the DFT here is
    n_image x chunk_k, freed with the chunk) and three adjoints are taken:
    the weighted data (for D), the DirtyImager-weighted data and the weights
    (for the products). The kernel accumulation is autoarray's own
    `nufft_precision_operator_from` on the chunk, which is a plain sum over
    visibilities and therefore additive across chunks -- and across
    accumulators: the MFS terms of a cube are the sum of its channels'
    (`SparseTerms.combine`).

    Three adjoints per chunk is three times the cost of the in-memory path's
    single pass; on a 202M-visibility dataset that is the price of never
    holding it. `use_jax=True` hands the kernel accumulation to autoarray's
    JAX backend where available.

    `pool_noise` replaces sigma_re and sigma_im by their quadrature mean per
    sample -- `uvdata.pooled_noise`, which is elementwise and so chunk-safe --
    *after* the re/im asymmetry has been measured on the originals. `api.run`
    pools when the asymmetry crosses `REIM_ASYMMETRY_WARN`; the streaming
    path makes that decision on the header's noise sample and passes it here,
    and the cache key carries it.
    """

    def __init__(self, geometry, mask, transformer_cls, *, use_jax: bool = False,
                 pool_noise: bool = False, rows: RowTracker | None = None):
        self.geometry = geometry
        self.mask = mask
        self.transformer_cls = transformer_cls
        self.use_jax = use_jax
        self.pool_noise = pool_noise
        self.rows = rows if rows is not None else RowTracker()
        self._own_rows = rows is None
        shape = self.shape = tuple(int(s) for s in geometry.shape_native)
        self.grid_radians = mask.derive_grid.all_false.in_radians.native.array
        self.kernel = np.zeros((2 * shape[0], 2 * shape[1]), dtype=np.float64)
        self.dirty = np.zeros(shape, dtype=np.float64)
        self.beam = np.zeros(shape, dtype=np.float64)
        self.data_dirty = np.zeros(shape, dtype=np.float64)
        self.sum_w = self.data_term = self.noise_norm = self.asym = 0.0
        self.n_vis = 0
        self.t0 = time.time()

    def add(self, ch: VisibilityChunk) -> None:
        from autoarray.inversion.inversion.interferometer import (
            inversion_interferometer_util as sparse_util,
        )
        from autoarray.structures.visibilities import Visibilities

        from .fitting import adjoint_image

        # per-row bookkeeping before the emptiness test: a fully flagged
        # block still says which rows are flagged
        if self._own_rows:
            self.rows.observe(ch)
        n = len(ch)
        if n == 0:
            return
        uv, d, s = ch.uv, ch.data, ch.noise
        self.asym += float(np.sum(np.abs(s.real - s.imag) / (0.5 * (s.real + s.imag))))
        if self.pool_noise:
            s = pooled_noise(s)
        sr, si = s.real, s.imag
        # -- the kernel, from autoarray's own accumulation on this chunk
        self.kernel += np.asarray(sparse_util.nufft_precision_operator_from(
            noise_map_real=sr, uv_wavelengths=uv,
            shape_masked_pixels_2d=self.shape, grid_radians_2d=self.grid_radians,
            chunk_k=max(n, 1), use_jax=self.use_jax,
        ))
        # -- the three adjoints, through the same transformer the fit would use
        transformer = self.transformer_cls(uv_wavelengths=uv, real_space_mask=self.mask)
        weighted = d.real * sr ** -2.0 + 1j * d.imag * si ** -2.0
        self.dirty += np.asarray(adjoint_image(transformer, Visibilities(weighted)).native)
        # The products use the kernel's own weight, 1/sigma_re^2, so that
        # dirty(data), dirty(model) = W~ * (M s) and the beam all share one
        # weighting and the residual map is a difference of like with like.
        # `DirtyImager` weights by 1/mean(var); with pooled noise -- which the
        # sparse path applies above `REIM_ASYMMETRY_WARN` -- the two are the
        # same array and the products are identical to the in-memory ones.
        w = sr ** -2.0
        self.beam += np.asarray(adjoint_image(transformer, Visibilities(w.astype(complex))).native)
        self.data_dirty += np.asarray(adjoint_image(transformer, Visibilities(d * w)).native)
        del transformer
        # -- the scalars
        self.sum_w += float(np.sum(w))
        self.data_term += float(np.sum(d.real ** 2 / sr ** 2) + np.sum(d.imag ** 2 / si ** 2))
        self.noise_norm += float(np.sum(np.log(2 * np.pi * sr ** 2))
                                 + np.sum(np.log(2 * np.pi * si ** 2)))
        self.n_vis += n

    def finish(self) -> SparseTerms:
        lengths, keep = self.rows.lengths_and_keep()
        return SparseTerms(
            kernel=self.kernel, dirty_image=self.dirty, beam_raw=self.beam,
            data_dirty_raw=self.data_dirty, sum_weights=self.sum_w,
            data_term=self.data_term, noise_normalization=self.noise_norm,
            n_vis=self.n_vis, row_lengths=lengths, row_keep=keep,
            reim_asymmetry_sum=self.asym, shape_native=self.shape,
            pixel_scale=float(self.geometry.pixel_scale),
            seconds=time.time() - self.t0,
        )


def accumulate_sparse_terms(
    chunks: Iterable[VisibilityChunk],
    geometry,
    mask,
    transformer_cls,
    *,
    use_jax: bool = False,
    pool_noise: bool = False,
    log_every: int = 50,
) -> SparseTerms:
    """One pass: fold every chunk into `SparseTerms` (see `TermsAccumulator`)."""
    acc = TermsAccumulator(geometry, mask, transformer_cls, use_jax=use_jax,
                           pool_noise=pool_noise)
    for i, ch in enumerate(chunks):
        acc.add(ch)
        if log_every and (i + 1) % log_every == 0:
            logger.info("  streamed %d visibilities in %.0f s", acc.n_vis, time.time() - acc.t0)
    return acc.finish()


def thinning_mask(n: int, thin: int, seed: int) -> np.ndarray:
    """Which of `n` samples a 1-in-`thin` thinning keeps; deterministic per seed.

    `api.run` thins the MFS data to one channel's worth before fitting the
    prior a cube's channels will share (`--cube-prior channel`): the prior is
    being chosen for the amount of data each channel fit has. In memory that
    is one `rng.choice` over the whole dataset; streamed, it is a Bernoulli
    draw per sample with the same expected fraction, seeded per chunk so a
    re-run and its cache agree.
    """
    if thin <= 1:
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    return rng.random(n) < 1.0 / float(thin)


def accumulate_channel_terms(
    chunks: Iterable[VisibilityChunk],
    geometry,
    mask,
    transformer_cls,
    *,
    use_jax: bool = False,
    pool_noise: bool = False,
    thin: int = 1,
    log_every: int = 50,
) -> tuple[dict[tuple[int, int], SparseTerms], SparseTerms]:
    """One pass, one `SparseTerms` per channel -- and the terms the shared
    prior is fitted on.

    `chunks` must come from a `per_channel=True` reader, so every chunk is
    stamped with its (spw, channel). Consecutive chunks of one channel are
    folded into that channel's accumulator; the channel's terms are finished
    when the stream moves on. Returns ``({(spw, channel): terms}, prior_terms)``
    where `prior_terms` is the MFS accumulation of a 1-in-`thin` thinning of
    every sample (`thin=1`: the plain sum of the channel terms, at no extra
    cost). Row bookkeeping is shared (`RowTracker`), so every channel's terms
    and the prior's carry the whole dataset's rows.
    """
    rows = RowTracker()
    per_channel: dict[tuple[int, int], SparseTerms] = {}
    current_key = None
    current: TermsAccumulator | None = None
    prior_acc = (
        TermsAccumulator(geometry, mask, transformer_cls, use_jax=use_jax,
                         pool_noise=pool_noise, rows=rows)
        if thin > 1 else None
    )
    t0 = time.time()
    n_seen = 0
    for i, ch in enumerate(chunks):
        if ch.channel is None:
            raise ValueError(
                "cube mode needs one channel per chunk: read the source with "
                "per_channel=True"
            )
        rows.observe(ch)
        key = (int(ch.spw), int(ch.channel))
        if key != current_key:
            if current is not None:
                per_channel[current_key] = current.finish()
            current_key = key
            current = TermsAccumulator(geometry, mask, transformer_cls, use_jax=use_jax,
                                       pool_noise=pool_noise, rows=rows)
        current.add(ch)
        if prior_acc is not None and len(ch):
            prior_acc.add(ch.thinned(thinning_mask(len(ch), thin, seed=i)))
        n_seen += len(ch)
        if log_every and (i + 1) % log_every == 0:
            logger.info("  streamed %d visibilities (%d channels done) in %.0f s",
                        n_seen, len(per_channel), time.time() - t0)
    if current is not None:
        per_channel[current_key] = current.finish()
    if prior_acc is not None:
        prior_terms = prior_acc.finish()
    else:
        prior_terms = SparseTerms.combine(list(per_channel.values()))
    # every channel's terms carry the whole dataset's row bookkeeping, which
    # is only complete once the stream has ended
    lengths, keep = rows.lengths_and_keep()
    for t in list(per_channel.values()) + [prior_terms]:
        t.row_lengths, t.row_keep = lengths, keep
    return per_channel, prior_terms


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def source_identity(source) -> str:
    """What names a cache entry: the file's identity, not its contents.

    A hash of the arrays would be a full pass over the data, which is what a
    cached re-fit exists to avoid. Path + size + mtime is the same trade
    CASA's `cfcache` makes; an in-memory dataset is hashed properly, since it
    is already resident.
    """
    if isinstance(source, (UVData, MultiSpwUVData)):
        h = hashlib.sha1()
        for spw in source.spws:
            for a in (spw.uvw, spw.frequencies, spw.data, spw.noise):
                h.update(np.ascontiguousarray(a).view(np.uint8))
            if spw.flags is not None:
                h.update(np.ascontiguousarray(spw.flags).view(np.uint8))
        return "mem:" + h.hexdigest()[:16]
    p = Path(source).resolve()
    if p.is_dir():
        parts = []
        for f in sorted(p.rglob("*.fits")):
            st = f.stat()
            parts.append(f"{f.relative_to(p)}:{st.st_size}:{int(st.st_mtime)}")
        return "dir:" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    st = p.stat()
    return f"file:{p}:{st.st_size}:{int(st.st_mtime)}"


def terms_key(source, geometry, mask_shape: str = "square", pool_noise: bool = False,
              centre: tuple[float, float] | None = None,
              channel: tuple[int, int] | None = None, thin: int = 1) -> str:
    """What names a `SparseTerms` cache entry.

    The source's identity, the geometry, the mask, whether the noise was
    pooled -- and, when they apply, the reconstruction centre (a recentred
    stream is different data), the (spw, channel) of a cube plane, and the
    thinning of a cube's prior terms. Any of these unset hashes exactly as it
    did before they existed, so old cache files stay valid.
    """
    ident = source_identity(source)
    geo = {
        "shape": [int(s) for s in geometry.shape_native],
        "pixel_scale": float(geometry.pixel_scale),
        "mask": mask_shape,
        "pooled": bool(pool_noise),
    }
    if centre is not None and (float(centre[0]) != 0.0 or float(centre[1]) != 0.0):
        geo["centre"] = [float(centre[0]), float(centre[1])]
    if channel is not None:
        geo["channel"] = [int(channel[0]), int(channel[1])]
    if thin > 1:
        geo["thin"] = int(thin)
    return hashlib.sha1((ident + "|" + json.dumps(geo, sort_keys=True)).encode()).hexdigest()[:16]


TERMS_SUFFIX = ".sparse_terms.npz"


def terms_cache_path(cache_dir, key: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"terms_{key}{TERMS_SUFFIX}"


def _load_cached(path, key: str, geometry) -> SparseTerms | None:
    """The cached terms at `path` if they are the ones `key` names, else None."""
    if path is None or not path.exists():
        return None
    try:
        terms, stored = SparseTerms.load(path)
    except Exception as e:  # a corrupt cache must never stop a fit
        logger.warning("could not read %s (%s); streaming again", path, e)
        return None
    if stored == key and terms.shape_native == tuple(int(v) for v in geometry.shape_native):
        return terms
    return None


def _save_cached(terms: SparseTerms, path, key: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        terms.save(path, key)
    except Exception as e:
        logger.warning("could not cache the terms at %s (%s)", path, e)


def sparse_terms_for(
    source,
    geometry,
    mask,
    transformer_cls,
    *,
    cache_dir=None,
    chunk_k: int = STREAM_CHUNK_K,
    use_jax: bool = False,
    mask_shape: str = "square",
    pool_noise: bool = False,
    reuse_cache: bool = True,
    centre: tuple[float, float] | None = None,
) -> SparseTerms:
    """The terms for `source`, from the cache when it has them, else one pass.

    `reuse_cache=False` streams again regardless and overwrites the entry --
    the cache is keyed on path, size and mtime, which a file rewritten in
    place with the same size can defeat. `centre` (y0, x0 arcsec on the grid)
    recentres every chunk on the way in (`recentred`); it is part of the key.
    """
    key = terms_key(source, geometry, mask_shape, pool_noise, centre=centre)
    path = terms_cache_path(cache_dir, key)
    if reuse_cache:
        terms = _load_cached(path, key, geometry)
        if terms is not None:
            logger.info(
                "reusing the streamed terms %s: %d visibilities, no data "
                "read this run", path.name, terms.n_vis,
            )
            return terms
    logger.info(
        "streaming the visibilities once (%d per chunk) to accumulate the "
        "w-tilde kernel, the dirty images and the chi^2 constants -- the only "
        "step whose cost scales with the data, and nothing per-visibility is "
        "kept", chunk_k,
    )
    chunks = iter_chunks(source, chunk_k)
    if centre is not None:
        chunks = recentred(chunks, centre)
    terms = accumulate_sparse_terms(
        chunks, geometry, mask, transformer_cls, use_jax=use_jax, pool_noise=pool_noise,
    )
    logger.info(
        "  %d visibilities in %.1f s; kernel %s, %.2f MB", terms.n_vis,
        terms.seconds, terms.kernel.shape, terms.kernel.nbytes / 1e6,
    )
    if path is not None:
        _save_cached(terms, path, key)
        if path.exists():
            logger.info("  cached as %s; a re-fit of this file reads no visibilities",
                        path.name)
    return terms


def wide_field_image_for(
    source,
    fov_arcsec: float,
    n_pixels: int,
    *,
    chunk_k: int = STREAM_CHUNK_K,
    cache_dir=None,
    reuse_cache: bool = True,
) -> tuple[np.ndarray, float]:
    """A naturally weighted wide-field dirty image of the whole stream.

    What `--image-centre auto` looks at before choosing where to reconstruct.
    One pass over the data by direct summation (`beam.WideFieldAccumulator`),
    ``n_pixels^2`` operations per sample -- 9216 at the default 96 pixels, so
    minutes on a few hundred thousand visibilities and hours on a hundred
    million -- and cached beside the output under the source's identity, the
    field and the pixel count, since neither the geometry nor the prior enters
    it and a re-run should not pay for it twice.
    """
    from .beam import WideFieldAccumulator

    ident = source_identity(source)
    key = hashlib.sha1(
        f"{ident}|wide|{float(fov_arcsec):.6g}|{int(n_pixels)}".encode()
    ).hexdigest()[:16]
    path = None if cache_dir is None else Path(cache_dir) / f"wide_{key}.npz"
    if reuse_cache and path is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    logger.info("reusing the cached wide-field image %s", path.name)
                    return np.asarray(z["image"]), float(z["rms"])
        except Exception as e:
            logger.warning("could not read %s (%s); imaging again", path, e)
    logger.info(
        "imaging a %.3g\" field at %d pixels by direct summation over the whole "
        "stream to find the source (%d operations per visibility; cached for "
        "the next run)...", fov_arcsec, n_pixels, n_pixels * n_pixels,
    )
    acc = WideFieldAccumulator(fov_arcsec, n_pixels)
    t0 = time.time()
    n = 0
    for i, ch in enumerate(iter_chunks(source, chunk_k)):
        acc.add(ch.uv, ch.data, ch.noise)
        n += len(ch)
        if (i + 1) % 50 == 0:
            logger.info("  imaged %d visibilities in %.0f s", n, time.time() - t0)
    image, rms = acc.image()
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, image=image, rms=rms, key=np.array(key))
        except Exception as e:
            logger.warning("could not cache the wide-field image at %s (%s)", path, e)
    return image, rms


def channel_terms_for(
    source,
    geometry,
    mask,
    transformer_cls,
    channels: list[tuple[int, int]],
    *,
    cache_dir=None,
    chunk_k: int = STREAM_CHUNK_K,
    use_jax: bool = False,
    mask_shape: str = "square",
    pool_noise: bool = False,
    reuse_cache: bool = True,
    centre: tuple[float, float] | None = None,
    thin: int = 1,
) -> tuple[dict[tuple[int, int], SparseTerms], SparseTerms]:
    """Cube mode: one `SparseTerms` per (spw, channel), plus the prior's terms.

    Every channel's terms and the prior's are cached under their own keys.
    When all of them are cached nothing is read; when any is missing (or
    `reuse_cache=False`) the whole file is streamed once, per channel, and
    every entry is (re)written -- one pass produces them all, so there is no
    saving in streaming only the missing channels.
    """
    common = dict(mask_shape=mask_shape, pool_noise=pool_noise, centre=centre)
    keys = {c: terms_key(source, geometry, channel=c, **common) for c in channels}
    prior_key = terms_key(source, geometry, thin=thin, **common) if thin > 1 else None
    if reuse_cache and cache_dir is not None:
        found = {c: _load_cached(terms_cache_path(cache_dir, k), k, geometry)
                 for c, k in keys.items()}
        prior_terms = (
            _load_cached(terms_cache_path(cache_dir, prior_key), prior_key, geometry)
            if prior_key is not None else None
        )
        if all(t is not None for t in found.values()) and (
            prior_key is None or prior_terms is not None
        ):
            if prior_terms is None:
                prior_terms = SparseTerms.combine([found[c] for c in channels])
            logger.info(
                "reusing the streamed terms of all %d channels (%d visibilities), "
                "no data read this run", len(channels), prior_terms.n_vis,
            )
            return found, prior_terms
    logger.info(
        "streaming the visibilities once (%d per chunk), one channel at a time, "
        "to accumulate every channel's w-tilde kernel, dirty images and chi^2 "
        "constants%s -- nothing per-visibility is kept", chunk_k,
        f" and the shared prior's on 1 sample in {thin}" if thin > 1 else "",
    )
    chunks = iter_chunks(source, chunk_k, per_channel=True)
    if centre is not None:
        chunks = recentred(chunks, centre)
    per_channel, prior_terms = accumulate_channel_terms(
        chunks, geometry, mask, transformer_cls, use_jax=use_jax,
        pool_noise=pool_noise, thin=thin,
    )
    missing = [c for c in channels if c not in per_channel]
    if missing:
        raise ValueError(
            f"the stream produced no samples for channels {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}; the header and the data disagree"
        )
    logger.info(
        "  %d visibilities in %.1f s across %d channels; kernels %s, %.2f MB each",
        prior_terms.n_vis if thin == 1 else sum(t.n_vis for t in per_channel.values()),
        sum(t.seconds for t in per_channel.values()), len(per_channel),
        prior_terms.kernel.shape, prior_terms.kernel.nbytes / 1e6,
    )
    if cache_dir is not None:
        for c, k in keys.items():
            _save_cached(per_channel[c], terms_cache_path(cache_dir, k), k)
        if prior_key is not None:
            _save_cached(prior_terms, terms_cache_path(cache_dir, prior_key), prior_key)
        logger.info("  cached %d channel entries beside the output; a re-fit reads "
                    "no visibilities", len(keys))
    return per_channel, prior_terms


# --------------------------------------------------------------------------
# Applying the kernel: the model's dirty image without any visibilities
# --------------------------------------------------------------------------

def apply_kernel(kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
    """W~ applied to an image on the (Ny, Nx) grid: the adjoint of N^-1 A image.

    The dirty image of a model's visibilities *is* this convolution -- that is
    the w-tilde identity -- so the residual dirty image, dirty(data) minus
    dirty(model), needs no model visibilities at all. Same FFT construction as
    `InterferometerSparseOperator`: zero-pad to (2Ny, 2Nx), multiply spectra,
    read the leading quadrant.
    """
    ny, nx = image.shape
    pad = np.zeros((2 * ny, 2 * nx), dtype=np.float64)
    pad[:ny, :nx] = image
    out = np.fft.ifft2(np.fft.fft2(pad) * np.fft.fft2(kernel)).real
    return out[:ny, :nx]


class KernelDirtyImager:
    """`beam.DirtyImager`'s interface, from streamed terms instead of a dataset.

    Same normalisation (sum of the DirtyImager weights, so a 1 Jy point at the
    phase centre reads 1 Jy/beam and `rms` is exact), same beam. The one
    thing it cannot do is image *arbitrary* visibilities -- there are none --
    so `dirty_image` takes an image-plane model and uses the kernel.
    """

    def __init__(self, terms: SparseTerms, mask=None):
        self.terms = terms
        self.mask = mask
        self._norm = float(terms.sum_weights)
        if not np.isfinite(self._norm) or self._norm <= 0:
            raise RuntimeError("dirty beam has non-positive weight sum")
        peak = float(np.nanmax(terms.beam_raw))
        ratio = peak / self._norm
        if not (0.5 <= ratio <= 1.0 + 1e-6):
            raise RuntimeError(
                f"the streamed dirty beam's sampled peak is {ratio:.4g} x sum(w); "
                "it must lie in [0.5, 1] -- the per-chunk adjoint is not on the "
                "mathematical scale"
            )
        self._beam = terms.beam_raw / self._norm
        self.dataset = None

    @property
    def dirty_beam(self) -> np.ndarray:
        return self._beam

    @property
    def rms(self) -> float:
        return float(1.0 / np.sqrt(self._norm))

    @property
    def inside(self) -> np.ndarray:
        """Where the image plane is defined -- everywhere, for a square mask."""
        if self.mask is None:
            return np.ones(self.terms.shape_native, dtype=bool)
        native = self.mask.native if hasattr(self.mask, "native") else self.mask
        return ~np.asarray(native).astype(bool)

    def dirty_image_of_data(self) -> np.ndarray:
        """Naturally weighted dirty image of the data [Jy/beam]."""
        return self.terms.data_dirty_raw / self._norm

    def dirty_image_of_model(self, model_image: np.ndarray) -> np.ndarray:
        """Dirty image [Jy/beam] the model's visibilities would give.

        W~ applied to the model image, over the same 1/sigma_re^2 weighting
        as `dirty_image_of_data` and the beam, so the residual map is a
        difference of like with like. Exact to ~1e-15 against imaging the
        model's visibilities on the mocks.
        """
        return apply_kernel(self.terms.kernel, np.asarray(model_image, dtype=float)) / self._norm

    def dirty_image(self, visibilities):  # pragma: no cover - interface guard
        raise NotImplementedError(
            "KernelDirtyImager has no visibilities to image; use "
            "dirty_image_of_data() / dirty_image_of_model(image)"
        )


# --------------------------------------------------------------------------
# Before the kernel exists: the geometry needs the baselines
# --------------------------------------------------------------------------

@dataclass
class SpwHeader:
    n_rows: int
    frequencies: np.ndarray
    n_samples: int                 # unflagged (channel, row) samples
    row_lengths: np.ndarray        # |uv| per row at this window's f_max
    row_keep: np.ndarray           # rows not flagged in every channel
    meta: dict

    @property
    def n_chan(self) -> int:
        return int(len(self.frequencies))


@dataclass
class StreamHeader:
    """What `UVData`/`MultiSpwUVData` know without the data: the small arrays.

    Mirrors the properties `api.run` and `_parameter_record` read off a
    dataset object -- counts, frequencies, baselines, meta -- so the streaming
    path can resolve the geometry and describe the run before a single data
    value has been read. `noise_sample` is the first chunk's sigma, for the
    median the record reports.
    """

    spws: list[SpwHeader]
    meta: dict
    noise_sample: np.ndarray | None = None

    @property
    def n_spw(self) -> int:
        return len(self.spws)

    @property
    def n_vis(self) -> int:
        return int(sum(s.n_rows for s in self.spws))

    @property
    def n_chan(self) -> int:
        return int(sum(s.n_chan for s in self.spws))

    @property
    def n_samples(self) -> int:
        return int(sum(s.n_samples for s in self.spws))

    @property
    def frequencies(self) -> np.ndarray:
        return np.sort(np.concatenate([s.frequencies for s in self.spws]))

    def channel_index(self) -> list[tuple[int, int]]:
        """Global channel order -> (spw, channel within it), by frequency --
        `MultiSpwUVData._channel_index`, so a streamed cube's planes come out
        in the same order as an in-memory one's."""
        pairs = [
            (float(f), i, c)
            for i, s in enumerate(self.spws)
            for c, f in enumerate(s.frequencies)
        ]
        pairs.sort()
        return [(i, c) for _, i, c in pairs]

    def with_centre(self, y0: float, x0: float) -> "StreamHeader":
        """The header of the recentred stream: `meta` carries the offset the
        output WCS has to follow (`uvdata._with_centre`)."""
        from .uvdata import _with_centre

        return StreamHeader(
            spws=self.spws, meta=_with_centre(self.meta, y0, x0),
            noise_sample=self.noise_sample,
        )

    @property
    def central_frequency(self) -> float:
        """`MultiSpwUVData.central_frequency`: sample-weighted mean."""
        total = num = 0.0
        for s in self.spws:
            w = float(s.n_samples) / max(s.n_chan, 1)
            num += w * float(np.sum(s.frequencies))
            total += w * s.n_chan
        return float(num / total) if total else float(np.mean(self.frequencies))

    @property
    def fractional_bandwidth(self) -> float:
        from .uvdata import _fractional_bandwidth
        return _fractional_bandwidth(self.frequencies)

    @property
    def row_lengths(self) -> np.ndarray:
        return np.concatenate([s.row_lengths for s in self.spws])

    @property
    def row_keep(self) -> np.ndarray:
        return np.concatenate([s.row_keep for s in self.spws])

    @property
    def max_baseline_wavelengths(self) -> float:
        return float(np.max(self.row_lengths))

    def baseline_percentile_wavelengths(self, percentile: float = 95.0) -> float:
        """`UVData.baseline_percentile_wavelengths`, on the same population:
        one entry per row not flagged in every channel."""
        keep = self.row_keep
        lengths = self.row_lengths
        lengths = lengths[keep] if keep.any() else lengths
        return float(np.percentile(lengths, percentile))

    @property
    def noise(self) -> np.ndarray:
        return self.noise_sample if self.noise_sample is not None else np.zeros(0)


def scan_header(source, chunk_k: int = STREAM_CHUNK_K) -> StreamHeader:
    """The header without the data: uvw, frequencies, meta, and the flags.

    The kernel's grid depends on the pixel scale, which depends on b_95 and
    b_max, so this has to run before `accumulate_sparse_terms`. Per row and per
    channel arrays are small; the flags are the one per-sample read -- a byte
    per sample, so 200 MB on a 202M-sample cube, streamed and not kept -- and
    they are needed for the sample count and the flagged-everywhere exclusion
    the in-memory definitions use.
    """
    import json as _json

    if isinstance(source, (UVData, MultiSpwUVData)):
        spws = []
        for spw in source.spws:
            uvw = np.asarray(spw.uvw, dtype=float)
            keep = (np.ones(uvw.shape[0], dtype=bool) if spw.flags is None
                    else ~np.all(spw.flags, axis=0))
            spws.append(SpwHeader(uvw.shape[0], np.asarray(spw.frequencies, dtype=float),
                                  spw.n_samples, _row_lengths(uvw, spw.frequencies),
                                  keep, dict(spw.meta)))
        sample = np.asarray(source.spws[0].noise).ravel()[:chunk_k]
        return StreamHeader(spws, dict(getattr(source, "meta", {}) or {}), sample)

    p = Path(source)
    if p.is_file() and p.suffix == ".npz":
        spws = []
        with zipfile.ZipFile(p) as zf:
            names = set(zf.namelist())

            def arr(name):
                st = _NpyMemberStream(zf, name + ".npy")
                try:
                    return st.read_all()
                finally:
                    st.close()

            meta = _json.loads(str(arr("meta"))) if "meta.npy" in names else {}
            multi = "n_spw.npy" in names
            prefixes = [f"spw{i:03d}_" for i in range(int(arr("n_spw")))] if multi else [""]
            per_spw = meta.get("per_spw_meta") or []
            for i, pre in enumerate(prefixes):
                uvw = np.asarray(arr(pre + "uvw"), dtype=float)
                freqs = np.atleast_1d(np.asarray(arr(pre + "frequencies"), dtype=float))
                n_rows, n_chan = uvw.shape[0], len(freqs)
                all_flagged = np.ones(n_rows, dtype=bool)
                n_flagged = 0
                if f"{pre}flags.npy" in names:
                    fs = _NpyMemberStream(zf, f"{pre}flags.npy")
                    try:
                        step = max(1, chunk_k // max(n_rows, 1))
                        for c0 in range(0, n_chan, step):
                            blk = fs.read_rows(min(step, n_chan - c0)).astype(bool)
                            all_flagged &= np.all(blk, axis=0)
                            n_flagged += int(np.count_nonzero(blk))
                    finally:
                        fs.close()
                else:
                    all_flagged[:] = False
                spws.append(SpwHeader(
                    n_rows, freqs, n_chan * n_rows - n_flagged,
                    _row_lengths(uvw, freqs), ~all_flagged,
                    dict(per_spw[i]) if i < len(per_spw) else dict(meta),
                ))
            # a sample of sigma for the record's median: the first channel
            ns = _NpyMemberStream(zf, f"{prefixes[0]}noise_re.npy")
            try:
                sample = ns.read_rows(1).ravel()[:chunk_k]
            finally:
                ns.close()
        return StreamHeader(spws, meta, sample)

    from astropy.io import fits

    from .uvdata import _FILES

    spw_dirs = sorted(d for d in p.iterdir() if d.is_dir() and d.name.startswith("spw")) \
        if not (p / _FILES["data"]).exists() else [p]
    top_meta = _json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
    spws = []
    for d in spw_dirs:
        uvw = np.asarray(fits.getdata(d / _FILES["uvw"]), dtype=float)
        freqs = np.atleast_1d(np.asarray(fits.getdata(d / _FILES["frequencies"]), dtype=float))
        n_rows, n_chan = uvw.shape[0], len(freqs)
        fp = d / _FILES["flags"]
        if fp.exists():
            with fits.open(fp, memmap=True) as hf:
                fl = np.asarray(hf[0].data).astype(bool)
                keep, n_flagged = ~np.all(fl, axis=0), int(np.count_nonzero(fl))
        else:
            keep, n_flagged = np.ones(n_rows, dtype=bool), 0
        m = d / "meta.json"
        spws.append(SpwHeader(
            n_rows, freqs, n_chan * n_rows - n_flagged, _row_lengths(uvw, freqs), keep,
            _json.loads(m.read_text()) if m.exists() else dict(top_meta),
        ))
    with fits.open(spw_dirs[0] / _FILES["noise"], memmap=True) as hn:
        sample = np.asarray(hn[0].data[0, :chunk_k, 0])
    return StreamHeader(spws, top_meta or (spws[0].meta if spws else {}), sample)


# --------------------------------------------------------------------------
# A dataset object the fit can be handed, holding the operator and no data
# --------------------------------------------------------------------------

def stub_dataset_from_terms(terms: SparseTerms, geometry, mask, batch_size: int = 128):
    """An `ag.Interferometer` carrying the w-tilde operator and eight visibilities.

    autoarray's inversion factory picks the sparse class whenever
    `dataset.sparse_operator` is set, and the sparse class reads F and D from
    the operator alone -- so the dataset behind it can be anything of the
    right type. It is eight zero visibilities with unit noise at small uv.
    Everything in pyuvimage that would otherwise read the data or noise arrays
    goes through `fitting.streamed_terms_of` / `fitting.n_data_of` and reads
    the terms instead; anything that would need real visibilities (model
    visibilities, `DirtyImager`) raises rather than silently using the stub.
    """
    import autoarray as aa
    import autogalaxy as ag
    from autoarray.inversion.inversion.interferometer import (
        inversion_interferometer_util as sparse_util,
    )
    from .fitting import STREAMED_TERMS_ATTR

    # the operator wants the dirty image slim (masked 1-D, autoarray's order):
    # that is what D = M^T dirty multiplies, since the mapping matrix is
    # n_image_slim x n_mesh
    dirty_slim = np.asarray(
        aa.Array2D(values=np.asarray(terms.dirty_image, dtype=float), mask=mask).slim
    )
    operator = sparse_util.InterferometerSparseOperator.from_nufft_precision_operator(
        nufft_precision_operator=np.asarray(terms.kernel, dtype=float),
        dirty_image=dirty_slim,
        batch_size=batch_size,
    )
    n_stub = 8
    rng = np.random.default_rng(0)
    uv = rng.normal(0.0, 1e3, size=(n_stub, 2))
    stub = ag.Interferometer(
        data=aa.Visibilities(np.zeros(n_stub, dtype=complex)),
        noise_map=aa.VisibilitiesNoiseMap(np.ones(n_stub, dtype=complex) * (1 + 1j)),
        uv_wavelengths=uv,
        real_space_mask=mask,
        transformer_class=ag.TransformerDFT,
        sparse_operator=operator,
        raise_error_dft_visibilities_limit=False,
    )
    setattr(stub, STREAMED_TERMS_ATTR, terms)
    return stub

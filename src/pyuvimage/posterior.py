"""Draws from the posterior: `posterior_samples.npz` (`--posterior-samples N`).

Each draw is the delivered model plus a correlated offset from the posterior
covariance -- at the fitted prior strength, or marginalised over it under
``--uncertainty bayesian`` (`SingleFit.posterior_deltas`). Compute anything
from each draw -- a moment map, a line ratio, a flux in a region -- and the
scatter over draws is its uncertainty, with every pixel-to-pixel correlation
carried through. That is how much structure finer than the beam the data
support: at high S/N the draws agree on it, at low S/N they spread as far as
the prior allows.

The draws are Gaussian: positivity is not imposed, so where the model sits at
zero they scatter either side of it. Analytic point components are not
sampled.

    from pyuvimage.posterior import PosteriorSamples
    post = PosteriorSamples("pyuvimage_out/posterior_samples.npz")
    model = fits.getdata("pyuvimage_out/model.fits")      # (n_chan, ny, nx)
    for k in range(post.n_samples):
        cube = post.draw(k, model)                         # one draw, same shape
    offsets = post.offsets(channel)                        # (n_samples, ny, nx)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class PosteriorSamples:
    """Lazy reader for `posterior_samples.npz`. Planes are in `model.fits`
    order and orientation (FITS: row 0 at the bottom)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._z = np.load(self.path, allow_pickle=False)
        self.n_planes = int(self._z["n_planes"])
        self.shape = tuple(int(v) for v in self._z["shape"])
        self.mode = str(self._z["mode"])
        self.planes = [i for i in range(self.n_planes) if f"p{i}_delta" in self._z.files]
        if not self.planes:
            raise ValueError(f"{self.path} holds no draws")
        self.n_samples = int(self._z[f"p{self.planes[0]}_delta"].shape[0])
        self._maps: dict[int, object] = {}

    def has(self, plane: int) -> bool:
        return plane in self.planes

    def _map(self, plane: int):
        import scipy.sparse as sp

        if plane not in self._maps:
            z = self._z
            n_pix = self.shape[0] * self.shape[1]
            n_par = z[f"p{plane}_delta"].shape[1]
            self._maps[plane] = sp.csr_matrix(
                (z[f"p{plane}_weight"], (z[f"p{plane}_pixel"], z[f"p{plane}_param"])),
                shape=(n_pix, n_par))
        return self._maps[plane]

    def offsets(self, plane: int, samples=None) -> np.ndarray:
        """(n, ny, nx) offsets of the draws from the model in one plane; zeros
        for a plane without draws."""
        idx = range(self.n_samples) if samples is None else samples
        idx = np.atleast_1d(np.asarray(list(idx) if not isinstance(idx, np.ndarray) else idx))
        if not self.has(plane):
            return np.zeros((len(idx), *self.shape), dtype=np.float32)
        delta = self._z[f"p{plane}_delta"][idx]                      # (n, n_par)
        img = (self._map(plane) @ delta.T).T                         # (n, n_pix)
        return np.asarray(img, dtype=np.float32).reshape(len(idx), *self.shape)

    def draw(self, k: int, model: np.ndarray) -> np.ndarray:
        """Draw ``k``: ``model`` (one plane or a cube, as in model.fits) plus
        its offsets."""
        model = np.asarray(model, dtype=np.float32)
        if model.ndim == 2:
            return model + self.offsets(self.planes[0], [k])[0]
        out = model.copy()
        for c in range(model.shape[0]):
            if self.has(c):
                out[c] += self.offsets(c, [k])[0]
        return out

# Several spectral windows

[← back to the README](../README.md)

```bash
pyuvimage import obs.ms mydata/ --spw all   # one (0), a list (0,2), a range (0-3), or all
pyuvimage fit mydata/ --fov 3.0             # --mode mfs (default) or --mode cube
```

Multiple windows are imported into one dataset and imaged together by
multifrequency synthesis as a single image.

Nothing is averaged or resampled to make them fit together. Every visibility
already carries its own (u, v) computed at its own channel frequency, so
combining windows is the same operation the single-window path has always
performed across channels — splitting a dataset into spectral windows and
imaging it gives a bit-identical image, which is a regression test.

Windows keep their own channels, their own rows and their own noise estimate,
because in a measurement set all three differ between them. On disk the dataset
becomes `spw000/`, `spw001/`, ...; single-window datasets written before this
keep working unchanged.

## The fractional-bandwidth warning

**MFS fits one frequency-independent image.** That is mild within one window
and can be strong across several: at fractional bandwidth *B*, a source with
spectral index α is mis-modelled by roughly |α|·*B* across the band — about
40% for α = −0.7 over a 2:1 frequency range.

pyuvimage has no Taylor-term expansion (CLEAN's `mtmfs`), so it warns above 20%
fractional bandwidth and leaves the judgement to you. The fit will still reach
its chi^2 target by absorbing the spectral structure into the image; read the
result as a band-averaged sky, and image the windows separately if you need
spectra.

## Cube mode across windows

`--mode cube` also works across windows: channels are ordered by frequency and
fitted independently, with the regularisation frozen from an MFS fit.

Their spacing is then irregular, which a linear FITS frequency axis cannot
express, so the true per-plane frequencies are written to the header as
`FRQ0000...` and to `frequencies.json`, and `FREQIRR` marks that `CDELT3` is
only indicative.

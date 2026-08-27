# Installing, and what to do when the install is the problem

[← back to the README](../README.md)

```bash
pip install -e .            # core (numpy backend)
pip install -e ".[ms]"      # + python-casacore, to read measurement sets
pip install -e ".[jax]"     # + JAX/nufftax
```

Python ≥ 3.12 is required by current PyAutoGalaxy releases (3.11 works with
`version: python_version_check: False` in a local `config/general.yaml`).

JAX is optional. The NumPy path is fully supported and is what most of the
measurements in these docs were made on — see
[large-datasets.md](large-datasets.md) for why the JAX NUFFT is not the right
backend for large fits anyway.

## An arm64 conda environment on Apple silicon

This is the one environment detail worth getting right up front, because the
symptom of getting it wrong appears much later, inside a fit.

Every x86 `jaxlib` wheel is built with AVX instructions; Rosetta does not
provide them, so an **x86 Python on Apple silicon cannot run JAX at all** —
the error is *"This version of jaxlib was built using AVX instructions, which
your CPU and/or operating system do not support"*, and no amount of
`pip install -U jax` fixes it. Conda will happily give you an x86 environment
on an arm64 Mac if its channel or subdir says so, and an environment created
before you migrated machines stays x86 forever.

Check what you have:

```bash
python -c "import platform; print(platform.machine())"   # want: arm64
```

If that prints `x86_64`, build a native one:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n native_env python=3.12
conda activate native_env
conda config --env --set subdir osx-arm64   # keep it arm64 for later installs
pip install -e ".[jax]"
python -c "import platform, jax; print(platform.machine(), jax.__version__)"
```

The `conda config --env --set subdir` line matters as much as the first: it
pins the environment, so packages installed into it months later do not
quietly reintroduce x86 builds.

## `ModuleNotFoundError: No module named 'pyuvimage'`, from the `pyuvimage` script itself

The command exists but the package it imports does not — so the console script
and the package have ended up in different places. Almost always one of: an
editable install that failed *after* writing the script, or one made into a
different environment from the one on your `PATH`.

```bash
python -m pip show pyuvimage | grep -i "location"   # start here
head -1 $(which pyuvimage)                          # which python the script uses
python -c 'import sys; print(sys.executable)'       # which python you are in
```

**Check `Editable project location` first.** An editable install records an
absolute path, and if that directory has since been moved, renamed or emptied,
`pip show` still reports the package as installed while the import fails — the
path is where the package *was*. Reinstall from where it actually lives:

```bash
python -m pip uninstall -y pyuvimage
cd /path/to/pyuvimage
python -m pip install -e .        # python -m pip, not pip: same interpreter
python -c 'import pyuvimage; print(pyuvimage.__file__)'
```

If instead the two `python` paths disagree, activate the environment the script
belongs to. If `pip show` finds nothing at all, the install did not complete —
rerun it **and read the output**, since a failure still leaves
`src/pyuvimage.egg-info` behind and looks like it worked.

Until that is sorted, this always works and needs no install at all:

```bash
PYTHONPATH=/path/to/pyuvimage/src python -m pyuvimage.cli fit mydata/ --fov 8
```

## A broken JAX install

**`AttributeError: partially initialized module 'jax' has no attribute
'version'`**, or any other crash mentioning jax. Your JAX install is broken,
not pyuvimage. The PyAuto libraries decide whether JAX exists by looking for it
on disk rather than importing it, so a broken install passes that check and
then fails deep inside a fit.

pyuvimage detects this at startup, falls back to the NumPy path, and tells you.
To fix JAX itself, in the environment you run pyuvimage from:

```bash
python -c 'import jax; print(jax.__version__)'   # see the real error
pip uninstall -y jax jaxlib jax-metal
pip install -U 'jax[cpu]'                        # a matched jax/jaxlib pair
```

A jax/jaxlib version mismatch is the usual cause; on Apple silicon the
`jax-metal` plugin is another, and an x86 environment is the one above. Mixing
conda-forge and pip installs of jax in one environment does it too. If you
would rather not use JAX, `pip uninstall jax jaxlib` is a clean answer.

`nufftax` is a separate matter: it is **not** part of a default install (it
lives in autoarray's `optional` extra), so a working JAX with no nufftax is
common and silent. `pip install 'nufftax>=0.6.1,<0.7.0'` — the floor matters,
since earlier versions cannot differentiate a batched `nufft2d2`.

"""Time the pieces a non-negative solve is made of, to find a slow environment.

    python scripts/check_speed.py

Prints the platform, the BLAS numpy uses and its threads, whether numba
actually compiles autoarray's Cholesky helpers, and wall times for: a dense
Cholesky, a matrix product, and one `fnnls_cholesky` solve on a synthetic
positive-definite system the size of a 45x45 mesh. Compare with the reference
column (a 2-core Linux container, Python 3.13, numba 0.67, OpenBLAS).
"""
from __future__ import annotations

import os
import platform
import sys
import time

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
import numpy as np

REFERENCE = {"cholesky": 0.08, "matmul": 0.11, "jit compile": 0.6,
             "jitted call": 0.002, "fnnls": 2.5}


def timed(label, fn, repeat=1):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    ref = REFERENCE.get(label)
    print(f"  {label:<14} {best:9.4f} s" + (f"   (reference {ref:g} s)" if ref else ""))
    return out


def main():
    print(f"python {sys.version.split()[0]} on {platform.platform()} "
          f"({platform.machine()}), {os.cpu_count()} cpus")
    if sys.platform == "darwin":
        import subprocess
        try:
            arm = subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                                 capture_output=True, text=True).stdout.strip()
            if arm == "1" and platform.machine() == "x86_64":
                print("  ** x86_64 Python on Apple Silicon: running under Rosetta **")
        except Exception:
            pass
    for var in ("NUMBA_DISABLE_JIT", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    try:
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            print(f"  BLAS: {p.get('internal_api')} {p.get('version')} "
                  f"threads={p.get('num_threads')} ({p.get('filepath')})")
    except ImportError:
        print("  (pip install threadpoolctl to see the BLAS)")
        np.show_config()

    import numba
    import autoarray
    print(f"  numpy {np.__version__}, numba {numba.__version__}, "
          f"autoarray {autoarray.__version__} ({autoarray.__file__})")
    print(f"  numba config: DISABLE_JIT={numba.config.DISABLE_JIT}, "
          f"cache dir={numba.config.CACHE_DIR or '(default)'}")

    rng = np.random.default_rng(1)
    n = 2025                                  # a 45x45 mesh
    a = rng.normal(size=(n, n))
    spd = a @ a.T / n + np.eye(n)
    print("timings:")
    timed("cholesky", lambda: np.linalg.cholesky(spd), repeat=3)
    timed("matmul", lambda: a @ a, repeat=3)

    from autoarray.util import cholesky_funcs as cf
    u = np.linalg.cholesky(spd).T.copy()
    b = rng.normal(size=n)
    timed("jit compile", lambda: cf._cho_solve_buffer(u, n, b))
    timed("jitted call", lambda: cf._cho_solve_buffer(u, n, b), repeat=5)
    target = getattr(cf._cho_solve_buffer, "__wrapped__", None)
    compiled = type(getattr(cf, "_cho_solve_buffer")).__module__
    print(f"  _cho_solve_buffer is now a {compiled} object"
          + ("" if "numba" in compiled else "  ** NOT compiled by numba **"))

    from autoarray.util.fnnls import fnnls_cholesky
    truth = np.clip(rng.normal(size=n), 0, None)
    rhs = spd @ truth + 0.1 * rng.normal(size=n)
    stats = {}
    timed("fnnls", lambda: fnnls_cholesky(spd, rhs, stats=stats))
    print(f"  fnnls iterations: outer {stats.get('outer_iterations')}, "
          f"inner {stats.get('inner_iterations')}, passive {stats.get('n_passive')}")


if __name__ == "__main__":
    main()

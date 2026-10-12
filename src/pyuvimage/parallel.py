"""Running independent fits concurrently (`--workers`).

Cube channels share a frozen prior and are otherwise independent, so they can
be fitted side by side. Each worker is a separate *process*: the fits spend
much of their time in Python (autoarray's inversion setup, the bracket
searches) and would serialise on the GIL as threads.

Three things make that safe and worth it:

* **Start method `spawn`**, on every platform. It is the macOS default, and
  forking a process that has already initialised BLAS, numba or JAX threads
  is unsafe on Linux too. Each worker therefore imports pyuvimage afresh, once.
* **BLAS threads divided between the workers.** N workers each using every
  core would oversubscribe the machine and run slower than one. The thread
  count is set in the environment the workers are spawned with -- BLAS reads
  it when it loads, before any code here could change it -- and again with
  threadpoolctl where available.
* **Memory capped.** Each worker holds one channel's fit at a time, so N
  workers need about N times one channel's peak (`per_fit_memory_gb`) plus
  each process's own baseline. The worker count is reduced until that fits
  in what looks available.
"""
from __future__ import annotations

import contextlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("pyuvimage")

#: resident size of one worker process before it fits anything [GB]:
#: interpreter, numpy/scipy, autoarray/autogalaxy and their numba caches
WORKER_BASELINE_GB = 0.6

#: environment variables the common BLAS / OpenMP runtimes read at load time
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)

#: the channel a worker is fitting, for its log lines (`_ChannelFilter`)
_CURRENT_TASK = "-"


def cpu_count() -> int:
    """Cores this process may use (the affinity mask where there is one)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):          # macOS has no sched_getaffinity
        return max(1, os.cpu_count() or 1)


def resolve_workers(requested: int | str | None, n_tasks: int,
                    per_task_gb: float | None = None) -> int:
    """How many worker processes to run ``n_tasks`` independent fits on.

    ``requested`` is 1 (serial, the default), a number, or 0 / "auto" for
    as many as there are cores. Never more than the tasks, and reduced until
    ``per_task_gb`` per worker (plus `WORKER_BASELINE_GB`) fits in the
    memory that looks available.
    """
    from .fitting import available_memory_gb

    if requested in (None, "", 1, "1"):
        return 1
    if str(requested).lower() in ("0", "auto"):
        n = cpu_count()
    else:
        n = int(requested)
        if n < 1:
            raise ValueError(f"workers must be >= 1, 0 or 'auto', not {requested!r}")
    n = max(1, min(n, n_tasks))
    if n > 1 and per_task_gb is not None:
        have = available_memory_gb()
        if have is not None:
            fits = int(have // (per_task_gb + WORKER_BASELINE_GB))
            if fits < n:
                logger.warning(
                    "--workers %d would need about %.1f GB (%.1f GB per "
                    "channel fit + %.1f GB per process) against %.1f GB "
                    "available; running %d", n,
                    n * (per_task_gb + WORKER_BASELINE_GB), per_task_gb,
                    WORKER_BASELINE_GB, have, max(fits, 1),
                )
                n = max(fits, 1)
    return n


class _ChannelFilter(logging.Filter):
    """Prefixes a worker's log lines with the task it is running."""

    def filter(self, record):
        if not getattr(record, "_tagged", False):
            record.msg = f"[{_CURRENT_TASK}] {record.msg}"
            record._tagged = True
        return True


def _init_worker(log_level: int, blas_threads: int, module_state: dict) -> None:
    """Runs once in each worker before its first task."""
    from . import fitting

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(blas_threads)
    except Exception:                            # optional; the env did it
        pass
    for name, value in module_state.items():
        setattr(fitting, name, value)
    log = logging.getLogger("pyuvimage")
    log.setLevel(log_level)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - pyuvimage - %(levelname)s - %(message)s"))
        log.addHandler(handler)
    log.addFilter(_ChannelFilter())


def set_task_label(label: str) -> None:
    """Called by a task as it starts, so its log lines say which it is."""
    global _CURRENT_TASK
    _CURRENT_TASK = label


@contextlib.contextmanager
def _main_module_hidden():
    """Stop spawned workers from re-running the caller's script.

    A `spawn` child re-imports ``__main__`` (by path, or by name under
    ``python -m``) so that functions defined there can be unpickled. A user's
    script without an ``if __name__ == "__main__":`` guard -- the usual way to
    call `pyuvimage.run` -- would then start the whole fit again in every
    worker. The tasks here are module-level functions in pyuvimage and need
    nothing from ``__main__``, so the parent hides it while workers start:
    with no ``__file__`` and no ``__spec__`` it is treated like an
    interactive session, which is not re-imported.
    """
    import sys

    main = sys.modules.get("__main__")
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__")
             if main is not None and k in main.__dict__}
    try:
        if main is not None:
            main.__dict__.pop("__file__", None)
            if "__spec__" in saved:
                main.__spec__ = None
        yield
    finally:
        if main is not None:
            main.__dict__.update(saved)


@contextlib.contextmanager
def worker_pool(n_workers: int):
    """A process pool of ``n_workers``, BLAS threads shared out between them.

    The parent's own BLAS is already loaded and keeps its threads; the
    environment change only reaches the spawned workers, and is undone on
    exit.
    """
    import multiprocessing

    from . import fitting

    blas_threads = max(1, cpu_count() // n_workers)
    saved = {k: os.environ.get(k) for k in BLAS_THREAD_VARIABLES}
    os.environ.update({k: str(blas_threads) for k in BLAS_THREAD_VARIABLES})
    level = logging.getLogger("pyuvimage").getEffectiveLevel()
    # module-level switches `run` sets that a fresh process would not have
    state = {"PYNUFFT_HALF_PIXEL_SHIFT": fitting.PYNUFFT_HALF_PIXEL_SHIFT}
    logger.info("running %d worker processes, %d BLAS thread(s) each",
                n_workers, blas_threads)
    try:
        with _main_module_hidden(), ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(level, blas_threads, state),
        ) as pool:
            yield pool
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

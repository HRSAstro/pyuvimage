"""A JAX that is installed but cannot be imported must not kill a fit.

The PyAuto stack decides whether JAX exists with
`importlib.util.find_spec("jax")`, which only asks whether the package is on
disk. A broken install passes that check and then blows up the first time
anything does a real `import jax.numpy` -- and `autoarray`'s guard there is
`except ImportError`, so a jax/jaxlib version mismatch (which raises
`AttributeError: partially initialized module 'jax' has no attribute
'version'`) escapes as an unhandled exception from inside array indexing.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BROKEN_JAX = '''
raise AttributeError(
    "partially initialized module 'jax' has no attribute 'version' "
    "(most likely due to a circular import)"
)
'''


@pytest.fixture
def broken_jax_path(tmp_path):
    """A directory holding a `jax` package that raises on import."""
    pkg = tmp_path / "jax"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(BROKEN_JAX)
    (pkg / "numpy.py").write_text("")
    return tmp_path


def _run(code, extra_path, env_extra=None):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(extra_path), env.get("PYTHONPATH", "")]
    )
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=env, timeout=600,
    )


def test_broken_jax_really_does_break_the_stack(broken_jax_path):
    """Guard the guard: without pyuvimage imported, this must still fail.

    If upstream ever fixes its `except ImportError`, this test starts failing
    and the workaround can be deleted.
    """
    out = _run('''
        import autogalaxy as ag
        mask = ag.Mask2D.all_false(shape_native=(16, 16), pixel_scales=0.1)
        try:
            mask.derive_indexes.border_slim
            print("NO_ERROR")
        except AttributeError as e:
            print("ATTRIBUTE_ERROR", e)
    ''', broken_jax_path)
    assert "ATTRIBUTE_ERROR" in out.stdout, out.stdout + out.stderr


def test_importing_pyuvimage_neutralises_a_broken_jax(broken_jax_path):
    """With pyuvimage imported first, the same call works on the NumPy path."""
    out = _run('''
        import pyuvimage
        from pyuvimage._jax_guard import DISABLED_REASON
        assert DISABLED_REASON is not None, "guard did not fire"
        import sys
        assert "jax" not in sys.modules, "a None sentinel breaks scipy"
        try:
            import jax
            raise SystemExit("jax imported when it should not")
        except ImportError:
            pass
        import autogalaxy as ag
        mask = ag.Mask2D.all_false(shape_native=(16, 16), pixel_scales=0.1)
        print("BORDER_OK", len(mask.derive_indexes.border_slim))
    ''', broken_jax_path)
    assert "BORDER_OK" in out.stdout, out.stdout + out.stderr


def test_scipy_still_works_with_the_guard_installed(broken_jax_path):
    """`sys.modules["jax"] = None` would break scipy; the finder must not.

    scipy's array_api_compat asks `sys.modules["jax"]` for an attribute
    whenever it type-checks an array, so a None there turns every
    `scipy.linalg.block_diag` call into `'NoneType' object has no attribute
    'Array'`.
    """
    out = _run('''
        import pyuvimage
        import numpy as np, scipy.linalg
        print("BLOCK_DIAG", scipy.linalg.block_diag(np.eye(2), np.eye(2)).shape)
    ''', broken_jax_path)
    assert "BLOCK_DIAG (4, 4)" in out.stdout, out.stdout + out.stderr


def test_a_working_environment_is_left_alone():
    """No JAX installed at all: the guard must do nothing and say nothing."""
    out = _run('''
        import pyuvimage
        from pyuvimage._jax_guard import DISABLED_REASON
        print("REASON", DISABLED_REASON)
    ''', Path("."))
    assert "REASON None" in out.stdout, out.stdout + out.stderr

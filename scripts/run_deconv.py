"""CLI entry point for UV-plane image deconvolution."""

import argparse
import logging
import sys
from pathlib import Path

try:
    _SCRIPT_FILE = __file__
except NameError:
    _SCRIPT_FILE = None

if _SCRIPT_FILE is not None:
    _search_paths = Path(_SCRIPT_FILE).resolve().parents
else:
    _search_paths = [Path.cwd(), *Path.cwd().parents]

for _parent in _search_paths:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        _repo_str = str(_parent)
        if _repo_str not in sys.path:
            sys.path.insert(0, _repo_str)
        break

from scripts.bootstrap import bootstrap_script
from src.deconv.pipeline import run_deconv
from src.deconv.settings import load_settings

ROOT = bootstrap_script(_SCRIPT_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generalized UV-plane image deconvolution (forward modelling)."
    )
    parser.add_argument(
        "--settings",
        required=True,
        help="Path to runner JSON settings file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings(path=args.settings, repo_root=ROOT)
    result = run_deconv(settings)
    print(f"Done ({result['mode']}): {result['output_path']}")
    return result


if __name__ == "__main__":
    main()

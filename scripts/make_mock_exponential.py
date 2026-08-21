"""Generate a noisy exponential-source mock for pyuvimage tests."""

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
from src.deconv.mock import make_exponential_mock

ROOT = bootstrap_script(_SCRIPT_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate an exponential source as dataprep FITS products."
    )
    parser.add_argument(
        "--output-dir",
        default="./data/mock_exponential",
        help="Directory for FITS products and preview PNG",
    )
    parser.add_argument("--fov", type=float, default=5.0, help="FOV in arcsec")
    parser.add_argument("--n-pixels", type=int, default=64)
    parser.add_argument("--n-vis", type=int, default=200)
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--effective-radius", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--uv-path",
        default=None,
        help="Optional uv_wavelengths FITS (else synthetic coverage)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = make_exponential_mock(
        output_dir=args.output_dir,
        fov=args.fov,
        n_pixels=args.n_pixels,
        uv_path=args.uv_path,
        n_vis=args.n_vis,
        noise_sigma=args.noise_sigma,
        intensity=args.intensity,
        effective_radius=args.effective_radius,
        seed=args.seed,
    )
    print(f"Wrote mock data to {result['output_dir']}")
    print(f"  n_vis={result['n_vis']}, preview={result['paths']['preview']}")
    return result


if __name__ == "__main__":
    main()

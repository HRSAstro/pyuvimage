"""CASA MeasurementSet → FITS visibility export."""

import argparse
import json
import os
import shutil
from pathlib import Path

from src.dataprep import dataprep_lib
from src.utils.io import (
    export_path,
    exported_array_exists,
    output_dir_from_settings,
)


def resolve_settings_path(path, repo_root=None):
    path = Path(path)
    if path.is_file():
        return path

    candidates = [path]
    if repo_root is not None:
        repo_root = Path(repo_root)
        candidates.extend(
            [
                repo_root / path,
                repo_root / "settings" / "dataprep" / path,
                repo_root / "settings" / "dataprep" / path.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Settings file not found: {path} (tried: {tried})")


def load_settings(path, repo_root=None):
    settings_path = resolve_settings_path(path=path, repo_root=repo_root)
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_casa_statwt(module_globals=None):
    if module_globals is not None and "statwt" in module_globals:
        return module_globals["statwt"]
    try:
        from casatasks import statwt

        return statwt
    except ImportError as exc:
        raise RuntimeError(
            "CASA task 'statwt' is not available. Data preparation must be run "
            "inside CASA, for example:\n"
            "  casa -c scripts/run_dataprep.py --settings settings/dataprep/example.json"
        ) from exc


def export_products(settings, module_globals=None):
    outputvis = settings["outputvis"]
    output_dir = output_dir_from_settings(settings)
    statwt = _require_casa_statwt(module_globals=module_globals)

    print(f"Reading MS from: {outputvis}")
    print(f"Writing products to: {output_dir}")

    filename_uv_wavelengths = export_path(settings, "uv_wavelengths")
    if not exported_array_exists(filename_uv_wavelengths):
        print(
            "Writing",
            dataprep_lib.export_uv_wavelengths(
                ms=outputvis, filename=str(filename_uv_wavelengths)
            ),
        )

    filename_visibilities = export_path(settings, "visibilities")
    if not exported_array_exists(filename_visibilities):
        print(
            "Writing",
            dataprep_lib.export_visibilities(
                ms=outputvis, filename=str(filename_visibilities)
            ),
        )

    filename_antennas = export_path(settings, "antennas")
    if not exported_array_exists(filename_antennas):
        print(
            "Writing",
            dataprep_lib.export_antennas(
                ms=outputvis, filename=str(filename_antennas)
            ),
        )

    filename_scans = export_path(settings, "scans")
    if not exported_array_exists(filename_scans):
        print(
            "Writing",
            dataprep_lib.export_scans(ms=outputvis, filename=str(filename_scans)),
        )

    filename_frequencies = export_path(settings, "frequencies")
    if not exported_array_exists(filename_frequencies):
        print(
            "Writing",
            dataprep_lib.export_frequencies(
                ms=outputvis, filename=str(filename_frequencies)
            ),
        )

    statwt_ms = outputvis + ".statwt"
    if not os.path.isdir(statwt_ms):
        shutil.copytree(outputvis, statwt_ms)
        print("Executing 'statwt'...")
        statwt(vis=statwt_ms, datacolumn="data")

    filename_sigma = export_path(settings, "sigma_statwt")
    if not exported_array_exists(filename_sigma):
        print(
            "Writing",
            dataprep_lib.export_sigma(ms=statwt_ms, filename=str(filename_sigma)),
        )

    filename_weights = export_path(settings, "weights")
    if not exported_array_exists(filename_weights):
        print(
            "Writing",
            dataprep_lib.export_weights(
                ms=statwt_ms, filename=str(filename_weights)
            ),
        )


def main(module_globals, default_settings_path, repo_root=None):
    parser = argparse.ArgumentParser(
        description="Export CASA MeasurementSet products to FITS."
    )
    parser.add_argument(
        "--settings",
        default=default_settings_path,
        help="Path to JSON settings file",
    )
    args = parser.parse_args()
    settings = load_settings(path=args.settings, repo_root=repo_root)
    export_products(settings=settings, module_globals=module_globals)

"""Load exported FITS visibility cubes for deconvolution."""

from pathlib import Path

import numpy as np

from src.utils.io import load_exported_array


def load_cube_data(settings):
    """
    Load frequencies, UV, visibilities, and sigma from dataprep FITS products.

    Concatenates XX and YY polarizations on the visibility axis (LensKin style).
    Returns arrays shaped for ``(n_channels, n_vis, 2)`` visibilities.
    """
    directory = Path(settings["data_directory"])
    uids = settings["uids"]
    width = settings["width"]
    patterns = settings["data_patterns"]
    extra_context = settings.get("pattern_context", {})

    list_of_frequencies = []
    list_of_uv_wavelengths = []
    list_of_visibilities = []
    list_of_sigma = []

    for uid in uids:
        fmt = {"uid": uid, "width": width, **extra_context}
        frequencies = load_exported_array(
            directory / patterns["frequencies"].format(**fmt)
        )
        list_of_frequencies.append(frequencies)

        uv_wavelengths = load_exported_array(
            directory / patterns["uv_wavelengths"].format(**fmt)
        )
        list_of_uv_wavelengths.append(
            np.concatenate((uv_wavelengths, uv_wavelengths), axis=1)
        )

        visibilities = load_exported_array(
            directory / patterns["visibilities"].format(**fmt)
        )
        list_of_visibilities.append(
            np.concatenate((visibilities[0], visibilities[1]), axis=1)
        )

        sigma = load_exported_array(directory / patterns["sigma"].format(**fmt))
        list_of_sigma.append(np.concatenate((sigma[0], sigma[1]), axis=1))

    if len(uids) == 1:
        frequencies = list_of_frequencies[0]
        uv_wavelengths = np.concatenate(list_of_uv_wavelengths, axis=0)
        visibilities = np.concatenate(list_of_visibilities, axis=0)
        sigma = np.concatenate(list_of_sigma, axis=0)
    else:
        frequencies = np.average(list_of_frequencies, axis=0)
        uv_wavelengths = np.concatenate(list_of_uv_wavelengths, axis=1)
        visibilities = np.concatenate(list_of_visibilities, axis=1)
        sigma = np.concatenate(list_of_sigma, axis=1)

    return frequencies, uv_wavelengths, visibilities, sigma


def load_cube_data_weights(settings):
    """Load optional CASA weights for MFS noise propagation."""
    patterns = settings.get("data_patterns", {})
    weight_pattern = patterns.get("weights")
    if weight_pattern is None:
        return None

    directory = Path(settings["data_directory"])
    uids = settings["uids"]
    width = settings["width"]
    extra_context = settings.get("pattern_context", {})
    list_of_weights = []

    for uid in uids:
        fmt = {"uid": uid, "width": width, **extra_context}
        weights = load_exported_array(directory / weight_pattern.format(**fmt))
        list_of_weights.append(np.concatenate((weights[0], weights[1]), axis=1))

    if len(uids) == 1:
        return np.concatenate(list_of_weights, axis=0)
    return np.concatenate(list_of_weights, axis=1)

import numpy as np
import os

from src.utils.io import write_exported_array

try:
    from astropy import constants, units

    _astropy_units_available = True
except Exception:
    _astropy_units_available = False

_tb = None


def _get_tb():
    global _tb
    if _tb is not None:
        return _tb
    try:
        from casatools import table

        _tb = table()
        return _tb
    except ImportError as exc:
        raise RuntimeError(
            "CASA table tool 'tb' is not available. Data preparation must be run "
            "inside CASA (see scripts/run_dataprep.py)."
        ) from exc


def getcol_wrapper(ms, table, colname):
    if not os.path.isdir(ms):
        raise IOError(f"{ms} does not exist")
    tb = _get_tb()
    tb.open(f"{ms}/{table}")
    col = np.squeeze(tb.getcol(colname))
    tb.close()
    return col


def convert_array_to_wavelengths(array, frequency):
    if _astropy_units_available:
        return (((array * units.m) * (frequency * units.Hz)) / constants.c).decompose().value
    return array * frequency / 299792458.0


def get_visibilities(ms):
    data = getcol_wrapper(ms=ms, table="", colname="DATA")
    return np.stack(arrays=(data.real, data.imag), axis=-1)


def export_visibilities(ms, filename):
    return write_exported_array(filename, get_visibilities(ms=ms))


def get_uv_wavelengths(ms):
    uvw = getcol_wrapper(ms=ms, table="", colname="UVW")
    chan_freq = getcol_wrapper(ms=ms, table="SPECTRAL_WINDOW", colname="CHAN_FREQ")
    chan_freq_shape = np.shape(chan_freq)
    if np.shape(chan_freq):
        u_wavelengths, v_wavelengths = np.zeros(shape=(2, chan_freq_shape[0], uvw.shape[1]))
        for i in range(chan_freq_shape[0]):
            u_wavelengths[i, :] = convert_array_to_wavelengths(array=uvw[0, :], frequency=chan_freq[i])
            v_wavelengths[i, :] = convert_array_to_wavelengths(array=uvw[1, :], frequency=chan_freq[i])
    else:
        u_wavelengths = convert_array_to_wavelengths(array=uvw[0, :], frequency=chan_freq)
        v_wavelengths = convert_array_to_wavelengths(array=uvw[1, :], frequency=chan_freq)
    return np.stack(arrays=(u_wavelengths, v_wavelengths), axis=-1)


def export_uv_wavelengths(ms, filename):
    return write_exported_array(filename, get_uv_wavelengths(ms=ms))


def get_sigma(ms):
    """
    Export CASA SIGMA values with shape ``(n_corr, n_chan, n_row, 2)``.

    If the MS stores a 2D SIGMA ``(n_corr, n_row)`` (no channel axis), the
    same value is broadcast to every channel. If SIGMA is already
    ``(n_corr, n_chan, n_row)``, per-channel statwt values are preserved.
    """
    sigma = np.asarray(getcol_wrapper(ms=ms, table="", colname="SIGMA"))
    chan_freq = getcol_wrapper(ms=ms, table="SPECTRAL_WINDOW", colname="CHAN_FREQ")
    n_chan = len(chan_freq) if np.ndim(chan_freq) else 1

    if sigma.ndim == 2:
        sigma = np.tile(sigma[:, np.newaxis, :], (1, n_chan, 1))
    elif sigma.ndim == 3:
        if sigma.shape[1] != n_chan:
            raise ValueError(
                f"SIGMA channel axis {sigma.shape[1]} does not match "
                f"SPECTRAL_WINDOW nchan={n_chan}."
            )
    else:
        raise ValueError(f"Unexpected CASA SIGMA shape: {sigma.shape}")

    return np.stack(arrays=(sigma, sigma), axis=-1)


def get_weights(ms):
    """Export CASA WEIGHT values with shape ``(n_corr, n_chan, n_row)``."""
    weight = np.asarray(getcol_wrapper(ms=ms, table="", colname="WEIGHT"))
    chan_freq = getcol_wrapper(ms=ms, table="SPECTRAL_WINDOW", colname="CHAN_FREQ")
    n_chan = len(chan_freq) if np.ndim(chan_freq) else 1

    if weight.ndim == 2:
        weight = np.tile(weight[:, np.newaxis, :], (1, n_chan, 1))
    elif weight.ndim == 3:
        if weight.shape[1] != n_chan:
            raise ValueError(
                f"WEIGHT channel axis {weight.shape[1]} does not match "
                f"SPECTRAL_WINDOW nchan={n_chan}."
            )
    else:
        raise ValueError(f"Unexpected CASA WEIGHT shape: {weight.shape}")

    return weight


def export_weights(ms, filename):
    return write_exported_array(filename, get_weights(ms=ms))


def export_sigma(ms, filename):
    return write_exported_array(filename, get_sigma(ms=ms))


def get_frequencies(ms):
    return getcol_wrapper(ms=ms, table="SPECTRAL_WINDOW", colname="CHAN_FREQ")


def export_frequencies(ms, filename):
    return write_exported_array(filename, get_frequencies(ms=ms))


def get_antennas(ms):
    antenna1 = getcol_wrapper(ms=ms, table="", colname="ANTENNA1")
    antenna2 = getcol_wrapper(ms=ms, table="", colname="ANTENNA2")
    return np.array([antenna1, antenna2])


def export_antennas(ms, filename):
    return write_exported_array(filename, get_antennas(ms=ms))


def get_scans(ms):
    return np.asarray(getcol_wrapper(ms=ms, table="", colname="SCAN_NUMBER"))


def export_scans(ms, filename):
    return write_exported_array(filename, get_scans(ms=ms))

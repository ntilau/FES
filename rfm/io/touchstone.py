"""
TouchStone file format support for the rfm RFM implementation.

Provides functions to read and write TouchStone (.snp) files, the standard
text format for RF/microwave network data.

Only Touchstone 1.x files are supported, with a single ``#`` option line and
S-parameter data in RI, MA or DB format.  The ``DUM`` stores S-parameters as
``(Nports, Nports, Nfreq)``.
"""

import re
from pathlib import Path
from typing import Union

import numpy as np

try:
    from ..dum import DUM
except (ImportError, ValueError):
    # Fallback for when this module is run directly.
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dum import DUM


_FREQ_UNITS = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9, 'THZ': 1e12}
_PARAM_TYPES = ('S', 'Y', 'Z', 'H', 'G')
_DATA_FORMATS = ('RI', 'MA', 'DB')


def _parse_option_line(line):
    """Parse a ``#`` option line into (frequency scale, parameter, format)."""
    freq_scale = 1e9  # Touchstone default is GHz
    param_type = 'S'
    data_format = 'RI'
    for token in line[1:].strip().split():
        upper = token.upper()
        if upper in _FREQ_UNITS:
            freq_scale = _FREQ_UNITS[upper]
        elif upper in _PARAM_TYPES:
            param_type = upper
        elif upper in _DATA_FORMATS:
            data_format = upper
    return freq_scale, param_type, data_format


def _count_ports(filename, first_data_line):
    """Determine the port count from the ``.sNp`` suffix or the data width."""
    match = re.search(r'\.s(\d+)p$', str(filename), re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fall back to the first data line: 1 + 2 * Nports**2 columns.
    n_columns = len(first_data_line.split())
    nports_squared = (n_columns - 1) // 2
    nports = int(round(np.sqrt(nports_squared)))
    if nports < 1 or 1 + 2 * nports * nports != n_columns:
        raise ValueError(
            "Cannot infer the number of ports; use a .sNp file extension"
        )
    return nports


def _to_complex(magnitude, phase, data_format):
    """Convert a Touchstone magnitude/phase pair to a complex value."""
    if data_format == 'RI':
        return complex(magnitude, phase)
    mag = magnitude if data_format == 'MA' else 10.0 ** (magnitude / 20.0)
    return mag * np.exp(1j * np.deg2rad(phase))


def read_touchstone(filename: Union[str, Path]) -> DUM:
    """
    Read a TouchStone (.snp) file and return a DUM object.

    Parameters
    ----------
    filename : str or Path
        Path to the TouchStone file to read.

    Returns
    -------
    DUM
        DUM object with ``S`` of shape ``(Nports, Nports, Nfreq)``.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"TouchStone file not found: {filename}")

    with open(filename, 'r') as handle:
        lines = handle.readlines()

    # Locate the option line and the first data line.
    freq_scale, param_type, data_format = 1e9, 'S', 'RI'
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            freq_scale, param_type, data_format = _parse_option_line(stripped)
        elif not stripped.startswith('!'):
            data_lines.append(stripped)

    if param_type != 'S':
        raise ValueError(
            f"Unsupported TouchStone parameter type {param_type!r}; "
            "only S-parameters are supported"
        )

    if not data_lines:
        dum = DUM()
        dum.Nports = 0
        dum.isfreq = 1
        dum.f = np.array([], dtype=float)
        dum.S = np.zeros((0, 0, 0), dtype=complex)
        return dum

    nports = _count_ports(filename, data_lines[0])
    values_per_record = 2 * nports * nports

    # Touchstone 1.x allows a record to be wrapped over several lines, so
    # parse a flat token stream instead of line by line.
    tokens = []
    for line in data_lines:
        tokens.extend(line.split('!', 1)[0].split())

    freq_list = []
    matrices = []
    position = 0
    while position + 1 + values_per_record <= len(tokens):
        try:
            freq_hz = float(tokens[position]) * freq_scale
        except ValueError as exc:
            raise ValueError(
                f"Invalid frequency value {tokens[position]!r} in {filename}"
            ) from exc
        position += 1

        matrix = np.zeros((nports, nports), dtype=complex)
        # Touchstone order is S11, S21, ..., SN1, S12, S22, ...: the output
        # port index varies fastest, so fill by column.
        for in_port in range(nports):
            for out_port in range(nports):
                try:
                    magnitude = float(tokens[position])
                    phase = float(tokens[position + 1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid S-parameter value in {filename}"
                    ) from exc
                position += 2
                matrix[out_port, in_port] = _to_complex(
                    magnitude, phase, data_format
                )

        freq_list.append(freq_hz)
        matrices.append(matrix)

    dum = DUM()
    dum.Nports = nports
    dum.isfreq = 1
    dum.istime = 0
    dum.f = np.array(freq_list, dtype=float)
    if matrices:
        # (Nfreq, Nports, Nports) -> (Nports, Nports, Nfreq)
        dum.S = np.array(matrices, dtype=complex).transpose(1, 2, 0)
    else:
        dum.S = np.zeros((nports, nports, 0), dtype=complex)

    return dum


def _normalize_s(dum, nports, nfreq):
    """Return S as ``(Nfreq, Nports, Nports)`` for writing."""
    s_data = np.asarray(dum.S)
    if s_data.ndim != 3:
        raise ValueError("DUM.S must be a 3-D array")
    if s_data.shape == (nports, nports, nfreq):
        return s_data.transpose(2, 0, 1)
    if s_data.shape == (nfreq, nports, nports):
        return s_data
    raise ValueError(
        "DUM.S must have shape (Nports, Nports, Nfreq); "
        f"got {s_data.shape} for Nports={nports}, Nfreq={nfreq}"
    )


def write_touchstone(dum: DUM, filename: Union[str, Path]) -> None:
    """
    Write a DUM object to a TouchStone (.snp) file.

    Parameters
    ----------
    dum : DUM
        DUM object to write.
    filename : str or Path
        Path to the TouchStone file to write.
    """
    filename = Path(filename)

    nports = int(dum.Nports)
    nfreq = len(dum.f) if hasattr(dum, 'f') else 0
    s_data = _normalize_s(dum, nports, nfreq) if nfreq else None

    with open(filename, 'w') as handle:
        handle.write("# Hz S RI R 50\n")
        handle.write("! Generated by the rfm package (FES bundle)\n")
        handle.write(f"! Number of ports: {nports}\n")
        handle.write(f"! Number of frequency points: {nfreq}\n")
        handle.write("\n")

        if s_data is None:
            return

        for index, freq in enumerate(np.asarray(dum.f, dtype=float)):
            fields = [f"{freq:.15g}"]
            for in_port in range(nports):
                for out_port in range(nports):
                    value = s_data[index, out_port, in_port]
                    fields.append(f"{value.real:.15g}")
                    fields.append(f"{value.imag:.15g}")
            handle.write(" ".join(fields) + "\n")


def read_csv(filename: Union[str, Path]) -> DUM:
    """
    Read a CSV file containing multiport network data.

    The upstream implementation is a placeholder that delegates to the
    TouchStone reader; the same limitation is retained here.  Use
    :func:`read_touchstone` for RI/MA/DB Touchstone data.

    Parameters
    ----------
    filename : str or Path
        Path to the file to read.

    Returns
    -------
    DUM
        DUM object containing the data from the file.
    """
    return read_touchstone(filename)

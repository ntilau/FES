"""
RFM file export utilities for the pyrrfm RFM implementation.

Provides functions to export Macromodel (pole-residue) data to portable
file formats: JSON for the macromodel parameters, and TouchStone for
the fitted frequency response.

Functions:
    export_rfm_model      — Save Macromodel to JSON
    import_rfm_model      — Load Macromodel from JSON
    export_rfm_touchstone — Save fitted RFM response as TouchStone (.s2p)
    rfm_to_dum            — Convert Macromodel to DUM with fitted response
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..dum import DUM
from ..macromodel import Macromodel


def _dc_scalar(dc_gain) -> complex:
    """Return the scalar DC term, treating an empty array as zero."""
    values = np.asarray(dc_gain).ravel()
    return complex(values[0]) if values.size else 0j


def rfm_response(freq: np.ndarray, poles: np.ndarray, residues: np.ndarray,
                 dc_gain: complex) -> np.ndarray:
    """Evaluate the pole-residue RFM H(jw) = sum(r_k / (jw - p_k)) + dc_gain."""
    jw = 1j * np.asarray(freq, dtype=float)
    h = np.zeros_like(jw, dtype=complex)
    residues = np.asarray(residues).ravel()
    for p, r in zip(np.asarray(poles).ravel(), residues):
        h += r / (jw - p)
    h += _dc_scalar(dc_gain)
    return h


def rfm_to_dum(model: Macromodel, freq: np.ndarray,
               s_param_index: tuple = (1, 0)) -> DUM:
    """Convert a Macromodel to a DUM with the fitted frequency response.

    Parameters
    ----------
    model : Macromodel
        Pole-residue macromodel to evaluate.
    freq : np.ndarray
        Frequency points (Hz) at which to evaluate the model.
    s_param_index : tuple
        (output_port, input_port) 1-indexed S-parameter to fill with the
        fitted response. All other S-parameters are copied as zeros.

    Returns
    -------
    DUM
        DUM object with fitted S-parameter data.
    """
    nports = int(model.Nports)
    if nports < 1:
        raise ValueError("model.Nports must be a positive integer")
    poles = np.asarray(model.poles).ravel()
    residues = np.asarray(model.residues).ravel()
    dc_gain = _dc_scalar(model.dc_gain)

    # Evaluate the fitted response
    h = rfm_response(freq, poles, residues, dc_gain)

    # Build full S matrix (zeros except the requested parameter)
    out_port, in_port = s_param_index[0] - 1, s_param_index[1] - 1
    if not (0 <= out_port < nports and 0 <= in_port < nports):
        raise ValueError(
            f"s_param_index {s_param_index} is out of range for "
            f"{nports} ports"
        )
    S = np.zeros((nports, nports, len(freq)), dtype=complex)
    S[out_port, in_port, :] = h

    dum = DUM()
    dum.Nports = nports
    dum.isfreq = 1
    dum.istime = 0
    dum.f = np.asarray(freq, dtype=float)
    dum.S = S
    return dum


def export_rfm_touchstone(model: Macromodel, freq: np.ndarray,
                          filename: Union[str, Path],
                          s_param_index: tuple = (1, 0)) -> None:
    """Export fitted RFM response as a TouchStone (.s2p) file.

    Parameters
    ----------
    model : Macromodel
        Pole-residue macromodel to evaluate and export.
    freq : np.ndarray
        Frequency points (Hz) at which to evaluate the model.
    filename : str or Path
        Output TouchStone file path.
    s_param_index : tuple
        (output_port, input_port) 1-indexed S-parameter to fill.
    """
    from ..io.touchstone import write_touchstone
    dum = rfm_to_dum(model, freq, s_param_index)
    write_touchstone(dum, filename)


def export_rfm_model(model: Macromodel, filename: Union[str, Path],
                     metadata: Optional[dict] = None) -> None:
    """Export Macromodel parameters to a JSON file.

    Parameters
    ----------
    model : Macromodel
        Pole-residue macromodel to export.
    filename : str or Path
        Output JSON file path.
    metadata : dict, optional
        Additional metadata to include in the export (e.g., source file,
        fit quality metrics, passivity status).
    """
    poles = np.asarray(model.poles).ravel()
    residues = np.asarray(model.residues).ravel()
    dc_gain = _dc_scalar(model.dc_gain)

    data = {
        "representation": "pole-residue",
        "nports": int(model.Nports),
        "n_poles": int(len(poles)),
        "poles": [{"real": float(p.real), "imag": float(p.imag)} for p in poles],
        "residues": [{"real": float(r.real), "imag": float(r.imag)} for r in residues],
        "dc_gain": {"real": float(dc_gain.real), "imag": float(dc_gain.imag)},
    }

    if metadata:
        data["metadata"] = metadata

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def import_rfm_model(filename: Union[str, Path]) -> Macromodel:
    """Import Macromodel parameters from a JSON file.

    Parameters
    ----------
    filename : str or Path
        Input JSON file path.

    Returns
    -------
    Macromodel
        Reconstructed pole-residue macromodel.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    n_poles = data["n_poles"]
    poles = np.array([complex(c["real"], c["imag"]) for c in data["poles"]])
    residues = np.array([complex(c["real"], c["imag"]) for c in data["residues"]])
    dc_gain = complex(data["dc_gain"]["real"], data["dc_gain"]["imag"])

    model = Macromodel(representation="pole-residue")
    model.Nports = int(data.get("nports", 1))
    model.poles = poles
    model.residues = residues
    model.dc_gain = dc_gain
    return model

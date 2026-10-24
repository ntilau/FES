"""
Data extraction utilities for the rfm RFM implementation.

Functions for extracting subsets and computing metrics from DUM structures.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from ..dum import DUM


def extract_subdum(dum: DUM, port_indices: Union[List[int], Tuple[int, ...]]) -> DUM:
    """
    Extract a sub-DUM containing only the specified ports.

    Parameters
    ----------
    dum : DUM
        DUM object to extract from
    port_indices : list or tuple of int
        List of port indices to extract (0-indexed)

    Returns
    -------
    DUM
        New DUM object containing only the specified ports
    """
    # Create a new DUM object
    sub_dum = DUM()

    # Copy basic properties
    sub_dum.Nports = len(port_indices)
    sub_dum.isfreq = getattr(dum, 'isfreq', 0)
    sub_dum.istime = getattr(dum, 'istime', 0)

    # Copy frequency data if available
    if hasattr(dum, 'f') and len(dum.f) > 0:
        sub_dum.f = dum.f.copy()

    # Extract S-parameter data for the specified ports.  The DUM convention is
    # (Nports, Nports, Nfreq); (Nfreq, Nports, Nports) is also accepted.
    empty_shape = (len(port_indices), len(port_indices), 0)
    if hasattr(dum, 'S') and dum.S.size > 0 and dum.S.ndim == 3:
        S_full = dum.S
        n_freq = len(dum.f)
        if S_full.shape == (dum.Nports, dum.Nports, n_freq):
            sub_dum.S = S_full[np.ix_(port_indices, port_indices)]
        elif S_full.shape == (n_freq, dum.Nports, dum.Nports):
            rows, cols = np.ix_(port_indices, port_indices)
            sub_dum.S = S_full[:, rows, cols].transpose(1, 2, 0)
        else:
            sub_dum.S = np.zeros(empty_shape, dtype=complex)
    else:
        sub_dum.S = np.zeros(empty_shape, dtype=complex)

    return sub_dum


def extract_dum_energy(dum: DUM) -> float:
    """
    Extract energy-related metrics from a DUM structure.

    Parameters
    ----------
    dum : DUM
        DUM object to analyze

    Returns
    -------
    float
        Energy metric (placeholder implementation)
    """
    # Placeholder implementation
    if hasattr(dum, 'S') and dum.S.size > 0:
        # Return average power (simplified)
        return float(np.mean(np.abs(dum.S)**2))
    return 0.0


def extract_largest_responses(dum: DUM, n: int = 5) -> list:
    """
    Extract the n largest S-parameter responses from a DUM.

    Parameters
    ----------
    dum : DUM
        DUM object to analyze
    n : int, optional
        Number of largest responses to extract, default is 5

    Returns
    -------
    list
        List of tuples (port_i, port_j, max_magnitude) for the n largest responses
    """
    if hasattr(dum, 'S') and dum.S.size > 0 and dum.Nports > 0:
        # Reduce over frequency, whichever orientation the caller used.
        S = dum.S
        n_freq = len(dum.f) if hasattr(dum, 'f') else 0
        if S.ndim == 3 and S.shape == (dum.Nports, dum.Nports, n_freq):
            max_mags = np.max(np.abs(S), axis=2)
        elif S.ndim == 3 and S.shape == (n_freq, dum.Nports, dum.Nports):
            max_mags = np.max(np.abs(S), axis=0)
        else:
            max_mags = np.max(np.abs(S), axis=0)

        # Flatten and get indices of largest values
        flat_indices = np.argsort(max_mags.flatten())[::-1][:n]

        results = []
        for idx in flat_indices:
            # Convert flat index to 2D indices
            i = idx // dum.Nports
            j = idx % dum.Nports
            max_val = max_mags[i, j]
            results.append((i, j, float(max_val)))

        return results
    return []
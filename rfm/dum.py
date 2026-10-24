"""
DUM (Descriptor Unit Model) class for the rfm RFM implementation.

This class represents the Descriptor Unit Model structure used to store
time and/or frequency domain multiport data from raw simulations.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any


class DUM:
    """
    Descriptor Unit Model (DUM) class for storing multiport frequency/time domain data.

    Attributes
    ----------
    Nports : int
        Number of ports
    isfreq : int
        Flag indicating if data is in frequency domain (1) or not (0)
    istime : int
        Flag indicating if data is in time domain (1) or not (0)
    f : np.ndarray
        Frequency vector (if isfreq=1)
    t : np.ndarray
        Time vector (if istime=1)
    S : np.ndarray
        S-parameter matrix of shape (Nports, Nports, Nfreq) or (Nports, Nports, Ntime)
    Y : np.ndarray
        Admittance matrix (optional)
    Z : np.ndarray
        Impedance matrix (optional)
    """

    def __init__(self):
        """Initialize an empty DUM object."""
        self.Nports = 0
        self.isfreq = 0
        self.istime = 0
        self.f = np.array([])
        self.t = np.array([])
        self.S = np.array([])
        self.Y = np.array([])
        self.Z = np.array([])

    def __repr__(self):
        """String representation of the DUM object."""
        return f"DUM(Nports={self.Nports}, isfreq={self.isfreq}, istime={self.istime}, f_shape={getattr(self.f, 'shape', 'N/A')}, S_shape={getattr(self.S, 'shape', 'N/A')})"
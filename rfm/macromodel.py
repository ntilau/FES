"""
Macromodel class for the rfm RFM implementation.

This class represents the model structure used to store fitted models in
state-space or pole-residue form.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any


class Macromodel:
    """
    Macromodel class for storing reduced-order models.

    Attributes
    ----------
    representation : str
        Model representation type ('pole-residue', 'state-space', etc.)
    Nports : int
        Number of ports
    poles : np.ndarray
        Complex poles of the model
    residues : np.ndarray
        Residue matrices (for pole-residue form)
    dc_gain : np.ndarray
        DC gain term
    A : np.ndarray
        State matrix (for state-space form)
    B : np.ndarray
        Input matrix (for state-space form)
    C : np.ndarray
        Output matrix (for state-space form)
    D : np.ndarray
        Feedthrough matrix (for state-space form)
    """

    def __init__(self, representation: str = 'pole-residue'):
        """
        Initialize a Macromodel object.

        Parameters
        ----------
        representation : str, optional
            Model representation type, default is 'pole-residue'
        """
        self.representation = representation
        self.Nports = 0
        self.poles = np.array([])
        self.residues = np.array([])
        self.dc_gain = np.array([])
        self.A = np.array([])
        self.B = np.array([])
        self.C = np.array([])
        self.D = np.array([])

    def __repr__(self):
        """String representation of the Macromodel object."""
        return f"Macromodel(representation='{self.representation}', Nports={self.Nports}, n_poles={len(self.poles)})"
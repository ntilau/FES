"""
rfm RFM Implementation

An RFM (Rational Function Model) implementation for Descriptor Unit Model (DUM)
creation, manipulation, analysis, and conversion. Designed for LTI system identification,
model order reduction, and RF/microwave applications.

This package provides functionality for:
- DUM (Descriptor Unit Model) and Macromodel data structures
- Vector Fitting algorithm for model order reduction
- Hamiltonian-based passivity checking and enforcement
- Import/export of various file formats (TouchStone, CSV, CST, FDTD)
- Data visualization and analysis utilities

Example usage:
    >>> import rfm
    >>> import numpy as np
    >>>
    >>> # Import TouchStone data
    >>> dum = rfm.io.read_touchstone('measurement.s2p')
    >>>
    >>> # Apply Vector Fitting to create macromodel
    >>> freq = dum.f
    >>> response = dum.S[0, 0, :]  # S11 parameter
    >>> poles, residues, d_term = rfm.fitting.vector_fit(freq, response)
    >>>
    >>> # Create macromodel and check passivity
    >>> model = rfm.Macromodel(representation='pole-residue')
    >>> model.poles = poles
    >>> model.residues = residues
    >>> model.dc_gain = d_term
    >>>
    >>> is_passive, passive_model = rfm.validation.check_hamiltonian(model, freq)
    >>> if not is_passive:
    >>>     model = passive_model
    >>>
    >>> # Export results
    >>> rfm.io.write_touchstone(dum, 'model.s2p')
    >>>
    >>> # Visualize
    >>> rfm.utils.plot_dum(dum, 'S21', scale='DB')
"""

__version__ = "0.1.0"
__author__ = "rfm RFM Team"

# Import main components for easy access
from .dum import DUM
from .macromodel import Macromodel
from .fitting import VectorFit, vector_fit, VectorFit_inf_dc0, vector_fit_inf_dc0, vector_fit_refined, En_Skimming, en_skimming, AddingSkimming, adding_skimming, Final_Skimming, final_skimming, MultiVectorFit, multivectorfit, MultiVectorFitPD
from .validation import check_hamiltonian, enforce_passivity, is_passive, check_data_passivity, enforce_data_passivity
from .io import read_touchstone, write_touchstone, read_csv
from .utils import plot_dum, show_dum, extract_subdum, convert_dum_units, \
    export_rfm_model, import_rfm_model, export_rfm_touchstone, \
    rfm_to_dum, rfm_response

# Define what gets imported with "from rfm import *"
__all__ = [
    # Core data structures
    'DUM',
    'Macromodel',

    # Fitting algorithms
    'VectorFit',
    'vector_fit',
    'VectorFit_inf_dc0',
    'vector_fit_inf_dc0',
    'vector_fit_refined',
    'En_Skimming',
    'en_skimming',
    'AddingSkimming',
    'adding_skimming',
    'Final_Skimming',
    'final_skimming',
    'MultiVectorFit',
    'multivectorfit',
    'MultiVectorFitPD',

    # Validation functions
    'check_hamiltonian',
    'enforce_passivity',
    'is_passive',
    'check_data_passivity',
    'enforce_data_passivity',

    # IO functions
    'read_touchstone',
    'write_touchstone',
    'read_csv',

    # Utility functions
    'plot_dum',
    'show_dum',
    'extract_subdum',
    'convert_dum_units',
    'export_rfm_model',
    'import_rfm_model',
    'export_rfm_touchstone',
    'rfm_to_dum',
    'rfm_response',
]

# Optional: GUI components (placeholder for future development)
# try:
#     from .gui import idem_setup
#     __all__.append('idem_setup')
# except ImportError:
#     pass  # GUI components not yet implemented
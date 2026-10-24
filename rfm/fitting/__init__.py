"""
Fitting package for the rfm RFM implementation.

Contains algorithms for model fitting including Vector Fitting and skimming methods.
"""
from .vectorfit import (
    VectorFit, vector_fit,
    VectorFit_inf_dc0, vector_fit_inf_dc0, vector_fit_refined,
    En_Skimming, en_skimming,
    AddingSkimming, adding_skimming,
    Final_Skimming, final_skimming,
    MultiVectorFit, multivectorfit, MultiVectorFitPD
)

__all__ = [
    # Vector Fitting algorithms
    'VectorFit',
    'vector_fit',
    'VectorFit_inf_dc0',
    'vector_fit_inf_dc0',
    'vector_fit_refined',

    # Skimming algorithms for model order reduction
    'En_Skimming',
    'en_skimming',
    'AddingSkimming',
    'adding_skimming',
    'Final_Skimming',
    'final_skimming',

    # Multi-instance Vector Fitting
    'MultiVectorFit',
    'multivectorfit',
    'MultiVectorFitPD'
]

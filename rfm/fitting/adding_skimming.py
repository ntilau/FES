"""
AddingSkimming - Skimming (balanced truncation) algorithm with incremental approach.

This function reduces the order of a macromodel using an incremental
balanced truncation approach where states are added one by one based on
their contribution to system energy.
"""

import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path
from ..macromodel import Macromodel
from ..dum import DUM
from scipy import linalg


def AddingSkimming(MOD: Macromodel,
                   ReduceOrd: int,
                   RootFigHandle=None,
                   *varargin) -> Macromodel:
    """
    AddingSkimming - Skimming (balanced truncation) algorithm with incremental approach.

    This function reduces the order of a macromodel using an incremental
    balanced truncation approach where states are added one by one based on
    their contribution to system energy.

    Args:
        MOD: Macromodel structure to reduce (should be in state-space form)
        ReduceOrd: Desired order of the reduced model
        RootFigHandle: Figure handle for GUI compatibility (unused in RFM implementation)
        *varargin: Additional arguments (kept for compatibility)

    Returns:
        Macromodel: Reduced macromodel
    """
    raise NotImplementedError(
        "AddingSkimming is a placeholder in the upstream RFM code; it is not implemented "
        "in the bundled package"
    )

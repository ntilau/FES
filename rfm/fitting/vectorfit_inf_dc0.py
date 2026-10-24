"""
VectorFit_inf_dc0 - Vector Fitting algorithm with infinity/DC constraints.

This class implements the Vector Fitting procedure with constraints:
- Enforces zero DC gain (s=0) when add_real_pole=False
- Enforces proper behavior at infinity
"""

import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
from scipy import linalg
from .vectorfit import VectorFit
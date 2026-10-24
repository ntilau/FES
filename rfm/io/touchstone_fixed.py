"""
TouchStone file format support for the rfm RFM implementation.

Provides functions to read and write TouchStone (.snp) files,
which are the standard format for RF/microwave component data.
Fixed version that properly parses S-parameter data.
"""

import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path

# Try to import DUM from the rfm package
try:
    from ..dum import DUM
except (ImportError, ValueError):
    # Fallback for when module is run directly or imported differently
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dum import DUM
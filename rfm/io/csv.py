"""
CSV file format support for the rfm RFM implementation.

Provides functions to read generic CSV files containing
multiport network data and convert to DUM structure.
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from pathlib import Path
import csv
from ..dum import DUM
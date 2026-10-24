"""
IO package for the rfm RFM implementation.

Contains functions for importing and exporting various file formats
used in the rfm toolkit.
"""
from .touchstone import read_touchstone, write_touchstone, read_csv

# Try to import other format readers, but don't fail if they're not available
try:
    from .ImportCSV import ImportCSV
except ImportError:
    ImportCSV = None

try:
    from .ImportCSDF import ImportCSDF
except ImportError:
    ImportCSDF = None

try:
    from .SpiceMOD2DUM import SpiceMOD2DUM
except ImportError:
    SpiceMOD2DUM = None

try:
    from .MOD2DUM import MOD2DUM
except ImportError:
    MOD2DUM = None

try:
    from .ResPoles2ABC import ResPoles2ABC
except ImportError:
    ResPoles2ABC = None

try:
    from .SS2ResPoles import SS2ResPoles
except ImportError:
    SS2ResPoles = None

try:
    from .MODFill import MODFill
except ImportError:
    MODFill = None

# TODO: Implement CST and FDTD imports
# try:
#     from .cst import read_cst
# except ImportError:
#     read_cst = None
# try:
#     from .fdtd import read_fdtd
# except ImportError:
#     read_fdtd = None

__all__ = [
    # TouchStone functions
    'read_touchstone',
    'write_touchstone',
    'read_csv',

    # CSV functions
    'ImportCSV',

    # CSDF functions
    'ImportCSDF',

    # SPICE functions
    'SpiceMOD2DUM',

    # Model conversion functions
    'MOD2DUM',
    'ResPoles2ABC',
    'SS2ResPoles',
    'MODFill',

    # TODO: Implement CST and FDTD imports
    # 'read_cst',
    # 'read_fdtd',
]

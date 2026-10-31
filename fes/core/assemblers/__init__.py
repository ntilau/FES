"""Formulation-specific assemblers ported from ``cpp/src/assemble_*.cpp``.

Each module exposes one ``assemble_*`` function with the same pipeline as the
reference C++: build the edge connectivity, number the DOFs, assemble the element
matrices, apply the boundary conditions, build the transfinite-element port
system, and solve.  The results are ordinary NumPy/SciPy objects so they can be
inspected and compared against the golden corpus.

The v0.1 C++ sources are frozen; the modules here are the replacement for them.
"""

from __future__ import annotations

from .em_av_fd import AVSystem, assemble_em_av_fd
from .em_e_fd import (
    FDPort,
    FDSystem,
    assemble_em_e_fd,
    compute_waveguide_modes,
)
from .em_e_fd_dd import assemble_em_e_fd_dd
from .em_e_fd_nl import assemble_em_e_fd_nl
from .em_e_qs import QSSystem, assemble_em_e_qs
from .em_e_tl_eig import TLEigSystem, assemble_em_e_tl_eig
from .em_ez_fd import TMzSystem, WavePort, assemble_em_ez_fd
from .errors import NullRightHandSideError

__all__ = [
    "AVSystem",
    "FDPort",
    "FDSystem",
    "NullRightHandSideError",
    "compute_waveguide_modes",
    "QSSystem",
    "TLEigSystem",
    "TMzSystem",
    "WavePort",
    "assemble_em_av_fd",
    "assemble_em_e_fd",
    "assemble_em_e_fd_dd",
    "assemble_em_e_fd_nl",
    "assemble_em_e_qs",
    "assemble_em_e_tl_eig",
    "assemble_em_ez_fd",
]

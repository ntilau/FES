"""Abort conditions shared by the assemblers.

The v0.1 solvers signal a handful of unrecoverable states by throwing a
message; ``main.cpp`` catches it and aborts.  The port uses exception classes
so the CLI can reproduce the reference's message and exit status without
swallowing unrelated ``RuntimeError``s.
"""

from __future__ import annotations

__all__ = ["NullRightHandSideError"]


class NullRightHandSideError(RuntimeError):
    """The original C++ ``### ERROR: Null Right Hand Side ###`` abort (exit 1).

    Raised when a model has neither waveports nor an incident plane wave
    (``em_e_fd``), or no boundary carries a voltage (``em_e_qs``).
    """

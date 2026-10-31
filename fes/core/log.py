"""Verbose solver event log mirroring the reference C++ trace.

Streams one line per stage, each with wall-clock time and resident memory,
exactly like ``cpp/src/main.cpp``'s ``++X s`` / ``+Memory: X MB resident``
trace.  It is standard-library only so the phase-1 CLI stays free of
NumPy/SciPy; the log is created lazily, only for commands that actually solve.

A :class:`SolverLog` is passed down into the drivers and assemblers as an
optional ``log`` argument that defaults to ``None``.  Every existing call site
and test therefore keeps working untouched, and the trace is emitted only when
a caller supplies a log.

Example (the reference trace shape)::

    # START: Sat Oct 31 20:38:05 2026
    COMPUTERNAME         = Apple-MacBook.local
    NUMBER_OF_PROCESSORS = 8
    OMP_NUM_THREADS      = ?
    Running with priority 0...
    Available memory: 16384 MB (total RAM)
    % Loading files:
    Project: WR90
    ...
    --- Frequency = 1e+10 ---
    % Assembly Symmetric:
    In solids: 0.0123693 s
    +Memory: 13.1719 MB resident
    On boundaries:
    \tPEC1: 0.000217458 s
    ...
"""

from __future__ import annotations

import os
import resource
import socket
import sys
import time
from datetime import datetime
from typing import List, Optional


def _resident_mb() -> float:
    """Resident (peak) set size in MiB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def now() -> float:
    """Monotonic wall clock for stage timers."""
    return time.perf_counter()


class SolverLog:
    """Accumulate and stream one solver run's timing/memory trace."""

    def __init__(self, project: str, p_ord: int = 1, h_ord: int = 0) -> None:
        self.project = project
        self.p_ord = p_ord
        self.h_ord = h_ord
        self.started = now()
        self.hostname = socket.gethostname()
        self.nproc = os.cpu_count() or 0
        self.total_mem_mb = self._total_memory_mb()
        self.lines: List[str] = []
        self._freq_start: Optional[float] = None

    def _total_memory_mb(self) -> float:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return 0.0
        return total / (1024.0 * 1024.0)

    # ── low level ──────────────────────────────────────────────────────

    def _emit(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def elapsed(self) -> float:
        return now() - self.started

    # ── header / footer ────────────────────────────────────────────────

    def header(self) -> None:
        self._emit(f"# START: {datetime.now():%a %b %d %H:%M:%S %Y}")
        self._emit(f"COMPUTERNAME         = {self.hostname}")
        self._emit(f"NUMBER_OF_PROCESSORS = {self.nproc}")
        self._emit("OMP_NUM_THREADS      = ?")
        self._emit("Running with priority 0...")
        self._emit(f"Available memory: {self.total_mem_mb:.0f} MB (total RAM)")

    def footer(self) -> None:
        self._emit(f"# END: {datetime.now():%a %b %d %H:%M:%S %Y}")

    # ── mesh loading ───────────────────────────────────────────────────

    def mesh_loaded(
        self,
        n_nodes: int,
        n_edges: int,
        n_faces: int,
        n_tetras: int,
        elapsed: float,
    ) -> None:
        self._emit("% Loading files:")
        self._emit(f"Project: {self.project}")
        self._emit("Homogeneous refinement: "
                   f"p = {self.p_ord}, h = {self.h_ord}")
        self._emit("Parsing TetGen based project file")
        self._emit(f"Nodes  = {n_nodes}")
        self._emit(f"Edges  = {n_edges}")
        self._emit(f"Faces  = {n_faces}")
        self._emit(f"Tetras = {n_tetras}")
        self._emit(f"++{elapsed:.6f} s")
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    # ── per frequency ──────────────────────────────────────────────────

    def frequency(self, freq: float) -> None:
        self._freq_start = now()
        self._emit(f"--- Frequency = {freq:g} ---")

    def assembly_solids(self, elapsed: float) -> None:
        self._emit("% Assembly Symmetric:")
        self._emit(f"In solids: {elapsed:.7f} s")
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    def boundaries_begin(self) -> None:
        self._emit("On boundaries:")

    def boundary(self, name: str, elapsed: float) -> None:
        self._emit(f"\t{name}: {elapsed:.7f} s")

    def boundaries_end(self) -> None:
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    def assembly_finish(self, dof: int, elapsed: float) -> None:
        self._emit("Finishing:")
        self._emit(f"\tSYS DoF = {dof},  {elapsed:.7f} s")

    def assembly_total(self, elapsed: float) -> None:
        self._emit(f"+{elapsed:.7f} s")
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    def assembly_and_solve(self, elapsed: float) -> None:
        """One combined assembly/solve stage for formulations the driver does
        not split into solids/boundaries/finishing."""
        self._emit("% Assembly and Solve:")
        self._emit(f"+{elapsed:.7f} s")
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    def solver_begin(self, mem_mb: float) -> None:
        self._emit("% Double precision Complex Solver  / Symmetric")
        self._emit(f"\tCommit memory: {mem_mb:.0f} MB")

    def solver_factor_solve(self, elapsed: float) -> None:
        self._emit(f"\tFactor and Solve: {elapsed:.7f} s")

    def solver_total(self, elapsed: float) -> None:
        self._emit(f"+{elapsed:.7f} s")
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    def saving_begin(self) -> None:
        self._emit("% Saving data:")

    def save_sparams(self, elapsed: float) -> None:
        self._emit(f"\tS parameters: {elapsed:.7f} s")

    def save_fields(self, elapsed: float) -> None:
        self._emit(f"\tFields: {elapsed:.7f} s")

    def saving_total(self, elapsed: float) -> None:
        self._emit(f"+{elapsed:.7f} s")
        self._emit(f"+Memory: {_resident_mb():.4f} MB resident")

    def frequency_total(self, elapsed: float) -> None:
        self._emit(f"++ {elapsed:.7f} s")

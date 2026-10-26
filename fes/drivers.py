"""Numerical drivers registered in :data:`fes.api.DRIVERS`.

Importing this module is cheap (no meshing, no assembly), but it does pull in
NumPy and SciPy.  The command-line front end therefore imports it lazily, only
for commands that actually solve.

The drivers mirror the ``main.cpp`` flow of the original C++ version:

1. read the ``.poly`` model and mesh it (Triangle),
2. assemble and solve the formulation,
3. hand the system to the post-processor, which appends one line per
   frequency to ``<project>.s<N>p`` (merging with frequencies already in the
   file, exactly like ``post_processor::save_data``).

``em_ez_fd`` (Phase 2), ``em_e_fd`` (Phase 6), ``em_e_tl_eig`` (Phase 7) and
``em_e_qs`` (Phase 11) have drivers; :func:`install` registers them and
:data:`fes.api.DRIVERS` remains the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import api
from .core.assemblers import (
    AVSystem,
    FDSystem,
    TLEigSystem,
    TMzSystem,
    assemble_em_av_fd,
    assemble_em_e_fd,
    assemble_em_e_fd_dd,
    assemble_em_e_fd_nl,
    assemble_em_e_qs,
    assemble_em_e_tl_eig,
    assemble_em_ez_fd,
)
from .core.assemblers.em_ez_fd import extract_solution_full
from .mesh.mesh2d import mesh_from_poly
from .mesh.mesh3d import mesh3d_from_poly
from .post.field import write_field_vtk
from .post.radiation import write_radiation_vtk
from .post.vtk import write_mesh_vtk

__all__ = [
    "SolverResult",
    "TouchstoneEntry",
    "install",
    "merge_touchstone",
    "read_touchstone",
    "run_em_av_fd",
    "run_em_e_fd",
    "run_em_e_fd_dd",
    "run_em_e_fd_nl",
    "run_em_e_qs",
    "run_em_e_tl_eig",
    "run_em_ez_fd",
    "write_touchstone",
]


# ── Touchstone I/O (post_processor::save_data) ──────────────────────────────


@dataclass
class TouchstoneEntry:
    """One frequency row of a ``.sNp`` file."""

    freq_ghz: float
    values: List[complex]


def _read_entry(line: str) -> Optional[TouchstoneEntry]:
    tokens = line.split()
    if not tokens:
        return None
    try:
        freq_ghz = float(tokens[0])
        numbers = [float(token) for token in tokens[1:]]
    except ValueError:
        return None
    if not numbers or len(numbers) % 2:
        return None
    values = [
        complex(mag * np.cos(np.radians(ang)), mag * np.sin(np.radians(ang)))
        for mag, ang in zip(numbers[0::2], numbers[1::2])
    ]
    return TouchstoneEntry(freq_ghz=freq_ghz, values=values)


def read_touchstone(path: str | Path) -> Dict[float, List[complex]]:
    """Return ``{frequency_GHz: [S values, column-major]}`` from *path*.

    Unknown or malformed lines are skipped, matching the reference C++'s
    ``std::istringstream``-per-line merge loop.
    """
    data: Dict[float, List[complex]] = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return data
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "!#":
            continue
        entry = _read_entry(line)
        if entry is not None:
            data[entry.freq_ghz] = entry.values
    return data


def _format_row(freq_ghz: float, values: Sequence[complex]) -> str:
    # ``std::setw(10) << std::scientific << std::left << std::setprecision(8)``
    parts = [f"{freq_ghz:<10.8e}"]
    for value in values:
        mag = abs(value)
        ang = np.degrees(np.angle(value))
        parts.append(f" {mag:<14.8e} {ang:<14.8e}")
    return "".join(parts)


def write_touchstone(
    path: str | Path, name: str, entries: Dict[float, Sequence[complex]]
) -> Path:
    """Write *entries* sorted by frequency in the reference format."""
    target = Path(path)
    lines = ["! FES S-parameters", f"! {name}", "# GHz S MA R 50"]
    for freq_ghz in sorted(entries):
        lines.append(_format_row(freq_ghz, entries[freq_ghz]))
    target.write_text("\n".join(lines) + "\n")
    return target


def merge_touchstone(
    path: str | Path,
    name: str,
    freq_ghz: float,
    values: Sequence[complex],
) -> Path:
    """Insert one row into *path*, replacing any existing row at *freq_ghz*."""
    entries = read_touchstone(path)
    entries[float(freq_ghz)] = list(values)
    return write_touchstone(path, name, entries)


# ── results ─────────────────────────────────────────────────────────────────


@dataclass
class SolverResult:
    """What a driver returns; ``str()`` is the one-screen summary."""

    project: str
    assembly: str
    frequencies: List[float] = field(default_factory=list)
    systems: List["TMzSystem | FDSystem | TLEigSystem"] = field(
        default_factory=list
    )
    sparam_files: List[Path] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_modes(self) -> int:
        """S-parameter count: ``n_modes`` in 2D, ``wave_ports_num`` in 3D."""
        if not self.systems:
            return 0
        system = self.systems[0]
        if isinstance(system, FDSystem):
            return int(system.wave_ports_num)
        if isinstance(system, AVSystem):
            return int(system.wave_ports_num)
        return int(system.n_modes)

    @property
    def dof(self) -> int:
        if not self.systems:
            return 0
        system = self.systems[0]
        if isinstance(system, FDSystem):
            return int(system.dofnum)
        if isinstance(system, AVSystem):
            return int(system.dofnum)
        return int(system.dof)

    @property
    def eigen_systems(self) -> List[TLEigSystem]:
        return [s for s in self.systems if isinstance(s, TLEigSystem)]

    def summary(self) -> str:
        lines = [
            f"{self.project}: {self.assembly} "
            f"{len(self.frequencies)} frequency(ies), {self.n_modes} mode(s), "
            f"{self.dof} DOF (p={self.systems[0].order})"
            if self.systems
            else f"{self.project}: {self.assembly} (no system)",
        ]
        for system in self.systems:
            if isinstance(system, TLEigSystem):
                lines.append(f"  f = {system.freq:.8g} Hz")
                lines.append(
                    f"  {system.n_modes} mode(s) from "
                    f"{system.n_edge_dofs} edge DOFs "
                    f"({system.free_edges.size} free), "
                    f"{system.n_node_dofs} node DOFs"
                )
                for index, (beta, alpha, kc) in enumerate(
                    zip(system.beta, system.alpha, system.kc), start=1
                ):
                    # ``assemble_em_e_tl_eig.cpp`` flags propagating modes.
                    flag = " p" if beta > system.k0 * 1e-6 else ""
                    lines.append(
                        f"    {index}: beta={beta:.6g} alpha={alpha:.6g} "
                        f"kc={kc:.6g}{flag}"
                    )
                continue
            if system.S is None:
                continue
            lines.append(f"  f = {system.freq:.8g} Hz")
            lines.append(f"  |S| =\n{20 * np.log10(np.abs(system.S))}")
        for path in self.sparam_files:
            lines.append(f"  wrote {path}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.summary()


# ── drivers ─────────────────────────────────────────────────────────────────


def _mesh_switches(plan: api.RunPlan) -> Optional[str]:
    return plan.options.poly_cmd or None


def _solve_system(
    mesh,
    freq: float,
    plan: api.RunPlan,
) -> TMzSystem:
    return assemble_em_ez_fd(
        mesh,
        freq,
        plan.options.p_ord,
        power=plan.options.power,
    )


def _new_result(plan: api.RunPlan) -> SolverResult:
    """Create the per-run result with its resolved frequency list."""
    return SolverResult(
        project=plan.project,
        assembly=plan.assembly,
        frequencies=list(plan.frequencies or [plan.options.freq]),
    )


def _prepare_output(plan: api.RunPlan, mesh) -> Path:
    """Create the output directory and write the mesh VTK before assembly."""
    output_dir = Path(plan.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The reference C++ writes the mesh before assembly, so 2D files carry the PLC
    # edges only (``Savefield`` predates the connectivity build).
    write_mesh_vtk(mesh, output_dir / plan.project)
    return output_dir


def _write_sparams(
    result: SolverResult,
    plan: api.RunPlan,
    output_dir: Path,
    freq: float,
    system,
    n_ports: int,
) -> None:
    """Append one row per frequency to ``<project>.s<N>p``."""
    if system.S is None or not plan.options.sparam or n_ports == 0:
        return
    path = output_dir / f"{plan.project}.s{n_ports}p"
    # Column-major, exactly like the reference C++.
    values = [system.S[j, i] for i in range(n_ports) for j in range(n_ports)]
    result.sparam_files.append(
        merge_touchstone(path, plan.project, freq / 1e9, values)
    )


def run_em_ez_fd(plan: api.RunPlan) -> SolverResult:
    """Driver for the 2D TMz scalar Helmholtz formulation."""
    options = plan.options
    result = _new_result(plan)

    mesh = mesh_from_poly(
        plan.model.path, _mesh_switches(plan), h_ord=options.h_ord
    )

    output_dir = _prepare_output(plan, mesh)
    for freq in result.frequencies:
        system = _solve_system(mesh, freq, plan)
        result.systems.append(system)
        if options.field:
            write_field_vtk(
                mesh,
                extract_solution_full(system),
                freq,
                output_dir / plan.project,
                assembly=plan.assembly,
                order=options.p_ord,
            )
        if options.rad:
            # ``rad`` is a 3D post-processor, but the reference C++ runs it for every
            # formulation; a 2D mesh simply has no radiation faces.
            write_radiation_vtk(
                mesh,
                extract_solution_full(system),
                freq,
                output_dir / plan.project,
                order=options.p_ord,
                n_theta=options.n_theta,
                n_phi=options.n_phi,
            )
        _write_sparams(result, plan, output_dir, freq, system, system.n_modes)

    return result


def run_em_av_fd(plan: api.RunPlan) -> SolverResult:
    """Driver for the 3D H(curl) x H(grad) A-V formulation.

    Like :func:`run_em_e_fd` but assembles the mixed vector-potential /
    scalar-potential system with the gauge-fixing projector.  The reference
    C++ re-meshes the PLC with TetGen on every run and never caches the
    result, so the mesh is generated here and reused for the whole frequency
    list.
    """
    options = plan.options
    result = _new_result(plan)

    mesh = mesh3d_from_poly(plan.model.path, _mesh_switches(plan))

    output_dir = _prepare_output(plan, mesh)
    for freq in result.frequencies:
        system = assemble_em_av_fd(
            mesh,
            freq,
            options.p_ord,
            power=options.power,
            e_inc=options.E if options.einc else None,
            k_inc=options.k if options.einc else None,
        )
        result.systems.append(system)
        if options.field:
            write_field_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                assembly=plan.assembly,
                order=options.p_ord,
                dofnum_s=system.dofnum_s,
            )
        if options.rad:
            write_radiation_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                order=options.p_ord,
                n_theta=options.n_theta,
                n_phi=options.n_phi,
                e_inc=options.E if options.einc else None,
            )
        n_ports = int(system.wave_ports_num)
        _write_sparams(result, plan, output_dir, freq, system, n_ports)

    return result


def run_em_e_fd(plan: api.RunPlan) -> SolverResult:
    """Driver for the 3D H(curl) curl-curl formulation.

    The reference C++ re-meshes the PLC with TetGen on every run and never caches the
    result, so unlike the 2D driver the mesh is generated here and reused for
    the whole frequency list.  The port bookkeeping (propagation constants and
    the shift) is printed only to stdout.
    """
    options = plan.options
    result = _new_result(plan)

    mesh = mesh3d_from_poly(plan.model.path, _mesh_switches(plan))

    output_dir = _prepare_output(plan, mesh)
    for freq in result.frequencies:
        system = assemble_em_e_fd(
            mesh,
            freq,
            options.p_ord,
            power=options.power,
            e_inc=options.E if options.einc else None,
            k_inc=options.k if options.einc else None,
        )
        result.systems.append(system)
        if options.field:
            write_field_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                assembly=plan.assembly,
                order=options.p_ord,
            )
        if options.rad:
            write_radiation_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                order=options.p_ord,
                n_theta=options.n_theta,
                n_phi=options.n_phi,
                e_inc=options.E if options.einc else None,
            )
        n_ports = int(system.wave_ports_num)
        _write_sparams(result, plan, output_dir, freq, system, n_ports)

    return result


def run_em_e_fd_dd(plan: api.RunPlan) -> SolverResult:
    """Driver for the 3D H(curl) curl-curl formulation with domain decomposition.

    Like :func:`run_em_e_fd` but solves the reduced system via Schur-complement
    domain decomposition, partitioning the mesh into *n_dd* subdomains and
    iterating on the interface with Jacobi or Gauss-Seidel relaxation.
    """
    options = plan.options
    result = _new_result(plan)

    if options.nl:
        result.notes.append("+nl harmonic balance is not ported yet (Phase 10)")

    mesh = mesh3d_from_poly(plan.model.path, _mesh_switches(plan))

    output_dir = _prepare_output(plan, mesh)
    for freq in result.frequencies:
        system = assemble_em_e_fd_dd(
            mesh,
            freq,
            options.p_ord,
            power=options.power,
            e_inc=options.E if options.einc else None,
            k_inc=options.k if options.einc else None,
            n_dd=options.n_dd or 1,
            preconditioner=options.dd_preconditioner or "gs",
        )
        result.systems.append(system)
        if options.field:
            write_field_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                assembly=plan.assembly,
                order=options.p_ord,
            )
        if options.rad:
            write_radiation_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                order=options.p_ord,
                n_theta=options.n_theta,
                n_phi=options.n_phi,
                e_inc=options.E if options.einc else None,
            )
        n_ports = int(system.wave_ports_num)
        _write_sparams(result, plan, output_dir, freq, system, n_ports)

    return result


def run_em_e_fd_nl(plan: api.RunPlan) -> SolverResult:
    """Driver for the 3D H(curl) curl-curl formulation with Kerr nonlinearity.

    Like :func:`run_em_e_fd` but solves the nonlinear system via fixed-point
    iteration on the Kerr-shifted permittivity.  The ``+nl`` flag selects the
    nonlinear material, number of harmonics, Kerr coefficient and relaxation
    factor.
    """
    options = plan.options
    result = _new_result(plan)

    mesh = mesh3d_from_poly(plan.model.path, _mesh_switches(plan))

    output_dir = _prepare_output(plan, mesh)
    for freq in result.frequencies:
        system = assemble_em_e_fd_nl(
            mesh,
            freq,
            options.p_ord,
            power=options.power,
            e_inc=options.E if options.einc else None,
            k_inc=options.k if options.einc else None,
            n_harm=options.n_harm,
            nl_mtrl_name=options.nl_mtrl_name,
            kerr=options.kerr,
            relax=options.relax,
        )
        result.systems.append(system)
        if options.field:
            write_field_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                assembly=plan.assembly,
                order=options.p_ord,
            )
        if options.rad:
            write_radiation_vtk(
                mesh,
                system.Sol,
                freq,
                output_dir / plan.project,
                order=options.p_ord,
                n_theta=options.n_theta,
                n_phi=options.n_phi,
                e_inc=options.E if options.einc else None,
            )
        n_ports = int(system.wave_ports_num)
        _write_sparams(result, plan, output_dir, freq, system, n_ports)

    return result


def run_em_e_tl_eig(plan: api.RunPlan) -> SolverResult:
    """Driver for the 2D transmission-line cross-section eigenproblem.

    The reference C++ meshes once (the project constructor) and runs the cross-section
    analysis once per frequency; ``equation_system`` skips solver and
    post-processing for this formulation, so the only file written next to the
    model is the mesh ``.vtk`` from the project constructor.
    """
    options = plan.options
    result = _new_result(plan)

    mesh = mesh_from_poly(
        plan.model.path, _mesh_switches(plan), h_ord=options.h_ord
    )
    _prepare_output(plan, mesh)
    for freq in result.frequencies:
        result.systems.append(
            assemble_em_e_tl_eig(mesh, freq, options.p_ord)
        )
    return result


def run_em_e_qs(plan: api.RunPlan) -> SolverResult:
    """Driver for 2D electrostatics (nodal P1, ``+volt`` Dirichlet values).

    The reference C++ assembles once per frequency, solves the non-symmetric system,
    and writes no Touchstone file (``wave_ports_num = 0``).  The field export
    is enabled by ``#Formula em_e_qs`` itself: ``detectFormula`` sets
    ``opt->field`` when the tag comes from the model file, which
    :func:`fes.api.plan_run` mirrors.
    """
    options = plan.options
    result = _new_result(plan)

    mesh = mesh_from_poly(
        plan.model.path, _mesh_switches(plan), h_ord=options.h_ord
    )
    output_dir = _prepare_output(plan, mesh)
    for freq in result.frequencies:
        system = assemble_em_e_qs(
            mesh,
            options.p_ord,
            options.Vbnd,
            freq=freq,
        )
        result.systems.append(system)
        if options.field:
            write_field_vtk(
                mesh,
                system.phi[:, None] if system.phi is not None else np.zeros((system.dof, 1)),
                freq,
                output_dir / plan.project,
                assembly=plan.assembly,
                order=options.p_ord,
            )
    return result


def install() -> None:
    """Register the available drivers in :data:`fes.api.DRIVERS`."""
    api.DRIVERS.setdefault("em_av_fd", run_em_av_fd)
    api.DRIVERS.setdefault("em_ez_fd", run_em_ez_fd)
    api.DRIVERS.setdefault("em_e_fd", run_em_e_fd)
    api.DRIVERS.setdefault("em_e_fd_dd", run_em_e_fd_dd)
    api.DRIVERS.setdefault("em_e_fd_nl", run_em_e_fd_nl)
    api.DRIVERS.setdefault("em_e_tl_eig", run_em_e_tl_eig)
    api.DRIVERS.setdefault("em_e_qs", run_em_e_qs)


install()

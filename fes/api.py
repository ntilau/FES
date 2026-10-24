"""High-level programmatic API.

Phase 1 provides model discovery, option resolution, and run planning.  The
numerical drivers are added phase by phase; until a formulation's driver lands,
:func:`run` raises :class:`NotPortedError` instead of silently doing something
different from the original C++ binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .core.options import (
    ASSEMBLY_TYPES,
    Options,
    UsageError,
    normalize_assembly,
    parse_argv,
)
from .mesh.poly_header import ModelInfo, read_poly_metadata
from .paths import model_path

#: Formulation → phase that supplies its Python driver.
FORMULATION_PHASE = {
    "em_ez_fd": 2,
    "em_e_fd": 6,
    "em_e_tl_eig": 7,
    "em_e_fd_dd": 9,
    "em_e_fd_nl": 10,
    "em_e_qs": 11,
}

#: Formulations the v0.1 option parser accepts but never dispatches.
#: ``equation_system.cpp`` has no case for them and throws
#: ``Formulation not available for this version``; there is nothing to port.
UNAVAILABLE_FORMULATIONS = frozenset({"em_h_qs"})


class NotPortedError(RuntimeError):
    """Raised when a model's formulation has no Python driver yet."""

    def __init__(self, plan: "RunPlan") -> None:
        self.plan = plan
        phase = FORMULATION_PHASE.get(plan.assembly, 15)
        super().__init__(
            f"formulation {plan.assembly!r} for {plan.project!r} is not ported yet "
            f"(planned for Phase {phase}); use --dry-run to inspect the resolved plan"
        )
        self.phase = phase


class FormulationUnavailableError(RuntimeError):
    """Raised for a formulation the original C++ version does not support."""

    def __init__(self, assembly: str) -> None:
        self.assembly = assembly
        super().__init__(
            f"Formulation not available for this version: {assembly!r}"
        )


def ensure_available(plan: "RunPlan") -> None:
    """Reject formulations the original C++ version cannot assemble."""
    if plan.assembly in UNAVAILABLE_FORMULATIONS:
        raise FormulationUnavailableError(plan.assembly)


@dataclass
class RunPlan:
    """Everything needed to execute (or compare) one solver run."""

    project: str
    model: ModelInfo
    options: Options
    assembly: str
    command: str = "run"
    frequencies: Optional[List[float]] = None
    output_dir: str = "."

    def __post_init__(self) -> None:
        if self.frequencies is None:
            self.frequencies = self.options.freqs()

    @property
    def dimension(self) -> int:
        return self.model.dimension

    @property
    def frequency(self) -> float:
        return self.options.freq

    def as_dict(self) -> dict:
        return {
            "project": self.project,
            "model": self.model.as_dict(),
            "assembly": self.assembly,
            "command": self.command,
            "frequencies": list(self.frequencies or []),
            "output_dir": self.output_dir,
            "options": self.options.as_dict(),
        }


def load_model(name: str, data: Optional[str | Path] = None) -> ModelInfo:
    """Resolve and read a model's metadata."""
    return read_poly_metadata(model_path(name, data))


def formula_is_explicit(options: Options) -> bool:
    """True when the CLI selected a formulation, so ``#Formula`` is ignored."""
    if getattr(options, "formula_explicit", False):
        return True
    keys = options.cli_override
    if "formula" in keys or "assembly" in keys:
        return True
    return any(name in keys for name in ASSEMBLY_TYPES)


def effective_assembly(options: Options, model: ModelInfo) -> str:
    """Resolve the formulation: explicit CLI flag, else the ``#Formula`` tag."""
    if formula_is_explicit(options):
        return options.assembly
    if model.formula:
        return normalize_assembly(model.formula)
    return options.assembly


def _validate(plan: RunPlan) -> None:
    if plan.options.p_ord < 1 or plan.options.p_ord > 4:
        raise UsageError(f"polynomial order must be 1..4, got {plan.options.p_ord}")
    if plan.model.dimension not in (2, 3):
        raise UsageError(
            f"could not determine the spatial dimension of {plan.model.path}"
        )
    if plan.command == "mesh":
        return
    if plan.assembly == "em_ez_fd" and plan.model.dimension != 2:
        raise UsageError("em_ez_fd requires a 2D .poly model")
    if plan.assembly == "em_e_qs" and plan.options.freq != 0.0:
        raise UsageError("em_e_qs requires +f 0")


def plan_run(
    project: str | None = None,
    options: Options | None = None,
    argv: Sequence[str] | None = None,
    command: str = "run",
    data: Optional[str | Path] = None,
) -> RunPlan:
    """Resolve a model and options into a :class:`RunPlan` without solving."""
    if options is None:
        options = parse_argv(argv or [])
    if project is not None:
        options.name = project
    if not options.name:
        raise UsageError("Project name is required")

    model = load_model(options.name, data=data)
    assembly = effective_assembly(options, model)
    # ``preprocessing::preprocessing`` calls ``detectFormula`` only while the
    # CLI still holds the default ``em_e_fd`` (an explicit ``+formula`` skips
    # it).  That function does more than pick the formulation: a
    # ``#Formula em_e_qs`` tag also forces ``opt->field``, which is why the
    # original C++ binary writes ``<model>_<freq>.vtk`` for the electrostatic models.
    if (
        not formula_is_explicit(options)
        and model.formula is not None
        and normalize_assembly(model.formula) == "em_e_qs"
    ):
        options.field = True
    plan = RunPlan(
        project=model.name,
        model=model,
        options=options,
        assembly=assembly,
        command=command,
    )
    _validate(plan)
    return plan


#: Filled in by later phases: assembly name → callable(plan) → result.
DRIVERS: Dict[str, Callable[[RunPlan], object]] = {}


def load_drivers() -> None:
    """Populate :data:`DRIVERS` by importing :mod:`fes.drivers`.

    Kept lazy so that ``fes info``/``fes models``/``--dry-run`` work without
    NumPy, SciPy, or a mesher installed.
    """
    from . import drivers  # noqa: F401  (registers the drivers on import)

    drivers.install()


def dispatch(plan: RunPlan) -> object:
    """Execute *plan* with its registered driver, importing drivers lazily."""
    ensure_available(plan)
    load_drivers()
    driver = DRIVERS.get(plan.assembly)
    if driver is None:
        raise NotPortedError(plan)
    return driver(plan)


def run(
    project: str | None = None,
    options: Options | None = None,
    argv: Sequence[str] | None = None,
    dry_run: bool = False,
    data: Optional[str | Path] = None,
) -> RunPlan | object:
    """Plan and (once ported) execute a solver run."""
    plan = plan_run(project, options=options, argv=argv, command="run", data=data)
    if dry_run:
        return plan
    return dispatch(plan)


def sweep(
    project: str | None = None,
    options: Options | None = None,
    argv: Sequence[str] | None = None,
    dry_run: bool = False,
    data: Optional[str | Path] = None,
) -> RunPlan:
    """Plan a frequency sweep (``+fr lf hf n`` or ``+l_freq/+h_freq/+n_freqs``)."""
    plan = plan_run(project, options=options, argv=argv, command="sweep", data=data)
    if plan.options.n_freqs <= 1:
        raise UsageError("sweep requires +fr <lf> <hf> <n> with n >= 2")
    if dry_run:
        return plan
    return dispatch(plan)


def info(name: str, data: Optional[str | Path] = None) -> ModelInfo:
    """Return metadata for one model."""
    return load_model(name, data=data)


def mesh(
    project: str | None = None,
    options: Options | None = None,
    argv: Sequence[str] | None = None,
    data: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    write: bool = True,
    dry_run: bool = False,
) -> "RunPlan | dict":
    """Mesh a model from its ``.poly`` file.

    A 2D model is meshed with Triangle and the result is written as
    ``<model>.npz`` next to the ``.poly`` file (or into *out_dir*),
    which is where :func:`fes.mesh.io_poly.read_poly` looks for it.  ``+poly_cmd``
    selects the mesher switches; the default ``pq34AfeQ`` is the one the v0.1
    C++ binary used.

    A 3D model is meshed with TetGen, which the original C++ binary runs on every solve and
    never caches, so no file is written and ``write`` is ignored.  TetGen is an
    optional shared library; when it is missing this raises :class:`UsageError`
    pointing at ``tools/build_tetgen.py``.
    """
    plan = plan_run(
        project, options=options, argv=argv, command="mesh", data=data
    )
    if dry_run:
        return plan
    if plan.options.h_ord:
        # The cache format carries its refinement level in the file name, and
        # ``mesh_arrays_from_poly`` writes the unrefined h1 arrays.  Solver
        # drivers refine at run time instead, like ``project::project``.
        raise UsageError(
            "`fes mesh` writes the unrefined h1 cache; pass +h to `fes run`/`fes sweep`"
        )

    switches = plan.options.poly_cmd or None

    if plan.model.dimension != 2:
        from .mesh.mesh3d import mesh3d_from_poly
        from .mesh.tetgen_backend import SWITCH_DEFAULT_3D, TetGenUnavailable

        try:
            mesh = mesh3d_from_poly(plan.model.path, switches=switches)
        except TetGenUnavailable as exc:
            raise UsageError(str(exc)) from exc
        return {
            "model": plan.model.name,
            "poly": str(plan.model.path),
            "dimension": 3,
            "switches": switches if switches else SWITCH_DEFAULT_3D,
            "n_nodes": mesh.n_nodes,
            "n_faces": mesh.n_faces,
            "n_edges": mesh.n_edges,
            "n_tetras": mesh.n_tetras,
            "cache": None,
        }

    from .mesh.topology import mesh_arrays_from_poly

    arrays = mesh_arrays_from_poly(plan.model.path, switches=switches)

    target = None
    if write:
        from .mesh.io_mat import write_cache

        directory = Path(out_dir) if out_dir else Path(plan.model.path).parent
        target = write_cache(
            directory / f"{plan.model.name}.npz", **arrays
        )

    return {
        "model": plan.model.name,
        "poly": str(plan.model.path),
        "dimension": 2,
        "switches": switches or "pq34AfeQ",
        "n_nodes": int(arrays["node"].shape[0]),
        "n_faces": int(arrays["ele"].shape[0]),
        "n_edges": int(arrays["spig2"].shape[0]),
        "n_tetras": 0,
        "cache": str(target) if target is not None else None,
    }


__all__ = [
    "DRIVERS",
    "FORMULATION_PHASE",
    "UNAVAILABLE_FORMULATIONS",
    "FormulationUnavailableError",
    "NotPortedError",
    "RunPlan",
    "effective_assembly",
    "ensure_available",
    "formula_is_explicit",
    "info",
    "load_drivers",
    "load_model",
    "mesh",
    "plan_run",
    "run",
    "sweep",
]

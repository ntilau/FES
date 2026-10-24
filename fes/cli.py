"""Command-line interface.

The legacy invocation is preserved::

    fes WR90 +f 1e10 +p 2 +poly AafeeQ

Subcommands add discoverable entry points::

    fes run WR90 +f 1e10 +p 2
    fes sweep WR90 +fr 8e9 12e9 5
    fes info WR90 --json
    fes models
    fes version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import __version__
from .api import (
    FormulationUnavailableError,
    NotPortedError,
    RunPlan,
    dispatch,
    info,
    plan_run,
)
from .core.options import Options, UsageError, parse_argv
from .paths import data_dir, list_models

COMMANDS = ("run", "sweep", "mesh", "sparam", "eigen", "info", "models", "version")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NOT_PORTED = 2
EXIT_IO = 3

_SWITCHES_WITH_VALUE = ("--out", "--data")


def _extract_switches(argv: Sequence[str]) -> Tuple[List[str], Dict[str, object]]:
    """Pull command-level switches out of *argv*, leaving solver flags."""
    rest: List[str] = []
    switches: Dict[str, object] = {}
    index = 0
    tokens = list(argv)
    while index < len(tokens):
        token = tokens[index]
        if token in (
            "--json",
            "--dry-run",
            "--legacy",
            "--no-write",
            "--remesh",
            "--help",
            "-h",
        ):
            switches[token.lstrip("-").replace("-", "_")] = True
            index += 1
            continue
        if token in ("--version", "-V"):
            switches["version"] = True
            index += 1
            continue
        matched = False
        for name in _SWITCHES_WITH_VALUE:
            if token == name:
                if index + 1 >= len(tokens):
                    raise UsageError(f"{name} requires a value")
                switches[name[2:]] = tokens[index + 1]
                index += 2
                matched = True
                break
            if token.startswith(name + "="):
                switches[name[2:]] = token.split("=", 1)[1]
                index += 1
                matched = True
                break
        if matched:
            continue
        rest.append(token)
        index += 1
    return rest, switches


def _json_dump(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=False, default=str))


def _error(message: str, code: int = EXIT_USAGE) -> int:
    print(f"fes: {message}", file=sys.stderr)
    return code


def _is_null_rhs(exc: RuntimeError) -> bool:
    """True for the assemblers' ``Null Right Hand Side`` abort.

    Imported lazily: the command front end must not pull NumPy/SciPy in until
    a command actually solves.
    """
    from .core.assemblers.errors import NullRightHandSideError

    return isinstance(exc, NullRightHandSideError)


def _null_rhs_abort(exc: RuntimeError) -> int:
    """Reproduce ``main.cpp``'s abort for an empty right-hand side."""
    print(f"### ERROR: {exc} ###")
    return EXIT_USAGE


def help_text() -> str:
    return f"""fes {__version__} — finite element solver for electromagnetics

Usage:
  fes <project> [frequency] [+flags]        legacy C++ invocation
  fes run <project> [+flags]                run one frequency
  fes sweep <project> +fr LF HF N           run a frequency sweep
  fes mesh <project> [+poly_cmd ARGS]       mesh only
  fes sparam <project> +f FREQ              extract S-parameters
  fes eigen <project> [+n_modes N]          eigenmode analysis
  fes info <project> [--json]               print model metadata
  fes models [--json]               list models in data/
  fes version [--json]                      print version information

Common flags:
  +f FREQ              frequency in Hz (0 selects electrostatics)
  +fr LF HF N          discrete frequency sweep
  +formula NAME        em_e_fd | em_ez_fd | em_e_tl_eig | em_e_qs |
                       em_e_fd_dd | em_e_fd_nl | em_h_qs
  +p N                 polynomial order (1-4)
  +h N                 uniform mesh refinement level
  +poly [CMD]          load .poly and mesh; CMD is passed to the mesher
  +poly_cmd CMD        mesher switches (default: qAfeeQ for TetGen)
  +volt BND V          voltage on a PerfectE boundary (repeatable)
  +direct              direct sparse solver (default)
  +gmres TOL [RESTART]  iterative solver
  +dd N                domain decomposition into N subdomains
  +nl H MAT KERR RELAX Kerr nonlinear harmonic balance
  +einc Ex,Ey,Ez,kx,ky,kz  incident plane wave
  +field               write VTK field data
  +rad NT NP            write a radiation pattern
  -sparam              disable S-parameter output
  -tfe                 disable transfinite port elements
  --dry-run            resolve and print the run plan without solving
  --out DIR            output directory (mesh cache for `fes mesh`)
  --no-write           `fes mesh`: report counts without writing a cache
  --remesh             `fes mesh`: accepted; meshing always restarts from .poly

Exit codes: 0 success, 1 usage error, 2 formulation not yet ported, 3 I/O error.
"""


# ── commands ────────────────────────────────────────────────────────────────


def _cmd_version(switches: Dict[str, object]) -> int:
    if switches.get("json"):
        import platform

        payload = {
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        for module in ("numpy", "scipy", "matplotlib"):
            try:
                payload[module] = __import__(module).__version__
            except Exception:
                payload[module] = None
        _json_dump(payload)
    else:
        print(f"fes {__version__}")
    return EXIT_OK


def _cmd_models(switches: Dict[str, object]) -> int:
    names = list_models()
    if switches.get("json"):
        rows = []
        for name in names:
            try:
                model = info(name)
                rows.append(
                    {
                        "name": name,
                        "dimension": model.dimension,
                        "formula": model.formula,
                        "n_nodes": model.n_nodes,
                        "n_boundaries": len(model.boundaries),
                    }
                )
            except Exception as exc:  # pragma: no cover - corrupt model
                rows.append({"name": name, "error": str(exc)})
        _json_dump({"data_dir": str(data_dir()), "models": rows})
    else:
        for name in names:
            print(name)
    return EXIT_OK


def _cmd_info(rest: Sequence[str], switches: Dict[str, object]) -> int:
    if not rest:
        return _error("info requires a model name")
    rows = []
    for name in rest:
        try:
            rows.append(info(name))
        except FileNotFoundError as exc:
            return _error(str(exc), EXIT_IO)
    if switches.get("json"):
        payload = rows[0].as_dict() if len(rows) == 1 else [r.as_dict() for r in rows]
        _json_dump(payload)
        return EXIT_OK

    for model in rows:
        dim = {2: "2-D", 3: "3-D"}.get(model.dimension, "?")
        print(f"{model.name}  ({dim})")
        print(f"  path:      {model.path}")
        print(f"  formula:   {model.formula or '(none; CLI/default decides)'}")
        counts = [f"nodes {model.n_nodes}"]
        if model.n_segments:
            counts.append(f"segments {model.n_segments}")
        if model.n_faces:
            counts.append(f"facets {model.n_faces}")
        counts.extend([f"holes {model.n_holes}", f"regions {model.n_regions}"])
        print("  counts:    " + ", ".join(counts))
        if model.bbox:
            low, high = model.bbox
            print("  bbox:      " + ", ".join(f"{v:.6g}" for v in low)
                  + " .. " + ", ".join(f"{v:.6g}" for v in high))
        if model.solids:
            print("  materials:")
            for solid in model.solids:
                print(
                    f"    {solid.name:<16} epsr={solid.epsr:g} mur={solid.mur:g} "
                    f"sigma={solid.sigma:g} tand={solid.tand:g} ({solid.material})"
                )
        if model.boundaries:
            print("  boundaries:")
            for boundary in model.boundaries:
                modes = (
                    f" modes={boundary.num_modes}"
                    if boundary.type in ("WavePort", "LumpedPort")
                    else ""
                )
                print(
                    f"    {boundary.name:<16} label={boundary.label:<4} "
                    f"{boundary.type}{modes}"
                )
    return EXIT_OK


def _print_plan(plan: RunPlan) -> None:
    opts = plan.options
    source = "CLI" if opts.formula_explicit else (
        "#Formula" if plan.model.formula else "default"
    )
    print(f"project:    {plan.project}")
    print(f"model:      {plan.model.path}")
    print(f"dimension:  {plan.model.dimension}")
    print(f"assembly:   {plan.assembly}  ({source})")
    print(f"command:    {plan.command}")
    if plan.command == "sweep":
        freqs = plan.frequencies or []
        print(f"frequencies: {len(freqs)} points, {freqs[0]:.6g} .. {freqs[-1]:.6g} Hz")
    else:
        print(f"frequency:  {opts.freq:.6g} Hz")
    print(f"solver:     {opts.solver}")
    if opts.solver == "gmres":
        restart = f", restart {opts.gmres_restart}" if opts.gmres_restart else ""
        print(f"gmres:      tol {opts.toll:g}{restart}")
    print(f"order:      p={opts.p_ord}, h={opts.h_ord}")
    if opts.n_dd:
        print(f"dd:         {opts.n_dd} subdomains ({opts.dd_preconditioner})")
    if opts.nl:
        print(
            f"nonlinear:  {opts.n_harm} harmonics, material {opts.nl_mtrl_name}, "
            f"kerr {opts.kerr:g}, relax {opts.relax:g}"
        )
    if opts.Vbnd:
        values = ", ".join(f"{k}={v:g}" for k, v in opts.Vbnd.items())
        print(f"voltages:   {values}")
    if opts.einc:
        print(
            "einc:       E=({:g}, {:g}, {:g}) k=({:g}, {:g}, {:g})".format(
                *opts.E, *opts.k
            )
        )
    print(f"outputs:    sparam={opts.sparam} field={opts.field} tfe={opts.tfe}")
    print(f"output dir: {plan.output_dir}")
    print("status:     dry run (no solve)")


def _execute(plan: RunPlan, switches: Dict[str, object]) -> int:
    if switches.get("dry_run"):
        if switches.get("json"):
            _json_dump(plan.as_dict())
        else:
            _print_plan(plan)
        return EXIT_OK
    # Imported lazily so the phase-1 CLI does not depend on numerical modules.
    result = dispatch(plan)
    if result is not None:
        print(result)
    return EXIT_OK


def _parse(rest: Sequence[str], switches: Dict[str, object]) -> Options:
    options = parse_argv(rest, legacy=bool(switches.get("legacy")))
    if switches.get("json") and not switches.get("dry_run"):
        # --json on run/sparam/eigen implies a plan dump in phase 1.
        switches["dry_run"] = True
    return options


def _run_command(
    rest: Sequence[str],
    switches: Dict[str, object],
    command: str,
    validate=None,
) -> int:
    """Shared body of ``run``/``eigen``/``sweep``: plan, execute, map errors."""
    if switches.get("help"):
        print(help_text())
        return EXIT_OK
    try:
        options = _parse(rest, switches)
        if not options.name:
            return _error(
                f"{command} requires a project name; see `fes models`",
                EXIT_USAGE,
            )
        if command == "eigen" and not options.formula_explicit:
            options.assembly = "em_e_tl_eig"
            options.formula_explicit = True
        plan = plan_run(
            options=options,
            command=command,
            data=switches.get("data"),
        )
        if switches.get("out"):
            plan.output_dir = str(switches["out"])
        if validate is not None:
            message = validate(plan)
            if message:
                return _error(message)
        return _execute(plan, switches)
    except UsageError as exc:
        return _error(str(exc))
    except FileNotFoundError as exc:
        return _error(str(exc), EXIT_IO)
    except FormulationUnavailableError as exc:
        # The original C++ parses the flag, then aborts with exit status 1.
        return _error(str(exc), EXIT_USAGE)
    except NotPortedError as exc:
        return _error(str(exc), EXIT_NOT_PORTED)
    except RuntimeError as exc:
        if not _is_null_rhs(exc):
            raise
        return _null_rhs_abort(exc)


def _cmd_run(rest: Sequence[str], switches: Dict[str, object], command: str = "run") -> int:
    return _run_command(rest, switches, command)


def _cmd_sweep(rest: Sequence[str], switches: Dict[str, object]) -> int:
    def _requires_sweep(plan: RunPlan) -> str | None:
        if plan.options.n_freqs <= 1:
            return "sweep requires +fr <lf> <hf> <n> with n >= 2"
        return None

    return _run_command(rest, switches, "sweep", validate=_requires_sweep)


def _cmd_mesh(rest: Sequence[str], switches: Dict[str, object]) -> int:
    """Mesh one model from its ``.poly`` and write the ``.h1.mat.npz`` cache."""
    if switches.get("help"):
        print(help_text())
        return EXIT_OK
    try:
        options = parse_argv(rest, legacy=bool(switches.get("legacy")))
        if not options.name:
            return _error(
                "mesh requires a project name; see `fes models`", EXIT_USAGE
            )
        from . import api

        result = api.mesh(
            options=options,
            data=switches.get("data"),
            out_dir=switches.get("out"),
            write=not switches.get("no_write"),
            dry_run=bool(switches.get("dry_run")),
        )
    except UsageError as exc:
        return _error(str(exc))
    except FileNotFoundError as exc:
        return _error(str(exc), EXIT_IO)

    if isinstance(result, RunPlan):
        if switches.get("json"):
            _json_dump(result.as_dict())
        else:
            _print_plan(result)
        return EXIT_OK

    if switches.get("json"):
        _json_dump(result)
        return EXIT_OK
    if result.get("dimension") == 3:
        print(
            f"{result['model']}: {result['n_nodes']} nodes, "
            f"{result['n_tetras']} tetrahedra, {result['n_faces']} faces, "
            f"{result['n_edges']} edges (switches {result['switches']})"
        )
    else:
        print(
            f"{result['model']}: {result['n_nodes']} nodes, "
            f"{result['n_faces']} triangles, {result['n_edges']} edges "
            f"(switches {result['switches']})"
        )
    if result["cache"]:
        print(f"cache: {result['cache']}")
    return EXIT_OK


# ── entry point ─────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    try:
        rest, switches = _extract_switches(raw)
    except UsageError as exc:
        return _error(str(exc))

    if switches.get("version"):
        return _cmd_version(switches)
    if not rest or switches.get("help"):
        print(help_text())
        return EXIT_OK

    command = rest[0]
    args = rest[1:]

    if command in ("help", "-h", "--help"):
        print(help_text())
        return EXIT_OK
    if command in ("version", "--version", "-V"):
        return _cmd_version(switches)
    if command == "models":
        return _cmd_models(switches)
    if command == "info":
        return _cmd_info(args, switches)
    if command == "run":
        return _cmd_run(args, switches, "run")
    if command == "sweep":
        return _cmd_sweep(args, switches)
    if command == "mesh":
        return _cmd_mesh(args, switches)
    if command == "sparam":
        return _cmd_run(args, switches, "sparam")
    if command == "eigen":
        return _cmd_run(args, switches, "eigen")

    # Legacy: `fes <project> [frequency] +flags`.
    try:
        known = set(list_models())
    except Exception:
        known = set()
    if (
        command in known
        or command.startswith(("+", "-"))
        or command.endswith(".poly")
        or Path(command).is_file()
    ):
        return _cmd_run(rest, switches, "run")

    return _error(
        f"unknown command or model {command!r}; run `fes help` or `fes models`"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

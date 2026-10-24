from __future__ import annotations
"""Small CLI for running the Python modal-matching benchmark scripts."""

import argparse
import sys
from typing import Callable

from .scripts import (
    BifurcationE,
    BifurcationH,
    HildebrandFull,
    HildebrandHalf,
    HildebrandSemiAuto,
    Riblet,
)

ScriptResult = tuple[list, list, dict]
ScriptFn = Callable[[bool], ScriptResult]

SCRIPT_ALIASES = {
    "bifurcatione": "BifurcationE",
    "bifurcationh": "BifurcationH",
    "hildebrandhalf": "HildebrandHalf",
    "hilderbrandhalf": "HildebrandHalf",
    "hildebrandsemiauto": "HildebrandSemiAuto",
    "hilderbrandsemiauto": "HildebrandSemiAuto",
    "hildebrandfull": "HildebrandFull",
    "hilderbrandfull": "HildebrandFull",
    "riblet": "Riblet",
}

CANONICAL_SCRIPT_NAMES = (
    "BifurcationE",
    "BifurcationH",
    "Riblet",
    "HildebrandHalf",
    "HildebrandSemiAuto",
    "HildebrandFull",
)

SCRIPT_RUNNERS: dict[str, ScriptFn] = {
    "BifurcationE": BifurcationE,
    "BifurcationH": BifurcationH,
    "HildebrandHalf": HildebrandHalf,
    "HildebrandSemiAuto": HildebrandSemiAuto,
    "HildebrandFull": HildebrandFull,
    "Riblet": Riblet,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Python MM benchmark scripts",
        epilog="""
Available scripts:
- BifurcationE: E-plane T-junction (3 ports, 10–25 GHz)
- BifurcationH: H-plane T-junction (3 ports, 10–25 GHz)
- Riblet: Riblet broadwall coupler (4 ports, 7–15 GHz)
- HildebrandHalf: Hildebrand coupler (half, exploits symmetry) (2 ports, 13–15 GHz)
- HildebrandSemiAuto: Hildebrand coupler (half, explicit assembly) (2 ports, 13–15 GHz)
- HildebrandFull: Hildebrand coupler (full 4-port, 6 devices) (4 ports, 13–15 GHz)

Aliases (case-insensitive): bifurcatione, bifurcationh, hildebrandhalf, hildebrandsemiauto, hildebrandfull, riblet
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "script",
        help="Script to run (see list above). Canonical names or aliases accepted.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser


def _resolve_script_runner(user_value: str) -> tuple[str, ScriptFn]:
    script_key = user_value.lower()
    canonical_name = SCRIPT_ALIASES.get(script_key)
    if canonical_name is None:
        allowed_aliases = ", ".join(sorted(SCRIPT_ALIASES))
        print(
            f"Unknown script '{user_value}'. Allowed values: {allowed_aliases}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return canonical_name, SCRIPT_RUNNERS[canonical_name]


def main() -> None:
    """Parse CLI arguments, resolve aliases, run one script, and print summary."""
    parser = _build_parser()
    args = parser.parse_args()

    script_name, script_runner = _resolve_script_runner(args.script)
    plot = not args.no_plot
    sf, sinfo, err = script_runner(plot=plot)

    fatal = err.get("fatal")
    print(f"{script_name}: computed {len(sf)} frequency points, {len(sinfo)} ports, fatal={fatal}")


if __name__ == "__main__":
    main()

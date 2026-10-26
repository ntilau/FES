"""Command-line option handling.

This module preserves the C++ solver's ``+flag value`` grammar and reproduces
``cpp/src/option.cpp`` exactly in *legacy* mode.  The default parser also
implements the flags that the C++ usage text documents but whose generic
one-token parser never actually applied (``+gmres``, ``+nl``, ``+rad``,
``+dd``, ``+einc``, ``+fr``), plus the output controls ``+tfe``/``+sparam``.

Two entry points:

``parse_legacy(argv)``
    Byte-for-byte behaviour of ``option::set`` + ``option::apply_cli``.  Used by
    ``fes --legacy``.

``parse_argv(argv)``
    Documented behaviour: multi-value flags, ``-flag`` negation for the output
    controls, and a positional frequency.

Both accept argv *without* the program name and return an :class:`Options`.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field as _dc_field
from typing import Dict, List, Sequence

ASSEMBLY_TYPES = (
    "em_av_fd",
    "em_e_fd",
    "em_e_fd_dd",
    "em_e_fd_nl",
    "em_e_qs",
    "em_h_qs",
    "em_ez_fd",
    "em_e_tl_eig",
)

SOLVER_TYPES = ("direct", "gmres")

#: Alternate spellings accepted by ``+formula``.  C++ falls back to em_e_fd for
#: anything unknown, so unknown names are accepted rather than rejected.
_FORMULA_ALIASES = {
    "em_e_fd_schur": "em_e_fd_dd",
    "em_e_fd_dd_schur": "em_e_fd_dd",
}


class UsageError(ValueError):
    """Raised when a flag is missing a required value."""


def normalize_assembly(name: str) -> str:
    """Map a formula/assembly name to its canonical lowercase spelling."""
    value = str(name).strip().lower()
    value = _FORMULA_ALIASES.get(value, value)
    return value if value in ASSEMBLY_TYPES else "em_e_fd"


def normalize_solver(name: str) -> str:
    """Map a solver name; anything other than ``gmres`` is ``direct``."""
    return "gmres" if str(name).strip().lower() == "gmres" else "direct"


# ── C++ numeric conversion semantics ────────────────────────────────────────
#
# std::stod / std::stoull accept a leading numeric prefix and ignore trailing
# junk; the callers catch conversion failures and use 0.  Python's float() is
# stricter, so emulate the prefix behaviour for parity.

_CPP_DOUBLE = re.compile(
    r"^[ \t\n\r\f\v]*[+-]?(?:"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|inf(?:inity)?"
    r"|nan(?:\([^)]*\))?"
    r")",
    re.IGNORECASE,
)
_CPP_INTEGER = re.compile(r"^[ \t\n\r\f\v]*[+-]?\d+")


def cpp_stod(text: str) -> float:
    """Emulate ``std::stod``; return 0.0 when no numeric prefix exists."""
    match = _CPP_DOUBLE.match(str(text))
    if not match:
        return 0.0
    token = match.group(0).strip().lower()
    sign = -1.0 if token.startswith("-") else 1.0
    unsigned = token.lstrip("+-")
    if unsigned.startswith("inf"):
        return sign * math.inf
    if unsigned.startswith("nan"):
        return math.nan
    try:
        return float(token)
    except ValueError:  # pragma: no cover - regex should guarantee validity
        return 0.0


def cpp_stoull(text: str) -> int:
    """Emulate ``std::stoull`` including its unsigned wraparound."""
    match = _CPP_INTEGER.match(str(text))
    if not match:
        return 0
    token = match.group(0).strip()
    try:
        value = int(token, 10)
    except ValueError:  # pragma: no cover
        return 0
    return value % (1 << 64)


def _as_bool(value: str) -> bool:
    """C++ ``getB``: only the exact string ``"1"`` is true."""
    return value == "1"


@dataclass
class Options:
    """Solver options shared by the CLI, API, and oracle harness.

    Field names and defaults mirror ``cpp/include/option.h``; the trailing
    group holds Python-only controls.
    """

    # C++ option members
    solver: str = "direct"
    assembly: str = "em_e_fd"
    name: str = ""
    dbg: bool = False
    dbl: bool = True
    niter: int = 0
    toll: float = 0.0
    h_ord: int = 0
    p_ord: int = 1
    freq: float = 5e9
    l_freq: float = 0.0
    h_freq: float = 0.0
    n_freqs: int = 1
    n_harm: int = 0
    relax: float = 0.0
    E: List[float] = _dc_field(default_factory=lambda: [0.0, 0.0, 0.0])
    k: List[float] = _dc_field(default_factory=lambda: [0.0, 0.0, 0.0])
    einc: bool = False
    field: bool = False
    rad: bool = False
    n_theta: float = 0.0
    n_phi: float = 0.0
    poly: bool = True
    poly_cmd: str = ""
    href_cmd: str = ""
    dd: bool = False
    ddn: bool = False
    dds: bool = False
    n_jor_gs: bool = True
    n_dd: int = 0
    nl: bool = False
    power: float = 1.0
    Vbnd: Dict[str, float] = _dc_field(default_factory=dict)
    cli_override: Dict[str, str] = _dc_field(default_factory=dict)

    # Python-only controls
    tfe: bool = True
    sparam: bool = True
    sgl: bool = False
    quiet: bool = False
    gmres_restart: int = 0
    n_modes: int = 0
    nl_mtrl_name: str = ""
    kerr: float = 0.0
    dd_preconditioner: str = "gs"
    #: User-specified shift for the eigenmode solver (0.0 = use default).
    shift: float = 0.0
    #: Orthogonalisation to suppress spurious modes (``+o``/``-o``).
    ortho: bool = True
    #: ARPACK tolerance for the sparse eigenmode solver path (0.0 = default).
    eig_tol: float = 0.0
    #: Eigen solver path: ``"auto"``, ``"dense"``, ``"arpack"`` or ``"arnoldi"``.
    eig_solver: str = "auto"
    positional_freqs: List[float] = _dc_field(default_factory=list)
    extra_positionals: List[str] = _dc_field(default_factory=list)
    extra: Dict[str, str] = _dc_field(default_factory=dict)
    legacy: bool = False
    formula_explicit: bool = False

    # ── derived values ──────────────────────────────────────────────────────
    def freqs(self) -> List[float]:
        """Return the frequency sweep encoded by the options.

        Mirrors ``eq_sys`` in ``cpp/src/equation_system.cpp``: below two
        frequencies the single ``freq`` is repeated (the C++ fills the whole
        vector with it), otherwise the range is ``arma::linspace(l_freq,
        h_freq, n_freqs)``.  There is no ``h_freq > l_freq`` guard — the
        oracle runs a non-increasing sweep as written.
        """
        if self.n_freqs < 2:
            return [self.freq]
        count = int(self.n_freqs)
        step = (self.h_freq - self.l_freq) / (count - 1)
        return [self.l_freq + i * step for i in range(count)]

    @property
    def electrostatic(self) -> bool:
        return self.freq == 0.0

    def as_dict(self) -> dict:
        """JSON-friendly dump (used by the oracle harness and tests)."""
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (list, tuple)):
                out[key] = list(value)
            elif isinstance(value, dict):
                out[key] = dict(value)
            else:
                out[key] = value
        return out


def _empty_name_error() -> UsageError:
    return UsageError("Project name is required")


# ── Legacy parser (exact C++ parity) ────────────────────────────────────────


def parse_legacy(argv: Sequence[str]) -> Options:
    """Parse *argv* with the exact semantics of ``option::set``/``apply_cli``."""
    opt = Options(legacy=True)
    tokens = list(argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "":
            raise UsageError("Empty command-line argument")

        # +f <freq>
        if token == "+f":
            index += 1
            if index >= len(tokens):
                raise UsageError("Missing frequency")
            opt.cli_override["freq"] = tokens[index]
            index += 1
            continue
        # +formula <name>
        if token == "+formula":
            index += 1
            if index >= len(tokens):
                raise UsageError("Missing formula name")
            opt.cli_override["formula"] = tokens[index]
            index += 1
            continue
        # +volt <boundary> <voltage>
        if token == "+volt":
            if index + 2 >= len(tokens):
                raise UsageError("Missing boundary name or voltage")
            opt.cli_override["volt:" + tokens[index + 1]] = tokens[index + 2]
            index += 3
            continue
        # project name: first non-flag token
        if token[0] not in "+-":
            if not opt.name:
                opt.name = token
            index += 1
            continue

        # generic +flag <value> / -flag / +flag
        key = token[1:]
        if key == "p":
            key = "p_ord"
        elif key == "h":
            key = "h_ord"

        if index + 1 < len(tokens) and tokens[index + 1][0] not in "+-":
            opt.cli_override[key] = tokens[index + 1]
            index += 2
        else:
            opt.cli_override[key] = "1" if token[0] == "+" else "0"
            index += 1

    _apply_legacy(opt)
    return opt


def _apply_legacy(opt: Options) -> None:
    """Port of ``option::apply_cli``."""
    ov = lambda key: key in opt.cli_override
    get_s = lambda key: opt.cli_override[key]
    get_b = lambda key: _as_bool(opt.cli_override[key])
    get_d = lambda key: cpp_stod(opt.cli_override[key])
    get_i = lambda key: cpp_stoull(opt.cli_override[key])

    if ov("solver"):
        opt.solver = normalize_solver(get_s("solver"))
    if ov("formula"):
        opt.assembly = normalize_assembly(get_s("formula"))
        opt.formula_explicit = True
    if ov("assembly"):
        opt.assembly = normalize_assembly(get_s("assembly"))
        opt.formula_explicit = True

    # shorthand flags: +em_e_fd, +em_ez_fd, ...
    for name in ASSEMBLY_TYPES:
        if ov(name):
            opt.assembly = name
            opt.formula_explicit = True

    if ov("dbg"):
        opt.dbg = get_b("dbg")
    if ov("dbl"):
        opt.dbl = get_b("dbl")
    if ov("einc"):
        opt.einc = get_b("einc")
    if ov("field"):
        opt.field = get_b("field")
    if ov("rad"):
        opt.rad = get_b("rad")
    if ov("dd"):
        opt.dd = get_b("dd")
    if ov("ddn"):
        opt.ddn = get_b("ddn")
    if ov("dds"):
        opt.dds = get_b("dds")
    if ov("nl"):
        opt.nl = get_b("nl")
    if ov("n_jor_gs"):
        opt.n_jor_gs = get_b("n_jor_gs")

    if ov("n_dd"):
        opt.n_dd = get_i("n_dd")
    if ov("niter"):
        opt.niter = get_i("niter")
    if ov("h_ord"):
        opt.h_ord = get_i("h_ord")
    if ov("p_ord"):
        opt.p_ord = get_i("p_ord")
    if ov("n_freqs"):
        opt.n_freqs = get_i("n_freqs")
    if ov("n_harm"):
        opt.n_harm = get_i("n_harm")

    if ov("freq"):
        opt.freq = get_d("freq")
    if ov("toll"):
        opt.toll = get_d("toll")
    if ov("l_freq"):
        opt.l_freq = get_d("l_freq")
    if ov("h_freq"):
        opt.h_freq = get_d("h_freq")
    if ov("relax"):
        opt.relax = get_d("relax")
    if ov("n_theta"):
        opt.n_theta = get_d("n_theta")
    if ov("n_phi"):
        opt.n_phi = get_d("n_phi")
    if ov("power"):
        opt.power = get_d("power")

    if ov("poly_cmd"):
        opt.poly_cmd = get_s("poly_cmd")
    if ov("href_cmd"):
        opt.href_cmd = get_s("href_cmd")

    for index, axis in enumerate(("Ex", "Ey", "Ez")):
        if ov(axis):
            opt.einc = True
            opt.E[index] = get_d(axis)
    for index, axis in enumerate(("kx", "ky", "kz")):
        if ov(axis):
            opt.einc = True
            opt.k[index] = get_d(axis)

    for key, value in opt.cli_override.items():
        if key.startswith("volt:"):
            opt.Vbnd[key[5:]] = cpp_stod(value)


# ── Documented parser ───────────────────────────────────────────────────────


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _is_value(token: str) -> bool:
    """A value token is anything that does not start a flag, or a number."""
    if token == "":
        return False
    if token[0] not in "+-":
        return True
    return _looks_numeric(token)


def _need(tokens: List[str], index: int, count: int, flag: str) -> List[str]:
    """Return *count* value tokens starting at *index*, or raise UsageError."""
    values = []
    cursor = index
    while len(values) < count:
        if cursor >= len(tokens) or not _is_value(tokens[cursor]):
            raise UsageError(
                f"{flag} requires {count} value"
                f"{'s' if count != 1 else ''}; got {len(values)}"
            )
        values.append(tokens[cursor])
        cursor += 1
    return values


def parse_argv(argv: Sequence[str], legacy: bool = False) -> Options:
    """Parse command-line options using the documented semantics."""
    if legacy:
        return parse_legacy(argv)

    opt = Options()
    tokens = list(argv)
    index = 0
    freq_explicit = False

    while index < len(tokens):
        token = tokens[index]
        if token == "":
            raise UsageError("Empty command-line argument")
        if token in ("--help", "-h"):
            opt.extra["help"] = "1"
            index += 1
            continue
        if token in ("--legacy",):
            opt.legacy = True
            index += 1
            continue
        if token in ("--dry-run", "--json", "--all"):
            opt.extra[token[2:]] = "1"
            index += 1
            continue

        # ── multi-value and renamed flags ───────────────────────────────
        if token == "+fr":
            values = _need(tokens, index + 1, 3, "+fr")
            opt.l_freq = cpp_stod(values[0])
            opt.h_freq = cpp_stod(values[1])
            opt.n_freqs = cpp_stoull(values[2])
            opt.freq = opt.l_freq
            freq_explicit = True
            index += 4
            continue
        if token == "+gmres":
            values = _need(tokens, index + 1, 1, "+gmres")
            opt.solver = "gmres"
            opt.toll = cpp_stod(values[0])
            index += 2
            if index < len(tokens) and _is_value(tokens[index]):
                opt.gmres_restart = cpp_stoull(tokens[index])
                index += 1
            continue
        if token == "+nl":
            values = _need(tokens, index + 1, 4, "+nl")
            opt.nl = True
            opt.n_harm = cpp_stoull(values[0])
            opt.nl_mtrl_name = values[1]
            opt.kerr = cpp_stod(values[2])
            opt.relax = cpp_stod(values[3])
            index += 5
            continue
        if token == "+rad":
            values = _need(tokens, index + 1, 2, "+rad")
            opt.rad = True
            opt.n_theta = cpp_stod(values[0])
            opt.n_phi = cpp_stod(values[1])
            index += 3
            continue
        if token in ("+dd", "+dds", "+ddn"):
            values = _need(tokens, index + 1, 1, token)
            opt.n_dd = cpp_stoull(values[0])
            if token == "+ddn":
                opt.ddn = True
            else:
                opt.dds = True
                opt.dd = True
            index += 2
            continue
        if token == "+einc":
            if index + 1 < len(tokens) and "," in tokens[index + 1]:
                values = [part for part in tokens[index + 1].split(",") if part != ""]
                if len(values) != 6:
                    raise UsageError("+einc needs six values: Ex,Ey,Ez,kx,ky,kz")
                index += 2
            else:
                values = _need(tokens, index + 1, 6, "+einc")
                index += 7
            opt.einc = True
            opt.E = [cpp_stod(v) for v in values[:3]]
            opt.k = [cpp_stod(v) for v in values[3:]]
            continue
        if token == "+volt":
            values = _need(tokens, index + 1, 2, "+volt")
            opt.cli_override["volt:" + values[0]] = values[1]
            opt.Vbnd[values[0]] = cpp_stod(values[1])
            index += 3
            continue
        if token == "+poly":
            opt.poly = True
            index += 1
            if (
                opt.name
                and index < len(tokens)
                and _is_value(tokens[index])
                and not _looks_numeric(tokens[index])
            ):
                opt.poly_cmd = tokens[index]
                index += 1
            continue
        if token in ("+pow", "+power"):
            values = _need(tokens, index + 1, 1, token)
            opt.power = cpp_stod(values[0])
            index += 2
            continue
        if token in ("+sgl", "+single"):
            opt.sgl = True
            opt.dbl = False
            index += 1
            continue
        if token in ("+dbl", "+double"):
            opt.sgl = False
            opt.dbl = True
            index += 1
            continue
        if token == "+direct":
            opt.solver = "direct"
            index += 1
            continue
        if token in ("+gs", "+jc"):
            opt.dd_preconditioner = "gs" if token == "+gs" else "jacobi"
            opt.n_jor_gs = token == "+gs"
            index += 1
            continue
        if token in ("+tfe", "-tfe", "+sparam", "-sparam", "+ortho", "-ortho"):
            key = token[1:]
            opt.cli_override[key] = "1" if token[0] == "+" else "0"
            index += 1
            continue
        if token in ("+verbose", "-verbose"):
            opt.quiet = token == "-verbose"
            index += 1
            continue
        if token == "+field":
            opt.field = True
            index += 1
            continue
        if token == "-field":
            opt.field = False
            index += 1
            continue
        if token == "+dbg":
            opt.dbg = True
            index += 1
            continue
        if token == "-dbg":
            opt.dbg = False
            index += 1
            continue
        if token == "++":
            opt.extra["priority"] = "high"
            index += 1
            continue

        # ── single-value flags that C++ handled generically ─────────────
        if token in ("+formula", "+assembly", "+solver", "+f", "+p", "+h",
                     "+poly_cmd", "+href", "+href_cmd", "+n_modes", "+shift",
                     "+tol", "+eigsolver"):
            values = _need(tokens, index + 1, 1, token)
            value = values[0]
            if token in ("+formula", "+assembly"):
                opt.assembly = normalize_assembly(value)
                opt.formula_explicit = True
            elif token == "+solver":
                opt.solver = normalize_solver(value)
            elif token == "+f":
                opt.freq = cpp_stod(value)
                freq_explicit = True
            elif token == "+p":
                opt.p_ord = cpp_stoull(value)
            elif token == "+h":
                opt.h_ord = cpp_stoull(value)
            elif token in ("+href", "+href_cmd"):
                opt.href_cmd = value
            elif token == "+poly_cmd":
                opt.poly_cmd = value
            elif token == "+n_modes":
                opt.n_modes = cpp_stoull(value)
            elif token == "+shift":
                opt.shift = cpp_stod(value)
            elif token == "+tol":
                opt.eig_tol = cpp_stod(value)
            elif token == "+eigsolver":
                opt.eig_solver = value
            index += 2
            continue

        # ── project name / positional arguments ─────────────────────────
        if token[0] not in "+-":
            if not opt.name:
                opt.name = token
            elif _looks_numeric(token):
                opt.positional_freqs.append(float(token))
                if not freq_explicit:
                    opt.freq = float(token)
                    freq_explicit = True
                    warnings.warn(
                        f"positional frequency {token} is deprecated; use +f {token}",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"ignoring extra positional value {token!r}",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                opt.extra_positionals.append(token)
            index += 1
            continue

        # ── unknown flag: keep it, and a following value if present ──────
        key = token[1:]
        if key == "p":
            key = "p_ord"
        elif key == "h":
            key = "h_ord"
        if index + 1 < len(tokens) and _is_value(tokens[index + 1]):
            opt.cli_override[key] = tokens[index + 1]
            opt.extra[key] = tokens[index + 1]
            index += 2
        else:
            opt.cli_override[key] = "1" if token[0] == "+" else "0"
            opt.extra[key] = opt.cli_override[key]
            index += 1

    # Apply generic keys with the C++ mapping first (this covers every key the
    # serialized option format knew about), then override with the explicit
    # multi-value results parsed above.
    _apply_legacy(opt)

    if "sparam" in opt.cli_override:
        opt.sparam = _as_bool(opt.cli_override["sparam"])
    if "tfe" in opt.cli_override:
        opt.tfe = _as_bool(opt.cli_override["tfe"])
    if "dbl" in opt.cli_override:
        opt.dbl = _as_bool(opt.cli_override["dbl"])
        opt.sgl = not opt.dbl
    if "nl" in opt.cli_override:
        opt.nl = _as_bool(opt.cli_override["nl"])
    if "dd" in opt.cli_override:
        opt.dd = _as_bool(opt.cli_override["dd"])
    if "href" in opt.cli_override:
        opt.href_cmd = opt.cli_override["href"]
    if "ortho" in opt.cli_override:
        opt.ortho = _as_bool(opt.cli_override["ortho"])
    return opt


__all__ = [
    "ASSEMBLY_TYPES",
    "SOLVER_TYPES",
    "Options",
    "UsageError",
    "cpp_stod",
    "cpp_stoull",
    "normalize_assembly",
    "normalize_solver",
    "parse_argv",
    "parse_legacy",
]

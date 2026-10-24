"""Filesystem locations for models, and optional native tools.

The package is normally used from a source checkout, where ``data/`` sits one
level above this file.  Installed copies can point at a model directory with
``FES_DATA``; native mesh generators can be overridden with ``FES_TRIANGLE`` and
``FES_TETGEN_LIB``.
"""

from __future__ import annotations

import os
from pathlib import Path

_THIS = Path(__file__).resolve()


def repo_root() -> Path:
    """Return the repository root, or the current directory if installed."""
    override = os.environ.get("FES_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in _THIS.parents:
        if (parent / "data").is_dir() and (parent / "fes").is_dir():
            return parent
    return Path.cwd().resolve()


def data_dir() -> Path:
    """Return the directory containing the shared ``.poly`` models.

    Resolution order: ``FES_DATA``, ``<repo>/data``, ``<cwd>/data``, ``<cwd>``.
    """
    override = os.environ.get("FES_DATA")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"FES_DATA does not name a directory: {path}")
        return path

    candidates = [repo_root() / "data", Path.cwd() / "data", Path.cwd()]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return repo_root() / "data"


def model_path(name: str, data: Path | str | None = None) -> Path:
    """Resolve a model name (with or without ``.poly``) to an existing file.

    Absolute and existing relative paths are accepted unchanged; bare names are
    looked up in the data directory.
    """
    raw = Path(name).expanduser()
    if raw.suffix != ".poly":
        raw_with_suffix = raw.with_name(raw.name + ".poly")
    else:
        raw_with_suffix = raw

    for candidate in (raw_with_suffix, raw):
        if candidate.is_file():
            return candidate.resolve()

    base = Path(data) if data is not None else data_dir()
    candidate = base / raw_with_suffix.name
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Model {name!r} not found; looked for {candidate}. "
        "Use `fes models` to list available models or set FES_DATA."
    )


def list_models(data: Path | str | None = None) -> list[str]:
    """Return model stems in *data*, sorted.
    """
    base = Path(data) if data is not None else data_dir()
    names = sorted(p.stem for p in base.glob("*.poly"))
    return names

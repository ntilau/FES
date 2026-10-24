"""FES Python FEM package."""


def _read_version() -> str:
    try:
        from .paths import repo_root

        version_file = repo_root() / "VERSION"
        if version_file.is_file():
            return version_file.read_text(encoding="utf-8").strip()
    except Exception:  # pragma: no cover - defensive for installed copies
        pass
    try:
        from importlib.metadata import version

        return version("fes")
    except Exception:  # pragma: no cover
        return "0.2.0"


__version__ = _read_version()

from . import constants
from . import mesh
from . import core
from . import post

__all__ = ["__version__", "constants", "core", "mesh", "post"]

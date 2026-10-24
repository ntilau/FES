"""Modal-matching waveguide solver (Python port of a MATLAB toolbox), bundled
under the MIT license; see the repository-root ``LICENSE``.

The public surface mirrors the original flat layout: :mod:`mm.core`
holds the numerical routines, :mod:`mm.lib` provides one wrapper module
per MATLAB ``lib/`` function, and :mod:`mm.scripts` contains the
benchmark entry points (BifurcationE/H, Riblet, Hildebrand).

Run a benchmark from the command line with ``python -m mm <script>``.
"""

from . import core, lib, scripts

__all__ = ["core", "lib", "scripts"]

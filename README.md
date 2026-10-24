# FES — Finite Element Software

[![cli](https://github.com/ntilau/FES/actions/workflows/cli.yml/badge.svg)](https://github.com/ntilau/FES/actions/workflows/cli.yml)

FES is a 2D/3D finite element solver for computational electromagnetics based on
the E-field formulation with H(curl) conforming elements. It supports S-parameter
extraction, eigenmode analysis, electrostatics, incident plane waves, and
far-field radiation patterns.

## Installation

FES requires Python 3.9 or newer.

```bash
make setup                     # create .venv and install the package (editable)
./configure                    # same as make setup
./configure --minimal          # FEM-only: skip ML/visualization extras
```

There is no `fes` console script; always invoke the package as
`.venv/bin/python -m fes`.

Meshing from a `.poly` file needs the optional native libraries, which are not
vendored as binaries:

```bash
python tools/build_triangle.py
python tools/build_tetgen.py
```

Without them, only 2D models with a pre-computed mesh cache can be run, and
tests marked `triangle`/`tetgen` are skipped.

## Quick Reference

| Task | Command |
|------|---------|
| **Install** | `make setup` |
| **Install FEM-only** | `./configure --minimal` |
| **Run tests** | `make test` |
| **CI-equivalent tests** | `.venv/bin/python -m pytest -q` |
| **Lint** | `ruff check fes tools tests mm n2f rfm setup.py` |
| **Build native meshers** | `python tools/build_triangle.py && python tools/build_tetgen.py` |
| **Run the CLI** | `.venv/bin/python -m fes run WR90 +f 1e10 +poly +p 2` |
| **Legacy CLI spelling** | `.venv/bin/python -m fes WR90 1e10 +poly +p 2` |
| **Inspect the resolved plan** | `.venv/bin/python -m fes run WR90 +f 1e10 --dry-run` |
| **Mesh only (2D cache)** | `.venv/bin/python -m fes mesh BilatFilter --out build/meshes` |
| **Frequency sweep** | `.venv/bin/python -m fes sweep BilatFilter +fr 140e9 160e9 3 +poly +p 1` |
| **S-parameters** | `.venv/bin/python -m fes sparam BilatFilter +f 150e9 +poly +p 1` |
| **Eigenmodes** | `.venv/bin/python -m fes eigen WR90_TL +f 1e10 +poly` |
| **List / inspect models** | `.venv/bin/python -m fes models` / `... info WR90 [--json]` |
| **Version** | `.venv/bin/python -m fes version [--json]` |
| **Python project example** | `.venv/bin/python -c "from fes.projects import run_waveguide; run_waveguide()"` |
| **DNN-GP surrogate** | `.venv/bin/python -c "from fes.projects import bilateral_filter_dnngp; bilateral_filter_dnngp()"` |
| **Modal-matching benchmark** | `.venv/bin/python -m mm Riblet --no-plot` |
| **Near-field → far-field** | `from n2f import sf_nf2ff_solver` (see `docs/bundled_modules.md`) |
| **RFM / vector fitting** | `from rfm import read_touchstone, vector_fit` (see `rfm/README.md`) |

## Quick Start

```bash
make setup                     # creates .venv and installs the package
make test                      # runs the pytest suite
```

```bash
# 3D waveguide — mesh with TetGen (default switches), then solve
.venv/bin/python -m fes run WR90 +f 1e10 +poly +p 2

# 2D TMz filter — mesh with Triangle, then solve
.venv/bin/python -m fes run BilatFilter +f 150e9 +poly +p 2

# 2D cross-section eigenmodes (WR90_TL is the 2D cross-section of WR90)
.venv/bin/python -m fes eigen WR90_TL +f 1e10 +poly

# Electrostatics (requires voltage assignment)
.venv/bin/python -m fes run CapSense +f 0 +poly +volt Elec 1

# Incident plane wave and far-field radiation pattern (3D, no waveports)
.venv/bin/python -m fes run data/RadCube.poly \
    +f 1e9 +poly +rad 37 73 +Ex 1 +kz 1
```

Programmatic use:

```python
from fes.projects import bilateral_filter, run_waveguide

freqs, Sparams = bilateral_filter(n_freqs=41)
sys, mesh = run_waveguide()
```

See [`docs/projects.md`](docs/projects.md) for the full project catalogue, the
DNN-GP surrogate, adaptive LHS, and mesh-generation examples.

## Subcommands

`python -m fes --help` prints the complete flag set. Exit codes are `0`
success, `1` usage error, `2` formulation not ported, `3` I/O error.

| Subcommand | Purpose |
|------------|---------|
| `run <model> +f FREQ` | Solve one frequency |
| `sweep <model> +fr LF HF N` | Solve a discrete frequency sweep |
| `mesh <model>` | Import and mesh only; write a 2D mesh cache (3D is never cached) |
| `sparam <model> +f FREQ` | Extract S-parameters |
| `eigen <model> [+n_modes N]` | Cross-section eigenmode analysis (`em_e_tl_eig`) |
| `info <model> [--json]` | Print model metadata (dimension, formula, solids, boundaries) |
| `models [--json]` | List models in the data directory |
| `version [--json]` | Print version information |

## Formulations

Formulations are auto-detected from the `#Formula` tag in each `.poly` file;
CLI flags (`+formula NAME`, `+em_e_fd`, ...) override the file tag.

| Formula | Description |
|---------|-------------|
| `em_e_fd` | 3D frequency-domain EM (curl-curl, default) |
| `em_ez_fd` | 2D TMz (scalar Helmholtz on a Triangle mesh) |
| `em_e_tl_eig` | 2D cross-section / 3D waveport eigenmodes |
| `em_e_qs` | Electrostatic quasistatic (`f = 0`, use `+volt bnd V`) |

The v0.1 C++ version's domain-decomposition (`em_e_fd_dd`) and nonlinear
(`em_e_fd_nl`) assemblers are not ported: both abort inside the C++ version at
every input tried, so there is no reference result to compare against.
`em_h_qs` is rejected by the C++ version's own dispatcher.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| | **Model I/O** | |
| `+poly [CMD]` | — | Import `.poly` (auto-detects 2D/3D) and mesh; optional CMD means `+poly_cmd CMD` |
| `+poly_cmd CMD` | `q34` / `q` | Mesher switches (Triangle / TetGen); see the note below |
| `--dry-run` | — | Resolve and print the run plan without solving |
| `--out DIR` | — | `fes mesh`: directory for the 2D `.h1.mat.npz` cache |
| `--no-write` | — | `fes mesh`: report counts without writing a cache |
| `--remesh` | — | `fes mesh`: accepted; meshing always restarts from `.poly` |
| | **Formulation** | |
| `+formula NAME` | — | Select formulation by snake_case name |
| `+em_e_fd` | — | 3D frequency-domain EM (curl-curl) |
| `+em_e_tl_eig` | — | Waveport eigenmodes / 2D cross-section eigenmode |
| `+em_ez_fd` | — | 2D TMz (scalar Helmholtz) |
| `+em_e_qs` | — | Electrostatic quasistatic |
| | **Frequency** | |
| `+f FREQ` | — | Main frequency [Hz] (required; 0 = electrostatic) |
| `+fr LF HF N` | — | Discrete frequency sweep over N points |
| | **Mesh & discretization** | |
| `+p N` | 1 | Polynomial order (1–4) |
| `+h N` | 0 | Homogeneous mesh refinement level (2D) |
| | **Output** | |
| `+field` | — | Export VTK field data |
| `+rad NT NP` | — | Export a far-field radiation pattern (`NT`/`NP` samples) |
| `+sparam` | on | Write S-parameters (`-sparam` to disable) |
| | **Ports & excitation** | |
| `+tfe` | on | Transfinite element formulation on waveports (`-tfe` to disable) |
| `+volt bnd V` | — | Apply voltage V to a PEC boundary (electrostatic) |
| `+pow P` | 1 | Port power scaling [W] |
| `+Ex/+Ey/+Ez/+kx/+ky/+kz` | — | Incident plane wave (enables `+einc`) |
| `+einc Ex,Ey,Ez,kx,ky,kz` | — | Incident plane wave, single-token spelling |
| | **Solver** | |
| `+direct` | default | Sparse direct solver |
| `+gmres tol [restart]` | — | Iterative solver spelling (parsed; direct solve is used) |
| | **Not ported** | |
| `+dd N`, `+nl ...` | — | Parsed, but the formulations are not ported |

`+poly_cmd` supplies only the variable middle of the mesher switch string: the
driver prefixes `p` and appends `AfeQ` (2D) or `AfeeQ` (3D), so `+poly_cmd q`
yields TetGen's `pqAfeeQ` and `+poly_cmd q34` yields Triangle's `pq34AfeQ`.
These defaults reproduce the v0.1 solver's meshes; changing them changes the
mesh and breaks numerical agreement.

## Architecture

### Pipeline

```
import → mesh → assemble → solve → export
```

### Two numerical paths

This is the single most important structural fact in the repository:

- **Reference path** — `fes/core/assemblers/` plus the reference kernels
  (`shape`, `shape3d`, `quadrature`, `element_matrix`, `dof`,
  `quadrature_tables3d`) and the mesh backends. It reproduces the v0.1 C++
  solver down to loop order, DOF numbering and output formatting.
- **Legacy Python path** — `core/assembly.py`, `shape_functions.py`,
  `quadrature.py`, `jacobian.py`, `dof.py`, `boundary.py`,
  `core/harmonic_balance.py`, used by `fes/projects/`. It does not feed the
  four assembled formulations.

Do not mix the two: a fix in `core/shape_functions.py` does not affect the
reference path, and a change in `core/assemblers/` does not affect the legacy projects.

### Source layout

```
fes/
├── cli.py            # `python -m fes` front end
├── api.py            # RunPlan, formulation dispatch, DRIVERS registry
├── drivers.py        # per-formulation drivers and Touchstone I/O
├── paths.py          # model/mesh discovery (FES_DATA, caches)
├── core/
│   ├── assemblers/   # em_e_fd, em_ez_fd, em_e_tl_eig, em_e_qs
│   ├── *_kernels.py     # reference kernels (shape, quadrature, DOF, matrices)
│   ├── options.py       # legacy and documented argv parsers
│   └── assembly.py, shape_functions.py, ...  # legacy Python path
├── mesh/             # .poly I/O, Triangle/TetGen backends, topology, caches
├── post/             # mesh, field and radiation VTK writers
└── projects/         # legacy Python projects and DNN-GP surrogates
mm/                   # modal-matching solver (MIT)
n2f/                  # near-field → far-field transforms (MIT)
rfm/                  # rational function models, vector fitting (MIT)
tools/                # reference capture scripts, mesher builds
tests/                # unit and integration gates
setup.py              # pip-installable package
configure             # virtual environment setup
data/                 # shared .poly models and .h<N>.mat{,.npz} mesh caches
dep/                  # vendored native deps; Triangle/TetGen sources feed tools/build_*.py
```

### Key features

- **H(curl) conforming elements** — hierarchical vector basis functions
  (orders 1–4) for the curl-curl E-field formulation.
- **Transfinite elements (TFE)** — exact waveport mode expansion for accurate
  S-parameters.
- **Incident plane wave and radiation pattern** — `+Ex/+Ey/+Ez/+kx/+ky/+kz`
  drive the portless solve; `+rad` writes the far field and RCS.
- **2D TMz solver** — scalar Helmholtz on Triangle triangulations.
- **2D electrostatic** — P1 triangle assembly for quasistatic analysis.
- **2D cross-section eigenmode** — TE/TM mode computation on 2D meshes.
- **DNN-GP surrogate modelling** — Deep Kernel Learning surrogates.
- **Auto-formulation** — the `#Formula` tag selects the assembly type.
- **Deterministic output** — waveport modes sorted by propagation constant;
  VTK and Touchstone writers match the reference formatting.

### Bundled solver modules

Three standalone top-level packages ship alongside `fes/` and bypass the FEM
pipeline:

- **`mm`** — modal-matching solver for rectangular waveguide GSMs
  (`BifurcationE/H`, `Riblet`, Hildebrand benchmarks) with a CLI:
  `python -m mm Riblet --no-plot`.
- **`n2f`** — near-field to far-field transformation (scalar and vector
  Huygens/Kottler solvers, SVD model-order reduction).
- **`rfm`** — rational function models: Touchstone I/O, Vector Fitting,
  pole-residue macromodels, and passivity checks/enforcement.

All three are MIT-licensed and bundled for convenience; see
[`mm/README.md`](mm/README.md),
[`n2f/README.md`](n2f/README.md) and [`rfm/README.md`](rfm/README.md) for
per-package APIs and examples, or
[`docs/bundled_modules.md`](docs/bundled_modules.md) for the combined overview.

## `.poly` File Format

Standard TetGen PLC sections (nodes, segments, faces, holes, regions) plus
custom trailing sections. `#Formula` may be the first line or appear after the
geometry; the parser is case-insensitive and order-independent.

```
#Formula EM_EZ_FD
# NODES: num_nodes  dim  num_attributes  num_markers
<idx> <x> <y> [<z>] <marker>
# SEGMENTS: num_segments  num_markers
# FACES: num_faces  num_markers
# HOLES: num_holes
# REGIONS: num_regions

#Regions <N>                      ← alias: #Solids
<name> <epsr> <mur> <sigma> <tand> <matname>
#Boundaries <M>
<name> <label> <type> [numModes]
```

`#Regions` lines carry no label column: materials are matched to the mesh's
region attributes by their line order. The `#Boundaries` line *does* carry the
boundary label used by the mesh markers. Boundary types: `PerfectE` (PEC),
`PerfectH` (PMC), `Radiation`, `WavePort`, `LumpedPort`.

`#Formula` values: `EM_E_FD` (3D), `EM_EZ_FD` (2D TMz), `EM_E_TL_EIG`
(eigenmode), `EM_E_QS` (electrostatic). A CLI formulation flag overrides the
file.

## Model Resolution and Mesh Caches

`fes/paths.py` resolves a bare model name to `data/<name>.poly` (also
`<cwd>/data`, or `$FES_DATA`). `fes models` lists available models.

A 2D model's mesh is loaded from `<name>.h<N>.mat.npz` or the legacy
`<name>.h<N>.mat`; if absent it is meshed from the `.poly` and written back as
`.npz`. `fes mesh --out DIR` writes the cache elsewhere. 3D models are meshed
with TetGen on every run and are never cached, matching the v0.1 C++ version.

Native library discovery can be overridden with `FES_TRIANGLE` (Triangle) and
`FES_TETGEN_LIB` (TetGen shim).

## Testing

```bash
make test                                           # pytest tests/ -v
.venv/bin/python -m pytest -q                       # CI equivalent
.venv/bin/python -m pytest tests/test_core.py::TestShapeFunctions::test_2d_linear -v
```

Markers:

- `triangle`, `tetgen` — mesh-generation tests; skipped unless the native
  library is available.

Lint with `ruff` (configured in `ruff.toml`; it is not a package dependency):

```bash
ruff check fes tools tests mm n2f rfm setup.py
```

## Dependencies

Python dependencies are declared in `setup.py`:

- Core: NumPy, SciPy, Matplotlib
- Optional visualization: pyvista (`fes[viz]`)
- Optional ML: torch, gpytorch, scikit-learn (`fes[ml]`)
- Development: pytest plus the `ml` and `viz` extras (`fes[dev]`)

The bundled `mm`, `n2f` and `rfm` modules use only the core dependencies.

## License

Copyright (c) 2026 Laurent Ntibarikure.

MIT License — see [LICENSE](LICENSE). The same single license file covers the
repository and the bundled `mm`, `n2f` and `rfm` packages; their duplicate
per-package license copies have been merged into it.

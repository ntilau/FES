# FES — Finite Element Solver

[![build](https://github.com/ntilau/FES/actions/workflows/build.yml/badge.svg)](https://github.com/ntilau/FES/actions/workflows/build.yml)

FES is a 2D/3D finite element solver for computational electromagnetics based on the E-field formulation
with H(curl) conforming elements. It supports S-parameter extraction, eigenmode analysis,
electrostatics, and nonlinear Kerr materials.

The solver has three language backends sharing the same model files:

| Backend | Dir | Description |
|---------|-----|-------------|
| **C++** | `cpp/` | Production 3D solver — curl-curl, MUMPS direct + GMRES iterative, DD, waveports |
| **Python** | `py/` | 2D solver + DNN-GP surrogate modelling — scalar Helmholtz, HB, ML surrogates |
| **MATLAB** | `m/` | Reference / legacy implementation — full-featured, research-oriented |

---

## Quick Start

### C++ backend (primary)

```bash
./setup                        # install dependencies (compiled from source)
make build                     # cmake configure + build (Release)
make test                      # run all model tests (load & mesh only)
```

The binary is at `cpp/build/fes`:

```bash
# 3D waveguide — mesh with TetGen, then solve
cd data && ../cpp/build/fes WR90 +poly AafeeQ +f 1e10 +p 2

# 2D TMz filter — mesh with Triangle, then solve
cd data && ../cpp/build/fes BilatFilter +poly q34a +f 150e9 +p 2

# Waveport eigenmodes
cd data && ../cpp/build/fes WR90 +poly AafeeQ +f 1e10 +p 2 +formula em_e_tl_eig

# Electrostatic (requires voltage assignment)
cd data && ../cpp/build/fes CapSense +poly +f 0 +volt Elec 1
```

### Python backend

```bash
./setup --py                    # or: cd py && ./configure
make py-test                    # or: py/.venv/bin/python -m pytest py/tests/ -v
```

```bash
# Run waveguide simulation
cd py && .venv/bin/python -c "from fes.projects import run_waveguide; run_waveguide()"

# Train DNN-GP surrogate model
cd py && .venv/bin/python -c "from fes.projects import bilateral_filter_dnngp; bilateral_filter_dnngp()"
```

### MATLAB backend

```bash
./setup --m                     # or: cd m && ./configure
make m-build                    # cd m && make all  (builds IOrMesh mesher)
# make m-test                   # runs MATLAB/Octave tests
```

```matlab
% In MATLAB/Octave:
addpath(genpath('m'));
ProjectWaveGuide;
```

---

## Formulations (C++ CLI)

Formulations are auto-detected from the `#Formula` tag in each `.poly` file.
CLI flags (`+formula em_e_fd`, `+em_e_fd`, etc.) override the file tag.

| CLI flag | Description |
|----------|-------------|
| `+formula em_e_fd` or `+em_e_fd` | 3D frequency-domain EM (curl-curl, default) |
| `+formula em_ez_fd` or `+em_ez_fd` | 2D TMz (scalar Helmholtz on Triangle mesh) |
| `+formula em_e_tl_eig` or `+em_e_tl_eig` | 2D cross-section eigenmode or 3D waveport eigenmodes |
| `+formula em_e_qs` or `+em_e_qs` | Electrostatic quasistatic (f = 0, use `+volt bnd V`) |

## CLI Options (C++)

| Flag | Default | Description |
|------|---------|-------------|
| | **Model I/O** | |
| `+poly [CMD]` | — | Import `.poly` file (auto-detects 2D/3D; optional TetGen quality switches) |
| | **Formulation** | |
| `+formula NAME` | — | Select formulation by snake_case name (`em_e_fd`, `em_ez_fd`, etc.) |
| `+em_e_fd` | — | 3D frequency-domain EM (curl-curl) |
| `+em_e_fd_dd N` | — | 3D EM with domain decomposition |
| `+em_e_fd_nl` | — | 3D EM with nonlinear Kerr |
| `+em_e_tl_eig` | — | Waveport eigenmodes / 2D cross-section eigenmode |
| `+em_ez_fd` | — | 2D TMz (scalar Helmholtz) |
| `+em_e_qs` | — | Electrostatic quasistatic |
| `+em_h_qs` | — | Magnetic quasistatic |
| | **Frequency** | |
| `+f FREQ` | — | Main frequency [Hz] (required; 0 = electrostatic) |
| | **Mesh & discretization** | |
| `+p N` | 1 | Polynomial order (1–4) |
| `+h N` | 0 | Homogeneous mesh refinement level |
| `+href CMD` | — | Quality mesh refinement with TetGen |
| | **Output** | |
| `+field` | — | Export VTK field data |
| `+rad Nθ Nφ` | — | Export far-field radiation pattern |
| `+sparam` | on | Write S-parameters (`-sparam` to disable) |
| | **Ports & excitation** | |
| `+tfe` | on | Transfinite element formulation on waveports (`-tfe` to disable) |
| `+volt bnd V` | — | Apply voltage V to PEC boundary (electrostatic) |
| `+pow P` | 1 | Port power scaling [W] |
| `+einc LABEL = {Ex,Ey,Ez,kx,ky,kz}` | — | Incident plane wave (disables S-params) |
| | **Frequency sweep** | |
| `+fr lf hf n` | — | Discrete frequency sweep over n points |
| | **Solver** | |
| `+direct` | default | MUMPS direct sparse solver |
| `+gmres tol [restart]` | — | GMRES iterative solver |
| `+sgl` | — | Single precision (default: double) |
| `+dbl` | — | Double precision (explicit) |
| `+dbg` | — | Debug output |
| | **Domain decomposition** | |
| `+dd N` | — | Partition mesh into N subdomains (Schur complement) |
| `+dds N` | — | DD with Schur complement (explicit) |
| `+ddn N` | — | DD with Neumann preconditioner |
| `+gs` | default | Gauss-Seidel DD preconditioner |
| `+jc` | — | Jacobi DD preconditioner |
| | **Nonlinear** | |
| `+nl H mat kerr relax` | — | Kerr nonlinear, H harmonics |
| | **Misc** | |
| `++` | — | Increase process priority |
| `-verbose` | — | Suppress console output |

## Architecture

### Pipeline

```
import → mesh → assemble → solve → export
```

This pipeline is implemented independently in each backend:

**C++ (`cpp/`):** Compiled binary, TetGen/Triangle meshing, H(curl) elements, sparse direct/iterative solvers.

**Python (`py/`):** Triangle meshing via io_poly, scipy sparse assembly, numpy/scipy solve, pyVista field rendering.

**MATLAB (`m/`):** IOrMesh/Triangle meshing, native MATLAB sparse matrices, direct/DD/HB solvers.

### Source layout

```
├── cpp/               # C++ FEM solver (production 3D)
│   ├── include/       #   25 headers: assembler, solver, mesh, options, ...
│   ├── src/           #   27 implementation files + main.cpp
│   └── CMakeLists.txt #   C++14, links dep/lib/*.a
├── py/                # Python FEM + ML surrogates
│   ├── fes/         #   FEM core: fem/, mesh/, post/, projects/
│   ├── tests/         #   pytest suite
│   └── setup.py       #   pip-installable package
├── m/                 # MATLAB reference implementation
│   ├── fes/           #   Package root (mirrors py/fes/)
│   │   ├── core/      #     Assembly routines (40+ files)
│   │   ├── mesh/      #     Mesh I/O, geometry writers
│   │   ├── post/      #     VTK field export
│   │   └── projects/  #     Simulation project drivers
│   ├── tests/         #   Standalone / debug / DD / NL scripts
│   └── Config.m       #   Path setup (addpath(genpath('.')))
├── data/              # Shared model files — .poly geometry + .h1.mat mesh caches (all backends)
└── dep/               # C++ library dependencies (dep/src, dep/build, dep/lib)
```

### Key features

- **H(curl) conforming elements** — hierarchical vector basis functions (orders 1–3)
  for the curl-curl E-field formulation
- **Transfinite elements (TFE)** — exact port mode expansion for accurate S-parameters
- **Domain decomposition** — additive Schwarz or Schur complement preconditioners
- **Nonlinear materials** — Kerr effect, iterative fixed-point relaxation
- **2D TMz solver** — P2 elements on Triangle triangulations, auto-detected from `#Formula`
- **2D electrostatic** — P1 triangle assembly for quasistatic analysis
- **2D cross-section eigenmode** — waveguide TE/TM mode computation on 2D meshes
- **DNN-GP surrogate modelling** — Deep Kernel Learning surrogates (fes, Wilson et al. 2016)
- **Auto-formulation** — `#Formula` tag in `.poly` selects assembly type automatically
- **OOP architecture** — polymorphic assembly (`assembler` base), strategy-pattern solvers (`solver` base)
- **Sparse matrices** — `arma::SpMat<complex<double>>` (C++), `scipy.sparse.csr` (Python)

### .poly file format

Standard TetGen PLC sections (nodes, facets, holes, regions) plus custom trailing sections:

```
# NODES: num_nodes  dim  num_attributes  num_markers
...
# SEGMENTS: num_segments  num_markers
...
# REGIONS: num_regions
...
#Formula EM_EZ_FD                 ← auto-selects formulation
#Regions N
<name> <label> <epsr> <mur> <sigma> <matname>
#Boundaries M
<name> <label> <type> [numModes]
```

Boundary types: `PerfectE`, `PerfectH`, `Radiation`, `WavePort`.

`#Formula` values: `EM_E_FD` (3D), `EM_EZ_FD` (2D TMz), `EM_E_TL_EIG` (eigenmode), `EM_E_QS` (electrostatic).

### .fes file format

The `.fes` file stores the complete simulation state — options, materials, boundary conditions,
and mesh — in a single file. It uses an XML header followed by binary mesh sections.

### File formats

| Format | Extension | Description |
|--------|-----------|-------------|
| Poly | `.poly` | TetGen PLC with `#Formula`/`#Regions`/`#Boundaries` sections |
| Native | `.fes` | FES project file (XML header + binary mesh) |
| Touchstone | `.sNp` | S-parameter output |

## Dependencies (C++)

| Library | Role |
|---------|------|
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) | Dense BLAS/LAPACK |
| [ARPACK-NG](https://github.com/opencollab/arpack-ng) | Sparse eigenvalue solver |
| [MUMPS](https://github.com/scivision/mumps-superbuild) | Direct sparse multifrontal solver |
| [METIS](https://github.com/KarypisLab/METIS) | Graph partitioning and mesh reordering |
| [Armadillo](https://gitlab.com/conradsnicta/armadillo-code) | Dense and sparse linear algebra |
| [Triangle](https://github.com/libigl/triangle) | 2D Delaunay triangulation |
| [TetGen](https://github.com/libigl/tetgen) | 3D tetrahedral mesh generation |

All C++ dependencies built via `./setup` into `dep/`.

Python dependencies installed via `pip` under `py/` (see `py/setup.py`).

## License

MIT — see [LICENSE](LICENSE).

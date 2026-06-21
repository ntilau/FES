# FES — Finite Element Solver

[![build](https://github.com/ntilau/FES/actions/workflows/build.yml/badge.svg)](https://github.com/ntilau/FES/actions/workflows/build.yml)

FES is a 2D/3D finite element solver for computational electromagnetics based on the E-field formulation
with H(curl) conforming elements. It supports S-parameter extraction, eigenmode analysis,
electrostatics, and nonlinear Kerr materials.

## Prerequisites

- **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- **Linux**: GCC ≥ 4.9 (for C++14), BLAS, LAPACK
- ~2 GB disk, ~15 minutes initial setup (builds all dependencies from source)

## Quick Start

```bash
./setup           # install dependencies (OpenBLAS, ARPACK-NG, MUMPS, METIS, Armadillo, TetGen, ...)
make build        # cmake configure + build (Release)
make test         # run 2D TMz, 3D EM, and waveport eigenmode tests
```

The binary is at `core/build/fes`. Models reside in `mdl/`:

```bash
# 3D waveguide — mesh with TetGen, then solve
cd mdl && ../core/build/fes WR90 +poly AafeeQ +f 1e10 +p 2

# 2D TMz filter — mesh with Triangle, then solve
cd mdl && ../core/build/fes BilatFilter +poly q34a +f 150e9 +p 2

# Waveport eigenmodes
cd mdl && ../core/build/fes WR90 +poly AafeeQ +f 1e10 +p 2 +formula em_e_tl_eig

# Electrostatic (requires voltage assignment)
cd mdl && ../core/build/fes CapSense +poly +f 0 +volt Elec 1
```

### CLI modifies the `.fes` file

When loading an existing `.fes` file, CLI flags are merged into the stored options
and the file is re-saved automatically. This means options persist across runs:

```bash
# First run: mesh + set options
./core/build/fes model +poly +f 1e10 +p 2 +rad 36 72

# Subsequent runs: just load the .fes — all options are already saved
./core/build/fes model +f 1e10

# Or override: change p_ord in the file permanently
./core/build/fes model +f 1e10 +p 3
```

## Formulations

Formulations are auto-detected from the `#Formula` tag in each `.poly` file.
CLI flags (`+formula em_e_fd`, `+em_e_fd`, etc.) override the file tag. All class names, method
names, and identifiers use **snake_case** throughout the C++ source.

| CLI flag | Description |
|----------|-------------|
| `+formula em_e_fd` or `+em_e_fd` | 3D frequency-domain EM (curl-curl, default) |
| `+formula em_ez_fd` or `+em_ez_fd` | 2D TMz (scalar Helmholtz on Triangle mesh) |
| `+formula em_e_tl_eig` or `+em_e_tl_eig` | 2D cross-section eigenmode or 3D waveport eigenmodes |
| `+formula em_e_qs` or `+em_e_qs` | Electrostatic quasistatic (f = 0, use `+volt bnd V`) |

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| | **Model I/O** | |
| `+poly [CMD]` | — | Import `.poly` file (auto-detects 2D/3D; optional TetGen quality switches) |
| `+hfss` | — | Import HFSS project |
| `+unv` | — | Import UNV mesh |
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
| `+sol` | — | Write solution vector |
| `+msh` | — | Export VTK mesh |
| `+sparam` | on | Write S-parameters (`-sparam` to disable) |
| `+matlab` | — | Dump system in MatrixMarket format |
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

1. `project` reads `.poly`, `.hfss`, or `.fes` — populates mesh, materials, BCs
2. `pre_processor` auto-detects dimension and reads `#Formula`/`#Regions`/`#Boundaries` from `.poly`
3. 2D models mesh with Triangle; 3D models mesh with TetGen
4. `eq_sys` dispatches to `assembler::create(type)->assemble()`
5. `solver::create(*opt)->solve()` — mumps_solver direct or custom gmres_solver
6. `post_processor::save_data()` — Touchstone S-params, VTK fields, radiation

### Source layout

```
core/
├── include/          # 25 headers (all snake_case)
│   ├── pre_processor.h  # Auto-detects 2D/3D, dispatches Triangle or TetGen
│   ├── assembler.h      # Abstract base + 6 derived assembly classes
│   ├── solver.h         # Abstract base + mumps_solver / gmres_solver
│   ├── post_processor.h # S-param / field / radiation export
│   ├── equation_system.h# eq_sys: frequency loop, wires assembly→solve→postproc
│   ├── mesh.h           # Mesh data: nodes, edges, faces, tetras
│   ├── project.h        # Model I/O: .poly/.fes/.hfss → binary .fes
│   ├── option.h         # CLI option parsing
│   ├── eigen_solver.h   # ARPACK eigenvalue solver (waveport modes)
│   ├── element_matrix.h # Element-level FE matrix computation
│   ├── shape.h          # Basis functions (hcurl, hgrad)
│   ├── quadrature.h     # Gauss-Legendre quadrature
│   ├── gmres.h          # DD-preconditioned GMRES templates
│   ├── field.h          # VTK field export
│   ├── radiation.h      # Far-field radiation pattern
│   └── ...
├── src/              # 27 implementation files + main.cpp
└── CMakeLists.txt    # C++14, links dep/lib/*.a and dep/lib/*.dylib
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
- **Auto-formulation** — `#Formula` tag in `.poly` selects assembly type automatically
- **OOP architecture** — polymorphic assembly (`assembler` base), strategy-pattern solvers (`solver` base)
- **Sparse Armadillo matrices** — `arma::SpMat<complex<double>>` throughout

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
| HFSS | `.hfss` | ANSYS HFSS mesh + boundary/material files |
| Touchstone | `.sNp` | S-parameter output |

## Dependencies

| Library | Role |
|---------|------|
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) | Dense BLAS/LAPACK |
| [ARPACK-NG](https://github.com/opencollab/arpack-ng) | Sparse eigenvalue solver |
| [MUMPS](https://github.com/scivision/mumps-superbuild) | Direct sparse multifrontal solver |
| [METIS](https://github.com/KarypisLab/METIS) | Graph partitioning and mesh reordering |
| [Armadillo](https://gitlab.com/conradsnicta/armadillo-code) | Dense and sparse linear algebra |
| [Triangle](https://github.com/libigl/triangle) | 2D Delaunay triangulation |
| [TetGen](https://github.com/libigl/tetgen) | 3D tetrahedral mesh generation |

All dependencies built via `./setup` into `dep/`.

## License

MIT — see [LICENSE](LICENSE).

# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Overview

This repo has three independent FEM solver implementations sharing model files in `mdl/`:

| Backend | Dir | Primary use |
|---------|-----|-------------|
| C++     | `cpp/` | Production 3D solver (curl-curl, waveports, MUMPS/GMRES, DD) |
| Python  | `py/`  | 2D solver + DNN-GP surrogate modelling |
| MATLAB  | `m/`   | Reference / legacy implementation |

---

## Build & Test

### C++ backend (primary)

```bash
make build          # cmake configure + build (Release)
make test           # build + run all .fes models in mdl/ (load & mesh check)
make clean          # remove cpp/build
```

Binary is at `cpp/build/fes`. Run directly:

```bash
./cpp/build/fes mdl/WR90 1e10 +poly AafeeQ +p 2       # 3D EM (default)
./cpp/build/fes mdl/WR90 1e10 +poly AafeeQ +p 2 +formula em_e_tl_eig  # waveport eigenmodes
./cpp/build/fes mdl/BilatFilter 150e9 +poly +p 2 +formula em_ez_fd   # 2D TMz
```

### Python backend

```bash
make py-setup       # create .venv + pip install (or: cd py && ./configure)
make py-test        # run pytest (or: cd py && .venv/bin/python -m pytest tests/ -v)
```

Projects run from `py/`:
```bash
cd py && .venv/bin/python -c "from pyfes.projects import run_waveguide; run_waveguide()"
```

### MATLAB backend

```bash
make m-build        # build IOrMesh and Triangle mesh tools (or: cd m && make all)
make m-test         # run MATLAB/Octave tests
make m-projects     # run all project scripts
```

### Setup

```bash
./setup             # everything: C++ deps + py venv + m check
./setup --py        # Python backend only
./setup --m         # MATLAB backend check only
./setup --compiler  # C++ deps only (OpenBLAS, ARPACK-NG, MUMPS, Armadillo, Triangle, TetGen)
```

---

## Architecture

The pipeline shared across all backends is: **import → mesh → assemble → solve → export**.

### C++ backend (`cpp/`)

#### Source layout

```
cpp/
├── include/          # 23 headers (all snake_case)
│   ├── option.h         # CLI option storage
│   ├── project.h        # Model I/O: reads .poly/.fes, writes binary .fes
│   ├── mesh.h           # Mesh data: nodes, edges, faces, tetras
│   ├── equation_system.h  # eq_sys — frequency loop, wires assembly→solve→postproc
│   ├── assembler.h      # Abstract base + 6 derived assembly classes
│   ├── solver.h         # Abstract base + mumps_solver / gmres_solver
│   ├── post_processor.h # S-param / field / radiation export
│   ├── pre_processor.h  # Auto-detects 2D/3D, dispatches Triangle or TetGen
│   ├── eigen_solver.h   # ARPACK eigenvalue solver for waveport modes
│   ├── element_matrix.h # Element-level FE matrix computation
│   ├── shape.h          # Basis function evaluation (hcurl, hgrad)
│   ├── quadrature.h     # Gauss-Legendre quadrature rules
│   ├── boundary_condition.h  # bc data (PEC, PMC, waveport, ABC)
│   ├── material.h       # Material properties (epsr, mur, sigma, kerr)
│   ├── field.h          # VTK field export
│   ├── radiation.h      # Far-field radiation pattern
│   ├── gmres.h          # DD-preconditioned GMRES templates
│   ├── mumps_constants.h  # MUMPS parameter constants
│   ├── tet_gen_wrap.h   # TetGen wrapper
│   ├── triangle_wrap.h  # Triangle wrapper (2D meshing)
│   ├── degree_of_freedom.h  # Local→global DOF numbering
│   ├── coupling.h       # Kerr nonlinear coupling tensor
│   ├── memory.h         # Memory reporting
│   ├── configuration.h  # System config / priority helpers
│   └── constants.h      # Physical constants
├── src/              # 27 implementation files + main.cpp
└── CMakeLists.txt    # C++14, links dep/lib/*.a and dep/lib/*.dylib
```

#### Solver flow (EM frequency-domain)

1. `main.cpp` → `option::set(argc, argv)` parses CLI flags → `option::apply_cli()` applies them
2. `project(log_file, &opt)` loads model (`.poly`→TetGen/Triangle, `.fes`→binary load)
3. `eq_sys(log_file, &prj)` — for each frequency:
   - `assembler::create(type)->assemble(log, sys)` — polymorphic assembly
   - `solver::create(*opt)->solve(sys, log)` — mumps_solver direct or gmres_solver iterative
   - `post_processor(sys, log).save_data()` — S-params, VTK fields, radiation

#### Formulations

| CLI flag | Assembler class | Description |
|---|---|---|
| `+formula em_e_fd` or `+em_e_fd` | `assembler_em_e_fd` | 3D frequency-domain (curl-curl + mass) |
| `+formula em_e_fd_dd` or `+em_e_fd_dd` | `assembler_em_e_fd_dd` | Domain decomposition |
| `+formula em_e_fd_nl` or `+em_e_fd_nl` | `assembler_em_e_fd_nl` | Nonlinear Kerr (harmonic balance) |
| `+formula em_ez_fd` or `+em_ez_fd` | `assembler_em_ez_fd` | 2D TMz (scalar Helmholtz) |
| `+formula em_e_qs` or `+em_e_qs` | `assembler_em_e_qs` | Electrostatic quasistatic |
| `+formula em_e_tl_eig` or `+em_e_tl_eig` | auto (3D→`assembler_em_e_fd`, 2D→`assembler_em_ez_fd`) | Waveport eigenmodes only |

#### Key types

| Type | Header | Storage |
|------|--------|---------|
| `mat_row_type` | `equation_system.h` | `arma::SpMat<std::complex<double>>` — CSR sparse |
| `mat_col_type` | `equation_system.h` | `arma::SpMat<std::complex<double>>` — CSC sparse |
| `vec_type` | `equation_system.h` | `arma::cx_vec` |
| DOF vectors | `equation_system.h` | `arma::cx_mat` — dense complex Armadillo matrices |

#### CLI defaults

`tfe=true`, `sparam=true`, `solver=direct`, `p_ord=1`, `assembly=em_e_fd`, `dbl=true`

#### Dependency notes

- **Armadillo 15.x** — `solve(out, A, B)` no longer accepts string solver type; use `solve_opts::opts`
- **OpenMP** is optional — CMake warns if not found but build succeeds without it

---

### Python backend (`py/`)

#### Source layout

```
py/
├── pyfes/
│   ├── fem/               # FEM core
│   │   ├── shape_functions.py  # Scalar Lagrange (1–4), H(curl) vector basis
│   │   ├── quadrature.py       # Gauss–Legendre, Duffy simplex quadrature
│   │   ├── jacobian.py         # Jacobian for triangles
│   │   ├── dof.py              # Global DOF numbering
│   │   ├── boundary.py         # Boundary DOF maps for DD
│   │   ├── assembly.py         # System matrix assembly, waveguide ports, BCs
│   │   └── harmonic_balance.py # Kerr nonlinearity, ferrite HB
│   ├── mesh/              # Mesh I/O and generation
│   │   ├── io_poly.py     # Read Triangle .poly, write .poly geometry
│   │   ├── build.py       # Regular triangular meshes
│   │   └── plot.py        # Matplotlib mesh plotting
│   ├── post/              # Post-processing
│   │   └── plot.py        # pyVista field rendering
│   └── projects/          # Simulation projects
│       ├── waveguide.py        # Rectangular waveguide S-params
│       ├── filter_design.py    # Bilateral/two-post filter scattering
│       ├── filter_dnngp.py     # DNN-GP surrogate model training
│       ├── modal_analysis.py   # TE mode cutoffs, open microstrip
│       ├── electrostatics.py   # Electrostatic potential
│       ├── thermal.py          # Heat conduction (standard + DG)
│       ├── circulator.py       # Ferrite circulator, intermodulation
│       ├── scattering.py       # Wave scattering with ABC, DD
│       ├── capacitive.py       # Coaxial capacitance, capacitive sensor
│       └── _utils.py           # Shared helper functions
├── data/                # .poly geometry files + .h1.mat mesh caches
├── iormesh/             # C mesher (Triangle wrapper) — builds binary
├── tests/               # pytest suite
├── setup.py             # pip installable package
└── configure            # venv setup script
```

#### Key conventions

- Mesh uses **0-based indexing** (MATLAB used 1-based)
- Sparse matrices use `scipy.sparse.csr_matrix`
- Shape functions are lambda functions evaluated at reference coordinates
- Reference triangle: vertices (0,0), (1,0), (0,1)
- `sys` dict carries all system state, `mesh` dict carries geometry/topology

### MATLAB backend (`m/`)

```
m/
├── FEass/              # Assembly routines (40+ files)
│   ├── AssembLin.m         Linear assembly
│   ├── AssembHB.m          Harmonic balance assembly
│   ├── AssembDD.m          Domain decomposition assembly
│   ├── AssembNL.m          Nonlinear assembly
│   └── ...                 (CalcShapeFunctions, GetCoupl*, Solv*, etc.)
├── FEpre/              # Pre-processing
│   ├── WriteWaveGuide.m    Geometry writers
│   ├── IOrPoly.m           .poly file I/O
│   ├── IOwPoly.m           .poly file output
│   ├── ...
│   ├── IGES/               IGES CAD file import toolbox
│   └── iormesh-src/        C source for IOrMesh mesher
├── FEpost/             # Post-processing
│   ├── IOwVTK.m            VTK field export
│   └── IOwVTKH.m           VTK H-field export
├── Tests/              # Test and debug scripts
│   ├── DD/                 Domain decomposition tests
│   ├── NL/                 Nonlinear tests
│   └── _Matlab/            Internal debug scripts
└── Project*.m          # 27 top-level project drivers
```

---

## .poly file format

Standard TetGen PLC sections (nodes, facets, holes, regions) plus custom trailing sections:

```
#Formula EM_E_FD
#Solids N
<name> <label> <epsr> <mur> <sigma> <type>
#Boundaries M
<name> <label> <type>
```

Solids populate materials (vacuum, dielectric, conductor). Boundaries define PEC, PMC, waveports, lumped ports, absorbing BCs.

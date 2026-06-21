# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
make build          # cmake configure + build (Release)
make test           # build + run all .fes models in mdl/ (load & mesh check)
make clean          # remove core/build
```

Binary is at `core/build/fes`. Run directly:

```bash
./core/build/fes mdl/WR90 1e10 +poly AafeeQ +p 2       # 3D EM (default)
./core/build/fes mdl/WR90 1e10 +poly AafeeQ +p 2 +formula em_e_tl_eig  # waveport eigenmodes
./core/build/fes mdl/BilatFilter 150e9 +poly +p 2 +formula em_ez_fd   # 2D TMz
```

Dependencies live in `dep/` — built via `./setup` (installs OpenBLAS, ARPACK-NG, MUMPS, Armadillo, Triangle, TetGen). If already built, `make build` suffices.

## Architecture

The pipeline is: **import → mesh → assemble → solve → export VTK/S-params**.

### Source layout

```
core/
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

### Solver flow (EM frequency-domain)

1. `main.cpp` → `option::set(argc, argv)` parses CLI flags → `option::apply_cli()` applies them
2. `project(log_file, &opt)` loads model (`.poly`→TetGen/Triangle, `.fes`→binary load)
3. `eq_sys(log_file, &prj)` — for each frequency:
   - `assembler::create(type)->assemble(log, sys)` — polymorphic assembly
   - `solver::create(*opt)->solve(sys, log)` — mumps_solver direct or gmres_solver iterative
   - `post_processor(sys, log).save_data()` — S-params, VTK fields, radiation

### Formulations

| CLI flag | Assembler class | Description |
|---|---|---|
| `+formula em_e_fd` or `+em_e_fd` | `assembler_em_e_fd` | 3D frequency-domain (curl-curl + mass) |
| `+formula em_e_fd_dd` or `+em_e_fd_dd` | `assembler_em_e_fd_dd` | Domain decomposition |
| `+formula em_e_fd_nl` or `+em_e_fd_nl` | `assembler_em_e_fd_nl` | Nonlinear Kerr (harmonic balance) |
| `+formula em_ez_fd` or `+em_ez_fd` | `assembler_em_ez_fd` | 2D TMz (scalar Helmholtz) |
| `+formula em_e_qs` or `+em_e_qs` | `assembler_em_e_qs` | Electrostatic quasistatic |
| `+formula em_e_tl_eig` or `+em_e_tl_eig` | auto (3D→`assembler_em_e_fd`, 2D→`assembler_em_ez_fd`) | Waveport eigenmodes only |

### Key types

| Type | Header | Storage |
|------|--------|---------|
| `mat_row_type` | `equation_system.h` | `arma::SpMat<std::complex<double>>` — CSR sparse |
| `mat_col_type` | `equation_system.h` | `arma::SpMat<std::complex<double>>` — CSC sparse |
| `vec_type` | `equation_system.h` | `arma::cx_vec` |
| DOF vectors | `equation_system.h` | `arma::cx_mat` — dense complex Armadillo matrices |

### CLI defaults (`option.cpp`)

`tfe=true`, `sparam=true`, `solver=direct`, `p_ord=1`, `assembly=em_e_fd`, `dbl=true`

### .poly file format

Standard TetGen PLC sections (nodes, facets, holes, regions) plus custom trailing sections:

```
#Solids N
<name> <label> <epsr> <mur> <sigma> <type>
#Boundaries M
<name> <label> <type>
```

Solids populate materials (vacuum, dielectric, conductor). Boundaries define PEC, PMC, waveports, lumped ports, absorbing BCs.

### Dependency notes

- **Armadillo 15.x** — `solve(out, A, B)` no longer accepts string solver type; use `solve_opts::opts`
- **OpenMP** is optional — CMake warns if not found but build succeeds without it

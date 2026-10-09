# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

| Task | Command |
|------|---------|
| **Install all dependencies** | `./setup` |
| **Install Python backend only** | `./setup --py` |
| **Install MATLAB backend only** | `./setup --m` |
| **Install C++ dependencies only** | `./setup --compiler` |
| **Build C++ solver** | `make build` |
| **Run all C++ model tests** | `make test` |
| **Run a specific C++ model** | `make <model>` (e.g., `make WR90`) |
| **Set up Python environment** | `make py-setup` |
| **Run Python tests** | `make py-test` |
| **Build MATLAB mesh tools** | `make m-build` |
| **Run MATLAB test cases** | `make m-test` |
| **Run MATLAB standalone scripts** | `make m-tests` |
| **Run C++ binary directly** | `./cpp/build/fes <model> <freq> [options]` |
| **Run Python project example** | `cd py && .venv/bin/python -c "from fes.projects import run_waveguide; run_waveguide()"` |
| **Run MATLAB example** | In MATLAB: `addpath(genpath('m')); ProjectWaveGuide;` |

## Overview

This repository contains three independent finite element method (FEM) solvers for computational electromagnetics, sharing the same `.poly` model files in the `data/` directory:

| Backend | Directory | Primary Use |
|---------|-----------|-------------|
| **C++** | `cpp/` | Production 3D solver (curl-curl formulation, MUMPS/GMRES, domain decomposition, waveports) |
| **Python** | `py/` | 2D solver + DNN-GP surrogate modeling (scalar Helmholtz, harmonic balance, machine learning surrogates) |
| **MATLAB** | `m/` | Reference / legacy implementation (full-featured, research-oriented) |

All backends follow the same pipeline: **import → mesh → assemble → solve → export**.

## Build & Test

### C++ Backend (`cpp/`)

- **Build**: `make build` (runs CMake configure and compile in Release mode)
- **Test all models**: `make test` (builds and runs mesh/check on all `.poly` files in `data/`)
- **Test a specific model**: `make <model>` (e.g., `make WR90` runs the WR90 waveguide model)
- **Clean build**: `make clean` (removes the `cpp/build` directory)
- **Reconfigure CMake**: `make config`
- **Binary location**: `cpp/build/fes`

**Direct usage examples**:
```bash
# 3D waveguide (default frequency-domain formulation)
cd data && ../cpp/build/fes WR90 1e10 +poly AafeeQ +p 2

# 2D TMz filter
cd data && ../cpp/build/fes BilatFilter 150e9 +poly +p 2 +formula em_ez_fd

# Waveport eigenmodes
cd data && ../cpp/build/fes WR90 1e10 +poly AafeeQ +p 2 +formula em_e_tl_eig

# Electrostatics (requires voltage assignment)
cd data && ../cpp/build/fes CapSense 0 +poly +volt Elec 1
```

### Python Backend (`py/`)

- **Setup**: `make py-setup` (creates virtual environment and installs dependencies)
- **Run tests**: `make py-test` (executes pytest suite)
- **Manual setup**: `cd py && ./configure`
- **Virtual environment**: `py/.venv`

**Example usage**:
```bash
# Run waveguide S-parameter simulation
cd py && .venv/bin/python -c "from fes.projects import run_waveguide; run_waveguide()"

# Train DNN-GP surrogate model for a bilateral filter
cd py && .venv/bin/python -c "from fes.projects import bilateral_filter_dnngp; bilateral_filter_dnngp()"
```

### MATLAB Backend (`m/`)

- **Build mesh tools**: `make m-build` (compiles IOrMesh and Triangle wrappers)
- **Run test cases**: `make m-test` (executes all FEM project test cases in `m/tests/`)
- **Run standalone scripts**: `make m-tests` (runs debug, domain decomposition, and nonlinear test scripts)
- **Manual build**: `cd m && make all`

**Usage in MATLAB/Octave**:
```matlab
addpath(genpath('m'));
ProjectWaveGuide; % Example: waveguide simulation
```

## Architecture

### Pipeline

All backends implement the same core pipeline:
1. **Import**: Read `.poly` file and parse geometry, materials, boundary conditions.
2. **Mesh**: Generate triangular (2D) or tetrahedral (3D) mesh using Triangle or TetGen.
3. **Assemble**: Construct system matrices and right-hand side vectors for the chosen formulation.
4. **Solve**: Solve the linear system (or eigenvalue problem) using direct or iterative solvers.
5. **Export**: Compute S-parameters, field data, radiation patterns, or other quantities of interest.

### Source Layout

```
cpp/
├── include/          # 25+ headers (snake_case): option, project, mesh, equation_system, assembler, solver, etc.
├── src/              # 27+ implementation files + main.cpp
└── CMakeLists.txt    # C++14, links prebuilt dependencies in dep/lib/

py/
├── fes/              # FEM core and projects
│   ├── core/         # Shape functions, quadrature, Jacobian, DOF, boundary, assembly, harmonic balance
│   ├── mesh/         # .poly I/O, mesh generation, plotting
│   ├── post/         # Field visualization (pyVista)
│   └── projects/     # Simulation scripts (waveguide, filter design, DNN-GP, modal analysis, etc.)
├── tests/            # Pytest suite
├── setup.py          # Pip-installable package
└── configure         # Virtual environment setup script

m/
├── fes/              # Package root (mirrors py/fes/)
│   ├── core/         # Assembly routines (40+ files)
│   ├── mesh/         # Mesh I/O, geometry writers
│   ├── post/         # VTK field export
│   └── projects/     # Simulation project drivers (Project*.m)
├── tests/            # Standalone scripts, domain decomposition, nonlinear tests
└── Config.m          # Path setup: addpath(genpath('.'))

data/                 # Shared .poly model files and mesh caches (.h1.mat)
dep/                  # C++ dependency libraries (OpenBLAS, ARPACK-NG, MUMPS, Armadillo, Triangle, TetGen)
```

### Key Features (C++ Backend)

- **H(curl) conforming elements**: Hierarchical vector basis functions (orders 1–4) for curl-curl formulation.
- **Transfinite Elements (TFE)**: Exact waveport mode expansion for accurate S-parameters.
- **Domain Decomposition**: Additive Schwarz or Schur complement preconditioners for large problems.
- **Nonlinear Materials**: Kerr effect modeled via harmonic balance and fixed-point iteration.
- **2D Solvers**: TMz (scalar Helmholtz), electrostatic, and cross-section eigenmode.
- **Automatic Formulation Selection**: `#Formula` tag in `.poly` files chooses assembly type; CLI flags (`+formula`) can override.
- **Sparse Linear Algebra**: Armadillo `SpMat<complex<double>>` for system matrices; MUMPS direct or GMRES iterative solvers.
- **Deterministic Waveport Ordering**: Eigenmodes sorted by propagation constant magnitude for reproducible results.

### .poly File Format

Standard TetGen PLC format with custom sections for materials and boundaries:

```
# NODES: <num_nodes> <dim> <num_attributes> <num_markers>
...
# SEGMENTS: <num_segments> <num_markers>
...
# REGIONS: <num_regions>
...
#Formula <TYPE>        ← Selects formulation (EM_E_FD, EM_EZ_FD, EM_E_TL_EIG, EM_E_QS)
#Solids <N>
<name> <label> <epsr> <mur> <sigma> <matname>
#Boundaries <M>
<name> <label> <type> [numModes]
```

- **Materials**: Defined in `#Solids`; vacuum, dielectric, conductor.
- **Boundaries**: Types include `PerfectE` (PEC), `PerfectH` (PMC), `Radiation` (ABC), `WavePort`.
- **Special flags**: `+poly` in CLI invokes TetGen/Triangle meshing; optional quality switches (e.g., `+poly q34a`).

### Common CLI Options (C++ Binary)

| Option | Description |
|--------|-------------|
| `+poly [CMD]` | Load `.poly` file; CMD passed to TetGen/Triangle (e.g., `+poly q34a` for quality mesh) |
| `+formula NAME` | Explicit formulation override (e.g., `em_e_fd`, `em_ez_fd`, `em_e_tl_eig`) |
| `+f FREQ` | Frequency in Hz (required; 0 for electrostatic) |
| `+p N` | Polynomial order (1–4) |
| `+tfe` | Enable transfinite element formulation on waveports (default on) |
| `+sparam` | Write S-parameter Touchstone file (default on) |
| `+field` | Export VTK field data |
| `+rad Nθ Nφ` | Export far-field radiation pattern |
| `+direct` | Use MUMPS direct solver (default) |
| `+gmres tol [restart]` | Use GMRES iterative solver |
| `+dd N` | Domain decomposition into N subdomains |
| `+nl H mat kerr relax` | Kerr nonlinearity: H harmonics, material label, kerr coefficient, relaxation factor |

## Development Tips

- To run a specific test for debugging, use the corresponding `make <target>` (e.g., `make WR90` for C++, or invoke Python/MATLAB examples directly).
- The `data/` directory contains numerous `.poly` files for various structures (waveguides, filters, antennas, etc.).
- When modifying formulations or solvers, focus on the `assembler*` and `solver*` classes in the C++ backend, or the `assembly.py` and harmonic balance modules in Python.
- MATLAB scripts in `m/tests/` and `m/projects/` serve as references for implementing new features.
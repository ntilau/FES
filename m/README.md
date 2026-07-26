# mFES

MATLAB Finite Element Solver for 2D electromagnetic problems.

Solves time-harmonic Maxwell's equations using high-order **p-adaptive** edge elements (hierarchical curl-conforming basis) on triangular meshes. Supports linear, harmonic balance, and domain decomposition formulations for waveguides, filters, circulators, and scattering structures.

## Structure

| Directory | Contents |
|---|---|
| root (`.m`) | Project scripts — one per application |
| `FEass/` | System matrix assembly — linear, harmonic balance, domain decomposition, waveguide ports |
| `FEpre/` | Mesh generation, geometry I/O, Triangle wrapper |
| `FEpost/` | VTK file output for visualization (ParaView) |
| `Tests/` | Unit tests, convergence studies, DD validation, plotting utilities |

## Requirements

- **C/C++ compiler** (gcc/g++) — to build the mesh generator
- **MATLAB** or **GNU Octave** — to run the solver

## Quick start

```sh
# Check dependencies and build the mesh generator
./configure
make
```

```matlab
% Add paths and set up
Config

% Run a project
ProjectWaveGuide
```

### Workflow

1. **Geometry** — a `.poly` file defines the 2D geometry and boundary regions
2. **Mesh** — `IOrPoly()` calls `IOrMesh` which invokes [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) to generate a triangular mesh, then writes a `.mat` file with node/element/edge data
3. **Solve** — the project script assembles the FEM system (mass, stiffness, port matrices), applies BCs, and solves for fields or S-parameters
4. **Visualize** — results are written to VTK files for viewing in ParaView

The bundled Triangle library handles all mesh generation — no external download needed.

## Makefile targets

| Target | Description |
|---|---|
| `make` / `make all` | Build IOrMesh and Triangle |
| `make test` | Run all tests |
| `make test-<Name>` | Run a single test (e.g. `make test-TestCoupling`) |
| `make project-<Name>` | Run a project from MATLAB (e.g. `make project-ProjectWaveGuide`) |
| `make clean` | Remove build artifacts |
| `make install` | Copy binaries to `FEpre/` |

## Solver capabilities

| Feature | Description |
|---|---|
| **Element types** | First- and second-order triangular elements (subparametric) |
| **Basis functions** | Hierarchical curl-conforming edge elements (p-refinement) |
| **Physics** | Time-harmonic Maxwell's equations (E-field, H-field formulations) |
| **BCs** | Dirichlet (PEC/PMC), Neumann, impedance, waveguide ports |
| **Domain decomposition** | Schur complement-based DD iteration for large problems |
| **Nonlinear** | Harmonic balance with Kerr nonlinearity (ferrite circulators, filters) |
| **Output** | VTK (ParaView), MATLAB `.mat` files |

## Projects

### Waveguides

| File | Description |
|---|---|
| `ProjectWaveGuide.m` | Rectangular waveguide S-parameters |
| `ProjectModalAnalysis.m` | TE/TM cutoff frequencies and dispersion |
| `ProjectModalAnalysisOpenStrip.m` | Open strip waveguide modes |

### Filters

| File | Description |
|---|---|
| `ProjectBilateralFilter.m` | Bilateral fin-line filter S-parameters |
| `ProjectBilateralFilterHB.m` | Harmonic balance with Kerr nonlinearity |
| `ProjectBilateralFilterNL.m` | Nonlinear bilateral filter |
| `ProjectTwoPostFilter.m` | Two-post waveguide filter |
| `ProjectTwoPostFilterHB.m` | Two-post filter, harmonic balance |
| `ProjectTwoPostFilterNL.m` | Two-post filter, nonlinear |

### Circulators

| File | Description |
|---|---|
| `ProjectCirculator.m` | Ferrite circulator full-wave simulation |
| `ProjectCirculatorDDschur.m` | Circulator with DD-Schur complement |
| `ProjectCirculatorIMP.m` | Circulator intermodulation products |
| `ProjectCirculatorIMP_TestHarmonicConv.m` | Circulator IMP harmonic convergence |
| `ProjectCirculatorIMPDDschur.m` | Circulator IMP with DD-Schur |
| `ProjectCirculatorIMPDDschur_Acceleration.m` | Accelerated circulator IMP-DD-Schur |

### Scattering

| File | Description |
|---|---|
| `ProjectWaveScatteringDD.m` | Wave scattering with DD iteration |
| `ProjectWaveScatteringDDglob.m` | Wave scattering, DD global formulation |
| `ProjectWaveScatteringEincEs.m` | Wave scattering, incident/scattered field split |
| `ProjectWaveScatteringFullField.m` | Wave scattering, full-field formulation |
| `ProjectNewDDscattering.m` | Domain decomposition scattering (high-order) |
| `ProjectScatteringFloquet.m` | Periodic structure scattering with Floquet BCs |

### Electrostatics & Thermal

| File | Description |
|---|---|
| `ProjectElectrostatics.m` | Electrostatic potential, Dirichlet BCs |
| `ProjectCapacitiveClearance.m` | Capacitive sensor capacitance |
| `ProjectCapacitiveClearanceCorrelated.m` | Capacitive sensor with correlated regions |
| `ProjectCableParassitics.m` | Coaxial capacitance extraction |
| `ProjectCoaxWideband.m` | Coaxial cable wideband capacitance |
| `ProjectThermalDistro.m` | Steady-state heat distribution |
| `ProjectThermalDistroDG.m` | Transient heat equation, DG in time |

## Tests

| File | Description |
|---|---|
| `fem1ode.m` | 1D FEM verification (ODE) |
| `TestCoupling.m` | Waveguide port coupling verification |
| `TestCouplingGeneric.m` | Generic coupling interface test |
| `EntCirc.m` | Circular geometry entity helper |
| `ThermalFE.m` | Thermal FEM test |
| `TestCurls.m` | Curl-conforming basis verification |
| `testGMRES.m` | GMRES solver test |
| `TestSurfInteg.m` | Surface integration test |
| `TestTetrahedronQuad.m` | Tetrahedron quadrature test |
| `TestTetrahedronQuadTraces.m` | Trace quadrature test |
| `PlotEfficiency.m` | Solver efficiency and memory usage plots |
| `PlotHarmPow.m` | Circulator harmonic field results |
| `PlotIMPConv.m` | Intermodulation product convergence plots |

Domain decomposition tests are in `Tests/DD/`, nonlinear tests in `Tests/NL/`.

## History

This is a readability-refactored copy of the original FESmat MATLAB code. The git history preserves the original source as the initial commit, followed by mechanical cleanup and a full readability pass (section headers, line breaks, dead code removal, consistent formatting).

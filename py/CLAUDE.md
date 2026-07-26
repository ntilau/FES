# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pyFES — Python Finite Element Software for electromagnetics.

2D Helmholtz / scalar wave / electrostatic / thermal FEM solver on triangular meshes, with waveguide ports, domain decomposition, harmonic balance (Kerr nonlinearity), ferrite circulators, and a DNN-GP surrogate modelling capability.

## Commands

### Setup
- Full install: `./configure` (creates .venv, installs package + pytest + torch + gpytorch)
- FEM-only: `./configure --minimal`
- Editable mode: `python -m pip install -e .`

### Tests
- Run all tests: `python -m pytest tests/ -v`
- Single test: `python -m pytest tests/test_core.py::TestShapeFunctions::test_2d_linear -v`

### Projects
- Bilateral filter (FEM): `python -c "from pyfes.projects import bilateral_filter; bilateral_filter()"`
- DNN-GP surrogate: `python -c "from pyfes.projects import bilateral_filter_dnngp; bilateral_filter_dnngp(mat_file='bilat_all_sparams_50epsr.mat')"`
- DNN-GP plot: `python -c "from pyfes.projects import plot_surrogate; plot_surrogate()"`
- Waveguide: `python -c "from pyfes.projects import run_waveguide; run_waveguide()"`

### Venv
- Activate: `source .venv/bin/activate`
- Path: `$CWD/.venv/`

## Architecture

### FEM core (`pyfes/fem/`)
| Module | Description |
|---|---|
| `shape_functions.py` | Scalar Lagrange (orders 1–4), H(curl) vector basis, DOF counting |
| `quadrature.py` | Gauss–Legendre on [0,1], Duffy-transformed simplex quadrature |
| `jacobian.py` | Jacobian determinant / inverse transpose for triangles |
| `dof.py` | Global DOF numbering, physical coordinate positioning |
| `boundary.py` | Boundary DOF maps for domain decomposition |
| `assembly.py` | System matrix assembly (S/T/St/Tt/G), waveguide ports, BCs |
| `harmonic_balance.py` | Harmonic balance (Kerr nonlinearity, ferrite) |

### Mesh (`pyfes/mesh/`)
| Module | Description |
|---|---|
| `io_poly.py` | Read Triangle .poly meshes, write .poly geometry files |
| `build.py` | Regular triangular meshes on the unit square |
| `plot.py` | Matplotlib mesh / geometry plotting |

### Post-processing (`pyfes/post/`)
| Module | Description |
|---|---|
| `plot.py` | pyVista-based in-process field rendering |

### Projects (`pyfes/projects/`)
| Module | Description |
|---|---|
| `filter_design.py` | Waveguide filter scattering (bilateral, two-post, HB) |
| `filter_dnngp.py` | **Deep Kernel Learning surrogate** (Wilson et al. 2016) — trains 4 DNN→GP models: S11 Re/Im (0.19%), S21 log|S|+phase (0.21%). |
| `waveguide.py` | Rectangular waveguide S-parameters |
| `modal_analysis.py` | TE mode cutoff frequencies, open microstrip |
| `electrostatics.py` | Electrostatic potential |
| `thermal.py` | Heat conduction (standard + DG) |
| `circulator.py` | Ferrite circulator S-parameters, intermodulation, DD |
| `scattering.py` | Wave scattering with ABC, DD, full-field |
| `capacitive.py` | Coaxial capacitance, capacitive sensor |

### Other
| Path | Description |
|---|---|
| `pyfes/constants.py` | Physical constants (c0, z0, eps0, mu0), dB/phase utilities |
| `data/` | .poly geometry files and .h1.mat mesh files |
| `iormesh/` | C mesher (Triangle wrapper, MAT .mat exporter) |

### Paper
- `paper/MoE-DKL-FE.pdf` — IEEE paper: *Mixture of Experts Deep Kernel Learning for Millimeter-Wave Passive Devices Model Order Reduction*
- `paper/MoE-DKL-FE.tex` — LaTeX source (IEEEtran, two-column)
- 8 vector figures from trained surrogate, 11 references, 5 pages

### Pre-generated data files
Files in project root (not tracked in git):
- `bilat_all_sparams_50epsr.mat` — 4050 samples (50 εᵣ × 81 freq), all 4 S-params
- `bilat_all_sparams_10epsr.mat` — 810 samples for quick iteration

## Key conventions

### FEM conventions
- Mesh uses 0-based indexing throughout (MATLAB used 1-based)
- Sparse matrices use `scipy.sparse.csr_matrix`
- Shape functions are returned as lambda functions that evaluate at reference coordinates
- The reference triangle has vertices (0,0), (1,0), (0,1)
- `sys` dict carries all system state (solver config, assembled matrices, solution)
- `mesh` dict carries all mesh geometry and topology

### DKL conventions (`filter_dnngp.py`)
- 4 independent DNN → ExactGP models (Deep Kernel Learning, Wilson et al. 2016)
- DNN: `Linear(8→1000)→ReLU→Linear(1000→1000)→ReLU→Linear(1000→500)→ReLU→Linear(500→64)`
- GP kernel: `ScaleKernel(RBFKernel(ard) × RBFKernel(ard))` — product kernel captures feature interactions
- Two-stage training: DNN pretrain via MSE, then joint DNN+GP via marginal likelihood
- Input features (8-dim): [εᵣ, f/f_scale, sin(ω), cos(ω), sin(2ω), cos(2ω), sin(3ω), cos(3ω)]
- Data is z-score normalized per output before training
- Validation split: adaptive (at most 1/3 of εᵣ values)

**Output transforms:**
- **S11 — 0.19%**: Re/Im directly (raw complex works well for reflection)
- **S21 — 0.21%**: log10|S| + detrended unwrapped phase (handles 14x amplitude range). Linear phase trend (slope+intercept) fitted per epsr, stored, and restored during reconstruction.

**Data I/O:**
- .mat files: epsr, freq, s11/s12/s21/s22 (complex), s*_re/s*_im (float)
- Save/load: torch.save for full model state + scalers
- Generated .mat/.npz/.pt files gitignored; regenerate via `bilateral_filter_dnngp(generate=True)`

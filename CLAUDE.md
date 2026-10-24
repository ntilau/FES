# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Set up the Python environment: `./configure` (creates `.venv`, installs the package editable). `./configure --minimal` installs base dependencies only (no ML, no pyvista, no pytest); `./configure --dev` (the default, also spelled `standard`) adds `[viz]`, pytest, torch/gpytorch/scikit-learn.
- `make setup` / `make test` wrap `./configure` and `pytest tests/ -v`.
- Lint: `ruff check fes tools tests mm n2f rfm setup.py`. `ruff` is not a package dependency (CI installs it explicitly); `ruff.toml` selects only correctness rules (`E9`, `F63`, `F7`, `F82`), so style and typing are not enforced.
- CI: `.github/workflows/cli.yml` runs the same `ruff check` and `pytest -q` on push to `main` and on pull requests, across Ubuntu/macOS and Python 3.10–3.14.
- Run all tests (CI-equivalent): `.venv/bin/python -m pytest -q`
- Run a single test: `.venv/bin/python -m pytest tests/test_core.py::TestShapeFunctions::test_2d_linear -v`
- Build native meshers (Triangle/TetGen): `.venv/bin/python tools/build_triangle.py && .venv/bin/python tools/build_tetgen.py`. Both compile from `dep/`, which is gitignored and absent from a clean checkout; `triangle`/`tetgen`-marked tests skip when the shared library is missing (CI never has `dep/`).
- Run the CLI: `.venv/bin/python -m fes run <model> +f <freq> +poly +p 2`
- Inspect the resolved plan without solving: `.venv/bin/python -m fes run <model> +f <freq> --dry-run`
- List / inspect models: `.venv/bin/python -m fes models` / `.venv/bin/python -m fes info <model> [--json]`

There is no `fes` console script; always invoke the package as `python -m fes`. `python -m fes --help` lists every subcommand (`run`, `sweep`, `mesh`, `sparam`, `eigen`, `info`, `models`, `version`) and the full flag set. Exit codes: `0` success, `1` usage error, `2` formulation not ported, `3` I/O error.

Pytest markers (from `pytest.ini`): `slow`, `triangle`, `tetgen`; the latter two are auto-skipped by `tests/conftest.py` when the native library is unavailable.

## Architecture

FES is Python-only since v0.2; the C++ and MATLAB backends were removed.

The package in `fes/` implements the full solver pipeline (import → mesh → assemble → solve → export) for four formulations:

- `em_e_fd` — 3D frequency-domain EM (curl-curl, H(curl))
- `em_ez_fd` — 2D TMz scalar Helmholtz
- `em_e_tl_eig` — 2D cross-section / 3D waveport eigenmodes
- `em_e_qs` — 2D electrostatics (nodal P1)

`em_e_fd_dd` and `em_e_fd_nl` are not ported (`NotPortedError`); `em_h_qs` is rejected (`FormulationUnavailableError`).

`api.py` resolves a `RunPlan` and registers per-formulation drivers in `api.DRIVERS`; `drivers.py` owns the drivers and Touchstone I/O; `cli.py` is a thin front end over `api`.

### Two parallel numerical code paths

This is the single most important structural fact in the repo:

- **Reference path** — `core/assemblers/` plus the reference kernels (`shape.py`, `shape3d.py`, `quadrature.py`, `element_matrix.py`, `dof.py`, `quadrature_tables3d.py`) and the mesh backends (`mesh/mesh2d.py`, `mesh/mesh3d.py`, `mesh/triangle_backend.py`, `mesh/tetgen_backend.py`, `mesh/topology.py`). Reproduces the v0.1 C++ solver down to loop order, DOF numbering, and output formatting.
- **Legacy Python path** — `core/shape_functions.py`, `quadrature_legacy.py`, `jacobian.py`, `dof_legacy.py`, `boundary.py`, `assembly.py`, `core/harmonic_balance.py`, used by `fes/projects/`. Does not feed the four assembled formulations.

Do not mix the two: a fix in `core/shape_functions.py` does not affect the reference path, and a change in `core/assemblers/` does not affect the legacy projects.

Import-time coupling is a real trap: `fes/__init__.py` eagerly imports `fes.core`, and `fes/core/__init__.py` eagerly imports the legacy path (`shape_functions`, `quadrature_legacy`, `dof_legacy`, `boundary`, `assembly`, `harmonic_balance`). An import error anywhere in that chain breaks every CLI command, including `fes version`. When renaming or moving a module, update `core/__init__.py` **and** the legacy modules' intra-package imports together.

### Run planning and dispatch

`api.py` resolves a `RunPlan`; the formulation comes from an explicit CLI flag (`+formula NAME` or an `+em_*` switch), otherwise the model's `#Formula` tag. The C++ `detectFormula` behavior is preserved: a `#Formula em_e_qs` tag also forces field output. `api.DRIVERS` is populated lazily by `drivers.install()`, so `fes info`, `fes models`, and `--dry-run` work without importing NumPy/SciPy or a mesher.

### Source layout

- `fes/cli.py`, `fes/api.py`, `fes/drivers.py`, `fes/paths.py` — CLI, plan dispatch/driver registry, Touchstone I/O, model/mesh discovery
- `fes/core/` — assemblers, reference kernels, legacy assembly
- `fes/mesh/` — `.poly` I/O, Triangle/TetGen ctypes backends, topology, caches
- `fes/post/` — mesh, field, and radiation VTK writers
- `fes/projects/` — legacy Python projects and DNN-GP surrogates
- `mm/`, `n2f/`, `rfm/` — self-contained bundled third-party packages (MIT), outside `fes`, with their own test suites (`tests/test_mm.py`, `tests/test_n2f.py`, `tests/test_rfm.py`)
- `tools/` — native mesher builds (`build_triangle.py`, `build_tetgen.py`, `tetgen_shim.cpp`); `tests/` — unit/integration gates; `data/` — `.poly` models and mesh caches; `dep/` — Triangle/TetGen sources (gitignored, so not present in every checkout)

### Model resolution and mesh caches

`fes/paths.py` resolves a bare model name to `data/<name>.poly` (also `<cwd>/data`, `<cwd>`, or `FES_DATA`; `FES_ROOT` overrides repo-root detection). `fes models` lists available models.

Only 2D meshes are cached. A 2D model loads `<name>.h<N>.mat.npz` (or the legacy `.mat`); if absent it is meshed from the `.poly` and written back as `.npz`. 3D models are re-meshed with TetGen on every run and never cached, matching v0.1. Native library discovery can be overridden with `FES_TRIANGLE` (Triangle) and `FES_TETGEN_LIB` (TetGen shim).

`+poly_cmd` is only the *middle* of the mesher switch string: the driver prefixes `p` and appends `AfeQ` (2D) or `AfeeQ` (3D), so the defaults `q34`/`q` become Triangle's `pq34AfeQ` and TetGen's `pqAfeeQ`. Changing them changes the mesh and breaks numerical agreement with the reference solver.

### `.poly` file format

Standard TetGen PLC sections (nodes, segments, faces; `#FACETS` is accepted as an alias for faces) plus custom `#Regions`/`#Solids` and `#Boundaries` tables. `#Formula` may be the first line or appear after the geometry; parsing is case-insensitive and order-independent. `#Formula` values: `EM_E_FD`, `EM_EZ_FD`, `EM_E_TL_EIG`, `EM_E_QS` (a CLI formulation flag overrides the file).

Material lines under `#Regions N` / `#Solids N` carry no label column; each material's `label` is its 0-based line index and is matched to the mesh's region attribute by that order. `#Boundaries M` lines do carry the label (`<name> <label> <type> [numModes]`). Boundary types: `PerfectE` (PEC), `PerfectH` (PMC), `Radiation`, `WavePort`, `LumpedPort`; tags are canonicalized case-insensitively (`wave_port`/`waveport` both work).

## DNN-GP surrogate conventions (`fes/projects/filter_dnngp.py`)

- 4 independent DNN → ExactGP models (Deep Kernel Learning), one per target: Re(S₁₁), Im(S₁₁), log₁₀|S₂₁|, detrended ∠S₂₁.
- Feature extractor is a mixture-of-experts (`MoEFeatureExtractor`): shared backbone `Linear(8→1000)→ReLU→Linear(1000→1000)→ReLU→Linear(1000→500)→ReLU`; two expert heads `Linear(500→64)`; a gate `Linear(500→128)→ReLU→Linear(128→64)→ReLU→Linear(64→2)` with a learnable softmax temperature blends the experts. GP kernel: `ScaleKernel(RBFKernel(ard, 64) × RBFKernel(ard, 64))`.
- Input features (8-dim): `[εᵣ, f/f_scale, sin(ω), cos(ω), sin(2ω), cos(2ω), sin(3ω), cos(3ω)]` with `ω = 2πf/f_scale` and `f_scale = 10 GHz` by default. Inputs and each output target are `StandardScaler`-normalized before training; the scalers are saved with the checkpoint.
- S11: Re/Im directly. S21: `log10|S|` + detrended unwrapped phase (per-εᵣ linear phase fit stored and restored during reconstruction).
- `mat_file` input expects `epsr`, `freq`, `s11`, `s21`. Generated data is cached as `bilat_dnngp_data.npz`; the adaptive-LHS path uses `adaptive_dnngp_data.npz`. Model checkpoints are `bilat_dnngp.pt` / `adaptive_dnngp.pt`. `save`/`load` use `torch.save` for the full model state (including GP training data) plus scalers. `.npz`/`.pt` outputs are gitignored; regenerate via `bilateral_filter_dnngp(generate=True)`.

## Documentation

- `docs/projects.md` — Python project catalogue, DNN-GP and adaptive-LHS usage, mesh-generation examples
- `docs/bundled_modules.md` — usage, layout, and licensing of the bundled `mm`, `n2f`, and `rfm` packages plus their test coverage
- `mm/README.md`, `n2f/README.md`, `rfm/README.md` — per-package API tours and examples; `rfm/README.md` records intentional deviations from the original implementation and known limitations

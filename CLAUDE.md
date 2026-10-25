# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential commands

The repo ships a venv at `.venv` (created by `./configure` / `make setup`). Activate it once so `python`, `pytest`, and `ruff` work directly:

```bash
source .venv/bin/activate
```

Without activating, prefix every command with `.venv/bin/python -m` (the sandbox has no `python` on PATH; use `python3` or `.venv/bin/python`).

| Task | Command |
|------|---------|
| Setup (venv + editable install + native meshers) | `./configure`  (same as `make setup`) |
| Run full test suite | `make test`  or  `pytest tests/ -q`  (CI equivalent) |
| Run a single test | `pytest tests/test_n2f.py::test_vf_near_field_solver -v` |
| Lint | `ruff check fes tools tests mm n2f rfm setup.py` |
| Run CLI (one frequency) | `python -m fes run WR90 +f 1e10 +poly +p 2` |
| Legacy CLI spelling | `python -m fes WR90 1e10 +poly +p 2` |
| Dry-run (plan only, no solve) | `python -m fes run WR90 +f 1e10 --dry-run` |
| List / inspect models | `python -m fes models` / `python -m fes info WR90 [--json]` |

Test config lives in `pytest.ini` (`testpaths = tests`, `pythonpath = .`, `addopts = -ra`). Markers: `slow`, `triangle`, `tetgen` — the latter two are skipped unless the native Triangle/TetGen shared libraries are built (`tools/build_triangle.py`, `tools/build_tetgen.py` from the gitignored `dep/` submodules).

Lint is phase-1 only (`ruff.toml`): `select = ["E9", "F63", "F7", "F82"]` — correctness errors, no style/typing enforcement yet.

CLI exit codes: `0` success, `1` usage error, `2` formulation not ported, `3` I/O error.

## High-level architecture

**Two independent numerical paths** (the single most important structural fact — read `README.md` "Architecture" before touching either):

- **Reference path** — `fes/core/assemblers/` plus the reference kernels (`shape`, `shape3d`, `quadrature`, `element_matrix`, `dof`, `quadrature_tables3d`) and the mesh backends. It reproduces the v0.1 C++ solver down to loop order, DOF numbering, and output formatting. This path owns the four assembled formulations: `em_e_fd` (curl-curl, default), `em_ez_fd` (2D TMz), `em_e_tl_eig` (eigenmodes), `em_e_qs` (electrostatic).
- **Legacy Python path** — `fes/core/assembly.py`, `shape_functions.py`, `quadrature.py`, `jacobian.py`, `dof.py`, `boundary.py`, `core/harmonic_balance.py`, used by `fes/projects/`. It does **not** feed the four assembled formulations.

Do not mix them: a fix in `core/shape_functions.py` does not affect the reference path, and a change in `core/assemblers/` does not affect the legacy projects.

**Request flow:** `fes/cli.py` (`main()`) parses `+flags` / subcommands (`run`, `sweep`, `mesh`, `sparam`, `eigen`, `info`, `models`) via `fes/core/options.py` (`parse_argv`, `Options`, `ASSEMBLY_TYPES`), builds a `RunPlan` in `fes/api.py` (`plan_run`), resolves the formulation (`effective_assembly`: explicit CLI flag wins, else the `#Formula` tag in the `.poly`), then `dispatch()` looks the assembly up in `fes.api.DRIVERS` (populated lazily by `fes.drivers.install()`). A formulation with no driver raises `NotPortedError`; `em_h_qs` raises `FormulationUnavailableError`. Drivers live in `fes/drivers.py` and hand results to `fes/post/` writers (VTK, Touchstone `.sNp`).

**Formulation phases** (`fes/api.py` FORMULATION_PHASE): `em_ez_fd`=2, `em_e_fd`=6, `em_e_tl_eig`=7, `em_e_fd_dd`=9 (not ported), `em_e_fd_nl`=10 (not ported), `em_e_qs`=11.

**Mesh pipeline:** models live as `.poly` files in `data/`; `#Formula` selects the formulation. Triangle meshes 2D and writes an `.npz` cache next to the `.poly`; TetGen meshes 3D on every run and never caches (matching the v0.1 C++ binary). Native mesher sources are expected under `dep/` and are not vendored as binaries.

**Bundled modules** (independent of the FEM pipeline, MIT licensed, only core deps): `mm` (modal matching), `n2f` (near-field → far-field Huygens/Kottler transforms), `rfm` (Touchstone I/O, Vector Fitting, passivity). Each has its own README and suite under `tests/`.

## Repository conventions

- Angles in radians for numerical solvers; degrees for excitation helpers (relevant to `n2f`).
- Surface patches must be closed with outwardly directed normals for Huygens/Kottler integrals (`n2f`).
- `.poly` format: TetGen PLC sections plus custom trailing sections; `#Regions` (alias `#Solids`) materials match mesh region attributes by line order; `#Boundaries` carries the boundary label.
- Do not add per-package license copies — the root `LICENSE` covers FES and the bundled `mm`, `n2f`, `rfm` packages.

## No CLAUDE.md existed before

This file was created from `README.md`, `pytest.ini`, `ruff.toml`, `setup.py`, `fes/{cli.py,api.py,drivers.py}`, and `configure`. There are no Cursor (`.cursor/`) or Copilot (`.github/copilot-instructions.md`) rule files in this repo. If you have an OpenAI Codex or Gemini CLI config you'd like imported here, reply `/import` to scan what's importable, then `/import --yes=<digest>` to apply it.

# Changelog

## Unreleased

### Bundled modules

- Bundled `rfm` (rational function models), a Python rational-function-model
  toolkit (upstream package name `pyrfm`). It adds the Descriptor Unit Model
  and pole-residue macromodel
  containers, Vector Fitting, Touchstone I/O, passivity checking/enforcement,
  and extraction/conversion utilities, all on the core NumPy/SciPy/Matplotlib
  dependencies. `rfm/README.md` documents the API and the changes from
  upstream.
- Defects in the upstream code that broke its documented workflow were fixed
  during the port: `read_touchstone` now returns `(Nports, Nports, Nfreq)` and
  honours the standard `S11, S21, S12, S22` order; `MA`/`DB` data, wrapped
  records and `THz` units are supported; `vector_fit` is a working
  Gustavsen–Semlyen implementation (upstream's could not fit a known
  three-pole response); `merge_options`, `convert_macromodel_units`,
  `extract_subdum`, `extract_largest_responses` and `plot_dum` were fixed; and
  the unimplemented skimming/multi-port routines now raise `NotImplementedError`
  instead of silently returning empty results.
- New `tests/test_rfm.py` (46 tests); `setup.py` ships `rfm`'s README and
  example Touchstone data, and CI lints `mm`, `n2f` and `rfm`.

### Licensing

- Consolidated the identical MIT license texts from the bundled `mm` and
  `n2f` packages into the single repository-root `LICENSE`. The per-package
  `LICENSE` files were removed, and the packaging metadata now declares MIT.
  This supersedes the AGPL-3.0-or-later relicensing note in v0.2.0.

## v0.2.0 — 2026-10-22

The C++ and MATLAB backends are consolidated into Python. During the port the
two trees were frozen at `v0.1` and CI verified they stayed unchanged; Phase 14
removes them from `main`. The complete three-backend state — including the C++
sources — is preserved at the `v0.1` tag.

### Relicensing

- The repository is now licensed under the GNU Affero General Public License
  v3.0 or later (AGPL-3.0-or-later); it was MIT through v0.1. The repository
  `LICENSE` carries the canonical AGPLv3 text; `m/LICENSE` at the `v0.1` tag is
  the same grant rather than C++ source.
- `py/setup.py` advertises `AGPL-3.0-or-later` in its metadata and trove
  classifier.

### Phase 1 — C++ reference, CLI scaffold, comparison corpus

- A harness builds and runs the frozen C++ binary in a scratch directory and
  records stdout, artifacts, Touchstone arrays and exit status.
- A comparison step checks a candidate run against the golden corpus using
  per-manifest tolerances.
- `py/fes/cli.py`, `py/fes/api.py`, `py/fes/core/options.py` and
  `py/fes/paths.py`: a Python CLI that reproduces the C++ `+flag` grammar and
  resolves the formulation from `#Formula` unless `+formula` is given.
- Comparison manifests for all 31 models, with derived fixtures for models that
  only carry MATLAB-supplied tables.

### Phase 2 — 2D consolidation

- New reference 2D kernels: `fes/mesh/mesh2d.py` (topology and the
  interior-edge numbering), `fes/core/quadrature.py`,
  `fes/core/shape.py` (scalar and H(curl) bases, orders 1-4) and
  `fes/core/dof.py`.
- `fes/core/assemblers/em_ez_fd.py`: 2D TMz scalar Helmholtz with TFE ports,
  port eigenmodes and S-parameters, matching the reference implementation to 1e-8.
- `fes/drivers.py`: the `em_ez_fd` driver, Touchstone reader/writer and merge
  semantics; `fes run` and `fes sweep` now execute 2D models end to end.
- `fes/mesh/io_mat.py`: one reader/writer for the legacy `.h1.mat` and
  `.h1.mat.npz` mesh caches, replacing the inline loader in `io_poly.py`.
- Candidate and per-order comparison corpora in the same layout as the C++ runner.

### Phase 3 — meshing

- `fes/mesh/geometry.py`: the Triangle `.poly` reader and writer, split out of
  `io_poly.py`, with the C++ scanner's prefix-parsing behaviour preserved.
- `fes/mesh/topology.py`: rebuilds the legacy `node/ele/spig/spig2/nlab/elab/slab`
  arrays directly from a Triangle result, so mesh caches no longer need the C
  `IOrMesh` tool.
- `fes/mesh/io_poly.py`: `read_poly` meshes the `.poly` when no cache exists
  (previously an error), and gains `remesh=` and `persist=` options.
- `fes mesh <model>`: new CLI command (and `fes.api.mesh`) that meshes a 2D
  model, reports node/triangle/edge counts and writes `.h1.mat.npz`.
- The mesher reproduces the reference `pq34AfeQ` switches; all 29 2D models with
  a golden log line match its node/triangle/segment counts exactly.
- Removed `py/iormesh/` (the C Triangle wrapper) and `py/Makefile`, so the
  package no longer needs a C/C++ build step. The only C++ left under `py/` is
  the optional C++-dump tooling in `py/tools/`.

### Phase 4 — 3D kernels and topology

- `fes/core/quadrature_tables3d.py` and `quadrature_3d()` in
  `fes/core/quadrature.py`: the reference tetrahedral rules for orders 1-4
  (Keast 1/4/7 and a 64-point Gauss rule; 4, 11, 24 and 64 points), stored
  with the exact doubles.
- `fes/core/shape3d.py`: tetrahedral H1 and H(curl) bases for orders 1-4
  (4/10/20/35 and 6/20/45/84 DOFs), in the per-order ordering and
  with its transform conventions (`invJ` for gradients and H(curl) values,
  `det(invJ) * J^{-1}` for curls).  Two v0.1 quirks are reproduced verbatim:
  `dNs(0,26)` at order 4 carries a spurious `l3` factor, and several H(curl)
  derivative columns are left at zero.
- `py/tools/extract_cpp_kernels.cpp` now dumps the order-4 tetrahedral bases,
  so `tests/kernels/kernels.json` pins all four orders for both reference
  simplexes.  The fixture was regenerated from the frozen C++ binary.
- `fes/mesh/mesh3d.py`: tetrahedral topology in the storage order —
  sorted tetrahedra, the full TetGen edge and face lists, and
  `tetEdges`/`tetFaces` built with `Createmesh`'s lookup semantics, including
  `std::map::operator[]`'s zero fallback for a missing key.  The global
  edge/face numbering is still the mesher's responsibility (Phase 6).
- `fes/core/dof.py`: the 3D DOF layout and per-tetrahedron maps from
  `dof.cpp`, sharing one front end with the 2D maps; `Mesh2D` and `Mesh3D`
  both expose `dim` so the call site does not change.

### Phase 5 — 3D H(curl) assembly (`em_e_fd`)

- `py/tools/extract_cpp_kernels.cpp` now links the untouched `element_matrix.cpp`,
  `coupling.cpp` and `material.cpp` and dumps both `ele_mat` constructors: the
  3D H(curl) bulk matrices (`S`, `T`, `Z`) and the embedded-face matrices
  (`St`, `Tt`, `Sz`, `Tz`, `G`, `STt`).  The committed fixture pins every entry
  for orders 1-3 on a skewed tetrahedron and one of its faces, so the tests
  exercise the full 3D Jacobian and the face frame rather than the reference
  simplex.
- `fes/core/element_matrix.py`: the per-element matrices, including the
  reference accumulation and scaling order (`Z` is built from the *unscaled*
  mass matrix), `epsr2 = -tand*epsr`, the `int_node`-flipped normal of the
  embedded face frame, and `arma::diagmat`'s diagonal-only `STt`.  The
  `quad(p_ord + 1)` ceiling means v0.1 can only assemble orders 1-3; the port
  raises the same error for order 4 instead of inventing a rule.
- `fes/core/assemblers/em_e_fd.py`: a line-by-line port of
  `assemble_em_e_fd.cpp` and `compute_waveport_modes` — tetrahedral assembly on
  the *global* DOF upper triangle, PEC/ABC faces, the port generalized
  pencils, the TFE modal reduction, the MUMPS-symmetric solve and
  `extract_solution_full`.  Two deliberate differences are recorded in
  `docs/tolerances.md`: the port eigenproblem is resolved densely with
  LAPACK (the ARPACK call truncates its Krylov space at `ncv == numtot`), and
  the solve uses SuperLU on the expanded one-triangle matrix.
- WR90 comparison: the bulk matrix and all four port blocks reproduce the
  reference implementation to `1e-11`, and S-parameters match the frozen
  golden to `2.1e-6` abs / `6.0e-6` rel — the residual is the Ritz-vector
  truncation. The `WR90` manifest therefore overrides the same-mesh tier
  (`sparam_abs 5e-6`, `sparam_rel 1e-5`, `eigen_rel 1e-5`); the tolerance
  table gained a per-model entry for it.
- `tests/test_em_e_fd.py` covers the structural reference bookkeeping
  (`wave_portIds`, `Nonwave`, `dofreal`), the goldens, the solution expansion,
  and the two geometry-order subtleties: the `cdof.v(i) <= cdof.v(j)` test runs
  on global DOF numbers, and Armadillo's `.t()` is the conjugate transpose.

### Phase 6 — 3D meshing and the `em_e_fd` driver

- `py/tools/tetgen_shim.cpp` and `py/tools/build_tetgen.py`: the vendored
  TetGen sources (`dep/src/tetgen`, the same ones behind `libtetgen.a`) are
  compiled into `py/build/libfetgen.*` together with a small C shim. TetGen's
  API is C++ — `tetgenio` has a virtual destructor — so the shim keeps the
  objects opaque, exposes the output arrays as plain pointers and turns the
  `TETLIBRARY` abort path into a return code.
- `fes/mesh/tetgen_backend.py`: the ctypes binding. It reproduces
  `pre_processor.cpp`'s switch grammar (`pqAfeeQ`, or `p<cmd>AfeeQ` for an
  explicit `+poly` argument), calls `load_poly` and returns the 1-based
  arrays TetGen emitted, `first_number` included. `TetGenUnavailable` is
  raised when the library is absent, so the mesh tests skip cleanly in CI.
- `fes/mesh/mesh3d.py`: `Mesh3D.from_tetgen` implements `Createmesh`
  (offset by `first_number`, raw face markers, a tet's label read from the
  *flat* attribute array, `maxLab` when TetGen produced no attributes) and
  `mesh3d_from_poly` ties it to `read_poly_metadata`. WR90 now reproduces the
  mesh **exactly** — every node, edge, face, tetrahedron, marker, label,
  `tetEdges`, `tetFaces`, `facEdges` and `facAdjTet` — which
  `tests/test_tetgen_mesh.py` checks with no tolerance.
- `fes/mesh/poly_header.py`: a `#Boundaries` line without a mode count now
  defaults to 1, matching `bc::bc() : num_modes(1)`. PEC entries in the
  tables carry 1, not 0.
- `fes/api.py` and `fes mesh`: 3D models are meshed with TetGen and reported
  as nodes/tetrahedra/faces/edges. No cache is written, matching the C++
  binary, which re-meshes on every run.
- `fes/drivers.py`: the `em_e_fd` driver — mesh once, assemble and solve each
  frequency, write `<project>.s<N>p` with the column-major ordering.
  `SolverResult` reports 3D port counts and DOFs alongside the 2D ones.
- `py/tools/py_runner.py` now records `port_beta_re`, `port_beta_im` and
  `port_shift` from the 3D systems, the arrays the stdout parser reads off
  the console.
- `fes/core/options.py`: `OptionSet.freqs()` mirrors `eq_sys` exactly — below
  two frequencies the single `freq` is repeated, and from two upwards the range
  is `linspace(l_freq, h_freq, n_freqs)` with no ordering guard, so a
  non-increasing sweep still runs every point. The C++ binary builds the mesh
  once per process and loops over frequencies *inside* `eq_sys`, which is the
  structure the driver reproduces.
- End-to-end WR90 comparison through the corpus harness: S-parameters to
  `2.1e-6` abs / `6.0e-6` rel, betas to `1.5e-6` rel and the shift to
  `8.3e-7` rel, within the manifest's Phase-5 tolerances.

### Phase 7 — 2D cross-section eigenmodes (`em_e_tl_eig`)

- `fes/core/element_matrix.py`: `element_matrices_2d_mixed` and
  `triangle_jacobian` port the `cDim == 2` branch of the first `ele_mat`
  constructor — `St`/`Tt` from the H(curl) basis, `Sz`/`Tz` from the scalar
  one, `G = Nv^T dNs`, all scaled by `|detJ|`.  The material scaling stays at
  the call site, as in the reference implementation, because the same element
  matrices serve both `1/mur` and `epsr`.
- `fes/core/assemblers/em_e_tl_eig.py`: the mixed H(curl)/H1 block pencil, the
  PEC Dirichlet set (including the `lev*nMeshEdges + edge` higher levels),
  the shift `-k0^2 epsr_max mur_max`, and the eigensolver pipeline.
  It reproduces `eigen.cpp` exactly, including the orthogonalisation matrix
  being `diag(I_numt, 0)` rather than the identity: `TT[zz,zz]` is never
  assigned, so Armadillo's `solve` falls back to the SVD least-squares answer
  for a zero matrix.  The dense path resolves the projected operator with
  LAPACK and applies the three sorts; see `docs/tolerances.md`.
- Systems above `DENSE_LIMIT` switch to a SuperLU factorisation plus
  `scipy.sparse.linalg.eigs` on the same operator.  That is the path that
  solves `Coaxial` (156k DOFs) in 43 s / 2.4 GB where the C++ binary is killed
  by the OOM killer; the two paths agree to `6.6e-12` relative on WR90_TL.
- `fes/drivers.py`: the `em_e_tl_eig` driver; `SolverResult.summary()` prints
  the mode table (`beta`, `alpha`, `kc` with six significant digits).
  The stdout recorder captures the modes as `eigen_beta`, `eigen_alpha` and
  `eigen_kc`.
- Comparison: the six solved models match the corrected reference to `4.4e-6`
  relative on `beta` and `2.4e-6` on `kc` — the goldens are parsed from
  six-significant-digit stdout, so the manifests use `eigen_rel = 1e-5`.  All
  seven `em_e_tl_eig` manifests point at `cxx_fixed`; `Coaxial`'s capture
  records the SIGKILL and the Python side is gated separately as the documented
  OOM deviation.
- `tests/test_em_e_tl_eig.py` and a new driver test cover the DOF bookkeeping,
  the failure messages, `solve=False`, the quadrature ceiling, the ordering
  pipeline, the dense/sparse equivalence and the corpus.

### Phase 8 — mesh and field export

- `fes/post/vtk.py`: `mesh::Savefield` ported byte for byte. The writer runs
  from the driver right after meshing, before assembly, which is observable in
  2D: only the PLC edges exist at that point, so the mesh `.vtk` carries no
  derived interior edges. 3D files hold every tetrahedron and face; 2D files
  hold every triangle and each boundary segment. Region and boundary labels
  reproduce the `(int)` casts, including `UINT_MAX` printing `-1`.
- `fes/post/field.py`: `field::field` ported for the 2D `em_ez_fd` and 3D
  `em_e_fd` branches — the `Solution data` mesh, the `sqrt(2)` RMS amplitude
  factor, node averaging (`NodeVal /= SumTimes`), and the `_norm/_abs/_real/
  _imag_` blocks for the summed solution (`cnt = 0`) and each solution column
  (while fewer than ten columns). The 3D branch adds the H blocks with
  `H = j/(2*pi*mu0*f) * curl(E)`. Six significant digits of the `float`
  values and the `<freq>` tag (`%.6g`) are formatted like `std::ostream`.
- `fes/core/assemblers/em_ez_fd.py`: `extract_solution_full` expands the
  reduced TFE solution back to the full DOF vector exactly like
  `solver::extract_solution_full`, so the 2D field export sees the same
  solution the C++ binary writes.
- `fes/drivers.py`: every driver now writes `<project>.vtk` (including
  `em_e_tl_eig`, whose C++ path `continue`s before the solver and therefore
  writes only that file), and `+field` writes `<project>_<freq>.vtk`.
- Comparison: the mesh `.vtk` is byte-identical for `WR90` (1196 B) and
  `ModelScatteringSquare` (368 B), and the 2D `+field` file for
  `ModelScatteringSquare` is byte-identical (901 B). The 3D field file has the
  same 52 blocks at the same positions and `E_norm_0` matches exactly; the
  remaining blocks inherit the documented 3D solution deviation
  (`3.0e-4` abs), measured at `1.0e-2` abs on a block scale of `1.7e3`
  (`5.9e-6` of the block maximum) and `4.8e-5` relative on near-cancelling
  entries. The criterion `|d| <= 2e-6*max|block| + 1e-5*|value|` bounds every
  entry. Near-zero values that print as `2.1e-15` versus Python's `0` account
  for the remaining text difference.
- `+rad` radiation-pattern export is not ported yet: `radiation.cpp` has no
  corpus model (only 3D models carry `Radiation` boundaries and none of them
  requests `+rad`), so there is no golden to gate it against. The drivers
  report the gap in `SolverResult.notes`.

### Phase 11 — 2D electrostatics (`em_e_qs`)

- `fes/core/assemblers/em_e_qs.py`: the `msh->nTetras == 0` branch of
  `assemble_em_e_qs.cpp`. One DOF per mesh node — the C++ binary always uses
  linear nodal elements in 2D, so `+p` changes nothing — and the stiffness is
  `area * epsr * grad(N_i) . grad(N_j)` per triangle. `epsr` is read from the
  material table keyed by the `#Solids` line index, using the element's region
  attribute as the key (the two coincide only when the file lists one solid
  per region in order). Every `PerfectE` PLC
  segment contributes its endpoints as Dirichlet DOFs with the `+volt` value
  (default 0), the rows and columns are eliminated sequentially in ascending
  DOF order (observable: each column updates the whole right-hand side), and
  the non-symmetric complex system is solved with SuperLU in place of MUMPS.
  With no prescribed voltage the assembler raises the `Null Right Hand Side`
  error.
- `fes/post/field.py`: the QS field branch. `field.cpp` loops over
  `msh->nTetras`, which is zero for a 2D mesh, so `SumTimes` stays zero and
  the final `NodeVal.row(i) /= SumTimes(i)` turns every component into NaN.
  The port writes the same file byte for byte — `CELLS 0 0`, one
  `Phi_0_[V]` scalar block of `nan`, and the `_E_` vector blocks.
- `fes/api.py`: a `#Formula em_e_qs` tag turns on `field` export through
  `detectFormula`'s side effect; an explicit `+formula em_e_qs` does not,
  because the C++ binary only calls `detectFormula` when the default assembly
  is `em_e_fd`.
- `fes/drivers.py`: the `em_e_qs` driver meshes, assembles once per frequency
  and writes `<project>.vtk` plus the field file when requested; electrostatics
  have no ports, so no Touchstone file is written.
- Comparison: `CapSense`, `CapSense2`, `CapSense2_1` and `coax` reproduce the
  mesh and field VTK by size and SHA-256. `tests/test_em_e_qs.py` additionally
  pins the P1 stiffness, the material lookup, the order being ignored and the
  `Null Right Hand Side` path.
- `em_h_qs` stays unavailable on purpose: `equation_system.cpp` has no case
  for it, so the C++ binary aborts with `Formulation not available for this
  version`. Planning still resolves it (the flag parses), but `fes run` raises
  `FormulationUnavailableError` and exits 1, like the C++ binary.
- Domain decomposition (Phase 9) and harmonic balance (Phase 10) remain
  unported because neither path can be gated. The documented
  `+nl H mtrl kerr relax` spelling stores `"H"` under the `nl` key, and
  `apply_cli`'s `getB` accepts only `"1"`, so harmonic balance never activates;
  forcing `+formula em_e_fd_nl +n_harm 3 +nl` aborts with
  `Mat::operator(): index out of bounds`. Domain decomposition needs the
  generic `+n_dd N +dd` spelling (with `+dd N`, `dd` is set to the string
  `"N"` and reads back false). Two or more subdomains then abort with
  `znaupd error = -9` at every order tried, and a single subdomain completes
  with `S = I`, which contradicts the non-DD WR90 result — the DD assembly
  is not a usable reference.

### Phase 12 — homogeneous refinement (`+h`)

- `fes/mesh/mesh2d.py`: `Mesh2D.refine_homogeneous()` ports the 2D branch of
  `mesh::refine_homogeneous`. Every triangle is split into four at its
  mid-edge nodes and every PLC segment into two; region and boundary labels
  are inherited, while the derived interior-edge connectivity is dropped for
  the assembler to rebuild, as `facEdges.reset()` does.
- Mid-edge numbering matches the reference implementation, which refines before
  `normalize_order`: faces are visited in raw Triangle order, each visiting
  local edges `(0,1), (1,2), (2,0)` with the endpoint key sorted, and a new
  node index is assigned on first encounter. `mesh_from_poly` gained `h_ord`
  and now refines before sorting, exactly like `project::project`.
- `fes/drivers.py`: the three 2D drivers (`em_ez_fd`, `em_e_tl_eig`,
  `em_e_qs`) pass `options.h_ord`. The 3D driver is unchanged because
  `refine_homogeneous` returns immediately when `nTetras > 0`.
- `fes mesh` rejects `+h`: the cache format carries its level in the file
  name and only the unrefined h1 arrays can be written. Refinement happens at
  solve time, like the reference implementation.
- Comparison: the `CapSense2_1` corpus case now runs `+h 1`. The QS test
  reproduces the refined mesh (827 -> 3283 nodes, 1630 -> 6520 triangles,
  157 -> 314 PLC segments) and both VTK files by SHA-256.
  `tests/test_mesh_refine.py` pins the mid-node and sub-triangle ordering on
  a hand-built mesh, label inheritance, two-level counts and the driver
  wiring.

### Phase 13 — incident plane wave (`+einc`) and radiation pattern (`+rad`)

- `fes/core/assemblers/em_e_fd.py`: the portless branch now assembles the
  incident-wave right-hand side (`kEinc *= k0`, `polEinc /= sqrt(2)`, and
  `einc_face_vector * i*k0` integrated over every `Radiation` face), zeroes the
  Dirichlet rows and columns with a unit diagonal, and solves the full
  `(dofnum, 1)` system. An all-zero right-hand side raises the new
  `fes.core.assemblers.errors.NullRightHandSideError`.
- `fes/post/radiation.py`: ports `radiation.cpp` including the lower-order
  `quad(p_ord)` far-field rule, `fullSol = sum(sol, 1) * sqrt(2)`, the
  `RefFace` reference triangles, `RefNorm = -n`, the accumulating `Prad`
  integral, `G`, and the einc/non-einc `DirCoeff` and `delta`. Note that
  `arma::dot` does **not** conjugate, so its projection maps to `numpy.dot`;
  using `vdot` leaves the far field with a radial component. `_render`
  reproduces `rad::Savefield` line for line, including the `(float) rMag * ...`
  cast, the newline-free cells block and six significant digits.
- `fes/drivers.py`: both `em_e_fd` and `em_ez_fd` call `write_radiation_vtk`
  when `+rad` is given. The C++ binary runs `rad` for 2D formulations too, where a
  mesh has no faces: Python then writes the same degenerate file (`Prad = 0`,
  `DirCoeff = inf`, `Dir = nan`), byte-identical to the C++ output.
- `fes/cli.py`: `NullRightHandSideError` prints `### ERROR: Null Right Hand
  Side ###` and exits 1, matching `main.cpp`; other `RuntimeError`s still
  propagate.
- Comparison: the frozen corpus cannot reach either path (`WR90` has waveports;
  the 2D `Radiation` boundaries are scalar), so Phase 13 adds the committed
  fixture `tests/fixtures/models/RadCube.poly` (vacuum cube, six radiation
  faces, no ports), the supplementary manifest `tests/manifest_rad/RadCube.json`
  and `tools/rad_corpus.py`, which captures `golden_rad/RadCube/p{1,2,3}` from
  the v0.1 binary. At every order the parsed far-field blocks are exactly equal
  and `points` agree to `1.5e-15`; `Prad`/`maxDir` match the stdout to
  `8.8e-8` relative. The 2D degenerate file is byte-identical.
- `tests/test_rad.py` gates all of it, including a regeneration test, the CLI
  abort, and a writer/parser round trip.

### Phase 14 — Python-only consolidation

- Removed `cpp/` and `m/` from `main` (the v0.1 tag preserves every file, and
  the C++ binary can be rebuilt from it). Domain decomposition
  (`em_e_fd_dd`) and nonlinear harmonic balance (`em_e_fd_nl`) stay unported:
  the v0.1 C++ binary aborts inside both assemblers on every input tried
  (`znaupd error = -9`, index-out-of-bounds, `coupling order not implemented
  yet`), so there is no reference result to gate against. `em_h_qs` is rejected
  by the binary's own dispatcher.
- Removed the C++ build and freeze workflows
  (`.github/workflows/build.yml`, `cxx-freeze.yml`); `py-ci.yml` is the only
  workflow.
- The root `Makefile` is Python-only (`py-setup`, `py-test`, `test`, `help`);
  root `setup` forwards to `py/configure`, and `./setup --compiler` / `--m`
  explain that those backends moved to the `v0.1` tag.
- Version 0.2.0 in `VERSION`, `py/setup.py`, `fes.__version__` and the
  `py/configure` banner.
- `.gitignore` drops the `dep/` and `cpp/` entries; `README.md` and
  `docs/tolerances.md` describe the Python-only repository and how to obtain
  the C++ sources.
- Comparison goldens and tests are retained unchanged. Tests that need the
  frozen binary skip when `cpp/build/fes` is absent; point `FES_SOLVER` at a
  binary built from the `v0.1` tag to run them.

### Fixed

- The 2D interior-edge connectivity port now sorts edge endpoints before
  inserting them. The v0.1 C++ code inserted unsorted pairs and looked them up
  sorted, collapsing DOFs and making the system singular; the frozen goldens
  for the eight 2D S-parameter models are therefore unusable (`S = -I` for
  BilatFilter and TwoPosts), as are the frozen `em_e_tl_eig` mode arrays.
  Regenerated goldens live in `tests/golden/cxx_fixed` and the affected
  manifests declare `"golden": "cxx_fixed"`. See `docs/tolerances.md`.
- `KoshibaCirculator26` and `KoshibaCirculator28` now run at `+p 3`, the first
  order at which their ports resolve a mode; the frozen `+p 2` capture held
  only an exit status. Both join the corrected corpus and match to `1e-8`.

## v0.1.0 — 2026-10-21

First tagged release. This tag is the final snapshot of the three-backend
repository before the Python-only consolidation planned for v0.2.

### Backends

- **C++ (`cpp/`)** — three-dimensional H(curl) tetrahedral time-harmonic FEM with
  Nédélec elements of order 1–4; PEC, ABC, waveguide ports, lumped ports, and
  plane-wave excitation; two-dimensional TMz; transmission-line eigenmodes;
  electrostatics; additive Schwarz and Schur domain decomposition; Kerr harmonic
  balance for 1–7 harmonics; TFE port modes; S-parameters and Touchstone output;
  passivity check; VTK field and radiation-pattern export; MUMPS direct solve;
  ARPACK shift-invert eigen solve; Jacobi-preconditioned complex GMRES; Triangle
  and TetGen meshing.
- **Python (`py/`)** — two-dimensional scalar and vector FEM, waveguide ports and
  S-parameters, ABC/Dirichlet/Neumann/DD boundary terms, partial Schur support,
  partial harmonic balance, example projects, and the DNN-GP surrogate.
- **MATLAB (`m/`)** — research reference implementation, IGES/NURBS toolbox,
  COMSOL `.mphtxt` reader, ferrite harmonic balance, DG time stepping, nonlinear
  solver, geometry writers, and DD/nonlinear validation scripts.

### Known limitations

- The Python backend is incomplete: no 3D, no TetGen, no MUMPS, no Touchstone
  writer, no VTK export, no TFE, and only partial harmonic balance.
- MATLAB requires MATLAB or Octave. The bundled `IOrMesh` and `triangle`
  executables are macOS arm64 binaries.
- Native dependencies (OpenBLAS, ARPACK-NG, MUMPS, Triangle, TetGen, Armadillo)
  are not included. Run `./setup` on macOS or Linux.
- Python 3.9+ is declared; the core test suite was verified on Python 3.14. The
  optional `viz` extra requires a VTK wheel (available for 3.14, not 3.15).
- `make test` is a generic smoke harness: it runs every model at `+p 1` and the
  default 5 GHz. Fifteen models require a model-specific frequency, polynomial
  order, formulation, or excitation and fail under those generic defaults:
  `CapSense`, `CapSense2`, `CapSense2_1`, `CircKoshiba26`, `CircKoshiba26_3`,
  `CircKoshiba26_5`, `CircKoshiba28`, `Coaxial`, `KoshibaDisc`, `WR90`,
  `ModelInfiniteScatteringDD`, `ModelScatteringDD`, `ModelScatteringDDborder`,
  `ModelScatteringSquareFlocket`, and `coax`. They run successfully with their
  documented options (for example, `WR90` at `+f 1e10 +p 2`).

### Verification at tag time

- C++: `make build` succeeds; `make test` passes 16 of 31 models under the
  generic defaults, with the 15 expected harness failures listed above.
- Python: `pytest tests/` passes 15 of 15 tests on Python 3.14.8.

### Quick start

```bash
./setup                 # build native dependencies into dep/
make build              # build cpp/build/fes
make test               # mesh/run all data/*.poly models
make py-setup           # create py/.venv and install the Python package
make py-test            # run the Python test suite
```

### License

MIT. See `LICENSE`.

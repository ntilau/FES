# Bundled solver modules: `mm`, `n2f` and `rfm`

Three independent Python packages live at the repository top level, alongside
the `fes` package. They are self-contained and do **not** use either FES
numerical path (neither the reference assemblers nor the legacy
`fes.core.assembly` path), and they do not touch the `.poly`/mesh pipeline.

| Package | Purpose |
|---------|---------|
| `mm` | Modal-matching (mode-matching) generalised scattering matrices for rectangular waveguide structures |
| `n2f` | Near-field to far-field transformation by Huygens/Kottler integration, with SVD model-order reduction |
| `rfm` | Rational function models: Touchstone I/O, Vector Fitting, pole-residue macromodels, passivity checking |

All three packages are MIT-licensed (Copyright (c) 2026 Laurent Ntibarikure).
Their license texts are identical and have been merged into the single
repository-root [`LICENSE`](../LICENSE), which covers FES and the bundled
packages alike; there are no per-package license copies.

Each package carries its own README with the full API tour and examples:
[`mm/README.md`](../mm/README.md), [`n2f/README.md`](../n2f/README.md) and
[`rfm/README.md`](../rfm/README.md). The sections below summarise usage; the
sub-READMEs are the detailed references.

## `mm` — modal matching

The upstream flat layout is preserved as a package:

- `mm.core` — numerical routines (modes, coupling integrals, cascades,
  GSM condensation).
- `mm.lib` — one wrapper per MATLAB `lib/*.m` function.
- `mm.scripts` — benchmark entry points (`BifurcationE`,
  `BifurcationH`, `Riblet`, `HildebrandHalf`, `HildebrandSemiAuto`,
  `HildebrandFull`); the HFSS reference CSVs are bundled alongside.
- `mm.run` — the original `argparse` CLI, available as
  `python -m mm`.

```python
from mm.scripts import Riblet

S, sinfo, err = Riblet(plot=False)
print(len(S), "frequency points, fatal:", err.get("fatal"))
```

```bash
.venv/bin/python -m mm BifurcationE --no-plot
.venv/bin/python -m mm hildebrandfull        # aliases are case-insensitive
```

Only NumPy and Matplotlib (already core FES dependencies) are needed;
Matplotlib is imported lazily inside the plotting helpers.

## `n2f` — near-field to far-field

The upstream `pyn2f` subpackage layout is preserved:

- `geometry` — arrays, boxes, spheres, triangulated spheres.
- `transforms` — Cartesian/spherical conversions and rotation matrices.
- `scalar` — scalar near-field and N2F solvers.
- `vector` — vector (Kottler) solvers and low-rank operators.
- `utils`, `plotting` — angle helpers, error metrics, 2-D/3-D plots.

```python
import numpy as np
from n2f import (
    build_array, build_box, get_box_dim, get_box_vectors,
    sf_excitations, sf_nf_solver, sf_nf2ff_solver, sf_compute_gain, deg2rad,
)

wl = 1.0
array_pos = build_array(wl, 3, 0.5, 3, 0.5)
dims = get_box_dim(wl, array_pos, 0.5, 0.1, 0.5)
box_pos, box_n, dS, _ = build_box([1] * 6, *dims, 1, 0, 0)
r_mag, n_dot_r = get_box_vectors(array_pos, box_pos, box_n)
psi, del_psi = sf_nf_solver(wl, sf_excitations(wl, array_pos, 0, 0),
                            r_mag, n_dot_r)
theta = deg2rad(np.linspace(-90, 90, 181))
far = sf_nf2ff_solver(wl, theta, deg2rad([0, 90]), box_pos, box_n, dS,
                      psi, del_psi)
gain = sf_compute_gain(far)
```

`plotting` is optional at import time: if Matplotlib is unavailable, the
package still imports and exposes the numerical API (the plotting names
become `None`).

## `rfm` — rational function models

The upstream `pyrfm` layout is preserved (renamed `rfm`):

- `rfm.dum`, `rfm.macromodel` — Descriptor Unit Model and pole-residue
  macromodel containers.
- `rfm.fitting` — Vector Fitting (`vector_fit`, `vector_fit_refined`,
  `vector_fit_inf_dc0`, `VectorFit`); the skimming/multi-port routines are
  retained as explicit `NotImplementedError` stubs.
- `rfm.io` — Touchstone reader/writer and the (stub) CSV/MOD import helpers.
- `rfm.utils` — sub-model extraction, unit conversion, option merging,
  plotting and JSON/Touchstone export.
- `rfm.validation` — positive-real/Hamiltonian passivity checks and
  enforcement.

```python
import rfm

dum = rfm.read_touchstone("rfm/data/example_model.s2p")   # S: (Nports, Nports, Nfreq)
poles, residues, d = rfm.vector_fit(dum.f, dum.S[1, 0, :], max_poles=10)

model = rfm.Macromodel(representation="pole-residue")
model.Nports = 1
model.poles, model.residues, model.dc_gain = poles, residues, d

is_passive, passive = rfm.check_hamiltonian(model, dum.f)
rfm.export_rfm_model(model, "fit.json")
rfm.export_rfm_touchstone(model, dum.f, "fit.s1p", s_param_index=(1, 1))
```

Models are evaluated as `H(jf) = d + sum_k r_k / (jf - p_k)` in the caller's
frequency units. Vector Fitting normalizes the band internally and rescales
poles and residues back, so Hz, kHz and rad/s inputs all work.

Run the bundled walkthrough with `python -m rfm.example_usage`.

The bundle fixes several upstream defects that broke this workflow; the list
is in [`rfm/README.md`](../rfm/README.md#changes-from-upstream). The most
important are the Touchstone orientation (`(Nports, Nports, Nfreq)` instead
of `(Nports, Nfreq, Nports)`), the standard `S11, S21, S12, S22` port order,
support for `MA`/`DB` data and wrapped records, and a working
Gustavsen–Semlyen Vector Fitting core (the upstream version could not fit a
known three-pole response). `Y`, `Z`, `H` and `G` Touchstone files raise
`ValueError`.

## Tests

All three packages have dedicated suites in the default `pytest` run (no native
meshers or ML extras required):

- `tests/test_mm.py` — modal primitives, mode ordering and symmetry, wave
  numbers/delay matrices, coupling integrals, port-block bookkeeping,
  frequency-sweep and junction validation, `ExtractSingleS`, CLI alias
  resolution, and the `BifurcationE`/`BifurcationH`/`Riblet` benchmarks
  (frequency counts and S-matrix shapes).
- `tests/test_n2f.py` — transforms and rotation matrices, utility/error
  metrics, array/box/sphere/triangulated-sphere geometry, sampling helpers,
  the scalar Huygens transformation (validated against the direct
  point-source pattern to < 5%), operator/solver consistency, and the vector
  direct far-field solver and gain.
- `tests/test_rfm.py` — Touchstone port ordering, formats (`RI`/`MA`/`DB`),
  wrapped records and round trips, recovery of a known rational response by
  Vector Fitting, pole-count and stability handling, the containers,
  extract/convert/merge utilities, JSON and Touchstone export, passivity
  checking/enforcement, the unimplemented-method guards, and plotting smoke
  tests.

Two vector routines are marked `xfail` because the upstream code is not
runnable as bundled: `vf_nf_solver` accumulates complex fields into real
arrays, and `vf_n2f_op_fields` builds operators of the wrong width. The
scalar `sf_nf2ff_solver` likewise stores its result in a real array, dropping
a small imaginary part; the tests compare against its real part.

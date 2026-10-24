# Python project examples

The `fes.projects` package carries the legacy Python simulation projects and
the DNN-GP surrogate models. They use the legacy assembly path
(`fes/core/assembly.py` et al.), not the reference assemblers.

## Quick start

```bash
./configure                    # create .venv and install the package
source .venv/bin/activate      # activate so `python` works directly
python -c "from fes.projects import run_waveguide; run_waveguide()"
```

## Run a project

```python
import numpy as np
from fes.projects import run_waveguide, bilateral_filter
from fes.post.plot import plot_field

# Full S-parameter sweep
freqs, Sparams = bilateral_filter(n_freqs=41)

# Single-frequency field plot
sys, mesh = run_waveguide()
plot_field(sys, mesh, field="u", component="abs", cmap="jet")
```

## DNN-GP surrogate model

```python
from fes.projects import bilateral_filter_dnngp

# Train from pre-generated .mat data (50 epsr x 81 freq)
model, metrics = bilateral_filter_dnngp(
    mat_file="bilat_all_sparams_50epsr.mat", n_epochs=500,
)
print(f"S11: {metrics['s11_rel_pct']:.2f}%  S21: {metrics['s21_rel_pct']:.2f}%")
# -> S11: 0.19%  S21: 0.21%  (both < 1% relative error)

# Predict S-parameters at new (epsr, freq) points
import numpy as np
X_new = np.column_stack([np.full(81, 2.15),
                         np.linspace(138e9, 158e9, 81)])
s11_pred, s21_pred = model.predict(X_new)

# Load a saved checkpoint
from fes.projects.filter_dnngp import BilateralFilterDNNGP
model = BilateralFilterDNNGP().load("bilat_dnngp.pt")
```

## Adaptive LHS — automatic convergence

```python
from fes.projects import adaptive_lhs_bilateral_filter

# Starts with 5 LHS epsr values, adds 5 per iteration,
# stops when both S11 and S21 relative error < 1%
model, history = adaptive_lhs_bilateral_filter(
    n_initial=5, batch_size=5, n_test=5,
    target_rel_error=1.0, max_n_epsr=50,
)
```

`adaptive_lhs_bilateral_filter` uses sequential Latin Hypercube Sampling to
add εᵣ training values until the surrogate reaches a prescribed accuracy
target. At each iteration, the unoccupied region of εᵣ ∈ [2.0, 2.2] is
partitioned into equal-width intervals and one point is sampled per interval —
true LHS stratification on the remaining space. A permanently held-out test set
(5 evenly-spaced εᵣ values) measures generalization error.

**Result** (25 training εᵣ values across 4 batches):

| Iteration | Train εᵣ | S₁₁ rel error | S₂₁ rel error |
|-----------|----------|:-------------:|:-------------:|
| 0 | 5 | 1.118% | 0.592% |
| 1 | 10 | 2.522% | 3.843% |
| 2 | 15 | 4.294% | 2.964% |
| 3 | 20 | 6.411% | 4.589% |
| **4** | **25** | **0.272%** | **0.258%** |

The non-monotonic early errors are expected — with few εᵣ samples the DNN-GP
optimizer can settle in worse local minima as the dataset grows. Once
sufficient coverage is reached (here 25 εᵣ), the model generalises sharply
below the 1% threshold.

Outputs: `adaptive_convergence.png` (error × cost curves),
`adaptive_dnngp_results.png` (final validation figure).

## Projects

| Function | Description |
|---|---|
| `run_electrostatics` | Electrostatic potential |
| `run_waveguide` | Rectangular waveguide S-parameters |
| `modal_analysis_rectangular` | TE mode cutoff frequencies |
| `modal_analysis_open_strip` | Open microstrip analysis |
| `bilateral_filter` | Finline filter frequency response |
| `bilateral_filter_hb` | Harmonic balance (Kerr nonlinearity) |
| `two_post_filter` | Two-post waveguide filter |
| `two_post_filter_hb` | Two-post filter harmonic balance |
| `circulator` | Ferrite circulator S-parameters |
| `circulator_imp` | Intermodulation products |
| `circulator_ddschur` | DD Schur complement circulator |
| `scattering_dd` | Wave scattering with ABC |
| `scattering_dd_iterative` | Iterative Schwarz DD solver |
| `scattering_full_field` | Full-wave scattering |
| `thermal_distribution` | Heat conduction |
| `thermal_distribution_dg` | Discontinuous Galerkin heat |
| `coaxial_capacitance` | Coax cable capacitance |
| `capacitive_clearance` | Capacitive sensor |
| `bilateral_filter_dnngp` | Deep Kernel Learning surrogate — S₁₁ 0.19%, S₂₁ 0.21% relative error |
| `adaptive_lhs_bilateral_filter` | Adaptive LHS-driven convergence to prescribed error target |

## Mesh generation (`.poly` → `.h1.mat`)

`.poly` files are geometry descriptions for the
[Triangle](https://www.cs.cmu.edu/~quake/triangle.html) mesh generator. The
solver meshes them in-process through the bundled Triangle library and caches
the result as `<name>.h1.mat.npz`; models whose `.poly` has no cache are meshed
automatically on first use.

```bash
source .venv/bin/activate      # activate so `python` works directly

# Mesh one model and write <model>.h1.mat.npz next to its .poly
python -m fes mesh WaveGuide_py

# Choose your own Triangle switches (default: pq34AfeQ)
python -m fes mesh WaveGuide_py +poly_cmd q33a1

# Report counts without writing a cache
python -m fes mesh WaveGuide_py --no-write --json

# Write the cache somewhere else
python -m fes mesh WaveGuide_py --out /tmp/meshes
```

Default Triangle arguments are `pq34AfeQ` (quality mesh, 34° min angle, region
attributes), matching the v0.1 C++ binary. `+poly_cmd` replaces the switch
string after the leading `p`.

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib, pyvista
- torch, gpytorch, scikit-learn (for DNN-GP surrogate)

## See also

- [`docs/bundled_modules.md`](bundled_modules.md) — the independent
  `mm` (modal matching) and `n2f` (near-field → far-field) packages.

# pyFES

Python Finite Element Software for electromagnetics.

Solves 2D Helmholtz / scalar wave / electrostatic / thermal problems on triangular meshes, with support for waveguide ports, domain decomposition (Schur complements), harmonic balance (Kerr nonlinearity), and ferrite circulators.

## Quick start

```bash
# Virtual environment + install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Or using setup.sh
./configure
```

## Run a project

```python
import numpy as np
from pyfes.projects import run_waveguide, bilateral_filter
from pyfes.post.plot import plot_field
from scipy import sparse

# Full S-parameter sweep
freqs, Sparams = bilateral_filter(n_freqs=41)

# Single-frequency field plot
sys, mesh = run_waveguide()
plot_field(sys, mesh, field="u", component="abs", cmap="jet")
```

Or from the command line:

```bash
python -c "from pyfes.projects import run_waveguide; run_waveguide()"
```

### DNN-GP surrogate model

```python
from pyfes.projects import bilateral_filter_dnngp

# Train from pre-generated .mat data (50 epsr x 81 freq)
model, metrics = bilateral_filter_dnngp(
    mat_file="bilat_all_sparams_50epsr.mat", n_epochs=500,
)
print(f"S11: {metrics['s11_rel_pct']:.2f}%  S21: {metrics['s21_rel_pct']:.2f}%")
# → S11: 0.19%  S21: 0.21%  (both < 1% relative error)

# Predict S-parameters at new (epsr, freq) points
import numpy as np
X_new = np.column_stack([np.full(81, 2.15),
                         np.linspace(138e9, 158e9, 81)])
s11_pred, s21_pred = model.predict(X_new)

# Load a saved checkpoint
from pyfes.projects.filter_dnngp import BilateralFilterDNNGP
model = BilateralFilterDNNGP().load("bilat_dnngp.pt")
```

### Adaptive LHS — automatic convergence

```python
from pyfes.projects import adaptive_lhs_bilateral_filter

# Starts with 5 LHS epsr values, adds 5 per iteration,
# stops when both S11 and S21 relative error < 1%
model, history = adaptive_lhs_bilateral_filter(
    n_initial=5, batch_size=5, n_test=5,
    target_rel_error=1.0, max_n_epsr=50,
)
```

The `adaptive_lhs_bilateral_filter` function uses **sequential Latin Hypercube Sampling** to gradually add εᵣ training values until the surrogate reaches a prescribed accuracy target. At each iteration, the unoccupied region of εᵣ ∈ [2.0, 2.2] is partitioned into equal-width intervals and one point is sampled per interval — true LHS stratification on the remaining space. A permanently held-out test set (5 evenly-spaced εᵣ values) measures generalization error.

**Result** (25 training εᵣ values across 4 batches):

| Iteration | Train εᵣ | S₁₁ rel error | S₂₁ rel error |
|-----------|----------|:------------:|:------------:|
| 0 | 5 | 1.118% | 0.592% |
| 1 | 10 | 2.522% | 3.843% |
| 2 | 15 | 4.294% | 2.964% |
| 3 | 20 | 6.411% | 4.589% |
| **4** | **25** | **0.272%** | **0.258%** |

The non-monotonic early errors are expected — with few εᵣ samples the DNN-GP optimizer can settle in worse local minima as the dataset grows. Once sufficient coverage is reached (here 25 εᵣ), the model generalises sharply below the 1% threshold.

Outputs: `adaptive_convergence.png` (error × cost curves), `adaptive_dnngp_results.png` (final validation figure).

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

`.poly` files are geometry descriptions for the [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) mesh generator. The `iormesh/` directory contains a C tool (`IOrMesh`) that runs Triangle and writes `.h1.mat` files consumed by the Python solver.

```bash
# Build the IOrMesh binary
make build

# List .poly files that still need meshing
make list

# Mesh a single file (with overridable Triangle arguments)
make data/WaveGuide.h1.mat ARGS="q34A"

# Mesh all missing .poly files
make all

# Re-mesh everything (overwrite existing .h1.mat)
make rebuild
```

Default Triangle arguments: `q34A` (quality mesh, 34° min angle, region attributes). Override per-file with `ARGS="q34a0.01A"`.

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib, pyvista
- torch, gpytorch, scikit-learn (for DNN-GP surrogate)
- C++ compiler (for `make build` to compile `IOrMesh`)

## Structure

| Path | Description |
|---|---|
| `pyfes/constants.py` | Physical constants (c₀, Z₀, ε₀, μ₀) |
| `pyfes/fem/shape_functions.py` | Scalar Lagrange (orders 1–4), H(curl) vector basis |
| `pyfes/fem/quadrature.py` | Gauss–Legendre on [0,1] and Duffy-transformed simplices |
| `pyfes/fem/jacobian.py` | Jacobian determinant / inverse transpose for triangles |
| `pyfes/fem/dof.py` | Global DOF numbering and hierarchical basis indices |
| `pyfes/fem/boundary.py` | Boundary DOF maps for domain decomposition |
| `pyfes/fem/assembly.py` | S/T/St/Tt/G matrix assembly, waveguide ports, BCs |
| `pyfes/fem/harmonic_balance.py` | Harmonic balance (Kerr nonlinearity, ferrite) |
| `pyfes/mesh/io_poly.py` | Triangle .poly I/O and .h1.mat reader (MATLAB v5 format) |
| `pyfes/mesh/build.py` | Regular triangular meshes on the unit square |
| `pyfes/mesh/plot.py` | Matplotlib mesh / geometry plotting |
| `pyfes/post/plot.py` | pyVista-based in-process field rendering |
| `pyfes/projects/` | Simulation examples (19 projects) |
| `pyfes/projects/filter_dnngp.py` | DNN-GP surrogate model (GPyTorch) |
| `iormesh/` | C mesher (Triangle wrapper, MATLAB .mat exporter) |
| `data/` | `.poly` geometry files and `.h1.mat` mesh files |
| `tests/` | pytest suite |
| `paper/` | IEEE paper and vector figures |

## Tests

```bash
pytest tests/ -v
```

## Related

- [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) — Jonathan Shewchuk's Delaunay triangulator

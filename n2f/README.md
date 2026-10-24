# `n2f` — near-field to far-field transformation

`n2f` transforms near fields sampled on a closed surface into far-field
radiation patterns using Huygens' principle in its scalar form and the
Kottler surface-current formulation in its vector form. It is bundled under
the MIT license (see the repository-root [`LICENSE`](../LICENSE)); the
original package name was `pyn2f`.

The package is independent of the FES finite-element pipeline and uses only
NumPy; Matplotlib is optional and used only by `n2f.plotting`.

## Installation

`n2f` is part of the FES repository and is installed with it:

```bash
./setup                # or: pip install -e .
source .venv/bin/activate   # activate so `python`/`pytest` work directly
```

## Quick start — scalar field

```python
import numpy as np
import n2f

wl = 1.0
array_pos = n2f.build_array(wl, 3, 0.5, 3, 0.5)

# Closed sampling box around the array.
dims = n2f.get_box_dim(wl, array_pos, 0.5, 0.25, 0.5)
box_pos, box_n, dS, _ = n2f.build_box([1] * 6, *dims, 1, 0, 0)

# Scalar near field on the box and its normal derivative.
r_mag, n_dot_r = n2f.get_box_vectors(array_pos, box_pos, box_n)
excitation = n2f.sf_excitations(wl, array_pos, 0, 0)
psi, del_psi = n2f.sf_nf_solver(wl, excitation, r_mag, n_dot_r)

# Far field (phi, theta) and power gain.
theta = n2f.deg2rad(np.linspace(0, 90, 181))
phi = n2f.deg2rad([0, 90])
far = n2f.sf_nf2ff_solver(wl, theta, phi, box_pos, box_n, dS, psi, del_psi)
gain = n2f.sf_compute_gain(far)
```

A direct point-source reference for validation is available as
`sf_direct_ff_solver`; on the box above the two agree to better than 5%.

## Quick start — vector field

```python
from n2f.utils import get_free_space_electrical_params

z0, k0, _ = get_free_space_electrical_params(1e9)
J, M = n2f.vf_excitations(k0, array_pos, 1.0, 0, 0, 0.0, 0, 0, 0, 0)

theta = n2f.deg2rad(np.linspace(0, 90, 19))
phi = n2f.deg2rad([0, 45, 90])
theta_ff, phi_ff = np.meshgrid(theta, phi)

et_ff, ep_ff = n2f.vf_direct_ff_solver(
    k0, z0, array_pos, J, M, theta_ff, phi_ff
)
gain_t, gain_p = n2f.vf_compute_gain(z0, et_ff, ep_ff, P=1.0)
```

See [Known limitations](#known-limitations) before using the vector
near-field or operator paths.

## Layout

```
n2f/
├── geometry/     # arrays, boxes, spheres, triangulated spheres
├── transforms/   # Cartesian ↔ spherical components, Rodrigues rotation
├── scalar/       # scalar excitations, near-field and N2F solvers
├── vector/       # vector excitations, Kottler solvers, N2F operators
├── utils/        # angle helpers, sampling, error metrics, electrical params
└── plotting/     # 2-D/3-D geometry, near-field and far-field plots
```

Main exports (all available directly from `n2f`):

- **Geometry** — `build_array`, `build_box`, `build_sphere`,
  `build_tri_sphere`, `get_box_dim`, `get_sph_radius`, `get_box_vectors`,
  `get_sph_vectors`, `get_tri_sph_mesh`,
  `get_spiraling_helicoidal_trajectory`
- **Transforms** — `cartesian2spherical`, `spherical2cartesian`,
  `cross_operator`, `get_rotation_matrix`
- **Scalar** — `sf_excitations`, `sf_compute_gain`, `sf_nf_solver`,
  `sf_nf2ff_solver`, `sf_direct_ff_solver`, `sf_nf2ff_operator`
- **Vector** — `vf_excitations`, `vf_compute_gain`, `vf_nf_solver`,
  `vf_nf2ff_solver`, `vf_direct_ff_solver`, `vf_n2f_op_fields`,
  `vf_n2f_op_fields_fft`
- **Utilities** — `deg2rad`, `rad2deg`, `vector2matrix`, `get_l1_error`,
  `get_l2_error`, `get_max_error`

## Conventions

- Angles are passed in **radians** to the numerical solvers; the excitation
  helpers take steering angles in **degrees**.
- `theta_ff`/`phi_ff` grids for the vector solvers have shape
  `(N_phi, N_theta)`, as produced by `numpy.meshgrid(theta, phi)`.
- Far-field arrays returned by the scalar solver have shape
  `(N_phi, N_theta)`.
- Surface patches must be closed and their normals outwardly directed for the
  Huygens/Kottler integral to be correct.

## Known limitations

These defects are present in the bundled upstream code and are recorded as
`xfail` in `tests/test_n2f.py`:

- `vf_nf_solver` accumulates complex fields into real-valued arrays and raises
  a casting error under NumPy 2.
- `vf_n2f_op_fields` assembles operator blocks with the wrong width and raises
  a broadcasting `ValueError`.
- `sf_nf2ff_solver` (and `sf_direct_ff_solver`) store the complex integral in a
  real array, silently discarding a small imaginary part.

The scalar `sf_nf2ff_operator` path is unaffected: applying the operator
reproduces the real part of `sf_nf2ff_solver` to machine precision.

## Tests

```bash
source .venv/bin/activate   # activate once, then:
pytest tests/test_n2f.py -q
```

The suite covers the geometry builders, transforms and rotation matrices,
utility metrics, the scalar box transformation against the direct
point-source pattern, operator/solver consistency, the vector direct solver,
and plotting smoke tests.

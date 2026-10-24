# `rfm` — rational function models for RF data

`rfm` builds reduced-order rational function models (RFM) from frequency-domain
S-parameter data. It provides the Descriptor Unit Model (DUM) container,
Vector Fitting, pole-residue macromodels, passivity checks and enforcement,
Touchstone/JSON I/O, and extraction/conversion utilities. It is bundled under
the MIT license (see the repository-root [`LICENSE`](../LICENSE)); the
original package name was `pyrfm`.

The package is independent of the FES finite-element pipeline and uses only
NumPy, SciPy and Matplotlib (all core FES dependencies).

## Installation

`rfm` is part of the FES repository and is installed with it:

```bash
./setup                # or: pip install -e .
source .venv/bin/activate   # activate so `python`/`pytest` work directly
```

## Quick start

```python
import numpy as np
import rfm

# Read a Touchstone file.  DUM.S has shape (Nports, Nports, Nfreq).
dum = rfm.read_touchstone("rfm/data/example_model.s2p")
print(dum.Nports, len(dum.f), dum.S.shape)

# Fit S21 with a 10-pole pole-residue model.
freq, s21 = dum.f, dum.S[1, 0, :]
poles, residues, d_term = rfm.vector_fit(freq, s21, max_poles=10)

# Build the macromodel and check passivity.
model = rfm.Macromodel(representation="pole-residue")
model.Nports = 1
model.poles = poles
model.residues = residues
model.dc_gain = d_term

is_passive, passive_model = rfm.check_hamiltonian(model, freq)
if not is_passive:
    model = passive_model

# Export the fit as JSON and as Touchstone.
rfm.export_rfm_model(model, "fit.json", metadata={"source": "example"})
rfm.export_rfm_touchstone(model, freq, "fit.s1p", s_param_index=(1, 1))
```

The models are evaluated with

```
H(j f) = d + sum_k r_k / (j f - p_k)
```

where `f`, `p_k` and `r_k` share the caller's frequency units.  Vector Fitting
normalizes the band internally for conditioning and scales the poles and
residues back, so inputs in Hz, kHz or rad/s all work.

## Command line

`rfm` has no CLI of its own.  The bundled walkthrough can be run with

```bash
python -m rfm.example_usage
```

## Touchstone conventions

- Only Touchstone 1.x S-parameter files are read.  The option line may specify
  `HZ`, `KHZ`, `MHZ`, `GHZ` or `THZ`, and `RI`, `MA` or `DB` data.  `Y`, `Z`,
  `H` and `G` files raise `ValueError`.
- Records may be wrapped across several lines.
- The port count comes from the `.sNp` suffix (recommended) or from the width
  of the first data line.
- Data order follows the standard `S11, S21, ..., SN1, S12, ...` convention.
- `read_touchstone` returns `DUM.S` with shape `(Nports, Nports, Nfreq)`.
- `write_touchstone` accepts either `(Nports, Nports, Nfreq)` (preferred) or
  `(Nfreq, Nports, Nports)` and always writes the documented orientation.
- `read_csv` is a placeholder that delegates to `read_touchstone`.

## Layout

```
rfm/
├── dum.py             # DUM container
├── macromodel.py      # Macromodel container
├── example_usage.py   # end-to-end walkthrough
├── data/              # bundled example .s2p
├── fitting/           # Vector Fitting and (unimplemented) skimming stubs
├── io/                # Touchstone/CSV readers and writers
├── utils/             # extract, merge, convert, plot, RFM export
└── validation/        # positive-real/Hamiltonian passivity checks
```

The main workflow is:

```
read_touchstone → vector_fit → Macromodel → check_hamiltonian
                → export_rfm_model / export_rfm_touchstone
```

## Changes from upstream

This bundle renames the package `pyrfm` → `rfm` and fixes defects that broke
the documented workflow.  The file layout is unchanged.

- **Touchstone orientation.** The upstream reader returned `S` as
  `(Nports, Nfreq, Nports)` despite documenting `(Nports, Nports, Nfreq)`,
  which broke `dum.S[i, j, :]` and downstream fitting.  The reader now returns
  the documented shape.
- **Port ordering.** The reader and writer now use the standard Touchstone
  order `S11, S21, S12, S22`; upstream filled rows in the wrong order and
  swapped `S12`/`S21`.
- **Formats and wrapping.** `MA`/`DB` data, wrapped records and `THz` units
  are handled; the upstream reader only understood unwrapped RI data.
- **Vector Fitting.** The upstream `vector_fit` solved a fixed-pole linear fit
  and applied a pole-relocation formula to the wrong coefficients; it could
  not fit a known three-pole response (relative RMS error ≈ 84%).  The
  implementation now performs the Gustavsen–Semlyen iteration with band
  normalization and recovers synthetic models to machine precision.
  `vector_fit_refined` random starts are scaled to the data band.
- **Utilities.** `merge_options` had no body; `convert_macromodel_units` wrote
  a nonexistent `Nprops` attribute, missed `THz`, scaled time in the wrong
  direction, and left residues inconsistent after a unit change.  All are
  fixed.  `extract_subdum`, `extract_largest_responses` and `plot_dum` now
  consistently use `(Nports, Nports, Nfreq)` (while still accepting the
  frequency-first layout).
- **Unimplemented methods raise.** `En_Skimming`, `AddingSkimming`,
  `Final_Skimming`, `MultiVectorFit` and `MultiVectorFitPD` were lambdas that
  silently returned `([], [], 0j)`; they now raise `NotImplementedError`.

## Known limitations

- The I/O conversion helpers `ImportCSDF`, `ImportCSV`, `MOD2DUM`, `MODFill`,
  `ResPoles2ABC`, `SpiceMOD2DUM` and `SS2ResPoles` are docstring-only stubs in
  the upstream code; they are retained for layout compatibility.
- `vector_fit_inf_dc0` fits a direct term but, as in upstream, does not impose
  a DC constraint.
- The passivity checks implement positive-real conditions
  (`H + H^H` positive semi-definite) for a transfer matrix, not the
  bounded-real condition `I - S^H S` for scattering parameters.  They are
  suitable for pole-residue models whose response is treated as an immittance
  or transfer function; do not use them alone to certify an amplifier's
  scattering passivity.
- Touchstone 2.0 features (multiple option lines, noise parameters, mixed-mode
  ordering) are not supported.

## Tests

```bash
source .venv/bin/activate   # activate once, then:
pytest tests/test_rfm.py -q
```

The suite covers the containers, Touchstone port ordering/formats/wrapping and
round trips, recovery of a known rational response by Vector Fitting,
passivity checking and enforcement, extraction/conversion utilities, JSON and
Touchstone export, the unimplemented-method guards, and the plotting helpers.

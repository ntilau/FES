# `mm` — modal-matching waveguide solver

`mm` computes generalised scattering matrices (GSMs) for rectangular
waveguide structures by mode matching. It is a Python port of a MATLAB
toolbox, bundled with FES under the MIT license (see the repository-root
[`LICENSE`](../LICENSE)).

The package is independent of the FES finite-element pipeline: it has no
`.poly`/mesh dependency and uses only NumPy (Matplotlib is imported lazily by
the plotting helpers).

## Installation

`mm` is part of the FES repository and is installed with it:

```bash
./setup                # or: pip install -e .
source .venv/bin/activate   # activate so `python`/`pytest` work directly
```

## Quick start

```python
from mm import core, scripts

# Run the Riblet broadwall-coupler benchmark (4 ports, 7-15 GHz).
S, sinfo, err = scripts.Riblet(plot=False)
print(len(S), "frequency points; fatal:", err.get("fatal"))
print("first S-matrix:", S[0].shape)
```

`S` is a list with one complex GSM per frequency, `sinfo` describes the modes
retained at each port, and `err` is `{}` on success (or carries a `fatal`
message).

Direct use of the numerical core:

```python
import numpy as np
from mm import core

seg = {"a": 0.02286, "b": 0.01016, "Nmodes": 4, "l": 0.01,
       "xo": 0.01143, "yo": 0.00508}
seg, _ = core.OrderModes(seg, {"x": 0, "y": 0, "H": 0, "E": 0})
core.EigenModes(seg)
core.NormCoeff(seg)
k0 = 2 * np.pi / 0.03
core.WaveNumbers(seg, k0)
core.DelayMatrix(seg, k0)
```

## Command line

```bash
python -m mm Riblet --no-plot
python -m mm bifurcatione            # aliases are case-insensitive
python -m mm hildebrandfull
```

| Script | Device | Ports | Band |
|--------|--------|:-----:|------|
| `BifurcationE` | E-plane T-junction | 3 | 10–25 GHz |
| `BifurcationH` | H-plane T-junction | 3 | 10–25 GHz |
| `Riblet` | Riblet broadwall coupler | 4 | 7–15 GHz |
| `HildebrandHalf` | Hildebrand coupler exploiting symmetry | 2 | 13–15 GHz |
| `HildebrandSemiAuto` | Hildebrand coupler, explicit assembly | 2 | 13–15 GHz |
| `HildebrandFull` | Hildebrand coupler, full 4-port | 4 | 13–15 GHz |

Aliases: `bifurcatione`, `bifurcationh`, `riblet`, `hildebrandhalf`,
`hildebrandsemiauto`, `hildebrandfull` (plus the `hilderbrand*` misspellings).

## Layout

```
mm/
├── core.py       # numerical routines: modes, coupling integrals, cascades, GSM condensation
├── lib/          # one thin wrapper module per MATLAB lib/*.m function
├── scripts.py    # benchmark definitions and HFSS reference overlays
├── run.py        # argparse CLI
├── __main__.py   # `python -m mm`
└── HFSSe.csv, HFSSh.csv   # bundled HFSS reference data
```

The numerical workflow is:

```
OrderModes → EigenModes → NormCoeff → WaveNumbers → DelayMatrix
          → SingleStep / Nto1Junction → Cascade
          → Renormalize / CondenseGSM → MultiPortDeviceSolve
          → ExtractPortS / ExtractSingleS
```

Port indices are 1-based, following the MATLAB original.

## Tests

```bash
source .venv/bin/activate   # activate once, then:
pytest tests/test_mm.py -q
```

The suite covers the modal primitives, mode ordering and symmetry, wave
numbers/delay matrices, coupling integrals, port-block bookkeeping,
frequency/junction validation, and the `BifurcationE`, `BifurcationH`, and
`Riblet` benchmarks.

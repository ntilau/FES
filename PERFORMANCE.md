# Performance — WR90 runtime comparison (Python port vs cFES)

Measurements of the Python reference path (`fes`, `em_e_fd`) against the cFES
C++ oracle on the same model, frequency and discretization. Both codes produce
agreeing S-parameters (see *Correctness anchor* below), so the timings are an
apples-to-apples comparison of the two implementations rather than of two
different problems.

Measured 2026-11-02.

## Workload

- `data/WR90.poly`, 10 GHz (one driven frequency), homogeneous refinement
  `p = 1, 2, 3, 4`, `h = 0`.
- Two `WavePort` boundaries with five modes each (`WP1`, `WP2`), plus `PEC1`.
- Both codes remesh the PLC with TetGen on every run (cFES prints
  `Meshing with TetGen command = pq34AfeeQ`; the Python loader's default is
  `pqAeafeeQ`). Both produce the identical mesh:
  556 nodes, 3,053 edges, 4,621 faces, 2,123 tetrahedra.
- Each run writes the 10-port Touchstone file (`WR90.s10p`). The cFES runs also
  saved the VTK field (`+field`); the Python runs used the default, which writes
  only the mesh VTK. Repeating cFES without `+field` gave 0.053 / 0.322 / 2.143 /
  6.885 s for p=1–4, i.e. field output is below the run-to-run noise.

## Environment

- Apple MacBook (`Apple-MacBook.local`), macOS (Darwin 27.2.0)
- 8 processors, 16 GB RAM
- Python 3.12 (repo venv), SciPy sparse complex LU
- cFES built from the local checkout (`bin/FES`)
- Single-threaded assembly in both; no other heavy work on the machine except
  where noted in *Caveats*.

## Total wall time

Each total is the program's own end-to-end timer: load + TetGen mesh +
assembly + port eigenmodes + factor/solve + output.

| p | total DOF | SYS DOF | Python | cFES | Python / cFES |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,053 | 1,938 | 2.52 s | 0.056 s | 45× |
| 2 | 15,348 | 11,608 | 22.79 s | 0.331 s | 69× |
| 3 | 43,254 | 35,389 | 175.80 s | 2.10 s | 84× |
| 4 | 93,140 | 79,650 | 601.52 s | 6.75 s | 89× |

`total DOF` is the mesh-wide `H(curl)` count (`FES DoF` in cFES); `SYS DOF` is
the reduced system after the port DOFs are replaced by modal amplitudes.

## Where the time goes

Assembly is the solids/port/DOF-construction stage of each program; the solve
is the sparse complex factorization plus one right-hand-side solve.

| p | Python assembly | cFES assembly | ratio | Python factor+solve | cFES factor+solve | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.49 s | 0.013 s | 195× | 0.030 s | 0.013 s | 2.4× |
| 2 | 20.04 s | 0.128 s | 157× | 2.750 s | 0.145 s | 19× |
| 3 | 133.92 s | 1.108 s | 121× | 41.855 s | 0.720 s | 58× |
| 4 | 390.98 s | 3.624 s | 108× | 210.297 s | 2.164 s | 97× |

Assembly share of the Python total: 99% (p=1), 88% (p=2), 76% (p=3), 65%
(p=4). The solve share grows correspondingly: 1%, 12%, 24%, 35%.

## Peak resident memory

| p | Python | cFES | ratio |
|---:|---:|---:|---:|
| 1 | 83 MB | 23.5 MB | 3.5× |
| 2 | 545 MB | 92.3 MB | 5.9× |
| 3 | 3.92 GB | 399 MB | 9.8× |
| 4 | 9.15 GB | 1.24 GB | 7.4× |

## Observations

- **Assembly dominates the Python runtime at every order.** The per-element
  Python loops in `element_matrices_3d` are ~100–200× slower than cFES's
  compiled assembly; cFES assembles all 93k DOF in 3.6 s while Python needs
  6.5 minutes.
- **The solve gap widens with DOF**: 2.4× at p=1 to 97× at p=4, with 7.4× the
  peak memory. At p=4 the SciPy sparse LU is the second-largest cost (35%) and
  is what makes the Python run take ten minutes.
- **Scaling per order** (p → p+1), total time: Python 9.0×, 7.7×, 3.4×;
  cFES 6.0×, 6.3×, 3.2×. The solve alone scales 90.8×, 15.2×, 5.0× in Python
  versus 11.3×, 4.96×, 3.0× in cFES; assembly scales 8.1×, 6.7×, 2.9× versus
  10.0×, 8.7×, 3.3×.
- **Correctness anchor.** WR90 S-parameters match cFES to max `|Δ|S||` of
  5.9e-8 (p=1), 1.1e-6 (p=2), 2.3e-8 (p=3) and 3.8e-8 (p=4, excluding the
  near-degenerate evanescent pairs, whose eigenvectors are basis-dependent).

## Reproduce

Python (run in an empty output directory so the Touchstone/VTK land there):

```bash
cd /tmp/wr90-py
python -m fes run /path/to/FES/data/WR90.poly +f 1e10 +p 4
```

cFES (the binary reads `WR90.poly` from its working directory):

```bash
mkdir -p results && cp /path/to/cFES/data/WR90.poly results/
cd results
/path/to/cFES/bin/FES WR90 1e10 +field +p 4
```

The `++ … s` line at the end of each log is the whole-run timer quoted above;
the per-stage lines (`In solids:`, `WP1:`, `SYS DoF =`, `Factor and Solve:` in
Python; `FES DoF =`, `Factor and Solve + … MB` in cFES) feed the breakdown
tables.

## Caveats

- Absolute times depend on the machine and on ambient load; the ratios are the
  useful number. Run-to-run variation is a few percent (repeat cFES totals
  above). The Python p=4 assembly overlapped a test-suite run on other cores
  for its first ~2 minutes, which is negligible because that stage is
  single-threaded interpreted code.
- The cFES numbers come from separate runs (`cpp_fix_p2/3/4`, `cpp_p1`); the
  Python numbers from `py_fix_p1/2/3/4`. Each run is a single frequency.
- Output cost is included on both sides (mesh VTK in Python, Touchstone in
  both); field output is well under 1% for either code.

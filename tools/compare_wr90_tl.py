"""Compare WR90_TL eigenmode results across dense, ARPACK, and Arnoldi solvers.

Sweeps frequency and plots propagation constants (beta) from all three solver
paths of em_e_tl_eig: dense, ARPACK shift-invert, and Arnoldi shift-invert.
Includes analytical WR90 cutoff frequencies for reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fes.mesh.mesh2d import mesh_from_poly
from fes.core.assemblers import assemble_em_e_tl_eig

REPO = Path("/Users/ntilau/Downloads/FES")
MESH = REPO / "data" / "WR90_TL.poly"

# Sweep 6-20 GHz to capture TE10 (6.56 GHz), TE20 (13.11 GHz), TE30 (19.67 GHz)
FREQS = np.linspace(6e9, 20e9, 29)
ORDER = 2
SOLVERS = ("dense", "arpack", "arnoldi")
SOLVER_LABELS = {"dense": "Dense", "arpack": "ARPACK", "arnoldi": "Arnoldi"}

# Analytical WR90 cutoff frequencies (GHz)
ANALYTICAL = {
    "TE10": 6.557,
    "TE20": 13.114,
    "TE30": 19.671,
    "TM01": 14.754,
    "TE11": 16.145,
}


def run_solver(freq: float, solver: str, n_modes: int = 6):
    mesh = mesh_from_poly(MESH)
    system = assemble_em_e_tl_eig(mesh, freq, ORDER, solver=solver, n_modes=n_modes)
    return system.mode_beta, system.kc, system.alpha


def main() -> None:
    n_modes = 6
    results = {s: {"beta": [], "kc": []} for s in SOLVERS}

    for f in FREQS:
        for solver in SOLVERS:
            bd, kd, _ = run_solver(f, solver, n_modes=n_modes)
            results[solver]["beta"].append(bd)
            results[solver]["kc"].append(kd)

    for solver in SOLVERS:
        results[solver]["beta"] = np.array(results[solver]["beta"]).real
        results[solver]["kc"] = np.array(results[solver]["kc"]).real

    # Save numerical results
    out_data = {
        "freqs_ghz": (FREQS / 1e9).tolist(),
        "solvers": {
            s: {
                "beta": results[s]["beta"].tolist(),
                "kc": results[s]["kc"].tolist(),
            }
            for s in SOLVERS
        },
    }
    data_path = REPO / "tools" / "wr90_tl_sweep.json"
    data_path.write_text(json.dumps(out_data, indent=2))
    print(f"Sweep data saved to {data_path}")

    # ---- Plots ----
    colors = plt.cm.tab10(np.linspace(0, 1, n_modes))
    line_styles = {"dense": "-", "arpack": "--", "arnoldi": ":"}
    markers = {"dense": "o", "arpack": "s", "arnoldi": "^"}

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # --- Panel 1: Beta vs frequency ---
    ax = fig.add_subplot(gs[0, :])
    for m in range(n_modes):
        for solver in SOLVERS:
            ax.plot(
                FREQS / 1e9,
                results[solver]["beta"][:, m],
                line_styles[solver],
                marker=markers[solver],
                color=colors[m],
                label=f"{SOLVER_LABELS[solver]} mode {m+1}" if m == 0 else None,
                markersize=4,
                alpha=0.85,
            )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Propagation constant β (rad/m)")
    ax.set_title("WR90_TL: β vs Frequency — Dense vs ARPACK vs Arnoldi (p=2)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    # --- Panel 2: kc vs frequency ---
    ax = fig.add_subplot(gs[1, :])
    for m in range(n_modes):
        for solver in SOLVERS:
            ax.plot(
                FREQS / 1e9,
                results[solver]["kc"][:, m],
                line_styles[solver],
                marker=markers[solver],
                color=colors[m],
                markersize=4,
                alpha=0.85,
            )
    # Analytical cutoff markers
    for name, fc in ANALYTICAL.items():
        ax.axvline(fc, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(
            fc,
            ax.get_ylim()[1] * 0.95,
            name,
            rotation=90,
            va="top",
            fontsize=7,
            color="gray",
        )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Cutoff wavenumber kc (rad/m)")
    ax.set_title("Cutoff Wavenumber vs Frequency (gray dashed = analytical)")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Solver spread in beta ---
    ax = fig.add_subplot(gs[2, 0])
    for m in range(n_modes):
        vals = np.stack([results[s]["beta"][:, m] for s in SOLVERS], axis=0)
        spread = vals.max(axis=0) - vals.min(axis=0)
        ax.semilogy(
            FREQS / 1e9,
            spread,
            "o-",
            color=colors[m],
            label=f"mode {m+1}",
            markersize=4,
        )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("max(β) − min(β) (rad/m)")
    ax.set_title("Solver Spread: β")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    # --- Panel 4: Solver spread in kc ---
    ax = fig.add_subplot(gs[2, 1])
    for m in range(n_modes):
        vals = np.stack([results[s]["kc"][:, m] for s in SOLVERS], axis=0)
        spread = vals.max(axis=0) - vals.min(axis=0)
        ax.semilogy(
            FREQS / 1e9,
            spread,
            "o-",
            color=colors[m],
            label=f"mode {m+1}",
            markersize=4,
        )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("max(kc) − min(kc) (rad/m)")
    ax.set_title("Solver Spread: kc")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    plot_path = REPO / "tools" / "wr90_tl_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # ---- Summary ----
    print("\nMaximum cross-solver differences per mode:")
    for m in range(n_modes):
        max_beta = 0.0
        max_kc = 0.0
        for i, s1 in enumerate(SOLVERS):
            for s2 in SOLVERS[i + 1 :]:
                max_beta = max(
                    max_beta,
                    float(np.max(np.abs(results[s1]["beta"][:, m] - results[s2]["beta"][:, m]))),
                )
                max_kc = max(
                    max_kc,
                    float(np.max(np.abs(results[s1]["kc"][:, m] - results[s2]["kc"][:, m]))),
                )
        print(f"  mode {m+1}: |Δβ|_max={max_beta:.2e}, |Δkc|_max={max_kc:.2e}")

    # Verify all match
    for s1 in SOLVERS:
        for s2 in SOLVERS:
            assert np.allclose(results[s1]["beta"], results[s2]["beta"], rtol=1e-6)
            assert np.allclose(results[s1]["kc"], results[s2]["kc"], rtol=1e-6)
    print("\nAll modes match across all three solver paths.")


if __name__ == "__main__":
    main()

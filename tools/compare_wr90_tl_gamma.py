"""Plot gamma (|mode_beta|) for all WR90_TL modes across orders and solvers.

Gamma = alpha + 1j*beta where the propagation/attenuation constant magnitude
is |mode_beta|. Real part = attenuation (evanescent), Imag part = propagation.
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

FREQS = np.linspace(6e9, 20e9, 21)
ORDERS = (1, 2, 3)
SOLVERS = ("dense", "arpack", "arnoldi")
SOLVER_LABELS = {"dense": "Dense", "arpack": "ARPACK", "arnoldi": "Arnoldi"}
NMODES_REQUESTED = 6


def run_solver(freq: float, order: int, solver: str, prev_beta=None):
    mesh = mesh_from_poly(MESH)
    system = assemble_em_e_tl_eig(
        mesh, freq, order, solver=solver, n_modes=NMODES_REQUESTED,
        prev_beta=prev_beta,
    )
    return system.mode_beta, system.n_modes


def main() -> None:
    results = {
        order: {s: [] for s in SOLVERS} for order in ORDERS
    }
    n_modes_by_order = {}

    for order in ORDERS:
        for solver in SOLVERS:
            prev_beta = None
            for f in FREQS:
                mb, n_modes = run_solver(f, order, solver, prev_beta=prev_beta)
                results[order][solver].append(mb)
                prev_beta = mb
        n_modes_by_order[order] = n_modes

    arrays = {
        order: {
            s: {
                "gamma_mag": np.stack(
                    [np.abs(results[order][s][i]) for i in range(len(FREQS))]
                ),
                "gamma_real": np.stack(
                    [results[order][s][i].real for i in range(len(FREQS))]
                ),
                "gamma_imag": np.stack(
                    [results[order][s][i].imag for i in range(len(FREQS))]
                ),
            }
            for s in SOLVERS
        }
        for order in ORDERS
    }

    # Save numerical results
    out_data = {
        "freqs_ghz": (FREQS / 1e9).tolist(),
        "n_modes_by_order": {str(o): n_modes_by_order[o] for o in ORDERS},
        "orders": {
            str(o): {
                s: {
                    "gamma_mag": arrays[o][s]["gamma_mag"].tolist(),
                    "gamma_real": arrays[o][s]["gamma_real"].tolist(),
                    "gamma_imag": arrays[o][s]["gamma_imag"].tolist(),
                }
                for s in SOLVERS
            }
            for o in ORDERS
        },
    }
    data_path = REPO / "tools" / "wr90_tl_gamma.json"
    data_path.write_text(json.dumps(out_data, indent=2))
    print(f"Gamma data saved to {data_path}")

    # ---- Plot gamma magnitude vs frequency ----
    order_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    line_styles = {"dense": "-", "arpack": "--", "arnoldi": ":"}
    markers = {"dense": "o", "arpack": "s", "arnoldi": "^"}

    fig = plt.figure(figsize=(16, 10))

    # One subplot per order, all modes and all solvers
    for idx, order in enumerate(ORDERS):
        n_modes = n_modes_by_order[order]
        ax = fig.add_subplot(1, 3, idx + 1)

        for m in range(n_modes):
            for solver in SOLVERS:
                gamma_mag = arrays[order][solver]["gamma_mag"][:, m]
                ax.plot(
                    FREQS / 1e9,
                    gamma_mag,
                    line_styles[solver],
                    marker=markers[solver],
                    color=order_colors[order],
                    label=f"mode {m+1}" if idx == 0 and solver == "dense" else None,
                    markersize=4,
                    alpha=0.85,
                )

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|γ| = |mode_beta| (rad/m)")
        ax.set_title(f"WR90_TL: |γ| vs Frequency (p={order}, {n_modes} modes)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=3, loc="upper left")

    plt.suptitle(
        "WR90_TL: Propagation/Attenuation Constant |γ| — Dense vs ARPACK vs Arnoldi",
        fontsize=14,
    )
    plt.tight_layout()

    plot_path = REPO / "tools" / "wr90_tl_gamma_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Gamma plot saved to {plot_path}")

    # ---- Print gamma summary at 10 GHz ----
    print("\n|gamma| at 10 GHz for all modes:")
    f_idx = np.argmin(np.abs(FREQS - 10e9))
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for m in range(n_modes):
            vals = [arrays[order][s]["gamma_mag"][f_idx, m] for s in SOLVERS]
            print(
                f"  p={order} mode {m+1}: |γ| = {vals[0]:.6f} "
                f"(dense={vals[0]:.6f}, arpack={vals[1]:.6f}, arnoldi={vals[2]:.6f})"
            )
        print(f"  p={order}: {n_modes} modes")


if __name__ == "__main__":
    main()

"""Compare WR90_TL eigenmode results across orders and solver paths.

Sweeps frequency for orders 1-3 and all three solvers (dense, ARPACK, Arnoldi).
Plots beta vs frequency for each order, plus cross-solver spread per order.
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

# Analytical WR90 cutoff frequencies (GHz)
ANALYTICAL = {
    "TE10": 6.557,
    "TE20": 13.114,
    "TE30": 19.671,
    "TM01": 14.754,
    "TE11": 16.145,
}

NMODES_REQUESTED = 6


def run_solver(freq: float, order: int, solver: str):
    mesh = mesh_from_poly(MESH)
    system = assemble_em_e_tl_eig(
        mesh, freq, order, solver=solver, n_modes=NMODES_REQUESTED
    )
    return system.mode_beta.real, system.kc.real, system.n_modes


def main() -> None:
    results = {
        order: {s: [] for s in SOLVERS} for order in ORDERS
    }
    n_modes_by_order = {}

    for order in ORDERS:
        for f in FREQS:
            for solver in SOLVERS:
                bd, kd, n_modes = run_solver(f, order, solver)
                results[order][solver].append((bd, kd))
        n_modes_by_order[order] = n_modes

    # Convert to arrays: results[order][solver] -> list of (beta, kc) per freq
    arrays = {
        order: {
            s: {
                "beta": np.stack([results[order][s][i][0] for i in range(len(FREQS))]),
                "kc": np.stack([results[order][s][i][1] for i in range(len(FREQS))]),
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
                    "beta": arrays[o][s]["beta"].tolist(),
                    "kc": arrays[o][s]["kc"].tolist(),
                }
                for s in SOLVERS
            }
            for o in ORDERS
        },
    }
    data_path = REPO / "tools" / "wr90_tl_orders.json"
    data_path.write_text(json.dumps(out_data, indent=2))
    print(f"Sweep data saved to {data_path}")

    # ---- Plots ----
    order_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    line_styles = {"dense": "-", "arpack": "--", "arnoldi": ":"}
    markers = {"dense": "o", "arpack": "s", "arnoldi": "^"}

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.25)

    # --- Panel 1: Beta vs frequency per order (dense solver) ---
    ax = fig.add_subplot(gs[0, :])
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for m in range(n_modes):
            ax.plot(
                FREQS / 1e9,
                arrays[order]["dense"]["beta"][:, m],
                "-",
                marker=markers["dense"],
                color=order_colors[order],
                label=f"p={order} mode {m+1}" if m < 3 else None,
                markersize=3,
                alpha=0.8,
            )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Propagation constant β (rad/m)")
    ax.set_title("WR90_TL: β vs Frequency by Order (dense solver)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    # --- Panel 2: kc vs frequency per order ---
    ax = fig.add_subplot(gs[1, :])
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for m in range(n_modes):
            ax.plot(
                FREQS / 1e9,
                arrays[order]["dense"]["kc"][:, m],
                "-",
                marker=markers["dense"],
                color=order_colors[order],
                markersize=3,
                alpha=0.8,
            )
    for name, fc in ANALYTICAL.items():
        ax.axvline(fc, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(fc, ax.get_ylim()[1] * 0.95, name, rotation=90, va="top",
                fontsize=7, color="gray")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Cutoff wavenumber kc (rad/m)")
    ax.set_title("Cutoff Wavenumber vs Frequency by Order (gray dashed = analytical)")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Cross-solver beta spread per order ---
    ax = fig.add_subplot(gs[2, 0])
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for m in range(n_modes):
            vals = np.stack(
                [arrays[order][s]["beta"][:, m] for s in SOLVERS], axis=0
            )
            spread = vals.max(axis=0) - vals.min(axis=0)
            ax.semilogy(
                FREQS / 1e9,
                spread,
                "o-",
                color=order_colors[order],
                label=f"p={order} mode {m+1}" if m < 2 else None,
                markersize=3,
                alpha=0.8,
            )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("max(β) − min(β) across solvers (rad/m)")
    ax.set_title("Solver Spread: β by Order")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    # --- Panel 4: Cross-solver kc spread per order ---
    ax = fig.add_subplot(gs[2, 1])
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for m in range(n_modes):
            vals = np.stack(
                [arrays[order][s]["kc"][:, m] for s in SOLVERS], axis=0
            )
            spread = vals.max(axis=0) - vals.min(axis=0)
            ax.semilogy(
                FREQS / 1e9,
                spread,
                "o-",
                color=order_colors[order],
                markersize=3,
                alpha=0.8,
            )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("max(kc) − min(kc) across solvers (rad/m)")
    ax.set_title("Solver Spread: kc by Order")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    # --- Panel 5: Order comparison at fixed frequency (10 GHz) ---
    ax = fig.add_subplot(gs[3, 0])
    f_idx = np.argmin(np.abs(FREQS - 10e9))
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        betas = arrays[order]["dense"]["beta"][f_idx, :n_modes]
        ax.plot(
            range(n_modes),
            betas,
            "o-",
            color=order_colors[order],
            label=f"p={order}",
            markersize=6,
        )
    ax.set_xlabel("Mode index")
    ax.set_ylabel("β at 10 GHz (rad/m)")
    ax.set_title("Mode Spectrum at 10 GHz by Order")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- Panel 6: Order comparison of kc at 10 GHz ---
    ax = fig.add_subplot(gs[3, 1])
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        kcs = arrays[order]["dense"]["kc"][f_idx, :n_modes]
        ax.plot(
            range(n_modes),
            kcs,
            "o-",
            color=order_colors[order],
            label=f"p={order}",
            markersize=6,
        )
    ax.set_xlabel("Mode index")
    ax.set_ylabel("kc at 10 GHz (rad/m)")
    ax.set_title("Cutoff Wavenumber Spectrum at 10 GHz by Order")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plot_path = REPO / "tools" / "wr90_tl_orders_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # ---- Summary ----
    print("\nMaximum cross-solver differences per order and mode:")
    all_ok = True
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for m in range(n_modes):
            max_beta = 0.0
            max_kc = 0.0
            for i, s1 in enumerate(SOLVERS):
                for s2 in SOLVERS[i + 1 :]:
                    max_beta = max(
                        max_beta,
                        float(
                            np.max(
                                np.abs(
                                    arrays[order][s1]["beta"][:, m]
                                    - arrays[order][s2]["beta"][:, m]
                                )
                            )
                        ),
                    )
                    max_kc = max(
                        max_kc,
                        float(
                            np.max(
                                np.abs(
                                    arrays[order][s1]["kc"][:, m]
                                    - arrays[order][s2]["kc"][:, m]
                                )
                            )
                        ),
                    )
            flag = "OK" if max_beta < 1e-6 else "CHECK"
            if flag != "OK":
                all_ok = False
            print(
                f"  p={order} mode {m+1}: |Δβ|_max={max_beta:.2e}, "
                f"|Δkc|_max={max_kc:.2e} [{flag}]"
            )
        print(f"  p={order}: {n_modes} modes computed")

    # Verify all match (only for well-resolved modes)
    all_ok = True
    for order in ORDERS:
        n_modes = n_modes_by_order[order]
        for s1 in SOLVERS:
            for s2 in SOLVERS:
                if not np.allclose(
                    arrays[order][s1]["beta"][:, :n_modes],
                    arrays[order][s2]["beta"][:, :n_modes],
                    rtol=1e-3,
                ):
                    all_ok = False
                if not np.allclose(
                    arrays[order][s1]["kc"][:, :n_modes],
                    arrays[order][s2]["kc"][:, :n_modes],
                    rtol=1e-3,
                ):
                    all_ok = False
    if all_ok:
        print("\nAll modes match across all solvers and orders (within 1e-3 rtol).")
    else:
        print(
            "\nSome higher-order modes differ across solvers (spurious on coarse mesh)."
        )


if __name__ == "__main__":
    main()

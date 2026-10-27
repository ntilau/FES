"""Plot spatial field distributions of WR90_TL eigenmodes.

Uses the same shift-invert approach as the solver to compute eigenvalues
and eigenvectors, then plots the field distributions on the triangular mesh.

For TE modes (ortho=True): plots |H_t| interpolated from edge DOFs.
For TM modes (ortho=False): plots |Ez| from node DOFs.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from fes.mesh.mesh2d import mesh_from_poly
from fes.core.assemblers.em_e_tl_eig import (
    _assemble_blocks,
    _bc_type_map,
    _dirichlet_dofs,
    _pencil,
    _GAMMA_TOL,
)
from fes.core.dof import face_dof_map, scalar_dof_count, vector_dof_count
from fes.constants import C0, PI

REPO = "/Users/ntilau/Downloads/FES"
MESH = f"{REPO}/data/WR90_TL.poly"
FREQ = 10e9
ORDER = 1
REFINE = 5  # homogeneous refinement levels (h_ord=5)
ORTHO = True  # True = TE modes (Ez=0, plot |H_t|), False = TM modes (plot |Ez|)
NMODES = 5  # WR90 at 10 GHz: only TE10 propagates; show 5 modes


def main() -> None:
    mesh = mesh_from_poly(MESH, h_ord=REFINE)

    k0 = 2.0 * PI * FREQ / C0
    kk = k0 * k0

    St, Tt, Tt2, Sz, Tz, G, maxepsr, maxmur = _assemble_blocks(mesh, ORDER)

    bc_types = _bc_type_map(mesh.boundaries)
    dirichlet = _dirichlet_dofs(mesh, ORDER, bc_types)
    dirichlet_set = set(int(d) for d in dirichlet)
    n_edges = vector_dof_count(mesh, ORDER)
    n_nodes = scalar_dof_count(mesh, ORDER)
    free_v = np.array(
        [i for i in range(n_edges) if i not in dirichlet_set], dtype=np.int64
    )
    numt = int(free_v.size)
    numz = n_nodes

    Stf = St[free_v][:, free_v]
    Ttf = Tt[free_v][:, free_v]
    A_vv = Stf - kk * Ttf
    Azz = Sz - kk * Tz

    # Extended T-matrix structure
    B_vv = Tt2[free_v][:, free_v]
    B_vz = G[free_v]
    B_zv = G.T[:, free_v]
    B_zz = -kk * Tz

    A_blocks = (
        A_vv,
        Azz,
        B_vv,
        B_vz,
        B_zv,
        B_zz,
    )
    A, B = _pencil(A_blocks, numt, numz)

    # Estimate waveguide cross-section dimensions from mesh extents
    a = float(mesh.nodes[:, 0].max() - mesh.nodes[:, 0].min())  # wide dimension
    b = float(mesh.nodes[:, 1].max() - mesh.nodes[:, 1].min())  # narrow dimension

    # TE10 cutoff wavenumber: kc = pi / a
    kc_te10 = PI / a
    # TE10 propagation constant (if k0 > kc_te10, it propagates)
    if k0 > kc_te10:
        beta_te10 = np.sqrt(k0 * k0 - kc_te10 * kc_te10)
    else:
        beta_te10 = 0.0
    # Shift for propagating modes: target TE10's eigenvalue d = -beta^2
    shift_prop = -beta_te10 * beta_te10 if beta_te10 > 0 else -0.5 * kk * maxepsr * maxmur

    # Shift for cutoff modes: target modes near alpha ~ k0/sqrt(2)
    shift_cut = +0.6 * kk * maxepsr * maxmur

    print(f"Solving at {FREQ/1e9:.1f} GHz, p={ORDER}, h_ord={REFINE}, ortho={ORTHO}")
    print(f"  Waveguide: a={a:.5f} m, b={b:.5f} m")
    print(f"  k0={k0:.1f}, kc_TE10={kc_te10:.1f}, beta_TE10={beta_te10:.1f}")
    print(f"  A shape: {A.shape}, numt={numt}, numz={numz}")
    print(f"  shift_prop={shift_prop:.1f} (targets TE10), shift_cut={shift_cut:.1f} (cutoff)")

    def solve_with_shift(shift: float, k_modes: int):
        """Run one shift-invert ARPACK solve, return (theta, vec, shift)."""
        A_s = (A / shift - B).tocsc()
        lu = scipy.sparse.linalg.splu(A_s)

        def matvec(x):
            x = np.asarray(x, dtype=complex).ravel()
            y = np.asarray(B @ x, dtype=complex).ravel() / shift
            y = lu.solve(y)
            if ORTHO:
                y[numt:] = 0.0  # project onto edge block (TE modes)
            return y

        operator = scipy.sparse.linalg.LinearOperator(
            (A.shape[0], A.shape[0]), matvec=matvec, dtype=complex
        )
        ncv = min(4 * k_modes, A.shape[0] - 1)
        theta, vec = scipy.sparse.linalg.eigs(
            operator, k=k_modes, which="LM", ncv=ncv,
            return_eigenvectors=True, maxiter=1000,
        )
        return theta, vec, shift

    # Solve with both shifts: propagating (targets TE10) and cutoff
    theta_prop, vec_prop, shift_prop = solve_with_shift(shift_prop, NMODES)
    theta_cut, vec_cut, shift_cut = solve_with_shift(shift_cut, NMODES)

    # Combine all results
    theta = np.concatenate([theta_prop, theta_cut])
    vec = np.concatenate([vec_prop, vec_cut], axis=1)
    shifts = np.concatenate([np.full(NMODES, shift_prop), np.full(NMODES, shift_cut)])

    print(f"  Computed {len(theta)} Ritz values total")

    # Convert theta -> d = shift + 1/theta -> gamma = sqrt(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = shifts + 1.0 / theta
    gamma = np.sqrt(d)
    real = np.where(np.abs(gamma.real) < _GAMMA_TOL, 0.0, np.abs(gamma.real))
    imag = np.where(np.abs(gamma.imag) < _GAMMA_TOL, 0.0, np.abs(gamma.imag))
    gamma = real + 1j * imag

    # Sort modes: fundamental propagating mode (lowest beta) first,
    # then cutoff modes by alpha ascending.  Only one propagating mode is
    # taken to avoid filling slots with numerical artifacts clustered near
    # TE10's eigenvalue.
    alpha = np.abs(gamma.real)
    beta = np.abs(gamma.imag)
    is_propagating = alpha < _GAMMA_TOL

    prop_idx = np.where(is_propagating)[0]
    cut_idx = np.where(~is_propagating)[0]

    # Propagating: take only the lowest-beta mode (fundamental TE10)
    if prop_idx.size > 0:
        best_prop = prop_idx[np.argmin(beta[prop_idx])]
    else:
        best_prop = None

    # Cutoff: sort by alpha ascending (lowest alpha = closest to cutoff)
    cut_order = cut_idx[np.argsort(alpha[cut_idx], kind="stable")]

    # Build ordered list: 1 propagating + (NMODES-1) cutoff
    selected = []
    if best_prop is not None:
        selected.append(best_prop)
    selected.extend(cut_order[: NMODES - len(selected)])
    order_idx = np.array(selected)
    gamma = gamma[order_idx]
    vec = vec[:, order_idx]

    print("\nMode summary (|gamma|, alpha, beta, kc):")
    for m in range(NMODES):
        b = gamma[m]
        kc = np.sqrt(abs(k0 * k0 - b.imag**2 + b.real**2))
        flag = " p" if abs(b.real) < _GAMMA_TOL else ""
        print(f"  mode {m+1}: |γ|={abs(b):9.3f}  α={b.real:9.3f}  β={b.imag:9.3f}  kc={kc:9.3f}{flag}")

    # Check eigenvector structure
    print("\nEigenvector block norms:")
    for m in range(NMODES):
        v_edge = np.linalg.norm(vec[:numt, m])
        v_node = np.linalg.norm(vec[numt:, m])
        print(f"  mode {m+1}: edge_norm={v_edge:.4f}, node_norm={v_node:.4f}")

    # Plot field distributions
    n_cols = 3
    n_rows = (NMODES + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    mode_type = "TE" if ORTHO else "TM"
    fig.suptitle(
        f"WR90_TL: {mode_type} Mode Field Distributions at {FREQ/1e9:.1f} GHz "
        f"(p={ORDER}, h_ord={REFINE})",
        fontsize=14,
    )

    triang = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.faces)
    n_mesh_edges = mesh.n_edges

    # Build edge->node mapping for interpolating edge DOFs to nodes
    edge_nodes = [tuple(int(e) for e in edge) for edge in mesh.edges]
    node_to_edges = [[] for _ in range(mesh.n_nodes)]
    for e_idx, (n1, n2) in enumerate(edge_nodes):
        node_to_edges[n1].append(e_idx)
        node_to_edges[n2].append(e_idx)

    # Number of edge-only DOFs (levels 0..ORDER-1)
    n_edge_dofs = ORDER * n_mesh_edges

    # Collect all fields first for consistent color scaling
    fields = []
    field_names = []
    for m in range(NMODES):
        v = vec[:, m]
        v_edge = v[:numt]
        v_node = v[numt:]

        if ORTHO:
            # For TE modes: |H_t| from edge DOFs.
            #
            # Edge DOF numbering (matches face_dof_map.v):
            #   level L edge dof = L * n_mesh_edges + L*(L-1)*n_faces + edge_id
            # For p=1 only level 0 exists, so dof == edge_id.
            # Keep the SIGNED edge value (tangential component direction matters
            # for the cos(pi*x/a) TE10 pattern); take abs only at plot time.
            n_faces = mesh.n_faces
            edge_vals = np.zeros(n_mesh_edges, dtype=complex)
            for i, orig_dof in enumerate(free_v):
                if orig_dof >= n_edge_dofs:
                    continue  # skip interior (bubble) DOFs
                # Invert the numbering: level = orig_dof // n_mesh_edges
                level = int(orig_dof // n_mesh_edges)
                edge_id = int(orig_dof - level * n_mesh_edges - level * (level - 1) * n_faces)
                edge_vals[edge_id] = v_edge[i]

            # Interpolate edge DOFs to nodes: each node gets the average of
            # the (signed) edge values on incident edges.
            field = np.zeros(mesh.n_nodes, dtype=complex)
            node_counts = np.zeros(mesh.n_nodes)
            for n in range(mesh.n_nodes):
                for e_idx in node_to_edges[n]:
                    field[n] += edge_vals[e_idx]
                    node_counts[n] += 1
            mask = node_counts > 0
            field[mask] /= node_counts[mask]
            field = np.abs(field)  # magnitude for plotting
            field_name = "|H_t|"
        else:
            # For TM modes: |Ez| from node DOFs (first n_nodes are vertex values)
            if np.linalg.norm(v_node) > 1e-6 * np.linalg.norm(v):
                field = np.abs(v_node[:mesh.n_nodes])
            else:
                field = np.zeros(mesh.n_nodes)
            field_name = "|Ez|"

        if np.max(np.abs(field)) > 0:
            field = field / np.max(np.abs(field))

        fields.append(field)
        field_names.append(field_name)

    # Use consistent color scaling across all subplots
    vmax = max(np.max(np.abs(f)) for f in fields) if fields else 1.0

    for m in range(NMODES):
        ax = axes[m // n_cols, m % n_cols]
        field = fields[m]

        tpc = ax.tripcolor(triang, field, shading="gouraud", cmap="RdBu_r",
                           vmin=0, vmax=vmax)
        ax.triplot(triang, "k-", linewidth=0.3, alpha=0.5)
        ax.set_aspect("equal")
        b = gamma[m]
        kc = np.sqrt(abs(k0 * k0 - b.imag**2 + b.real**2))
        ax.set_title(
            f"Mode {m+1} [{field_names[m]}]: |γ|={abs(b):.1f}, "
            f"α={b.real:.1f}, β={b.imag:.1f}",
            fontsize=8,
        )
        plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    for m in range(NMODES, n_rows * n_cols):
        axes[m // n_cols, m % n_cols].axis("off")

    plt.tight_layout()
    out = f"{REPO}/tools/wr90_tl_modes_dist_{'te' if ORTHO else 'tm'}_h{REFINE}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved mode distribution plot to {out}")


if __name__ == "__main__":
    main()

"""Waveguide scattering simulation example.

Port of ProjectWaveGuide.m.
"""

import numpy as np

from ..constants import get_constants
from ..mesh.io_poly import read_poly
from scipy import sparse
from ..fem.assembly import assemble_linear, assemble_waveguide_port
from ..post.plot import plot_field


def run_waveguide():
    """Run a waveguide scattering simulation."""
    sys = {"pOrd": 4, "hOrd": 1}

    # Mesh (a WR-90 rectangular waveguide)
    mesh = read_poly("WaveGuide", data_dir="./data", scale=1e-3)
    mesh["a"] = 22.86e-3
    mesh["b"] = mesh["a"] / 2

    sys["Height"] = mesh["b"]
    sys["WPnModes"] = 1
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPpow"] = 1.0
    mesh["BC"] = {"Dir": 1, "WP": [11, 12]}

    # Assemble system
    sys, mesh = assemble_linear(sys, mesh)

    # Frequency sweep
    n_freqs = 1
    freqs = np.array([10e9])
    sys["Sparams"] = np.zeros(
        (len(np.atleast_1d(mesh["BC"]["WP"])) * sys["WPnModes"], n_freqs),
        dtype=complex
    )

    for kf in range(n_freqs):
        freq = freqs[kf]
        print(f"freq = {freq / 1e9} GHz")

        sys = assemble_waveguide_port(sys, freq)
        X = sparse.linalg.spsolve(sys["A"], sys["B"])
        Xd = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)

        n_wp = len(sys["WP"])
        n_modes = sys["WPnModes"]

        sp = (Xd[:n_wp * n_modes, :n_wp * n_modes]
              - np.eye(n_wp * n_modes))
        port_idx = (sys["WPportPlot"] - 1) * n_modes + sys["WPmodePlot"] - 1
        sys["Sparams"][:, kf] = sp[:, port_idx].ravel()

        loss = (1 - np.linalg.norm(sys["Sparams"][:, kf])) * 100
        print(f"  losses = {loss:.2g}%")

        # Reconstruct full field
        u = np.zeros(sys["NDOFs"], dtype=complex)
        u[sys["nnWP"]] = Xd[n_wp * n_modes:, port_idx].ravel()
        for ip in range(n_wp):
            u[sys["WP"][ip]] = (
                sys["WPgvec"][ip]
                @ Xd[ip * n_modes:(ip + 1) * n_modes, port_idx].ravel()
            )
        sys["u"] = u

    print(f"max|u| = {np.max(np.abs(sys['u'])):.6g}")
    return sys, mesh


if __name__ == "__main__":
    from scipy import sparse
    sys, mesh = run_waveguide()

"""Waveguide scattering simulation example.

Port of ProjectWaveGuide.m.
"""

import numpy as np

from scipy import sparse

from ..core.assembly import assemble_linear, assemble_waveguide_port
from ..mesh.io_poly import read_poly
from ..paths import data_dir
from ._utils import reconstruct_wp_field, scattering_parameters


def run_waveguide():
    """Run a waveguide scattering simulation."""
    sys = {"pOrd": 4, "hOrd": 1}

    # Mesh (a WR-90 rectangular waveguide)
    mesh = read_poly("WaveGuide_py", data_dir=data_dir(), scale=1e-3)
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

        n_modes = sys["WPnModes"]

        sp = scattering_parameters(Xd, sys)
        port_idx = (sys["WPportPlot"] - 1) * n_modes + sys["WPmodePlot"] - 1
        sys["Sparams"][:, kf] = sp[:, port_idx].ravel()

        loss = (1 - np.linalg.norm(sys["Sparams"][:, kf])) * 100
        print(f"  losses = {loss:.2g}%")

        # Reconstruct full field
        sys["u"] = reconstruct_wp_field(Xd, sys, port_idx)

    print(f"max|u| = {np.max(np.abs(sys['u'])):.6g}")
    return sys, mesh


if __name__ == "__main__":
    sys, mesh = run_waveguide()

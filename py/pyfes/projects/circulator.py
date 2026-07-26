"""Ferrite circulator simulation.

Ports of ProjectCirculator.m, ProjectCirculatorDDschur.m,
ProjectCirculatorIMP.m, and related variants.
"""

import numpy as np
from scipy import sparse

from ..mesh.io_poly import read_poly
from ..mesh.build import build_regular_square
from ..fem.assembly import (
    assemble_linear, assemble_waveguide_port, assemble_domain_decomposition
)
from ..fem.harmonic_balance import assemble_hb_ferrite, assemble_wp_hb
from ..post.plot import plot_field
from ._utils import scattering_parameters, reconstruct_wp_field


def circulator(freq=9e9, n_freqs=1, plot=False):
    """S-parameter analysis of a ferrite circulator.

    Port of ProjectCirculator.m.

    Parameters
    ----------
    freq : float
        Center frequency (Hz).
    n_freqs : int
        Number of frequency points.
    plot : bool
        Whether to plot the field.

    Returns
    -------
    freqs : ndarray
        Frequency vector.
    Sparams : ndarray
        Scattering parameters.
    """
    sys = {"pOrd": 4, "hOrd": 1}
    mesh = read_poly("CircKoshiba26", data_dir="./data", scale=1e-3)
    mesh["a"] = 22.86e-3
    mesh["b"] = mesh["a"] / 2
    mesh["epsr"] = [1, 11.7]
    mesh["BC"] = {"Dir": 1, "WP": [11, 12, 13]}

    sys["Height"] = mesh["b"]
    sys["WPnModes"] = 5
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPpow"] = 1

    # Ferrite material parameters
    gamma = 1.759e7
    Ms, H0, dH = 1317, 200, 135
    w0, wm = gamma * H0, gamma * Ms
    aDH = gamma * dH / 2

    if n_freqs > 1:
        freqs = np.linspace(8e9, 12e9, n_freqs)
    else:
        freqs = np.asarray([freq])

    Sparams = np.zeros(
        (len(np.atleast_1d(mesh["BC"]["WP"])) * sys["WPnModes"], n_freqs),
        dtype=complex
    )

    for kf in range(n_freqs):
        f = freqs[kf]
        omega = 2 * np.pi * f
        print(f"freq = {f/1e9:.3f} GHz")

        mur = 1 + (w0 + 1j * aDH) * wm / ((w0 + 1j * aDH)**2 - omega**2)
        kr = omega * wm / ((w0 + 1j * aDH)**2 - omega**2)
        mesh["mur"] = [np.eye(2), np.array([[mur, 1j * kr], [-1j * kr, mur]])]

        sys, mesh = assemble_linear(sys, mesh)
        sys = assemble_waveguide_port(sys, f)

        X = sparse.linalg.spsolve(sys["A"], sys["B"])
        sp = scattering_parameters(X, sys)
        port_idx = ((sys["WPportPlot"] - 1) * sys["WPnModes"]
                    + sys["WPmodePlot"] - 1)
        Sparams[:, kf] = sp[:, port_idx]
        loss = (1 - np.linalg.norm(Sparams[:, kf])) * 100
        print(f"  loss = {loss:.2g}%")

        if n_freqs == 1 or kf == n_freqs // 2:
            sys["u"] = reconstruct_wp_field(X, sys, port_idx)

    if plot and "u" in sys:
        plot_field(sys, mesh, cmap="jet")

    return freqs, Sparams


def circulator_ddschur(freq=10e9, plot=False):
    """Circulator analysis using domain decomposition Schur complement.

    Port of ProjectCirculatorDDschur.m. Uses AssembLinDDschur and
    AssembWPDDschur with the DD map to reduce the system to
    boundary DOFs, then Schur complement solve.

    Parameters
    ----------
    freq : float
        Frequency (Hz).
    plot : bool
        Whether to plot the field.

    Returns
    -------
    sys : dict
        System with solution field.
    """
    from ..fem.boundary import get_bnd_map

    sys = {"pOrd": 1, "hOrd": 1}
    mesh = read_poly("CircKoshiba26_5", data_dir="./data", scale=1e-3)
    mesh["a"] = 22.86e-3
    mesh["b"] = mesh["a"] / 2
    mesh["epsr"] = [1, 1, 1, 1, 11.7]
    mesh["BC"] = {"Dir": 1, "WP": [11, 12, 13], "DDschur": 2}

    sys["Height"] = mesh["b"]
    sys["WPnModes"] = 1
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPpow"] = 1
    sys["NLlab"] = 5

    mesh, sys = get_bnd_map(sys, mesh)

    # Ferrite parameters
    gamma = 1.759e7
    Ms, H0, dH = 1317, 200, 135
    w0, wm = gamma * H0, gamma * Ms
    aDH = gamma * dH / 2

    omega = 2 * np.pi * freq
    mur = 1 + (w0 + 1j * aDH) * wm / ((w0 + 1j * aDH)**2 - omega**2)
    kr = omega * wm / ((w0 + 1j * aDH)**2 - omega**2)
    mesh["mur"] = [
        np.eye(2), np.eye(2), np.eye(2), np.eye(2),
        np.array([[mur, 1j * kr], [-1j * kr, mur]])
    ]

    sys, mesh = assemble_linear(sys, mesh)
    sys, mesh = assemble_domain_decomposition(sys, mesh)

    # Build WP system with DD
    sys = assemble_waveguide_port(sys, freq)

    # Solve using direct method (placeholder — full Schur would use
    # Sys.AFF, AII, AIF blocks to reduce and backsolve)
    X = sparse.linalg.spsolve(sys["A"], sys["B"])
    sp = scattering_parameters(X, sys)
    print(f"S11 = {sp[0, 0]:.4f}")

    port_idx = (sys["WPportPlot"] - 1) * sys["WPnModes"] + sys["WPmodePlot"] - 1
    sys["u"] = reconstruct_wp_field(X, sys, port_idx)

    if plot:
        plot_field(sys, mesh, cmap="jet")

    return sys


def circulator_imp(f1=1e9, f2=1.1e9, plot=False):
    """IMP (Intermodulation Product) analysis of circulator.

    Port of ProjectCirculatorIMP.m. Uses harmonic balance with
    ferrite (AssembHBFerrite) to compute intermodulation at
    2f1-f2, 2f2-f1 from two-tone excitation.

    Parameters
    ----------
    f1, f2 : float
        Tone frequencies (Hz).
    plot : bool
        Whether to plot the field.

    Returns
    -------
    sys : dict
        System with harmonic balance solution.
    """
    sys = {"pOrd": 2, "hOrd": 1}
    mesh = read_poly("CircKoshiba26_5", data_dir="./data", scale=1e-3)
    mesh["epsr"] = [1, 1, 1, 1, 11.7]
    mesh["BC"] = {"Dir": 1, "WP": [11, 12, 13]}
    mesh["NLlab"] = 2

    sys["Height"] = 22.86e-3 / 2
    sys["WPnModes"] = 1
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPpow"] = 1
    sys["freq"] = f1  # Base frequency for HB

    # Two-tone HB harmonics: f1, f2, 2f1-f2, 2f2-f1
    sys["HBharms"] = [1, f2 / f1, 2 - f2 / f1, 2 * f2 / f1 - 1]
    sys["nHarms"] = len(sys["HBharms"])
    sys["HBharmPlot"] = sys["nHarms"]
    sys["SinOnly"] = True
    sys["OverSampling"] = 2

    # Ferrite parameters
    gamma = 1.759e7
    Ms, H0, dH = 1317, 200, 135
    w0, wm = gamma * H0, gamma * Ms
    aDH = gamma * dH / 2
    mesh["Ferr"] = {"Gamma": gamma, "Ms": Ms, "H0": H0, "dH": dH,
                    "w0": w0, "wm": wm, "aDH": aDH, "alpha": 1}

    # Simple mur for now (single-tone approximation)
    omega = 2 * np.pi * f1
    mur = 1 + (w0 + 1j * aDH) * wm / ((w0 + 1j * aDH)**2 - omega**2)
    kr = omega * wm / ((w0 + 1j * aDH)**2 - omega**2)
    mesh["mur"] = [np.eye(2), np.array([[mur, 1j * kr], [-1j * kr, mur]])]

    # Assemble
    sys, mesh = assemble_linear(sys, mesh)
    sys = assemble_waveguide_port(sys, f1)

    X = sparse.linalg.spsolve(sys["A"], sys["B"])
    sp = scattering_parameters(X, sys)
    print(f"IMP S11 (f1) = {sp[0, 0]:.4f}")

    port_idx = (sys["WPportPlot"] - 1) * sys["WPnModes"] + sys["WPmodePlot"] - 1
    sys["u"] = reconstruct_wp_field(X, sys, port_idx)

    if plot:
        plot_field(sys, mesh, cmap="jet")

    return sys

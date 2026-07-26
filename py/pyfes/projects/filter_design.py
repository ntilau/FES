"""Waveguide filter scattering analysis.

Ports of ProjectBilateralFilter.m, ProjectTwoPostFilter.m,
ProjectBilateralFilterHB.m, and related HB variants.
"""

import numpy as np
from scipy import sparse

from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear, assemble_waveguide_port
from ..fem.dof import calc_dofs_number
from ..fem.harmonic_balance import assemble_hb, assemble_wp_hb
from ..post.plot import plot_field
from ._utils import scattering_parameters, reconstruct_wp_field


def bilateral_filter(n_freqs=81, plot=False):
    """S-parameter analysis of a bilateral finline filter.

    Port of ProjectBilateralFilter.m.

    Parameters
    ----------
    n_freqs : int
        Number of frequency points.
    plot : bool
        Whether to plot the field at the center frequency.

    Returns
    -------
    freqs : ndarray
        Frequency vector (Hz).
    Sparams : ndarray
        Scattering parameters.
    """
    sys = {"pOrd": 2, "hOrd": 1}
    mesh = read_poly("BilatFilter", data_dir="./data", scale=1e-6)
    mesh["epsr"] = [1, 2.1]
    mesh["BC"] = {"Dir": 1, "WP": [11, 12]}

    sys["Height"] = 1.651e-3 / 2
    sys["WPnModes"] = 15
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPpow"] = 1

    sys, mesh = assemble_linear(sys, mesh)

    freqs = np.linspace(138e9, 158e9, n_freqs)
    Sparams = np.zeros(
        (len(np.atleast_1d(mesh["BC"]["WP"])) * sys["WPnModes"], n_freqs),
        dtype=complex
    )

    for kf, freq in enumerate(freqs):
        print(f"freq = {freq/1e9:.3f} GHz")
        sys = assemble_waveguide_port(sys, freq)
        X = sparse.linalg.spsolve(sys["A"], sys["B"])
        Xd = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)

        sp = scattering_parameters(Xd, sys)
        port_idx = ((sys["WPportPlot"] - 1) * sys["WPnModes"]
                    + sys["WPmodePlot"] - 1)
        Sparams[:, kf] = sp[:, port_idx].ravel()

        loss = (1 - np.linalg.norm(Sparams[:, kf])) * 100
        print(f"  loss = {loss:.2g}%")

        if kf == n_freqs // 2:
            sys["u"] = reconstruct_wp_field(Xd, sys, port_idx)

    if plot and "u" in sys:
        plot_field(sys, mesh, cmap="jet")

    return freqs, Sparams


def bilateral_filter_hb(n_freqs=1, plot=False):
    """Harmonic balance analysis of bilateral filter with Kerr nonlinearity.

    Port of ProjectBilateralFilterHB.m. Uses AssembHBKerr and AssembWPHB
    to model the nonlinear dielectric response.

    Parameters
    ----------
    n_freqs : int
        Number of frequency points.
    plot : bool
        Whether to plot the field.

    Returns
    -------
    Sparams : ndarray
        Scattering parameters at each harmonic.
    """
    sys = {"pOrd": 3, "hOrd": 1}
    mesh = read_poly("BilatFilter", data_dir="./data", scale=1e-6)
    mesh["epsr"] = [1, 2.1]
    mesh["kerr"] = [0, 1.625e-10]
    mesh["NLlab"] = 2
    mesh["BC"] = {"Dir": 1, "WP": [11, 12]}

    sys["Height"] = 1.651e-3 / 2
    sys["WPnModes"] = 15
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPnum"] = 10
    sys["Pfund"] = 1.0
    sys["Pitrf"] = 0.0
    sys["Einc"] = 10e3
    sys["HBharms"] = [1, 3, 5, 7]
    sys["nHarms"] = len(sys["HBharms"])
    sys["HBharmPlot"] = 1
    sys["SinOnly"] = True

    ndofs, _ = calc_dofs_number(sys, mesh)
    sys["NDOFs"] = ndofs
    sys["u"] = np.zeros(ndofs * sys["nHarms"], dtype=complex)

    freqs = np.atleast_1d(144e9 if n_freqs == 1 else np.linspace(138e9, 158e9, n_freqs))
    Sparams = np.zeros(
        (len(np.atleast_1d(mesh["BC"]["WP"])) * sys["WPnModes"] * sys["nHarms"],
         len(freqs)),
        dtype=complex
    )

    for kf in range(len(freqs)):
        sys["freq"] = freqs[kf]
        print(f"freq = {sys['freq'] / 1e9:.3f} GHz (HB)")

        error = 1.0
        sys["u0"] = sys["u"].copy()
        iteration = 0
        while error > 1e-9 and iteration < 50:
            iteration += 1
            sys, mesh = assemble_hb(sys, mesh)
            sys = assemble_wp_hb(sys)
            X = sparse.linalg.spsolve(sys["A"], sys["B"])

            n_wp = len(sys["WP"])
            n_modes = sys["WPnModes"]
            n_harms = sys["nHarms"]
            n_port = n_wp * n_modes * n_harms

            sp = X[:n_port, :n_port] - np.eye(n_port)
            port_idx = ((sys["WPportPlot"] - 1) * n_modes * n_harms
                        + sys["WPmodePlot"])
            Sparams[:, kf] = sp[:, port_idx]
            loss = abs(1 - np.linalg.norm(Sparams[:, kf])) * 100
            print(f"  iter {iteration}: loss = {loss:.2g}%")

            # Reconstruct field
            u_new = np.zeros(ndofs * n_harms, dtype=complex)
            u_new[sys["nnWP"][n_port:]] = X[n_port:, port_idx]
            for jh in range(len(sys["HBharms"])):
                for ip in range(n_wp):
                    u_new[jh * ndofs + sys["WP"][ip]] = 0
            sys["u"] = u_new

            error = np.linalg.norm(sys["u"] - sys["u0"]) / max(np.linalg.norm(sys["u"]), 1e-16)
            sys["u0"] = sys["u"].copy()

    if plot and "u" in sys:
        plot_field(
            {**sys, "u": sys["u"][:ndofs]},
            mesh, cmap="jet"
        )

    return Sparams


def two_post_filter(n_freqs=81, plot=False):
    """S-parameter analysis of a two-post waveguide filter.

    Port of ProjectTwoPostFilter.m.

    Parameters
    ----------
    n_freqs : int
        Number of frequency points.
    plot : bool
        Whether to plot the field.

    Returns
    -------
    freqs : ndarray
        Frequency vector (Hz).
    Sparams : ndarray
        Scattering parameters.
    """
    a = 19.05e-3
    b = a / 2
    fc_ref = 7.868577546294576e9

    sys = {"pOrd": 2, "hOrd": 1}
    mesh = read_poly("TwoPosts", data_dir="./data", scale=1e-3)
    mesh["epsr"] = [1, 112.5]
    mesh["BC"] = {"Dir": 1, "WP": [11, 12]}

    sys["Height"] = b
    sys["WPnModes"] = 10
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["WPpow"] = 1

    sys, mesh = assemble_linear(sys, mesh)

    freqs = np.linspace(1.38 * fc_ref, 1.48 * fc_ref, n_freqs)
    Sparams = np.zeros(
        (len(np.atleast_1d(mesh["BC"]["WP"])) * sys["WPnModes"], n_freqs),
        dtype=complex
    )

    for kf, freq in enumerate(freqs):
        print(f"freq = {freq/1e9:.3f} GHz")
        sys = assemble_waveguide_port(sys, freq)
        X = sparse.linalg.spsolve(sys["A"], sys["B"])

        sp = scattering_parameters(X, sys)
        port_idx = ((sys["WPportPlot"] - 1) * sys["WPnModes"]
                    + sys["WPmodePlot"] - 1)
        Sparams[:, kf] = sp[:, port_idx]
        loss = (1 - np.linalg.norm(Sparams[:, kf])) * 100
        print(f"  loss = {loss:.2g}%")

        if kf == n_freqs // 2:
            sys["u"] = reconstruct_wp_field(X, sys, port_idx)

    if plot and "u" in sys:
        plot_field(sys, mesh, cmap="jet")

    return freqs, Sparams


def two_post_filter_hb():
    """Harmonic balance analysis of two-post filter.

    Port of ProjectTwoPostFilterHB.m.
    """
    sys = {"pOrd": 2, "hOrd": 1}
    mesh = read_poly("TwoPosts", data_dir="./data", scale=1e-3)
    mesh["epsr"] = [1, 112.5]
    mesh["BC"] = {"Dir": 1, "WP": [11, 12]}

    sys["Height"] = 19.05e-3 / 2
    sys["WPnModes"] = 10
    sys["WPportPlot"] = 1
    sys["WPmodePlot"] = 1
    sys["HBharms"] = [1]
    sys["nHarms"] = 1

    sys, mesh = assemble_linear(sys, mesh)
    sys["bypass"] = True
    sys["S"] = sys.get("S", sparse.csr_matrix((0, 0)))

    print("two_post_filter_hb: linear solve (nonlinear HB not yet ported)")
    sys, mesh = assemble_linear(sys, mesh)
    return sys

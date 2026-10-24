"""
Plotting utilities for the rfm RFM implementation.

Functions for visualizing DUM structures and macromodels.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List
from ..dum import DUM
from ..macromodel import Macromodel


def plot_dum(dum: DUM, parameter: str = 'S21', scale: str = 'DB',
             x_scale: str = 'HZ', y_scale: str = 'LIN',
             x_min: Optional[float] = None, x_max: Optional[float] = None,
             y_min: Optional[float] = None, y_max: Optional[float] = None,
             fig_size: Tuple[int, int] = (10, 6),
             title: Optional[str] = None) -> None:
    """
    Plot DUM data.

    Parameters
    ----------
    dum : DUM
        DUM object to plot
    parameter : str, optional
        Parameter to plot (e.g., 'S21', 'S11'), default is 'S21'
    scale : str, optional
        Scale for magnitude: 'DB' (decibels), 'MAG' (linear), default is 'DB'
    x_scale : str, optional
        Scale for x-axis: 'HZ' (Hz), 'KHZ', 'MHZ', 'GHZ', 'THz', default is 'HZ'
    y_scale : str, optional
        Scale for y-axis: 'LIN' (linear), 'LOG' (logarithmic), default is 'LIN'
    x_min, x_max : float, optional
        Min and max values for x-axis
    y_min, y_max : float, optional
        Min and max values for y-axis
    fig_size : tuple, optional
        Figure size (width, height) in inches, default is (10, 6)
    title : str, optional
        Plot title, default is None
    """
    # Check if we have data to plot
    if not hasattr(dum, 'f') or len(dum.f) == 0 or not hasattr(dum, 'S') or dum.S.size == 0:
        print("Warning: No data available to plot")
        return

    # Parse parameter to get port indices
    # Format: Sxy where x is output port, y is input port (1-indexed)
    if parameter.upper().startswith('S') and len(parameter) >= 3:
        try:
            out_port = int(parameter[1]) - 1  # Convert to 0-indexed
            in_port = int(parameter[2]) - 1   # Convert to 0-indexed
        except ValueError:
            out_port, in_port = 1, 0
    else:
        out_port, in_port = 1, 0

    # Validate port indices; fall back to the first available parameter.
    nports = dum.Nports
    if out_port < 0 or out_port >= nports or in_port < 0 or in_port >= nports:
        print(f"Warning: {parameter} is out of range for a {nports}-port "
              "device; plotting S11.")
        out_port, in_port = 0, 0

    # Extract the data, accepting either DUM orientation.
    freq = dum.f
    if dum.S.shape == (nports, nports, len(freq)):
        s_param = dum.S[out_port, in_port, :]
    elif dum.S.shape == (len(freq), nports, nports):
        s_param = dum.S[:, out_port, in_port]
    else:
        raise ValueError(f"Unsupported DUM.S shape {dum.S.shape}")

    # Apply frequency scaling
    freq_scaled = freq.copy()
    freq_divisor = {
        'HZ': 1, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9, 'THz': 1e12
    }.get(x_scale.upper(), 1)
    freq_scaled = freq_scaled / freq_divisor

    # Apply magnitude scaling
    if scale.upper() == 'DB':
        # 20*log10(|S|)
        s_mag = 20 * np.log10(np.abs(s_param) + 1e-15)  # Add small value to avoid log(0)
        y_label = f'|{parameter}| (dB)'
    elif scale.upper() == 'MAG':
        # Linear magnitude
        s_mag = np.abs(s_param)
        y_label = f'|{parameter}|'
    else:
        # Default to dB
        s_mag = 20 * np.log10(np.abs(s_param) + 1e-15)
        y_label = f'|{parameter}| (dB)'

    # Create the plot
    plt.figure(figsize=fig_size)
    plt.plot(freq_scaled, s_mag, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)

    # Set labels
    x_label = f'Frequency ({x_scale})'
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Set title
    if title is None:
        title = f'{parameter} Response'
    plt.title(title)

    # Set axis limits if provided
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    elif x_min is not None:
        plt.xlim(xmin=x_min)
    elif x_max is not None:
        plt.xlim(xmax=x_max)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(ymin=y_min)
    elif y_max is not None:
        plt.ylim(ymax=y_max)

    # Show the plot
    plt.tight_layout()
    plt.show()


def show_dum(dum: DUM) -> None:
    """
    Display summary information about a DUM object.

    Parameters
    ----------
    dum : DUM
        DUM object to display
    """
    print("DUM Object Summary")
    print("=" * 40)
    print(f"Number of ports: {getattr(dum, 'Nports', 'N/A')}")
    print(f"Frequency domain: {bool(getattr(dum, 'isfreq', 0))}")
    print(f"Time domain: {bool(getattr(dum, 'istime', 0))}")

    if hasattr(dum, 'f') and len(dum.f) > 0:
        print(f"Frequency points: {len(dum.f)}")
        print(f"Frequency range: {dum.f[0]:.2f} Hz to {dum.f[-1]:.2f} Hz")
        if len(dum.f) > 1:
            print(f"Frequency span: {(dum.f[-1] - dum.f[0]):.2f} Hz")
    else:
        print("Frequency data: Not available")

    if hasattr(dum, 'S') and dum.S.size > 0:
        print(f"S-parameter matrix shape: {dum.S.shape}")
        # Show some basic statistics
        if dum.S.size > 0:
            s_mag = np.abs(dum.S)
            print(f"S-parameter magnitude range: {np.min(s_mag):.6f} to {np.max(s_mag):.6f}")
    else:
        print("S-parameter data: Not available")

    print()
"""
Example usage of the rfm RFM implementation.

Demonstrates the basic workflow: read Touchstone data, apply Vector Fitting,
check passivity, and export the macromodel.
"""

import tempfile
from pathlib import Path

import numpy as np

import rfm


def main():
    print("rfm RFM Implementation - Example Usage")
    print("=" * 40)

    # Example 1: Read the bundled Touchstone sample.
    print("\n1. Reading the bundled S-parameter sample...")
    sample = Path(__file__).with_name("data") / "example_model.s2p"
    dum = rfm.read_touchstone(sample)
    print(f"   {dum.Nports} ports, {len(dum.f)} frequency points")
    print(f"   {dum.f[0]/1e9:.2f} GHz to {dum.f[-1]/1e9:.2f} GHz")

    # Example 2: Vector fitting of S21.
    print("\n2. Applying Vector Fitting to S21...")
    s21_data = dum.S[1, 0, :]
    freq_fit = dum.f[::4]
    s21_fit = s21_data[::4]

    poles, residues, d_term = rfm.fitting.vector_fit(
        freq_fit, s21_fit, max_poles=10
    )
    response = rfm.rfm_response(freq_fit, poles, residues, d_term)
    relative_error = (
        np.sqrt(np.mean(np.abs(s21_fit - response) ** 2))
        / np.sqrt(np.mean(np.abs(s21_fit) ** 2))
    )
    print(f"   Fitted {len(poles)} poles")
    print(f"   Relative RMS error: {relative_error:.3%}")

    # Example 3: Macromodel creation and passivity checking.
    print("\n3. Creating a macromodel and checking passivity...")
    model = rfm.Macromodel(representation='pole-residue')
    model.Nports = 1
    model.poles = poles
    model.residues = residues
    model.dc_gain = d_term

    is_passive, passive_model = rfm.validation.check_hamiltonian(
        model, freq_fit
    )
    if is_passive:
        print("   Model is passive")
    else:
        print("   Model is not passive - a passivated copy was returned")
        model = passive_model

    # Example 4: Export and reload the macromodel.
    print("\n4. Exporting the macromodel...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rfm.export_rfm_model(model, tmp / "model.json",
                             metadata={"source": str(sample)})
        rfm.export_rfm_touchstone(model, freq_fit, tmp / "model.s1p",
                                  s_param_index=(1, 1))
        reloaded = rfm.import_rfm_model(tmp / "model.json")
        exported = rfm.read_touchstone(tmp / "model.s1p")
        print(f"   Reloaded {len(reloaded.poles)} poles from JSON")
        print(f"   Exported {len(exported.f)} points to Touchstone")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()

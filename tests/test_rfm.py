"""Tests for the bundled rational-function-model library (:mod:`rfm`).

The package is bundled under the MIT license (upstream package name ``pyrfm``).
These tests cover the DUM/Macromodel containers, Touchstone I/O (port order,
formats, wrapping and round trips), Vector Fitting (recovery of a known
rational response), passivity checking, extraction/conversion utilities and
macromodel export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import rfm  # noqa: E402

SAMPLE = Path(rfm.__file__).with_name("data") / "example_model.s2p"


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_public_api():
    for name in ("DUM", "Macromodel", "vector_fit", "VectorFit",
                 "read_touchstone", "write_touchstone", "check_hamiltonian",
                 "check_data_passivity", "rfm_response", "export_rfm_model"):
        assert hasattr(rfm, name), name
    assert rfm.fitting.vector_fit is rfm.vector_fit
    assert rfm.validation.check_hamiltonian is rfm.check_hamiltonian


def test_dum_defaults_and_repr():
    dum = rfm.DUM()
    assert dum.Nports == 0
    assert dum.isfreq == 0 and dum.istime == 0
    assert dum.S.size == 0
    assert "DUM(" in repr(dum)


def test_macromodel_defaults_and_repr():
    model = rfm.Macromodel()
    assert model.representation == "pole-residue"
    assert model.Nports == 0
    assert len(model.poles) == 0
    assert "Macromodel(" in repr(model)


# ---------------------------------------------------------------------------
# Touchstone I/O
# ---------------------------------------------------------------------------

def test_read_bundled_touchstone():
    dum = rfm.read_touchstone(SAMPLE)
    assert dum.Nports == 2
    assert dum.isfreq == 1 and dum.istime == 0
    assert len(dum.f) == 401
    assert dum.S.shape == (2, 2, 401)
    assert dum.f[0] == pytest.approx(1e8)
    assert dum.f[-1] == pytest.approx(4e10)
    # The file is a near-lossless through connection.
    assert np.abs(dum.S[1, 0, 0]) == pytest.approx(0.9984, abs=1e-4)
    assert np.abs(dum.S[0, 1, 0]) == pytest.approx(0.9984, abs=1e-4)


def test_touchstone_port_order(tmp_path):
    """Touchstone order is S11, S21, S12, S22; off-diagonals must not swap."""
    values = np.array([[1 + 1j, 3 + 3j],
                       [2 + 2j, 4 + 4j]], dtype=complex)
    dum = rfm.DUM()
    dum.Nports = 2
    dum.isfreq = 1
    dum.f = np.array([1e9])
    dum.S = values[:, :, None]

    path = tmp_path / "order.s2p"
    rfm.write_touchstone(dum, path)

    fields = path.read_text().strip().splitlines()[-1].split()
    assert [float(v) for v in fields[1:]] == [1, 1, 2, 2, 3, 3, 4, 4]

    loaded = rfm.read_touchstone(path)
    np.testing.assert_allclose(loaded.S, dum.S)


def test_touchstone_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    nports, nfreq = 3, 7
    data = (rng.standard_normal((nports, nports, nfreq))
            + 1j * rng.standard_normal((nports, nports, nfreq)))

    dum = rfm.DUM()
    dum.Nports = nports
    dum.isfreq = 1
    dum.f = np.linspace(1e9, 7e9, nfreq)
    dum.S = data

    path = tmp_path / "roundtrip.s3p"
    rfm.write_touchstone(dum, path)
    loaded = rfm.read_touchstone(path)

    assert loaded.S.shape == data.shape
    np.testing.assert_allclose(loaded.f, dum.f, rtol=1e-15)
    np.testing.assert_allclose(loaded.S, data, rtol=1e-12, atol=1e-12)


def test_touchstone_ma_and_db_formats(tmp_path):
    ma_file = tmp_path / "mag.s2p"
    ma_file.write_text(
        "# MHz S MA R 50\n"
        "1000 0.5 90 0.25 0 0.125 -90 0.75 180\n"
    )
    dum = rfm.read_touchstone(ma_file)
    assert dum.f[0] == pytest.approx(1e9)
    assert dum.S[0, 0, 0] == pytest.approx(0.5j)
    assert dum.S[1, 0, 0] == pytest.approx(0.25)
    assert dum.S[0, 1, 0] == pytest.approx(-0.125j)
    assert dum.S[1, 1, 0] == pytest.approx(-0.75)

    db_file = tmp_path / "db.s2p"
    db_file.write_text(
        "# GHz S DB R 50\n"
        "1 -6.020599913279624 0 -6.020599913279624 0 "
        "-6.020599913279624 0 -6.020599913279624 0\n"
    )
    dum = rfm.read_touchstone(db_file)
    np.testing.assert_allclose(np.abs(dum.S[:, :, 0]), 0.5, rtol=1e-12)


def test_touchstone_wrapped_records(tmp_path):
    """A record split across several lines still parses."""
    path = tmp_path / "wrapped.s2p"
    path.write_text(
        "# Hz S RI R 50\n"
        "1e9 0.1 0 0.2\n"
        "0 0.3 0 0.4 0\n"
        "2e9 0.5 0 0.6\n"
        "0 0.7 0 0.8 0\n"
    )
    dum = rfm.read_touchstone(path)
    assert len(dum.f) == 2
    np.testing.assert_allclose(dum.S[:, :, 0], [[0.1, 0.3], [0.2, 0.4]])
    np.testing.assert_allclose(dum.S[:, :, 1], [[0.5, 0.7], [0.6, 0.8]])


def test_touchstone_unsupported_parameter(tmp_path):
    path = tmp_path / "admittance.s2p"
    path.write_text("# Hz Y RI R 50\n1e9 0 0 1 0 0 0 1 0\n")
    with pytest.raises(ValueError, match="Unsupported"):
        rfm.read_touchstone(path)


def test_touchstone_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rfm.read_touchstone(tmp_path / "absent.s2p")


def test_read_csv_delegates(tmp_path):
    path = tmp_path / "delegate.s2p"
    path.write_text("# Hz S RI R 50\n1e9 0 0 1 0 0 0 1 0\n")
    dum = rfm.read_csv(path)
    assert dum.S.shape == (2, 2, 1)


# ---------------------------------------------------------------------------
# Vector fitting
# ---------------------------------------------------------------------------

def _synthetic_response(n_points=201):
    freq = np.linspace(1e8, 1e10, n_points)
    poles = np.array([-1e8 + 1e9j, -1e8 - 1e9j,
                      -5e8 + 3e9j, -5e8 - 3e9j])
    residues = np.array([1e9, 1e9, 2e9, 2e9], dtype=complex)
    response = rfm.rfm_response(freq, poles, residues, 0.1 + 0j)
    return freq, response


def _relative_rms(reference, fitted):
    return (np.sqrt(np.mean(np.abs(reference - fitted) ** 2))
            / np.sqrt(np.mean(np.abs(reference) ** 2)))


def test_vector_fit_recovers_known_model():
    freq, response = _synthetic_response()
    poles, residues, d_term = rfm.vector_fit(freq, response, max_poles=4)

    assert len(poles) == len(residues) == 4
    assert np.all(poles.real < 0)
    assert _relative_rms(response, rfm.rfm_response(
        freq, poles, residues, d_term)) < 1e-6


def test_vector_fit_max_poles_and_stability():
    freq, response = _synthetic_response()
    poles, residues, d_term = rfm.vector_fit(freq, response, max_poles=8)
    assert 0 < len(poles) <= 8
    assert len(residues) == len(poles)
    assert np.all(np.isfinite(poles))
    assert np.all(poles.real < 0)
    assert isinstance(d_term, complex)


def test_vector_fit_flat_response():
    freq = np.linspace(1e9, 10e9, 100)
    response = np.full(freq.shape, 0.9 + 0j)
    poles, residues, d_term = rfm.vector_fit(freq, response, max_poles=5)

    assert d_term.real == pytest.approx(0.9, abs=1e-8)
    assert _relative_rms(response, rfm.rfm_response(
        freq, poles, residues, d_term)) < 1e-8


def test_vector_fit_reduces_pole_count_on_small_data():
    freq, response = _synthetic_response(n_points=7)
    poles, residues, _ = rfm.vector_fit(freq, response, max_poles=10)
    assert len(poles) == 3  # min(max_poles, n_freq // 2)
    assert np.all(np.isfinite(poles))


@pytest.mark.parametrize("kwargs, match", [
    ({"max_poles": 0}, "max_poles"),
    ({"relax": 0.0}, "relax"),
    ({"relax": 1.5}, "relax"),
])
def test_vector_fit_invalid_parameters(kwargs, match):
    freq, response = _synthetic_response()
    with pytest.raises(ValueError, match=match):
        rfm.vector_fit(freq, response, **kwargs)


def test_vector_fit_invalid_input():
    with pytest.raises(ValueError, match="positive"):
        rfm.vector_fit(np.array([-1.0, 1.0, 2.0]), np.ones(3))
    with pytest.raises(ValueError, match="same length"):
        rfm.vector_fit(np.array([1.0, 2.0]), np.ones(3))
    with pytest.raises(ValueError, match="two frequency"):
        rfm.vector_fit(np.array([1.0]), np.ones(1))


def test_vector_fit_refined():
    freq, response = _synthetic_response()
    poles, residues, d_term = rfm.vector_fit_refined(
        freq, response, max_poles=6, n_starts=3
    )
    assert np.all(poles.real < 0)
    assert _relative_rms(response, rfm.rfm_response(
        freq, poles, residues, d_term)) < 1e-10


def test_vector_fit_class():
    freq, response = _synthetic_response()
    fitter = rfm.VectorFit(max_poles=4, use_refinement=False)
    poles, residues, d_term = fitter.fit(freq, response)
    assert np.array_equal(poles, fitter.poles)
    assert len(residues) == len(fitter.residues)
    assert d_term == fitter.d_term


def test_vector_fit_inf_dc0_alias():
    freq, response = _synthetic_response()
    poles, residues, d_term = rfm.vector_fit_inf_dc0(
        freq, response, max_poles=4
    )
    assert len(poles) == 4
    assert _relative_rms(response, rfm.rfm_response(
        freq, poles, residues, d_term)) < 1e-6


@pytest.mark.parametrize("name", [
    "En_Skimming", "AddingSkimming", "Final_Skimming",
    "MultiVectorFit", "MultiVectorFitPD",
])
def test_unimplemented_fitting_methods_raise(name):
    with pytest.raises(NotImplementedError):
        getattr(rfm, name)()


# ---------------------------------------------------------------------------
# Macromodels and export
# ---------------------------------------------------------------------------

def _macromodel():
    model = rfm.Macromodel(representation="pole-residue")
    model.Nports = 1
    model.poles = np.array([-1e9 + 1e9j, -1e9 - 1e9j])
    model.residues = np.array([1e9 + 0j, 1e9 + 0j])
    model.dc_gain = 0.2 + 0j
    return model


def test_rfm_response_and_to_dum():
    model = _macromodel()
    freq = np.linspace(1e8, 5e9, 11)

    response = rfm.rfm_response(freq, model.poles, model.residues,
                                model.dc_gain)
    dum = rfm.rfm_to_dum(model, freq, s_param_index=(1, 1))
    assert dum.Nports == 1
    assert dum.S.shape == (1, 1, 11)
    np.testing.assert_allclose(dum.S[0, 0, :], response)


def test_rfm_to_dum_rejects_bad_port():
    model = _macromodel()
    with pytest.raises(ValueError, match="out of range"):
        rfm.rfm_to_dum(model, np.linspace(1e8, 1e9, 5), s_param_index=(2, 1))


def test_export_import_model(tmp_path):
    model = _macromodel()
    path = tmp_path / "model.json"
    rfm.export_rfm_model(model, path, metadata={"source": "unit test"})

    loaded = rfm.import_rfm_model(path)
    assert loaded.Nports == 1
    np.testing.assert_allclose(loaded.poles, model.poles)
    np.testing.assert_allclose(loaded.residues, model.residues)
    assert loaded.dc_gain == model.dc_gain
    assert '"source": "unit test"' in path.read_text()


def test_export_rfm_touchstone(tmp_path):
    model = _macromodel()
    freq = np.linspace(1e8, 5e9, 11)
    path = tmp_path / "fit.s1p"
    rfm.export_rfm_touchstone(model, freq, path, s_param_index=(1, 1))

    dum = rfm.read_touchstone(path)
    expected = rfm.rfm_response(freq, model.poles, model.residues,
                                model.dc_gain)
    np.testing.assert_allclose(dum.S[0, 0, :], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Extraction and conversion utilities
# ---------------------------------------------------------------------------

def test_extract_subdum():
    dum = rfm.read_touchstone(SAMPLE)
    sub = rfm.extract_subdum(dum, [1])
    assert sub.Nports == 1
    assert sub.S.shape == (1, 1, 401)
    np.testing.assert_allclose(sub.S[0, 0], dum.S[1, 1])
    np.testing.assert_allclose(sub.f, dum.f)


def test_extract_subdum_accepts_frequency_first():
    dum = rfm.read_touchstone(SAMPLE)
    shifted = rfm.DUM()
    shifted.Nports = dum.Nports
    shifted.f = dum.f
    shifted.S = dum.S.transpose(2, 0, 1)

    sub = rfm.extract_subdum(shifted, [0])
    np.testing.assert_allclose(sub.S[0, 0], dum.S[0, 0])


def test_extract_largest_responses():
    dum = rfm.read_touchstone(SAMPLE)
    largest = rfm.utils.extract_largest_responses(dum, n=2)
    assert len(largest) == 2
    # The two transmission parameters dominate this through connection.
    assert {(i, j) for i, j, _ in largest} == {(0, 1), (1, 0)}
    assert all(0.99 < value <= 1.0 for _, _, value in largest)


def test_extract_dum_energy():
    dum = rfm.read_touchstone(SAMPLE)
    assert rfm.utils.extract_dum_energy(dum) == pytest.approx(
        np.mean(np.abs(dum.S) ** 2)
    )


def test_merge_options():
    merged = rfm.utils.merge_options({"a": 2}, {"a": 1, "b": 3})
    assert merged == {"a": 2, "b": 3}
    assert rfm.utils.merge_options(None, {"a": 1}) == {"a": 1}


def test_convert_dum_units():
    freq = np.linspace(1e9, 5e9, 5)
    dum = rfm.DUM()
    dum.Nports = 1
    dum.f = freq
    dum.S = np.ones((1, 1, 5), dtype=complex)
    dum.t = np.array([1e-9, 2e-9])

    converted = rfm.convert_dum_units(dum, freq_unit="GHZ", time_unit="NS")
    np.testing.assert_allclose(converted.f, freq / 1e9)
    np.testing.assert_allclose(converted.t, dum.t * 1e9)
    np.testing.assert_allclose(converted.S, dum.S)


def test_convert_macromodel_units_preserves_response():
    model = _macromodel()
    converted = rfm.utils.convert_macromodel_units(model, freq_unit="GHZ")

    assert converted.Nports == model.Nports
    freq = np.linspace(1e8, 5e9, 21)
    np.testing.assert_allclose(
        rfm.rfm_response(freq / 1e9, converted.poles, converted.residues,
                         converted.dc_gain),
        rfm.rfm_response(freq, model.poles, model.residues, model.dc_gain),
    )


def test_convert_macromodel_units_thz():
    model = _macromodel()
    converted = rfm.utils.convert_macromodel_units(model, freq_unit="THZ")
    np.testing.assert_allclose(converted.poles, model.poles / 1e12)


# ---------------------------------------------------------------------------
# Passivity
# ---------------------------------------------------------------------------

def test_check_data_passivity():
    freq = np.linspace(1e9, 5e9, 11)
    passive = np.zeros((2, 2, 11), dtype=complex)
    passive[0, 0] = 0.5
    passive[1, 1] = 0.5
    assert rfm.check_data_passivity(freq, passive)

    active = np.full(11, -0.5 + 0.1j)
    assert not rfm.check_data_passivity(freq, active)


def test_enforce_data_passivity():
    freq = np.linspace(1e9, 5e9, 7)
    response = np.full(7, -0.5 + 0.1j)
    out_freq, out = rfm.enforce_data_passivity(freq, response)
    assert out.shape == response.shape
    np.testing.assert_allclose(out_freq, freq)
    assert np.all(out.real >= -1e-8)
    assert rfm.check_data_passivity(freq, out)


def test_check_hamiltonian_passive():
    model = rfm.Macromodel()
    model.Nports = 1
    model.poles = np.array([-1e9 + 0j])
    model.residues = np.array([0.5e9 + 0j])
    model.dc_gain = 0.5 + 0j

    freq = np.linspace(1e8, 1e10, 21)
    is_passive, returned = rfm.check_hamiltonian(model, freq)
    assert is_passive
    assert returned is model


def test_check_hamiltonian_passivates_unstable_model():
    model = rfm.Macromodel()
    model.Nports = 1
    model.poles = np.array([1e9 + 0j])
    model.residues = np.array([1e9 + 0j])
    model.dc_gain = 0.2 + 0j

    freq = np.linspace(1e8, 1e10, 21)
    is_passive, returned = rfm.check_hamiltonian(model, freq)
    assert not is_passive
    assert returned is not model
    assert np.all(returned.poles.real < 0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def test_show_dum(capsys):
    rfm.show_dum(rfm.read_touchstone(SAMPLE))
    output = capsys.readouterr().out
    assert "Number of ports: 2" in output
    assert "Frequency points: 401" in output


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_dum_smoke():
    dum = rfm.read_touchstone(SAMPLE)
    plt.close("all")
    rfm.plot_dum(dum, parameter="S21", scale="DB", x_scale="GHZ")
    assert plt.get_fignums()
    plt.close("all")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_dum_single_port_falls_back_to_s11(capsys):
    dum = rfm.DUM()
    dum.Nports = 1
    dum.f = np.linspace(1e9, 2e9, 5)
    dum.S = np.ones((1, 1, 5), dtype=complex) * 0.5

    plt.close("all")
    rfm.plot_dum(dum)
    assert "out of range" in capsys.readouterr().out
    plt.close("all")

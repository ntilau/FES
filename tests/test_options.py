"""Option parser tests: legacy spelling plus documented extensions."""

import pytest

from fes.core.options import (
    UsageError,
    cpp_stod,
    cpp_stoull,
    normalize_assembly,
    normalize_solver,
    parse_argv,
    parse_legacy,
)


def test_legacy_rejects_missing_values():
    with pytest.raises(UsageError):
        parse_legacy(["WR90", "+f"])
    with pytest.raises(UsageError):
        parse_legacy(["WR90", "+formula"])
    with pytest.raises(UsageError):
        parse_legacy(["WR90", "+volt", "Elec"])


def test_improved_rejects_missing_values():
    with pytest.raises(UsageError):
        parse_argv(["WR90", "+f"])
    with pytest.raises(UsageError):
        parse_argv(["WR90", "+gmres"])
    with pytest.raises(UsageError):
        parse_argv(["WR90", "+nl", "3", "Kerr"])
    with pytest.raises(UsageError):
        parse_argv(["WR90", "+rad", "10"])
    with pytest.raises(UsageError):
        parse_argv(["WR90", "+einc", "1,0,0"])


def test_frequency_sweep():
    options = parse_argv(["WR90", "+fr", "1e9", "3e9", "5"])
    assert options.l_freq == 1e9
    assert options.h_freq == 3e9
    assert options.n_freqs == 5
    freqs = options.freqs()
    assert freqs == [1e9, 1.5e9, 2e9, 2.5e9, 3e9]


def test_frequency_sweep_follows_the_reference_linspace():
    # ``eq_sys`` only tests ``n_freqs < 2``: a non-increasing range is still a
    # sweep of ``n_freqs`` points, not a single frequency.
    options = parse_argv(["WR90", "+l_freq", "3e9", "+h_freq", "1e9", "+n_freqs", "3"])
    assert options.freqs() == [3e9, 2e9, 1e9]
    # Below two frequencies the range is ignored and ``freq`` is repeated.
    options = parse_argv(["WR90", "+f", "5e9", "+l_freq", "1e9", "+h_freq", "3e9"])
    assert options.freqs() == [5e9]


def test_gmres_with_restart():
    options = parse_argv(["WR90", "+gmres", "1e-8", "30"])
    assert options.solver == "gmres"
    assert options.toll == 1e-8
    assert options.gmres_restart == 30


def test_nonlinear_parameters():
    options = parse_argv(["WR90", "+nl", "3", "Kerr", "1e-10", "0.5"])
    assert options.nl is True
    assert options.n_harm == 3
    assert options.nl_mtrl_name == "Kerr"
    assert options.kerr == 1e-10
    assert options.relax == 0.5


def test_radiation_parameters():
    options = parse_argv(["WR90", "+rad", "10", "20"])
    assert options.rad is True
    assert options.n_theta == 10
    assert options.n_phi == 20


def test_domain_decomposition():
    options = parse_argv(["WR90", "+dd", "4"])
    assert options.dd is True
    assert options.dds is True
    assert options.n_dd == 4


def test_incident_wave_comma_and_token_forms():
    comma = parse_argv(["WR90", "+einc", "1,0,0,0,0,1"])
    tokens = parse_argv(["WR90", "+einc", "1", "0", "0", "0", "0", "1"])
    assert comma.einc and tokens.einc
    assert comma.E == [1.0, 0.0, 0.0]
    assert comma.k == [0.0, 0.0, 1.0]
    assert tokens.E == comma.E and tokens.k == comma.k


def test_poly_command():
    options = parse_argv(["WR90", "+poly", "AafeeQ", "+p", "2"])
    assert options.poly is True
    assert options.poly_cmd == "AafeeQ"
    assert options.p_ord == 2


def test_output_negation():
    options = parse_argv(["WR90", "-sparam", "-tfe", "-field"])
    assert options.sparam is False
    assert options.tfe is False
    assert options.field is False


def test_power_and_href_aliases():
    options = parse_argv(["WR90", "+pow", "2.5", "+href", "q__a__Y"])
    assert options.power == 2.5
    assert options.href_cmd == "q__a__Y"


def test_voltage_repeatable():
    options = parse_argv(["CapSense", "+volt", "Elec", "1", "+volt", "Gnd", "0"])
    assert options.Vbnd == {"Elec": 1.0, "Gnd": 0.0}


def test_unknown_flags_are_preserved():
    options = parse_argv(["WR90", "+custom", "value", "+mystery"])
    assert options.extra["custom"] == "value"
    assert options.extra["mystery"] == "1"


def test_negative_numeric_value_is_a_value():
    options = parse_argv(["WR90", "+Ex", "-1"])
    assert options.einc is True
    assert options.E[0] == -1.0


def test_positional_frequency_warns_and_is_honored():
    with pytest.warns(DeprecationWarning):
        options = parse_argv(["WR90", "1e10", "+p", "2"])
    assert options.freq == 1e10
    assert options.positional_freqs == [1e10]


def test_explicit_formula_disables_file_formula():
    options = parse_argv(["BilatFilter", "+formula", "em_e_fd"])
    assert options.formula_explicit is True
    assert options.assembly == "em_e_fd"
    legacy = parse_legacy(["BilatFilter", "+formula", "em_e_fd"])
    assert legacy.formula_explicit is True


def test_matlab_flag():
    options = parse_argv(["WR90", "+matlab"])
    assert options.matlab is True
    assert options.sparam is False
    assert options.matlab_dir is None

    options = parse_argv(["WR90", "+matlab", "@../FESOld/"])
    assert options.matlab is True
    assert options.sparam is False
    assert options.matlab_dir == "../FESOld/"

    options = parse_argv(["WR90", "+matlab", "/tmp/mats", "+p", "3"])
    assert options.matlab_dir == "/tmp/mats"
    assert options.p_ord == 3

    legacy = parse_legacy(["WR90", "+matlab"])
    assert legacy.matlab is True
    assert legacy.sparam is False

    legacy = parse_legacy(["WR90", "+matlab", "@../FESOld/"])
    assert legacy.matlab is True
    assert legacy.matlab_dir == "../FESOld/"


def test_normalize_helpers():
    assert normalize_assembly("EM_EZ_FD") == "em_ez_fd"
    assert normalize_assembly("em_e_fd_schur") == "em_e_fd_dd"
    assert normalize_assembly("nonsense") == "em_e_fd"
    assert normalize_solver("GMRES") == "gmres"
    assert normalize_solver("bogus") == "direct"


def test_cpp_numeric_conversions():
    assert cpp_stod("1e10junk") == 1e10
    assert cpp_stod("abc") == 0.0
    assert cpp_stod(" -2.5") == -2.5
    assert cpp_stoull("3.9") == 3
    assert cpp_stoull("abc") == 0

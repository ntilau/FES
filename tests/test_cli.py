"""End-to-end tests for the command-line entry point."""

import json

import pytest

from fes.cli import (
    EXIT_IO,
    EXIT_NOT_PORTED,
    EXIT_OK,
    EXIT_USAGE,
    main,
)


def test_version(capsys):
    assert main(["version"]) == EXIT_OK
    assert capsys.readouterr().out.strip().startswith("fes ")


def test_version_json(capsys):
    assert main(["version", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"]
    assert payload["python"]


def test_help(capsys):
    assert main(["help"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "fes run <project>" in text
    assert "+gmres" in text


def test_models(capsys):
    assert main(["models"]) == EXIT_OK
    names = capsys.readouterr().out.split()
    assert "WR90" in names
    assert "BilatFilter" in names


def test_models_json(capsys):
    assert main(["models", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["data_dir"].endswith("data")
    names = {row["name"] for row in payload["models"]}
    assert "WR90" in names


def test_info_json(capsys):
    assert main(["info", "WR90", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dimension"] == 3
    assert payload["formula"] == "em_e_fd"
    assert [b["type"] for b in payload["boundaries"]] == [
        "PerfectE",
        "WavePort",
        "WavePort",
    ]


def test_info_text(capsys):
    assert main(["info", "BilatFilter"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "BilatFilter" in text
    assert "em_ez_fd" in text
    assert "vacuum2" in text


def test_info_missing_model(capsys):
    assert main(["info", "_missing_"]) == EXIT_IO
    assert "not found" in capsys.readouterr().err


def test_info_requires_name(capsys):
    assert main(["info"]) == EXIT_USAGE
    assert "requires a model name" in capsys.readouterr().err


def test_run_dry_run(capsys):
    assert main(["run", "WR90", "+f", "1e10", "+p", "2", "--dry-run"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "assembly:   em_e_fd" in text
    assert "frequency:  1e+10 Hz" in text
    assert "status:     dry run" in text


def test_run_json_implies_dry_run(capsys):
    assert main(["run", "WR90", "+f", "1e10", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"] == "WR90"
    assert payload["assembly"] == "em_e_fd"


def test_legacy_positional_frequency(capsys):
    with pytest.warns(DeprecationWarning):
        assert main(["WR90", "1e10", "+p", "2", "--dry-run"]) == EXIT_OK
    assert "frequency:  1e+10 Hz" in capsys.readouterr().out


def test_run_without_driver_reports_not_ported(capsys):
    # Electrostatics has a driver since Phase 11; domain decomposition
    # (Phase 9) and harmonic balance (Phase 10) remain unported.
    assert main(["run", "WR90", "+f", "1e10", "+formula", "em_e_fd_dd"]) == EXIT_NOT_PORTED
    assert "not ported yet" in capsys.readouterr().err


def test_em_h_qs_rejected(capsys):
    # The formulation is unavailable in the v0.1 solver.
    assert main(["run", "WR90", "+f", "1e10", "+em_h_qs"]) == EXIT_USAGE
    assert "Formulation not available" in capsys.readouterr().err


def test_sweep_dry_run(capsys):
    assert main(["sweep", "WR90", "+fr", "8e9", "12e9", "5", "--dry-run"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "frequencies: 5 points" in text


def test_sweep_requires_range(capsys):
    assert main(["sweep", "WR90", "+f", "1e10", "--dry-run"]) == EXIT_USAGE
    assert "+fr" in capsys.readouterr().err


def test_eigen_forces_eigenmode_formula(capsys):
    assert main(["eigen", "WR90", "+f", "1e10", "--dry-run"]) == EXIT_OK
    assert "assembly:   em_e_tl_eig  (CLI)" in capsys.readouterr().out


def test_mesh_dry_run_for_electrostatic_model(capsys):
    assert main(["mesh", "CapSense", "--dry-run"]) == EXIT_OK
    assert "command:    mesh" in capsys.readouterr().out


def test_missing_project(capsys):
    assert main(["run", "+f", "1e10"]) == EXIT_USAGE
    assert "requires a project name" in capsys.readouterr().err


def test_unknown_command(capsys):
    assert main(["frobnicate"]) == EXIT_USAGE
    assert "unknown command" in capsys.readouterr().err

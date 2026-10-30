"""Tests for :mod:`fes.drivers` — the Touchstone writer and the 2D driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fes import api
from fes.core.options import parse_argv
from fes.drivers import (
    merge_touchstone,
    read_touchstone,
    write_touchstone,
)

REPO = Path(__file__).resolve().parents[1]


def test_write_matches_the_oracle_format(tmp_path: Path) -> None:
    values = [0.5 + 0.5j, -0.25 + 0.75j, -0.25 + 0.75j, 0.5 - 0.5j]
    path = write_touchstone(tmp_path / "x.s2p", "x", {1.5: values})
    text = path.read_text()
    assert text.splitlines()[:3] == [
        "! FES S-parameters",
        "! x",
        "# GHz S MA R 50",
    ]
    row = text.splitlines()[3]
    tokens = row.split()
    assert tokens[0] == "1.50000000e+00"
    assert len(tokens) == 1 + 2 * 4
    # Same padding/justification as std::setw(14) << std::left.
    assert row == (
        "1.50000000e+00 7.07106781e-01 4.50000000e+01 "
        "7.90569415e-01 1.08434949e+02 7.90569415e-01 1.08434949e+02 "
        "7.07106781e-01 -4.50000000e+01"
    )


def test_round_trip(tmp_path: Path) -> None:
    values = [0.1 + 0.2j, -0.3 + 0.4j]
    path = write_touchstone(tmp_path / "y.s1p", "y", {10.0: values})
    data = read_touchstone(path)
    assert list(data) == [10.0]
    np.testing.assert_allclose(data[10.0], values, rtol=0, atol=5e-9)


def test_merge_replaces_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "z.s1p"
    write_touchstone(path, "z", {3.0: [0.1 + 0j], 1.0: [0.2 + 0j]})
    merge_touchstone(path, "z", 2.0, [0.3 + 0j])
    data = read_touchstone(path)
    assert list(data) == [1.0, 2.0, 3.0]
    merge_touchstone(path, "z", 2.0, [0.9 + 0j])
    data = read_touchstone(path)
    assert list(data) == [1.0, 2.0, 3.0]
    assert abs(data[2.0][0] - 0.9) < 5e-9


def test_read_skips_noise_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "noisy.s1p"
    path.write_text(
        "! comment\n# GHz S MA R 50\n\n1.0 5.0e-01 9.0e+01\nnot a row\n"
    )
    data = read_touchstone(path)
    assert list(data) == [1.0]
    assert read_touchstone(tmp_path / "absent.s1p") == {}


def test_drivers_are_registered() -> None:
    api.load_drivers()
    assert "em_ez_fd" in api.DRIVERS
    assert "em_e_fd" in api.DRIVERS
    assert "em_e_tl_eig" in api.DRIVERS


@pytest.mark.triangle
def test_run_writes_touchstone(tmp_path: Path) -> None:
    options = parse_argv([str(REPO / "data" / "BilatFilter.poly"), "+f", "150e9", "+p", "2"])
    plan = api.plan_run(options=options, command="run")
    plan.output_dir = str(tmp_path)
    api.load_drivers()
    result = api.DRIVERS[plan.assembly](plan)
    path = tmp_path / "BilatFilter.s2p"
    assert path.is_file()
    data = read_touchstone(path)
    assert list(data) == [150.0]
    np.testing.assert_allclose(
        np.asarray(data[150.0]).reshape(2, 2), result.systems[0].S,
        rtol=0, atol=5e-9,
    )
    # A second run at the same frequency must not duplicate the row.
    api.DRIVERS[plan.assembly](plan)
    assert len(read_touchstone(path)) == 1


@pytest.mark.triangle
def test_matlab_mode_dumps_system_matrices(tmp_path: Path) -> None:
    """``+matlab`` writes A.mm/B.mm and skips solve, fields and S-parameters."""
    options = parse_argv(
        [
            str(REPO / "data" / "BilatFilter.poly"),
            "+f",
            "150e9",
            "+p",
            "2",
            "+matlab",
            str(tmp_path),
        ]
    )
    plan = api.plan_run(options=options, command="run")
    plan.output_dir = str(tmp_path)
    api.load_drivers()
    result = api.DRIVERS[plan.assembly](plan)
    assert (tmp_path / "A.mm").is_file()
    assert (tmp_path / "B.mm").is_file()
    assert not (tmp_path / "BilatFilter.s2p").is_file()
    assert not (tmp_path / "BilatFilter.vtk").is_file()
    assert any("A.mm" in note for note in result.notes)
    assert any("B.mm" in note for note in result.notes)

    # The dumped matrices match a normal (non-matlab) assembly.
    other = tmp_path / "solve"
    other.mkdir()
    plan2 = api.plan_run(
        options=parse_argv(
            [str(REPO / "data" / "BilatFilter.poly"), "+f", "150e9", "+p", "2"]
        ),
        command="run",
    )
    plan2.output_dir = str(other)
    result2 = api.DRIVERS[plan2.assembly](plan2)
    from scipy.io import mmread

    np.testing.assert_allclose(
        np.asarray(result2.systems[0].A.toarray()),
        np.asarray(mmread(tmp_path / "A.mm").toarray()),
    )
    np.testing.assert_allclose(
        np.asarray(result2.systems[0].B), np.asarray(mmread(tmp_path / "B.mm"))
    )


@pytest.mark.tetgen
def test_run_em_e_fd_writes_a_six_port_touchstone(tmp_path: Path) -> None:
    options = parse_argv([str(REPO / "data" / "WR90.poly"), "+f", "1e10", "+p", "2"])
    plan = api.plan_run(options=options, command="run")
    plan.output_dir = str(tmp_path)
    api.load_drivers()
    result = api.DRIVERS[plan.assembly](plan)
    assert plan.assembly == "em_e_fd"
    path = tmp_path / "WR90.s6p"
    assert path.is_file()
    data = read_touchstone(path)
    assert list(data) == [10.0]
    # The Touchstone file keeps nine significant digits per value; the 3D
    # entries reach O(1), so the round trip needs a relative bound.
    np.testing.assert_allclose(
        np.asarray(data[10.0]).reshape(6, 6), result.systems[0].S,
        rtol=1e-7, atol=5e-9,
    )


@pytest.mark.triangle
def test_run_em_e_tl_eig_prints_the_oracle_mode_table(tmp_path: Path) -> None:
    options = parse_argv(
        [str(REPO / "data" / "WR90_TL.poly"), "+f", "1e10", "+p", "2"]
    )
    plan = api.plan_run(options=options, command="run")
    plan.output_dir = str(tmp_path)
    api.load_drivers()
    result = api.DRIVERS[plan.assembly](plan)
    assert plan.assembly == "em_e_tl_eig"
    lines = [line.strip() for line in str(result).splitlines()]
    # Six significant digits and the propagating flag, as ``std::cout`` prints
    # the oracle's table.
    assert "1: beta=72.5371 alpha=0 kc=196.632 p" in lines
    assert "5: beta=0 alpha=41.7617 kc=213.705" in lines
    # The cross-section problem skips solver and post-processing, so the only
    # file written is the mesh VTK from the project constructor.
    assert [path.name for path in tmp_path.iterdir()] == ["WR90_TL.vtk"]


@pytest.mark.triangle
def test_sweep_covers_every_frequency(tmp_path: Path) -> None:
    options = parse_argv(
        [str(REPO / "data" / "BilatFilter.poly"),
         "+fr", "140e9", "160e9", "3", "+p", "1"]
    )
    plan = api.plan_run(options=options, command="sweep")
    plan.output_dir = str(tmp_path)
    api.load_drivers()
    result = api.DRIVERS[plan.assembly](plan)
    assert len(result.systems) == 3
    data = read_touchstone(tmp_path / "BilatFilter.s2p")
    assert list(data) == [140.0, 150.0, 160.0]

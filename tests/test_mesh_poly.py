"""Tests for ``.poly`` reading and the meshed-array conversion.

These pin the pieces behind ``read_poly``: the legacy array layout
(:mod:`fes.mesh.topology`), the cacheless meshing path, and the CLI/API
surface that writes a 2D mesh cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fes.mesh.geometry import read_poly_plc
from fes.mesh.io_mat import read_cache
from fes.mesh.io_poly import read_poly
from fes.mesh.topology import (
    build_ele2edge,
    derive_edges,
    legacy_arrays,
    mesh_arrays_from_poly,
)
from fes.mesh.triangle_backend import triangulate_plc

PY_DIR = Path(__file__).resolve().parents[1]
REPO = PY_DIR


# ── legacy array layout (replaces py/iormesh) ───────────────────────────────


def test_build_ele2edge_matches_the_documented_sign_convention() -> None:
    """``spig`` is signed and 1-based; the middle column is negative."""
    elements = np.array([[0, 1, 2]])
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    spig = build_ele2edge(elements, edges)
    assert spig.tolist() == [[3, -2, 1]]


def test_build_ele2edge_zeroes_missing_edges() -> None:
    spig = build_ele2edge(np.array([[0, 1, 2]]), np.array([[0, 1]]))
    assert spig.tolist() == [[0, 0, 1]]


def test_derive_edges_is_unique_and_first_appearance() -> None:
    edges = derive_edges(np.array([[0, 1, 2], [1, 2, 3]]))
    assert edges.tolist() == [[0, 1], [1, 2], [0, 2], [2, 3], [1, 3]]


def test_build_ele2edge_from_first_appearance_edges() -> None:
    triangles = np.array([[0, 1, 2], [1, 2, 3]])
    edges = derive_edges(triangles)
    spig = build_ele2edge(triangles, edges)
    for row, (n0, n1, n2) in zip(spig, triangles):
        for column, (a, b) in enumerate(((n1, n2), (n0, n2), (n0, n1))):
            edge_id = abs(int(row[column])) - 1
            assert row[column] != 0
            assert sorted(edges[edge_id].tolist()) == sorted([a, b])


@pytest.mark.triangle
def test_legacy_arrays_are_internally_consistent() -> None:
    path = REPO / "data" / "TwoPosts.poly"
    if not path.is_file():
        pytest.skip("data/TwoPosts.poly is not available")
    arrays = mesh_arrays_from_poly(path)

    assert {name: value.shape for name, value in arrays.items()}["ele"][1] == 3
    assert np.all(arrays["ele"][:, 0] < arrays["ele"][:, 1])
    assert np.all(arrays["ele"][:, 1] < arrays["ele"][:, 2])
    assert np.all(arrays["spig2"][:, 0] < arrays["spig2"][:, 1])
    assert arrays["node"].shape[0] == arrays["nlab"].shape[0]
    assert arrays["ele"].shape[0] == arrays["elab"].shape[0]
    assert arrays["spig2"].shape[0] == arrays["slab"].shape[0]

    # Every element corner references a real edge that contains it.
    n_edges = arrays["spig2"].shape[0]
    for element, row in zip(arrays["ele"], arrays["spig"]):
        assert np.all(row != 0)
        for column, edge_id in enumerate(np.abs(row)):
            assert 1 <= edge_id <= n_edges
            endpoints = arrays["spig2"][edge_id - 1]
            assert set(endpoints.tolist()) <= set(element.tolist())

    # Planar triangulation: V - E + F = 1 (one boundary component).
    assert arrays["node"].shape[0] - n_edges + arrays["ele"].shape[0] == 1


@pytest.mark.triangle
def test_legacy_arrays_without_an_edge_list() -> None:
    triangulation = triangulate_plc(read_poly_plc(REPO / "data" / "ModelHeatEquation.poly"))
    with_edges = legacy_arrays(triangulation)
    triangulation.edges = None
    triangulation.edge_markers = None
    derived = legacy_arrays(triangulation)
    np.testing.assert_array_equal(with_edges["ele"], derived["ele"])
    np.testing.assert_array_equal(derived["slab"], np.zeros_like(derived["slab"]))
    assert derived["spig2"].shape[0] >= with_edges["spig2"].shape[0]


# ── cacheless reading ───────────────────────────────────────────────────────


@pytest.mark.triangle
def test_read_poly_meshes_when_no_cache_exists(tmp_path: Path) -> None:
    import shutil

    source = REPO / "data" / "ModelHeatEquation.poly"
    shutil.copy(source, tmp_path / source.name)

    mesh = read_poly("ModelHeatEquation", data_dir=tmp_path)
    assert (mesh["NNODE"], mesh["NELE"], mesh["NSPIG"]) == (4, 2, 5)
    assert set(mesh) == {
        "node", "ele", "spig", "spig2", "nlab", "elab", "slab",
        "NNODE", "NELE", "NSPIG",
    }
    # No cache is written unless asked for.
    assert not list(tmp_path.glob("*.npz"))


@pytest.mark.triangle
def test_read_poly_scales_node_coordinates(tmp_path: Path) -> None:
    import shutil

    source = REPO / "data" / "ModelHeatEquation.poly"
    shutil.copy(source, tmp_path / source.name)

    plain = read_poly("ModelHeatEquation", data_dir=tmp_path)
    scaled = read_poly("ModelHeatEquation", data_dir=tmp_path, scale=2.0)
    np.testing.assert_allclose(scaled["node"], 2.0 * plain["node"])


@pytest.mark.triangle
def test_read_poly_persist_writes_a_reusable_cache(tmp_path: Path) -> None:
    import shutil

    source = REPO / "data" / "ModelHeatEquation.poly"
    shutil.copy(source, tmp_path / source.name)

    generated = read_poly("ModelHeatEquation", data_dir=tmp_path, persist=True)
    assert (tmp_path / "ModelHeatEquation.h1.mat.npz").is_file()

    # The cache serves the second call; ``scale`` still applies to it.
    cached = read_poly("ModelHeatEquation", data_dir=tmp_path, scale=2.0)
    np.testing.assert_allclose(cached["node"], 2.0 * generated["node"])

    # ``remesh`` ignores the cache and regenerates from the .poly.
    remeshed = read_poly("ModelHeatEquation", data_dir=tmp_path, remesh=True)
    np.testing.assert_allclose(remeshed["node"], generated["node"])


@pytest.mark.triangle
def test_read_poly_reports_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_poly("NoSuchModel", data_dir=tmp_path)


# ── the CLI and API surfaces ────────────────────────────────────────────────


@pytest.mark.triangle
def test_api_mesh_writes_a_loadable_cache(tmp_path: Path) -> None:
    from fes import api

    result = api.mesh("ModelHeatEquation", out_dir=tmp_path)
    assert result["n_nodes"] == 4
    assert result["n_faces"] == 2
    assert result["n_edges"] == 5
    assert result["switches"] == "pq34AfeQ"
    assert Path(result["cache"]).is_file()

    arrays = read_cache(tmp_path, "ModelHeatEquation", 1)
    assert arrays["node"].shape == (4, 2)
    assert arrays["ele"].shape == (2, 3)
    assert arrays["spig2"].shape == (5, 2)


@pytest.mark.triangle
def test_api_mesh_can_skip_writing(tmp_path: Path) -> None:
    from fes import api

    result = api.mesh("ModelHeatEquation", out_dir=tmp_path, write=False)
    assert result["cache"] is None
    assert not list(tmp_path.iterdir())


@pytest.mark.triangle
def test_cli_mesh_reports_counts_and_writes(tmp_path: Path) -> None:
    from fes.cli import main

    assert main(["mesh", "ModelHeatEquation", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "ModelHeatEquation.h1.mat.npz").is_file()


@pytest.mark.triangle
def test_cli_mesh_dry_run_prints_the_plan(tmp_path: Path, capsys) -> None:
    from fes.cli import main

    assert main(["mesh", "ModelHeatEquation", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "command:    mesh" in output
    assert "status:     dry run" in output

"""Bit-level parity with the reference C++ solver.

The fixture ``tests/fixtures/cpp_oracle.json`` is produced by
``tools/parity/build_and_dump.py``, which compiles the real cFES
``Shape.cpp`` / ``Quad.cpp`` / ``EleMat.cpp`` / ``DoF.cpp`` and dumps every
array the Python reference path has to reproduce.  Regenerate it with::

    python tools/parity/build_and_dump.py --cfes ../cFES

The comparisons are tight (``atol=1e-10``): a swapped basis column or a shifted
DOF number differs by O(1), so this catches both permutations and sign errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fes.core.dof import (
    face_dof_map,
    scalar_dof_count,
    scalar_dof_map,
    vector_dof_count,
    vector_dof_map,
)
from fes.core.element_matrix import (
    Material,
    einc_face_vector,
    element_matrices_2d_mixed,
    element_matrices_3d,
    element_matrices_face,
)
from fes.core.quadrature import quadrature_2d, quadrature_3d
from fes.core.shape import shape_2d
from fes.core.shape3d import shape_3d
from fes.mesh.mesh2d import Mesh2D
from fes.mesh.mesh3d import Mesh3D

FIXTURE = Path(__file__).parent / "fixtures" / "cpp_oracle.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason="cpp_oracle.json missing; run tools/parity/build_and_dump.py",
)

ORDERS = (1, 2, 3, 4)
ATOL = 1e-10

#: Reference gradients of the barycentric coordinates, ``(dim, n_nodes)``,
#: exactly the matrices ``Jacobian::Jacobian`` hard-codes.
_DNS_REF_2D = np.array(
    [
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ]
)
_DNS_REF_3D = np.array(
    [
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads(FIXTURE.read_text())


def _real(entry: dict) -> np.ndarray:
    return np.asarray(entry["data"], dtype=float).reshape(entry["rows"], entry["cols"])


def _complex(entry: dict) -> np.ndarray:
    pairs = np.asarray(entry["data"], dtype=float)
    pairs = pairs.reshape(entry["rows"], entry["cols"], 2)
    return pairs[..., 0] + 1j * pairs[..., 1]


def _inv_jacobian(dns_ref: np.ndarray, geo: np.ndarray) -> np.ndarray:
    return np.linalg.inv(dns_ref @ geo)


def _assert_allclose(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert np.allclose(actual, expected, atol=ATOL, rtol=0.0), (
        f"{label}: max |diff| = {np.max(np.abs(actual - expected))}"
    )


@pytest.mark.parametrize("rule", [1, 2, 3, 4, 5])
def test_quadrature_2d_matches_the_reference(oracle: dict, rule: int) -> None:
    points, weights = quadrature_2d(rule)
    _assert_allclose(points, _real(oracle["quad2d"][str(rule)]["xq"]), "xq2")
    _assert_allclose(weights, _real(oracle["quad2d"][str(rule)]["wq"]).ravel(), "wq2")


@pytest.mark.parametrize("rule", [1, 2, 3, 4, 5])
def test_quadrature_3d_matches_the_reference(oracle: dict, rule: int) -> None:
    points, weights = quadrature_3d(rule)
    _assert_allclose(points, _real(oracle["quad3d"][str(rule)]["xq"]), "xq3")
    _assert_allclose(weights, _real(oracle["quad3d"][str(rule)]["wq"]).ravel(), "wq3")


@pytest.mark.parametrize("geometry", ["tri_ref", "tri_skew"])
@pytest.mark.parametrize("order", ORDERS)
def test_shape_2d_matches_the_reference(oracle: dict, geometry: str, order: int) -> None:
    geo = np.asarray(oracle["geo"][geometry], dtype=float)
    inv_j = _inv_jacobian(_DNS_REF_2D, geo)
    points, _ = quadrature_2d(order + 1)
    basis = shape_2d(order, points, inv_j)
    ref = oracle["shape2d"][geometry][str(order)]
    _assert_allclose(basis.Ns, _real(ref["Ns"]), "Ns")
    _assert_allclose(basis.dNs[:, :, 0], _real(ref["dNs_0"]), "dNs_0")
    _assert_allclose(basis.dNs[:, :, 1], _real(ref["dNs_1"]), "dNs_1")
    _assert_allclose(basis.Nv[:, :, 0], _real(ref["Nv_0"]), "Nv_0")
    _assert_allclose(basis.Nv[:, :, 1], _real(ref["Nv_1"]), "Nv_1")
    _assert_allclose(basis.dNv, _real(ref["dNv"]), "dNv")


@pytest.mark.parametrize("geometry", ["tet_ref", "tet_skew"])
@pytest.mark.parametrize("order", ORDERS)
def test_shape_3d_matches_the_reference(oracle: dict, geometry: str, order: int) -> None:
    geo = np.asarray(oracle["geo"][geometry], dtype=float)
    inv_j = _inv_jacobian(_DNS_REF_3D, geo)
    points, _ = quadrature_3d(order + 1)
    basis = shape_3d(order, points, inv_j)
    ref = oracle["shape3d"][geometry][str(order)]

    hgrad = ref["hgrad"]
    _assert_allclose(basis.Ns, _real(hgrad["Ns"]), "H1 Ns")
    for comp in range(3):
        _assert_allclose(
            basis.dNs[:, :, comp], _real(hgrad[f"dNs_{comp}"]), f"H1 dNs_{comp}"
        )

    hcurl = ref["hcurl"]
    for comp in range(3):
        _assert_allclose(
            basis.Nv[:, :, comp], _real(hcurl[f"Nv_{comp}"]), f"Nv_{comp}"
        )
        _assert_allclose(
            basis.dNv[:, :, comp], _real(hcurl[f"dNv_{comp}"]), f"dNv_{comp}"
        )


@pytest.fixture(scope="module")
def material(oracle: dict) -> Material:
    raw = oracle["geo"]["material"]
    return Material(
        epsr=raw["epsr"], mur=raw["mur"], sigma=raw["sigma"], tand=raw["tand"]
    )


@pytest.mark.parametrize("order", ORDERS)
def test_element_matrices_3d_match_the_reference(
    oracle: dict, material: Material, order: int
) -> None:
    geo = np.asarray(oracle["geo"]["tet_skew"], dtype=float)
    element = element_matrices_3d(order, geo, material)
    ref = oracle["elemat3d"][str(order)]
    _assert_allclose(element.S, _complex(ref["S"]), "S")
    _assert_allclose(element.T, _complex(ref["T"]), "T")
    _assert_allclose(element.Z, _complex(ref["Z"]), "Z")


@pytest.mark.parametrize("order", ORDERS)
def test_element_matrices_face_match_the_reference(oracle: dict, order: int) -> None:
    geo = np.asarray(oracle["geo"]["face_geo"], dtype=float)
    int_node = np.asarray(oracle["geo"]["face_int_node"], dtype=float).reshape(3)
    element = element_matrices_face(order, geo, int_node)
    ref = oracle["elemface"][str(order)]
    for name in ("St", "Tt", "Sz", "Tz", "G", "STt"):
        _assert_allclose(getattr(element, name), _complex(ref[name]), name)


@pytest.mark.parametrize("order", ORDERS)
def test_element_matrices_2d_mixed_match_the_reference(oracle: dict, order: int) -> None:
    geo = np.asarray(oracle["geo"]["tri_skew"], dtype=float)
    element = element_matrices_2d_mixed(order, geo)
    ref = oracle["elemat2d"][str(order)]
    for name in ("St", "Tt", "Sz", "Tz", "G"):
        _assert_allclose(getattr(element, name), _complex(ref[name]), name)


@pytest.mark.parametrize("order", ORDERS)
def test_einc_face_vector_matches_the_reference(oracle: dict, order: int) -> None:
    geo = np.asarray(oracle["geo"]["face_geo"], dtype=float)
    int_node = np.asarray(oracle["geo"]["face_int_node"], dtype=float).reshape(3)
    k_einc = np.asarray(oracle["geo"]["k_einc"], dtype=float).reshape(3)
    pol_einc = np.asarray(oracle["geo"]["pol_einc"], dtype=float).reshape(3)
    vector = einc_face_vector(order, geo, int_node, k_einc, pol_einc)
    _assert_allclose(vector.reshape(-1, 1), _complex(oracle["einc"][str(order)]["f"]), "f")


@pytest.fixture(scope="module")
def mesh2d(oracle: dict) -> Mesh2D:
    raw = oracle["geo"]["dof_mesh2d"]
    return Mesh2D(
        nodes=np.zeros((raw["n_nodes"], 3)),
        faces=np.asarray(raw["fac_nodes"], dtype=np.int64),
        face_labels=np.zeros(len(raw["fac_nodes"]), dtype=np.int64),
        edges=np.zeros((raw["n_edges"], 2), dtype=np.int64),
        edge_labels=np.zeros(raw["n_edges"], dtype=np.int64),
        face_edges=np.asarray(raw["fac_edges"], dtype=np.int64),
    )


@pytest.fixture(scope="module")
def mesh3d(oracle: dict) -> Mesh3D:
    raw = oracle["geo"]["dof_mesh3d"]
    return Mesh3D(
        nodes=np.zeros((raw["n_nodes"], 3)),
        tetras=np.asarray(raw["tet_nodes"], dtype=np.int64),
        tet_labels=np.zeros(len(raw["tet_nodes"]), dtype=np.int64),
        edges=np.zeros((raw["n_edges"], 2), dtype=np.int64),
        edge_labels=np.zeros(raw["n_edges"], dtype=np.int64),
        faces=np.asarray(raw["fac_nodes"], dtype=np.int64),
        face_labels=np.zeros(len(raw["fac_nodes"]), dtype=np.int64),
        face_edges=np.asarray(raw["fac_edges"], dtype=np.int64),
        tet_edges=np.asarray([[2, 5, 1, 0, 6, 8], [2, 5, 7, 0, 4, 3]], dtype=np.int64),
        tet_faces=np.asarray([[0, 5, 4, 6], [3, 2, 1, 6]], dtype=np.int64),
    )


@pytest.mark.parametrize("order", ORDERS)
def test_2d_face_dof_map_matches_the_reference(
    oracle: dict, mesh2d: Mesh2D, order: int
) -> None:
    for face_id, entries in oracle["dof2d"].items():
        ref = entries[str(order)]
        got = face_dof_map(mesh2d, order, int(face_id))
        assert list(got.s) == ref["s"], f"face {face_id} scalar"
        assert list(got.v) == ref["v"], f"face {face_id} vector"


@pytest.mark.parametrize("order", ORDERS)
def test_3d_tet_dof_map_matches_the_reference(
    oracle: dict, mesh3d: Mesh3D, order: int
) -> None:
    for tet_id, entries in oracle["dof3d"].items():
        ref = entries[str(order)]
        assert list(scalar_dof_map(mesh3d, order, int(tet_id))) == ref["s"]
        assert list(vector_dof_map(mesh3d, order, int(tet_id))) == ref["v"]


@pytest.mark.parametrize("order", ORDERS)
def test_3d_face_dof_map_matches_the_reference(
    oracle: dict, mesh3d: Mesh3D, order: int
) -> None:
    for face_id, entries in oracle["dof3dface"].items():
        ref = entries[str(order)]
        got = face_dof_map(mesh3d, order, int(face_id))
        assert list(got.s) == ref["s"], f"face {face_id} scalar"
        assert list(got.v) == ref["v"], f"face {face_id} vector"


@pytest.mark.parametrize("order", ORDERS)
def test_dof_counts_match_the_reference(
    oracle: dict, mesh2d: Mesh2D, mesh3d: Mesh3D, order: int
) -> None:
    assert scalar_dof_count(mesh2d, order) == oracle["counts2d"][str(order)]["s"]
    assert vector_dof_count(mesh2d, order) == oracle["counts2d"][str(order)]["v"]
    assert scalar_dof_count(mesh3d, order) == oracle["counts3d"][str(order)]["s"]
    assert vector_dof_count(mesh3d, order) == oracle["counts3d"][str(order)]["v"]

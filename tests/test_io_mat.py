"""Tests for the legacy ``.h<order>.mat`` mesh caches (:mod:`fes.mesh.io_mat`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from fes.mesh.io_mat import (
    MESH_KEYS,
    cache_paths,
    load_cache,
    read_cache,
    write_cache,
)
from fes.mesh.io_poly import read_poly

DATA = Path(__file__).resolve().parents[1] / "data"


def _arrays(n_nodes: int = 4, one_based: bool = False) -> dict:
    offset = 1 if one_based else 0
    return {
        "node": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "ele": np.array([[0, 1, 2], [1, 3, 2]]) + offset,
        "spig": np.array([[1, 2, 3], [-1, -2, 3]]),
        "spig2": np.array([[0, 1], [1, 3], [2, 3], [0, 2]]) + offset,
        "nlab": np.array([1, 0, 0, 2]),
        "elab": np.array([1, 1]),
        "slab": np.array([1, 0, 2, 2]),
    }


def test_npz_round_trip(tmp_path: Path) -> None:
    path = write_cache(tmp_path / "m.h1.mat.npz", **_arrays())
    loaded = load_cache(path)
    for key in MESH_KEYS:
        np.testing.assert_array_equal(loaded[key], _arrays()[key])
    assert loaded["ele"].dtype == np.int64


def test_scale_applies_to_nodes_only(tmp_path: Path) -> None:
    write_cache(tmp_path / "m.h1.mat.npz", **_arrays())
    scaled = read_cache(tmp_path, "m", scale=2.5)
    np.testing.assert_allclose(scaled["node"], _arrays()["node"] * 2.5)
    np.testing.assert_array_equal(scaled["ele"], _arrays()["ele"])


def test_matlab_cache_is_converted(tmp_path: Path) -> None:
    """MATLAB stores 1-based indices and labels as column vectors."""
    scipy.io.savemat(tmp_path / "m.h1.mat", _arrays(one_based=True))
    loaded = load_cache(tmp_path / "m.h1.mat")
    np.testing.assert_array_equal(loaded["ele"], [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(loaded["spig2"], [[0, 1], [1, 3], [2, 3], [0, 2]])
    np.testing.assert_array_equal(loaded["nlab"], [1, 0, 0, 2])
    np.testing.assert_array_equal(loaded["elab"], [1, 1])


def test_zero_based_cache_is_left_alone(tmp_path: Path) -> None:
    scipy.io.savemat(tmp_path / "m.h1.mat", _arrays(one_based=False))
    loaded = load_cache(tmp_path / "m.h1.mat")
    np.testing.assert_array_equal(loaded["ele"], _arrays()["ele"])


def test_npz_precedes_mat(tmp_path: Path) -> None:
    scipy.io.savemat(tmp_path / "m.h1.mat", _arrays())
    arrays = _arrays()
    arrays["node"] = arrays["node"] + 10.0
    write_cache(tmp_path / "m.h1.mat.npz", **arrays)
    np.testing.assert_allclose(read_cache(tmp_path, "m")["node"], arrays["node"])
    npz_path, mat_path = cache_paths(tmp_path, "m", 1)
    assert npz_path.name == "m.h1.mat.npz" and mat_path.name == "m.h1.mat"


def test_missing_cache_mentions_both_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        read_cache(tmp_path, "absent")
    assert "absent.h1.mat.npz" in str(caught.value)
    assert "absent.h1.mat" in str(caught.value)


def test_unparsable_cache_is_a_file_error(tmp_path: Path) -> None:
    (tmp_path / "m.h1.mat.npz").write_bytes(b"not a zip")
    with pytest.raises(FileNotFoundError):
        read_cache(tmp_path, "m")


def test_write_rejects_unknown_and_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        write_cache(tmp_path / "x.npz", **_arrays(), extra=np.zeros(1))
    partial = _arrays()
    partial.pop("slab")
    with pytest.raises(KeyError):
        write_cache(tmp_path / "x.npz", **partial)


def test_real_cache_loads(tmp_path: Path) -> None:
    arrays = _arrays()
    write_cache(tmp_path / "BilatFilter_py.h1.mat.npz", **arrays)
    loaded = read_cache(tmp_path, "BilatFilter_py", 1)
    n_nodes = loaded["node"].shape[0]
    assert n_nodes > 0
    assert loaded["ele"].max() < n_nodes
    assert loaded["spig2"].max() < n_nodes
    assert loaded["nlab"].shape == (n_nodes,)


def test_read_poly_uses_the_cache(tmp_path: Path) -> None:
    write_cache(tmp_path / "m.h1.mat.npz", **_arrays())
    mesh = read_poly("m", data_dir=str(tmp_path), scale=2.0)
    assert mesh["NNODE"] == 4
    assert mesh["NELE"] == 2
    assert mesh["NSPIG"] == 4
    np.testing.assert_allclose(mesh["node"][1], [2.0, 0.0])
    np.testing.assert_array_equal(mesh["ele"], [[0, 1, 2], [1, 3, 2]])

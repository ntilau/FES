"""Model path resolution and model discovery."""

from pathlib import Path

import pytest

from fes.paths import data_dir, list_models, model_path, repo_root


def test_repo_root_contains_data_and_package():
    root = repo_root()
    assert (root / "data").is_dir()
    assert (root / "fes").is_dir()


def test_data_dir_is_repo_data():
    assert data_dir() == repo_root() / "data"


def test_model_list_includes_all():
    names = list_models()
    assert "WR90" in names
    assert "BilatFilter_py" in names


def test_model_path_forms():
    root = model_path("WR90")
    assert root.name == "WR90.poly" and root.is_file()
    assert model_path("WR90.poly") == root
    assert model_path(str(root)) == root


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        model_path("_does_not_exist_")

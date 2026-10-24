"""Metadata parsing for the custom .poly sections."""

from fes.mesh.poly_header import read_poly_metadata
from fes.paths import model_path


def test_wr90_metadata():
    model = read_poly_metadata(model_path("WR90"))
    assert model.name == "WR90"
    assert model.dimension == 3
    assert model.formula == "em_e_fd"
    assert model.n_nodes == 8
    assert model.n_faces == 6
    assert model.n_holes == 0
    assert model.n_regions == 1
    assert [solid.name for solid in model.solids] == ["WG1"]
    assert model.solids[0].material == "vacuum"
    assert [boundary.name for boundary in model.boundaries] == ["PEC1", "WP1", "WP2"]
    assert [boundary.type for boundary in model.boundaries] == [
        "PerfectE",
        "WavePort",
        "WavePort",
    ]
    # ``bc::bc()`` defaults the mode count to 1; only ports spell it out here.
    assert [boundary.num_modes for boundary in model.boundaries] == [1, 3, 3]
    assert model.bbox is not None
    assert model.bbox[0][0] == -0.01143


def test_bilat_filter_metadata():
    model = read_poly_metadata(model_path("BilatFilter"))
    assert model.dimension == 2
    assert model.formula == "em_ez_fd"
    assert model.n_nodes == 28
    assert model.n_segments == 28
    assert model.n_holes == 0
    assert model.n_regions == 2
    assert [solid.epsr for solid in model.solids] == [1.0, 2.1]
    assert [solid.material for solid in model.solids] == ["vacuum", "diel"]
    assert [boundary.num_modes for boundary in model.boundaries] == [1, 1, 1]


def test_metadata_as_dict_is_json_ready():
    payload = read_poly_metadata(model_path("CapSense")).as_dict()
    assert payload["dimension"] == 2
    assert payload["formula"] == "em_e_qs"
    assert payload["solids"][0]["name"] == "cap1"
    assert isinstance(payload["boundaries"], list)

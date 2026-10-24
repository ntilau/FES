"""Tests for the bundled modal-matching solver (:mod:`mm`).

The package is bundled under the MIT license; these tests
cover the modal primitives, the port-block bookkeeping, the frequency/topology
validation helpers and the bundled benchmark scripts.
"""

from __future__ import annotations

import numpy as np
import pytest

from mm import core, lib, scripts
from mm.run import _resolve_script_runner

A = 0.02286  # WR-90 broad wall [m]
B = 0.01016  # WR-90 narrow wall [m]


def _segment(
    a: float = A,
    b: float = B,
    nmodes: int = 4,
    length: float = 0.01,
    xo: float | None = None,
    yo: float | None = None,
) -> dict:
    """Return a segment with modes, normalizations and (optionally) delays."""
    seg = {
        "a": a,
        "b": b,
        "Nmodes": nmodes,
        "l": length,
        "xo": a / 2 if xo is None else xo,
        "yo": b / 2 if yo is None else yo,
    }
    seg, _ = core.OrderModes(seg, {"x": 0, "y": 0, "H": 0, "E": 0})
    core.EigenModes(seg)
    core.NormCoeff(seg)
    return seg


def _propagating_segment(k0: float = 2 * np.pi / 0.03) -> dict:
    seg = _segment()
    core.WaveNumbers(seg, k0)
    core.DelayMatrix(seg, k0)
    return seg


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_package_surface() -> None:
    assert lib.Cascade.__name__ == "Cascade"
    assert callable(core.MultiPortDevice)
    assert callable(scripts.Riblet)
    assert {"core", "lib", "scripts"} <= set(dir(__import__("mm")))


def test_lib_wrappers_delegate_to_core() -> None:
    kx, ky = lib.OneModeEigens(A, B, 2, 1)
    assert (kx, ky) == core.OneModeEigens(A, B, 2, 1)
    assert lib.ExtractPortS.__name__ == "ExtractPortS"


# ---------------------------------------------------------------------------
# Modal primitives
# ---------------------------------------------------------------------------


def test_one_mode_eigens() -> None:
    assert core.OneModeEigens(A, B, 1, 0) == (np.pi / A, 0.0)
    assert core.OneModeEigens(A, B, 0, 1) == (0.0, np.pi / B)
    kx, ky = core.OneModeEigens(A, B, 2, 3)
    assert kx == 2 * np.pi / A
    assert ky == 3 * np.pi / B


def test_one_mode_norm_coeff() -> None:
    kx, ky = core.OneModeEigens(A, B, 1, 0)
    norm = core.OneModeNormCoeff(A, B, 1, 0, kx, ky)
    # n == 0 doubles the kx contribution (the sqrt(1 + delta) factors).
    expected = 2.0 / (np.sqrt(A * B) * np.sqrt(2 * kx**2))
    assert np.isclose(norm, expected)
    assert norm > 0

    kx2, ky2 = core.OneModeEigens(A, B, 1, 1)
    assert np.isclose(
        core.OneModeNormCoeff(A, B, 1, 1, kx2, ky2),
        2.0 / (np.sqrt(A * B) * np.sqrt(kx2**2 + ky2**2)),
    )


def test_order_modes_selects_lowest_cutoff_modes() -> None:
    seg, info = core.OrderModes(
        {"a": A, "b": B, "Nmodes": 4}, {"x": 0, "y": 0, "H": 0, "E": 0}
    )
    assert info["info"] == ["modes used: 4"]
    # TE10, TE20 and TE11 are the three lowest cutoffs; TE01 is the E mode.
    assert seg["mh"] == [1, 2, 1]
    assert seg["nh"] == [0, 0, 1]
    assert seg["me"] == [1]
    assert seg["ne"] == [1]
    assert seg["Nh"] == 3 and seg["Ne"] == 1

    cutoffs = [
        (m * np.pi / A) ** 2 + (n * np.pi / B) ** 2
        for m, n in zip(seg["mh"], seg["nh"])
    ]
    assert cutoffs == sorted(cutoffs)


@pytest.mark.parametrize("symmetry", [{"x": 1}, {"y": 1}, {"H": 1}, {"E": 1}])
def test_order_modes_symmetry_reduces_mode_set(symmetry: dict) -> None:
    full = {"x": 0, "y": 0, "H": 0, "E": 0}
    full.update(symmetry)
    seg, _ = core.OrderModes({"a": A, "b": B, "Nmodes": 4}, full)

    assert seg["Nh"] <= 4
    # Every E mode is also a propagating H candidate.
    for m, n in zip(seg["me"], seg["ne"]):
        assert m > 0 and n > 0
        assert any(
            m == mh and n == nh for mh, nh in zip(seg["mh"], seg["nh"])
        )
    if symmetry == {"E": 1}:
        # E symmetry only keeps m-varying modes.
        assert seg["Ne"] == 0
        assert seg["Nh"] == 4


def test_eigen_modes_and_norm_coeff() -> None:
    seg = _segment()
    assert seg["kh"]["x"].shape == (seg["Nh"],)
    assert seg["kh"]["y"].shape == (seg["Nh"],)
    assert seg["ke"]["x"].shape == (seg["Ne"],)
    assert np.all(seg["Ah"] > 0)
    assert np.all(seg["Ae"] > 0)
    assert np.isclose(seg["kh"]["x"][0], np.pi / A)
    assert seg["kh"]["y"][0] == 0.0


def test_wave_numbers_and_delay_matrix() -> None:
    k0 = 2 * np.pi / 0.03  # above the TE10 cutoff
    seg = _propagating_segment(k0)

    mn = seg["kh"]["mn"]
    # TE10 propagates, the next two modes are below cutoff.
    assert np.isclose(mn[0].imag, 0.0)
    assert mn[0].real > 0
    assert np.all(mn[1:].imag < 0)

    D = seg["D"]
    assert D.shape == (seg["Nh"] + seg["Ne"],) * 2
    assert np.all(np.abs(np.diag(D)) <= 1 + 1e-12)
    # Propagating modes only accumulate phase.
    assert np.isclose(abs(D[0, 0]), 1.0)
    # Off-diagonals are zero.
    assert np.allclose(D - np.diag(np.diag(D)), 0)


def test_integrals_orthogonality() -> None:
    step = [{"a": A, "b": B, "xo": A / 2, "yo": B / 2}] * 2
    kx, ky = core.OneModeEigens(A, B, 1, 0)
    iint = core.Integrals(kx, ky, kx, ky, step)
    assert np.allclose(iint[:3], [B / 2, B / 2, A / 2])
    assert np.isclose(iint[3], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Single-step scattering matrices
# ---------------------------------------------------------------------------


def test_single_step_shapes() -> None:
    seg_a = _segment()
    seg_b = _segment(a=2 * A + 1e-4, b=B, length=0.005, xo=A, yo=B / 2)
    k0 = 2 * np.pi / 0.03
    for seg in (seg_a, seg_b):
        core.WaveNumbers(seg, k0)
        core.DelayMatrix(seg, k0)

    step = core.SingleStep([seg_a, seg_b], 0, k0)
    ndof = seg_a["Nh"] + seg_a["Ne"]
    assert set(step) == {"S11", "S12", "S21", "S22"}
    for block in step.values():
        assert block.shape == (ndof, ndof)
        assert np.isfinite(block).all()


def test_mxx_and_u_matrices() -> None:
    seg_a = _segment()
    seg_b = _segment(a=2 * A + 1e-4, b=B)
    mhh, mhe, meh, mee = core.MxxMatrices([seg_a, seg_b])
    assert mhh.shape == (seg_a["Nh"], seg_b["Nh"])
    assert mhe.shape == (seg_a["Nh"], seg_b["Ne"])
    assert meh.shape == (seg_a["Ne"], seg_b["Nh"])
    assert mee.shape == (seg_a["Ne"], seg_b["Ne"])
    assert np.isfinite(mhh).all()

    uh1, uh2, ue1, ue2 = core.UMatrices([seg_a, seg_b])
    assert np.array_equal(uh1, np.eye(seg_a["Nh"]))
    assert np.array_equal(uh2, np.eye(seg_b["Nh"]))
    assert np.array_equal(ue1, np.eye(seg_a["Ne"]))
    assert np.array_equal(ue2, np.eye(seg_b["Ne"]))


# ---------------------------------------------------------------------------
# Port block bookkeeping
# ---------------------------------------------------------------------------


def test_extract_insert_port_round_trip() -> None:
    rng = np.random.default_rng(0)
    stot = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    dim = (3, 3)

    block = core.ExtractPortS(stot, dim, 2, 3)
    assert np.array_equal(block, stot[3:6, 6:9])

    out = core.InsertPortS(stot.copy(), block, dim, 2, 3)
    assert np.array_equal(out, stot)

    # Building from an empty matrix grows it to fit the largest port index.
    built = core.InsertPortS(np.zeros((0, 0), complex), block, dim, 2, 3)
    assert built.shape == (9, 9)
    assert np.array_equal(core.ExtractPortS(built, dim, 2, 3), block)
    assert np.count_nonzero(built) == np.count_nonzero(block)


def test_reverse_structure_is_involution() -> None:
    wgs = [_segment(length=i + 1) for i in range(3)]
    reversed_ = core.ReverseWaveGuideStructure(wgs)
    assert [w["l"] for w in reversed_] == [3, 2, 1]
    assert core.ReverseWaveGuideStructure(reversed_) == wgs


def test_two_port_port_segments() -> None:
    wgs = [_segment(length=i + 1) for i in range(3)]
    port1, other1, all1 = core.TwoPortDeviceGetPortSegment(wgs, 1)
    assert (port1, other1, all1) == (wgs[0], wgs[-1], wgs)
    port2, other2, _ = core.TwoPortDeviceGetPortSegment(wgs, 2)
    assert (port2, other2) == (wgs[-1], wgs[0])
    with pytest.raises(ValueError):
        core.TwoPortDeviceGetPortSegment(wgs, 3)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_frequency_sweep_validate_linspace() -> None:
    fs, err = core.FrequencySweepValidate({"start": 1e9, "end": 2e9, "N": 5})
    assert err == {}
    assert fs["N"] == 5
    assert np.allclose(fs["f"], np.linspace(1e9, 2e9, 5))


def test_frequency_sweep_validate_explicit_and_invalid() -> None:
    fs, err = core.FrequencySweepValidate({"f": [1.0, 2.0, 3.0]})
    assert err == {}
    assert fs["N"] == 3
    assert np.array_equal(fs["f"], [1.0, 2.0, 3.0])

    fs, err = core.FrequencySweepValidate({"start": 1e9, "end": 2e9, "N": 1})
    assert err == {} and fs["N"] == 1
    assert np.array_equal(fs["f"], [1e9])

    _, err = core.FrequencySweepValidate({"f": [0.0, 2.0]})
    assert "fatal" in err
    _, err = core.FrequencySweepValidate({"f": [1.0, -2.0]})
    assert "fatal" in err


def test_multiport_topology_two_to_one() -> None:
    top, err = core.MultiPortDeviceTopology(
        [], [{"SideOne": [1, 2], "SideTwo": [1]}], []
    )
    assert err == {}
    nto1 = top["Nto1"][0]
    assert nto1["nFurcation"] == 0
    assert nto1["Dimensions"] == 3
    assert len(nto1["PortToCondense"]["a"]) == len(nto1["PortToCondense"]["b"])
    assert nto1["OpenPorts"] == [1, 3, 6]


def test_multiport_topology_one_to_two() -> None:
    top, err = core.MultiPortDeviceTopology(
        [], [{"SideOne": [1], "SideTwo": [1, 2]}], []
    )
    assert err == {}
    assert top["Nto1"][0]["nFurcation"] == 1


def test_multiport_topology_rejects_degenerate_connections() -> None:
    for side_one, side_two in (([1], [1]), ([1, 2], [1, 2]), ([], [1])):
        top, err = core.MultiPortDeviceTopology(
            [], [{"SideOne": side_one, "SideTwo": side_two}], []
        )
        assert top == {}
        assert "fatal" in err


def test_extract_single_s() -> None:
    sf, sinfo, err = scripts.BifurcationH(plot=False)
    assert err.get("fatal") is None
    s = np.asarray(sf[0])

    out_info = sinfo[0]
    in_info = sinfo[1]
    dim = (
        len(out_info["mh"]) + len(out_info["me"]),
        len(in_info["nh"]) + len(in_info["ne"]),
    )
    block = core.ExtractPortS(s, dim, 1, 2)

    # First H mode (m=1, n=0) of both ports.
    j = out_info["mh"].index(1)
    i = in_info["mh"].index(1)
    assert core.ExtractSingleS(s, sinfo, 1, 2, "h", 1, 0, "h", 1, 0) == block[j, i]
    # A mode that is not in the retained set yields zero.
    assert core.ExtractSingleS(s, sinfo, 1, 2, "e", 99, 99, "h", 1, 0) == 0


# ---------------------------------------------------------------------------
# Command-line entry point and benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("bifurcatione", "BifurcationE"),
        ("BifurcationE", "BifurcationE"),
        ("bifurcationh", "BifurcationH"),
        ("riblet", "Riblet"),
        ("hilderbrandhalf", "HildebrandHalf"),
        ("HildebrandSemiAuto", "HildebrandSemiAuto"),
        ("hildebrandfull", "HildebrandFull"),
    ],
)
def test_script_aliases(alias: str, expected: str) -> None:
    name, runner = _resolve_script_runner(alias)
    assert name == expected
    assert callable(runner)


def test_bifurcation_e_benchmark() -> None:
    sf, sinfo, err = scripts.BifurcationE(plot=False)
    assert err.get("fatal") is None
    assert len(sf) == 51
    assert np.asarray(sf[0]).shape == (30, 30)
    assert len(sinfo) == 3
    assert all(np.isfinite(np.asarray(s)).all() for s in sf)


def test_bifurcation_h_benchmark() -> None:
    sf, sinfo, err = scripts.BifurcationH(plot=False)
    assert err.get("fatal") is None
    assert len(sf) == 61
    assert np.asarray(sf[0]).shape == (48, 48)
    assert len(sinfo) == 3
    assert all(np.isfinite(np.asarray(s)).all() for s in sf)


def test_riblet_benchmark() -> None:
    sf, sinfo, err = scripts.Riblet(plot=False)
    assert err.get("fatal") is None
    assert len(sf) == 251
    assert np.asarray(sf[0]).shape == (36, 36)
    assert len(sinfo) == 4
    assert all(np.isfinite(np.asarray(s)).all() for s in sf)

    # Below the TE10 cutoff the coupler is essentially invisible; it becomes
    # transparent above cutoff.
    first = np.max(np.abs(np.asarray(sf[0])))
    last = np.max(np.abs(np.asarray(sf[-1])))
    assert first < 1e-3
    assert last > 0.5

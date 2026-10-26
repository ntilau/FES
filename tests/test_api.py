"""Run-plan resolution and formulation selection."""

import pytest

from fes.api import (
    DRIVERS,
    FORMULATION_PHASE,
    FormulationUnavailableError,
    effective_assembly,
    load_drivers,
    load_model,
    mesh,
    plan_run,
    run,
    sweep,
)
from fes.core.options import UsageError, parse_argv


def test_plan_uses_cli_frequency_and_order():
    plan = plan_run("WR90", argv=["+f", "1e10", "+p", "2"])
    assert plan.project == "WR90"
    assert plan.assembly == "em_e_fd"
    assert plan.frequency == 1e10
    assert plan.options.p_ord == 2
    assert plan.frequencies == [1e10]


def test_file_formula_selects_2d_solver():
    plan = plan_run("BilatFilter", argv=["+f", "150e9", "+p", "2"])
    assert plan.assembly == "em_ez_fd"
    assert plan.dimension == 2


def test_explicit_formula_overrides_file():
    plan = plan_run("BilatFilter", argv=["+formula", "em_e_fd"])
    assert plan.assembly == "em_e_fd"
    options = parse_argv(["BilatFilter", "+formula", "em_e_fd"])
    assert effective_assembly(options, load_model("BilatFilter")) == "em_e_fd"


def test_electrostatic_requires_zero_frequency():
    plan = plan_run("CapSense", argv=["+f", "0"])
    assert plan.assembly == "em_e_qs"
    with pytest.raises(UsageError):
        plan_run("CapSense", argv=["+f", "1e9"])


def test_formula_tag_forces_field_export_for_electrostatics():
    """``detectFormula`` sets ``opt->field`` for a ``#Formula em_e_qs`` tag.

    An explicit ``+formula`` skips ``detectFormula`` in the oracle, so the
    side effect must not fire there.
    """
    tagged = plan_run("CapSense", argv=["+f", "0"])
    assert tagged.assembly == "em_e_qs"
    assert tagged.options.field is True

    explicit = plan_run("CapSense", argv=["+f", "0", "+formula", "em_e_qs"])
    assert explicit.assembly == "em_e_qs"
    assert explicit.options.field is False


def test_2d_scalar_rejects_3d_model():
    with pytest.raises(UsageError):
        plan_run("WR90", argv=["+formula", "em_ez_fd"])


def test_invalid_polynomial_order():
    with pytest.raises(UsageError):
        plan_run("WR90", argv=["+p", "5"])
    with pytest.raises(UsageError):
        plan_run("WR90", argv=["+p", "0"])


def test_missing_project():
    with pytest.raises(UsageError):
        plan_run(argv=["+f", "1e10"])


def test_dry_run_returns_plan():
    plan = run("WR90", argv=["+f", "1e10"], dry_run=True)
    assert plan.command == "run"


def test_em_e_fd_nl_is_ported():
    # em_e_fd_nl (Phase 10) now has a driver; without +nl options it falls
    # back to the linear em_e_fd solve.
    result = run("WR90", argv=["+f", "1e10", "+formula", "em_e_fd_nl", "+p", "2"])
    assert result.systems[0].S is not None


def test_em_h_qs_is_unavailable_like_the_oracle():
    """``equation_system.cpp`` has no case for ``em_h_qs``.

    The oracle parses the flag and then aborts with
    ``Formulation not available for this version``, so planning still works
    (``--dry-run``) but execution must fail the same way.
    """
    plan = plan_run("WR90", argv=["+f", "1e10", "+em_h_qs"])
    assert plan.assembly == "em_h_qs"
    assert "em_h_qs" not in FORMULATION_PHASE
    with pytest.raises(FormulationUnavailableError) as caught:
        run("WR90", argv=["+f", "1e10", "+em_h_qs"])
    assert "Formulation not available" in str(caught.value)


def test_ported_formulations_have_a_driver():
    """The registry carries exactly the formulations that have a driver.

    Phases 2, 6, 7, 9, 10, 11 and 12 landed drivers.
    """
    ported = {"em_av_fd", "em_ez_fd", "em_e_fd", "em_e_fd_dd", "em_e_fd_nl", "em_e_tl_eig", "em_e_qs"}
    load_drivers()
    for name in FORMULATION_PHASE:
        if name in ported:
            assert name in DRIVERS, name
        else:
            assert name not in DRIVERS, name


def test_em_av_fd_is_ported():
    """em_av_fd (Phase 12) now has a driver; without waveports it raises
    ``Null Right Hand Side`` like em_e_fd.  WR90 carries WavePort boundaries,
    so at order 1 the port mode computation fails with ValueError instead
    ("not enough dofs on wave_ports")."""
    from fes.core.assemblers.errors import NullRightHandSideError
    with pytest.raises((NullRightHandSideError, ValueError)):
        run("WR90", argv=["+f", "1e10", "+formula", "em_av_fd", "+p", "1"])


def test_sweep_plan():
    plan = sweep("WR90", argv=["+fr", "8e9", "12e9", "5"], dry_run=True)
    assert plan.command == "sweep"
    assert len(plan.frequencies) == 5
    assert plan.frequencies[0] == 8e9
    assert plan.frequencies[-1] == 12e9


def test_sweep_requires_multiple_points():
    with pytest.raises(UsageError):
        sweep("WR90", argv=["+f", "1e10"], dry_run=True)


def test_mesh_plan_skips_frequency_validation():
    plan = plan_run("CapSense", argv=[], command="mesh")
    assert plan.command == "mesh"
    assert plan.assembly == "em_e_qs"


def test_mesh_rejects_h_refinement():
    """The cache is the unrefined h1 array set; refinement happens at run time."""
    plan = plan_run("CapSense", argv=["+h", "1"], command="mesh")
    assert plan.options.h_ord == 1  # --dry-run still resolves the plan
    with pytest.raises(UsageError, match="h1 cache"):
        mesh("CapSense", argv=["+h", "1"])


def test_plan_as_dict_is_json_ready():
    payload = plan_run("WR90", argv=["+f", "1e10"]).as_dict()
    assert payload["project"] == "WR90"
    assert payload["assembly"] == "em_e_fd"
    assert payload["options"]["freq"] == 1e10

"""Focused contract tests for canonical stationary forward execution."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fluxemu.exceptions import MappingError, ValidationError
from fluxemu.execution import CanonicalFluxState, run_stationary_forward
from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    CanonicalModelError,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    IsotopeMetabolite,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    LinearObjective,
    MappingBranch,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
)


def _science() -> tuple[CanonicalModel, StationaryExperimentSemantics]:
    flux = FluxModel(
        (FluxMetabolite("source", False), FluxMetabolite("product", False)),
        (
            FluxReaction(
                "R",
                (StoichiometricTerm("source", -1), StoichiometricTerm("product", 1)),
                0,
                10,
            ),
        ),
        LinearObjective("maximise", ()),
    )
    transition = AtomTransition(AtomPosition("source", 1), AtomPosition("product", 1))
    isotope = IsotopeModel(
        (IsotopeMetabolite("source", 1, True, False), IsotopeMetabolite("product", 1, True, False)),
        (
            IsotopeReaction(
                "R",
                "forward",
                True,
                (IsotopeParticipant("source", (1,)),),
                (IsotopeParticipant("product", (1,)),),
                (MappingBranch("primary", 1.0, (transition,)),),
            ),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("source", (("#1", 1.0),), "no"),),
        (Target("product", "product", (1,), "intermediate", "C", "no"),),
    )
    return CanonicalModel(flux, isotope), experiment


@pytest.mark.parametrize(
    ("values", "message"),
    [({}, "missing reaction"), ({"R": 1.0, "extra": 2.0}, "unknown reaction"), ({"R": 11.0}, "outside")],
)
def test_invalid_complete_flux_states_fail_before_backend_import(values, message):
    model, experiment = _science()
    with pytest.raises(MappingError, match=message):
        run_stationary_forward(model, experiment, values)


def test_duplicate_flux_ids_are_rejected():
    model, experiment = _science()
    state = CanonicalFluxState("duplicate", (("R", 1.0), ("R", 1.0)))
    with pytest.raises(MappingError, match="duplicate"):
        run_stationary_forward(model, experiment, (state,))


def test_invalid_canonical_model_fails_before_backend_import():
    model, experiment = _science()
    reaction = replace(model.flux_model.reactions[0], lower_bound=20)
    broken = replace(model, flux_model=replace(model.flux_model, reactions=(reaction,)))
    with pytest.raises(CanonicalModelError, match="lower bound exceeds"):
        run_stationary_forward(broken, experiment, {"R": 1.0})


def test_invalid_stationary_experiment_fails_before_backend_import():
    model, experiment = _science()
    broken = replace(experiment, targets=(replace(experiment.targets[0], metabolite_id="missing"),))
    with pytest.raises(CanonicalModelError, match="unknown isotope metabolite"):
        run_stationary_forward(model, broken, {"R": 1.0})


def test_incomplete_atom_map_and_malformed_weights_fail_before_execution():
    model, experiment = _science()
    reaction = model.isotope_model.reactions[0]
    incomplete = replace(reaction.mapping_branches[0], transitions=())
    broken_map = replace(
        model,
        isotope_model=replace(
            model.isotope_model,
            reactions=(replace(reaction, mapping_branches=(incomplete,)),),
        ),
    )
    with pytest.raises(CanonicalModelError, match="must contain transitions"):
        run_stationary_forward(broken_map, experiment, {"R": 1.0})

    malformed_weight = replace(reaction.mapping_branches[0], weight=0.5)
    broken_weight = replace(
        model,
        isotope_model=replace(
            model.isotope_model,
            reactions=(replace(reaction, mapping_branches=(malformed_weight,)),),
        ),
    )
    with pytest.raises(CanonicalModelError, match="must sum to one"):
        run_stationary_forward(broken_weight, experiment, {"R": 1.0})


def test_fluxemu_model_import_remains_backend_independent():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys; import fluxemu.model; "
        "forbidden={'scipy','cobra','mfapy','matplotlib'}; "
        "assert not forbidden.intersection(sys.modules), forbidden.intersection(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_result_does_not_expose_backend_objects(monkeypatch):
    model, experiment = _science()

    from fluxemu.execution import StationaryForwardResult, StationaryMID, StationaryMIDValue

    expected = StationaryForwardResult(
        (StationaryMID("state-0", "product", (0.0, 1.0)),),
        (
            StationaryMIDValue("state-0", "product", 0, 0.0),
            StationaryMIDValue("state-0", "product", 1, 1.0),
        ),
        1e-8,
        0.0,
    )
    import types

    fake = types.ModuleType("fluxemu.backends.mfapy")
    fake.execute_stationary = lambda *args: expected
    monkeypatch.setitem(sys.modules, "fluxemu.backends.mfapy", fake)
    assert run_stationary_forward(model, experiment, {"R": 1.0}) == expected


def test_backend_missing_target_is_a_fluxemu_validation_failure(monkeypatch):
    """A backend response omitting a requested MID must fail FluxEMU validation."""

    flux = FluxModel(
        (
            FluxMetabolite("source", False),
            FluxMetabolite("middle", True),
            FluxMetabolite("product", False),
        ),
        (
            FluxReaction(
                "R1",
                (StoichiometricTerm("source", -1), StoichiometricTerm("middle", 1)),
                0,
                10,
            ),
            FluxReaction(
                "R2",
                (StoichiometricTerm("middle", -1), StoichiometricTerm("product", 1)),
                0,
                10,
            ),
        ),
        LinearObjective("maximise", ()),
    )
    source_to_middle = AtomTransition(AtomPosition("source", 1), AtomPosition("middle", 1))
    middle_to_product = AtomTransition(AtomPosition("middle", 1), AtomPosition("product", 1))
    isotope = IsotopeModel(
        (
            IsotopeMetabolite("source", 1, True, False),
            IsotopeMetabolite("middle", 1, True, False),
            IsotopeMetabolite("product", 1, True, False),
        ),
        (
            IsotopeReaction(
                "R1",
                "forward",
                True,
                (IsotopeParticipant("source", (1,)),),
                (IsotopeParticipant("middle", (1,)),),
                (MappingBranch("primary", 1.0, (source_to_middle,)),),
            ),
            IsotopeReaction(
                "R2",
                "forward",
                True,
                (IsotopeParticipant("middle", (1,)),),
                (IsotopeParticipant("product", (1,)),),
                (MappingBranch("primary", 1.0, (middle_to_product,)),),
            ),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("source", (("#1", 1.0),), "no"),),
        (Target("product", "product", (1,), "intermediate", "C", "no"),),
    )
    model = CanonicalModel(flux, isotope)

    from fluxemu._mfapy import load_mfapy

    monkeypatch.setattr(load_mfapy().optimize, "calc_MDV_from_flux", lambda *args: ([], {"X_list": []}))
    with pytest.raises(ValidationError, match="missing requested target"):
        run_stationary_forward(model, experiment, {"R1": 1.0, "R2": 1.0})

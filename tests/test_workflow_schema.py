"""Strict declarations and identity for the finite hypothesis workflow."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import pytest
import yaml

from fluxemu.exceptions import InputValidationError
from fluxemu.workflow.schema import (
    HypothesisSpecification, ReactionBoundConstraint, StateGenerationSpecification,
    TestingSpecification as WorkflowTestingSpecification,
    WorkflowSpecification, load_hypothesis_testing_spec,
)


FIXTURES = Path(__file__).parent / "fixtures" / "native_portability"


def workflow_document():
    return {
        "schema_version": 1, "model": "model.xml",
        "experiments": [{
            "experiment_id": "tracer", "specification": "experiment.yaml",
            "counts": [{"target_id": "portable_mid", "replicate_id": "r0", "total_count": 2}],
        }],
        "hypotheses": {
            role: {
                "description": f"{role} pathway interval",
                "reaction_bounds": [{"reaction_id": "foreign_hx", "lower_bound": lower, "upper_bound": upper}],
                "state_generation": {"method": "hit_and_run", "count": 2, "seed": seed},
            }
            for role, lower, upper, seed in (("H0", 1, 3, 10), ("H1", 7, 9, 20))
        },
        "observations": {"semantics": "genuine_counts", "independent_blocks": False},
        "testing": {"epsilon": 0.1, "procedures": ["exact_minimax"]},
        "output": {"directory": "results"},
    }


def write_workflow(tmp_path, document=None):
    for name in ("model.xml", "experiment.yaml"):
        shutil.copyfile(FIXTURES / name, tmp_path / name)
    path = tmp_path / "hypotheses.yaml"
    path.write_text(yaml.safe_dump(workflow_document() if document is None else document, sort_keys=False))
    return path


def test_minimal_specification_and_path_independent_identity(tmp_path):
    path = write_workflow(tmp_path)
    spec = load_hypothesis_testing_spec(path)
    assert spec.null.state_generation.count == 2
    assert spec.null.state_generation.burn_in == 100
    assert spec.testing.max_outcomes == 100_000
    assert spec.output_directory == tmp_path / "results"
    assert [item.kind for item in spec.input_sources] == ["workflow", "model", "experiment"]
    assert spec.input_sources[-1].forward_fva_fraction_of_optimum == 1.0
    assert spec.fingerprint == load_hypothesis_testing_spec(path).fingerprint
    assert replace(spec, output_directory=tmp_path / "elsewhere", input_sources=()).fingerprint == spec.fingerprint
    other = tmp_path / "copy"
    other.mkdir()
    assert load_hypothesis_testing_spec(write_workflow(other)).fingerprint == spec.fingerprint


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "2", None])
def test_count_totals_are_literal_positive_counts(tmp_path, value):
    document = workflow_document()
    document["experiments"][0]["counts"][0]["total_count"] = value
    with pytest.raises(InputValidationError):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))


@pytest.mark.parametrize("bounds", [
    [{"reaction_id": "unknown", "upper_bound": 1}],
    [{"reaction_id": "foreign_hx"}],
    [{"reaction_id": "foreign_hx", "lower_bound": 3, "upper_bound": 1}],
    [{"reaction_id": "foreign_hx", "upper_bound": True}],
    [{"reaction_id": "foreign_hx", "upper_bound": "3"}],
    [{"reaction_id": "foreign_hx", "upper_bound": None}],
    [{"reaction_id": "foreign_hx", "upper_bound": float("inf")}],
    [{"reaction_id": "foreign_hx", "upper_bound": 1}] * 2,
    [{"reaction_id": "foreign_hx", "upper_bound": 1, "pathway": "inferred"}],
])
def test_malformed_hypothesis_constraints_fail(tmp_path, bounds):
    document = workflow_document()
    document["hypotheses"]["H1"]["reaction_bounds"] = bounds
    with pytest.raises(InputValidationError):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))


@pytest.mark.parametrize("settings", [
    {"epsilon": 0}, {"epsilon": 1}, {"epsilon": True}, {"epsilon": "0.1"},
    {"procedures": ["unknown"]}, {"procedures": ["exact_minimax"] * 2},
    {"procedures": ["composite_converse"]},
    {"procedures": ["candidate_score"], "score_orders": [1]},
    {"procedures": ["candidate_score"], "score_orders": [0.5, 0.5]},
    {"converse_orders": [2]}, {"score_orders": [0.5]},
    {"max_outcomes": True}, {"max_outcomes": 0},
])
def test_invalid_testing_settings_fail(tmp_path, settings):
    document = workflow_document()
    document["testing"].update(settings)
    with pytest.raises(InputValidationError):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))


@pytest.mark.parametrize("settings", [
    {"count": 0}, {"count": True}, {"seed": -1}, {"seed": "1"},
    {"method": "independent_fva"}, {"burn_in": -1}, {"thinning": 0},
    {"max_direction_attempts": 1.0}, {"fraction_of_optimum": 1},
])
def test_invalid_generation_settings_fail(tmp_path, settings):
    document = workflow_document()
    document["hypotheses"]["H0"]["state_generation"].update(settings)
    with pytest.raises(InputValidationError):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))


def test_independence_and_experiment_count_order(tmp_path):
    document = workflow_document()
    document["experiments"][0]["counts"].append({"target_id": "portable_mid", "replicate_id": "r1", "total_count": 3})
    with pytest.raises(InputValidationError, match="independence"):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))
    document["observations"]["independent_blocks"] = True
    spec = load_hypothesis_testing_spec(write_workflow(tmp_path, document))
    assert [(item.replicate_id, item.total_count) for item in spec.observation.experiments[0].specifications] == [("r0", 2), ("r1", 3)]
    document["observations"]["independent_blocks"] = "true"
    with pytest.raises(InputValidationError, match="boolean"):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))


def test_no_inferred_count_semantics_and_unknown_root_fields(tmp_path):
    document = workflow_document()
    document["observations"]["semantics"] = "normalized_mid"
    with pytest.raises(InputValidationError, match="genuine_counts"):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))
    document = workflow_document()
    document["prior"] = [0.5, 0.5]
    with pytest.raises(InputValidationError, match="unknown"):
        load_hypothesis_testing_spec(write_workflow(tmp_path, document))


def test_experiment_scalar_coercions_rejected_before_native_loader(tmp_path):
    path = write_workflow(tmp_path)
    experiment = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    experiment["experiment"]["targets"][0]["atom_positions"][0] = 1.5
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    with pytest.raises(InputValidationError, match="integer"):
        load_hypothesis_testing_spec(path)


def test_incompatible_experiment_normalizes_native_validation_error(tmp_path):
    path = write_workflow(tmp_path)
    spec = load_hypothesis_testing_spec(path)
    experiment = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    experiment["experiment"]["targets"][0]["atom_positions"] = [7]
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    with pytest.raises(InputValidationError, match="invalid workflow model or isotope experiment"):
        load_hypothesis_testing_spec(path)
    block = spec.observation.experiments[0]
    invalid_target = replace(block.experiment.targets[0], atom_positions=(7,))
    invalid_experiment = replace(block.experiment, targets=(invalid_target,))
    observation = replace(spec.observation, experiments=(replace(block, experiment=invalid_experiment),))
    with pytest.raises(InputValidationError, match="invalid workflow model or isotope experiment"):
        replace(spec, observation=observation)


def test_explicit_model_override_and_duplicate_keys(tmp_path):
    document = workflow_document()
    del document["model"]
    path = write_workflow(tmp_path, document)
    spec = load_hypothesis_testing_spec(path, model_path=tmp_path / "model.xml")
    assert spec.model.flux_model.reactions[0].reaction_id == "foreign_hx"
    path.write_text(path.read_text() + "schema_version: 1\n")
    with pytest.raises(InputValidationError, match="duplicate"):
        load_hypothesis_testing_spec(path, model_path=tmp_path / "model.xml")


def test_typed_scientific_records_reject_mutable_or_malformed_inputs(tmp_path):
    with pytest.raises(InputValidationError):
        ReactionBoundConstraint("r", upper_bound=True)
    with pytest.raises(InputValidationError):
        StateGenerationSpecification("hit_and_run", 1, False)
    with pytest.raises(InputValidationError):
        HypothesisSpecification("a", [], StateGenerationSpecification("hit_and_run", 1, 0))
    with pytest.raises(InputValidationError):
        WorkflowTestingSpecification(0.1, ["exact_minimax"])
    spec = load_hypothesis_testing_spec(write_workflow(tmp_path))
    with pytest.raises(InputValidationError):
        replace(spec, independent_blocks=1)
    with pytest.raises(InputValidationError):
        WorkflowSpecification(None, spec.null, spec.alternative, spec.testing, False)


def test_scientific_constraint_and_order_changes_change_identity(tmp_path):
    document = workflow_document()
    original = load_hypothesis_testing_spec(write_workflow(tmp_path, document))
    changed = deepcopy(document)
    changed["hypotheses"]["H0"]["reaction_bounds"][0]["upper_bound"] = 4
    assert load_hypothesis_testing_spec(write_workflow(tmp_path, changed)).fingerprint != original.fingerprint
    changed = deepcopy(document)
    changed["hypotheses"]["H0"], changed["hypotheses"]["H1"] = changed["hypotheses"]["H1"], changed["hypotheses"]["H0"]
    assert load_hypothesis_testing_spec(write_workflow(tmp_path, changed)).fingerprint != original.fingerprint

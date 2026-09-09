"""Declarative corrected-MID Dirichlet workflow and refusal semantics."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import StationaryDirichletObservationSpecification
from fluxemu.testing import (
    DirichletCompositeBinaryTestingProblem,
    StationaryDirichletCompositeTestingResult,
)
from fluxemu.workflow import (
    load_hypothesis_testing_spec,
    run_hypothesis_testing_workflow,
    workflow_report,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "hypothesis_workflow"
DIRICHLET = FIXTURE / "workflow_dirichlet.yaml"
COUNT = FIXTURE / "workflow.yaml"


def document():
    value = yaml.safe_load(DIRICHLET.read_text())
    value["model"] = str(FIXTURE / "model.xml")
    value["experiments"][0]["specification"] = str(FIXTURE / "experiment.yaml")
    return value


def write(tmp_path, value):
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def acceptance(tmp_path_factory):
    output = tmp_path_factory.mktemp("dirichlet-workflow")
    return run_hypothesis_testing_workflow(DIRICHLET, output_directory=output)


def test_schema_loads_parallel_continuous_observation_declaration():
    specification = load_hypothesis_testing_spec(DIRICHLET)
    assert specification.observation_semantics == "corrected_mid_dirichlet"
    assert isinstance(specification.observation, StationaryDirichletObservationSpecification)
    assert specification.observation.correction.status == "externally_corrected"
    item = specification.observation.experiments[0].specifications[0]
    assert item.precision == 50.0
    assert item.precision_source == "fixed_external"
    assert item.replicate_count == 3
    assert item.independent_replicates is True
    assert item.replicate_semantics == "technical_measurement_variability"
    assert not hasattr(item, "total_count")


def test_end_to_end_workflow_reports_valid_bounds_and_continuous_refusals(acceptance):
    assert isinstance(acceptance.stationary, StationaryDirichletCompositeTestingResult)
    assert isinstance(acceptance.problem, DirichletCompositeBinaryTestingProblem)
    records = {item.procedure: item for item in acceptance.testing_results}
    assert records["composite_converse"].status == "evaluated"
    assert records["candidate_score"].status == "evaluated"
    assert records["candidate_score"].value.uniform_moment_bounds_verified
    assert records["analytical_score_bound"].status == "evaluated"
    for name in ("exact_minimax", "deterministic_score_error", "calibrated_score_error"):
        assert records[name].status == "refused"
        assert records[name].refusal.category == "unsupported_for_continuous_observation_space"
        assert "unsupported_for_continuous_observation_space" in records[name].refusal.reason
    assert len(acceptance.relationship_checks) == 1
    relationship = acceptance.relationship_checks[0]
    assert relationship.lower_quantity == "composite_converse"
    assert relationship.upper_quantity == "analytical_score_bound"
    assert relationship.passed


def test_machine_report_retains_laws_support_precision_and_correction(acceptance):
    report = workflow_report(acceptance)
    assert report["status"] == "completed_with_refusals"
    observations = report["observations"]
    assert observations["semantics"] == "corrected_mid_dirichlet"
    assert observations["continuous_observation_space"] is True
    assert observations["precision_is_count"] is False
    assert observations["correction"] == {
        "status": "externally_corrected",
        "method": "public-synthetic-declaration",
        "provenance": "tests/fixtures/hypothesis_workflow/workflow_dirichlet.yaml",
    }
    block = observations["block_order"][0]
    assert block["active_support"] == [0, 6]
    assert block["structural_zero_mass_classes"] == [1, 2, 3, 4, 5]
    assert block["precision"] == 50.0
    assert block["precision_source"] == "fixed_external"
    assert block["replicate_count"] == 3
    assert "total_count" not in block
    for family in observations["families"]:
        for member in family["members"]:
            law = member["blocks"][0]
            assert law["predicted_mean_mid"][1:6] == [0.0] * 5
            assert len(law["dirichlet_parameters"]) == 2
            assert law["correction"] == observations["correction"]
    candidate = next(
        item for item in report["results"] if item["procedure"] == "candidate_score"
    )["result"]
    assert candidate["analytic_score_blocks"][0]["active_support"] == [0, 6]
    assert candidate["analytic_score_blocks"][0]["replicate_count"] == 3
    assert "continuous" in " ".join(report["scope"]).lower()
    assert "not a count" in acceptance.summary


def test_report_persistence_and_cli_refusals_exit_successfully(tmp_path):
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable, "-m", "fluxemu.cli", "test-hypotheses",
            "--specification", str(DIRICHLET), "--output", str(output),
        ],
        cwd=ROOT.parent, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "corrected-MID Dirichlet" in completed.stdout
    assert "REFUSED exact_minimax" in completed.stdout
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed_with_refusals"
    assert (output / "summary.txt").read_text().strip() == completed.stdout.strip()


def test_same_data_plugin_keeps_workflow_but_refuses_known_precision_guarantees(tmp_path):
    value = document()
    value["observations"]["blocks"][0]["noise_model"]["precision_source"] = "same_data_plugin"
    result = run_hypothesis_testing_workflow(write(tmp_path, value))
    records = {item.procedure: item for item in result.testing_results}
    for name in ("composite_converse", "candidate_score"):
        assert records[name].status == "refused"
        assert records[name].refusal.category == "model_assumption_or_calibration"
        assert "same_data_plugin" in records[name].refusal.reason
    assert records["analytical_score_bound"].refusal.category == "prerequisite_refused"
    assert records["exact_minimax"].refusal.category == "unsupported_for_continuous_observation_space"


@pytest.mark.parametrize("mutation", [
    "missing_correction", "unknown_correction", "uncorrected", "precision_string",
    "precision_nan", "replicate_float", "replicate_semantics", "independence_string",
    "missing_independence", "distribution", "invented_n", "uncertainty_field",
    "counts_in_experiment", "missing_block", "unknown_experiment",
])
def test_dirichlet_schema_never_coerces_or_invents_scientific_semantics(tmp_path, mutation):
    value = document()
    observations = value["observations"]
    block = observations["blocks"][0]
    noise = block["noise_model"]
    if mutation == "missing_correction":
        observations.pop("correction")
    elif mutation == "unknown_correction":
        observations["correction"]["status"] = "unknown"
    elif mutation == "uncorrected":
        observations["correction"]["status"] = "uncorrected"
    elif mutation == "precision_string":
        noise["precision"] = "50"
    elif mutation == "precision_nan":
        noise["precision"] = float("nan")
    elif mutation == "replicate_float":
        block["replicate_count"] = 3.0
    elif mutation == "replicate_semantics":
        block["replicate_semantics"] = "replicates"
    elif mutation == "independence_string":
        block["independent_replicates"] = "true"
    elif mutation == "missing_independence":
        block.pop("independent_replicates")
    elif mutation == "distribution":
        noise["distribution"] = "multinomial"
    elif mutation == "invented_n":
        noise["effective_sample_size"] = 100
    elif mutation == "uncertainty_field":
        block["standard_error"] = [0.01, 0.01]
    elif mutation == "counts_in_experiment":
        value["experiments"][0]["counts"] = [
            {"target_id": "pool-mid", "replicate_id": "r", "total_count": 3}
        ]
    elif mutation == "missing_block":
        observations["blocks"] = []
    else:
        block["experiment_id"] = "unknown"
    with pytest.raises(InputValidationError):
        load_hypothesis_testing_spec(write(tmp_path, value))


def test_multiple_dirichlet_blocks_require_explicit_block_independence(tmp_path):
    value = document()
    second = deepcopy(value["observations"]["blocks"][0])
    second["replicate_id"] = "technical-series-2"
    second["noise_model"]["precision"] = 80.0
    value["observations"]["blocks"].append(second)
    with pytest.raises(InputValidationError, match="explicit independence"):
        load_hypothesis_testing_spec(write(tmp_path, value))
    value["observations"]["independent_blocks"] = True
    specification = load_hypothesis_testing_spec(write(tmp_path, value))
    assert tuple(
        item.precision for item in specification.observation.experiments[0].specifications
    ) == (50.0, 80.0)


def test_blocks_may_declare_distinct_external_correction_provenance(tmp_path):
    value = document()
    second = deepcopy(value["observations"]["blocks"][0])
    second["replicate_id"] = "technical-series-2"
    second["correction"] = {
        "status": "externally_corrected",
        "method": "fragment-specific-method",
        "provenance": "public synthetic fragment-specific source",
    }
    value["observations"]["blocks"].append(second)
    value["observations"]["independent_blocks"] = True
    result = run_hypothesis_testing_workflow(write(tmp_path, value))
    laws = result.problem.null.members[0].blocks
    assert laws[0].correction.method == "public-synthetic-declaration"
    assert laws[1].correction.method == "fragment-specific-method"
    report = workflow_report(result)
    assert [item["correction"]["method"] for item in report["observations"]["block_order"]] == [
        "public-synthetic-declaration", "fragment-specific-method",
    ]


def test_precision_correction_and_replicate_changes_modify_workflow_identity(tmp_path):
    original_value = document()
    original = load_hypothesis_testing_spec(write(tmp_path, original_value)).fingerprint
    identities = {original}
    mutations = (
        ("precision", 51.0),
        ("precision_source", "independent_calibration"),
        ("replicate_semantics", "total_replicate_variability"),
        ("replicate_count", 4),
        ("correction", "another-method"),
    )
    for index, (name, changed) in enumerate(mutations):
        value = document()
        if name == "precision":
            value["observations"]["blocks"][0]["noise_model"][name] = changed
        elif name == "precision_source":
            value["observations"]["blocks"][0]["noise_model"][name] = changed
        elif name == "correction":
            value["observations"]["correction"]["method"] = changed
        else:
            value["observations"]["blocks"][0][name] = changed
        identities.add(load_hypothesis_testing_spec(write(tmp_path, value)).fingerprint)
    assert len(identities) == 1 + len(mutations)


def test_genuine_count_scientific_fingerprints_and_report_contract_are_unchanged():
    specification = load_hypothesis_testing_spec(COUNT)
    assert specification.observation_semantics == "genuine_counts"
    assert specification.fingerprint == "e89ccff29ae63af96d163882c7db3da9f763504d856466008bf2869833524348"
    assert specification.observation.fingerprint == "7b850c7c46d0f270d30399e25d292ba4845b946deacd9bb2cd8d60056a72779c"
    result = run_hypothesis_testing_workflow(specification)
    report = workflow_report(result)
    assert report["observations"]["semantics"] == "genuine_counts"
    assert report["observations"]["block_order"] == [{
        "experiment_id": "mixed-glucose", "target_id": "pool-mid",
        "replicate_id": "counts-1", "total_count": 2,
        "mass_classes": [0, 1, 2, 3, 4, 5, 6],
    }]
    assert "continuous_observation_space" not in report["observations"]
    assert all(item.status == "evaluated" for item in result.testing_results)
    assert not result.refusals

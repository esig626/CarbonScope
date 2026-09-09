"""Acceptance from real SBML and declarative bounds to persisted testing output."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fluxemu import run_hypothesis_testing_workflow


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "hypothesis_workflow"


def _command(specification: Path, output: Path, *extra: str):
    return subprocess.run(
        [sys.executable, "-m", "fluxemu.cli", "test-hypotheses",
         "--specification", str(specification), "--output", str(output), *extra],
        cwd=ROOT.parent, check=False, capture_output=True, text=True,
    )


def _declaration():
    declaration = yaml.safe_load((FIXTURE / "workflow.yaml").read_text())
    declaration["model"] = str(FIXTURE / "model.xml")
    for experiment in declaration["experiments"]:
        experiment["specification"] = str(FIXTURE / "experiment.yaml")
    return declaration


@pytest.fixture(scope="module")
def acceptance(tmp_path_factory):
    output = tmp_path_factory.mktemp("hypothesis-acceptance")
    return run_hypothesis_testing_workflow(
        FIXTURE / "workflow.yaml", output_directory=output,
    )


def test_real_sbml_bound_regions_generate_native_complete_count_laws(acceptance):
    result = acceptance
    assert result.stationary.hypotheses.null_states == result.null_family.states
    assert result.stationary.hypotheses.alternative_states == result.alternative_family.states
    for family, source, laws, low, high in (
        (result.null_family, result.stationary.null_observation_laws,
         result.problem.null, 7.0, 8.0),
        (result.alternative_family, result.stationary.alternative_observation_laws,
         result.problem.alternative, 2.0, 3.0),
    ):
        assert len(family.states) == len(laws.members) == 2
        assert source.states == family.states
        assert source.validation.valid
        for state, component, member in zip(
            family.states, source.components, laws.members, strict=True,
        ):
            assert tuple(reaction for reaction, _ in state.values) == ("HX_U", "HX_L", "SINK")
            flux = dict(state.values)
            assert low <= flux["HX_U"] <= high
            assert 0 <= flux["HX_L"] <= 10
            assert flux["SINK"] == 10.0
            assert flux["HX_U"] + flux["HX_L"] == pytest.approx(10.0, abs=1e-12)
            # Independent steady-state dilution control, not a supplied law:
            # each mapped input preserves its all-light or all-heavy skeleton.
            probabilities = component.law.probabilities
            assert probabilities[0] == pytest.approx(flux["HX_U"] / 10.0, abs=2e-15)
            assert probabilities[6] == pytest.approx(flux["HX_L"] / 10.0, abs=2e-15)
            assert probabilities[1:6] == (0.0,) * 5
            assert member.blocks == (component.law,)
            assert member.block_totals == (2,)
            assert member.block_identities == (("mixed-glucose", "pool-mid", "counts-1"),)
    assert result.problem.null.members != result.problem.alternative.members


def test_acceptance_all_requested_quantities_are_separate_and_enumerate_full_space(acceptance):
    expected = (
        "composite_converse", "exact_minimax", "candidate_score",
        "analytical_score_bound", "deterministic_score_error", "calibrated_score_error",
    )
    assert tuple(item.procedure for item in acceptance.testing_results) == expected
    assert all(item.requested and item.status == "evaluated" for item in acceptance.testing_results)
    assert not acceptance.refusals
    results = {item.procedure: item.value for item in acceptance.testing_results}
    minimax = results["exact_minimax"]
    assert len(minimax.outcomes) == 28  # C(2 + 7 - 1, 7 - 1), including zero-mass outcomes.
    # For these separated binary-support mixtures, two heavy observations
    # have the greatest likelihood ratio for the nearest pair. At this budget
    # the NP rule randomises only there and simultaneously controls every
    # represented member, giving an independent closed-form minimax control.
    null_heavy = tuple(member.blocks[0].probabilities[6] for member in acceptance.problem.null.members)
    alternative_heavy = tuple(
        member.blocks[0].probabilities[6] for member in acceptance.problem.alternative.members
    )
    assert max(null_heavy) < min(alternative_heavy)
    assert 0.05 < max(null_heavy) ** 2
    expected_beta = 1.0 - 0.05 * min(alternative_heavy) ** 2 / max(null_heavy) ** 2
    assert minimax.minimax_type_ii_error == pytest.approx(expected_beta, abs=2e-12)
    assert results["candidate_score"].uniform_moment_bounds_verified
    assert results["composite_converse"].type_ii_lower_bound <= minimax.minimax_type_ii_error + 2e-9
    for procedure in ("deterministic_score_error", "calibrated_score_error"):
        achieved = results[procedure]
        assert minimax.minimax_type_ii_error <= achieved.worst_type_ii_error + 2e-9
        assert achieved.worst_type_i_error <= 0.05 + 1e-12
    assert all(check.passed for check in acceptance.relationship_checks)


def test_acceptance_persisted_report_binds_public_result_and_summary(acceptance):
    paths = acceptance.output_paths
    assert paths is not None
    assert paths.report.name == "report.json"
    assert paths.summary.name == "summary.txt"
    report = json.loads(paths.report.read_text())
    assert report["schema_version"] == 1
    assert report["status"] == "completed"
    assert report["identity"]["composite_problem_fingerprint"] == acceptance.problem.fingerprint
    assert report["identity"]["specification_fingerprint"] == acceptance.specification.fingerprint
    assert [(item["procedure"], item["status"]) for item in report["results"]] == [
        (item.procedure, item.status) for item in acceptance.testing_results
    ]
    assert report["refusals"] == []
    assert paths.summary.read_text().strip() == acceptance.summary.strip()
    assert "H0" in acceptance.summary and "H1" in acceptance.summary
    assert "finite" in acceptance.summary.lower()


def test_workflow_preserves_experiment_target_replicate_and_count_order(tmp_path):
    experiment = yaml.safe_load((FIXTURE / "experiment.yaml").read_text())
    experiment["experiment"]["targets"].append({
        "target_id": "tail-mid", "metabolite_id": "g6p", "atom_positions": [6],
    })
    experiment_path = tmp_path / "ordered-experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))
    declaration = _declaration()
    declaration["experiments"] = [
        {
            "experiment_id": "z-exp", "specification": str(experiment_path),
            "counts": [
                {"target_id": "pool-mid", "replicate_id": "z", "total_count": 1},
                {"target_id": "tail-mid", "replicate_id": "a", "total_count": 1},
            ],
        },
        {
            "experiment_id": "a-exp", "specification": str(experiment_path),
            "counts": [
                {"target_id": "tail-mid", "replicate_id": "z", "total_count": 2},
                {"target_id": "pool-mid", "replicate_id": "a", "total_count": 1},
            ],
        },
    ]
    declaration["observations"]["independent_blocks"] = True
    path = tmp_path / "ordered.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False))
    result = run_hypothesis_testing_workflow(path, output_directory=tmp_path / "output")
    identities = (
        ("z-exp", "pool-mid", "z"), ("z-exp", "tail-mid", "a"),
        ("a-exp", "tail-mid", "z"), ("a-exp", "pool-mid", "a"),
    )
    totals = (1, 1, 2, 1)
    for states, source, family in (
        (result.null_family.states, result.stationary.null_observation_laws, result.problem.null),
        (result.alternative_family.states, result.stationary.alternative_observation_laws,
         result.problem.alternative),
    ):
        assert tuple(item.state for item in source.components) == tuple(
            state for state in states for _ in identities
        )
        assert tuple((item.experiment_id, item.target_id, item.replicate_id)
                     for item in source.components) == identities * len(states)
        for member in family.members:
            assert member.block_identities == identities
            assert member.block_totals == totals
            assert member.blocks[0].probabilities[1:6] == (0.0,) * 5
            assert member.blocks[3].probabilities[1:6] == (0.0,) * 5
    report = json.loads(result.output_paths.report.read_text())
    assert tuple((block["experiment_id"], block["target_id"], block["replicate_id"])
                 for block in report["observations"]["block_order"]) == identities
    assert tuple(block["total_count"] for block in report["observations"]["block_order"]) == totals
    exact = next(item for item in result.testing_results if item.procedure == "exact_minimax")
    assert exact.status == "evaluated"
    assert len(exact.value.outcomes) == 294


def test_hypothesis_cli_replays_identical_persisted_reports(tmp_path):
    outputs = (tmp_path / "first", tmp_path / "second")
    for output in outputs:
        completed = _command(FIXTURE / "workflow.yaml", output)
        assert completed.returncode == 0, completed.stderr
        assert "H0" in completed.stdout and "H1" in completed.stdout
        assert {path.name for path in output.iterdir()} == {"report.json", "summary.txt"}
    for name in ("report.json", "summary.txt"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


@pytest.mark.parametrize("invalid", ["counts", "semantics", "reaction", "budget", "independence"])
def test_hypothesis_cli_malformed_science_fails_without_report(tmp_path, invalid):
    declaration = _declaration()
    if invalid == "counts":
        declaration["experiments"][0]["counts"][0].pop("total_count")
    elif invalid == "semantics":
        declaration["observations"]["semantics"] = "peak_areas"
    elif invalid == "reaction":
        declaration["hypotheses"]["H0"]["reaction_bounds"][0]["reaction_id"] = "UNKNOWN"
    elif invalid == "budget":
        declaration["testing"]["epsilon"] = 1.0
    else:
        declaration["observations"]["independent_blocks"] = "false"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False))
    output = tmp_path / "output"
    completed = _command(path, output)
    assert completed.returncode == 2
    assert "fluxemu:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (output / "report.json").exists()


def test_hypothesis_cli_explicit_exact_refusal_is_successful_workflow(tmp_path):
    declaration = _declaration()
    declaration["testing"]["procedures"] = ["composite_converse", "exact_minimax"]
    declaration["testing"]["score_orders"] = []
    declaration["testing"]["max_outcomes"] = 1
    path = tmp_path / "refused.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False))
    output = tmp_path / "output"
    completed = _command(path, output)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed_with_refusals"
    outcomes = {item["procedure"]: item for item in report["results"]}
    assert outcomes["composite_converse"]["status"] == "evaluated"
    assert outcomes["exact_minimax"]["status"] == "refused"
    refusal = outcomes["exact_minimax"]["refusal"]
    assert refusal["category"] == "enumeration_limit"
    assert refusal["reason"]
    assert report["refusals"]


def test_hypothesis_cli_model_override_is_explicit_and_usable(tmp_path):
    declaration = _declaration()
    declaration["model"] = "not-used.xml"
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False))
    output = tmp_path / "output"
    completed = _command(path, output, "--model", str(FIXTURE / "model.xml"))
    assert completed.returncode == 0, completed.stderr
    assert (output / "report.json").is_file()


def test_hypothesis_cli_requires_declared_output_directory():
    completed = subprocess.run(
        [sys.executable, "-m", "fluxemu.cli", "test-hypotheses",
         "--specification", str(FIXTURE / "workflow.yaml")],
        cwd=ROOT.parent, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "requires --output or specification output.directory" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_hypothesis_cli_uses_specification_output_directory(tmp_path):
    declaration = _declaration()
    declaration["output"] = {"directory": "declared-output"}
    path = tmp_path / "declared-output.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False))
    completed = subprocess.run(
        [sys.executable, "-m", "fluxemu.cli", "test-hypotheses",
         "--specification", str(path)],
        cwd=ROOT.parent, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "declared-output" / "report.json").is_file()

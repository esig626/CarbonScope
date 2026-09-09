"""Independent controls for scientific labels, numerical refusals and identity."""

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import pytest

from fluxemu import testing
from fluxemu.exceptions import InputValidationError, ValidationError
from fluxemu.workflow import (
    ReactionBoundConstraint, load_hypothesis_testing_spec,
    persist_workflow_report, run_hypothesis_testing_workflow, workflow_report,
)
from fluxemu.workflow import runner


FIXTURE = Path(__file__).parent / "fixtures" / "hypothesis_workflow"


def _spec():
    return load_hypothesis_testing_spec(FIXTURE / "workflow.yaml")


def _bounds(hypothesis, lower, upper):
    return replace(hypothesis, reaction_bounds=(ReactionBoundConstraint("HX_U", lower, upper),))


def _singleton_spec():
    spec = _spec()
    block = spec.observation.experiments[0]
    block = replace(block, specifications=(replace(block.specifications[0], total_count=1),))
    return replace(spec, null=_bounds(spec.null, 8, 8), alternative=_bounds(spec.alternative, 2, 2),
                   observation=replace(spec.observation, experiments=(block,)))


def test_native_singletons_match_independent_binary_np_and_holder_controls():
    result = run_hypothesis_testing_workflow(_singleton_spec())
    values = {item.procedure: item.value for item in result.testing_results}
    # Light/heavy single-count probabilities are (.8,.2) versus (.2,.8).
    # NP rejects a quarter of heavy observations: alpha=.2*.25; beta=.8.
    exact = values["exact_minimax"]
    assert exact.minimax_type_ii_error == pytest.approx(.8, abs=5e-10)
    assert exact.worst_type_i_error == pytest.approx(.05, abs=1e-15)
    assert values["calibrated_score_error"].worst_type_ii_error == pytest.approx(.8, abs=5e-10)
    assert values["deterministic_score_error"].worst_type_ii_error == pytest.approx(1.)
    assert values["composite_converse"].type_ii_lower_bound == pytest.approx(
        1 - math.sqrt(.05 * (.2**2 / .8 + .8**2 / .2)), abs=2e-15,
    )
    report = workflow_report(result)
    assert len({item["result"]["quantity"] for item in report["results"]}) == 6
    assert not values["candidate_score"].finite_n_least_favourable_claimed


def test_failed_native_score_moments_refuse_analytical_but_keep_direct_calibration():
    spec = _spec()
    spec = replace(spec, null=_bounds(spec.null, 1, 9), alternative=_bounds(spec.alternative, 3, 7))
    result = run_hypothesis_testing_workflow(spec)
    records = {item.procedure: item for item in result.testing_results}
    candidate = records["candidate_score"].value
    assert not candidate.uniform_moment_bounds_verified
    assert candidate.verification_failures
    assert records["analytical_score_bound"].refusal.category == "score_verification"
    assert records["deterministic_score_error"].refusal.category == "prerequisite_refused"
    assert records["calibrated_score_error"].status == "evaluated"
    assert records["calibrated_score_error"].value.worst_type_i_error <= spec.testing.epsilon + 1e-12
    assert all(item.passed for item in result.relationship_checks)


def test_native_exact_coefficient_floor_refusal_preserves_positive_laws(tmp_path):
    spec = _singleton_spec()
    block = spec.observation.experiments[0]
    experiment = replace(block.experiment, targets=(replace(block.experiment.targets[0], atom_positions=(1,)),))
    block = replace(block, experiment=experiment, specifications=(replace(block.specifications[0], total_count=40),))
    spec = replace(spec, null=_bounds(spec.null, 5, 5), alternative=_bounds(spec.alternative, 4, 4),
                   observation=replace(spec.observation, experiments=(block,)),
                   testing=replace(spec.testing, procedures=("exact_minimax",), converse_orders=(), score_orders=()))
    result = run_hypothesis_testing_workflow(spec, output_directory=tmp_path)
    assert len(result.refusals) == 1
    assert "coefficient" in result.refusals[0].reason
    assert result.refusals[0].exception_type == "NumericalLimitError"
    assert all(p > 0 for law in result.problem.alternative.members for p in law.blocks[0].probabilities)
    assert json.loads(result.output_paths.report.read_text())["status"] == "completed_with_refusals"


def test_tiny_valid_budget_refuses_exact_only():
    spec = _singleton_spec()
    spec = replace(spec, testing=replace(spec.testing, epsilon=1e-13))
    result = run_hypothesis_testing_workflow(spec)
    records = {item.procedure: item for item in result.testing_results}
    assert records["exact_minimax"].status == "refused"
    assert records["composite_converse"].status == "evaluated"
    assert result.specification.testing.epsilon == 1e-13
    assert "no budget substitution" in records["exact_minimax"].refusal.reason


def test_requested_procedure_and_order_ordering_and_visible_prerequisites():
    spec = _spec()
    settings = replace(spec.testing, procedures=("deterministic_score_error", "composite_converse"),
                       score_orders=(.7, .3), converse_orders=(3., 2.))
    result = run_hypothesis_testing_workflow(replace(spec, testing=settings))
    assert [(item.procedure, item.order) for item in result.testing_results if item.requested] == [
        ("deterministic_score_error", .7), ("deterministic_score_error", .3),
        ("composite_converse", 3.), ("composite_converse", 2.),
    ]
    prerequisites = [item for item in result.testing_results if not item.requested and item.status == "evaluated"]
    assert {item.procedure for item in prerequisites} == {"candidate_score", "analytical_score_bound"}
    assert next(item for item in result.testing_results if item.procedure == "exact_minimax").status == "not_requested"


@pytest.mark.parametrize("error", [RuntimeError("programming failure"), InputValidationError("invalid primitive binding"),
                                  ValidationError("unexpected invalid law")])
def test_unexpected_primitive_errors_are_never_statistical_refusals(monkeypatch, error, tmp_path):
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(testing, "exact_finite_composite_minimax", broken)
    with pytest.raises(type(error), match=str(error)):
        run_hypothesis_testing_workflow(_spec(), output_directory=tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_relationship_inconsistency_is_failed_construction_not_accepted_report(monkeypatch):
    original = testing.exact_finite_composite_minimax
    def corrupt(*args, **kwargs):
        return replace(original(*args, **kwargs), minimax_type_ii_error=0.)
    monkeypatch.setattr(testing, "exact_finite_composite_minimax", corrupt)
    with pytest.raises(ValidationError, match="workflow integration failed"):
        run_hypothesis_testing_workflow(_singleton_spec())


def test_relocated_inputs_and_output_paths_do_not_change_scientific_identity(tmp_path):
    original = run_hypothesis_testing_workflow(_spec())
    copied = tmp_path / "inputs"
    shutil.copytree(FIXTURE, copied)
    relocated = run_hypothesis_testing_workflow(copied / "workflow.yaml", output_directory=tmp_path / "output")
    assert original.provenance == relocated.provenance
    assert original.null_family.fingerprint == relocated.null_family.fingerprint
    assert original.alternative_family.fingerprint == relocated.alternative_family.fingerprint
    assert workflow_report(original) == workflow_report(relocated)
    assert str(tmp_path) not in json.dumps(workflow_report(relocated))


def test_supported_infinities_are_explicit_json_strings_without_support_repair():
    spec = _spec()
    result = run_hypothesis_testing_workflow(replace(spec, null=_bounds(spec.null, 10, 10),
                                                    alternative=_bounds(spec.alternative, 0, 0)))
    report = workflow_report(result)
    encoded = json.dumps(report, allow_nan=False)
    assert '"Infinity"' in encoded
    assert '"-Infinity"' in encoded
    assert result.problem.null.members[0].blocks[0].probabilities == (1., 0., 0., 0., 0., 0., 0.)
    assert result.problem.alternative.members[0].blocks[0].probabilities == (0., 0., 0., 0., 0., 0., 1.)


def test_typed_spec_does_not_accept_a_second_hidden_model():
    with pytest.raises(InputValidationError, match="file specification only"):
        run_hypothesis_testing_workflow(_spec(), model_path=FIXTURE / "model.xml")


def test_report_contains_complete_tracer_and_model_science():
    report = workflow_report(run_hypothesis_testing_workflow(_spec()))
    science = report["scientific_specification"]
    assert science["common_model"]["flux_model"]["reactions"][0]["reaction_id"] == "HX_U"
    experiment = science["ordered_experiments"][0]
    assert experiment["experiment"]["tracers"][0]["isotopomers"]
    assert experiment["specifications"][0]["total_count"] == 2


def test_persistence_refuses_symlink_to_input_and_preserves_source(tmp_path):
    copied = tmp_path / "input"
    shutil.copytree(FIXTURE, copied)
    result = run_hypothesis_testing_workflow(copied / "workflow.yaml")
    source = copied / "model.xml"
    original = source.read_bytes()
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").symlink_to(source)
    with pytest.raises(InputValidationError, match="overwrite a scientific input"):
        persist_workflow_report(result, output)
    assert source.read_bytes() == original


def test_report_writes_do_not_follow_existing_unrelated_symlink(tmp_path):
    result = run_hypothesis_testing_workflow(_spec())
    target = tmp_path / "unrelated.txt"
    target.write_text("preserve me")
    output = tmp_path / "out"
    output.mkdir()
    (output / "summary.txt").symlink_to(target)
    paths = persist_workflow_report(result, output)
    assert target.read_text() == "preserve me"
    assert not paths.summary.is_symlink()
    assert paths.summary.read_text() == result.summary


def test_invalid_second_output_path_does_not_publish_a_success_report(tmp_path):
    result = run_hypothesis_testing_workflow(_spec())
    (tmp_path / "summary.txt").mkdir()
    with pytest.raises(InputValidationError, match="existing directory"):
        persist_workflow_report(result, tmp_path)
    assert not (tmp_path / "report.json").exists()

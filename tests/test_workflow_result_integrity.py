"""Inconsistent replacements cannot become scientific workflow reports."""

from dataclasses import replace
from pathlib import Path

import pytest

from fluxemu.exceptions import InputValidationError, ValidationError
from fluxemu.workflow import run_hypothesis_testing_workflow, workflow_report


FIXTURE = Path(__file__).parent / "fixtures" / "hypothesis_workflow" / "workflow.yaml"


@pytest.fixture(scope="module")
def result():
    return run_hypothesis_testing_workflow(FIXTURE)


def _replace_procedure(result, procedure, **changes):
    return replace(result, testing_results=tuple(
        replace(item, **changes) if item.procedure == procedure else item
        for item in result.testing_results
    ))


@pytest.mark.parametrize("change", ["role_swap", "hypothesis", "model", "family_fingerprint",
                                  "hypothesis_fingerprint", "states", "seed", "validation", "fva",
                                  "fba", "witnesses"])
def test_result_rejects_misaligned_state_family(result, change):
    family = result.null_family
    if change == "role_swap":
        family = result.alternative_family
    elif change == "hypothesis":
        family = replace(family, hypothesis=result.alternative_family.hypothesis)
    elif change == "model":
        family = replace(family, model=result.alternative_family.model)
    elif change == "family_fingerprint":
        family = replace(family, fingerprint="different-family")
    elif change == "hypothesis_fingerprint":
        family = replace(family, hypothesis_fingerprint="different-hypothesis")
    elif change == "states":
        family = replace(family, sampling=replace(family.sampling, states=family.states[::-1]))
    elif change == "seed":
        family = replace(family, sampling=replace(
            family.sampling, provenance=replace(family.sampling.provenance, seed=99),
        ))
    elif change == "validation":
        family = replace(family, sampling=replace(
            family.sampling, validation=replace(family.sampling.validation, model_fingerprint="other"),
        ))
    elif change == "fva":
        ranges = family.fva.ranges.copy()
        ranges.iloc[0, 0] += 0.25
        family = replace(family, fva=replace(family.fva, ranges=ranges))
    elif change == "fba":
        family = replace(family, fba=result.alternative_family.fba)
    else:
        family = replace(family, region_witnesses=())
    with pytest.raises((InputValidationError, ValidationError)):
        replace(result, null_family=family)


@pytest.mark.parametrize("change", ["budget", "count_total", "score_order"])
def test_result_rejects_scientific_specification_rebinding(result, change):
    specification = result.specification
    if change == "budget":
        specification = replace(specification, testing=replace(specification.testing, epsilon=0.1))
    elif change == "score_order":
        specification = replace(specification, testing=replace(specification.testing, score_orders=(0.7,)))
    else:
        block = specification.observation.experiments[0]
        block = replace(block, specifications=(replace(block.specifications[0], total_count=3),))
        specification = replace(specification, observation=replace(
            specification.observation, experiments=(block,),
        ))
    with pytest.raises((InputValidationError, ValidationError)):
        replace(result, specification=specification)


@pytest.mark.parametrize("change", ["reorder", "mutable", "missing", "requested", "wrong_value",
                                  "budget", "order", "problem", "empty_refusal"])
def test_result_rejects_malformed_or_rebound_testing_records(result, change):
    records = result.testing_results
    exact = next(item for item in records if item.procedure == "exact_minimax")
    converse = next(item for item in records if item.procedure == "composite_converse")
    with pytest.raises((InputValidationError, ValidationError)):
        if change == "reorder":
            replace(result, testing_results=records[::-1])
        elif change == "mutable":
            replace(result, testing_results=list(records))
        elif change == "missing":
            replace(result, testing_results=records[:-1])
        elif change == "requested":
            _replace_procedure(result, "exact_minimax", requested=False)
        elif change == "wrong_value":
            _replace_procedure(result, "exact_minimax", value=converse.value)
        elif change == "budget":
            _replace_procedure(result, "exact_minimax", value=replace(
                exact.value, constraint=replace(exact.value.constraint, epsilon=0.1),
            ))
        elif change == "order":
            _replace_procedure(result, "composite_converse", value=replace(converse.value, order=3.0))
        elif change == "problem":
            changed_problem = replace(result.problem, null=result.problem.alternative,
                                      alternative=result.problem.null)
            _replace_procedure(result, "exact_minimax", value=replace(exact.value, problem=changed_problem))
        else:
            _replace_procedure(result, "exact_minimax", status="refused", value=None, refusal=None)


def test_result_rejects_detached_score_prerequisite(result):
    bound = next(item for item in result.testing_results if item.procedure == "analytical_score_bound")
    changed = replace(bound.value, threshold=bound.value.threshold + 1.0)
    with pytest.raises((InputValidationError, ValidationError)):
        _replace_procedure(result, "analytical_score_bound", value=changed)


def test_result_rejects_changed_identity_and_relationship_evidence(result):
    with pytest.raises((InputValidationError, ValidationError)):
        replace(result, provenance=replace(result.provenance, null_hypothesis_fingerprint="other"))
    with pytest.raises((InputValidationError, ValidationError)):
        replace(result, relationship_checks=())


def test_export_rechecks_integrity_without_sampling_or_emu(result, monkeypatch):
    from fluxemu.workflow import runner
    from fluxemu import testing

    def forbidden(*args, **kwargs):
        pytest.fail("result integrity validation repeated scientific evaluation")

    monkeypatch.setattr(runner, "generate_hypothesis_state_families", forbidden)
    monkeypatch.setattr(testing, "evaluate_stationary_composite_hypotheses", forbidden)
    report = workflow_report(result)
    assert report["identity"]["composite_problem_fingerprint"] == result.problem.fingerprint
    # Frozen records are not authentication, but export must still detect a
    # bypassed constructor whose retained scientific records no longer align.
    altered = replace(result)
    object.__setattr__(altered, "provenance", replace(result.provenance, model_fingerprint="other"))
    with pytest.raises((InputValidationError, ValidationError)):
        workflow_report(altered)

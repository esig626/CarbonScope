"""Native EMU to common-face Dirichlet observation bridge tests."""

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import (
    MIDCorrectionProvenance,
    StationaryDirichletBlockSpecification,
    StationaryDirichletObservationExperiment,
    StationaryDirichletObservationSpecification,
    evaluate_stationary_dirichlet_observation_families,
)
from fluxemu.testing import evaluate_stationary_dirichlet_composite_hypotheses
from fluxemu.workflow import (
    generate_hypothesis_state_families,
    load_hypothesis_testing_spec,
)
import fluxemu.observation.dirichlet_stationary as stationary


FIXTURE = "tests/fixtures/hypothesis_workflow/workflow.yaml"


@pytest.fixture(scope="module")
def native_inputs():
    count_specification = load_hypothesis_testing_spec(FIXTURE)
    null, alternative = generate_hypothesis_state_families(count_specification)
    return count_specification, null, alternative


def specification(count_specification, *blocks, correction=None):
    experiment = count_specification.observation.experiments[0]
    return StationaryDirichletObservationSpecification(
        count_specification.model,
        (StationaryDirichletObservationExperiment(
            experiment.experiment_id, experiment.experiment, tuple(blocks),
        ),),
        correction or MIDCorrectionProvenance(
            "externally_corrected", "declared-method", "public synthetic source",
        ),
    )


def block(**overrides):
    values = dict(
        target_id="pool-mid",
        precision=75.0,
        precision_source="fixed_external",
        precision_provenance="independent synthetic declaration",
        replicate_count=3,
        replicate_semantics="technical_measurement_variability",
        independent_replicates=True,
        replicate_id="mid-series",
    )
    values.update(overrides)
    return StationaryDirichletBlockSpecification(**values)


def test_native_bridge_removes_only_common_structural_zeros(native_inputs):
    count_specification, null, alternative = native_inputs
    declared = specification(count_specification, block())
    result = evaluate_stationary_dirichlet_composite_hypotheses(
        declared,
        null_states=null.states,
        alternative_states=alternative.states,
    )
    expected = ((('mixed-glucose', 'pool-mid', 'mid-series'), (0, 6)),)
    assert result.null_observation_laws.common_active_supports == expected
    assert result.alternative_observation_laws.common_active_supports == expected
    assert len(result.problem.null.members) == len(null.states) == 2
    for family in (result.problem.null, result.problem.alternative):
        for member in family.members:
            law = member.blocks[0]
            assert law.mass_classes == tuple(range(7))
            assert law.active_support == (0, 6)
            assert law.structural_zero_mass_classes == (1, 2, 3, 4, 5)
            assert law.precision == 75.0
            assert law.replicate_count == 3
            assert law.parameters[0] + law.parameters[1] == pytest.approx(75.0)
            assert all(law.predicted_mid[index] == 0.0 for index in range(1, 6))
    assert result.hypotheses.null_states == null.states
    assert result.hypotheses.alternative_states == alternative.states
    assert len(result.fingerprint) == 64


def test_source_family_api_uses_same_face_in_both_roles(native_inputs):
    count_specification, null, alternative = native_inputs
    declared = specification(count_specification, block(replicate_count=1, independent_replicates=False))
    left, right = evaluate_stationary_dirichlet_observation_families(
        declared, null_states=null.states, alternative_states=alternative.states,
    )
    assert left.common_active_supports == right.common_active_supports
    assert tuple(component.active_support for component in left.components) == ((0, 6), (0, 6))
    assert tuple(component.active_support for component in right.components) == ((0, 6), (0, 6))


def test_multiple_blocks_require_explicit_independence(native_inputs):
    count_specification, null, alternative = native_inputs
    declared = specification(
        count_specification,
        block(replicate_id="first"),
        block(replicate_id="second", precision=120.0),
    )
    with pytest.raises(InputValidationError, match="independent_blocks=True"):
        evaluate_stationary_dirichlet_composite_hypotheses(
            declared, null_states=null.states, alternative_states=alternative.states,
            independent_blocks=False,
        )
    result = evaluate_stationary_dirichlet_composite_hypotheses(
        declared, null_states=null.states, alternative_states=alternative.states,
        independent_blocks=True,
    )
    assert len(result.problem.null.members[0].blocks) == 2
    assert result.problem.null.members[0].replicate_counts == (3, 3)
    assert tuple(item.precision for item in result.problem.null.members[0].blocks) == (75.0, 120.0)


def test_precision_correction_and_replicate_semantics_change_identity(native_inputs):
    count_specification, _, _ = native_inputs
    original = specification(count_specification, block())
    changed_precision = specification(count_specification, block(precision=76.0))
    changed_source = specification(
        count_specification,
        block(precision_source="independent_calibration"),
    )
    changed_semantics = specification(
        count_specification,
        block(replicate_semantics="total_replicate_variability"),
    )
    changed_correction = specification(
        count_specification,
        block(),
        correction=MIDCorrectionProvenance(
            "externally_corrected", "different-method", "public synthetic source",
        ),
    )
    fingerprints = {
        item.fingerprint
        for item in (
            original, changed_precision, changed_source,
            changed_semantics, changed_correction,
        )
    }
    assert len(fingerprints) == 5


def test_same_data_plugin_is_retained_for_later_statistical_refusal(native_inputs):
    count_specification, null, alternative = native_inputs
    declared = specification(
        count_specification, block(precision_source="same_data_plugin"),
    )
    result = evaluate_stationary_dirichlet_composite_hypotheses(
        declared, null_states=null.states, alternative_states=alternative.states,
    )
    assert not result.problem.null.members[0].rigorous_testing_suitable
    assert result.problem.null.members[0].blocks[0].precision_source == "same_data_plugin"


def test_joint_face_detection_refuses_state_dependent_support_without_epsilon(native_inputs):
    count_specification, _, _ = native_inputs
    declared = specification(count_specification, block())
    first = stationary._PredictionBatch(
        states=(), rows=(((0.5, 0.0, 0.5),),), plans=(), validation=None,
    )
    second = stationary._PredictionBatch(
        states=(), rows=(((0.5, 0.1, 0.4),),), plans=(), validation=None,
    )
    with pytest.raises(InputValidationError, match="state-dependent active supports"):
        stationary._common_active_supports(declared, (first, second))


@pytest.mark.parametrize("overrides", [
    {"replicate_id": ""},
    {"target_id": ""},
    {"replicate_count": 2, "independent_replicates": False},
    {"precision": 0},
    {"precision_source": "estimated-somehow"},
])
def test_dirichlet_block_declarations_are_strict(overrides):
    with pytest.raises(InputValidationError):
        block(**overrides)

"""Raw count validation, scientific ordering, and law provenance contracts."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.observation import (
    MAX_MULTINOMIAL_TOTAL,
    MIDCountObservation,
    MultinomialMIDLaw,
    StationaryCountSample,
    StationaryCountSpecification,
    StationaryObservationExperiment,
    StationaryObservationLawComponent,
    StationaryObservationSpecification,
    stationary_observation_specification_fingerprint,
    validate_stationary_observation_specification,
)
from test_mfa_schema import _problem


def _specification():
    baseline = _problem()
    experiment = baseline.experiments[0]
    counts = (
        StationaryCountSpecification("a-one", 13, "second"),
        StationaryCountSpecification("z-full", 7),
        StationaryCountSpecification("a-one", 101, "first"),
        StationaryCountSpecification("composite", 11),
    )
    return StationaryObservationSpecification(
        baseline.model,
        (StationaryObservationExperiment("z-exp", experiment.experiment, counts),),
    )


def _component():
    return StationaryObservationLawComponent(
        CanonicalFluxState("truth", (("R", 1.0),)),
        "exp", "target", "rep", MultinomialMIDLaw(4, (0.25, 0.75)),
        "model-fingerprint", "experiment-fingerprint", "specification-fingerprint",
    )


def test_raw_counts_total_order_and_derived_mid_remain_auditable_and_immutable():
    observation = MIDCountObservation((np.int64(0), np.int64(3), np.int64(1)), np.int64(4))
    assert observation.counts == (0, 3, 1)
    assert all(type(item) is int for item in observation.counts)
    assert type(observation.n) is int
    assert observation.n == 4
    assert observation.mass_classes == (0, 1, 2)
    assert observation.empirical_mid == (0.0, 0.75, 0.25)
    with pytest.raises(FrozenInstanceError):
        observation.n = 8
    with pytest.raises(FrozenInstanceError):
        observation.counts = (0, 6, 2)


@pytest.mark.parametrize("counts", [
    (), [], [0, 1], (0, 0), (-1, 2), (0.0, 1.0), (0, 1.5),
    (False, 1), (0, True), (np.bool_(False), 1), (0, math.nan),
    (0, math.inf), (0, "1"), ((0,), 1),
])
def test_counts_never_accept_float_intensities_booleans_or_invalid_measurements(counts):
    with pytest.raises(InputValidationError):
        MIDCountObservation(counts, 1)


@pytest.mark.parametrize("n", [0, -1, 1.0, True, False, math.nan, math.inf, "1", 2**63])
def test_total_is_explicit_positive_integer_with_declared_representability(n):
    with pytest.raises(InputValidationError):
        MIDCountObservation((1, 0), n)
    with pytest.raises(InputValidationError):
        StationaryCountSpecification("target", n)


def test_raw_counts_must_match_total_without_any_rescaling_or_rounding():
    with pytest.raises(InputValidationError, match="sum exactly"):
        MIDCountObservation((30, 70), 1000)
    observation = MIDCountObservation((MAX_MULTINOMIAL_TOTAL, 0), MAX_MULTINOMIAL_TOTAL)
    assert observation.empirical_mid == (1.0, 0.0)
    assert observation.n == MAX_MULTINOMIAL_TOTAL


def test_count_fingerprint_binds_raw_scale_and_class_order():
    first = MIDCountObservation((1, 3), 4)
    same_composition = MIDCountObservation((2, 6), 8)
    assert first.empirical_mid == same_composition.empirical_mid
    assert first.fingerprint != same_composition.fingerprint
    assert first.fingerprint != MIDCountObservation((3, 1), 4).fingerprint
    assert first.fingerprint == MIDCountObservation((1, 3), 4).fingerprint


def test_stationary_declarations_reuse_native_target_and_replicate_order():
    specification = _specification()
    validate_stationary_observation_specification(specification)
    block = specification.experiments[0]
    assert tuple((item.target_id, item.replicate_id, item.total_count)
                 for item in block.specifications) == (
        ("a-one", "second", 13), ("z-full", "0", 7),
        ("a-one", "first", 101), ("composite", "0", 11),
    )
    assert block.experiment.targets[0].atom_positions == (2, 1)
    with pytest.raises(FrozenInstanceError):
        block.specifications[0].total_count = 100


def test_duplicate_experiment_and_observation_identities_and_unknown_targets_fail():
    specification = _specification()
    block = specification.experiments[0]
    with pytest.raises(InputValidationError, match="duplicate experiment_id"):
        validate_stationary_observation_specification(
            replace(specification, experiments=(block, block)))
    with pytest.raises(InputValidationError, match="duplicate observation identity"):
        validate_stationary_observation_specification(replace(specification, experiments=(
            replace(block, specifications=block.specifications * 2),)))
    with pytest.raises(InputValidationError, match="unknown target_id.*z-exp"):
        validate_stationary_observation_specification(replace(specification, experiments=(
            replace(block, specifications=(StationaryCountSpecification("missing", 3),)),)))
    second = replace(block, experiment_id="a-exp")
    multiple = replace(specification, experiments=(block, second))
    validate_stationary_observation_specification(multiple)
    assert tuple(item.experiment_id for item in multiple.experiments) == ("z-exp", "a-exp")


@pytest.mark.parametrize("factory", [
    lambda: StationaryCountSpecification("", 1),
    lambda: StationaryCountSpecification("t", 1, ""),
    lambda: replace(_specification(), model="bad"),
    lambda: replace(_specification(), experiments=[]),
    lambda: replace(_specification(), experiments=()),
    lambda: replace(_specification(), experiments=("bad",)),
    lambda: replace(_specification().experiments[0], experiment="bad"),
    lambda: replace(_specification().experiments[0], specifications=[]),
    lambda: replace(_specification().experiments[0], specifications=()),
    lambda: replace(_specification().experiments[0], specifications=("bad",)),
])
def test_scientific_record_structure_requires_immutable_native_records(factory):
    with pytest.raises(InputValidationError):
        factory()


def test_native_model_and_experiment_validation_are_authoritative():
    specification = _specification()
    block = specification.experiments[0]
    duplicate_targets = replace(block.experiment, targets=block.experiment.targets * 2)
    with pytest.raises(ValueError, match="duplicate target ID"):
        validate_stationary_observation_specification(replace(specification, experiments=(
            replace(block, experiment=duplicate_targets),)))
    flux_model = specification.model.flux_model
    broken_model = replace(specification.model, flux_model=replace(
        flux_model, reactions=flux_model.reactions * 2))
    with pytest.raises(ValueError, match="duplicate"):
        validate_stationary_observation_specification(replace(specification, model=broken_model))


def test_specification_fingerprint_binds_science_identity_totals_and_order():
    specification = _specification()
    block = specification.experiments[0]
    first, *rest = block.specifications
    original = specification.fingerprint
    variants = [
        replace(specification, experiments=(replace(block, experiment_id="different"),)),
        replace(specification, experiments=(replace(block, specifications=block.specifications[::-1]),)),
        replace(specification, experiments=(replace(block, specifications=(replace(first, total_count=14), *rest)),)),
        replace(specification, experiments=(replace(block, specifications=(replace(first, replicate_id="other"), *rest)),)),
        replace(specification, experiments=(replace(block, experiment=replace(
            block.experiment, targets=block.experiment.targets[::-1])),)),
        replace(specification, model=replace(specification.model, flux_model=replace(
            specification.model.flux_model, reactions=(replace(
                specification.model.flux_model.reactions[0], upper_bound=9),)))),
    ]
    assert original == stationary_observation_specification_fingerprint(_specification())
    assert len(original) == 64
    assert all(variant.fingerprint != original for variant in variants)


def test_component_fingerprint_binds_state_probability_total_and_all_identities():
    component = _component()
    assert component.sample_id == "truth"
    assert component.predicted_mid is component.law.probabilities
    assert component.mass_classes == (0, 1)
    changes = (
        {"state": CanonicalFluxState("other", (("R", 1.0),))},
        {"state": CanonicalFluxState("truth", (("R", 2.0),))},
        {"law": MultinomialMIDLaw(5, (0.25, 0.75))},
        {"law": MultinomialMIDLaw(4, (0.5, 0.5))},
        {"experiment_id": "other"}, {"target_id": "other"}, {"replicate_id": "other"},
        {"model_fingerprint": "other"}, {"experiment_fingerprint": "other"},
        {"specification_fingerprint": "other"},
    )
    assert component.fingerprint == _component().fingerprint
    assert all(replace(component, **change).fingerprint != component.fingerprint for change in changes)


@pytest.mark.parametrize("state", [
    CanonicalFluxState(["mutable"], (("R", 1.0),)),
    CanonicalFluxState({"mutable": 1}, (("R", 1.0),)),
    CanonicalFluxState(("nested", []), (("R", 1.0),)),
    CanonicalFluxState("truth", (["R", 1.0],)),
    CanonicalFluxState("truth", (("R", [1.0]),)),
    CanonicalFluxState("truth", (("R", True),)),
    CanonicalFluxState("truth", (("R", math.inf),)),
    CanonicalFluxState("truth", (("R", 1.0), ("R", 2.0))),
])
def test_component_rejects_mutable_or_malformed_state_provenance(state):
    with pytest.raises(InputValidationError):
        replace(_component(), state=state)


def test_component_retains_recursively_immutable_sample_identity():
    identity = ("batch", 3, ("truth", None))
    component = replace(_component(), state=CanonicalFluxState(identity, (("R", 1.0),)))
    assert component.sample_id == identity
    assert component.fingerprint == replace(
        _component(), state=CanonicalFluxState(identity, (("R", 1.0),))).fingerprint


def test_law_paired_count_record_requires_exact_dimension_and_total():
    component = _component()
    sample = StationaryCountSample(component, MIDCountObservation((1, 3), 4))
    assert sample.observation.counts == (1, 3)
    assert sample.component.predicted_mid == (0.25, 0.75)
    with pytest.raises(InputValidationError, match="total"):
        StationaryCountSample(component, MIDCountObservation((1, 4), 5))
    with pytest.raises(InputValidationError, match="mass classes"):
        StationaryCountSample(component, MIDCountObservation((1, 2, 1), 4))

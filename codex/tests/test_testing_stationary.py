"""Fixed feasible flux hypotheses, genuine-count scaling, and provenance."""

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from fluxemu import observation as observation_api
from fluxemu.exceptions import AnalysisError, InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.observation import (
    MIDCountObservation,
    StationaryCountSpecification,
    StationaryObservationExperiment,
    StationaryObservationSpecification,
)
from fluxemu.observation import stationary as observation_module
from fluxemu.testing import (
    BrunoTheoremAssumptionError,
    SimpleBinaryFluxHypotheses,
    bruno_converse_at_order,
    evaluate_stationary_simple_hypotheses,
    log_likelihood_ratio,
)
from test_native_stationary_analysis import _alternate_optimum_science
from test_observation_stationary import _specification as _ordered_specification


def _state(sample_id, unlabelled=2.5, total=10.0):
    return CanonicalFluxState(sample_id, (
        ("Z_IN", unlabelled), ("A_IN", total - unlabelled), ("M_OUT", total),
    ))


def _specification(n=4):
    model, experiment = _alternate_optimum_science()
    return StationaryObservationSpecification(model, (
        StationaryObservationExperiment("fixed-tracer", experiment, (
            StationaryCountSpecification("O-mid", n, "genuine-counts"),
        )),
    ))


def test_fixed_flux_pair_retains_native_sources_mids_roles_and_certificate_fingerprints():
    specification = _specification()
    null, alternative = _state("null"), _state("alternative", 5.0)
    result = evaluate_stationary_simple_hypotheses(
        specification, null_state=null, alternative_state=alternative,
    )
    assert result.hypotheses.null_state is null
    assert result.hypotheses.alternative_state is alternative
    assert result.null_observation_laws.states == (null,)
    assert result.alternative_observation_laws.states == (alternative,)
    assert result.null_observation_laws.validation.valid
    assert result.alternative_observation_laws.validation.valid
    assert result.null_predicted_mids == ((0.25, 0.75),)
    assert result.alternative_predicted_mids == ((0.5, 0.5),)
    assert result.law_pair.null is result.null_components[0].law
    assert result.law_pair.alternative is result.alternative_components[0].law
    assert result.law_pair.count_totals == (4,)
    assert result.law_pair.block_identities == (("fixed-tracer", "O-mid", "genuine-counts"),)
    certificate = bruno_converse_at_order(result, epsilon=0.05, order=1.7)
    assert certificate.pair is result.law_pair
    assert certificate.null_state_fingerprint == result.hypotheses.null_state_fingerprint
    assert certificate.alternative_state_fingerprint == result.hypotheses.alternative_state_fingerprint
    assert certificate.observation_specification_fingerprint == specification.fingerprint
    assert certificate.null_fingerprint == result.null_components[0].law.fingerprint
    assert certificate.alternative_fingerprint == result.alternative_components[0].law.fingerprint
    replay = evaluate_stationary_simple_hypotheses(
        specification, null_state=null, alternative_state=alternative,
    )
    assert result.fingerprint == replay.fingerprint
    assert log_likelihood_ratio(result, MIDCountObservation((1, 3), 4)) == pytest.approx(math.log(16 / 27))
    with pytest.raises(FrozenInstanceError):
        result.hypotheses = replay.hypotheses


def test_larger_explicit_counts_change_full_law_separation_and_certificate_with_fixed_mids():
    certificates = []
    for n in (1, 2, 4, 16, 64):
        result = evaluate_stationary_simple_hypotheses(
            _specification(n), null_state=_state("v0"), alternative_state=_state("v1", 5.0),
        )
        assert result.null_predicted_mids == ((0.25, 0.75),)
        assert result.alternative_predicted_mids == ((0.5, 0.5),)
        certificate = bruno_converse_at_order(result, epsilon=0.05, order=2.0)
        # Independent analytical order-two MID sums, with explicit total n.
        assert certificate.reverse_renyi == pytest.approx(n * math.log(4 / 3), abs=3e-14)
        assert certificate.forward_renyi == pytest.approx(n * math.log(5 / 4), abs=3e-14)
        certificates.append(certificate)
    assert all(a.reverse_renyi < b.reverse_renyi for a, b in zip(certificates, certificates[1:]))
    assert all(a.type_ii_lower_bound > b.type_ii_lower_bound for a, b in zip(certificates, certificates[1:]))
    assert len({item.observation_specification_fingerprint for item in certificates}) == len(certificates)
    assert len({item.null_state_fingerprint for item in certificates}) == 1


def test_increasing_fixed_state_mid_separation_changes_certificate_at_fixed_total():
    certificates = []
    for unlabelled in (3.75, 5.0, 7.5):
        result = evaluate_stationary_simple_hypotheses(
            _specification(4), null_state=_state("v0"), alternative_state=_state("v1", unlabelled),
        )
        certificates.append(bruno_converse_at_order(result, epsilon=0.05, order=2.0))
    assert all(a.reverse_renyi < b.reverse_renyi for a, b in zip(certificates, certificates[1:]))
    assert all(a.type_ii_lower_bound > b.type_ii_lower_bound for a, b in zip(certificates, certificates[1:]))


def test_identical_record_and_distinct_states_sharing_source_id_preserve_identity():
    null = _state("shared-source")
    identical = evaluate_stationary_simple_hypotheses(
        _specification(), null_state=null, alternative_state=null,
    )
    assert identical.hypotheses.null_state is identical.hypotheses.alternative_state is null
    assert identical.null_observation_laws.states == identical.alternative_observation_laws.states == (null,)
    certificate = bruno_converse_at_order(identical, epsilon=0.05, order=2.0)
    assert certificate.reverse_renyi == certificate.forward_renyi == 0.0
    assert certificate.type_ii_lower_bound == pytest.approx(0.95**2)
    distinct = evaluate_stationary_simple_hypotheses(
        _specification(), null_state=null, alternative_state=_state("shared-source", 5.0),
    )
    assert distinct.null_components[0].sample_id == distinct.alternative_components[0].sample_id == "shared-source"
    assert distinct.null_predicted_mids != distinct.alternative_predicted_mids
    assert distinct.hypotheses.null_state_fingerprint != distinct.hypotheses.alternative_state_fingerprint


def test_very_close_matching_support_control_preserves_exact_predictions():
    alternative_fraction = 0.25 + 2**-20
    result = evaluate_stationary_simple_hypotheses(
        _specification(), null_state=_state("v0"), alternative_state=_state("v1", 10 * alternative_fraction),
    )
    assert result.alternative_predicted_mids == ((alternative_fraction, 1 - alternative_fraction),)
    assert result.law_pair.mutually_absolutely_continuous
    certificate = bruno_converse_at_order(result, epsilon=0.05, order=2.0)
    assert 0 < certificate.reverse_renyi < 1e-9
    assert 0 < certificate.forward_renyi < 1e-9
    assert 0.95**2 - 1e-9 < certificate.type_ii_lower_bound < 0.95**2


def test_flux_endpoint_support_mismatch_remains_visible_and_rejects_bruno_without_smoothing():
    result = evaluate_stationary_simple_hypotheses(
        _specification(2), null_state=_state("v0", 10.0), alternative_state=_state("v1", 5.0),
    )
    assert result.null_predicted_mids == ((1.0, 0.0),)
    assert not result.law_pair.mutually_absolutely_continuous
    with pytest.raises(BrunoTheoremAssumptionError, match="mutual absolute continuity.*fixed-tracer"):
        bruno_converse_at_order(result, epsilon=0.05, order=2.0)
    assert log_likelihood_ratio(result, (0, 2)) == math.inf
    assert result.null_predicted_mids == ((1.0, 0.0),)


def test_multiple_experiments_targets_replicates_and_mass_classes_retain_declared_order():
    specification = _ordered_specification()
    result = evaluate_stationary_simple_hypotheses(
        specification, null_state=_state("v0"), alternative_state=_state("v1", 5.0),
        independent_blocks=True,
    )
    identities = tuple(
        (block.experiment_id, item.target_id, item.replicate_id)
        for block in specification.experiments for item in block.specifications
    )
    totals = tuple(item.total_count for block in specification.experiments for item in block.specifications)
    assert result.law_pair.independent
    assert result.law_pair.block_identities == identities
    assert result.law_pair.count_totals == totals == (37, 59, 11, 23, 23, 11, 59, 37)
    for source in (result.null_observation_laws, result.alternative_observation_laws):
        assert tuple((item.experiment_id, item.target_id, item.replicate_id) for item in source.components) == identities
        assert tuple(key for key, _ in source.experiment_fingerprints) == ("z-exp", "a-exp")
        for item, n in zip(source.components, totals, strict=True):
            assert item.law.n == n
            assert item.mass_classes == tuple(range(len(item.predicted_mid)))
            assert item.specification_fingerprint == specification.fingerprint
    certificate = bruno_converse_at_order(result, epsilon=0.05, order=1.5)
    assert certificate.pair.block_identities == identities
    assert certificate.reverse_renyi == observation_api.independent_product_renyi(
        result.law_pair.alternative_laws, result.law_pair.null_laws, 1.5,
    )


def test_missing_product_independence_is_rejected_before_observation_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("undeclared independence reached observation evaluation")

    monkeypatch.setattr(observation_api, "evaluate_stationary_observation_laws", forbidden)
    with pytest.raises(InputValidationError, match="independent_blocks=True"):
        evaluate_stationary_simple_hypotheses(
            _ordered_specification(), null_state=_state("v0"), alternative_state=_state("v1", 5.0),
        )


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_independence_must_be_an_explicit_bool(value):
    with pytest.raises(InputValidationError, match="explicit bool"):
        evaluate_stationary_simple_hypotheses(
            _specification(), null_state=_state("v0"), alternative_state=_state("v1", 5.0),
            independent_blocks=value,
        )


def test_both_states_are_independently_validated_without_a_biological_objective_restriction(monkeypatch):
    calls = []
    original = observation_module.prepare_highs_flux_region

    def record(model, objective_fraction):
        calls.append(objective_fraction)
        return original(model, objective_fraction)

    monkeypatch.setattr(observation_module, "prepare_highs_flux_region", record)
    # M_OUT has an explicit biological maximization objective and upper bound10.
    # These complete states have output4 and are feasible without being optimal.
    null, alternative = _state("v0", 1.0, 4.0), _state("v1", 2.0, 4.0)
    result = evaluate_stationary_simple_hypotheses(
        _specification(), null_state=null, alternative_state=alternative,
    )
    assert calls == [None, None]
    assert result.null_observation_laws.validation.valid
    assert result.alternative_observation_laws.validation.valid
    assert result.null_predicted_mids == ((0.25, 0.75),)
    assert result.alternative_predicted_mids == ((0.5, 0.5),)


@pytest.mark.parametrize("role", ["null", "alternative"])
@pytest.mark.parametrize("invalid", [
    CanonicalFluxState("incomplete", (("Z_IN", 2.5), ("A_IN", 7.5))),
    CanonicalFluxState("reordered", (("A_IN", 7.5), ("Z_IN", 2.5), ("M_OUT", 10.0))),
    CanonicalFluxState("unbalanced", (("Z_IN", 2.5), ("A_IN", 7.5), ("M_OUT", 9.0))),
    CanonicalFluxState("outside-bounds", (("Z_IN", 11.0), ("A_IN", 0.0), ("M_OUT", 11.0))),
])
def test_each_role_rejects_incomplete_reordered_or_infeasible_states_before_its_emu_evaluation(monkeypatch, role, invalid):
    evaluated = []
    original = observation_module.evaluate_stationary

    def record(plan, states):
        evaluated.extend(states)
        return original(plan, states)

    monkeypatch.setattr(observation_module, "evaluate_stationary", record)
    null = invalid if role == "null" else _state("v0")
    alternative = invalid if role == "alternative" else _state("v1", 5.0)
    with pytest.raises((AnalysisError, InputValidationError), match=f"{role} flux hypothesis"):
        evaluate_stationary_simple_hypotheses(_specification(), null_state=null, alternative_state=alternative)
    assert invalid not in evaluated


@pytest.mark.parametrize("invalid", [object(), {"Z_IN": 2.5},
    CanonicalFluxState("boolean", (("Z_IN", True), ("A_IN", 9.0), ("M_OUT", 10.0))),
    CanonicalFluxState("nonfinite", (("Z_IN", math.nan), ("A_IN", 7.5), ("M_OUT", 10.0))),
])
def test_hypotheses_require_immutable_canonical_state_records(invalid):
    with pytest.raises(InputValidationError):
        SimpleBinaryFluxHypotheses(null_state=invalid, alternative_state=_state("v1", 5.0))


def test_result_rejects_swapped_sources_and_unbound_pair_provenance():
    result = evaluate_stationary_simple_hypotheses(
        _specification(), null_state=_state("v0"), alternative_state=_state("v1", 5.0),
    )
    with pytest.raises(InputValidationError, match="null observation source"):
        replace(result, null_observation_laws=result.alternative_observation_laws)
    with pytest.raises(InputValidationError, match="bind both flux states"):
        replace(result, law_pair=replace(result.law_pair, observation_specification_fingerprint="wrong"))
    with pytest.raises(InputValidationError, match="block order"):
        replace(result, law_pair=replace(result.law_pair, block_identities=(("wrong", "O-mid", "genuine-counts"),)))


def test_source_component_provenance_cannot_diverge_from_native_source():
    result = evaluate_stationary_simple_hypotheses(
        _specification(), null_state=_state("v0"), alternative_state=_state("v1", 5.0),
    )
    malformed_source = replace(result.null_observation_laws, components=(
        replace(result.null_components[0], model_fingerprint="different-model"),
    ))
    with pytest.raises(InputValidationError, match="component provenance"):
        replace(result, null_observation_laws=malformed_source)

"""Continuous Dirichlet finite-family converse and projected-score tests."""

import math

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.observation import DirichletMIDLaw, MIDCorrectionProvenance, renyi_dirichlet
from fluxemu.testing import (
    DirichletCompositeBinaryTestingProblem,
    DirichletCompositeMIDLawFamily,
    DirichletTestingAssumptionError,
    IndependentDirichletMIDProductLaw,
    UnsupportedContinuousObservationError,
    calibrate_dirichlet_composite_score_test,
    composite_dirichlet_renyi_converse_at_order,
    composite_dirichlet_renyi_score_candidate,
    composite_dirichlet_score_bound_at_order,
    evaluate_dirichlet_composite_score_test,
    exact_dirichlet_composite_minimax,
    verified_composite_dirichlet_renyi_score,
)


CORRECTION = MIDCorrectionProvenance(
    "externally_corrected", "synthetic", "public synthetic fixture",
)


def mid_law(
    probabilities,
    precision=30.0,
    *,
    identity=("experiment", "target", "replicates"),
    replicate_count=1,
    precision_source="fixed_external",
):
    probabilities = tuple(probabilities)
    return DirichletMIDLaw(
        mass_classes=tuple(range(len(probabilities))),
        active_support=tuple(index for index, value in enumerate(probabilities) if value > 0),
        predicted_mid=probabilities,
        precision=precision,
        observation_identity=identity,
        correction=CORRECTION,
        precision_source=precision_source,
        precision_provenance="declared synthetic precision",
        replicate_count=replicate_count,
        replicate_semantics="technical_measurement_variability",
        independent_replicates=replicate_count > 1,
    )


def product(*laws):
    return IndependentDirichletMIDProductLaw(blocks=tuple(laws))


def family(members, prefix):
    members = tuple(members)
    return DirichletCompositeMIDLawFamily(
        members=members,
        member_ids=tuple(f"{prefix}-{index}" for index in range(len(members))),
    )


def problem(null, alternative):
    return DirichletCompositeBinaryTestingProblem(
        null=family((product(item) for item in null), "P"),
        alternative=family((product(item) for item in alternative), "Q"),
    )


def ordered_problem(replicate_count=2):
    return problem(
        (
            mid_law((0.7, 0.3), replicate_count=replicate_count),
            mid_law((0.65, 0.35), replicate_count=replicate_count),
        ),
        (
            mid_law((0.3, 0.7), replicate_count=replicate_count),
            mid_law((0.35, 0.65), replicate_count=replicate_count),
        ),
    )


def test_product_and_family_preserve_continuous_observation_identity():
    first = mid_law((0.2, 0.3, 0.5), identity=("e", "a", "r"), replicate_count=2)
    second = mid_law((0.4, 0.6), identity=("e", "b", "r"), replicate_count=3)
    law = product(first, second)
    assert law.block_identities == (("e", "a", "r"), ("e", "b", "r"))
    assert law.replicate_counts == (2, 3)
    assert law.rigorous_testing_suitable
    assert len(law.fingerprint) == 64
    assert len(law.sample(seed=7)) == 2
    with pytest.raises(InputValidationError, match="unique"):
        product(first, mid_law((0.4, 0.6), identity=("e", "a", "r")))


def test_family_and_problem_refuse_state_dependent_active_support():
    full = product(mid_law((0.2, 0.3, 0.5)))
    face = product(mid_law((0.2, 0.0, 0.8)))
    with pytest.raises(InputValidationError, match="state-dependent support"):
        family((full, face), "P")
    null = family((full,), "P")
    alternative = family((face,), "Q")
    with pytest.raises(InputValidationError, match="common ordered active support"):
        DirichletCompositeBinaryTestingProblem(null=null, alternative=alternative)


def test_replicate_and_multiblock_renyi_additivity():
    p1 = mid_law((0.2, 0.3, 0.5), 20, identity=("e", "a", "r"), replicate_count=3)
    q1 = mid_law((0.3, 0.25, 0.45), 28, identity=("e", "a", "r"), replicate_count=3)
    p2 = mid_law((0.6, 0.4), 12, identity=("e", "b", "r"), replicate_count=2)
    q2 = mid_law((0.45, 0.55), 16, identity=("e", "b", "r"), replicate_count=2)
    actual = product(p1, p2).renyi_divergence(product(q1, q2), 0.5)
    expected = 3 * renyi_dirichlet(p1, q1, 0.5) + 2 * renyi_dirichlet(p2, q2, 0.5)
    assert actual == pytest.approx(expected, abs=2e-14)
    assert product(p1).renyi_divergence(product(q1), 0.5) == pytest.approx(
        3 * renyi_dirichlet(p1, q1, 0.5), abs=2e-14,
    )


def test_composite_converse_uses_reverse_direction_and_declared_order():
    p = mid_law((0.2, 0.3, 0.5), 20)
    q = mid_law((0.3, 0.25, 0.45), 35)
    declared = problem((p,), (q,))
    result = composite_dirichlet_renyi_converse_at_order(
        declared, epsilon=0.05, order=1.2,
    )
    assert result.reverse_renyi == pytest.approx(renyi_dirichlet(q, p, 1.2), abs=2e-13)
    assert result.reverse_renyi != pytest.approx(renyi_dirichlet(p, q, 1.2))
    assert result.null_member_id == "P-0" and result.alternative_member_id == "Q-0"
    assert result.global_order_envelope_evaluated is False


def test_integrability_boundary_gives_infinite_directed_divergence_and_vacuous_converse():
    p = mid_law((0.2, 0.3, 0.5), 20)
    q = mid_law((0.3, 0.25, 0.45), 35)
    assert renyi_dirichlet(p, q, 2.0) == math.inf
    # Reverse direction is finite here; swap the testing roles to exercise the infinite minimum.
    result = composite_dirichlet_renyi_converse_at_order(
        problem((q,), (p,)), epsilon=0.05, order=2.0,
    )
    assert result.reverse_renyi == math.inf
    assert result.type_ii_lower_bound == 0.0


def test_ordered_family_candidate_has_verified_analytic_uniform_moments():
    declared = ordered_problem()
    candidate = composite_dirichlet_renyi_score_candidate(declared, order=0.5)
    assert (candidate.null_member_index, candidate.alternative_member_index) == (1, 1)
    assert candidate.uniform_moment_bounds_verified
    assert candidate.verification_failures == ()
    assert candidate.maximum_log_null_moment <= candidate.log_hellinger_integral
    assert candidate.maximum_log_alternative_moment <= candidate.log_hellinger_integral
    assert candidate.log_hellinger_integral == pytest.approx(
        (candidate.order - 1) * candidate.renyi, abs=2e-14,
    )
    assert verified_composite_dirichlet_renyi_score(declared, order=0.5) == candidate
    assert not candidate.finite_n_least_favourable_claimed
    assert not candidate.joint_convex_projection_claimed


def test_product_score_evaluation_and_moments_add_over_blocks_and_replicates():
    first = ordered_problem(replicate_count=3)
    candidate = composite_dirichlet_renyi_score_candidate(first, order=0.5)
    sampled = first.null.members[0].sample(seed=71)
    expected_score = sum(
        block_score.evaluate(value)
        for block_score, replicates in zip(candidate.score.block_scores, sampled, strict=True)
        for value in replicates
    )
    assert candidate.score.evaluate(sampled) == pytest.approx(expected_score, abs=1e-14)
    reference = first.null.members[0]
    single = candidate.score.block_scores[0].log_moment(reference.blocks[0], 0.5)
    assert candidate.score.log_moment(reference, 0.5) == pytest.approx(3 * single, abs=2e-14)


def test_nonordered_finite_family_exposes_failed_uniform_moment_conditions():
    null_parameters = (
        ((0.7195905564607262, 0.23859715661602354, 0.04181228692325023), 28.08735010527839),
        ((0.2184749394536289, 0.06459673950800936, 0.7169283210383618), 9.082527919351032),
        ((0.3018998051457259, 0.12159908997706506, 0.5765011048772091), 49.11005307166139),
    )
    alternative_parameters = (
        ((0.13034424964509725, 0.18524665150934197, 0.6844090988455609), 78.2943831224152),
        ((0.22583372514230593, 0.11982196352220961, 0.6543443113354845), 14.627844211252203),
        ((0.3430609917254396, 0.101073092563854, 0.5558659157107064), 11.04599688848159),
    )
    declared = problem(
        tuple(mid_law(p, k) for p, k in null_parameters),
        tuple(mid_law(p, k) for p, k in alternative_parameters),
    )
    candidate = composite_dirichlet_renyi_score_candidate(declared, order=0.5)
    assert not candidate.uniform_moment_bounds_verified
    assert any("null-side" in failure for failure in candidate.verification_failures)
    assert any("alternative-side" in failure for failure in candidate.verification_failures)
    with pytest.raises(DirichletTestingAssumptionError, match="only a candidate"):
        verified_composite_dirichlet_renyi_score(declared, order=0.5)
    with pytest.raises(DirichletTestingAssumptionError, match="lacks verified"):
        composite_dirichlet_score_bound_at_order(candidate, epsilon=0.05)


def test_analytical_projected_bound_and_type_i_simulation_control():
    declared = ordered_problem(replicate_count=2)
    candidate = verified_composite_dirichlet_renyi_score(declared, order=0.5)
    bound = composite_dirichlet_score_bound_at_order(candidate, epsilon=0.05)
    assert bound.raw_exponential_upper_bound < bound.constant_randomised_upper_bound
    assert bound.minimax_type_ii_upper_bound == bound.raw_exponential_upper_bound
    # Deterministic Monte Carlo is a sanity check on, not the proof of, Markov control.
    simulations = 30_000
    for index, null in enumerate(declared.null.members):
        generator_seed = 9300 + index
        scores = tuple(
            candidate.score.evaluate(null.sample(seed=generator_seed + draw))
            for draw in range(simulations)
        )
        alpha_hat = sum(value >= bound.threshold for value in scores) / simulations
        standard_error = math.sqrt(max(alpha_hat * (1 - alpha_hat), 1 / simulations) / simulations)
        assert alpha_hat <= 0.05 + 5 * standard_error


def test_exact_minimax_and_exact_score_cdf_procedures_refuse_continuous_space():
    declared = ordered_problem()
    candidate = verified_composite_dirichlet_renyi_score(declared, order=0.5)
    bound = composite_dirichlet_score_bound_at_order(candidate, epsilon=0.05)
    with pytest.raises(UnsupportedContinuousObservationError, match="unsupported_for_continuous"):
        exact_dirichlet_composite_minimax(declared, epsilon=0.05)
    with pytest.raises(UnsupportedContinuousObservationError, match="no implemented certified exact CDF"):
        evaluate_dirichlet_composite_score_test(bound)
    with pytest.raises(UnsupportedContinuousObservationError, match="no simplex discretisation"):
        calibrate_dirichlet_composite_score_test(candidate, epsilon=0.05)


def test_same_data_plugin_precision_refuses_rigorous_testing_quantities():
    declared = problem(
        (mid_law((0.7, 0.3), precision_source="same_data_plugin"),),
        (mid_law((0.3, 0.7), precision_source="same_data_plugin"),),
    )
    for operation in (
        lambda: composite_dirichlet_renyi_converse_at_order(
            declared, epsilon=0.05, order=1.2,
        ),
        lambda: composite_dirichlet_renyi_score_candidate(declared, order=0.5),
    ):
        with pytest.raises(DirichletTestingAssumptionError, match="same_data_plugin"):
            operation()


@pytest.mark.parametrize("order", [0, 1, 1.1, -0.5, True, math.nan, math.inf, "0.5"])
def test_candidate_orders_are_explicitly_in_zero_one_interval(order):
    with pytest.raises(InputValidationError):
        composite_dirichlet_renyi_score_candidate(ordered_problem(), order=order)

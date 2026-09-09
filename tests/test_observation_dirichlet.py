"""Dirichlet MID law, support, identity, and calibration contracts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.special import digamma, gammaln

from fluxemu.exceptions import InputValidationError, ValidationError
from fluxemu.observation import (
    DIRICHLET_NEAR_ONE_REFUSAL_RADIUS,
    DirichletMIDLaw,
    DirichletNumericalError,
    MIDCorrectionProvenance,
    diagnose_dirichlet_covariance,
    dirichlet_pairwise_log_likelihood_score,
    dirichlet_score_log_moment,
    estimate_dirichlet_precision_from_replicates,
    implied_dirichlet_precision_from_uncertainty,
    kl_dirichlet,
    renyi_dirichlet,
)


CORRECTION = MIDCorrectionProvenance(
    "externally_corrected", "synthetic-control", "public synthetic fixture",
)


def law(
    probabilities=(0.2, 0.3, 0.5),
    precision=20.0,
    *,
    precision_source="fixed_external",
    replicate_count=1,
    independent_replicates=False,
    identity=("experiment", "target", "replicate-series"),
):
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
        independent_replicates=independent_replicates,
    )


def scipy_log_beta(parameters):
    return float(np.sum(gammaln(parameters)) - gammaln(np.sum(parameters)))


def scipy_kl(left, right):
    alpha, beta = np.asarray(left.parameters), np.asarray(right.parameters)
    return float(
        scipy_log_beta(beta)
        - scipy_log_beta(alpha)
        + np.sum((alpha - beta) * (digamma(alpha) - digamma(np.sum(alpha))))
    )


def scipy_renyi(left, right, order):
    alpha, beta = np.asarray(left.parameters), np.asarray(right.parameters)
    mixed = order * alpha + (1.0 - order) * beta
    if np.any(mixed <= 0):
        return math.inf
    return float(
        (scipy_log_beta(mixed) - order * scipy_log_beta(alpha)
         - (1.0 - order) * scipy_log_beta(beta)) / (order - 1.0)
    )


def test_valid_law_retains_full_face_identity_and_is_immutable():
    observed = law((0.25, 0.0, 0.75), 40.0)
    assert observed.mean == observed.predicted_mid == (0.25, 0.0, 0.75)
    assert observed.active_support == (0, 2)
    assert observed.active_mass_classes == (0, 2)
    assert observed.structural_zero_mass_classes == (1,)
    assert observed.parameters == (10.0, 30.0)
    assert observed.kappa == 40.0
    assert observed.precision_is_count is False
    assert observed.rigorous_testing_suitable is True
    assert observed.fingerprint == law((0.25, 0.0, 0.75), 40.0).fingerprint
    assert observed.fingerprint != law((0.25, 0.0, 0.75), 41.0).fingerprint
    with pytest.raises(FrozenInstanceError):
        observed.precision = 41.0


@pytest.mark.parametrize("probabilities", [
    (), (1.0,), (0.2, 0.2), (-1e-16, 1.0), (math.nan, 1.0),
    (math.inf, 0.0), (True, False), {0.2, 0.8}, {0.2: 0, 0.8: 1},
])
def test_malformed_means_are_never_repaired(probabilities):
    with pytest.raises((InputValidationError, TypeError)):
        law(probabilities)


@pytest.mark.parametrize("precision", [0, -1, True, math.nan, math.inf, "20"])
def test_precision_must_be_positive_finite_and_is_not_a_count(precision):
    with pytest.raises(InputValidationError):
        law(precision=precision)


def test_positive_support_erasure_and_alpha_underflow_refuse():
    with pytest.raises(ValidationError, match="erased"):
        law((Fraction(1, 10**400), 1))
    tiny = math.ulp(0.0)
    with pytest.raises(DirichletNumericalError, match="alpha"):
        law((tiny, 1.0), tiny)


def test_active_support_must_equal_positive_support_without_epsilon():
    base = dict(
        mass_classes=(0, 1, 2), predicted_mid=(0.25, 0.0, 0.75), precision=20.0,
        observation_identity=("e", "t", "r"), correction=CORRECTION,
        precision_source="fixed_external", precision_provenance="source",
        replicate_count=1, replicate_semantics="technical_measurement_variability",
        independent_replicates=False,
    )
    for active in ((0, 1, 2), (0,), (2, 0), (0, 0, 2)):
        with pytest.raises(InputValidationError):
            DirichletMIDLaw(active_support=active, **base)


def test_external_correction_and_replicate_semantics_are_mandatory():
    for status in ("unknown", "uncorrected", True, None):
        with pytest.raises(InputValidationError):
            MIDCorrectionProvenance(status, "method", "source")
    with pytest.raises(InputValidationError, match="independent_replicates"):
        law(replicate_count=2)
    repeated = law(replicate_count=3, independent_replicates=True)
    assert len(repeated.sample_replicates(seed=12)) == 3
    plugin = law(precision_source="same_data_plugin")
    assert plugin.rigorous_testing_suitable is False


def test_density_matches_independent_log_beta_expression():
    observed = law((0.2, 0.3, 0.5), 20.0)
    value = (0.15, 0.25, 0.6)
    expected = -scipy_log_beta(np.asarray(observed.parameters)) + sum(
        (alpha - 1.0) * math.log(y)
        for alpha, y in zip(observed.parameters, value, strict=True)
    )
    assert observed.log_density(value) == pytest.approx(expected, abs=2e-14)
    assert observed.log_prob(value) == observed.log_density(value)


def test_observed_active_zero_and_structural_zero_violation_refuse():
    observed = law((0.25, 0.0, 0.75), 20.0)
    with pytest.raises(InputValidationError, match="observed exact zero"):
        observed.log_density((0.0, 0.0, 1.0))
    with pytest.raises(InputValidationError, match="structural-zero"):
        observed.log_density((0.2, 0.1, 0.7))
    with pytest.raises(InputValidationError, match="sum to one"):
        observed.log_density((0.2, 0.0, 0.7))


def test_mean_covariance_identities_and_simplex_constraint():
    observed = law((0.2, 0.3, 0.5), 9.0)
    expected = tuple(
        tuple(
            (p * (1 - p) if i == j else -p * observed.mean[j]) / 10.0
            for j in range(3)
        )
        for i, p in enumerate(observed.mean)
    )
    assert np.asarray(observed.covariance) == pytest.approx(np.asarray(expected), abs=1e-16)
    assert np.sum(observed.covariance, axis=0) == pytest.approx(np.zeros(3), abs=5e-18)
    assert np.sum(observed.covariance, axis=1) == pytest.approx(np.zeros(3), abs=5e-18)
    concentrated = law((0.2, 0.3, 0.5), 90.0)
    assert all(concentrated.covariance[i][i] < observed.covariance[i][i] for i in range(3))


def test_seeded_samples_stay_on_declared_face_and_reproduce():
    observed = law((0.25, 0.0, 0.75), 20.0)
    first = observed.sample(seed=91)
    assert first == observed.sample(seed=np.int64(91))
    assert first[1] == 0.0 and first[0] > 0 and first[2] > 0
    assert sum(first) == pytest.approx(1.0, abs=1e-15)
    left, right = np.random.default_rng(18), np.random.default_rng(18)
    assert [observed.sample(rng=left) for _ in range(4)] == [observed.sample(rng=right) for _ in range(4)]


def test_sampled_mean_and_covariance_match_analytic_values():
    observed = law((0.15, 0.35, 0.5), 30.0)
    generator = np.random.default_rng(20260909)
    samples = generator.dirichlet(observed.parameters, size=80_000)
    assert np.mean(samples, axis=0) == pytest.approx(observed.mean, abs=8e-4)
    assert np.cov(samples, rowvar=False) == pytest.approx(
        np.asarray(observed.covariance), rel=0.035, abs=5e-5,
    )
    assert np.max(np.abs(np.sum(samples, axis=1) - 1.0)) < 5e-16


def test_renyi_identities_direction_integrability_and_kl_limit():
    p = law((0.2, 0.3, 0.5), 20.0)
    q = law((0.3, 0.25, 0.45), 35.0)
    assert renyi_dirichlet(p, p, 0.5) == 0.0
    assert renyi_dirichlet(p, p, 1.0) == 0.0
    assert renyi_dirichlet(p, q, 0.5) == pytest.approx(
        renyi_dirichlet(q, p, 0.5), rel=3e-14,
    )
    assert renyi_dirichlet(p, q, 0.7) != pytest.approx(renyi_dirichlet(q, p, 0.7))
    assert renyi_dirichlet(p, q, 0.7) == pytest.approx(scipy_renyi(p, q, 0.7), rel=3e-13)
    assert kl_dirichlet(p, q) == pytest.approx(scipy_kl(p, q), rel=3e-12)
    assert renyi_dirichlet(p, q, 1.0 + 1e-3) == pytest.approx(
        kl_dirichlet(p, q), rel=2e-2,
    )
    assert renyi_dirichlet(p, q, 2.0) == math.inf


def test_renyi_conditioning_policy_refuses_unresolved_large_precision_case():
    p = law((0.2, 0.3, 0.5), 1_000_000.0)
    q = law((0.3, 0.25, 0.45), 35.0)
    with pytest.raises(DirichletNumericalError, match="ill-conditioned"):
        renyi_dirichlet(p, q, 1.0 + 1e-5)


def test_near_one_order_policy_is_explicit_and_does_not_snap():
    p, q = law(), law((0.25, 0.25, 0.5), 25.0)
    for order in (
        1.0 - DIRICHLET_NEAR_ONE_REFUSAL_RADIUS / 2,
        1.0 + DIRICHLET_NEAR_ONE_REFUSAL_RADIUS / 2,
    ):
        with pytest.raises(DirichletNumericalError, match="too close to one"):
            renyi_dirichlet(p, q, order)
    for order in (0, -1, True, math.nan, math.inf, "0.5"):
        with pytest.raises(InputValidationError):
            renyi_dirichlet(p, q, order)


def test_pairwise_score_and_analytic_moments_match_independent_calculations():
    p = law((0.2, 0.3, 0.5), 20.0)
    q = law((0.3, 0.25, 0.45), 35.0)
    reference = law((0.25, 0.35, 0.4), 28.0)
    value = (0.18, 0.31, 0.51)
    score = p.pairwise_score(q)
    assert score.evaluate(value) == pytest.approx(q.log_density(value) - p.log_density(value), abs=2e-14)
    assert dirichlet_pairwise_log_likelihood_score(value, p, q) == score.evaluate(value)
    for exponent in (-0.35, 0.25, 0.7):
        shifted = np.asarray(reference.parameters) + exponent * (
            np.asarray(q.parameters) - np.asarray(p.parameters)
        )
        expected = (
            exponent * (scipy_log_beta(np.asarray(p.parameters)) - scipy_log_beta(np.asarray(q.parameters)))
            + scipy_log_beta(shifted) - scipy_log_beta(np.asarray(reference.parameters))
        )
        assert score.log_moment(reference, exponent) == pytest.approx(expected, abs=5e-14)
        assert dirichlet_score_log_moment(reference, p, q, exponent) == pytest.approx(expected, abs=5e-14)


def test_nonintegrable_score_moment_is_infinite_without_clipping():
    p = law((0.2, 0.3, 0.5), 20.0)
    q = law((0.8, 0.1, 0.1), 20.0)
    assert p.pairwise_score(q).log_moment(p, -2.0) == math.inf


def test_se_and_sd_precision_diagnostic_returns_every_component_without_pooling():
    p = (0.2, 0.3, 0.5)
    kappa = 99.0
    sd = tuple(math.sqrt(value * (1 - value) / (kappa + 1)) for value in p)
    se = tuple(value / math.sqrt(4) for value in sd)
    for kind, uncertainty in (("sd", sd), ("se", se)):
        result = implied_dirichlet_precision_from_uncertainty(
            p, uncertainty, replicate_count=4, uncertainty_kind=kind,
        )
        assert tuple(item.implied_precision for item in result.components) == pytest.approx((kappa,) * 3)
        assert result.dispersion.valid_component_count == 3
        assert result.pooled_precision is None
        assert result.precision_source == "same_data_plugin"
        assert result.suitable_for_rigorous_testing is False
    with pytest.raises(InputValidationError, match="explicitly"):
        implied_dirichlet_precision_from_uncertainty(
            p, sd, replicate_count=4, uncertainty_kind="unknown",
        )


def test_inconsistent_and_invalid_component_precisions_are_exposed():
    result = implied_dirichlet_precision_from_uncertainty(
        (0.2, 0.3, 0.5), (0.01, 0.2, 0.0),
        replicate_count=3, uncertainty_kind="se",
    )
    assert [item.status for item in result.components] == ["valid", "valid", "nonfinite"]
    assert result.dispersion.maximum_to_minimum_ratio > 100
    assert result.warnings
    assert result.pooled_precision is None


def test_replicate_calibration_and_covariance_diagnostic_are_structured():
    generator = np.random.default_rng(812)
    center = (0.2, 0.3, 0.5)
    replicates = tuple(tuple(row) for row in generator.dirichlet(np.asarray(center) * 80, size=600))
    calibration = estimate_dirichlet_precision_from_replicates(
        replicates, center=center,
        replicate_semantics="technical_measurement_variability",
        independent_of_test_data=True, bootstrap_samples=40, bootstrap_seed=77,
    )
    assert calibration.estimate == pytest.approx(80, rel=0.12)
    assert calibration.method == "scalar_covariance_frobenius_method_of_moments_v1"
    assert calibration.precision_source == "independent_calibration"
    assert calibration.suitable_for_rigorous_testing is True
    assert 0 < calibration.bootstrap_p_value <= 1
    diagnostic = diagnose_dirichlet_covariance(
        replicates, center=center, precision=calibration.estimate,
        replicate_semantics="technical_measurement_variability",
    )
    assert diagnostic.replicate_count == 600
    assert len(diagnostic.empirical_covariance) == 3
    assert diagnostic.covariance_relative_frobenius_error < 0.15
    assert diagnostic.suitable_for_rigorous_testing is False


def test_same_data_and_biological_calibration_labels_do_not_overclaim():
    replicates = law(precision=30).sample_replicates(
        seed=9,
    ) + law(precision=30).sample_replicates(seed=10)
    result = estimate_dirichlet_precision_from_replicates(
        replicates,
        replicate_semantics="biological_replicate_variability",
        independent_of_test_data=False,
    )
    assert result.precision_source == "same_data_plugin"
    assert result.suitable_for_rigorous_testing is False
    assert any("biological" in warning for warning in result.warnings)
    assert any("same-data" in warning for warning in result.warnings)


def test_replicate_calibration_rejects_malformed_and_boundary_vectors_without_normalizing():
    valid = ((0.2, 0.3, 0.5), (0.21, 0.29, 0.5))
    for bad in (
        ((0.2, 0.3, 0.4), valid[1]),
        ((0.0, 0.5, 0.5), valid[1]),
        ((0.2, 0.8), valid[1]),
    ):
        with pytest.raises(InputValidationError):
            estimate_dirichlet_precision_from_replicates(
                bad, replicate_semantics="total_replicate_variability",
                independent_of_test_data=False,
            )
    with pytest.raises(InputValidationError, match="bootstrap_seed"):
        estimate_dirichlet_precision_from_replicates(
            valid, replicate_semantics="technical_measurement_variability",
            independent_of_test_data=True, bootstrap_samples=10,
        )

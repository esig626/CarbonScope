"""Reproducible independent numerical campaign for Dirichlet MID V1.

Production formulas use ``math.lgamma`` and the FluxEMU implementation.
Independent controls use SciPy special functions, mpmath at 80 digits, direct
NumPy simulation, and empirical error campaigns.  Monte Carlo checks are
diagnostics, not mathematical certificates.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import mpmath as mp
import numpy as np
import scipy
from scipy.special import digamma, gammaln

from fluxemu.observation import (
    DirichletMIDLaw,
    DirichletNumericalError,
    MIDCorrectionProvenance,
    estimate_dirichlet_precision_from_replicates,
    kl_dirichlet,
    renyi_dirichlet,
)
from fluxemu.testing import (
    DirichletCompositeBinaryTestingProblem,
    DirichletCompositeMIDLawFamily,
    IndependentDirichletMIDProductLaw,
    composite_dirichlet_renyi_score_candidate,
    composite_dirichlet_score_bound_at_order,
)


DEFAULT_SEED = 20260909
CORRECTION = MIDCorrectionProvenance(
    "externally_corrected", "synthetic-validation", "public deterministic campaign",
)


def _simplex(rng: np.random.Generator, dimension: int, shape: float) -> tuple[float, ...]:
    raw = rng.gamma(shape, 1.0, size=dimension)
    first = tuple(float(value / math.fsum(raw)) for value in raw[:-1])
    last = 1.0 - math.fsum(first)
    if last <= 0:
        return _simplex(rng, dimension, shape)
    result = (*first, last)
    for _ in range(3):
        residual = 1.0 - math.fsum(result)
        if residual == 0.0:
            return result
        result = (*result[:-1], result[-1] + residual)
    return _simplex(rng, dimension, shape)


def _law(
    probabilities,
    precision,
    *,
    identity=("validation", "target", "replicates"),
    replicate_count=1,
):
    probabilities = tuple(float(value) for value in probabilities)
    return DirichletMIDLaw(
        mass_classes=tuple(range(len(probabilities))),
        active_support=tuple(index for index, value in enumerate(probabilities) if value > 0),
        predicted_mid=probabilities,
        precision=float(precision),
        observation_identity=identity,
        correction=CORRECTION,
        precision_source="fixed_external",
        precision_provenance="public deterministic campaign input",
        replicate_count=replicate_count,
        replicate_semantics="technical_measurement_variability",
        independent_replicates=replicate_count > 1,
    )


def _log_beta_scipy(parameters) -> float:
    parameters = np.asarray(parameters, dtype=float)
    return float(np.sum(gammaln(parameters)) - gammaln(np.sum(parameters)))


def _renyi_scipy(left, right, order) -> float:
    alpha, beta = np.asarray(left.parameters), np.asarray(right.parameters)
    if order == 1:
        return _kl_scipy(left, right)
    mixed = order * alpha + (1.0 - order) * beta
    if np.any(mixed <= 0):
        return math.inf
    return (
        _log_beta_scipy(mixed)
        - order * _log_beta_scipy(alpha)
        - (1.0 - order) * _log_beta_scipy(beta)
    ) / (order - 1.0)


def _kl_scipy(left, right) -> float:
    alpha, beta = np.asarray(left.parameters), np.asarray(right.parameters)
    return float(
        _log_beta_scipy(beta)
        - _log_beta_scipy(alpha)
        + np.sum((alpha - beta) * (digamma(alpha) - digamma(np.sum(alpha))))
    )


def _renyi_mpmath(left, right, order) -> float:
    with mp.workdps(80):
        lam = mp.mpf(str(order))
        alpha = tuple(mp.mpf(repr(value)) for value in left.parameters)
        beta = tuple(mp.mpf(repr(value)) for value in right.parameters)
        mixed = tuple(lam * a + (1 - lam) * b for a, b in zip(alpha, beta, strict=True))
        if any(value <= 0 for value in mixed):
            return math.inf

        def log_beta(values):
            return mp.fsum(mp.loggamma(value) for value in values) - mp.loggamma(mp.fsum(values))

        return float(
            (log_beta(mixed) - lam * log_beta(alpha) - (1 - lam) * log_beta(beta))
            / (lam - 1)
        )


def _relative_error(actual, expected) -> float:
    return abs(actual - expected) / max(1.0, abs(expected))


def _distribution_campaign(rng, *, scale):
    configurations = (
        ((0.5, 0.5), 0.5),
        ((0.2, 0.3, 0.5), 2.0),
        ((0.05, 0.15, 0.3, 0.5), 25.0),
        ((0.01, 0.04, 0.1, 0.2, 0.25, 0.4), 250.0),
        ((0.001, 0.009, 0.04, 0.1, 0.2, 0.25, 0.4), 10_000.0),
    )
    draws = max(10_000, int(60_000 * scale))
    maximum_mean_z = 0.0
    maximum_covariance_relative_error = 0.0
    maximum_simplex_residual = 0.0
    for probabilities, precision in configurations:
        law = _law(probabilities, precision)
        samples = rng.dirichlet(law.parameters, size=draws)
        maximum_simplex_residual = max(
            maximum_simplex_residual,
            float(np.max(np.abs(np.sum(samples, axis=1) - 1.0))),
        )
        empirical_mean = np.mean(samples, axis=0)
        analytic_covariance = np.asarray(law.active_covariance)
        empirical_covariance = np.cov(samples, rowvar=False)
        standard_error = np.sqrt(np.diag(analytic_covariance) / draws)
        maximum_mean_z = max(
            maximum_mean_z,
            float(np.max(np.abs(empirical_mean - np.asarray(law.active_mean)) / standard_error)),
        )
        covariance_error = np.linalg.norm(empirical_covariance - analytic_covariance)
        covariance_scale = np.linalg.norm(analytic_covariance)
        maximum_covariance_relative_error = max(
            maximum_covariance_relative_error,
            float(covariance_error / covariance_scale),
        )
        assert np.max(np.abs(np.sum(analytic_covariance, axis=0))) < 5e-16
        assert np.max(np.abs(np.sum(analytic_covariance, axis=1))) < 5e-16
    assert maximum_mean_z < 6.0
    assert maximum_covariance_relative_error < 0.08
    assert maximum_simplex_residual < 8e-16
    base = _law((0.2, 0.3, 0.5), 2.0)
    concentrated = _law((0.2, 0.3, 0.5), 200.0)
    assert all(
        concentrated.covariance[index][index] < base.covariance[index][index]
        for index in range(3)
    )
    return {
        "configurations": len(configurations),
        "draws_per_configuration": draws,
        "total_draws": draws * len(configurations),
        "maximum_mean_standard_errors": maximum_mean_z,
        "maximum_covariance_relative_frobenius_error": maximum_covariance_relative_error,
        "maximum_sample_simplex_residual": maximum_simplex_residual,
        "increasing_precision_decreased_every_marginal_variance": True,
        "analytic_covariance_rows_and_columns_sum_to_zero": True,
    }


def _renyi_campaign(rng, *, scale):
    stress_cases = max(300, int(3_000 * scale))
    orders = (0.01, 0.2, 0.5, 0.9, 0.99999, 1.00001, 1.2, 2.0, 10.0, 50.0)
    accepted = 0
    infinite = 0
    explicit_refusals = 0
    scipy_comparisons = 0
    high_precision_comparisons = 0
    maximum_scipy_error = 0.0
    maximum_high_precision_error = 0.0
    direction_differences = 0
    reverse_numerical_refusals = 0
    half_order_numerical_refusals = 0
    for index in range(stress_cases):
        dimension = 2 + index % 11
        p = _simplex(rng, dimension, 0.2 if index % 4 == 0 else 1.5)
        q = _simplex(rng, dimension, 0.25 if index % 5 == 0 else 1.2)
        precision_p = 10 ** rng.uniform(-1.3, 7.0)
        precision_q = 10 ** rng.uniform(-1.3, 7.0)
        left, right = _law(p, precision_p), _law(q, precision_q)
        order = orders[index % len(orders)]
        expected = _renyi_scipy(left, right, order)
        try:
            actual = renyi_dirichlet(left, right, order)
        except DirichletNumericalError:
            explicit_refusals += 1
            continue
        accepted += 1
        if expected == math.inf:
            assert actual == math.inf
            infinite += 1
            continue
        assert math.isfinite(actual) and actual >= 0
        error = _relative_error(actual, expected)
        maximum_scipy_error = max(maximum_scipy_error, error)
        scipy_comparisons += 1
        assert error < 2e-7, {
            "case": index,
            "order": order,
            "actual": actual,
            "scipy": expected,
            "relative_error": error,
            "left_parameters": left.parameters,
            "right_parameters": right.parameters,
        }
        if index % 50 == 0:
            independent = _renyi_mpmath(left, right, order)
            high_error = _relative_error(actual, independent)
            maximum_high_precision_error = max(maximum_high_precision_error, high_error)
            high_precision_comparisons += 1
            assert high_error < 3e-7
        if index < 100 and order < 1 and order != 0.5:
            try:
                reverse = renyi_dirichlet(right, left, order)
            except DirichletNumericalError:
                reverse_numerical_refusals += 1
                reverse = None
            if reverse is not None and not math.isclose(
                actual, reverse, rel_tol=1e-9, abs_tol=1e-11,
            ):
                direction_differences += 1
        assert renyi_dirichlet(left, left, order) == 0.0
        try:
            half_forward = renyi_dirichlet(left, right, 0.5)
            half_reverse = renyi_dirichlet(right, left, 0.5)
        except DirichletNumericalError:
            half_order_numerical_refusals += 1
        else:
            assert math.isclose(
                half_forward, half_reverse, rel_tol=2e-10, abs_tol=2e-10,
            )
    assert direction_differences > 0

    near_one_refusals = 0
    control_p, control_q = _law((0.2, 0.3, 0.5), 20), _law((0.3, 0.25, 0.45), 35)
    for order in (1 - 1e-9, 1 + 1e-9):
        try:
            renyi_dirichlet(control_p, control_q, order)
        except DirichletNumericalError:
            near_one_refusals += 1
    assert near_one_refusals == 2
    kl = kl_dirichlet(control_p, control_q)
    assert _relative_error(kl, _kl_scipy(control_p, control_q)) < 5e-12
    near_limit_orders = (0.999, 1.001)
    near_limits = tuple(
        renyi_dirichlet(control_p, control_q, order) for order in near_limit_orders
    )
    assert all(_relative_error(value, kl) < 0.02 for value in near_limits)
    assert renyi_dirichlet(control_p, control_q, 2.0) == math.inf

    repeated_p = _law((0.6, 0.4), 25, replicate_count=7)
    repeated_q = _law((0.4, 0.6), 25, replicate_count=7)
    product_p = IndependentDirichletMIDProductLaw(blocks=(repeated_p,))
    product_q = IndependentDirichletMIDProductLaw(blocks=(repeated_q,))
    assert math.isclose(
        product_p.renyi_divergence(product_q, 0.5),
        7 * renyi_dirichlet(repeated_p, repeated_q, 0.5),
        rel_tol=2e-14,
    )
    return {
        "stress_cases": stress_cases,
        "orders": list(orders),
        "dimensions": [2, 12],
        "precision_range": [0.05, 10_000_000.0],
        "accepted": accepted,
        "infinite_integrability_results": infinite,
        "explicit_numerical_refusals": explicit_refusals,
        "reverse_numerical_refusals": reverse_numerical_refusals,
        "half_order_numerical_refusals": half_order_numerical_refusals,
        "scipy_comparisons": scipy_comparisons,
        "high_precision_mpmath_comparisons": high_precision_comparisons,
        "maximum_relative_error_vs_scipy": maximum_scipy_error,
        "maximum_relative_error_vs_mpmath": maximum_high_precision_error,
        "asymmetric_direction_controls": direction_differences,
        "near_one_explicit_refusals": near_one_refusals,
        "near_one_accepted_orders": list(near_limit_orders),
        "near_one_accepted_limit_controls": len(near_limits),
        "replicate_additivity_verified": True,
        "integrability_boundary_returned_infinity": True,
    }


def _score_moment_scipy(reference, null, alternative, exponent):
    gamma = np.asarray(reference.parameters)
    difference = np.asarray(alternative.parameters) - np.asarray(null.parameters)
    shifted = gamma + exponent * difference
    if np.any(shifted <= 0):
        return math.inf
    constant = _log_beta_scipy(null.parameters) - _log_beta_scipy(alternative.parameters)
    return exponent * constant + _log_beta_scipy(shifted) - _log_beta_scipy(gamma)


def _score_campaign(rng, *, scale):
    stress_cases = max(200, int(1_500 * scale))
    exponents = (-0.7, -0.2, 0.2, 0.5, 1.1)
    finite = 0
    nonintegrable = 0
    maximum_error = 0.0
    for index in range(stress_cases):
        dimension = 2 + index % 9
        p = _law(_simplex(rng, dimension, 0.5), 10 ** rng.uniform(-0.5, 4))
        q = _law(_simplex(rng, dimension, 0.5), 10 ** rng.uniform(-0.5, 4))
        reference = _law(_simplex(rng, dimension, 0.8), 10 ** rng.uniform(-0.5, 4))
        exponent = exponents[index % len(exponents)]
        expected = _score_moment_scipy(reference, p, q, exponent)
        actual = p.pairwise_score(q).log_moment(reference, exponent)
        if expected == math.inf:
            assert actual == math.inf
            nonintegrable += 1
        else:
            error = _relative_error(actual, expected)
            maximum_error = max(maximum_error, error)
            assert error < 3e-8
            finite += 1

    monte_carlo_cases = (
        ((0.2, 0.3, 0.5), 25, (0.25, 0.3, 0.45), 30, 0.25),
        ((0.4, 0.6), 12, (0.5, 0.5), 18, -0.2),
        ((0.1, 0.2, 0.3, 0.4), 80, (0.12, 0.18, 0.31, 0.39), 70, 0.5),
    )
    draws = max(30_000, int(150_000 * scale))
    maximum_monte_carlo_z = 0.0
    for p, kp, q, kq, exponent in monte_carlo_cases:
        null, alternative = _law(p, kp), _law(q, kq)
        score = null.pairwise_score(alternative)
        samples = rng.dirichlet(null.parameters, size=draws)
        values = exponent * (
            score.constant + np.sum(np.log(samples) * np.asarray(score.weights), axis=1)
        )
        pivot = float(np.max(values))
        weights = np.exp(values - pivot)
        estimate = pivot + math.log(float(np.mean(weights)))
        standard_error = float(np.std(weights, ddof=1) / math.sqrt(draws) / np.mean(weights))
        expected = score.log_moment(null, exponent)
        z = abs(estimate - expected) / max(standard_error, 1e-15)
        maximum_monte_carlo_z = max(maximum_monte_carlo_z, z)
        assert abs(estimate - expected) < max(0.015, 7 * standard_error)
    return {
        "analytic_stress_cases": stress_cases,
        "finite_scipy_comparisons": finite,
        "nonintegrable_moments": nonintegrable,
        "maximum_relative_error_vs_scipy": maximum_error,
        "monte_carlo_cases": len(monte_carlo_cases),
        "draws_per_monte_carlo_case": draws,
        "maximum_monte_carlo_standard_errors": maximum_monte_carlo_z,
    }


def _product(*laws):
    return IndependentDirichletMIDProductLaw(blocks=tuple(laws))


def _family(members, prefix):
    members = tuple(members)
    return DirichletCompositeMIDLawFamily(
        members=members,
        member_ids=tuple(f"{prefix}-{index}" for index in range(len(members))),
    )


def _problem(null, alternative):
    return DirichletCompositeBinaryTestingProblem(
        null=_family(null, "P"), alternative=_family(alternative, "Q"),
    )


def _simulated_scores(member, candidate, draws, rng):
    result = np.zeros(draws)
    for block, score in zip(member.blocks, candidate.score.block_scores, strict=True):
        samples = rng.dirichlet(block.parameters, size=draws * block.replicate_count)
        samples = samples.reshape(draws, block.replicate_count, len(block.parameters))
        result += block.replicate_count * score.constant
        result += np.sum(np.log(samples) * np.asarray(score.weights), axis=(1, 2))
    return result


def _wilson(successes, count, z=1.96):
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z / denominator * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return max(0.0, center - radius), min(1.0, center + radius)


def _type_i_campaign(rng, *, scale):
    cases = []
    # Ordered one-block family with two conditionally independent replicates.
    cases.append((
        "ordered-one-block",
        _problem(
            (_product(_law((0.7, 0.3), 30, replicate_count=2)),
             _product(_law((0.65, 0.35), 30, replicate_count=2))),
            (_product(_law((0.3, 0.7), 30, replicate_count=2)),
             _product(_law((0.35, 0.65), 30, replicate_count=2))),
        ),
    ))
    cases.append((
        "singleton-one-block",
        _problem(
            (_product(_law((0.7, 0.3), 30)),),
            (_product(_law((0.3, 0.7), 30)),),
        ),
    ))
    cases.append((
        "ordered-two-block",
        _problem(
            (
                _product(
                    _law((0.65, 0.35), 24, identity=("e", "a", "r"), replicate_count=2),
                    _law((0.25, 0.3, 0.45), 35, identity=("e", "b", "r")),
                ),
                _product(
                    _law((0.6, 0.4), 24, identity=("e", "a", "r"), replicate_count=2),
                    _law((0.27, 0.3, 0.43), 35, identity=("e", "b", "r")),
                ),
            ),
            (
                _product(
                    _law((0.35, 0.65), 24, identity=("e", "a", "r"), replicate_count=2),
                    _law((0.45, 0.3, 0.25), 35, identity=("e", "b", "r")),
                ),
                _product(
                    _law((0.4, 0.6), 24, identity=("e", "a", "r"), replicate_count=2),
                    _law((0.43, 0.3, 0.27), 35, identity=("e", "b", "r")),
                ),
            ),
        ),
    ))
    draws = max(20_000, int(80_000 * scale))
    epsilon = 0.05
    records = []
    total_null_members = 0
    for name, problem in cases:
        candidate = composite_dirichlet_renyi_score_candidate(problem, order=0.5)
        assert candidate.uniform_moment_bounds_verified, candidate.verification_failures
        bound = composite_dirichlet_score_bound_at_order(candidate, epsilon=epsilon)
        null_results = []
        for member_id, member in zip(problem.null.member_ids, problem.null.members, strict=True):
            scores = _simulated_scores(member, candidate, draws, rng)
            rejections = int(np.sum(scores >= bound.threshold))
            rate = rejections / draws
            interval = _wilson(rejections, draws)
            standard_error = math.sqrt(max(rate * (1 - rate), 1 / draws) / draws)
            assert rate <= epsilon + 6 * standard_error
            null_results.append({
                "member_id": member_id,
                "rejections": rejections,
                "draws": draws,
                "empirical_type_i": rate,
                "wilson_95_interval": list(interval),
            })
            total_null_members += 1
        alternative_results = []
        for member_id, member in zip(
            problem.alternative.member_ids, problem.alternative.members, strict=True,
        ):
            scores = _simulated_scores(member, candidate, draws, rng)
            failures = int(np.sum(scores < bound.threshold))
            rate = failures / draws
            standard_error = math.sqrt(max(rate * (1 - rate), 1 / draws) / draws)
            assert rate <= bound.raw_exponential_upper_bound + 6 * standard_error
            alternative_results.append({
                "member_id": member_id,
                "empirical_type_ii": rate,
                "draws": draws,
            })
        records.append({
            "name": name,
            "order": 0.5,
            "epsilon": epsilon,
            "threshold": bound.threshold,
            "analytical_deterministic_score_type_ii_upper_bound": (
                bound.raw_exponential_upper_bound
            ),
            "analytical_projected_type_ii_upper_bound": bound.minimax_type_ii_upper_bound,
            "null_members": null_results,
            "alternative_members": alternative_results,
        })
    return {
        "certified_projected_test_cases": len(cases),
        "all_represented_null_members_simulated": total_null_members,
        "draws_per_member": draws,
        "campaigns": records,
        "interpretation": "Monte Carlo sanity check only; analytic moments provide the guarantee",
    }


def _diagnostic_campaign(rng, *, scale):
    count = max(400, int(1_200 * scale))
    center = (0.2, 0.3, 0.5)
    true_data = tuple(
        tuple(float(value) for value in row)
        for row in rng.dirichlet(np.asarray(center) * 80.0, size=count)
    )
    true_fit = estimate_dirichlet_precision_from_replicates(
        true_data,
        center=center,
        replicate_semantics="technical_measurement_variability",
        independent_of_test_data=True,
        bootstrap_samples=max(40, int(150 * scale)),
        bootstrap_seed=991,
    )
    # An anisotropic logistic-normal composition deliberately violates the
    # scalar Dirichlet covariance shape, especially its fixed negative
    # cross-covariance ratios.
    logits = rng.multivariate_normal(
        np.log(np.asarray(center)),
        np.asarray(((0.55, 0.0, 0.0), (0.0, 0.015, 0.0), (0.0, 0.0, 0.08))),
        size=count,
    )
    exponentials = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    logistic = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    non_dirichlet_data = tuple(tuple(float(value) for value in row) for row in logistic)
    non_fit = estimate_dirichlet_precision_from_replicates(
        non_dirichlet_data,
        center=None,
        replicate_semantics="total_replicate_variability",
        independent_of_test_data=False,
        bootstrap_samples=max(40, int(150 * scale)),
        bootstrap_seed=992,
    )
    assert true_fit.estimate is not None
    assert abs(true_fit.estimate - 80.0) / 80.0 < 0.15
    assert true_fit.covariance_relative_frobenius_error is not None
    assert non_fit.covariance_relative_frobenius_error is not None
    assert non_fit.covariance_relative_frobenius_error > 2 * true_fit.covariance_relative_frobenius_error
    assert non_fit.suitable_for_rigorous_testing is False
    assert any("cannot be separated" in warning for warning in non_fit.warnings)
    return {
        "replicates_per_dataset": count,
        "true_dirichlet": {
            "generating_precision": 80.0,
            "estimated_precision": true_fit.estimate,
            "covariance_relative_frobenius_error": true_fit.covariance_relative_frobenius_error,
            "bootstrap_p_value": true_fit.bootstrap_p_value,
            "suitable_for_rigorous_testing_when_independent": true_fit.suitable_for_rigorous_testing,
        },
        "deliberately_non_dirichlet_logistic_normal": {
            "estimated_precision": non_fit.estimate,
            "covariance_relative_frobenius_error": non_fit.covariance_relative_frobenius_error,
            "bootstrap_p_value": non_fit.bootstrap_p_value,
            "suitable_for_rigorous_testing": non_fit.suitable_for_rigorous_testing,
            "warnings": list(non_fit.warnings),
        },
        "non_dirichlet_covariance_error_exceeded_true_dirichlet": True,
    }


def _source_hashes(root: Path):
    paths = (
        root / "src/fluxemu/observation/dirichlet.py",
        root / "src/fluxemu/testing/dirichlet_composite.py",
        root / "src/fluxemu/observation/dirichlet_stationary.py",
        root / "src/fluxemu/testing/dirichlet_stationary.py",
    )
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def run_campaign(*, seed=DEFAULT_SEED, scale=1.0):
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = np.random.default_rng(seed)
    root = Path(__file__).resolve().parents[2]
    return {
        "status": "passed",
        "seed": seed,
        "scale": scale,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mp.__version__,
        "source_sha256": _source_hashes(root),
        "distribution_identities": _distribution_campaign(rng, scale=scale),
        "renyi_identities_and_stress": _renyi_campaign(rng, scale=scale),
        "score_moments": _score_campaign(rng, scale=scale),
        "type_i_simulation": _type_i_campaign(rng, scale=scale),
        "model_diagnostics": _diagnostic_campaign(rng, scale=scale),
        "claims": {
            "monte_carlo_is_proof": False,
            "dirichlet_is_universally_adequate": False,
            "continuous_flux_family_solved": False,
            "exact_continuous_minimax_computed": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    result = run_campaign(seed=arguments.seed, scale=arguments.scale)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(arguments.output),
        "renyi_stress_cases": result["renyi_identities_and_stress"]["stress_cases"],
        "score_stress_cases": result["score_moments"]["analytic_stress_cases"],
        "type_i_null_members": result["type_i_simulation"]["all_represented_null_members_simulated"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

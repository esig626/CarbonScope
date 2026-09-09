"""Independent finite-control gates for production composite testing.

The Decimal vertex oracle does not import the runtime testing solver, and its
PMF construction uses integer factorials rather than runtime log PMFs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem, CompositeMIDLawFamily, IndependentMIDProductLaw,
    NumericalLimitError, exact_finite_composite_minimax,
)


_ORACLE_PATH = Path(__file__).resolve().parents[1] / 'tools/composite_testing_validation/oracle.py'
_SPEC = importlib.util.spec_from_file_location('composite_validation_oracle', _ORACLE_PATH)
oracle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = oracle
_SPEC.loader.exec_module(oracle)


def problem(null, alternative, n=1):
    def family(rows, prefix):
        return CompositeMIDLawFamily(
            members=tuple(IndependentMIDProductLaw(blocks=(MultinomialMIDLaw(n, tuple(p)),)) for p in rows),
            member_ids=tuple(f'{prefix}{i}' for i in range(len(rows))))
    return CompositeBinaryTestingProblem(null=family(null, 'P'), alternative=family(alternative, 'Q'))


def compare(case, epsilon, *, vertices=False):
    outcomes, p, q = oracle.enumerate_masses(case)
    reference = oracle.solve(p, q, epsilon, vertices=vertices)
    result = exact_finite_composite_minimax(case, epsilon=epsilon)
    indices = {y: i for i, y in enumerate(result.outcomes)}
    phi = tuple(result.rejection_probabilities[indices[y]] for y in outcomes)
    alpha, beta = oracle.errors(p, q, phi)
    assert all(0 <= x <= 1 for x in phi)
    assert float(max(alpha)) <= epsilon * (1 + 5e-9)
    assert result.null_type_i_errors == pytest.approx(tuple(map(float, alpha)), abs=2e-12)
    assert result.alternative_type_ii_errors == pytest.approx(tuple(map(float, beta)), abs=2e-12)
    assert result.minimax_type_ii_error == pytest.approx(reference.beta, abs=2e-9)
    assert result.minimax_type_ii_error == pytest.approx(float(max(beta)), abs=2e-12)
    assert result.solver_objective == pytest.approx(result.epigraph_variable, abs=result.numerical_tolerance)
    assert result.epigraph_variable == pytest.approx(result.minimax_type_ii_error, abs=result.numerical_tolerance)
    assert result.dual_lower_bound <= result.minimax_type_ii_error + result.numerical_tolerance
    return result.minimax_type_ii_error


@pytest.mark.parametrize('n', [1, 2, 4, 6])
@pytest.mark.parametrize('epsilon', [.01, .2, .8])
def test_singleton_recovers_independent_randomised_neyman_pearson(n, epsilon):
    compare(problem(((.6, .3, .1),), ((.2, .3, .5),), n), epsilon)


@pytest.mark.parametrize('epsilon', [1e-8, .05, .9])
def test_identical_laws_have_one_minus_budget(epsilon):
    assert compare(problem(((.7, .3),), ((.7, .3),), 3), epsilon) == pytest.approx(1 - epsilon, abs=2e-9)


@pytest.mark.parametrize('epsilon', [.01, .5])
def test_disjoint_support_has_zero_type_ii(epsilon):
    assert compare(problem(((1., 0., 0.),), ((0., .4, .6),)), epsilon) == pytest.approx(0., abs=1e-12)


def test_structural_partial_support_matches_analytical_value():
    assert compare(problem(((1., 0.),), ((.3, .7),)), .2) == pytest.approx(.24, abs=1e-12)


def test_duplicates_and_class_permutations_preserve_value():
    p, q = ((.9, .1), (.8, .2)), ((.1, .9), (.2, .8))
    expected = compare(problem(p, q), .1, vertices=True)
    assert expected == pytest.approx(.6, abs=1e-12)
    assert compare(problem(p[::-1], q[::-1]), .1, vertices=True) == pytest.approx(expected)
    assert compare(problem(p + p[:1], q + q[:1]), .1, vertices=True) == pytest.approx(expected)


def test_increasing_budget_cannot_increase_minimax_error():
    case = problem(((.9, .1), (.8, .2)), ((.1, .9), (.2, .8)))
    values = [compare(case, epsilon, vertices=True) for epsilon in (.01, .05, .1, .4, .9)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_enlarging_either_class_cannot_reduce_minimax_error():
    p, q = ((.9, .1),), ((.1, .9),)
    base = compare(problem(p, q), .1)
    assert compare(problem(p + ((.8, .2),), q), .1, vertices=True) >= base - 1e-12
    assert compare(problem(p, q + ((.2, .8),)), .1, vertices=True) >= base - 1e-12


@pytest.mark.parametrize('seed', [41, 42, 43, 44])
def test_decimal_vertices_and_independent_dual_agree(seed):
    rng = np.random.default_rng(seed)
    case = problem(rng.dirichlet(np.ones(3), 2), rng.dirichlet(np.ones(3), 2))
    _, p, q = oracle.enumerate_masses(case)
    vertex = oracle.vertex_minimax(p, q, .15)
    dual = oracle.dual_minimax(p, q, .15)
    assert dual.beta == pytest.approx(vertex.beta, abs=2e-10)
    compare(case, .15, vertices=True)


def test_independent_mass_enumeration_preserves_tiny_positive_support():
    case = problem(((1e-200, 1.),), ((2e-200, 1.),), n=2)
    _, p, q = oracle.enumerate_masses(case)
    assert min(p[0]) > 0
    assert min(q[0]) > 0
    assert float(min(p[0])) == 0  # Decimal reference survives float underflow.


@pytest.mark.parametrize('k,n,accepted', [(2, 39, True), (2, 40, False), (3, 25, True), (3, 26, False)])
def test_uniform_count_coefficient_resolution_boundaries(k, n, accepted):
    uniform = (1. / k,) * k
    case = problem((uniform,), (uniform,), n)
    if accepted:
        assert compare(case, .05) == pytest.approx(.95, abs=2e-12)
    else:
        with pytest.raises(NumericalLimitError, match='matrix resolution'):
            exact_finite_composite_minimax(case, epsilon=.05)


@pytest.mark.parametrize('null,alternative,epsilon', [
    ((2.4584366475093418e-14, .9999999999999755),
     (7.583662395957528e-10, .9999999992416337), .39542481278532665),
    ((1.6847478613508015e-18, 1.),
     (5.089931857333033e-10, .9999999994910068), .6985446913288124),
])
def test_real_baseline_accepted_np_gap_exceeds_declared_resolution(null, alternative, epsilon):
    # Unchanged main accepted both and lost approximately 7.58e-10 / 5.09e-10
    # of power, exceeding its declared 5e-10 objective-resolution tolerance.
    case = problem((null,), (alternative,))
    _, p, q = oracle.enumerate_masses(case)
    reference = oracle.neyman_pearson(p[0], q[0], epsilon)
    try:
        result = exact_finite_composite_minimax(case, epsilon=epsilon)
    except NumericalLimitError:
        return  # The positive coefficients must be retained or refused.
    assert result.minimax_type_ii_error == pytest.approx(reference.beta, abs=5e-10)

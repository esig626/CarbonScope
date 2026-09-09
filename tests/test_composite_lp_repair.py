"""Adversarial LP results must be checked beyond the solver success flag."""
from types import SimpleNamespace

import numpy as np
import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem, CompositeMIDLawFamily,
    CompositeOptimizationError, IndependentMIDProductLaw, NumericalLimitError,
    exact_finite_composite_minimax,
)


def problem(p=(0.5, 0.5), q=(0.5, 0.5), n=1):
    def family(probabilities):
        return CompositeMIDLawFamily(members=(IndependentMIDProductLaw(
            blocks=(MultinomialMIDLaw(n, probabilities),)),))
    return CompositeBinaryTestingProblem(null=family(p), alternative=family(q))


def test_lp_rejects_success_with_inconsistent_objective(monkeypatch):
    import scipy.optimize
    original = scipy.optimize.linprog
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.fun += 0.05
        return result
    monkeypatch.setattr(scipy.optimize, 'linprog', corrupt)
    with pytest.raises(CompositeOptimizationError, match='objective'):
        exact_finite_composite_minimax(problem(), epsilon=0.1)


def test_lp_rejects_feasible_suboptimal_optimal_status(monkeypatch):
    import scipy.optimize
    original = scipy.optimize.linprog
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        # Feasible test with perfectly consistent epigraph/objective, yet beta
        # 0.95 is strictly above the independently known optimum 0.9.
        result.x = np.array([0.05, 0.05, 0.95])
        result.fun = 0.95
        return result
    monkeypatch.setattr(scipy.optimize, 'linprog', corrupt)
    with pytest.raises(CompositeOptimizationError, match='dual|optimal|gap|consistency'):
        exact_finite_composite_minimax(problem(), epsilon=0.1)


def test_lp_refuses_positive_probability_below_solver_matrix_resolution():
    with pytest.raises(NumericalLimitError, match='coefficient|resolution'):
        exact_finite_composite_minimax(problem(q=(1e-13, 1.0-1e-13)), epsilon=0.1)


@pytest.mark.parametrize('bad_phi', [-1e-12, 1.0+1e-12])
def test_lp_does_not_clip_invalid_rejection_probabilities(monkeypatch, bad_phi):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **kw:
                        SimpleNamespace(success=True, x=np.array([bad_phi, 0.1, 0.9]), fun=0.9))
    with pytest.raises(CompositeOptimizationError, match='rejection probabilities'):
        exact_finite_composite_minimax(problem(), epsilon=0.1)


@pytest.mark.parametrize('bad_x', [np.array([0.1, 0.1, 0.9, 0.9]), np.array([[0.1], [0.1], [0.9]])])
def test_malformed_solver_vector_is_an_explicit_optimization_refusal(monkeypatch, bad_x):
    import scipy.optimize
    original = scipy.optimize.linprog
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result.x = bad_x
        return result
    monkeypatch.setattr(scipy.optimize, 'linprog', corrupt)
    with pytest.raises(CompositeOptimizationError):
        exact_finite_composite_minimax(problem(), epsilon=0.1)

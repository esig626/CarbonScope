"""Numerical safeguards for exact finite-composite minimax LP evaluation."""

from __future__ import annotations

import pytest

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    MIN_EXACT_COMPOSITE_EPSILON,
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    IndependentMIDProductLaw,
    NumericalLimitError,
    exact_finite_composite_minimax,
)


IDENTITY = (("e", "target", "counts"),)


def _member(probabilities):
    return IndependentMIDProductLaw(
        blocks=(MultinomialMIDLaw(1, probabilities),),
        block_identities=IDENTITY,
    )


def _problem(null, alternative):
    return CompositeBinaryTestingProblem(
        null=CompositeMIDLawFamily(members=(_member(null),), member_ids=("P",)),
        alternative=CompositeMIDLawFamily(
            members=(_member(alternative),), member_ids=("Q",),
        ),
    )


def test_scaled_lp_respects_small_type_i_budget_in_rare_event_problem():
    # Rejecting the rare first null outcome with probability 0.1 uses exactly
    # epsilon=1e-8 while gaining 0.1 power under a point-mass alternative.
    problem = _problem((1e-7, 1.0 - 1e-7), (1.0, 0.0))
    result = exact_finite_composite_minimax(problem, epsilon=1e-8)
    assert result.worst_type_i_error == pytest.approx(1e-8, rel=5e-9, abs=0.0)
    assert result.minimax_type_ii_error == pytest.approx(0.9, abs=2e-10)
    assert result.randomised


def test_budget_below_lp_numerical_floor_fails_without_substitution():
    problem = _problem((0.5, 0.5), (0.25, 0.75))
    assert MIN_EXACT_COMPOSITE_EPSILON == 1e-12
    with pytest.raises(NumericalLimitError, match="LP numerical floor.*no budget substitution"):
        exact_finite_composite_minimax(
            problem, epsilon=MIN_EXACT_COMPOSITE_EPSILON / 10,
        )

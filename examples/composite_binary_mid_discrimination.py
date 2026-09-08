"""Compare exact minimax and Rényi-based finite composite MID testing."""

from __future__ import annotations

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    IndependentMIDProductLaw,
    calibrate_composite_score_test,
    composite_renyi_converse_at_order,
    composite_score_bound_at_order,
    exact_finite_composite_minimax,
    verified_composite_renyi_score,
)


TYPE_I_BUDGET = 0.05
BLOCK_IDS = (("experiment", "signal-mid", "counts"), ("experiment", "control-mid", "counts"))


def member(signal_mid):
    return IndependentMIDProductLaw(
        blocks=(
            MultinomialMIDLaw(4, signal_mid),
            MultinomialMIDLaw(2, (0.5, 0.5)),
        ),
        block_identities=BLOCK_IDS,
    )


problem = CompositeBinaryTestingProblem(
    null=CompositeMIDLawFamily(
        members=(member((0.8, 0.2)), member((0.7, 0.3))),
        member_ids=("H0-mechanism-a", "H0-mechanism-b"),
    ),
    alternative=CompositeMIDLawFamily(
        members=(member((0.3, 0.7)), member((0.2, 0.8))),
        member_ids=("H1-mechanism-a", "H1-mechanism-b"),
    ),
)

converse = composite_renyi_converse_at_order(
    problem, epsilon=TYPE_I_BUDGET, order=2.0,
)
exact = exact_finite_composite_minimax(problem, epsilon=TYPE_I_BUDGET)
candidate = verified_composite_renyi_score(problem, order=0.5)
bound = composite_score_bound_at_order(candidate, epsilon=TYPE_I_BUDGET)
calibrated = calibrate_composite_score_test(
    candidate, epsilon=TYPE_I_BUDGET,
)

print("Finite composite MID-product discrimination")
print(f"H0 members: {problem.null.member_ids}")
print(f"H1 members: {problem.alternative.member_ids}")
print(f"joint block IDs: {problem.null.members[0].block_identities}")
print(f"Type-I budget: {TYPE_I_BUDGET:g}")
print(
    "order-2 composite Type-II lower bound: "
    f"{converse.type_ii_lower_bound:.9g}"
)
print(
    "exact randomised minimax Type-II error: "
    f"{exact.minimax_type_ii_error:.9g}"
)
print(
    "verified order-1/2 candidate score pair: "
    f"{candidate.null_member_id} vs {candidate.alternative_member_id}"
)
print(
    "analytical minimax Type-II upper bound: "
    f"{bound.minimax_type_ii_upper_bound:.9g}"
)
print(
    "calibrated score-family Type-II error: "
    f"{calibrated.worst_type_ii_error:.9g}"
)
print(
    "calibrated worst Type-I error: "
    f"{calibrated.worst_type_i_error:.9g}"
)
print(
    "The finite-family Rényi minimum is a candidate score construction. "
    "It is neither a joint convex-class projection nor a finite-n "
    "least-favourable-pair claim."
)

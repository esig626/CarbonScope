"""Compare exact minimax and Rényi-based finite composite MID testing."""

from __future__ import annotations

from fluxemu.observation import MultinomialMIDLaw
from fluxemu.testing import (
    CompositeBinaryTestingProblem,
    CompositeMIDLawFamily,
    calibrate_composite_projected_test,
    composite_renyi_converse_at_order,
    exact_finite_composite_minimax,
    verified_composite_renyi_projection,
)


COUNT_TOTAL = 4
TYPE_I_BUDGET = 0.05


problem = CompositeBinaryTestingProblem(
    null=CompositeMIDLawFamily(
        members=(
            MultinomialMIDLaw(COUNT_TOTAL, (0.8, 0.2)),
            MultinomialMIDLaw(COUNT_TOTAL, (0.7, 0.3)),
        ),
        member_ids=("H0-mechanism-a", "H0-mechanism-b"),
    ),
    alternative=CompositeMIDLawFamily(
        members=(
            MultinomialMIDLaw(COUNT_TOTAL, (0.3, 0.7)),
            MultinomialMIDLaw(COUNT_TOTAL, (0.2, 0.8)),
        ),
        member_ids=("H1-mechanism-a", "H1-mechanism-b"),
    ),
)

converse = composite_renyi_converse_at_order(
    problem, epsilon=TYPE_I_BUDGET, order=2.0,
)
exact = exact_finite_composite_minimax(problem, epsilon=TYPE_I_BUDGET)
projection = verified_composite_renyi_projection(problem, order=0.5)
calibrated = calibrate_composite_projected_test(
    projection, epsilon=TYPE_I_BUDGET,
)

print("Finite composite MID discrimination")
print(f"H0 members: {problem.null.member_ids}")
print(f"H1 members: {problem.alternative.member_ids}")
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
    "verified order-1/2 score pair: "
    f"{projection.null_member_id} vs {projection.alternative_member_id}"
)
print(
    "calibrated projected Type-II error: "
    f"{calibrated.worst_type_ii_error:.9g}"
)
print(
    "calibrated worst Type-I error: "
    f"{calibrated.worst_type_i_error:.9g}"
)
print(
    "The Rényi-minimising pair is a verified composite score construction; "
    "it is not labelled as a finite-n least-favourable pair."
)

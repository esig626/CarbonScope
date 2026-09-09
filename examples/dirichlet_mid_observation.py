"""Continuous corrected-MID law, diagnostics, and projected finite testing.

From a source checkout with the testing extra installed::

    python examples/dirichlet_mid_observation.py

The numbers are synthetic. Dirichlet precision is never a count, and the
finite represented families do not exhaust a continuous flux family.
"""

from __future__ import annotations

import numpy as np

from fluxemu.observation import (
    DirichletMIDLaw,
    MIDCorrectionProvenance,
    estimate_dirichlet_precision_from_replicates,
    implied_dirichlet_precision_from_uncertainty,
)
from fluxemu.testing import (
    DirichletCompositeBinaryTestingProblem,
    DirichletCompositeMIDLawFamily,
    IndependentDirichletMIDProductLaw,
    composite_dirichlet_renyi_converse_at_order,
    composite_dirichlet_score_bound_at_order,
    verified_composite_dirichlet_renyi_score,
)


TYPE_I_BUDGET = 0.05
CORRECTION = MIDCorrectionProvenance(
    status="externally_corrected",
    method="synthetic-example-method",
    provenance="examples/dirichlet_mid_observation.py",
)


def law(m0: float) -> DirichletMIDLaw:
    return DirichletMIDLaw(
        mass_classes=(0, 1, 2),
        active_support=(0, 2),
        predicted_mid=(m0, 0.0, 1.0 - m0),
        precision=30.0,
        precision_source="fixed_external",
        precision_provenance="synthetic declared design value",
        observation_identity=("experiment", "fragment", "technical-series"),
        correction=CORRECTION,
        replicate_count=2,
        replicate_semantics="technical_measurement_variability",
        independent_replicates=True,
    )


def product(m0: float) -> IndependentDirichletMIDProductLaw:
    return IndependentDirichletMIDProductLaw(blocks=(law(m0),))


problem = DirichletCompositeBinaryTestingProblem(
    null=DirichletCompositeMIDLawFamily(
        members=(product(0.70), product(0.65)),
        member_ids=("H0-a", "H0-b"),
    ),
    alternative=DirichletCompositeMIDLawFamily(
        members=(product(0.30), product(0.35)),
        member_ids=("H1-a", "H1-b"),
    ),
)

converse = composite_dirichlet_renyi_converse_at_order(
    problem, epsilon=TYPE_I_BUDGET, order=1.2,
)
candidate = verified_composite_dirichlet_renyi_score(problem, order=0.5)
bound = composite_dirichlet_score_bound_at_order(
    candidate, epsilon=TYPE_I_BUDGET,
)

uncertainty = implied_dirichlet_precision_from_uncertainty(
    (0.65, 0.35),
    (0.03, 0.03),
    replicate_count=3,
    uncertainty_kind="se",
)

# An independent synthetic calibration panel is used only to demonstrate the
# structured calibration API. Production provenance must identify a real,
# scientifically appropriate calibration source.
rng = np.random.default_rng(626)
active_replicates = tuple(
    (sample[0], sample[2])
    for sample in (law(0.65).sample(rng=rng) for _ in range(500))
)
calibration = estimate_dirichlet_precision_from_replicates(
    active_replicates,
    center=(0.65, 0.35),
    replicate_semantics="technical_measurement_variability",
    independent_of_test_data=True,
    bootstrap_samples=50,
    bootstrap_seed=626,
)

print("Corrected continuous MID Dirichlet example")
print(f"active mass classes: {law(0.65).active_mass_classes}")
print(f"structural zero mass classes: {law(0.65).structural_zero_mass_classes}")
print("precision 30 is a concentration, not a count")
print(f"order-1.2 converse lower bound: {converse.type_ii_lower_bound:.9g}")
print(
    "verified order-1/2 projected upper bound: "
    f"{bound.minimax_type_ii_upper_bound:.9g}"
)
print(
    "component-wise SE-implied concentrations: "
    f"{tuple(item.implied_precision for item in uncertainty.components)}"
)
print(f"independent-panel fitted concentration: {calibration.estimate:.9g}")
print(
    "The exact count-space minimax LP and exact score CDF are intentionally "
    "unavailable for this continuous observation law."
)

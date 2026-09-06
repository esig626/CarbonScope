"""Simple binary error certificates for explicitly declared observation laws.

H0 is the fixed null law P0; H1 is the fixed alternative law P1. Type I is
P0(decide H1), and Type II is P1(decide H0). MFA fitting remains separate.
"""

from .simple import (
    BrunoOrderCertificate,
    BrunoTheoremAssumptionError,
    NumericalLimitError,
    SimpleBinaryLawPair,
    SimpleBinaryTestingConstraint,
    validate_bruno_assumptions,
    validate_renyi_order,
)
from .bruno import bruno_converse_at_order

__all__ = [
    "BrunoOrderCertificate", "BrunoTheoremAssumptionError", "NumericalLimitError",
    "SimpleBinaryLawPair", "SimpleBinaryTestingConstraint",
    "bruno_converse_at_order",
    "validate_bruno_assumptions", "validate_renyi_order",
]

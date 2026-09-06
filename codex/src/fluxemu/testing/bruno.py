"""Full-law, finite-order converse from Bruno et al., arXiv:2601.09550v2.

Theorem 1 / Eq. (5) and Appendix A / Eqs. (14)--(17) give the reverse
component; Appendix A's following unnumbered inequality gives the forward
component. Both full-law inequalities precede i.i.d. tensorisation. Existing
``fluxemu.observation`` functions own the multinomial identity and independent
product additivity. This module neither changes those laws nor implements
another divergence kernel.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

from .. import observation as _observation
from ..exceptions import InputValidationError, ValidationError
from .simple import (
    BrunoOrderCertificate as BrunoOrderBound,
    NumericalLimitError,
    SimpleBinaryLawPair,
    SimpleBinaryTestingConstraint,
    validate_renyi_order,
)

if TYPE_CHECKING:
    from .stationary import StationarySimpleTestingResult


def _law_renyi(
    left: _observation.MultinomialMIDLaw | tuple[_observation.MultinomialMIDLaw, ...],
    right: _observation.MultinomialMIDLaw | tuple[_observation.MultinomialMIDLaw, ...],
    order: float,
    direction: str,
) -> float:
    """Call the existing full-law API and expose its numerical limits."""
    try:
        if isinstance(left, _observation.MultinomialMIDLaw):
            value = _observation.renyi_multinomial(left, right, order)
        else:
            value = _observation.independent_product_renyi(left, right, order)
    except (ArithmeticError, ValidationError) as error:
        raise NumericalLimitError(
            f"{direction} full-law Rényi evaluation failed in the existing "
            "observation divergence kernel; no probabilities or order were repaired"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise NumericalLimitError(
            f"{direction} full-law Rényi evaluation returned inherited "
            f"nonfinite or negative law divergence {value!r}; "
            "signed roundoff is not clipped"
        )
    return value


def bruno_converse_at_order(
    laws: SimpleBinaryLawPair | StationarySimpleTestingResult,
    *,
    epsilon: Real,
    order: Real,
) -> BrunoOrderBound:
    """Return a lower bound on optimal Type II at one finite real order >1.

    ``laws`` is a :class:`SimpleBinaryLawPair` or the result of the stationary
    simple-hypothesis bridge, which exposes its validated ``law_pair``.
    H0=P0 is null and H1=P1 is alternative. Among deterministic tests with
    Type I=P0(decide H1) <= ``epsilon``, optimal Type II=P1(decide H0) is at
    least the returned ``type_ii_lower_bound``.

    Reverse Rényi is D_order(P1 || P0), and forward is D_order(P0 || P1).
    Mutual absolute continuity is checked before either divergence is
    evaluated. No support repair, count inference, fitting, or order search
    occurs. Each finite runtime order greater than one is used unchanged;
    unrepresentable mathematical inputs raise :class:`NumericalLimitError`.

    Natural logarithms, ``expm1``, and ``log1p`` retain small differences near
    order one and interior epsilon endpoints. A vacuous reverse component
    below floating-point range is exposed as -inf with its finite log-space
    diagnostic; the forward component and combined bound remain finite.
    Forward underflow retains its finite log bound. The result is the bound
    at this order, without a claim about the full continuous-order envelope.
    """
    pair = getattr(laws, "law_pair", laws)
    if not isinstance(pair, SimpleBinaryLawPair):
        raise InputValidationError(
            "Bruno bound evaluation requires SimpleBinaryLawPair or a stationary "
            "simple-testing result with a validated law_pair"
        )
    constraint = SimpleBinaryTestingConstraint(epsilon=epsilon)
    finite_order = validate_renyi_order(order)
    pair.require_mutual_absolute_continuity()

    reverse_renyi = _law_renyi(
        pair.alternative, pair.null, finite_order, "reverse D_lambda(P1 || P0)",
    )
    forward_renyi = _law_renyi(
        pair.null, pair.alternative, finite_order, "forward D_lambda(P0 || P1)",
    )

    offset = finite_order - 1.0
    try:
        reverse_log_power = (offset / finite_order) * math.fsum((
            math.log(constraint.epsilon), reverse_renyi,
        ))
        log_forward_lower_bound = math.fsum((
            (finite_order / offset) * math.log1p(-constraint.epsilon),
            -forward_renyi,
        ))
    except ArithmeticError as error:
        raise NumericalLimitError(
            "Bruno log-domain components exceeded finite numerical range"
        ) from error
    if not math.isfinite(reverse_log_power) or not math.isfinite(log_forward_lower_bound):
        raise NumericalLimitError(
            "Bruno log-domain components are not finite at this numerical precision"
        )

    # The raw reverse expression can be arbitrarily negative. Overflow here
    # signifies a vacuous component; it must not erase a useful forward bound.
    try:
        reverse_lower_bound = -math.expm1(reverse_log_power)
    except OverflowError:
        reverse_lower_bound = -math.inf
    forward_lower_bound = math.exp(log_forward_lower_bound)
    type_ii_lower_bound = max(reverse_lower_bound, forward_lower_bound)
    if not math.isfinite(type_ii_lower_bound) or not 0 <= type_ii_lower_bound <= 1:
        raise NumericalLimitError(
            "combined Bruno Type-II lower bound exceeded [0, 1] at numerical precision; "
            "raw components are not clipped"
        )

    return BrunoOrderBound(
        pair=pair,
        constraint=constraint,
        order=finite_order,
        reverse_renyi=reverse_renyi,
        forward_renyi=forward_renyi,
        reverse_lower_bound=reverse_lower_bound,
        forward_lower_bound=forward_lower_bound,
        type_ii_lower_bound=type_ii_lower_bound,
        log_forward_lower_bound=log_forward_lower_bound,
        reverse_log_power=reverse_log_power,
    )


__all__ = ["bruno_converse_at_order"]

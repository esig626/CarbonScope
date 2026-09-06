"""Immutable roles, assumptions, and provenance for simple binary testing.

``null`` always means H0=P0 and ``alternative`` always means H1=P1.
Type I is P0(decide H1); Type II is P1(decide H0). These records own no
observation-law implementation and never change stored MID probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Real

from ..exceptions import InputValidationError, ValidationError
from ..model import deterministic_serialise
from ..observation import MultinomialMIDLaw


class BrunoTheoremAssumptionError(InputValidationError):
    """Bruno et al. Theorem 1 cannot apply to the declared law pair."""


class NumericalLimitError(ValidationError):
    """An admissible mathematical input exceeds supported numerical precision."""


def _testing_digest(value: object) -> str:
    return sha256(deterministic_serialise(value).encode("utf-8")).hexdigest()


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name} must be a nonempty immutable string")


def _finite_real(value: object, name: str) -> float:
    """Validate source scalars before narrowing to the runtime float type."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(f"{name} must be a finite real scalar, not bool")
    if value != value or value == math.inf or value == -math.inf:
        raise InputValidationError(f"{name} must be finite, not NaN or infinity")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise NumericalLimitError(f"{name} is outside finite float representability") from error
    if not math.isfinite(result):
        raise NumericalLimitError(f"{name} is outside finite float representability")
    return result


def validate_renyi_order(order: Real) -> float:
    """Accept a finite real order strictly above one, without a grid or snapping.

    If a finite source scalar cannot be represented as a finite runtime float
    greater than one, fail explicitly instead of replacing it by an endpoint.
    In particular, ``nextafter(1, +inf)`` is retained exactly.
    """
    result = _finite_real(order, "Rényi order lambda")
    if order <= 1:
        raise InputValidationError("Rényi order lambda must be strictly greater than 1")
    if result <= 1:
        raise NumericalLimitError("Rényi order lambda rounds to 1 at float precision")
    return result


def _epsilon(epsilon: Real) -> float:
    result = _finite_real(epsilon, "Type-I constraint epsilon")
    if not 0 < epsilon < 1:
        raise InputValidationError("Type-I constraint epsilon must satisfy 0 < epsilon < 1")
    if not 0 < result < 1:
        raise NumericalLimitError("Type-I constraint epsilon rounds to an endpoint at float precision")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleBinaryTestingConstraint:
    """The sole testing constraint: P0(decide H1) <= epsilon, 0 < epsilon < 1."""

    epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _epsilon(self.epsilon))

    @property
    def type_i_constraint(self) -> float:
        return self.epsilon

    @property
    def fingerprint(self) -> str:
        return _testing_digest(("simple-binary-type-i-constraint-v1", self.epsilon))


@dataclass(frozen=True, slots=True, kw_only=True)
class SimpleBinaryLawPair:
    """Two fixed laws with explicit H0/P0 and H1/P1 roles.

    For one block, pass ``null=law0, alternative=law1``. For a product, pass
    immutable tuples and explicitly declare ``independent=True``. Corresponding
    blocks share their genuine count total and ordered mass-class space;
    different blocks need not share a total. No common total is fabricated.

    Exact support mismatch is retained here for likelihood-ratio evaluation.
    Bruno certification must call :func:`validate_bruno_assumptions` first.
    ``block_identities``, when supplied, contains ordered
    ``(experiment_id, target_id, replicate_id)`` triples.
    """

    null: MultinomialMIDLaw | tuple[MultinomialMIDLaw, ...]
    alternative: MultinomialMIDLaw | tuple[MultinomialMIDLaw, ...]
    independent: bool = False
    block_identities: tuple[tuple[str, str, str], ...] = ()
    null_state_fingerprint: str | None = None
    alternative_state_fingerprint: str | None = None
    observation_specification_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.independent) is not bool:
            raise InputValidationError("independent must be an explicit bool")
        single = isinstance(self.null, MultinomialMIDLaw)
        if single:
            if not isinstance(self.alternative, MultinomialMIDLaw):
                raise InputValidationError("null and alternative must both be single laws or ordered tuples")
        else:
            if not isinstance(self.null, tuple) or not isinstance(self.alternative, tuple):
                raise InputValidationError("products require immutable ordered law tuples in both roles")
            if not self.independent:
                raise InputValidationError("product laws require explicitly declared independent=True")
            if not self.null or len(self.null) != len(self.alternative):
                raise InputValidationError("null and alternative require the same positive number of blocks")
            if not all(isinstance(law, MultinomialMIDLaw)
                       for law in (*self.null, *self.alternative)):
                raise InputValidationError("every null and alternative block must be MultinomialMIDLaw")
        for index, (null, alternative) in enumerate(zip(
            self.null_laws, self.alternative_laws, strict=True,
        )):
            if null.n != alternative.n:
                raise InputValidationError(f"block {index}: null and alternative count totals must match")
            if null.mass_classes != alternative.mass_classes:
                raise InputValidationError(f"block {index}: ordered mass-class spaces must match")
        if not isinstance(self.block_identities, tuple):
            raise InputValidationError("block_identities must be an immutable ordered tuple")
        if self.block_identities and len(self.block_identities) != len(self.null_laws):
            raise InputValidationError("block_identities must identify every law block in order")
        for identity in self.block_identities:
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise InputValidationError("each block identity must be an immutable experiment/target/replicate triple")
            for value, name in zip(identity, ("experiment_id", "target_id", "replicate_id"), strict=True):
                _identifier(value, name)
        if len(set(self.block_identities)) != len(self.block_identities):
            raise InputValidationError("block_identities contain a duplicate experiment/target/replicate identity")
        for name in (
            "null_state_fingerprint", "alternative_state_fingerprint",
            "observation_specification_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                _identifier(value, name)
        if (self.null_state_fingerprint is None) != (self.alternative_state_fingerprint is None):
            raise InputValidationError("state provenance must identify both null and alternative")

    @property
    def null_laws(self) -> tuple[MultinomialMIDLaw, ...]:
        return (self.null,) if isinstance(self.null, MultinomialMIDLaw) else self.null

    @property
    def alternative_laws(self) -> tuple[MultinomialMIDLaw, ...]:
        return (self.alternative,) if isinstance(self.alternative, MultinomialMIDLaw) else self.alternative

    @property
    def count_totals(self) -> tuple[int, ...]:
        return tuple(law.n for law in self.null_laws)

    @property
    def null_fingerprint(self) -> str:
        if isinstance(self.null, MultinomialMIDLaw):
            return self.null.fingerprint
        return _testing_digest(("declared-independent-multinomial-product-v1",
                                tuple(law.fingerprint for law in self.null)))

    @property
    def alternative_fingerprint(self) -> str:
        if isinstance(self.alternative, MultinomialMIDLaw):
            return self.alternative.fingerprint
        return _testing_digest(("declared-independent-multinomial-product-v1",
                                tuple(law.fingerprint for law in self.alternative)))

    @property
    def fingerprint(self) -> str:
        return _testing_digest((
            "simple-binary-law-pair-v1", ("H0=P0", self.null_fingerprint),
            ("H1=P1", self.alternative_fingerprint), self.independent,
            self.block_identities, self.null_state_fingerprint,
            self.alternative_state_fingerprint, self.observation_specification_fingerprint,
        ))

    @property
    def mutually_absolutely_continuous(self) -> bool:
        return all(
            tuple(p > 0 for p in null.probabilities)
            == tuple(p > 0 for p in alternative.probabilities)
            for null, alternative in zip(self.null_laws, self.alternative_laws, strict=True)
        )

    def require_mutual_absolute_continuity(self) -> None:
        validate_bruno_assumptions(self)


def validate_bruno_assumptions(pair: SimpleBinaryLawPair) -> None:
    """Require Bruno Theorem 1 mutual absolute continuity with exact support.

    Every fixed-total block must have the same positive-support mass classes
    under H0 and H1. A finite one-direction divergence is insufficient.
    """
    if not isinstance(pair, SimpleBinaryLawPair):
        raise InputValidationError("Bruno theorem validation requires SimpleBinaryLawPair")
    for index, (null, alternative) in enumerate(zip(
        pair.null_laws, pair.alternative_laws, strict=True,
    )):
        null_support = tuple(i for i, p in enumerate(null.probabilities) if p > 0)
        alternative_support = tuple(i for i, p in enumerate(alternative.probabilities) if p > 0)
        if null_support != alternative_support:
            identity = f" {pair.block_identities[index]!r}" if pair.block_identities else ""
            raise BrunoTheoremAssumptionError(
                "Bruno et al. Theorem 1 requires mutual absolute continuity: "
                f"block {index}{identity} has null positive support {null_support} "
                f"and alternative positive support {alternative_support}; "
                "exact supports are retained without smoothing"
            )


def _diagnostic(value: object, name: str, *, negative_infinity: bool = False) -> float:
    if negative_infinity and isinstance(value, Real) and not isinstance(value, bool) and value == -math.inf:
        return -math.inf
    return _finite_real(value, name)


@dataclass(frozen=True, slots=True, kw_only=True)
class BrunoOrderCertificate:
    """Order-specific lower certificate on optimal Type-II error.

    Among deterministic tests with Type I=P0(decide H1) <= epsilon, optimal
    Type II=P1(decide H0) is at least ``type_ii_lower_bound`` at this order.
    The fields do not describe an observed sample's error probability. Natural
    logarithms are used. An order-specific certificate never claims to certify
    the global continuous-order envelope.

    ``reverse_renyi`` is D_lambda(P1 || P0); ``forward_renyi`` is
    D_lambda(P0 || P1). A vacuous reverse component may be represented as
    negative infinity on overflow; ``reverse_log_power`` retains its log-space
    diagnostic. Forward underflow retains ``log_forward_lower_bound``.
    """

    pair: SimpleBinaryLawPair
    constraint: SimpleBinaryTestingConstraint
    order: float
    reverse_renyi: float
    forward_renyi: float
    reverse_lower_bound: float
    forward_lower_bound: float
    type_ii_lower_bound: float
    log_forward_lower_bound: float
    reverse_log_power: float

    def __post_init__(self) -> None:
        validate_bruno_assumptions(self.pair)
        if not isinstance(self.constraint, SimpleBinaryTestingConstraint):
            raise InputValidationError("constraint must be SimpleBinaryTestingConstraint")
        object.__setattr__(self, "order", validate_renyi_order(self.order))
        for name in (
            "reverse_renyi", "forward_renyi", "reverse_lower_bound", "forward_lower_bound",
            "type_ii_lower_bound", "log_forward_lower_bound", "reverse_log_power",
        ):
            value = _diagnostic(getattr(self, name), name, negative_infinity=(
                name in ("reverse_lower_bound", "log_forward_lower_bound")))
            object.__setattr__(self, name, value)
        if self.reverse_renyi < 0 or self.forward_renyi < 0:
            raise NumericalLimitError("law Rényi diagnostics must be nonnegative; signed roundoff is not clipped")
        if self.reverse_lower_bound > 1:
            raise InputValidationError("raw reverse lower bound cannot exceed 1")
        if not 0 <= self.forward_lower_bound <= 1 or self.log_forward_lower_bound > 0:
            raise InputValidationError("forward lower bound must lie in [0, 1] with nonpositive logarithm")
        if not 0 <= self.type_ii_lower_bound <= 1:
            raise InputValidationError("combined Type-II lower certificate must be finite and lie in [0, 1]")
        if self.type_ii_lower_bound != max(self.reverse_lower_bound, self.forward_lower_bound):
            raise InputValidationError("Type-II lower certificate must equal the maximum of both raw components")

    @property
    def type_i_constraint(self) -> float:
        return self.constraint.epsilon

    @property
    def null_fingerprint(self) -> str:
        return self.pair.null_fingerprint

    @property
    def alternative_fingerprint(self) -> str:
        return self.pair.alternative_fingerprint

    @property
    def null_state_fingerprint(self) -> str | None:
        return self.pair.null_state_fingerprint

    @property
    def alternative_state_fingerprint(self) -> str | None:
        return self.pair.alternative_state_fingerprint

    @property
    def observation_specification_fingerprint(self) -> str | None:
        return self.pair.observation_specification_fingerprint

    @property
    def global_envelope_certified(self) -> bool:
        return False

    @property
    def reverse_component_overflowed(self) -> bool:
        return self.reverse_lower_bound == -math.inf

    @property
    def forward_component_underflowed(self) -> bool:
        return self.forward_lower_bound == 0

    @property
    def numerical_diagnostics(self) -> tuple[str, ...]:
        return tuple(name for active, name in (
            (self.reverse_component_overflowed, "reverse_component_overflow"),
            (self.forward_component_underflowed, "forward_component_underflow"),
        ) if active)

    @property
    def fingerprint(self) -> str:
        # Hex strings preserve signed zero and explicit overflow diagnostics,
        # while canonical serialization correctly rejects nonfinite floats.
        values = tuple(getattr(self, name).hex() for name in (
            "order", "reverse_renyi", "forward_renyi", "reverse_lower_bound",
            "forward_lower_bound", "type_ii_lower_bound", "log_forward_lower_bound",
            "reverse_log_power",
        ))
        return _testing_digest(("bruno-order-certificate-v1", self.pair.fingerprint,
                                self.constraint.fingerprint, values))


__all__ = [
    "BrunoOrderCertificate", "BrunoTheoremAssumptionError", "NumericalLimitError",
    "SimpleBinaryLawPair", "SimpleBinaryTestingConstraint",
    "validate_bruno_assumptions", "validate_renyi_order",
]

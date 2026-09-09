"""Continuous Dirichlet laws for externally corrected MID compositions.

This module is deliberately parallel to :mod:`fluxemu.observation.multinomial`.
Dirichlet precision is a concentration parameter, never a count or an inferred
effective sample size.  Inputs are validated without clipping, pseudocounts,
or hidden renormalisation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from ..exceptions import InputValidationError, ValidationError
from .multinomial import _resolve_rng
from .schema import _observation_digest


DIRICHLET_SIMPLEX_TOLERANCE = 1e-12
DIRICHLET_NEAR_ONE_REFUSAL_RADIUS = 1e-8
DIRICHLET_RENYI_ROUNDOFF_RELATIVE_BUDGET = 1e-8
DIRICHLET_PRECISION_SOURCES = (
    "fixed_external",
    "external_calibration",
    "independent_calibration",
    "same_data_plugin",
)
DIRICHLET_REPLICATE_SEMANTICS = (
    "technical_measurement_variability",
    "biological_replicate_variability",
    "total_replicate_variability",
)


class DirichletNumericalError(ValidationError):
    """A valid Dirichlet expression exceeds the supported float policy."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{name} must be a nonempty string")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(f"{name} must be a finite real scalar, not bool")
    if value != value or value in (math.inf, -math.inf):
        raise InputValidationError(f"{name} must be finite")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise DirichletNumericalError(
            f"{name} is outside finite float representability"
        ) from error
    if not math.isfinite(result):
        raise DirichletNumericalError(
            f"{name} is outside finite float representability"
        )
    return result


def _positive_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if value <= 0 or result <= 0:
        raise InputValidationError(f"{name} must be strictly positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InputValidationError(f"{name} must be a positive integer, not bool")
    return int(value)


def _ordered_values(values: object, name: str) -> tuple[object, ...]:
    if isinstance(values, (Mapping, Set)):
        raise InputValidationError(f"{name} requires a declared positional order")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise InputValidationError(f"{name} must be an ordered iterable") from error
    if not result:
        raise InputValidationError(f"{name} must be nonempty")
    return result


def _simplex(values: object, name: str, *, strictly_positive: bool) -> tuple[float, ...]:
    source = _ordered_values(values, name)
    if len(source) < 2:
        raise InputValidationError(f"{name} must contain at least two components")
    converted = []
    for index, value in enumerate(source):
        number = _finite_real(value, f"{name}[{index}]")
        if value < 0 or number < 0:
            raise InputValidationError(f"{name}[{index}] must be nonnegative")
        if value > 1 or number > 1:
            raise InputValidationError(f"{name}[{index}] must not exceed one")
        if value > 0 and number == 0:
            raise DirichletNumericalError(
                f"{name}[{index}] positive support is erased by float conversion"
            )
        if strictly_positive and number <= 0:
            raise InputValidationError(f"{name}[{index}] must be strictly positive")
        converted.append(number)
    result = tuple(converted)
    mass = math.fsum(result)
    if not math.isfinite(mass) or abs(mass - 1.0) > DIRICHLET_SIMPLEX_TOLERANCE:
        raise InputValidationError(
            f"{name} must sum to one within {DIRICHLET_SIMPLEX_TOLERANCE:g}; "
            "values are not renormalised"
        )
    return result


def _mass_classes(values: object, size: int) -> tuple[int, ...]:
    source = _ordered_values(values, "mass_classes")
    if len(source) != size:
        raise InputValidationError("mass_classes must identify every MID coordinate")
    result = []
    for index, value in enumerate(source):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise InputValidationError(
                f"mass_classes[{index}] must be a nonnegative integer, not bool"
            )
        result.append(int(value))
    if len(set(result)) != len(result):
        raise InputValidationError("mass_classes must be unique")
    return tuple(result)


def _active_support(values: object, size: int) -> tuple[int, ...]:
    source = _ordered_values(values, "active_support")
    result = []
    for index, value in enumerate(source):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise InputValidationError(
                f"active_support[{index}] must be an integer coordinate index"
            )
        coordinate = int(value)
        if not 0 <= coordinate < size:
            raise InputValidationError("active_support contains an out-of-range coordinate")
        result.append(coordinate)
    if tuple(sorted(set(result))) != tuple(result):
        raise InputValidationError(
            "active_support must be unique and follow the declared coordinate order"
        )
    if len(result) < 2:
        raise InputValidationError(
            "Dirichlet V1 requires at least two active coordinates on its simplex face"
        )
    return tuple(result)


def _log_beta_with_roundoff_bound(
    parameters: tuple[float, ...],
) -> tuple[float, float]:
    """Return log B and a conservative binary64 cancellation indicator.

    The second value is not an interval-arithmetic certificate.  It is a
    deliberately conservative first-order bound based on the magnitudes of
    every ``lgamma`` term.  Rényi calculations use it to refuse ill-conditioned
    subtraction instead of reporting digits that binary64 cannot resolve.
    """

    if not parameters or any(value <= 0 or not math.isfinite(value) for value in parameters):
        raise InputValidationError("multivariate beta parameters must be finite and positive")
    component_terms = tuple(math.lgamma(value) for value in parameters)
    total_term = math.lgamma(math.fsum(parameters))
    result = math.fsum((*component_terms, -total_term))
    if not math.isfinite(result):
        raise DirichletNumericalError(
            "multivariate log-beta is outside finite float representability"
        )
    magnitude = math.fsum(abs(value) for value in (*component_terms, total_term, result))
    roundoff_bound = 32.0 * math.ulp(1.0) * max(1.0, magnitude)
    return result, roundoff_bound


def _log_beta(parameters: tuple[float, ...]) -> float:
    return _log_beta_with_roundoff_bound(parameters)[0]


def _digamma(value: float) -> float:
    """Binary64 digamma without importing the optional SciPy dependency."""

    if value <= 0 or not math.isfinite(value):
        raise InputValidationError("digamma argument must be finite and positive")
    result = 0.0
    x = value
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inverse = 1.0 / x
    square = inverse * inverse
    # Asymptotic Bernoulli expansion through x^-14.
    result += math.log(x) - 0.5 * inverse
    result -= square * (
        1.0 / 12.0
        - square * (
            1.0 / 120.0
            - square * (
                1.0 / 252.0
                - square * (
                    1.0 / 240.0
                    - square * (5.0 / 660.0 - square * (691.0 / 32760.0))
                )
            )
        )
    )
    if not math.isfinite(result):
        raise DirichletNumericalError("digamma evaluation is nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class MIDCorrectionProvenance:
    """Auditable external natural-abundance correction declaration."""

    status: str
    method: str
    provenance: str

    def __post_init__(self) -> None:
        if self.status != "externally_corrected":
            raise InputValidationError(
                "Dirichlet V1 requires correction.status='externally_corrected'; "
                "unknown or uncorrected MID data are not accepted"
            )
        _identifier(self.method, "correction method")
        _identifier(self.provenance, "correction provenance")

    @property
    def fingerprint(self) -> str:
        return _observation_digest(("external-mid-correction-v1", self))


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletMIDLaw:
    """One ``Dirichlet(kappa * p)`` law on an explicitly declared face.

    ``predicted_mid`` retains the full ordered mass-class vector, including
    exact structural zeros. ``active_support`` contains positional indices into
    that vector.  Every active probability is strictly positive and every
    inactive probability is exactly zero.  No epsilon or pseudocount is added.

    ``replicate_count`` is experiment-design provenance.  Density and ``sample``
    concern one replicate; independent-product testing and ``sample_replicates``
    apply the explicitly declared number of conditionally independent draws.
    Precision is a concentration parameter and is never exposed as a count.
    """

    mass_classes: tuple[int, ...]
    active_support: tuple[int, ...]
    predicted_mid: tuple[float, ...]
    precision: float
    observation_identity: tuple[str, str, str]
    correction: MIDCorrectionProvenance
    precision_source: str
    precision_provenance: str
    replicate_count: int
    replicate_semantics: str
    independent_replicates: bool

    def __post_init__(self) -> None:
        mean = _simplex(self.predicted_mid, "predicted_mid", strictly_positive=False)
        classes = _mass_classes(self.mass_classes, len(mean))
        active = _active_support(self.active_support, len(mean))
        expected_active = tuple(index for index, value in enumerate(mean) if value > 0)
        if active != expected_active:
            raise InputValidationError(
                "active_support must identify exactly the strictly positive predicted MID "
                "coordinates; structural zeros are not replaced"
            )
        if math.fsum(mean[index] for index in active) != 1.0:
            raise InputValidationError(
                "the active predicted MID must have an exactly representable unit sum; "
                "Dirichlet V1 does not renormalise EMU predictions"
            )
        precision = _positive_real(self.precision, "Dirichlet precision")
        parameters = tuple(precision * mean[index] for index in active)
        if any(value <= 0 or not math.isfinite(value) for value in parameters):
            raise DirichletNumericalError(
                "Dirichlet alpha=kappa*p contains an unrepresentable parameter"
            )
        identity = _ordered_values(self.observation_identity, "observation_identity")
        if len(identity) != 3:
            raise InputValidationError(
                "observation_identity must be an experiment/target/replicate triple"
            )
        identity = tuple(
            _identifier(value, name)
            for value, name in zip(
                identity, ("experiment_id", "target_id", "replicate_id"), strict=True
            )
        )
        if not isinstance(self.correction, MIDCorrectionProvenance):
            raise InputValidationError("correction must be MIDCorrectionProvenance")
        if self.precision_source not in DIRICHLET_PRECISION_SOURCES:
            raise InputValidationError(
                "precision_source must be one of " + ", ".join(DIRICHLET_PRECISION_SOURCES)
            )
        _identifier(self.precision_provenance, "precision provenance")
        replicate_count = _positive_integer(self.replicate_count, "replicate_count")
        if self.replicate_semantics not in DIRICHLET_REPLICATE_SEMANTICS:
            raise InputValidationError(
                "replicate_semantics must be one of "
                + ", ".join(DIRICHLET_REPLICATE_SEMANTICS)
            )
        if type(self.independent_replicates) is not bool:
            raise InputValidationError("independent_replicates must be an explicit bool")
        if replicate_count > 1 and not self.independent_replicates:
            raise InputValidationError(
                "multiple Dirichlet replicates require explicit independent_replicates=True"
            )
        object.__setattr__(self, "predicted_mid", mean)
        object.__setattr__(self, "mass_classes", classes)
        object.__setattr__(self, "active_support", active)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "observation_identity", identity)
        object.__setattr__(self, "replicate_count", replicate_count)

    @property
    def kappa(self) -> float:
        return self.precision

    @property
    def precision_is_count(self) -> bool:
        return False

    @property
    def parameters(self) -> tuple[float, ...]:
        return tuple(
            self.precision * self.predicted_mid[index] for index in self.active_support
        )

    @property
    def alpha(self) -> tuple[float, ...]:
        return self.parameters

    @property
    def mean(self) -> tuple[float, ...]:
        return self.predicted_mid

    @property
    def active_mean(self) -> tuple[float, ...]:
        return tuple(self.predicted_mid[index] for index in self.active_support)

    @property
    def active_mass_classes(self) -> tuple[int, ...]:
        return tuple(self.mass_classes[index] for index in self.active_support)

    @property
    def structural_zero_mass_classes(self) -> tuple[int, ...]:
        active = set(self.active_support)
        return tuple(
            mass_class for index, mass_class in enumerate(self.mass_classes) if index not in active
        )

    @property
    def rigorous_testing_suitable(self) -> bool:
        return self.precision_source != "same_data_plugin"

    @property
    def log_normalizer(self) -> float:
        return _log_beta(self.parameters)

    @property
    def covariance(self) -> tuple[tuple[float, ...], ...]:
        result = []
        active = set(self.active_support)
        denominator = self.precision + 1.0
        if not math.isfinite(denominator):
            raise DirichletNumericalError(
                "Dirichlet covariance denominator is outside float representability"
            )
        for i, p_i in enumerate(self.predicted_mid):
            row = []
            for j, p_j in enumerate(self.predicted_mid):
                if i not in active or j not in active:
                    value = 0.0
                elif i == j:
                    value = p_i * (1.0 - p_i) / denominator
                else:
                    value = -p_i * p_j / denominator
                row.append(value)
            result.append(tuple(row))
        return tuple(result)

    @property
    def active_covariance(self) -> tuple[tuple[float, ...], ...]:
        full = self.covariance
        return tuple(
            tuple(full[i][j] for j in self.active_support) for i in self.active_support
        )

    @property
    def fingerprint(self) -> str:
        return _observation_digest((
            "dirichlet-mid-law-v1",
            self.mass_classes,
            self.active_support,
            self.predicted_mid,
            self.precision,
            self.observation_identity,
            self.correction,
            self.precision_source,
            self.precision_provenance,
            self.replicate_count,
            self.replicate_semantics,
            self.independent_replicates,
        ))

    def _observation(self, value: object) -> tuple[float, ...]:
        observation = _simplex(value, "Dirichlet MID observation", strictly_positive=False)
        if len(observation) != len(self.predicted_mid):
            raise InputValidationError(
                "Dirichlet MID observation has the wrong number of mass classes"
            )
        active = set(self.active_support)
        for index, component in enumerate(observation):
            if index in active and component <= 0:
                raise InputValidationError(
                    "an observed exact zero on a continuously positive active coordinate "
                    "requires a censoring/detection-limit model; no epsilon was added"
                )
            if index not in active and component != 0:
                raise InputValidationError(
                    "Dirichlet MID observation assigns mass to a declared structural-zero coordinate"
                )
        return observation

    def log_density(self, observation: object) -> float:
        values = self._observation(observation)
        terms = [
            (parameter - 1.0) * math.log(values[index])
            for index, parameter in zip(self.active_support, self.parameters, strict=True)
        ]
        result = math.fsum(terms) - self.log_normalizer
        if not math.isfinite(result):
            raise DirichletNumericalError(
                "Dirichlet log density is outside finite float representability"
            )
        return result

    def log_prob(self, observation: object) -> float:
        """Alias for the face-relative log density, not a discrete probability."""

        return self.log_density(observation)

    def sample(
        self, *, seed: int | None = None, rng: np.random.Generator | None = None,
    ) -> tuple[float, ...]:
        generator = _resolve_rng(seed=seed, rng=rng)
        active_sample = generator.dirichlet(self.parameters)
        if not np.isfinite(active_sample).all() or np.any(active_sample <= 0):
            raise DirichletNumericalError(
                "Dirichlet sampler produced an unrepresentable boundary/nonfinite draw; "
                "the sample was not clipped"
            )
        result = [0.0] * len(self.predicted_mid)
        for index, value in zip(self.active_support, active_sample, strict=True):
            result[index] = float(value)
        if abs(math.fsum(result) - 1.0) > DIRICHLET_SIMPLEX_TOLERANCE:
            raise DirichletNumericalError(
                "Dirichlet sampler output is outside the documented simplex tolerance"
            )
        return tuple(result)

    def sample_replicates(
        self, *, seed: int | None = None, rng: np.random.Generator | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        generator = _resolve_rng(seed=seed, rng=rng)
        return tuple(self.sample(rng=generator) for _ in range(self.replicate_count))

    def renyi_divergence(self, other: "DirichletMIDLaw", order: Real) -> float:
        return renyi_dirichlet(self, other, order)

    def kl_divergence(self, other: "DirichletMIDLaw") -> float:
        return kl_dirichlet(self, other)

    def pairwise_score(self, alternative: "DirichletMIDLaw") -> "DirichletLogLikelihoodScore":
        return DirichletLogLikelihoodScore(null=self, alternative=alternative)


def _comparable(left: DirichletMIDLaw, right: DirichletMIDLaw) -> None:
    if not isinstance(left, DirichletMIDLaw) or not isinstance(right, DirichletMIDLaw):
        raise InputValidationError("Dirichlet comparison requires two DirichletMIDLaw records")
    if left.mass_classes != right.mass_classes:
        raise InputValidationError("Dirichlet laws require the same ordered mass classes")
    if left.active_support != right.active_support:
        raise InputValidationError(
            "Dirichlet V1 requires common active support; state-dependent support is refused"
        )


def kl_dirichlet(left: DirichletMIDLaw, right: DirichletMIDLaw) -> float:
    """Analytic ``D_KL(left || right)`` on a common active simplex face."""

    _comparable(left, right)
    alpha, beta = left.parameters, right.parameters
    if alpha == beta:
        return 0.0
    total = math.fsum(alpha)
    psi_total = _digamma(total)
    result = _log_beta(beta) - _log_beta(alpha) + math.fsum(
        (a - b) * (_digamma(a) - psi_total)
        for a, b in zip(alpha, beta, strict=True)
    )
    scale = max(1.0, abs(_log_beta(alpha)), abs(_log_beta(beta)))
    if not math.isfinite(result) or result < -1e-12 * scale:
        raise DirichletNumericalError(
            "Dirichlet KL is nonfinite or negative at supported numerical precision"
        )
    if result < 0:
        raise DirichletNumericalError(
            "Dirichlet KL is unresolved near zero; no divergence was clipped"
        )
    return result


def renyi_dirichlet(
    left: DirichletMIDLaw, right: DirichletMIDLaw, order: Real,
) -> float:
    """Analytic directed Rényi divergence on a common active face.

    The direction is exactly ``D_order(left || right)``.  At order one the
    analytic KL value is returned.  Non-unit orders closer than
    ``DIRICHLET_NEAR_ONE_REFUSAL_RADIUS`` are refused because direct binary64
    subtraction cannot reliably resolve the removable singularity.  If any
    mixed parameter is nonpositive, the defining integral diverges and
    ``math.inf`` is returned; parameters are never clipped.
    """

    _comparable(left, right)
    value = _positive_real(order, "Dirichlet Rényi order")
    if value == 1.0:
        return kl_dirichlet(left, right)
    if abs(value - 1.0) < DIRICHLET_NEAR_ONE_REFUSAL_RADIUS:
        raise DirichletNumericalError(
            "non-unit Dirichlet Rényi order is too close to one for the documented "
            "binary64 policy; use order=1 for KL or move farther from one"
        )
    alpha, beta = left.parameters, right.parameters
    if alpha == beta:
        return 0.0
    mixed = tuple(
        value * a + (1.0 - value) * b
        for a, b in zip(alpha, beta, strict=True)
    )
    if any(parameter <= 0 for parameter in mixed):
        return math.inf
    if any(not math.isfinite(parameter) for parameter in mixed):
        raise DirichletNumericalError("mixed Dirichlet Rényi parameters are nonfinite")
    log_mixed, mixed_error = _log_beta_with_roundoff_bound(mixed)
    log_alpha, alpha_error = _log_beta_with_roundoff_bound(alpha)
    log_beta, beta_error = _log_beta_with_roundoff_bound(beta)
    numerator_terms = (
        log_mixed,
        -value * log_alpha,
        -(1.0 - value) * log_beta,
    )
    numerator = math.fsum(numerator_terms)
    result = numerator / (value - 1.0)
    scale = max(1.0, abs(numerator), abs(log_alpha), abs(log_beta))
    if not math.isfinite(result) or result < -1e-11 * scale:
        raise DirichletNumericalError(
            "Dirichlet Rényi divergence is nonfinite or negative at supported precision"
        )
    if result < 0:
        raise DirichletNumericalError(
            "Dirichlet Rényi divergence is unresolved near zero; it was not clipped"
        )
    propagated_roundoff = math.fsum((
        mixed_error,
        abs(value) * alpha_error,
        abs(1.0 - value) * beta_error,
        32.0 * math.ulp(1.0) * math.fsum(abs(term) for term in numerator_terms),
    )) / abs(value - 1.0)
    if propagated_roundoff > (
        DIRICHLET_RENYI_ROUNDOFF_RELATIVE_BUDGET * max(1.0, abs(result))
    ):
        raise DirichletNumericalError(
            "Dirichlet Rényi subtraction is too ill-conditioned for the documented "
            "binary64 roundoff budget; no divergence was reported"
        )
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class DirichletLogLikelihoodScore:
    """The one-replicate score ``log q(y) - log p(y)`` for a selected pair."""

    null: DirichletMIDLaw
    alternative: DirichletMIDLaw

    def __post_init__(self) -> None:
        _comparable(self.null, self.alternative)
        if self.null.replicate_count != self.alternative.replicate_count:
            raise InputValidationError("score laws require the same replicate_count")
        if self.null.independent_replicates != self.alternative.independent_replicates:
            raise InputValidationError("score laws require aligned replicate independence")

    @property
    def constant(self) -> float:
        return self.null.log_normalizer - self.alternative.log_normalizer

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(
            b - a
            for a, b in zip(self.null.parameters, self.alternative.parameters, strict=True)
        )

    def evaluate(self, observation: object) -> float:
        values = self.null._observation(observation)
        result = self.constant + math.fsum(
            weight * math.log(values[index])
            for index, weight in zip(self.null.active_support, self.weights, strict=True)
        )
        if not math.isfinite(result):
            raise DirichletNumericalError("Dirichlet pairwise score is nonfinite")
        return result

    def evaluate_replicates(self, observations: object) -> float:
        values = _ordered_values(observations, "replicate MID observations")
        if len(values) != self.null.replicate_count:
            raise InputValidationError(
                "replicate MID observations must match the declared replicate_count"
            )
        return math.fsum(self.evaluate(value) for value in values)

    def log_moment(self, reference: DirichletMIDLaw, exponent: Real) -> float:
        """Return ``log E_reference[exp(exponent * score(Y))]`` analytically."""

        _comparable(reference, self.null)
        t = _finite_real(exponent, "score-moment exponent")
        shifted = tuple(
            gamma + t * weight
            for gamma, weight in zip(reference.parameters, self.weights, strict=True)
        )
        if any(parameter <= 0 for parameter in shifted):
            return math.inf
        if any(not math.isfinite(parameter) for parameter in shifted):
            raise DirichletNumericalError("shifted score-moment parameters are nonfinite")
        result = t * self.constant + _log_beta(shifted) - reference.log_normalizer
        if not math.isfinite(result):
            raise DirichletNumericalError("Dirichlet log score moment is nonfinite")
        return result

    def log_moment_for_declared_replicates(
        self, reference: DirichletMIDLaw, exponent: Real,
    ) -> float:
        if reference.replicate_count != self.null.replicate_count:
            raise InputValidationError("score moment laws require the same replicate_count")
        single = self.log_moment(reference, exponent)
        return single if single == math.inf else reference.replicate_count * single


def dirichlet_pairwise_log_likelihood_score(
    observation: object,
    null: DirichletMIDLaw,
    alternative: DirichletMIDLaw,
) -> float:
    return DirichletLogLikelihoodScore(null=null, alternative=alternative).evaluate(observation)


def dirichlet_score_log_moment(
    reference: DirichletMIDLaw,
    null: DirichletMIDLaw,
    alternative: DirichletMIDLaw,
    exponent: Real,
) -> float:
    return DirichletLogLikelihoodScore(
        null=null, alternative=alternative
    ).log_moment(reference, exponent)


@dataclass(frozen=True, slots=True)
class ComponentPrecisionEstimate:
    mass_class: int
    mean: float
    uncertainty: float
    implied_precision: float
    status: str


@dataclass(frozen=True, slots=True)
class PrecisionDispersion:
    valid_component_count: int
    minimum: float | None
    median: float | None
    maximum: float | None
    maximum_to_minimum_ratio: float | None
    median_absolute_deviation: float | None


@dataclass(frozen=True, slots=True)
class ImpliedPrecisionDiagnostic:
    """Component-wise same-data precision diagnostic; never a known law input."""

    uncertainty_kind: str
    replicate_count: int
    components: tuple[ComponentPrecisionEstimate, ...]
    dispersion: PrecisionDispersion
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    pooled_precision: None = None
    precision_source: str = "same_data_plugin"
    suitable_for_rigorous_testing: bool = False


def implied_dirichlet_precision_from_uncertainty(
    mean_mid: object,
    uncertainty: object,
    *,
    replicate_count: int,
    uncertainty_kind: str,
    mass_classes: object | None = None,
) -> ImpliedPrecisionDiagnostic:
    """Return every marginally implied concentration from declared SD or SE.

    For SD, ``kappa_i=p_i(1-p_i)/SD_i^2-1``.  For the SE of an
    independent-replicate mean, the denominator is ``R*SE_i^2``.  No agreement
    threshold or pooled estimate is manufactured.
    """

    mean = _simplex(mean_mid, "mean_mid", strictly_positive=False)
    raw_uncertainty = _ordered_values(uncertainty, "uncertainty")
    if len(raw_uncertainty) != len(mean):
        raise InputValidationError("uncertainty must identify every MID component")
    values = tuple(
        _finite_real(value, f"uncertainty[{index}]")
        for index, value in enumerate(raw_uncertainty)
    )
    if any(value < 0 for value in values):
        raise InputValidationError("uncertainty components must be nonnegative")
    count = _positive_integer(replicate_count, "replicate_count")
    if uncertainty_kind not in {"sd", "se"}:
        raise InputValidationError("uncertainty_kind must be explicitly 'sd' or 'se'")
    classes = (
        tuple(range(len(mean)))
        if mass_classes is None
        else _mass_classes(mass_classes, len(mean))
    )
    factor = count if uncertainty_kind == "se" else 1
    components = []
    for mass_class, p, spread in zip(classes, mean, values, strict=True):
        numerator = p * (1.0 - p)
        denominator = factor * spread * spread
        if denominator == 0.0:
            estimate = math.nan if numerator == 0.0 else math.inf
        else:
            estimate = numerator / denominator - 1.0
        status = (
            "valid" if math.isfinite(estimate) and estimate > 0
            else "nonpositive" if math.isfinite(estimate)
            else "nonfinite"
        )
        components.append(ComponentPrecisionEstimate(
            mass_class, p, spread, estimate, status,
        ))
    valid = sorted(item.implied_precision for item in components if item.status == "valid")
    if valid:
        middle = len(valid) // 2
        median = (
            valid[middle]
            if len(valid) % 2
            else 0.5 * (valid[middle - 1] + valid[middle])
        )
        deviations = sorted(abs(value - median) for value in valid)
        dm = len(deviations) // 2
        mad = (
            deviations[dm]
            if len(deviations) % 2
            else 0.5 * (deviations[dm - 1] + deviations[dm])
        )
        ratio = valid[-1] / valid[0]
        dispersion = PrecisionDispersion(
            len(valid), valid[0], median, valid[-1], ratio, mad,
        )
    else:
        dispersion = PrecisionDispersion(0, None, None, None, None, None)
    warnings = []
    invalid = tuple(item.mass_class for item in components if item.status != "valid")
    if invalid:
        warnings.append(
            "nonpositive or nonfinite component-wise precision at mass classes "
            + ", ".join(str(value) for value in invalid)
        )
    if len(valid) > 1 and valid[-1] > valid[0]:
        warnings.append(
            "component-wise precisions differ; inspect the reported dispersion rather "
            "than treating marginal-SE agreement as proof of a Dirichlet model"
        )
    assumptions = (
        "reported uncertainty convention is exactly the declared SD or SE convention",
        "replicate MIDs are independent when SE uses replicate_count",
        "Dirichlet marginal variance p_i(1-p_i)/(kappa+1) is the diagnostic model",
    )
    return ImpliedPrecisionDiagnostic(
        uncertainty_kind, count, tuple(components), dispersion, assumptions,
        tuple(warnings),
    )


def _replicate_vectors(replicates: object) -> tuple[tuple[float, ...], ...]:
    raw = _ordered_values(replicates, "replicate MID vectors")
    if len(raw) < 2:
        raise InputValidationError("replicate calibration requires at least two MID vectors")
    result = tuple(
        _simplex(value, f"replicate MID vector {index}", strictly_positive=True)
        for index, value in enumerate(raw)
    )
    dimension = len(result[0])
    if any(len(value) != dimension for value in result):
        raise InputValidationError("replicate MID vectors must share one declared dimension")
    return result


def _empirical_mean(replicates: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    count = len(replicates)
    return tuple(math.fsum(row[index] for row in replicates) / count
                 for index in range(len(replicates[0])))


def _empirical_covariance(
    replicates: tuple[tuple[float, ...], ...],
    center: tuple[float, ...],
    *,
    estimated_center: bool,
) -> tuple[tuple[float, ...], ...]:
    denominator = len(replicates) - 1 if estimated_center else len(replicates)
    return tuple(
        tuple(
            math.fsum((row[i] - center[i]) * (row[j] - center[j]) for row in replicates)
            / denominator
            for j in range(len(center))
        )
        for i in range(len(center))
    )


def _model_covariance(center: tuple[float, ...], precision: float) -> tuple[tuple[float, ...], ...]:
    denominator = precision + 1.0
    return tuple(
        tuple(
            (p_i * (1.0 - p_i) if i == j else -p_i * center[j]) / denominator
            for j in range(len(center))
        )
        for i, p_i in enumerate(center)
    )


def _matrix_residual(
    empirical: tuple[tuple[float, ...], ...],
    model: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], float, float]:
    residual = tuple(
        tuple(a - b for a, b in zip(left, right, strict=True))
        for left, right in zip(empirical, model, strict=True)
    )
    residual_norm = math.sqrt(math.fsum(value * value for row in residual for value in row))
    empirical_norm = math.sqrt(math.fsum(value * value for row in empirical for value in row))
    relative = residual_norm / empirical_norm if empirical_norm > 0 else math.inf
    maximum = max(abs(value) for row in residual for value in row)
    return residual, relative, maximum


@dataclass(frozen=True, slots=True)
class DirichletCovarianceDiagnostic:
    replicate_count: int
    replicate_semantics: str
    declared_center: tuple[float, ...]
    empirical_mean: tuple[float, ...]
    empirical_covariance: tuple[tuple[float, ...], ...]
    model_covariance: tuple[tuple[float, ...], ...]
    covariance_residual: tuple[tuple[float, ...], ...]
    covariance_relative_frobenius_error: float
    maximum_absolute_covariance_residual: float
    mean_root_mean_square_error: float
    component_precision_diagnostic: ImpliedPrecisionDiagnostic
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    suitable_for_rigorous_testing: bool = False


def diagnose_dirichlet_covariance(
    replicates: object,
    *,
    center: object,
    precision: Real,
    replicate_semantics: str,
) -> DirichletCovarianceDiagnostic:
    """Compare empirical replicate moments with a declared Dirichlet law."""

    values = _replicate_vectors(replicates)
    declared = _simplex(center, "declared Dirichlet center", strictly_positive=True)
    if len(declared) != len(values[0]):
        raise InputValidationError("declared center and replicate dimension differ")
    kappa = _positive_real(precision, "Dirichlet precision")
    if replicate_semantics not in DIRICHLET_REPLICATE_SEMANTICS:
        raise InputValidationError(
            "replicate_semantics must be one of "
            + ", ".join(DIRICHLET_REPLICATE_SEMANTICS)
        )
    empirical_mean = _empirical_mean(values)
    empirical_covariance = _empirical_covariance(
        values, empirical_mean, estimated_center=True,
    )
    model_covariance = _model_covariance(declared, kappa)
    residual, relative, maximum = _matrix_residual(
        empirical_covariance, model_covariance,
    )
    mean_rmse = math.sqrt(math.fsum(
        (observed - expected) ** 2
        for observed, expected in zip(empirical_mean, declared, strict=True)
    ) / len(declared))
    component_sd = tuple(
        math.sqrt(max(0.0, empirical_covariance[index][index]))
        for index in range(len(declared))
    )
    implied = implied_dirichlet_precision_from_uncertainty(
        declared, component_sd, replicate_count=len(values), uncertainty_kind="sd",
    )
    warnings = list(implied.warnings)
    if replicate_semantics == "biological_replicate_variability":
        warnings.append(
            "biological replicate variation must not be labelled analytical measurement precision"
        )
    elif replicate_semantics == "total_replicate_variability":
        warnings.append(
            "analytical and biological variability are aggregated and cannot be separated here"
        )
    assumptions = (
        "replicate vectors are independent observations for this diagnostic",
        "the declared center is fixed independently of covariance comparison",
        "a scalar concentration imposes covariance (diag(p)-p p^T)/(kappa+1)",
        "fit diagnostics are model checks, not proof of a Dirichlet law",
    )
    return DirichletCovarianceDiagnostic(
        len(values), replicate_semantics, declared, empirical_mean,
        empirical_covariance, model_covariance, residual, relative, maximum,
        mean_rmse, implied, assumptions, tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class ReplicateDirichletCalibration:
    """Scalar concentration estimate plus covariance and bootstrap diagnostics."""

    estimate: float | None
    method: str
    center: tuple[float, ...]
    center_source: str
    empirical_mean: tuple[float, ...]
    empirical_covariance: tuple[tuple[float, ...], ...]
    implied_covariance: tuple[tuple[float, ...], ...] | None
    covariance_relative_frobenius_error: float | None
    component_precision_diagnostic: ImpliedPrecisionDiagnostic
    replicate_count: int
    replicate_semantics: str
    precision_source: str
    independent_of_test_data: bool
    bootstrap_samples: int
    bootstrap_seed: int | None
    bootstrap_p_value: float | None
    bootstrap_failed_fits: int
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    suitable_for_rigorous_testing: bool


def _estimate_precision_core(
    replicates: tuple[tuple[float, ...], ...],
    center: tuple[float, ...],
    *,
    estimated_center: bool,
) -> tuple[float | None, tuple[tuple[float, ...], ...], float | None]:
    covariance = _empirical_covariance(
        replicates, center, estimated_center=estimated_center,
    )
    shape = tuple(
        tuple((p_i * (1.0 - p_i) if i == j else -p_i * center[j])
              for j in range(len(center)))
        for i, p_i in enumerate(center)
    )
    denominator = math.fsum(value * value for row in shape for value in row)
    scale = math.fsum(
        covariance[i][j] * shape[i][j]
        for i in range(len(center)) for j in range(len(center))
    ) / denominator
    if not math.isfinite(scale) or scale <= 0:
        return None, covariance, None
    estimate = 1.0 / scale - 1.0
    if not math.isfinite(estimate) or estimate <= 0:
        return None, covariance, estimate
    return estimate, covariance, scale


def estimate_dirichlet_precision_from_replicates(
    replicates: object,
    *,
    center: object | None = None,
    replicate_semantics: str,
    independent_of_test_data: bool,
    bootstrap_samples: int = 0,
    bootstrap_seed: int | None = None,
) -> ReplicateDirichletCalibration:
    """Estimate scalar ``kappa`` by covariance-shape method of moments.

    The fitted scale ``c=1/(kappa+1)`` is the Frobenius least-squares
    projection of the empirical covariance onto ``diag(p)-p p^T``.  If
    ``center`` is omitted, the replicate mean is declared as the fitted centre
    and unbiased sample covariance is used.  A supplied centre uses deviations
    around that fixed centre with divisor ``R``.

    Optional parametric bootstrap refits this same statistic with a deterministic
    seed and reports a goodness-of-fit p-value for covariance residual magnitude.
    It is a model diagnostic, never a proof.
    """

    values = _replicate_vectors(replicates)
    empirical_mean = _empirical_mean(values)
    if center is None:
        declared_center = empirical_mean
        if abs(math.fsum(declared_center) - 1.0) > DIRICHLET_SIMPLEX_TOLERANCE:
            raise DirichletNumericalError("replicate mean is not numerically on the simplex")
        center_source = "replicate_mean"
        estimated_center = True
    else:
        declared_center = _simplex(
            center, "declared Dirichlet center", strictly_positive=True,
        )
        if len(declared_center) != len(values[0]):
            raise InputValidationError("declared center and replicate dimension differ")
        center_source = "declared_center"
        estimated_center = False
    if any(value <= 0 for value in declared_center):
        raise InputValidationError(
            "replicate calibration centre must be strictly positive on its declared face"
        )
    if replicate_semantics not in DIRICHLET_REPLICATE_SEMANTICS:
        raise InputValidationError(
            "replicate_semantics must be one of "
            + ", ".join(DIRICHLET_REPLICATE_SEMANTICS)
        )
    if type(independent_of_test_data) is not bool:
        raise InputValidationError("independent_of_test_data must be an explicit bool")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, Integral) or bootstrap_samples < 0:
        raise InputValidationError("bootstrap_samples must be a nonnegative integer, not bool")
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples:
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral) or bootstrap_seed < 0:
            raise InputValidationError(
                "positive bootstrap_samples require an explicit nonnegative integer bootstrap_seed"
            )
        bootstrap_seed = int(bootstrap_seed)
    elif bootstrap_seed is not None:
        raise InputValidationError("bootstrap_seed is unused when bootstrap_samples is zero")

    estimate, covariance, _ = _estimate_precision_core(
        values, declared_center, estimated_center=estimated_center,
    )
    component_sd = tuple(
        math.sqrt(max(0.0, covariance[index][index]))
        for index in range(len(declared_center))
    )
    component_diagnostic = implied_dirichlet_precision_from_uncertainty(
        declared_center, component_sd, replicate_count=len(values), uncertainty_kind="sd",
    )
    warnings = list(component_diagnostic.warnings)
    implied_covariance = None
    relative_error = None
    bootstrap_p_value = None
    failed = 0
    if estimate is None:
        warnings.append(
            "the covariance-shape projection did not yield a positive finite precision"
        )
    else:
        implied_covariance = _model_covariance(declared_center, estimate)
        _, relative_error, observed_statistic = _matrix_residual(
            covariance, implied_covariance,
        )
        if bootstrap_samples:
            generator = np.random.default_rng(bootstrap_seed)
            greater_or_equal = 0
            alpha = np.asarray(declared_center) * estimate
            for _ in range(bootstrap_samples):
                sampled_array = generator.dirichlet(alpha, size=len(values))
                sampled = tuple(tuple(float(value) for value in row) for row in sampled_array)
                bootstrap_center = _empirical_mean(sampled) if estimated_center else declared_center
                fitted, bootstrap_covariance, _ = _estimate_precision_core(
                    sampled, bootstrap_center, estimated_center=estimated_center,
                )
                if fitted is None:
                    failed += 1
                    continue
                fitted_covariance = _model_covariance(bootstrap_center, fitted)
                _, _, statistic = _matrix_residual(
                    bootstrap_covariance, fitted_covariance,
                )
                if statistic >= observed_statistic:
                    greater_or_equal += 1
            usable = bootstrap_samples - failed
            bootstrap_p_value = (
                (greater_or_equal + 1.0) / (usable + 1.0) if usable else None
            )
            if failed:
                warnings.append(
                    f"{failed} bootstrap replicate fits were numerically unavailable"
                )
    if replicate_semantics == "biological_replicate_variability":
        warnings.append(
            "the estimate describes biological replicate variation, not analytical precision"
        )
    elif replicate_semantics == "total_replicate_variability":
        warnings.append(
            "analytical and biological contributions cannot be separated from these replicates"
        )
    if not independent_of_test_data:
        warnings.append(
            "same-data plug-in precision is diagnostic/model-conditional and does not inherit "
            "known-precision finite-sample guarantees"
        )
    precision_source = (
        "independent_calibration" if independent_of_test_data else "same_data_plugin"
    )
    assumptions = (
        "replicate vectors are independent for calibration",
        "all replicates share one compositional centre and scalar concentration",
        "covariance follows (diag(p)-p p^T)/(kappa+1)",
        "bootstrap goodness of fit is diagnostic rather than proof",
    )
    suitable = estimate is not None and independent_of_test_data
    return ReplicateDirichletCalibration(
        estimate, "scalar_covariance_frobenius_method_of_moments_v1",
        declared_center, center_source, empirical_mean, covariance,
        implied_covariance, relative_error, component_diagnostic, len(values),
        replicate_semantics, precision_source, independent_of_test_data,
        bootstrap_samples, bootstrap_seed, bootstrap_p_value, failed,
        assumptions, tuple(warnings), suitable,
    )


__all__ = [
    "DIRICHLET_NEAR_ONE_REFUSAL_RADIUS",
    "DIRICHLET_PRECISION_SOURCES",
    "DIRICHLET_REPLICATE_SEMANTICS",
    "DIRICHLET_SIMPLEX_TOLERANCE",
    "ComponentPrecisionEstimate",
    "DirichletCovarianceDiagnostic",
    "DirichletLogLikelihoodScore",
    "DirichletMIDLaw",
    "DirichletNumericalError",
    "ImpliedPrecisionDiagnostic",
    "MIDCorrectionProvenance",
    "PrecisionDispersion",
    "ReplicateDirichletCalibration",
    "diagnose_dirichlet_covariance",
    "dirichlet_pairwise_log_likelihood_score",
    "dirichlet_score_log_moment",
    "estimate_dirichlet_precision_from_replicates",
    "implied_dirichlet_precision_from_uncertainty",
    "kl_dirichlet",
    "renyi_dirichlet",
]

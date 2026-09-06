"""Immutable genuine-count observations and native stationary law declarations.

These records never infer counts or a count total from a probability MID,
percentage, peak area, intensity, or standard deviation. A count declaration
asserts that the measurement has literal fixed-total count semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Integral
from typing import TYPE_CHECKING

from fluxemu.exceptions import InputValidationError
from fluxemu.execution import CanonicalFluxState
from fluxemu.model import (
    CanonicalModel,
    StationaryExperimentSemantics,
    deterministic_serialise,
    validate_canonical_model,
    validate_stationary_experiment,
)

if TYPE_CHECKING:
    from fluxemu.flux_analysis.sampling import FluxSampleValidationReport
    from .multinomial import MultinomialMIDLaw


# NumPy's count sampler represents both its total and output in signed int64.
# This explicit boundary is shared by laws, raw observations, and declarations.
MAX_MULTINOMIAL_TOTAL = 2**63 - 1


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name} must be a nonempty string")


def _record_tuple(value: object, name: str, record_type: type) -> None:
    if not isinstance(value, tuple) or not value:
        raise InputValidationError(f"{name} must be a nonempty immutable tuple")
    if not all(isinstance(item, record_type) for item in value):
        raise InputValidationError(f"{name} must contain {record_type.__name__} records")


def _count_total(value: object, name: str = "n") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InputValidationError(f"{name} must be a positive integer count, not a float or bool")
    result = int(value)
    if not 1 <= result <= MAX_MULTINOMIAL_TOTAL:
        raise InputValidationError(
            f"{name} must be an integer in [1, {MAX_MULTINOMIAL_TOTAL}]"
        )
    return result


def _count_vector(
    counts: object, *, expected_size: int | None = None
) -> tuple[int, ...]:
    if not isinstance(counts, tuple) or not counts:
        raise InputValidationError("counts must be a nonempty immutable tuple")
    if expected_size is not None and len(counts) != expected_size:
        raise InputValidationError(
            f"counts have {len(counts)} mass classes; expected {expected_size}"
        )
    result: list[int] = []
    for index, value in enumerate(counts):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise InputValidationError(
                f"counts[{index}] must be a nonnegative integer, not a float or bool"
            )
        result.append(int(value))
    return tuple(result)


def _observation_digest(value: object) -> str:
    try:
        serialized = deterministic_serialise(value)
    except ValueError as error:
        raise InputValidationError(
            f"observation provenance is not canonical-serializable: {error}"
        ) from error
    return sha256(serialized.encode("utf-8")).hexdigest()


def _immutable_identity(value: object) -> bool:
    if isinstance(value, tuple):
        return all(_immutable_identity(item) for item in value)
    return value is None or isinstance(value, (str, bool, int, float))


def _validate_state_record(state: object) -> None:
    """Check immutable native record structure without inventing flux feasibility."""

    if not isinstance(state, CanonicalFluxState):
        raise InputValidationError("state must be CanonicalFluxState")
    if not _immutable_identity(state.sample_id):
        raise InputValidationError("sample_id must be a canonical scalar or recursively immutable tuple")
    if not isinstance(state.values, tuple) or not state.values:
        raise InputValidationError("state values must be a nonempty immutable tuple")
    reaction_ids: set[str] = set()
    for pair in state.values:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise InputValidationError("state values must contain immutable reaction/value pairs")
        reaction_id, value = pair
        _identifier(reaction_id, "reaction_id")
        if reaction_id in reaction_ids:
            raise InputValidationError("state values contain duplicate reaction IDs")
        reaction_ids.add(reaction_id)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputValidationError("state flux values must be finite real scalars, not bool")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise InputValidationError("state flux values must be finite real scalars")
    _observation_digest(state)


@dataclass(frozen=True, slots=True)
class MIDCountObservation:
    """One complete raw count vector with its explicit positive total.

    Floating-point values are rejected even when mathematically integral.
    ``empirical_mid`` is derived; raw counts and total remain available.
    Positions mean native mass classes M+0, M+1, ... in their given order.
    """

    counts: tuple[int, ...]
    n: int

    def __post_init__(self) -> None:
        counts = _count_vector(self.counts)
        n = _count_total(self.n)
        if sum(counts) != n:
            raise InputValidationError("observed counts must sum exactly to the explicit n")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "n", n)

    @property
    def empirical_mid(self) -> tuple[float, ...]:
        return tuple(value / self.n for value in self.counts)

    @property
    def mass_classes(self) -> tuple[int, ...]:
        return tuple(range(len(self.counts)))

    @property
    def fingerprint(self) -> str:
        return _observation_digest(("mid-count-observation-v1", self.n, self.counts, self.mass_classes))


@dataclass(frozen=True, slots=True)
class StationaryCountSpecification:
    """Explicit count total for one declared native target and replicate."""

    target_id: str
    total_count: int
    replicate_id: str = "0"

    def __post_init__(self) -> None:
        _identifier(self.target_id, "target_id")
        _identifier(self.replicate_id, "replicate_id")
        object.__setattr__(self, "total_count", _count_total(self.total_count, "total_count"))


@dataclass(frozen=True, slots=True)
class StationaryObservationExperiment:
    """One identified tracer experiment and its ordered count declarations."""

    experiment_id: str
    experiment: StationaryExperimentSemantics
    specifications: tuple[StationaryCountSpecification, ...]

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        if not isinstance(self.experiment, StationaryExperimentSemantics):
            raise InputValidationError("experiment must be StationaryExperimentSemantics")
        _record_tuple(self.specifications, "count specifications", StationaryCountSpecification)


@dataclass(frozen=True, slots=True)
class StationaryObservationSpecification:
    """Native model and ordered experiments, separate from any MFA objective."""

    model: CanonicalModel
    experiments: tuple[StationaryObservationExperiment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, CanonicalModel):
            raise InputValidationError("model must be a CanonicalModel")
        _record_tuple(self.experiments, "observation experiments", StationaryObservationExperiment)

    @property
    def fingerprint(self) -> str:
        return stationary_observation_specification_fingerprint(self)


def validate_stationary_observation_specification(
    specification: StationaryObservationSpecification,
) -> None:
    """Reuse native validation and check explicit experiment/target identities.

    Native stationary targets define the full ordered M+0 through M+c count
    space. There is no per-class selection, inferred total, or replicate mean.
    """

    if not isinstance(specification, StationaryObservationSpecification):
        raise InputValidationError("specification must be StationaryObservationSpecification")
    validate_canonical_model(specification.model)
    experiment_ids: set[str] = set()
    for block in specification.experiments:
        if block.experiment_id in experiment_ids:
            raise InputValidationError(f"duplicate experiment_id: {block.experiment_id!r}")
        experiment_ids.add(block.experiment_id)
        validate_stationary_experiment(specification.model, block.experiment)
        target_ids = {target.target_id for target in block.experiment.targets}
        target_ids.update(target.target_id for target in block.experiment.observation_targets)
        observation_ids: set[tuple[str, str]] = set()
        for item in block.specifications:
            identity = (item.target_id, item.replicate_id)
            context = f"experiment {block.experiment_id!r}, observation {identity!r}"
            if identity in observation_ids:
                raise InputValidationError(f"duplicate observation identity in {context}")
            observation_ids.add(identity)
            if item.target_id not in target_ids:
                raise InputValidationError(f"unknown target_id in {context}")


def stationary_observation_specification_fingerprint(
    specification: StationaryObservationSpecification,
) -> str:
    """Bind model, ordered experiment science, identities, and explicit totals."""

    validate_stationary_observation_specification(specification)
    return _observation_digest(("stationary-multinomial-specification-v1", specification))


@dataclass(frozen=True, slots=True)
class StationaryObservationLawComponent:
    """One native prediction/law with complete scientific identity and provenance."""

    state: CanonicalFluxState
    experiment_id: str
    target_id: str
    replicate_id: str
    law: MultinomialMIDLaw
    model_fingerprint: str
    experiment_fingerprint: str
    specification_fingerprint: str

    def __post_init__(self) -> None:
        from .multinomial import MultinomialMIDLaw

        _validate_state_record(self.state)
        if not isinstance(self.law, MultinomialMIDLaw):
            raise InputValidationError("law must be MultinomialMIDLaw")
        for name in (
            "experiment_id", "target_id", "replicate_id", "model_fingerprint",
            "experiment_fingerprint", "specification_fingerprint",
        ):
            _identifier(getattr(self, name), name)

    @property
    def sample_id(self) -> object:
        return self.state.sample_id

    @property
    def predicted_mid(self) -> tuple[float, ...]:
        return self.law.probabilities

    @property
    def mass_classes(self) -> tuple[int, ...]:
        return self.law.mass_classes

    @property
    def fingerprint(self) -> str:
        return _observation_digest((
            "stationary-observation-law-component-v1", self.state, self.experiment_id,
            self.target_id, self.replicate_id, self.model_fingerprint,
            self.experiment_fingerprint, self.specification_fingerprint,
            self.mass_classes, self.law.fingerprint,
        ))


@dataclass(frozen=True, slots=True)
class StationaryObservationLawResult:
    """Ordered state/experiment/replicate laws and native state validation."""

    states: tuple[CanonicalFluxState, ...]
    components: tuple[StationaryObservationLawComponent, ...]
    model_fingerprint: str
    experiment_fingerprints: tuple[tuple[str, str], ...]
    specification_fingerprint: str
    validation: FluxSampleValidationReport

    def __post_init__(self) -> None:
        from fluxemu.flux_analysis.sampling import FluxSampleValidationReport

        _record_tuple(self.states, "states", CanonicalFluxState)
        for state in self.states:
            _validate_state_record(state)
        _record_tuple(self.components, "law components", StationaryObservationLawComponent)
        if not isinstance(self.validation, FluxSampleValidationReport):
            raise InputValidationError("validation must be FluxSampleValidationReport")
        if not self.validation.valid or self.validation.sample_count != len(self.states):
            raise InputValidationError("law result requires successful native validation of its state batch")
        _identifier(self.model_fingerprint, "model_fingerprint")
        _identifier(self.specification_fingerprint, "specification_fingerprint")
        if not isinstance(self.experiment_fingerprints, tuple) or not self.experiment_fingerprints:
            raise InputValidationError("experiment fingerprints must be a nonempty immutable tuple")
        for pair in self.experiment_fingerprints:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise InputValidationError("experiment fingerprints must contain identity/digest pairs")
            _identifier(pair[0], "experiment_id")
            _identifier(pair[1], "experiment_fingerprint")


@dataclass(frozen=True, slots=True)
class StationaryCountSample:
    """Raw counts paired with their source law, prediction, and full identity."""

    component: StationaryObservationLawComponent
    observation: MIDCountObservation

    def __post_init__(self) -> None:
        if not isinstance(self.component, StationaryObservationLawComponent):
            raise InputValidationError("component must be StationaryObservationLawComponent")
        if not isinstance(self.observation, MIDCountObservation):
            raise InputValidationError("observation must be MIDCountObservation")
        if self.observation.n != self.component.law.n:
            raise InputValidationError("sample count total must match its source law")
        _count_vector(self.observation.counts, expected_size=len(self.component.predicted_mid))


__all__ = [
    "MAX_MULTINOMIAL_TOTAL", "MIDCountObservation", "StationaryCountSample",
    "StationaryCountSpecification", "StationaryObservationExperiment",
    "StationaryObservationLawComponent", "StationaryObservationLawResult",
    "StationaryObservationSpecification", "stationary_observation_specification_fingerprint",
    "validate_stationary_observation_specification",
]

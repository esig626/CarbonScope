"""Native stationary EMU predictions joined to Dirichlet MID laws."""

from __future__ import annotations

from dataclasses import dataclass

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import ForwardEMUError, InputValidationError, ValidationError
from fluxemu.execution import CanonicalFluxState, _validate_states
from fluxemu.flux_analysis.highs import prepare_highs_flux_region
from fluxemu.flux_analysis.sampling import (
    FluxSampleValidationReport,
    _require_valid_flux_states,
    validate_flux_states,
)
from fluxemu.model import (
    CanonicalModel,
    StationaryExperimentSemantics,
    validate_canonical_model,
    validate_stationary_experiment,
)

from .dirichlet import (
    DIRICHLET_PRECISION_SOURCES,
    DIRICHLET_REPLICATE_SEMANTICS,
    DirichletMIDLaw,
    MIDCorrectionProvenance,
    _identifier,
    _positive_integer,
    _positive_real,
)
from .schema import _observation_digest, _record_tuple, _validate_state_record


@dataclass(frozen=True, slots=True)
class StationaryDirichletBlockSpecification:
    """One target's explicit continuous-noise and replicate declaration."""

    target_id: str
    precision: float
    precision_source: str
    precision_provenance: str
    replicate_count: int
    replicate_semantics: str
    independent_replicates: bool
    replicate_id: str = "0"
    correction: MIDCorrectionProvenance | None = None

    def __post_init__(self) -> None:
        _identifier(self.target_id, "target_id")
        _identifier(self.replicate_id, "replicate_id")
        object.__setattr__(self, "precision", _positive_real(
            self.precision, "Dirichlet precision",
        ))
        if self.precision_source not in DIRICHLET_PRECISION_SOURCES:
            raise InputValidationError(
                "precision_source must be one of " + ", ".join(DIRICHLET_PRECISION_SOURCES)
            )
        _identifier(self.precision_provenance, "precision_provenance")
        count = _positive_integer(self.replicate_count, "replicate_count")
        object.__setattr__(self, "replicate_count", count)
        if self.replicate_semantics not in DIRICHLET_REPLICATE_SEMANTICS:
            raise InputValidationError(
                "replicate_semantics must be one of "
                + ", ".join(DIRICHLET_REPLICATE_SEMANTICS)
            )
        if type(self.independent_replicates) is not bool:
            raise InputValidationError("independent_replicates must be an explicit bool")
        if count > 1 and not self.independent_replicates:
            raise InputValidationError(
                "multiple Dirichlet replicates require explicit independent_replicates=True"
            )
        if self.correction is not None and not isinstance(
            self.correction, MIDCorrectionProvenance
        ):
            raise InputValidationError(
                "block correction must be MIDCorrectionProvenance or None"
            )


@dataclass(frozen=True, slots=True)
class StationaryDirichletObservationExperiment:
    experiment_id: str
    experiment: StationaryExperimentSemantics
    specifications: tuple[StationaryDirichletBlockSpecification, ...]

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        if not isinstance(self.experiment, StationaryExperimentSemantics):
            raise InputValidationError("experiment must be StationaryExperimentSemantics")
        _record_tuple(
            self.specifications,
            "Dirichlet block specifications",
            StationaryDirichletBlockSpecification,
        )


@dataclass(frozen=True, slots=True)
class StationaryDirichletObservationSpecification:
    model: CanonicalModel
    experiments: tuple[StationaryDirichletObservationExperiment, ...]
    correction: MIDCorrectionProvenance

    def __post_init__(self) -> None:
        validate_stationary_dirichlet_observation_specification(self)

    @property
    def fingerprint(self) -> str:
        return stationary_dirichlet_observation_specification_fingerprint(self)


def validate_stationary_dirichlet_observation_specification(
    specification: StationaryDirichletObservationSpecification,
) -> None:
    if not isinstance(specification, StationaryDirichletObservationSpecification):
        raise InputValidationError(
            "specification must be StationaryDirichletObservationSpecification"
        )
    if not isinstance(specification.model, CanonicalModel):
        raise InputValidationError("model must be CanonicalModel")
    validate_canonical_model(specification.model)
    _record_tuple(
        specification.experiments,
        "Dirichlet observation experiments",
        StationaryDirichletObservationExperiment,
    )
    if not isinstance(specification.correction, MIDCorrectionProvenance):
        raise InputValidationError("correction must be MIDCorrectionProvenance")
    experiment_ids = set()
    block_identities = set()
    for block in specification.experiments:
        if block.experiment_id in experiment_ids:
            raise InputValidationError(f"duplicate experiment_id: {block.experiment_id!r}")
        experiment_ids.add(block.experiment_id)
        validate_stationary_experiment(specification.model, block.experiment)
        target_ids = {target.target_id for target in block.experiment.targets}
        target_ids.update(target.target_id for target in block.experiment.observation_targets)
        for item in block.specifications:
            identity = (block.experiment_id, item.target_id, item.replicate_id)
            if identity in block_identities:
                raise InputValidationError(
                    f"duplicate Dirichlet observation identity: {identity!r}"
                )
            block_identities.add(identity)
            if item.target_id not in target_ids:
                raise InputValidationError(
                    f"unknown target_id in Dirichlet observation {identity!r}"
                )


def stationary_dirichlet_observation_specification_fingerprint(
    specification: StationaryDirichletObservationSpecification,
) -> str:
    validate_stationary_dirichlet_observation_specification(specification)
    return _observation_digest(("stationary-dirichlet-observation-specification-v1", specification))


@dataclass(frozen=True, slots=True)
class StationaryDirichletLawComponent:
    state: CanonicalFluxState
    experiment_id: str
    target_id: str
    replicate_id: str
    law: DirichletMIDLaw
    model_fingerprint: str
    experiment_fingerprint: str
    specification_fingerprint: str

    def __post_init__(self) -> None:
        _validate_state_record(self.state)
        if not isinstance(self.law, DirichletMIDLaw):
            raise InputValidationError("law must be DirichletMIDLaw")
        for name in (
            "experiment_id", "target_id", "replicate_id", "model_fingerprint",
            "experiment_fingerprint", "specification_fingerprint",
        ):
            _identifier(getattr(self, name), name)
        if self.law.observation_identity != (
            self.experiment_id, self.target_id, self.replicate_id,
        ):
            raise InputValidationError(
                "Dirichlet component identity differs from its law identity"
            )

    @property
    def sample_id(self) -> object:
        return self.state.sample_id

    @property
    def predicted_mid(self) -> tuple[float, ...]:
        return self.law.predicted_mid

    @property
    def mass_classes(self) -> tuple[int, ...]:
        return self.law.mass_classes

    @property
    def active_support(self) -> tuple[int, ...]:
        return self.law.active_support

    @property
    def fingerprint(self) -> str:
        return _observation_digest((
            "stationary-dirichlet-law-component-v1",
            self.state,
            self.experiment_id,
            self.target_id,
            self.replicate_id,
            self.model_fingerprint,
            self.experiment_fingerprint,
            self.specification_fingerprint,
            self.law.fingerprint,
        ))


@dataclass(frozen=True, slots=True)
class StationaryDirichletObservationLawResult:
    states: tuple[CanonicalFluxState, ...]
    components: tuple[StationaryDirichletLawComponent, ...]
    model_fingerprint: str
    experiment_fingerprints: tuple[tuple[str, str], ...]
    specification_fingerprint: str
    common_active_supports: tuple[tuple[tuple[str, str, str], tuple[int, ...]], ...]
    validation: FluxSampleValidationReport

    def __post_init__(self) -> None:
        _record_tuple(self.states, "states", CanonicalFluxState)
        for state in self.states:
            _validate_state_record(state)
        _record_tuple(
            self.components, "Dirichlet law components", StationaryDirichletLawComponent,
        )
        if not isinstance(self.validation, FluxSampleValidationReport):
            raise InputValidationError("validation must be FluxSampleValidationReport")
        if not self.validation.valid or self.validation.sample_count != len(self.states):
            raise InputValidationError(
                "Dirichlet law result requires successful native state validation"
            )
        _identifier(self.model_fingerprint, "model_fingerprint")
        _identifier(self.specification_fingerprint, "specification_fingerprint")
        if not isinstance(self.experiment_fingerprints, tuple) or not self.experiment_fingerprints:
            raise InputValidationError("experiment_fingerprints must be a nonempty tuple")
        for experiment_id, fingerprint in self.experiment_fingerprints:
            _identifier(experiment_id, "experiment_id")
            _identifier(fingerprint, "experiment_fingerprint")
        if not isinstance(self.common_active_supports, tuple) or not self.common_active_supports:
            raise InputValidationError("common_active_supports must be a nonempty tuple")


@dataclass(frozen=True, slots=True)
class _PredictionBatch:
    states: tuple[CanonicalFluxState, ...]
    rows: tuple[tuple[tuple[float, ...], ...], ...]
    plans: tuple[object, ...]
    validation: FluxSampleValidationReport


def _block_declarations(specification: StationaryDirichletObservationSpecification):
    return tuple(
        (experiment, item)
        for experiment in specification.experiments
        for item in experiment.specifications
    )


def _evaluate_prediction_batch(
    specification: StationaryDirichletObservationSpecification,
    states: tuple[CanonicalFluxState, ...],
) -> _PredictionBatch:
    validate_stationary_dirichlet_observation_specification(specification)
    if not isinstance(states, tuple) or not states:
        raise InputValidationError(
            "states must be a nonempty immutable tuple of CanonicalFluxState records"
        )
    for state in states:
        _validate_state_record(state)
    prepared = prepare_highs_flux_region(specification.model.flux_model, None)
    validation = validate_flux_states(prepared, states)
    _require_valid_flux_states(validation, "stationary Dirichlet observation-law state batch")
    _validate_states(specification.model, states)
    plans = tuple(
        compile_emu_plan(specification.model, block.experiment)
        for block in specification.experiments
    )
    predictions = []
    for block, plan in zip(specification.experiments, plans, strict=True):
        try:
            native = evaluate_stationary(plan, states)
        except (ForwardEMUError, InputValidationError, ValidationError) as error:
            raise type(error)(
                f"stationary Dirichlet experiment {block.experiment_id!r}: {error}"
            ) from error
        predictions.append({
            (item.sample_id, item.target_id): tuple(item.fractions)
            for item in native.forward.predictions
        })
    rows = []
    for state in states:
        state_rows = []
        for block, predicted in zip(specification.experiments, predictions, strict=True):
            for item in block.specifications:
                state_rows.append(predicted[(state.sample_id, item.target_id)])
        rows.append(tuple(state_rows))
    return _PredictionBatch(states, tuple(rows), plans, validation)


def _common_active_supports(
    specification: StationaryDirichletObservationSpecification,
    batches: tuple[_PredictionBatch, ...],
) -> tuple[tuple[tuple[str, str, str], tuple[int, ...]], ...]:
    declarations = _block_declarations(specification)
    result = []
    for block_index, (experiment, item) in enumerate(declarations):
        identity = (experiment.experiment_id, item.target_id, item.replicate_id)
        supports = []
        dimensions = set()
        for batch in batches:
            for row in batch.rows:
                predicted = row[block_index]
                dimensions.add(len(predicted))
                supports.append(tuple(index for index, value in enumerate(predicted) if value > 0))
        if len(dimensions) != 1:
            raise InputValidationError(
                f"Dirichlet block {identity!r} has inconsistent predicted MID dimensions"
            )
        unique_supports = tuple(dict.fromkeys(supports))
        if len(unique_supports) != 1:
            raise InputValidationError(
                f"Dirichlet block {identity!r} has state-dependent active supports "
                f"{unique_supports!r}; V1 refuses rather than adding epsilon"
            )
        active = unique_supports[0]
        if len(active) < 2:
            raise InputValidationError(
                f"Dirichlet block {identity!r} has fewer than two active coordinates"
            )
        result.append((identity, active))
    return tuple(result)


def _materialise_batch(
    specification: StationaryDirichletObservationSpecification,
    batch: _PredictionBatch,
    common_supports: tuple[tuple[tuple[str, str, str], tuple[int, ...]], ...],
) -> StationaryDirichletObservationLawResult:
    specification_fingerprint = specification.fingerprint
    declarations = _block_declarations(specification)
    if len(common_supports) != len(declarations):
        raise InputValidationError("common active support declaration count is inconsistent")
    components = []
    for state, row in zip(batch.states, batch.rows, strict=True):
        for block_index, ((experiment, item), predicted, support_record) in enumerate(zip(
            declarations, row, common_supports, strict=True,
        )):
            identity, active = support_record
            expected_identity = (experiment.experiment_id, item.target_id, item.replicate_id)
            if identity != expected_identity:
                raise InputValidationError(
                    "common active support identities differ from declared block order"
                )
            experiment_index = specification.experiments.index(experiment)
            plan = batch.plans[experiment_index]
            try:
                law = DirichletMIDLaw(
                    mass_classes=tuple(range(len(predicted))),
                    active_support=active,
                    predicted_mid=predicted,
                    precision=item.precision,
                    observation_identity=identity,
                    correction=item.correction or specification.correction,
                    precision_source=item.precision_source,
                    precision_provenance=item.precision_provenance,
                    replicate_count=item.replicate_count,
                    replicate_semantics=item.replicate_semantics,
                    independent_replicates=item.independent_replicates,
                )
            except (InputValidationError, ValidationError) as error:
                raise type(error)(
                    f"stationary Dirichlet block {identity!r}, state {state.sample_id!r}: {error}"
                ) from error
            components.append(StationaryDirichletLawComponent(
                state,
                identity[0], identity[1], identity[2],
                law,
                plan.model_fingerprint,
                plan.experiment_fingerprint,
                specification_fingerprint,
            ))
    return StationaryDirichletObservationLawResult(
        batch.states,
        tuple(components),
        batch.plans[0].model_fingerprint,
        tuple(
            (block.experiment_id, plan.experiment_fingerprint)
            for block, plan in zip(specification.experiments, batch.plans, strict=True)
        ),
        specification_fingerprint,
        common_supports,
        batch.validation,
    )


def evaluate_stationary_dirichlet_observation_families(
    specification: StationaryDirichletObservationSpecification,
    *,
    null_states: tuple[CanonicalFluxState, ...],
    alternative_states: tuple[CanonicalFluxState, ...],
) -> tuple[StationaryDirichletObservationLawResult, StationaryDirichletObservationLawResult]:
    """Jointly determine one active face per block across both represented roles."""

    null = _evaluate_prediction_batch(specification, null_states)
    alternative = _evaluate_prediction_batch(specification, alternative_states)
    supports = _common_active_supports(specification, (null, alternative))
    return (
        _materialise_batch(specification, null, supports),
        _materialise_batch(specification, alternative, supports),
    )


__all__ = [
    "StationaryDirichletBlockSpecification",
    "StationaryDirichletLawComponent",
    "StationaryDirichletObservationExperiment",
    "StationaryDirichletObservationLawResult",
    "StationaryDirichletObservationSpecification",
    "evaluate_stationary_dirichlet_observation_families",
    "stationary_dirichlet_observation_specification_fingerprint",
    "validate_stationary_dirichlet_observation_specification",
]

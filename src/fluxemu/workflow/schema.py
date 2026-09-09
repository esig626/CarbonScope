"""Immutable declarations for the native finite hypothesis workflow.

V1 hypotheses intersect reaction bounds with a common physical model.  The
region is the entire resulting steady-state polytope, with no retained
biological-optimum constraint.  Paths and source byte digests are provenance,
not scientific specification identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

from fluxemu.exceptions import InputValidationError
from fluxemu.model import (
    CanonicalModel, CanonicalModelError, deterministic_serialise, load_sbml_flux_model,
)
from fluxemu.native_io import load_native_stationary_spec
from fluxemu.observation import (
    StationaryCountSpecification, StationaryObservationExperiment,
    StationaryObservationSpecification, validate_stationary_observation_specification,
)

from ._yaml import load_unique_yaml, validate_native_experiment_document


PROCEDURES = (
    "composite_converse", "exact_minimax", "candidate_score",
    "analytical_score_bound", "deterministic_score_error", "calibrated_score_error",
)
SCORE_PROCEDURES = frozenset(PROCEDURES[2:])


def scientific_fingerprint(value: object) -> str:
    """Hash canonical ordered scientific values without runtime metadata."""

    return sha256(deterministic_serialise(value).encode("utf-8")).hexdigest()


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{name} must be a nonempty string")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise InputValidationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise InputValidationError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise InputValidationError(f"{name} must be an integer >= {minimum}, not bool")


def _records(value: object, name: str, record_type: type, *, nonempty: bool = True) -> None:
    if not isinstance(value, tuple) or (nonempty and not value):
        raise InputValidationError(f"{name} must be {'a nonempty' if nonempty else 'an'} immutable tuple")
    if not all(isinstance(item, record_type) for item in value):
        raise InputValidationError(f"{name} must contain {record_type.__name__} records")


@dataclass(frozen=True, slots=True)
class ReactionBoundConstraint:
    reaction_id: str
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.reaction_id, "reaction_id")
        if self.lower_bound is None and self.upper_bound is None:
            raise InputValidationError("reaction bound must declare lower_bound or upper_bound")
        for name in ("lower_bound", "upper_bound"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise InputValidationError("reaction lower_bound exceeds upper_bound")


@dataclass(frozen=True, slots=True)
class StateGenerationSpecification:
    method: str
    count: int
    seed: int
    burn_in: int = 100
    thinning: int = 10
    max_direction_attempts: int = 100

    def __post_init__(self) -> None:
        if self.method != "hit_and_run":
            raise InputValidationError("state-generation method must be exactly 'hit_and_run'")
        for name in ("count", "thinning", "max_direction_attempts"):
            _integer(getattr(self, name), name)
        for name in ("seed", "burn_in"):
            _integer(getattr(self, name), name, 0)


@dataclass(frozen=True, slots=True)
class HypothesisSpecification:
    description: str
    reaction_bounds: tuple[ReactionBoundConstraint, ...]
    state_generation: StateGenerationSpecification

    def __post_init__(self) -> None:
        _identifier(self.description, "hypothesis description")
        _records(self.reaction_bounds, "reaction_bounds", ReactionBoundConstraint, nonempty=False)
        identifiers = tuple(item.reaction_id for item in self.reaction_bounds)
        if len(identifiers) != len(set(identifiers)):
            raise InputValidationError("duplicate reaction constraint ID")
        if not isinstance(self.state_generation, StateGenerationSpecification):
            raise InputValidationError("state_generation must be StateGenerationSpecification")


@dataclass(frozen=True, slots=True)
class TestingSpecification:
    epsilon: float
    procedures: tuple[str, ...]
    converse_orders: tuple[float, ...] = ()
    score_orders: tuple[float, ...] = ()
    max_outcomes: int = 100_000

    def __post_init__(self) -> None:
        epsilon = _finite(self.epsilon, "epsilon")
        if not 0 < epsilon < 1:
            raise InputValidationError("epsilon must lie strictly between zero and one")
        object.__setattr__(self, "epsilon", epsilon)
        _records(self.procedures, "procedures", str)
        if len(self.procedures) != len(set(self.procedures)):
            raise InputValidationError("duplicate requested testing procedure")
        if any(item not in PROCEDURES for item in self.procedures):
            raise InputValidationError("unknown requested testing procedure")
        _integer(self.max_outcomes, "max_outcomes")
        for name, used in (
            ("converse_orders", "composite_converse" in self.procedures),
            ("score_orders", bool(SCORE_PROCEDURES.intersection(self.procedures))),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise InputValidationError(f"{name} must be an immutable tuple")
            normalized = tuple(_finite(value, name) for value in values)
            if len(set(normalized)) != len(normalized):
                raise InputValidationError(f"duplicate {name}")
            if any(not (value > 1 if name == "converse_orders" else 0 < value < 1) for value in normalized):
                interval = "> 1" if name == "converse_orders" else "in (0, 1)"
                raise InputValidationError(f"{name} must be finite and {interval}")
            if used and not normalized:
                raise InputValidationError(f"requested procedures require explicit {name}")
            if normalized and not used:
                raise InputValidationError(f"unused {name} are not allowed")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True, slots=True)
class WorkflowInputSource:
    kind: str
    identifier: str
    path: Path
    sha256: str
    forward_fva_fraction_of_optimum: float | None = None

    def __post_init__(self) -> None:
        for name in ("kind", "identifier", "sha256"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.path, Path):
            raise InputValidationError("input source path must be pathlib.Path")


@dataclass(frozen=True, slots=True)
class WorkflowSpecification:
    observation: StationaryObservationSpecification
    null: HypothesisSpecification
    alternative: HypothesisSpecification
    testing: TestingSpecification
    independent_blocks: bool
    output_directory: Path | None = None
    input_sources: tuple[WorkflowInputSource, ...] = ()

    def __post_init__(self) -> None:
        validate_hypothesis_testing_specification(self)

    @property
    def model(self) -> CanonicalModel:
        return self.observation.model

    @property
    def fingerprint(self) -> str:
        validate_hypothesis_testing_specification(self)
        return scientific_fingerprint((
            "native-hypothesis-workflow-v1", self.observation,
            ("H0", self.null), ("H1", self.alternative), self.testing,
            self.independent_blocks, "genuine_counts", "full-constrained-region",
        ))


def validate_hypothesis_testing_specification(specification: WorkflowSpecification) -> None:
    """Validate declarations; feasibility and region distinction require native LPs."""

    if not isinstance(specification, WorkflowSpecification):
        raise InputValidationError("specification must be WorkflowSpecification")
    if not isinstance(specification.observation, StationaryObservationSpecification):
        raise InputValidationError("observation must be StationaryObservationSpecification")
    try:
        validate_stationary_observation_specification(specification.observation)
    except CanonicalModelError as error:
        raise InputValidationError(f"invalid workflow model or isotope experiment: {error}") from error
    if type(specification.independent_blocks) is not bool:
        raise InputValidationError("independent_blocks must be an explicit boolean")
    block_count = sum(len(item.specifications) for item in specification.observation.experiments)
    if block_count > 1 and not specification.independent_blocks:
        raise InputValidationError("multiple observation blocks require explicit independence")
    if not isinstance(specification.testing, TestingSpecification):
        raise InputValidationError("testing must be TestingSpecification")
    physical = {item.reaction_id for item in specification.model.flux_model.reactions}
    for role, hypothesis in (("H0", specification.null), ("H1", specification.alternative)):
        if not isinstance(hypothesis, HypothesisSpecification):
            raise InputValidationError(f"{role} must be HypothesisSpecification")
        for constraint in hypothesis.reaction_bounds:
            if constraint.reaction_id not in physical:
                raise InputValidationError(f"{role} constraint references unknown reaction {constraint.reaction_id!r}")
    if specification.output_directory is not None and not isinstance(specification.output_directory, Path):
        raise InputValidationError("output_directory must be pathlib.Path or None")
    _records(specification.input_sources, "input_sources", WorkflowInputSource, nonempty=False)


def _mapping(value: object, name: str, allowed: set[str], required: set[str] = frozenset()) -> Mapping:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise InputValidationError(f"{name} must be a mapping with string keys")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise InputValidationError(f"{name} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise InputValidationError(f"{name} is missing field(s): {', '.join(sorted(missing))}")
    return value


def _list(value: object, name: str, *, nonempty: bool = True) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        raise InputValidationError(f"{name} must be {'a nonempty' if nonempty else 'a'} list")
    return value


def _path(value: object, base: Path, name: str) -> Path:
    _identifier(value, name)
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _hypothesis(raw: object, role: str) -> HypothesisSpecification:
    fields = {"description", "reaction_bounds", "state_generation"}
    item = _mapping(raw, role, fields, fields)
    constraints = []
    for bound in _list(item["reaction_bounds"], f"{role}.reaction_bounds", nonempty=False):
        bound = _mapping(bound, f"{role}.reaction_bounds", {"reaction_id", "lower_bound", "upper_bound"}, {"reaction_id"})
        if any(bound[key] is None for key in ("lower_bound", "upper_bound") if key in bound):
            raise InputValidationError("declared reaction bounds cannot be null; omit an unspecified bound")
        constraints.append(ReactionBoundConstraint(**bound))
    generation = _mapping(
        item["state_generation"], f"{role}.state_generation",
        {"method", "count", "seed", "burn_in", "thinning", "max_direction_attempts"},
        {"method", "count", "seed"},
    )
    return HypothesisSpecification(item["description"], tuple(constraints), StateGenerationSpecification(**generation))


def load_hypothesis_testing_spec(path: str | Path, *, model_path: str | Path | None = None) -> WorkflowSpecification:
    """Load workflow YAML and referenced SBML/FBC and ordered native experiments.

    YAML file references resolve relative to the workflow file; an explicit
    ``model_path`` override resolves relative to the caller's current directory.
    Existing experiment ``fva_fraction_of_optimum`` is recorded as forward-only
    source metadata and is never applied to either workflow hypothesis.
    """

    try:
        return _load_hypothesis_testing_spec(path, model_path=model_path)
    except CanonicalModelError as error:
        raise InputValidationError(f"invalid workflow model or isotope experiment: {error}") from error
    except OSError as error:
        raise InputValidationError(f"cannot read workflow input: {error}") from error


def _load_hypothesis_testing_spec(path: str | Path, *, model_path: str | Path | None) -> WorkflowSpecification:
    path = Path(path).resolve()
    raw = load_unique_yaml(path, "hypothesis workflow")
    fields = {"schema_version", "model", "experiments", "hypotheses", "observations", "testing", "output"}
    root = _mapping(raw, "workflow", fields, fields - {"model", "output"})
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise InputValidationError("workflow must declare integer schema_version: 1")
    if "model" in root:
        _identifier(root["model"], "model path")
    physical_path = Path(model_path).resolve() if model_path is not None else _path(root.get("model"), path.parent, "model path")
    flux_model = load_sbml_flux_model(physical_path)
    sources = [
        WorkflowInputSource("workflow", "workflow", path, sha256(path.read_bytes()).hexdigest()),
        WorkflowInputSource("model", "common", physical_path, sha256(physical_path.read_bytes()).hexdigest()),
    ]
    experiments = []
    common_model = None
    for item in _list(root["experiments"], "experiments"):
        item = _mapping(item, "experiment declaration", {"experiment_id", "specification", "counts"}, {"experiment_id", "specification", "counts"})
        _identifier(item["experiment_id"], "experiment_id")
        experiment_path = _path(item["specification"], path.parent, "experiment specification path")
        document = load_unique_yaml(experiment_path, "native experiment")
        validate_native_experiment_document(document)
        model, experiment, fraction = load_native_stationary_spec(experiment_path, flux_model)
        if common_model is None:
            common_model = model
        elif common_model != model:
            raise InputValidationError("all experiments must declare the same common isotope model and ordering")
        counts = []
        for count in _list(item["counts"], "experiment counts"):
            count = _mapping(count, "count declaration", {"target_id", "replicate_id", "total_count"}, {"target_id", "replicate_id", "total_count"})
            counts.append(StationaryCountSpecification(**count))
        experiments.append(StationaryObservationExperiment(item["experiment_id"], experiment, tuple(counts)))
        sources.append(WorkflowInputSource("experiment", item["experiment_id"], experiment_path, sha256(experiment_path.read_bytes()).hexdigest(), fraction))
    hypotheses = _mapping(root["hypotheses"], "hypotheses", {"H0", "H1"}, {"H0", "H1"})
    observations = _mapping(root["observations"], "observations", {"semantics", "independent_blocks"}, {"semantics", "independent_blocks"})
    if observations["semantics"] != "genuine_counts":
        raise InputValidationError("workflow V1 requires explicit genuine_counts observation semantics")
    testing = _mapping(root["testing"], "testing", {"epsilon", "procedures", "converse_orders", "score_orders", "max_outcomes"}, {"epsilon", "procedures"})
    testing_values = dict(testing)
    for key in ("procedures", "converse_orders", "score_orders"):
        if key in testing_values:
            testing_values[key] = tuple(_list(testing_values[key], key, nonempty=key == "procedures"))
    output = None
    if "output" in root:
        output_raw = _mapping(root["output"], "output", {"directory"}, {"directory"})
        output = _path(output_raw["directory"], path.parent, "output directory")
    return WorkflowSpecification(
        StationaryObservationSpecification(common_model, tuple(experiments)),
        _hypothesis(hypotheses["H0"], "H0"), _hypothesis(hypotheses["H1"], "H1"),
        TestingSpecification(**testing_values), observations["independent_blocks"], output, tuple(sources),
    )


__all__ = [
    "ReactionBoundConstraint", "StateGenerationSpecification", "HypothesisSpecification",
    "TestingSpecification", "WorkflowInputSource", "WorkflowSpecification",
    "load_hypothesis_testing_spec", "validate_hypothesis_testing_specification",
]

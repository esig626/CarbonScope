"""Strict experiment-YAML parsing for FluxEMU."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import yaml

from .exceptions import ConfigurationError


SamplerName = Literal["achr", "optgp"]
CorrectionMode = Literal["yes", "no"]
DEFAULT_TRACER_NORMALIZATION_TOLERANCE = 1e-9


def _nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty string")
    return value.strip()


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ConfigurationError(f"{context} must be a finite number")
    return result


def _positive_float(value: Any, context: str) -> float:
    result = _finite_float(value, context)
    if result <= 0:
        raise ConfigurationError(f"{context} must be greater than zero")
    return result


def _strict_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{context} must be an integer")
    if value < minimum:
        comparator = "positive" if minimum == 1 else f"at least {minimum}"
        raise ConfigurationError(f"{context} must be {comparator}")
    return value


def _correction_mode(value: Any, context: str) -> CorrectionMode:
    # PyYAML 1.1 resolves unquoted yes/no to booleans, so accept their exact
    # boolean equivalents while retaining the mfapy-facing yes/no spelling.
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return value.strip().lower()  # type: ignore[return-value]
    raise ConfigurationError(f"{context} must be 'yes' or 'no'")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context} keys must be strings")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{context} must be a list")
    return value


def _check_keys(
    data: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
    context: str,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(required - set(data))
    if missing:
        raise ConfigurationError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _one_alias(
    data: Mapping[str, Any], names: tuple[str, ...], context: str
) -> Any:
    present = [name for name in names if name in data]
    if not present:
        raise ConfigurationError(
            f"{context} is missing required field '{names[0]}'"
        )
    if len(present) > 1:
        raise ConfigurationError(
            f"{context} must use only one of: {', '.join(names)}"
        )
    return data[present[0]]


@dataclass(frozen=True)
class NumericalTolerances:
    """Numerical acceptance tolerances used throughout one pipeline run."""

    bounds: float = 1e-7
    mass_balance: float = 1e-7
    objective_floor: float = 1e-7
    mid: float = 1e-8
    tracer_normalization: float = DEFAULT_TRACER_NORMALIZATION_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "bounds",
            "mass_balance",
            "objective_floor",
            "mid",
            "tracer_normalization",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), f"tolerances.{name}"),
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "bounds": self.bounds,
            "mass_balance": self.mass_balance,
            "objective_floor": self.objective_floor,
            "mid": self.mid,
            "tracer_normalization": self.tracer_normalization,
        }


@dataclass(frozen=True)
class OutputSettings:
    """Formatting and destination-reuse settings for required outputs."""

    overwrite: bool = False
    float_precision: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.overwrite, bool):
            raise ConfigurationError("output.overwrite must be true or false")
        precision = _strict_int(
            self.float_precision, "output.float_precision", minimum=1
        )
        if precision > 17:
            raise ConfigurationError("output.float_precision must be at most 17")
        object.__setattr__(self, "float_precision", precision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overwrite": self.overwrite,
            "float_precision": self.float_precision,
        }


@dataclass(frozen=True)
class TracerMixture:
    """An mfapy-compatible isotopomer mixture for one source metabolite."""

    metabolite_id: str
    isotopomer_fractions: Mapping[str, float]
    correction: CorrectionMode = "no"
    normalization_tolerance: float = field(
        default=DEFAULT_TRACER_NORMALIZATION_TOLERANCE,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metabolite_id",
            _nonempty_text(self.metabolite_id, "tracer.metabolite_id"),
        )
        object.__setattr__(
            self,
            "correction",
            _correction_mode(self.correction, "tracer.correction"),
        )
        tolerance = _positive_float(
            self.normalization_tolerance, "tolerances.tracer_normalization"
        )
        object.__setattr__(self, "normalization_tolerance", tolerance)

        if not isinstance(self.isotopomer_fractions, Mapping):
            raise ConfigurationError("tracer.isotopomer_fractions must be a mapping")
        if not self.isotopomer_fractions:
            raise ConfigurationError(
                "tracer.isotopomer_fractions must contain at least one isotopomer"
            )

        normalized: dict[str, float] = {}
        carbon_counts: set[int] = set()
        for raw_pattern, raw_fraction in self.isotopomer_fractions.items():
            pattern = _nonempty_text(raw_pattern, "tracer isotopomer")
            if not pattern.startswith("#") or any(
                bit not in {"0", "1"} for bit in pattern[1:]
            ):
                raise ConfigurationError(
                    f"invalid isotopomer '{pattern}'; expected '#' followed by 0/1"
                )
            if len(pattern) == 1:
                raise ConfigurationError("an isotopomer must contain at least one atom")
            carbon_counts.add(len(pattern) - 1)
            fraction = _finite_float(
                raw_fraction, f"fraction for isotopomer {pattern}"
            )
            if fraction < 0 or fraction > 1:
                raise ConfigurationError(
                    f"fraction for isotopomer {pattern} must be between zero and one"
                )
            normalized[pattern] = fraction

        if len(carbon_counts) != 1:
            raise ConfigurationError(
                f"tracer '{self.metabolite_id}' has inconsistent isotopomer lengths"
            )
        total = sum(normalized.values())
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
            raise ConfigurationError(
                f"tracer '{self.metabolite_id}' isotopomer fractions sum to "
                f"{total:.17g}, not one"
            )
        object.__setattr__(
            self, "isotopomer_fractions", MappingProxyType(normalized)
        )

    @property
    def carbon_count(self) -> int:
        return len(next(iter(self.isotopomer_fractions))) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "metabolite_id": self.metabolite_id,
            "isotopomer_fractions": dict(self.isotopomer_fractions),
            "correction": self.correction,
        }


@dataclass(frozen=True)
class TargetFragment:
    """A requested metabolite fragment and its ordered carbon positions."""

    fragment_id: str
    metabolite_id: str
    atom_positions: tuple[int, ...]
    analytical_method: str
    formula: str
    correction: CorrectionMode = "no"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fragment_id",
            _nonempty_text(self.fragment_id, "target.fragment_id"),
        )
        object.__setattr__(
            self,
            "metabolite_id",
            _nonempty_text(self.metabolite_id, "target.metabolite_id"),
        )
        object.__setattr__(
            self,
            "analytical_method",
            _nonempty_text(
                self.analytical_method, "target.analytical_method"
            ).lower(),
        )
        object.__setattr__(
            self, "formula", _nonempty_text(self.formula, "target.formula")
        )
        object.__setattr__(
            self,
            "correction",
            _correction_mode(self.correction, "target.correction"),
        )

        if isinstance(self.atom_positions, (str, bytes)):
            raise ConfigurationError("target.atom_positions must be a list")
        try:
            positions = tuple(self.atom_positions)
        except TypeError as error:
            raise ConfigurationError("target.atom_positions must be a list") from error
        if not positions:
            raise ConfigurationError("target.atom_positions must not be empty")
        for position in positions:
            _strict_int(position, "target atom position", minimum=1)
        if len(set(positions)) != len(positions):
            raise ConfigurationError("target.atom_positions contains duplicates")
        object.__setattr__(self, "atom_positions", positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "metabolite_id": self.metabolite_id,
            "atom_positions": list(self.atom_positions),
            "analytical_method": self.analytical_method,
            "formula": self.formula,
            "correction": self.correction,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully validated experiment and sampling configuration."""

    tracers: tuple[TracerMixture, ...]
    targets: tuple[TargetFragment, ...]
    fraction_of_optimum: float
    sample_count: int
    sampler: SamplerName
    seed: int
    tolerances: NumericalTolerances = field(default_factory=NumericalTolerances)
    output: OutputSettings = field(default_factory=OutputSettings)
    schema_version: int = 1

    def __post_init__(self) -> None:
        tracers = tuple(self.tracers)
        targets = tuple(self.targets)
        if not tracers:
            raise ConfigurationError("tracers must contain at least one tracer")
        if not targets:
            raise ConfigurationError("targets must contain at least one target fragment")
        if not all(isinstance(item, TracerMixture) for item in tracers):
            raise ConfigurationError("tracers contains an invalid tracer entry")
        if not all(isinstance(item, TargetFragment) for item in targets):
            raise ConfigurationError("targets contains an invalid target entry")

        tracer_ids = [item.metabolite_id for item in tracers]
        if len(set(tracer_ids)) != len(tracer_ids):
            raise ConfigurationError("tracer metabolite IDs must be unique")
        target_ids = [item.fragment_id for item in targets]
        if len(set(target_ids)) != len(target_ids):
            raise ConfigurationError("target fragment IDs must be unique")

        fraction = _finite_float(
            self.fraction_of_optimum, "fraction_of_optimum"
        )
        if fraction <= 0 or fraction > 1:
            raise ConfigurationError(
                "fraction_of_optimum must be greater than zero and at most one"
            )
        sample_count = _strict_int(self.sample_count, "sample_count", minimum=1)
        sampler = _nonempty_text(self.sampler, "sampler").lower()
        if sampler not in {"achr", "optgp"}:
            raise ConfigurationError("sampler must be 'achr' or 'optgp'")
        seed = _strict_int(self.seed, "seed", minimum=0)
        if seed >= 2**31:
            raise ConfigurationError("seed must be smaller than 2**31")
        schema_version = _strict_int(
            self.schema_version, "schema_version", minimum=1
        )
        if schema_version != 1:
            raise ConfigurationError(
                f"unsupported experiment schema_version {schema_version}"
            )

        object.__setattr__(self, "tracers", tracers)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "fraction_of_optimum", fraction)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def tracer_by_metabolite(self) -> dict[str, TracerMixture]:
        return {tracer.metabolite_id: tracer for tracer in self.tracers}

    @property
    def target_by_id(self) -> dict[str, TargetFragment]:
        return {target.fragment_id: target for target in self.targets}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tracers": [tracer.to_dict() for tracer in self.tracers],
            "targets": [target.to_dict() for target in self.targets],
            "fraction_of_optimum": self.fraction_of_optimum,
            "sample_count": self.sample_count,
            "sampler": self.sampler,
            "seed": self.seed,
            "tolerances": self.tolerances.to_dict(),
            "output": self.output.to_dict(),
        }


def _parse_tolerances(value: Any) -> NumericalTolerances:
    if value is None:
        return NumericalTolerances()
    data = _mapping(value, "tolerances")
    allowed = {
        "bounds",
        "mass_balance",
        "objective_floor",
        "mid",
        "tracer_normalization",
    }
    _check_keys(data, allowed=allowed, context="tolerances")
    defaults = NumericalTolerances()
    return NumericalTolerances(
        bounds=data.get("bounds", defaults.bounds),
        mass_balance=data.get("mass_balance", defaults.mass_balance),
        objective_floor=data.get("objective_floor", defaults.objective_floor),
        mid=data.get("mid", defaults.mid),
        tracer_normalization=data.get(
            "tracer_normalization", defaults.tracer_normalization
        ),
    )


def _parse_output(value: Any) -> OutputSettings:
    if value is None:
        return OutputSettings()
    data = _mapping(value, "output")
    _check_keys(
        data,
        allowed={"overwrite", "float_precision"},
        context="output",
    )
    return OutputSettings(
        overwrite=data.get("overwrite", False),
        float_precision=data.get("float_precision", 12),
    )


def _parse_tracer(
    value: Any, index: int, tolerance: float
) -> TracerMixture:
    context = f"tracers[{index}]"
    data = _mapping(value, context)
    allowed = {
        "metabolite_id",
        "metabolite",
        "isotopomer_fractions",
        "isotopomers",
        "correction",
    }
    _check_keys(data, allowed=allowed, context=context)
    metabolite_id = _one_alias(
        data, ("metabolite_id", "metabolite"), context
    )
    fractions = _one_alias(
        data, ("isotopomer_fractions", "isotopomers"), context
    )
    return TracerMixture(
        metabolite_id=metabolite_id,
        isotopomer_fractions=_mapping(
            fractions, f"{context}.isotopomer_fractions"
        ),
        correction=data.get("correction", "no"),
        normalization_tolerance=tolerance,
    )


def _parse_target(value: Any, index: int) -> TargetFragment:
    context = f"targets[{index}]"
    data = _mapping(value, context)
    allowed = {
        "fragment_id",
        "id",
        "metabolite_id",
        "metabolite",
        "atom_positions",
        "atoms",
        "analytical_method",
        "formula",
        "correction",
    }
    _check_keys(
        data,
        allowed=allowed,
        required={"analytical_method", "formula"},
        context=context,
    )
    return TargetFragment(
        fragment_id=_one_alias(data, ("fragment_id", "id"), context),
        metabolite_id=_one_alias(
            data, ("metabolite_id", "metabolite"), context
        ),
        atom_positions=tuple(
            _sequence(
                _one_alias(data, ("atom_positions", "atoms"), context),
                f"{context}.atom_positions",
            )
        ),
        analytical_method=data["analytical_method"],
        formula=data["formula"],
        correction=data.get("correction", "no"),
    )


def parse_experiment_config(value: Any) -> ExperimentConfig:
    """Validate a decoded YAML object and return immutable dataclasses."""

    data = _mapping(value, "experiment")
    allowed = {
        "schema_version",
        "tracers",
        "targets",
        "fraction_of_optimum",
        "sample_count",
        "number_of_flux_samples",
        "sampler",
        "seed",
        "random_seed",
        "tolerances",
        "output",
        "output_settings",
    }
    _check_keys(
        data,
        allowed=allowed,
        required={"tracers", "targets", "fraction_of_optimum", "sampler"},
        context="experiment",
    )

    tolerances = _parse_tolerances(data.get("tolerances"))
    tracer_values = _sequence(data["tracers"], "tracers")
    target_values = _sequence(data["targets"], "targets")
    sample_count = _one_alias(
        data, ("sample_count", "number_of_flux_samples"), "experiment"
    )
    seed = _one_alias(data, ("seed", "random_seed"), "experiment")

    output_keys = [key for key in ("output", "output_settings") if key in data]
    if len(output_keys) > 1:
        raise ConfigurationError(
            "experiment must use only one of: output, output_settings"
        )
    output_value = data[output_keys[0]] if output_keys else None

    return ExperimentConfig(
        tracers=tuple(
            _parse_tracer(
                tracer, index, tolerances.tracer_normalization
            )
            for index, tracer in enumerate(tracer_values)
        ),
        targets=tuple(
            _parse_target(target, index)
            for index, target in enumerate(target_values)
        ),
        fraction_of_optimum=data["fraction_of_optimum"],
        sample_count=sample_count,
        sampler=data["sampler"],
        seed=seed,
        tolerances=tolerances,
        output=_parse_output(output_value),
        schema_version=data.get("schema_version", 1),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(
            f"could not read experiment YAML '{source}': {error}"
        ) from error
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"invalid YAML in experiment file '{source}': {error}"
        ) from error
    if value is None:
        raise ConfigurationError(f"experiment YAML '{source}' is empty")
    return parse_experiment_config(value)


# Short public spelling used by the pipeline and CLI.
load_experiment = load_experiment_config

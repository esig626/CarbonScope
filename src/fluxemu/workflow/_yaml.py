"""Lossless YAML ingestion checks for the public forward workflow.

The native experiment reader owns scientific binding and model validation.  Its
historical scalar conversions are guarded here so a workflow declaration cannot
silently turn a boolean into a count or a numeric identifier into a string.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from fluxemu.exceptions import InputValidationError


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader with duplicate detection, including YAML merge collisions."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "expected a mapping node", node.start_mark
            )
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "mapping keys must be hashable scalars", key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"duplicate mapping key {key!r}", key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_unique_yaml(path: str | Path, context: str) -> object:
    """Read safe YAML without permitting silent replacement of mapping keys."""
    try:
        content = Path(path).read_text(encoding="utf-8")
        return yaml.load(content, Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError,
            RecursionError) as exc:
        raise InputValidationError(f"cannot load {context} YAML: {exc}") from exc


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputValidationError(f"{context} must be a mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise InputValidationError(f"{context} keys must be nonempty strings")
    return value


def _record(value: object, context: str, *, required: set[str],
            optional: set[str] = frozenset()) -> dict[str, Any]:
    result = _mapping(value, context)
    missing = required - result.keys()
    if missing:
        raise InputValidationError(f"{context} missing required field(s): {', '.join(sorted(missing))}")
    unknown = result.keys() - required - optional
    if unknown:
        raise InputValidationError(f"{context} unknown field(s): {', '.join(sorted(unknown))}")
    return result


def _string(value: object, context: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{context} must be a nonempty string")


def _list(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise InputValidationError(f"{context} must be a list")
    return value


def _positive_integer(value: object, context: str) -> None:
    if type(value) is not int or value < 1:
        raise InputValidationError(f"{context} must be a positive integer, not a boolean")


def _finite_number(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{context} must be a finite number, not a boolean or string")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise InputValidationError(f"{context} must be a finite number")


def _positions(value: object, context: str) -> None:
    positions = _list(value, context)
    if not positions:
        raise InputValidationError(f"{context} must be nonempty")
    for index, position in enumerate(positions):
        _positive_integer(position, f"{context}[{index}]")


def _correction(record: dict[str, Any], context: str) -> None:
    if "correction" in record:
        value = record["correction"]
        if not isinstance(value, str) or value not in {"yes", "no"}:
            raise InputValidationError(
                f"{context}.correction must be the string 'yes' or 'no'; "
                "quote the value in YAML"
            )


def validate_native_experiment_document(raw: object) -> None:
    """Check native schema structure and literal types before native ingestion.

    Scientific constraints, authoritative mappings, atom ranges and probability
    sums remain the responsibility of the existing native implementation.
    """
    root = _record(raw, "native experiment", required={
        "schema_version", "isotope_model", "experiment",
    }, optional={"fva_fraction_of_optimum"})
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise InputValidationError("native experiment schema_version must be the integer 1")
    if "fva_fraction_of_optimum" in root:
        fraction = root["fva_fraction_of_optimum"]
        _finite_number(fraction, "fva_fraction_of_optimum")
        if not 0 < fraction <= 1:
            raise InputValidationError("fva_fraction_of_optimum must be in (0, 1]")

    isotope = _record(root["isotope_model"], "isotope_model", required={
        "metabolites", "assignments", "required_reactions",
    })
    metabolites = _mapping(isotope["metabolites"], "isotope_model.metabolites")
    if not metabolites:
        raise InputValidationError("isotope_model.metabolites must be nonempty")
    for metabolite_id, declaration in metabolites.items():
        context = f"isotope_model.metabolites.{metabolite_id}"
        metabolite = _record(declaration, context, required={"carbon_count"},
                             optional={"isotope_visible", "symmetry"})
        _positive_integer(metabolite["carbon_count"], f"{context}.carbon_count")
        for field in ("isotope_visible", "symmetry"):
            if field in metabolite and type(metabolite[field]) is not bool:
                raise InputValidationError(f"{context}.{field} must be a literal boolean")
    for index, declaration in enumerate(_list(isotope["assignments"], "isotope_model.assignments")):
        context = f"isotope_model.assignments[{index}]"
        assignment = _record(declaration, context, required={
            "reaction_id", "transition_id", "direction", "metabolite_map",
        })
        for field in ("reaction_id", "transition_id", "direction"):
            _string(assignment[field], f"{context}.{field}")
        if assignment["direction"] not in {"forward", "reverse"}:
            raise InputValidationError(f"{context}.direction must be forward or reverse")
        names = _mapping(assignment["metabolite_map"], f"{context}.metabolite_map")
        for participant, metabolite_id in names.items():
            _string(metabolite_id, f"{context}.metabolite_map.{participant}")
    for index, reaction_id in enumerate(_list(isotope["required_reactions"], "isotope_model.required_reactions")):
        _string(reaction_id, f"isotope_model.required_reactions[{index}]")

    experiment = _record(root["experiment"], "experiment", required=set(),
                         optional={"tracers", "targets", "observation_targets"})
    for index, declaration in enumerate(_list(experiment.get("tracers", []), "experiment.tracers")):
        context = f"experiment.tracers[{index}]"
        tracer = _record(declaration, context, required={"metabolite_id", "isotopomers"},
                         optional={"correction"})
        _string(tracer["metabolite_id"], f"{context}.metabolite_id")
        isotopomers = _mapping(tracer["isotopomers"], f"{context}.isotopomers")
        if not isotopomers:
            raise InputValidationError(f"{context}.isotopomers must be nonempty")
        for pattern, fraction in isotopomers.items():
            _finite_number(fraction, f"{context}.isotopomers[{pattern!r}]")
            if not 0 <= fraction <= 1:
                raise InputValidationError(f"{context}.isotopomers[{pattern!r}] must be in [0, 1]")
        _correction(tracer, context)
    for index, declaration in enumerate(_list(experiment.get("targets", []), "experiment.targets")):
        context = f"experiment.targets[{index}]"
        target = _record(declaration, context, required={
            "target_id", "metabolite_id", "atom_positions",
        }, optional={"analytical_method", "formula", "correction"})
        for field in ("target_id", "metabolite_id", "analytical_method", "formula"):
            if field in target:
                _string(target[field], f"{context}.{field}")
        _positions(target["atom_positions"], f"{context}.atom_positions")
        _correction(target, context)
    for index, declaration in enumerate(_list(experiment.get("observation_targets", []), "experiment.observation_targets")):
        context = f"experiment.observation_targets[{index}]"
        observation = _record(declaration, context, required={
            "target_id", "carbon_count", "precursors",
        })
        _string(observation["target_id"], f"{context}.target_id")
        _positive_integer(observation["carbon_count"], f"{context}.carbon_count")
        for precursor_index, declaration in enumerate(_list(observation["precursors"], f"{context}.precursors")):
            precursor_context = f"{context}.precursors[{precursor_index}]"
            precursor = _record(declaration, precursor_context,
                                required={"metabolite_id", "atom_positions"})
            _string(precursor["metabolite_id"], f"{precursor_context}.metabolite_id")
            _positions(precursor["atom_positions"], f"{precursor_context}.atom_positions")

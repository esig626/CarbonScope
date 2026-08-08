"""Order-preserving identity records for canonical isotope models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from cobra import Model

from .cobra_to_mfapy import MfapyModelBundle
from .configuration import ExperimentConfig
from .isotope_metadata import collect_isotope_metadata


FINGERPRINT_SCHEMA = "fluxemu.canonical_ordered_model.v1"


class ModelIdentityError(ValueError):
    """Raised when an ordered canonical model invariant is violated."""


@dataclass(frozen=True)
class InverseReactionConstraint:
    """Inverse-only working constraint, separate from canonical model bounds."""

    reaction_id: str
    parameter_type: str
    value: float
    lower_bound: float
    upper_bound: float
    stdev: float = 1.0

    def __post_init__(self) -> None:
        if not self.reaction_id:
            raise ModelIdentityError("inverse reaction ID must not be empty")
        if self.parameter_type not in {"free", "fixed", "fitting"}:
            raise ModelIdentityError(
                f"unsupported inverse parameter type {self.parameter_type!r}"
            )
        values = (self.value, self.lower_bound, self.upper_bound, self.stdev)
        if not all(isfinite(float(value)) for value in values):
            raise ModelIdentityError("inverse constraint values must be finite")
        if self.lower_bound > self.upper_bound:
            raise ModelIdentityError("inverse lower bound exceeds upper bound")
        if self.stdev <= 0.0:
            raise ModelIdentityError("inverse standard deviation must be positive")
        if self.parameter_type == "fixed" and not (
            self.lower_bound <= self.value <= self.upper_bound
        ):
            raise ModelIdentityError("fixed inverse value is outside its bounds")


@dataclass(frozen=True)
class InverseParameterConfiguration:
    """Ordered inverse-only constraints excluded from model identity."""

    reactions: tuple[InverseReactionConstraint, ...]

    def __post_init__(self) -> None:
        reactions = tuple(self.reactions)
        ids = [item.reaction_id for item in reactions]
        if len(ids) != len(set(ids)):
            raise ModelIdentityError("inverse reaction constraints contain duplicates")
        object.__setattr__(self, "reactions", reactions)


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, Mapping):
        return [
            {"field": str(key), "value": _json_compatible(item)}
            for key, item in value.items()
        ]
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    raise ModelIdentityError(
        f"unsupported value {type(value).__name__} in canonical fingerprint"
    )


def _metabolite_role(item: Any) -> str:
    if item.is_carbon_source:
        return "carbon_source"
    if item.is_excreted:
        return "product_boundary"
    return "internal"


def _ordered_target_selections(
    atommap: str, internal_to_original: Mapping[str, str]
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for fragment in atommap.split("+"):
        internal_id, positions = fragment.rsplit("_", 1)
        selections.append(
            {
                "metabolite_id": internal_to_original[internal_id],
                "internal_metabolite_id": internal_id,
                "atom_positions": [int(value) for value in positions.split(":")],
            }
        )
    return selections


def _coefficient(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


def ordered_model_fingerprint(
    cobra_model: Model,
    bundle: MfapyModelBundle,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    """Return a deterministic, order-preserving canonical model record.

    The record comes from canonical construction data, not mutable inverse
    parameter state on ``bundle.model``.
    """

    if not isinstance(cobra_model, Model):
        raise ModelIdentityError("ordered_model_fingerprint requires cobra.Model")
    metadata = collect_isotope_metadata(cobra_model)
    reaction_order = [reaction.id for reaction in cobra_model.reactions]
    metabolite_order = [metabolite.id for metabolite in cobra_model.metabolites]

    mapping_order = [
        item.original_cobra_reaction_id for item in bundle.reaction_mappings
    ]
    if mapping_order != reaction_order:
        raise ModelIdentityError(
            "reaction mapping order differs from canonical SBML reaction order"
        )
    internal_reaction_order = [
        item.internal_mfapy_reaction_id for item in bundle.reaction_mappings
    ]
    if list(bundle.model.reaction_ids) != internal_reaction_order:
        raise ModelIdentityError(
            "mfapy reaction order differs from the explicit reaction mapping"
        )

    expected_internal_metabolites = [
        bundle.metabolite_to_internal[metabolite_id]
        for metabolite_id in metabolite_order
    ]
    if list(bundle.metabolites) != expected_internal_metabolites:
        raise ModelIdentityError(
            "mfapy metabolite dictionary order differs from canonical SBML order"
        )

    metabolite_records: list[dict[str, Any]] = []
    steady_state_row_order: list[str] = []
    steady_state_internal_row_order: list[str] = []
    for index, metabolite in enumerate(cobra_model.metabolites):
        item = metadata.metabolites[metabolite.id]
        role = _metabolite_role(item)
        internal_id = bundle.metabolite_to_internal[metabolite.id]
        if role == "internal":
            steady_state_row_order.append(metabolite.id)
            steady_state_internal_row_order.append(internal_id)
        metabolite_records.append(
            {
                "metabolite_id": metabolite.id,
                "metabolite_index": index,
                "internal_metabolite_id": internal_id,
                "compartment": metabolite.compartment,
                "role": role,
                "carbon_count": item.carbon_count,
                "carbon_atom_order": list(range(1, item.carbon_count + 1)),
                "symmetry": bool(item.symmetry),
            }
        )
    if list(bundle.model.metabolite_ids) != steady_state_internal_row_order:
        raise ModelIdentityError(
            "mfapy steady-state metabolite order differs from canonical role order"
        )

    reaction_records: list[dict[str, Any]] = []
    mapping_by_original = {
        item.original_cobra_reaction_id: item for item in bundle.reaction_mappings
    }
    for index, reaction in enumerate(cobra_model.reactions):
        item = metadata.reactions[reaction.id]
        mapping = mapping_by_original[reaction.id]
        internal = bundle.reactions[mapping.internal_mfapy_reaction_id]
        reaction_records.append(
            {
                "reaction_id": reaction.id,
                "reaction_index": index,
                "internal_reaction_id": mapping.internal_mfapy_reaction_id,
                "mfapy_reaction_index": mapping.mfapy_reaction_order,
                "direction": item.direction,
                "direction_factor": mapping.direction,
                "directional_id": item.directional_id,
                "canonical_lower_bound": float(reaction.lower_bound),
                "canonical_upper_bound": float(reaction.upper_bound),
                "ordered_reactants": [
                    {
                        "metabolite_id": participant.metabolite_id,
                        "atom_labels": list(participant.atom_labels),
                    }
                    for participant in item.substrates
                ],
                "ordered_products": [
                    {
                        "metabolite_id": participant.metabolite_id,
                        "atom_labels": list(participant.atom_labels),
                    }
                    for participant in item.products
                ],
                "mfapy_stoichiometry": internal["stoichiometry"],
                "mfapy_atom_transition": internal["atommap"],
            }
        )

    full_values = [
        [
            _coefficient(reaction.metabolites.get(metabolite, 0.0))
            for reaction in cobra_model.reactions
        ]
        for metabolite in cobra_model.metabolites
    ]
    steady_positions = [metabolite_order.index(item) for item in steady_state_row_order]
    steady_values = [full_values[index] for index in steady_positions]

    internal_to_original = {
        internal: original
        for original, internal in bundle.metabolite_to_internal.items()
    }
    configured_targets = experiment.target_by_id
    target_order = list(bundle.target_fragments)
    if list(bundle.model.target_fragments) != target_order:
        raise ModelIdentityError(
            "mfapy target-fragment order differs from canonical bundle order"
        )
    target_records: list[dict[str, Any]] = []
    for index, internal_target_id in enumerate(target_order):
        definition = bundle.target_fragments[internal_target_id]
        selections = _ordered_target_selections(
            definition["atommap"], internal_to_original
        )
        component_count = sum(
            len(selection["atom_positions"]) for selection in selections
        ) + 1
        mfapy_component_count = int(
            bundle.model.target_fragments[internal_target_id]["number"]
        )
        if component_count != mfapy_component_count:
            raise ModelIdentityError(
                f"MID component count differs for target {internal_target_id!r}"
            )
        original_target_id = next(
            (
                original
                for original, internal in bundle.target_to_internal.items()
                if internal == internal_target_id
            ),
            internal_target_id,
        )
        configured = configured_targets.get(original_target_id)
        target_records.append(
            {
                "target_id": original_target_id,
                "target_index": index,
                "internal_target_id": internal_target_id,
                "analytical_method": definition["type"],
                "use": definition["use"],
                "formula": definition["formula"],
                "correction": configured.correction if configured else "no",
                "ordered_atom_selections": selections,
                "mid_component_order": [
                    f"M+{mass}" for mass in range(component_count)
                ],
            }
        )

    source_order_from_metadata = [
        metabolite.id
        for metabolite in cobra_model.metabolites
        if metadata.metabolites[metabolite.id].is_carbon_source
    ]
    tracer_order = [tracer.metabolite_id for tracer in experiment.tracers]
    if tracer_order != source_order_from_metadata:
        raise ModelIdentityError(
            "tracer order differs from canonical carbon-source metabolite order"
        )
    source_records: list[dict[str, Any]] = []
    for index, tracer in enumerate(experiment.tracers):
        item = metadata.metabolites[tracer.metabolite_id]
        source_records.append(
            {
                "metabolite_id": tracer.metabolite_id,
                "source_index": index,
                "internal_metabolite_id": bundle.metabolite_to_internal[
                    tracer.metabolite_id
                ],
                "carbon_atom_order": list(range(1, item.carbon_count + 1)),
                "ordered_isotopomers": [
                    {"pattern": pattern, "fraction": float(fraction)}
                    for pattern, fraction in tracer.isotopomer_fractions.items()
                ],
                "natural_isotope_correction": tracer.correction,
            }
        )

    reversible_records = [
        {
            "internal_reversible_id": reversible_id,
            "reversible_index": index,
            "ordered_definition_fields": _json_compatible(definition),
        }
        for index, (reversible_id, definition) in enumerate(
            bundle.reversible_reactions.items()
        )
    ]
    if list(bundle.model.reversible_ids) != [
        item["internal_reversible_id"] for item in reversible_records
    ]:
        raise ModelIdentityError("mfapy reversible-reaction order differs")

    return {
        "schema": FINGERPRINT_SCHEMA,
        "reaction_order": reaction_order,
        "internal_reaction_order": internal_reaction_order,
        "metabolite_order": metabolite_order,
        "internal_metabolite_order": expected_internal_metabolites,
        "steady_state_row_order": steady_state_row_order,
        "internal_steady_state_row_order": steady_state_internal_row_order,
        "stoichiometric_matrix": {
            "row_order": metabolite_order,
            "column_order": reaction_order,
            "values": full_values,
        },
        "steady_state_stoichiometric_matrix": {
            "row_order": steady_state_row_order,
            "column_order": reaction_order,
            "values": steady_values,
        },
        "metabolites": metabolite_records,
        "reactions": reaction_records,
        "reversible_reaction_order": list(bundle.reversible_reactions),
        "reversible_reactions": reversible_records,
        "target_fragment_order": target_order,
        "configured_target_order": [
            target.fragment_id for target in experiment.targets
        ],
        "target_fragments": target_records,
        "carbon_source_order": tracer_order,
        "carbon_sources": source_records,
        "mfapy_dynamic_metabolite_order": list(
            bundle.model.dynamic_metabolite_ids
        ),
        "mfapy_emu_order_in_X": _json_compatible(bundle.model.emu_order_in_X),
        "mfapy_emu_order_in_y": _json_compatible(bundle.model.emu_order_in_y),
        "mfapy_carbon_source_emu_order": list(bundle.model.carbon_source_emu),
        "mfapy_carbon_source_emus": [
            {
                "emu_id": emu_id,
                "ordered_definition_fields": _json_compatible(definition),
            }
            for emu_id, definition in bundle.model.carbon_source_emu.items()
        ],
        "generated_calmdv_sha256": sha256(
            bundle.model.calmdv_text.encode("utf-8")
        ).hexdigest(),
    }


def serialize_ordered_fingerprint(fingerprint: Mapping[str, Any]) -> str:
    """Serialize a fingerprint without reordering any mapping keys."""

    return json.dumps(
        fingerprint,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def fingerprint_sha256(fingerprint: Mapping[str, Any]) -> str:
    """Return SHA-256 of the stable ordered fingerprint serialization."""

    return sha256(serialize_ordered_fingerprint(fingerprint).encode("utf-8")).hexdigest()


def first_ordered_fingerprint_mismatch(
    left: Any, right: Any, path: str = "$"
) -> dict[str, Any] | None:
    """Return the first exact value or ordering mismatch, if one exists."""

    if type(left) is not type(right):
        return {
            "path": path,
            "reason": "type mismatch",
            "forward": type(left).__name__,
            "inverse": type(right).__name__,
        }
    if isinstance(left, Mapping):
        left_keys = list(left)
        right_keys = list(right)
        if left_keys != right_keys:
            return {
                "path": path,
                "reason": "field order mismatch",
                "forward": left_keys,
                "inverse": right_keys,
            }
        for key in left_keys:
            mismatch = first_ordered_fingerprint_mismatch(
                left[key], right[key], f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {
                "path": path,
                "reason": "list length mismatch",
                "forward": len(left),
                "inverse": len(right),
            }
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = first_ordered_fingerprint_mismatch(
                left_item, right_item, f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if left != right:
        return {
            "path": path,
            "reason": "value mismatch",
            "forward": left,
            "inverse": right,
        }
    return None


def fingerprint_document(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a canonical fingerprint with its digest."""

    return {
        "sha256": fingerprint_sha256(fingerprint),
        "fingerprint": fingerprint,
    }


def write_fingerprint_document(
    path: str | Path, fingerprint: Mapping[str, Any]
) -> None:
    """Write one readable fingerprint document without changing field order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            fingerprint_document(fingerprint),
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )


def apply_inverse_parameter_configuration(
    bundle: MfapyModelBundle, configuration: InverseParameterConfiguration
) -> None:
    """Apply explicitly separate inverse constraints to mfapy's working copy."""

    mapping = {
        item.original_cobra_reaction_id: item.internal_mfapy_reaction_id
        for item in bundle.reaction_mappings
    }
    configured_order = [item.reaction_id for item in configuration.reactions]
    canonical_order = [
        item.original_cobra_reaction_id for item in bundle.reaction_mappings
    ]
    if configured_order != canonical_order:
        raise ModelIdentityError(
            "inverse constraints must cover reactions in canonical reaction order"
        )
    for item in configuration.reactions:
        internal_id = mapping[item.reaction_id]
        if bundle.model.set_constraint(
            "reaction",
            internal_id,
            item.parameter_type,
            value=item.value,
            stdev=item.stdev,
        ) is not True:
            raise ModelIdentityError(
                f"mfapy rejected inverse constraint for {item.reaction_id!r}"
            )
        if bundle.model.set_boundary(
            "reaction", internal_id, item.lower_bound, item.upper_bound
        ) is not True:
            raise ModelIdentityError(
                f"mfapy rejected inverse bounds for {item.reaction_id!r}"
            )
    if bundle.model.update() is not True:
        raise ModelIdentityError("mfapy failed to update inverse constraints")


__all__ = [
    "FINGERPRINT_SCHEMA",
    "InverseParameterConfiguration",
    "InverseReactionConstraint",
    "ModelIdentityError",
    "apply_inverse_parameter_configuration",
    "fingerprint_document",
    "fingerprint_sha256",
    "first_ordered_fingerprint_mismatch",
    "ordered_model_fingerprint",
    "serialize_ordered_fingerprint",
    "write_fingerprint_document",
]

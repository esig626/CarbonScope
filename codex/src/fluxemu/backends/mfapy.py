"""Compile canonical stationary science into the temporary mfapy backend."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import string
from typing import Any

from .._mfapy import load_mfapy
from ..exceptions import ForwardEMUError, MappingError
from ..execution import (
    CanonicalFluxState,
    StationaryForwardResult,
    StationaryMID,
    StationaryMIDValue,
)
from ..mfapy_model import BoundaryTargetMetabolicModel
from ..model import AtomPosition, CanonicalModel, IsotopeReaction, StationaryExperimentSemantics
from ..validation import validate_mid_batch


_ATOM_LABELS = string.ascii_lowercase + string.ascii_uppercase + string.digits


def _internal(kind: str, identifier: str) -> str:
    return f"f{kind}{sha256(identifier.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class CompiledBranch:
    reaction_id: str
    direction: str
    branch_id: str
    weight: float
    substrates: tuple[tuple[str, tuple[int, ...]], ...]
    products: tuple[tuple[str, tuple[int, ...]], ...]
    transitions: tuple[tuple[str, int, str, int], ...]
    internal_reaction_id: str


@dataclass(frozen=True, slots=True)
class CompiledMfapyModel:
    model: Any
    branches: tuple[CompiledBranch, ...]
    metabolite_ids: dict[str, str]
    target_ids: dict[str, str]


def _side(reaction_side: tuple[Any, ...], metabolite_ids: dict[str, str]) -> str:
    return "+".join(metabolite_ids[item.metabolite_id] for item in reaction_side)


def _branch_atommap(reaction: IsotopeReaction, branch: Any) -> str:
    sources = tuple(
        AtomPosition(participant.metabolite_id, position)
        for participant in reaction.substrates
        for position in participant.atom_positions
    )
    if len(sources) > len(_ATOM_LABELS):
        raise MappingError(
            f"reaction {reaction.reaction_id!r} exceeds mfapy's single-character atom-label capacity"
        )
    labels = dict(zip(sources, _ATOM_LABELS))
    destination_sources = {item.destination: item.source for item in branch.transitions}

    def text(participants: tuple[Any, ...], *, products: bool) -> str:
        groups: list[str] = []
        for participant in participants:
            atoms = [AtomPosition(participant.metabolite_id, position) for position in participant.atom_positions]
            try:
                groups.append(
                    "".join(labels[destination_sources[atom]] if products else labels[atom] for atom in atoms)
                )
            except KeyError as error:  # validation should make this unreachable
                raise MappingError(
                    f"mapping branch {branch.branch_id!r} in reaction {reaction.reaction_id!r} is incomplete"
                ) from error
        return "+".join(groups)

    return f"{text(reaction.substrates, products=False)}-->{text(reaction.products, products=True)}"


def compile_stationary_model(
    canonical: CanonicalModel, experiment: StationaryExperimentSemantics
) -> CompiledMfapyModel:
    """Compile records without consulting COBRA or legacy isotope metadata."""

    flux_by_id = {item.reaction_id: item for item in canonical.flux_model.reactions}
    isotope_metabolites = tuple(item for item in canonical.isotope_model.metabolites if item.isotope_visible)
    metabolite_ids = {
        item.metabolite_id: _internal("m", item.metabolite_id) for item in isotope_metabolites
    }
    target_ids = {item.target_id: _internal("t", item.target_id) for item in experiment.targets}
    tracer_ids = {item.metabolite_id for item in experiment.tracers}
    balanced = {
        item.metabolite_id: item.steady_state_balanced for item in canonical.flux_model.metabolites
    }

    metabolites: dict[str, dict[str, Any]] = {}
    for order, item in enumerate(isotope_metabolites):
        if item.metabolite_id not in balanced:
            raise MappingError(f"isotope metabolite {item.metabolite_id!r} has no flux-model record")
        metabolites[metabolite_ids[item.metabolite_id]] = {
            "C_number": item.carbon_count,
            # Explicit canonical branches carry symmetry; mfapy must not invent it.
            "symmetry": "no",
            "carbonsource": "carbonsource" if item.metabolite_id in tracer_ids else "no",
            "excreted": "no" if balanced[item.metabolite_id] else "excreted",
            "order": order,
            "externalids": f"fluxemu:{item.metabolite_id}",
            "lb": 0.0,
            "ub": 1_000_000.0,
        }

    reactions: dict[str, dict[str, Any]] = {}
    compiled: list[CompiledBranch] = []
    order = 0
    seen_isotope_reactions: set[str] = set()
    for reaction in canonical.isotope_model.reactions:
        if not reaction.isotope_enabled:
            continue
        if reaction.reaction_id in seen_isotope_reactions or reaction.reaction_id not in flux_by_id:
            raise MappingError(f"isotope reaction {reaction.reaction_id!r} has no unique flux reaction")
        seen_isotope_reactions.add(reaction.reaction_id)
        direction_factor = -1.0 if reaction.direction == "reverse" else 1.0
        flux_reaction = flux_by_id[reaction.reaction_id]
        directed_upper = max(
            direction_factor * float(flux_reaction.lower_bound),
            direction_factor * float(flux_reaction.upper_bound),
        )
        for branch_index, branch in enumerate(reaction.mapping_branches):
            internal_id = _internal(
                "r", f"{reaction.reaction_id}\0{branch_index}\0{branch.branch_id}"
            )
            substrate_text = _side(reaction.substrates, metabolite_ids)
            product_text = _side(reaction.products, metabolite_ids)
            reactions[internal_id] = {
                "stoichiometry": f"{substrate_text}-->{product_text}",
                "reaction": f"{substrate_text}-->{product_text}",
                "atommap": _branch_atommap(reaction, branch),
                "externalids": f"fluxemu:{reaction.reaction_id}:{branch.branch_id}",
                "order": order,
                "lb": 0.0,
                "ub": max(0.0, directed_upper * float(branch.weight)),
            }
            compiled.append(
                CompiledBranch(
                    reaction.reaction_id,
                    reaction.direction,
                    branch.branch_id,
                    float(branch.weight),
                    tuple((item.metabolite_id, item.atom_positions) for item in reaction.substrates),
                    tuple((item.metabolite_id, item.atom_positions) for item in reaction.products),
                    tuple(
                        (
                            item.source.metabolite_id,
                            item.source.position,
                            item.destination.metabolite_id,
                            item.destination.position,
                        )
                        for item in branch.transitions
                    ),
                    internal_id,
                )
            )
            order += 1
    if not reactions:
        raise MappingError("canonical model contains no enabled isotope reactions")

    targets: dict[str, dict[str, Any]] = {}
    for order, target in enumerate(experiment.targets):
        if target.metabolite_id not in metabolite_ids:
            raise MappingError(f"target {target.target_id!r} is missing from the compiled backend")
        positions = ":".join(str(item) for item in target.atom_positions)
        targets[target_ids[target.target_id]] = {
            "type": target.analytical_method,
            "atommap": f"{metabolite_ids[target.metabolite_id]}_{positions}",
            "use": "use",
            "order": order,
            "formula": target.formula,
        }
    try:
        backend_model = BoundaryTargetMetabolicModel(reactions, {}, metabolites, targets)
    except Exception as error:
        raise ForwardEMUError(f"mfapy canonical model construction failed: {error}") from error
    expected = tuple(item.internal_reaction_id for item in compiled)
    if tuple(backend_model.reaction_ids) != expected:
        raise MappingError("mfapy changed canonical branch ordering")
    return CompiledMfapyModel(backend_model, tuple(compiled), metabolite_ids, target_ids)


def execute_stationary(
    canonical: CanonicalModel,
    experiment: StationaryExperimentSemantics,
    states: tuple[CanonicalFluxState, ...],
    tolerance: float,
) -> StationaryForwardResult:
    compiled = compile_stationary_model(canonical, experiment)
    carbon_source = compiled.model.generate_carbon_source_template()
    for tracer in experiment.tracers:
        accepted = carbon_source.set_each_isotopomer(
            compiled.metabolite_ids[tracer.metabolite_id], dict(tracer.isotopomers), correction=tracer.correction
        )
        if accepted is not True:
            raise ForwardEMUError(f"mfapy rejected tracer {tracer.metabolite_id!r}")
    source_mdvs = carbon_source.generate_dict()
    mfapy = load_mfapy()
    requested = [compiled.target_ids[item.target_id] for item in experiment.targets]
    predictions: dict[Any, dict[str, list[float]]] = {}
    for state in states:
        flux_by_id = dict(state.values)
        vector = [
            flux_by_id[item.reaction_id]
            * (-1.0 if item.direction == "reverse" else 1.0)
            * item.weight
            for item in compiled.branches
        ]
        if any(item < -1e-9 for item in vector):
            raise MappingError(f"flux state {state.sample_id!r} opposes a declared isotope direction")
        try:
            _, raw = mfapy.optimize.calc_MDV_from_flux(
                vector, requested, source_mdvs, compiled.model.func
            )
        except Exception as error:
            raise ForwardEMUError(
                f"mfapy canonical forward calculation failed for {state.sample_id!r}: {error}"
            ) from error
        sample: dict[str, list[float]] = {}
        for target in experiment.targets:
            internal = compiled.target_ids[target.target_id]
            if internal in raw:
                sample[target.target_id] = [float(item) for item in raw[internal]]
        predictions[state.sample_id] = sample

    summary = validate_mid_batch(
        predictions,
        {item.target_id: len(item.atom_positions) + 1 for item in experiment.targets},
        tolerance,
    )
    mids: list[StationaryMID] = []
    values: list[StationaryMIDValue] = []
    for state in states:
        for target in experiment.targets:
            fractions = tuple(predictions[state.sample_id][target.target_id])
            mids.append(StationaryMID(state.sample_id, target.target_id, fractions))
            values.extend(
                StationaryMIDValue(state.sample_id, target.target_id, index, fraction)
                for index, fraction in enumerate(fractions)
            )
    return StationaryForwardResult(
        tuple(mids), tuple(values), tolerance, summary["max_normalization_error"]
    )


__all__ = ["CompiledBranch", "CompiledMfapyModel", "compile_stationary_model"]

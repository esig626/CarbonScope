"""Reproduce the Antoniewicz–Kelleher–Stephanopoulos TCA EMU benchmark.

The transcribed network is limited to the eight Table 5 reactions.  It is an
isolated example: no E. coli artefact, mapping database, or general transition
library is used.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from cobra import Metabolite, Model, Reaction
from cobra.io import read_sbml_model, write_sbml_model

from fluxemu.mfapy_model import BoundaryTargetMetabolicModel
from fluxemu.cobra_to_mfapy import MfapyModelBundle
from fluxemu.configuration import ExperimentConfig, load_experiment, parse_experiment_config
from fluxemu.flux_conversion import FluxConverter, ReactionMapping
from fluxemu.forward import run_batch_forward
from fluxemu.isotope_metadata import (
    AtomMappedParticipant,
    MetaboliteIsotopeMetadata,
    ReactionIsotopeMetadata,
    collect_isotope_metadata,
    set_metabolite_metadata,
    set_reaction_metadata,
)

from direct_isotopomer_solver import FLUXES, glutamate_mid


EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR / "antoniewicz_tca.xml"
STATIONARY_EXPERIMENT = EXAMPLE_DIR / "experiment_stationary.yaml"
TIMECOURSE_EXPERIMENT = EXAMPLE_DIR / "experiment_timecourse.yaml"
PUBLISHED_REFERENCE = EXAMPLE_DIR / "published_reference_mid.csv"
STATIONARY_MIDS = EXAMPLE_DIR / "stationary_mids.csv"
TIMECOURSE_MIDS = EXAMPLE_DIR / "timecourse_mids.csv"

PAPER_REFERENCE = (
    "Antoniewicz, Kelleher and Stephanopoulos, Metabolic Engineering; "
    "uploaded NIH manuscript nihms16238.pdf"
)
TABLE5_PAGE = "paper page 37, Table 5"
TABLE6_PAGE = "paper page 38, Table 6"
FIGURE12_PAGE = "paper page 29, Figure 12"
SECTION32_PAGES = "paper pages 12-14, Section 3.2"

METABOLITE_CARBONS: Mapping[str, int] = {
    "OAC": 4,
    "AcCoA": 2,
    "citrate": 6,
    "AKG": 5,
    "glutamate": 5,
    "succinate": 4,
    "fumarate": 4,
    "aspartate": 4,
    "CO2": 1,
}
CARBON_SOURCES = {"AcCoA", "aspartate"}
PRODUCT_BOUNDARIES = {"CO2", "glutamate"}
SYMMETRIC_METABOLITES = {"succinate", "fumarate"}
CANONICAL_FLUX_BOUNDS = (0.0, 1_000_000.0)
CANONICAL_REACTION_ORDER = tuple(f"v{number}" for number in range(1, 9))
GROUND_TRUTH_FLUXES: Mapping[str, float] = MappingProxyType(dict(FLUXES))

PUBLISHED_TABLE6_MID = np.array(
    (0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039), dtype=float
)
TIMEPOINTS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0)
# Numerically convenient, dimensionless pool quantities.  They are explicitly
# not represented as biological time units.
POOL_SIZES: Mapping[str, float] = {
    "OAC": 1.0,
    "citrate": 1.0,
    "AKG": 1.0,
    "glutamate": 1.0,
    "succinate": 1.0,
    "fumarate": 1.0,
}


def _participant(metabolite_id: str, labels: str) -> AtomMappedParticipant:
    return AtomMappedParticipant(metabolite_id, tuple(labels))


@dataclass(frozen=True)
class Table5Reaction:
    """One exact Table 5 reaction and the corresponding FluxEMU participants."""

    reaction_id: str
    substrates: tuple[tuple[str, str], ...]
    products: tuple[tuple[str, str], ...]
    printed_transition: str


TABLE5_REACTIONS: tuple[Table5Reaction, ...] = (
    Table5Reaction("v1", (("OAC", "abcd"), ("AcCoA", "ef")), (("citrate", "dcbfea"),), "abcd + ef -> dcbfea"),
    Table5Reaction("v2", (("citrate", "abcdef"),), (("AKG", "abcde"), ("CO2", "f")), "abcdef -> abcde + f"),
    Table5Reaction("v3", (("AKG", "abcde"),), (("glutamate", "abcde"),), "abcde -> abcde"),
    Table5Reaction("v4", (("AKG", "abcde"),), (("succinate", "bcde"), ("CO2", "a")), "abcde -> bcde + a"),
    Table5Reaction("v5", (("succinate", "abcd"),), (("fumarate", "abcd"),), "0.5 abcd + 0.5 dcba -> 0.5 abcd + 0.5 dcba"),
    Table5Reaction("v6", (("fumarate", "abcd"),), (("OAC", "abcd"),), "0.5 abcd + 0.5 dcba -> abcd"),
    Table5Reaction("v7", (("OAC", "abcd"),), (("fumarate", "abcd"),), "abcd -> 0.5 abcd + 0.5 dcba"),
    Table5Reaction("v8", (("aspartate", "abcd"),), (("OAC", "abcd"),), "abcd -> abcd"),
)
TABLE5_BY_ID = {item.reaction_id: item for item in TABLE5_REACTIONS}


def _reaction_formula(spec: Table5Reaction) -> str:
    return " + ".join(metabolite for metabolite, _ in spec.substrates) + " --> " + " + ".join(
        metabolite for metabolite, _ in spec.products
    )


def build_cobra_model() -> Model:
    """Create the exact eight-reaction Table 5 SBML network."""

    model = Model("antoniewicz_tca_table5")
    metabolites = {
        metabolite_id: Metabolite(metabolite_id, compartment="c")
        for metabolite_id in METABOLITE_CARBONS
    }
    model.add_metabolites(metabolites.values())

    for metabolite_id, metabolite in metabolites.items():
        set_metabolite_metadata(
            metabolite,
            MetaboliteIsotopeMetadata(
                original_cobra_metabolite_id=metabolite_id,
                carbon_count=METABOLITE_CARBONS[metabolite_id],
                is_carbon_source=metabolite_id in CARBON_SOURCES,
                is_excreted=metabolite_id in PRODUCT_BOUNDARIES,
                symmetry=metabolite_id in SYMMETRIC_METABOLITES,
                include_in_isotope_model=True,
            ),
        )

    for order, spec in enumerate(TABLE5_REACTIONS):
        reaction = Reaction(
            spec.reaction_id,
            name=f"Antoniewicz Table 5 {spec.reaction_id}",
            lower_bound=CANONICAL_FLUX_BOUNDS[0],
            upper_bound=CANONICAL_FLUX_BOUNDS[1],
        )
        coefficients = {
            metabolites[metabolite_id]: -1.0 for metabolite_id, _ in spec.substrates
        }
        coefficients.update(
            {metabolites[metabolite_id]: 1.0 for metabolite_id, _ in spec.products}
        )
        reaction.add_metabolites(coefficients)
        set_reaction_metadata(
            reaction,
            ReactionIsotopeMetadata(
                original_cobra_reaction_id=spec.reaction_id,
                directional_id=f"{spec.reaction_id}:forward",
                direction="forward",
                include_in_isotope_model=True,
                substrates=tuple(_participant(*item) for item in spec.substrates),
                products=tuple(_participant(*item) for item in spec.products),
            ),
        )
        model.add_reactions([reaction])
        assert (reaction.lower_bound, reaction.upper_bound) == CANONICAL_FLUX_BOUNDS
        assert reaction.annotation.setdefault("table5_order", order + 1) == order + 1

    model.objective = model.reactions.v3
    model.objective_direction = "max"
    model.notes["FLUXEMU_BENCHMARK_SOURCE"] = PAPER_REFERENCE
    model.notes["FLUXEMU_TABLE5_REFERENCE"] = TABLE5_PAGE
    model.notes["FLUXEMU_FIGURE12_REFERENCE"] = FIGURE12_PAGE
    return model


def write_model(path: Path = MODEL_PATH) -> Path:
    """Write and round-trip validate the exact annotated Table 5 model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_sbml_model(build_cobra_model(), str(path))
    reloaded = read_sbml_model(str(path))
    validate_table5_model(reloaded)
    return path


def validate_table5_model(model: Model) -> None:
    """Validate the paper transcription and all map-level carbon bookkeeping."""

    if tuple(reaction.id for reaction in model.reactions) != tuple(FLUXES):
        raise ValueError("the model must contain exactly the eight Figure 12 reactions")
    metadata = collect_isotope_metadata(model)
    if set(metadata.metabolites) != set(METABOLITE_CARBONS):
        raise ValueError("the model metabolite set differs from Table 5")
    for spec in TABLE5_REACTIONS:
        reaction = model.reactions.get_by_id(spec.reaction_id)
        item = metadata.reactions[spec.reaction_id]
        observed = (
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in item.substrates),
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in item.products),
        )
        expected = (spec.substrates, spec.products)
        if observed != expected:
            raise ValueError(f"{spec.reaction_id} atom transition differs from Table 5")
        if (reaction.lower_bound, reaction.upper_bound) != CANONICAL_FLUX_BOUNDS:
            raise ValueError(
                f"{spec.reaction_id} does not use canonical irreversible bounds"
            )
        if set(item.substrate_atom_labels) != set(item.product_atom_labels):
            raise ValueError(f"{spec.reaction_id} does not conserve labelled carbon")
    for metabolite_id in SYMMETRIC_METABOLITES:
        if not metadata.metabolites[metabolite_id].symmetry:
            raise ValueError(f"{metabolite_id} must retain Table 5 symmetry")


def _side_text(participants: tuple[AtomMappedParticipant, ...]) -> str:
    return "+".join(participant.metabolite_id for participant in participants)


def _atom_text(participants: tuple[AtomMappedParticipant, ...]) -> str:
    return "+".join("".join(participant.atom_labels) for participant in participants)


def _mfapy_targets(experiment: ExperimentConfig) -> dict[str, dict[str, Any]]:
    # The time-course uses all intermediates.  Keeping this fixed target set in
    # the constructed model makes the same diffmdv function serve both runs.
    all_targets = (
        ("OAC", "OAC", 4),
        ("citrate", "citrate", 6),
        ("AKG", "AKG", 5),
        ("succinate", "succinate", 4),
        ("fumarate", "fumarate", 4),
        ("glutamate", "glutamate", 5),
    )
    configured = {target.fragment_id: target for target in experiment.targets}
    result: dict[str, dict[str, Any]] = {}
    for order, (fragment_id, metabolite_id, carbon_count) in enumerate(all_targets):
        target = configured.get(fragment_id)
        result[fragment_id] = {
            "type": "intermediate",
            "atommap": f"{metabolite_id}_" + ":".join(str(i) for i in range(1, carbon_count + 1)),
            "use": "use" if target is not None else "no",
            "order": order,
            "formula": target.formula if target is not None else "",
        }
    return result


def build_table5_mfapy_bundle(
    model: Model,
    experiment: ExperimentConfig,
    *,
    symmetric: bool = True,
) -> MfapyModelBundle:
    """Construct the existing FluxEMU conversion/forward interfaces in memory."""

    validate_table5_model(model)
    metadata = collect_isotope_metadata(model)
    reaction_dictionary: dict[str, dict[str, Any]] = {}
    mappings: list[ReactionMapping] = []
    for order, reaction in enumerate(model.reactions):
        item = metadata.reactions[reaction.id]
        reaction_dictionary[reaction.id] = {
            "stoichiometry": f"{_side_text(item.substrates)}-->{_side_text(item.products)}",
            "reaction": f"{_side_text(item.substrates)}-->{_side_text(item.products)}",
            "atommap": f"{_atom_text(item.substrates)}-->{_atom_text(item.products)}",
            "externalids": f"cobra:{reaction.id}",
            "order": order,
            "lb": float(reaction.lower_bound),
            "ub": float(reaction.upper_bound),
        }
        mappings.append(ReactionMapping(reaction.id, reaction.id, order))

    metabolite_dictionary: dict[str, dict[str, Any]] = {}
    for order, metabolite in enumerate(model.metabolites):
        item = metadata.metabolites[metabolite.id]
        metabolite_dictionary[metabolite.id] = {
            "C_number": item.carbon_count,
            "symmetry": "symmetry" if symmetric and item.symmetry else "no",
            "carbonsource": "carbonsource" if item.is_carbon_source else "no",
            "excreted": "excreted" if item.is_excreted else "no",
            "order": order,
            "externalids": f"cobra:{metabolite.id}",
            "lb": 1.0,
            "ub": 1_000_000.0,
        }

    mfapy_model = BoundaryTargetMetabolicModel(
        reaction_dictionary, {}, metabolite_dictionary, _mfapy_targets(experiment)
    )
    converter = FluxConverter(mappings, tuple(mfapy_model.reaction_ids))
    target_to_internal = {
        target.fragment_id: target.fragment_id for target in experiment.targets
    }
    return MfapyModelBundle(
        model=mfapy_model,
        converter=converter,
        reaction_mappings=tuple(mappings),
        metabolite_to_internal={metabolite.id: metabolite.id for metabolite in model.metabolites},
        target_to_internal=target_to_internal,
        reactions=reaction_dictionary,
        reversible_reactions={},
        metabolites=metabolite_dictionary,
        target_fragments=_mfapy_targets(experiment),
    )


def _carbon_source(bundle: MfapyModelBundle, experiment: ExperimentConfig):
    carbon_source = bundle.model.generate_carbon_source_template()
    for tracer in experiment.tracers:
        accepted = carbon_source.set_each_isotopomer(
            bundle.metabolite_to_internal[tracer.metabolite_id],
            dict(tracer.isotopomer_fractions),
            correction=tracer.correction,
        )
        if accepted is not True:
            raise ValueError(f"mfapy rejected tracer {tracer.metabolite_id}")
    return carbon_source


def ground_truth_flux_frame(model: Model) -> pd.DataFrame:
    """Return the explicit generating state, separate from canonical bounds."""

    return pd.DataFrame(
        [[GROUND_TRUTH_FLUXES[reaction.id] for reaction in model.reactions]],
        index=pd.Index(["figure12"], name="sample_id"),
        columns=[reaction.id for reaction in model.reactions],
    )


def _fixed_flux_frame(model: Model) -> pd.DataFrame:
    """Backward-compatible spelling for the separate ground-truth state."""

    return ground_truth_flux_frame(model)


def run_stationary() -> tuple[MfapyModelBundle, np.ndarray, pd.DataFrame]:
    """Run the published flux state through FluxEMU's batch mfapy forward path."""

    model = read_sbml_model(str(MODEL_PATH))
    experiment = load_experiment(STATIONARY_EXPERIMENT)
    bundle = build_table5_mfapy_bundle(model, experiment)
    forward = run_batch_forward(bundle, _fixed_flux_frame(model), experiment)
    mid = np.asarray(forward.predictions["figure12"]["glutamate"], dtype=float)
    return bundle, mid, forward.mids


def _unlabelled_y0(bundle: MfapyModelBundle) -> list[float]:
    return [1.0 if isotopologue_index == 0 else 0.0 for _, isotopologue_index in bundle.model.emu_order_in_y]


def run_timecourse() -> pd.DataFrame:
    """Use mfapy's existing diffmdv implementation for the Table 5 network."""

    model = read_sbml_model(str(MODEL_PATH))
    timecourse_data = yaml.safe_load(TIMECOURSE_EXPERIMENT.read_text(encoding="utf-8"))
    assert isinstance(timecourse_data, dict)
    timecourse_data.pop("timecourse")
    experiment = parse_experiment_config(timecourse_data)
    bundle = build_table5_mfapy_bundle(model, experiment)
    carbon_source = _carbon_source(bundle, experiment)
    fluxes = bundle.converter.convert_row(_fixed_flux_frame(model).iloc[0])
    pools = [
        POOL_SIZES[metabolite_id]
        for metabolite_id in bundle.model.dynamic_metabolite_ids
    ]
    _, predicted = bundle.model.func["diffmdv"](
        fluxes,
        pools,
        list(TIMEPOINTS),
        list(bundle.model.target_fragments),
        carbon_source.generate_dict(),
        _unlabelled_y0(bundle),
    )
    rows: list[dict[str, float | str]] = []
    for target, values_by_time in predicted.items():
        if target == "X_list":
            continue
        for timepoint, mid in zip(TIMEPOINTS, values_by_time):
            for mass, fraction in enumerate(mid):
                rows.append(
                    {
                        "time": timepoint,
                        "metabolite": target,
                        "mass_isotopologue": f"M+{mass}",
                        "fraction": float(fraction),
                    }
                )
    return pd.DataFrame(rows, columns=("time", "metabolite", "mass_isotopologue", "fraction"))


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_results() -> None:
    """Write reproducible stationary and dynamic benchmark artefacts."""

    _, mfapy_mid, _ = run_stationary()
    direct_mid = glutamate_mid()
    _write_csv(
        PUBLISHED_REFERENCE,
        ("metabolite", "mass_isotopologue", "fraction", "paper_reference"),
        [
            ("glutamate", f"M+{index}", float(value), TABLE6_PAGE)
            for index, value in enumerate(PUBLISHED_TABLE6_MID)
        ],
    )
    _write_csv(
        STATIONARY_MIDS,
        ("metabolite", "mass_isotopologue", "fluxemu_mfapy", "direct_isotopomer", "published_table6"),
        [
            ("glutamate", f"M+{index}", float(mfapy_mid[index]), float(direct_mid[index]), float(PUBLISHED_TABLE6_MID[index]))
            for index in range(6)
        ],
    )
    timecourse = run_timecourse()
    timecourse.to_csv(TIMECOURSE_MIDS, index=False, float_format="%.15g")


def main() -> None:
    write_model()
    write_results()


if __name__ == "__main__":
    main()

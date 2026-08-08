"""Synthetic U-13C6 glucose entry into the frozen Antoniewicz TCA benchmark.

This module is deliberately benchmark-local.  It imports the frozen Table 5
TCA transition definitions and its independent isotopomer solver, adds only
the documented synthetic glycolytic carbon skeleton, and calls mfapy directly
through FluxEMU's established in-memory forward interface.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from cobra import Metabolite, Model, Reaction
from cobra.io import read_sbml_model, write_sbml_model

from fluxemu.mfapy_model import BoundaryTargetMetabolicModel
from fluxemu.cobra_to_mfapy import MfapyModelBundle
from fluxemu.configuration import ExperimentConfig, load_experiment, parse_experiment_config
from fluxemu.configuration import TracerMixture
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


EXAMPLE_DIR = Path(__file__).resolve().parent
FROZEN_TCA_DIR = EXAMPLE_DIR.parent / "antoniewicz_tca"
MODEL_PATH = EXAMPLE_DIR / "antoniewicz_tca_glucose.xml"
STATIONARY_EXPERIMENT = EXAMPLE_DIR / "experiment_u13c6_glucose.yaml"
TIMECOURSE_EXPERIMENT = EXAMPLE_DIR / "experiment_u13c6_glucose_timecourse.yaml"
STATIONARY_MIDS = EXAMPLE_DIR / "stationary_mids.csv"
INDEPENDENT_TCA_MIDS = EXAMPLE_DIR / "independent_tca_mids.csv"
TIMECOURSE_MIDS = EXAMPLE_DIR / "timecourse_mids.csv"


def _load_frozen_tca_module():
    """Load the immutable benchmark under a private, collision-free name."""

    module_name = "_fluxemu_frozen_antoniewicz_tca_build_model"
    if module_name in sys.modules:
        return sys.modules[module_name]
    # The frozen builder imports its adjacent direct_isotopomer_solver module.
    sys.path.insert(0, str(FROZEN_TCA_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, FROZEN_TCA_DIR / "build_model.py"
        )
        if spec is None or spec.loader is None:  # pragma: no cover - filesystem invariant
            raise ImportError("could not load frozen Antoniewicz benchmark")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


_FROZEN_TCA = _load_frozen_tca_module()


def _load_independent_tca_solver():
    """Load the adjacent NumPy-only solver without relying on sys.path state."""

    name = "_fluxemu_glucose_independent_tca_solver"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / "independent_tca_solver.py")
    if spec is None or spec.loader is None:  # pragma: no cover - filesystem invariant
        raise ImportError("could not load independent pure-M+2 TCA solver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_INDEPENDENT_TCA_SOLVER = _load_independent_tca_solver()


@dataclass(frozen=True)
class GlycolysisReaction:
    """One carbon-skeleton reaction added upstream of the frozen TCA model."""

    reaction_id: str
    substrates: tuple[tuple[str, str], ...]
    products: tuple[tuple[str, str], ...]
    printed_transition: str


GLYCOLYSIS_REACTIONS: tuple[GlycolysisReaction, ...] = (
    GlycolysisReaction("GLC_IN", (("glucose_ext", "abcdef"),), (("glucose_c", "abcdef"),), "abcdef -> abcdef"),
    GlycolysisReaction("HEX", (("glucose_c", "abcdef"),), (("G6P", "abcdef"),), "abcdef -> abcdef"),
    GlycolysisReaction("PGI", (("G6P", "abcdef"),), (("F6P", "abcdef"),), "abcdef -> abcdef"),
    GlycolysisReaction("PFK", (("F6P", "abcdef"),), (("FBP", "abcdef"),), "abcdef -> abcdef"),
    GlycolysisReaction("FBA", (("FBP", "abcdef"),), (("DHAP", "cba"), ("GAP", "def")), "abcdef -> cba + def"),
    GlycolysisReaction("TPI", (("DHAP", "abc"),), (("GAP", "abc"),), "abc -> abc"),
    GlycolysisReaction("GAPD", (("GAP", "abc"),), (("BPG", "abc"),), "abc -> abc"),
    GlycolysisReaction("PGK", (("BPG", "abc"),), (("3PG", "abc"),), "abc -> abc"),
    GlycolysisReaction("PGM", (("3PG", "abc"),), (("2PG", "abc"),), "abc -> abc"),
    GlycolysisReaction("ENO", (("2PG", "abc"),), (("PEP", "abc"),), "abc -> abc"),
    GlycolysisReaction("PYK", (("PEP", "abc"),), (("pyruvate", "abc"),), "abc -> abc"),
    GlycolysisReaction("LDH", (("pyruvate", "abc"),), (("lactate", "abc"),), "abc -> abc"),
    GlycolysisReaction("PDH", (("pyruvate", "abc"),), (("AcCoA", "bc"), ("CO2", "a")), "abc -> bc + a"),
)

# These objects, including their exact map strings, are imported from the
# frozen benchmark.  They are not independently transcribed in this example.
TCA_REACTIONS = tuple(_FROZEN_TCA.TABLE5_REACTIONS)
ALL_REACTIONS = GLYCOLYSIS_REACTIONS + TCA_REACTIONS
REACTION_BY_ID = {reaction.reaction_id: reaction for reaction in ALL_REACTIONS}

UPSTREAM_FLUXES: Mapping[str, float] = {
    "GLC_IN": 50.0,
    "HEX": 50.0,
    "PGI": 50.0,
    "PFK": 50.0,
    "FBA": 50.0,
    "TPI": 50.0,
    "GAPD": 100.0,
    "PGK": 100.0,
    "PGM": 100.0,
    "ENO": 100.0,
    "PYK": 100.0,
    "LDH": 0.0,
    "PDH": 100.0,
}
GROUND_TRUTH_FLUXES: Mapping[str, float] = MappingProxyType(
    {**UPSTREAM_FLUXES, **dict(_FROZEN_TCA.GROUND_TRUTH_FLUXES)}
)
# Compatibility name retained for existing forward regressions. This mapping
# is a generating state and is not encoded as canonical SBML bounds.
FIXED_FLUXES = GROUND_TRUTH_FLUXES
CANONICAL_FLUX_BOUNDS = _FROZEN_TCA.CANONICAL_FLUX_BOUNDS
CANONICAL_REACTION_ORDER = tuple(item.reaction_id for item in ALL_REACTIONS)

METABOLITE_CARBONS: Mapping[str, int] = {
    "glucose_ext": 6,
    "glucose_c": 6,
    "G6P": 6,
    "F6P": 6,
    "FBP": 6,
    "DHAP": 3,
    "GAP": 3,
    "BPG": 3,
    "3PG": 3,
    "2PG": 3,
    "PEP": 3,
    "pyruvate": 3,
    "lactate": 3,
    "AcCoA": 2,
    "OAC": 4,
    "citrate": 6,
    "AKG": 5,
    "glutamate": 5,
    "succinate": 4,
    "fumarate": 4,
    "aspartate": 4,
    "CO2": 1,
}
CARBON_SOURCES = frozenset(("glucose_ext", "aspartate"))
EXCRETED_METABOLITES = frozenset(("CO2", "lactate", "glutamate"))
# The frozen Table 5 model deliberately has glutamate as its observed terminal
# product.  It therefore has an implicit measurement outflow, not a ninth
# reaction; all other non-source/non-excreted metabolites balance explicitly.
TERMINAL_OBSERVED_PRODUCTS = frozenset(("glutamate",))
SYMMETRIC_METABOLITES = frozenset(("succinate", "fumarate"))
BALANCED_CARBON_METABOLITES = tuple(
    metabolite_id
    for metabolite_id in METABOLITE_CARBONS
    if metabolite_id not in CARBON_SOURCES
    and metabolite_id not in EXCRETED_METABOLITES
)

STATIONARY_TARGETS = (
    "glucose_c",
    "G6P",
    "F6P",
    "FBP",
    "DHAP",
    "GAP",
    "BPG",
    "3PG",
    "2PG",
    "PEP",
    "pyruvate",
    "AcCoA",
    "citrate",
    "AKG",
    "succinate",
    "fumarate",
    "OAC",
    "glutamate",
)
INDEPENDENT_TCA_TARGETS = ("OAC", "citrate", "AKG", "glutamate", "succinate", "fumarate")
BASE_TIMECOURSE_TIMEPOINTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
BASE_TIMEPOINTS = BASE_TIMECOURSE_TIMEPOINTS
DIAGNOSTIC_TIMECOURSE_TIMEPOINTS = (
    0.0,
    1.0,
    2.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    60.0,
    80.0,
    120.0,
    160.0,
)
DEFAULT_TIMECOURSE_TIMEPOINTS = DIAGNOSTIC_TIMECOURSE_TIMEPOINTS
POOL_SIZE = 100.0
DYNAMIC_CONVERGENCE_TOLERANCE = 1.0e-5
DYNAMIC_ROUNDING_TOLERANCE = 1.0e-6
DIAGNOSTIC_GLUCOSE_ENRICHMENT_SETTINGS: tuple[tuple[str, float], ...] = (
    ("100", 1.0),
    ("80", 0.8),
    ("70", 0.7),
    ("50", 0.5),
)

# mfapy's generated functions use metabolite IDs as Python variables and EMU
# names split at underscores.  These internal IDs preserve the COBRA/SBML IDs
# at the boundary while avoiding that implementation constraint.
MFAPY_METABOLITE_IDS: Mapping[str, str] = {
    "glucose_ext": "glucoseext",
    "glucose_c": "glucosec",
    "G6P": "G6P",
    "F6P": "F6P",
    "FBP": "FBP",
    "DHAP": "DHAP",
    "GAP": "GAP",
    "BPG": "BPG",
    "3PG": "m3PG",
    "2PG": "m2PG",
    "PEP": "PEP",
    "pyruvate": "pyruvate",
    "lactate": "lactate",
    "AcCoA": "AcCoA",
    "OAC": "OAC",
    "citrate": "citrate",
    "AKG": "AKG",
    "glutamate": "glutamate",
    "succinate": "succinate",
    "fumarate": "fumarate",
    "aspartate": "aspartate",
    "CO2": "CO2",
}


def _participant(metabolite_id: str, labels: str) -> AtomMappedParticipant:
    return AtomMappedParticipant(metabolite_id, tuple(labels))


def _side_text(
    participants: Sequence[AtomMappedParticipant], *, internal: bool = False
) -> str:
    return "+".join(
        MFAPY_METABOLITE_IDS[item.metabolite_id] if internal else item.metabolite_id
        for item in participants
    )


def _atom_text(participants: Sequence[AtomMappedParticipant]) -> str:
    return "+".join("".join(participant.atom_labels) for participant in participants)


def build_cobra_model() -> Model:
    """Build the fixed-flux synthetic glucose-to-Table-5 carbon model."""

    model = Model("antoniewicz_tca_glucose")
    metabolites = {
        metabolite_id: Metabolite(
            metabolite_id,
            compartment="e" if metabolite_id == "glucose_ext" else "c",
        )
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
                is_excreted=metabolite_id in EXCRETED_METABOLITES,
                symmetry=metabolite_id in SYMMETRIC_METABOLITES,
                include_in_isotope_model=True,
            ),
        )

    for order, spec in enumerate(ALL_REACTIONS):
        reaction = Reaction(
            spec.reaction_id,
            name=(
                f"Synthetic glucose carbon skeleton {spec.reaction_id}"
                if spec.reaction_id in UPSTREAM_FLUXES
                else f"Frozen Antoniewicz Table 5 {spec.reaction_id}"
            ),
            lower_bound=CANONICAL_FLUX_BOUNDS[0],
            upper_bound=CANONICAL_FLUX_BOUNDS[1],
        )
        coefficients = {metabolites[metabolite_id]: -1.0 for metabolite_id, _ in spec.substrates}
        coefficients.update({metabolites[metabolite_id]: 1.0 for metabolite_id, _ in spec.products})
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
        reaction.annotation["benchmark_order"] = order + 1
        model.add_reactions([reaction])

    model.objective = model.reactions.v1
    model.objective_direction = "max"
    model.notes["FLUXEMU_BENCHMARK_PURPOSE"] = (
        "Synthetic U-13C6 glucose-to-Antoniewicz-TCA numerical isotope-propagation benchmark"
    )
    model.notes["FLUXEMU_TCA_MAPPING_SOURCE"] = (
        "Exact frozen Antoniewicz Table 5 definitions imported from antoniewicz_tca"
    )
    return model


def fixed_flux_frame(model: Model | None = None) -> pd.DataFrame:
    """Return the one complete ground-truth flux vector in reaction order."""

    reaction_ids = [reaction.reaction_id for reaction in ALL_REACTIONS] if model is None else [reaction.id for reaction in model.reactions]
    return pd.DataFrame(
        [[FIXED_FLUXES[reaction_id] for reaction_id in reaction_ids]],
        index=pd.Index(["glucose_to_tca_fixed"], name="sample_id"),
        columns=reaction_ids,
    )


def carbon_mass_balance(model: Model | None = None) -> pd.Series:
    """Calculate net molecular flux for every explicitly balanced carbon pool."""

    if model is None:
        model = build_cobra_model()
    fluxes = fixed_flux_frame(model).iloc[0]
    residuals: dict[str, float] = {}
    for metabolite_id in BALANCED_CARBON_METABOLITES:
        metabolite = model.metabolites.get_by_id(metabolite_id)
        residuals[metabolite_id] = float(
            sum(
                float(reaction.metabolites[metabolite]) * float(fluxes[reaction.id])
                for reaction in metabolite.reactions
            )
        )
    return pd.Series(residuals, name="net_molecular_flux")


def validate_glucose_tca_model(model: Model) -> None:
    """Validate imported maps, fixed bounds, and the complete internal balance."""

    if tuple(reaction.id for reaction in model.reactions) != tuple(item.reaction_id for item in ALL_REACTIONS):
        raise ValueError("the model reaction order differs from the defined fixed benchmark")
    metadata = collect_isotope_metadata(model)
    if set(metadata.metabolites) != set(METABOLITE_CARBONS):
        raise ValueError("the model metabolite set is incomplete")
    for spec in ALL_REACTIONS:
        item = metadata.reactions[spec.reaction_id]
        observed = (
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in item.substrates),
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in item.products),
        )
        if observed != (spec.substrates, spec.products):
            raise ValueError(f"{spec.reaction_id} atom transition differs from its benchmark definition")
        if set(item.substrate_atom_labels) != set(item.product_atom_labels):
            raise ValueError(f"{spec.reaction_id} does not conserve its carbon labels")
        reaction = model.reactions.get_by_id(spec.reaction_id)
        if (reaction.lower_bound, reaction.upper_bound) != CANONICAL_FLUX_BOUNDS:
            raise ValueError(
                f"{spec.reaction_id} does not use canonical irreversible bounds"
            )
    if tuple(TCA_REACTIONS) != tuple(_FROZEN_TCA.TABLE5_REACTIONS):
        raise ValueError("the TCA definitions are not the frozen Table 5 definitions")
    residuals = carbon_mass_balance(model)
    if not np.allclose(residuals.to_numpy(), 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"fixed flux vector is not internally balanced: {residuals.to_dict()}")


def write_model(path: Path = MODEL_PATH) -> Path:
    """Write and SBML-round-trip validate the complete benchmark model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_sbml_model(build_cobra_model(), str(path))
    validate_glucose_tca_model(read_sbml_model(str(path)))
    return path


def _mfapy_targets(experiment: ExperimentConfig) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for order, target in enumerate(experiment.targets):
        metabolite_id = MFAPY_METABOLITE_IDS[target.metabolite_id]
        positions = ":".join(str(position) for position in target.atom_positions)
        result[target.fragment_id] = {
            "type": target.analytical_method,
            "atommap": f"{metabolite_id}_{positions}",
            "use": "use",
            "order": order,
            "formula": target.formula,
        }
    return result


def build_glucose_tca_mfapy_bundle(
    model: Model,
    experiment: ExperimentConfig,
    *,
    symmetric: bool = True,
) -> MfapyModelBundle:
    """Build the new example through FluxEMU's existing in-memory contract."""

    validate_glucose_tca_model(model)
    metadata = collect_isotope_metadata(model)
    reactions: dict[str, dict[str, Any]] = {}
    mappings: list[ReactionMapping] = []
    for order, reaction in enumerate(model.reactions):
        item = metadata.reactions[reaction.id]
        internal_reaction_id = f"r{order}"
        reactions[internal_reaction_id] = {
            "stoichiometry": f"{_side_text(item.substrates, internal=True)}-->{_side_text(item.products, internal=True)}",
            "reaction": f"{_side_text(item.substrates, internal=True)}-->{_side_text(item.products, internal=True)}",
            "atommap": f"{_atom_text(item.substrates)}-->{_atom_text(item.products)}",
            "externalids": f"cobra:{reaction.id}",
            "order": order,
            "lb": float(reaction.lower_bound),
            "ub": float(reaction.upper_bound),
        }
        mappings.append(ReactionMapping(reaction.id, internal_reaction_id, order))

    metabolites: dict[str, dict[str, Any]] = {}
    for order, metabolite in enumerate(model.metabolites):
        item = metadata.metabolites[metabolite.id]
        metabolites[MFAPY_METABOLITE_IDS[metabolite.id]] = {
            "C_number": item.carbon_count,
            "symmetry": "symmetry" if symmetric and item.symmetry else "no",
            "carbonsource": "carbonsource" if item.is_carbon_source else "no",
            "excreted": "excreted" if item.is_excreted else "no",
            "order": order,
            "externalids": f"cobra:{metabolite.id}",
            "lb": 1.0,
            "ub": 1_000_000.0,
        }

    target_fragments = _mfapy_targets(experiment)
    mfapy_model = BoundaryTargetMetabolicModel(
        reactions, {}, metabolites, target_fragments
    )
    expected_order = tuple(f"r{order}" for order in range(len(model.reactions)))
    if tuple(mfapy_model.reaction_ids) != expected_order:
        raise ValueError("mfapy reordered the fixed benchmark reaction vector")
    return MfapyModelBundle(
        model=mfapy_model,
        converter=FluxConverter(mappings, tuple(mfapy_model.reaction_ids)),
        reaction_mappings=tuple(mappings),
        metabolite_to_internal=dict(MFAPY_METABOLITE_IDS),
        target_to_internal={target.fragment_id: target.fragment_id for target in experiment.targets},
        reactions=reactions,
        reversible_reactions={},
        metabolites=metabolites,
        target_fragments=target_fragments,
    )


def run_stationary(
    *,
    symmetric: bool = True,
    experiment: ExperimentConfig | None = None,
) -> tuple[MfapyModelBundle, dict[str, np.ndarray], pd.DataFrame]:
    """Run the complete fixed glucose-to-TCA model through FluxEMU/mfapy."""

    model = read_sbml_model(str(MODEL_PATH))
    if experiment is None:
        experiment = load_experiment(STATIONARY_EXPERIMENT)
    bundle = build_glucose_tca_mfapy_bundle(model, experiment, symmetric=symmetric)
    result = run_batch_forward(bundle, fixed_flux_frame(model), experiment)
    predictions = {
        target: np.asarray(result.predictions["glucose_to_tca_fixed"][target], dtype=float)
        for target in STATIONARY_TARGETS
    }
    return bundle, predictions, result.mids


def solve_independent_tca(*, symmetric: bool = True) -> dict[str, np.ndarray]:
    """Run the NumPy-only frozen TCA state solver with pure #11 acetyl-CoA."""

    return _INDEPENDENT_TCA_SOLVER.solve_stationary(symmetric=symmetric)


def independent_tca_mids(*, symmetric: bool = True) -> dict[str, np.ndarray]:
    """Collapse the independent full isotopomer state into complete TCA MIDs."""

    return _INDEPENDENT_TCA_SOLVER.mass_isotopomer_distributions(symmetric=symmetric)


def _unlabelled_y0(bundle: MfapyModelBundle) -> list[float]:
    return [1.0 if isotopologue_index == 0 else 0.0 for _, isotopologue_index in bundle.model.emu_order_in_y]


def _load_timecourse_experiment() -> ExperimentConfig:
    """Parse the standard fields while retaining benchmark-local YAML metadata."""

    raw = yaml.safe_load(TIMECOURSE_EXPERIMENT.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):  # pragma: no cover - YAML fixture invariant
        raise ValueError("time-course experiment must be a mapping")
    raw.pop("timecourse")
    return parse_experiment_config(raw)


def timecourse_experiment_with_glucose_enrichment(
    experiment: ExperimentConfig,
    labelled_fraction: float,
) -> ExperimentConfig:
    """Return a copy of ``experiment`` with glucose_ext enrichment adjusted."""

    if not 0.0 <= labelled_fraction <= 1.0:
        raise ValueError("labelled fraction must be between zero and one")
    unlabeled_fraction = 1.0 - labelled_fraction

    updated_tracers: list[TracerMixture] = []
    replaced = False
    for tracer in experiment.tracers:
        if tracer.metabolite_id == "glucose_ext":
            updated_tracers.append(
                TracerMixture(
                    metabolite_id=tracer.metabolite_id,
                    isotopomer_fractions={
                        "#111111": labelled_fraction,
                        "#000000": unlabeled_fraction,
                    },
                    correction=tracer.correction,
                )
            )
            replaced = True
        else:
            updated_tracers.append(tracer)

    if not replaced:
        raise ValueError("experiment does not define a glucose_ext tracer")

    return ExperimentConfig(
        tracers=tuple(updated_tracers),
        targets=experiment.targets,
        fraction_of_optimum=experiment.fraction_of_optimum,
        sample_count=experiment.sample_count,
        sampler=experiment.sampler,
        seed=experiment.seed,
        tolerances=experiment.tolerances,
        output=experiment.output,
        schema_version=experiment.schema_version,
    )


def _timecourse_frame(
    bundle: MfapyModelBundle,
    timepoints: Sequence[float],
    experiment: ExperimentConfig | None = None,
) -> pd.DataFrame:
    if experiment is None:
        experiment = _load_timecourse_experiment()
    carbon_source = bundle.model.generate_carbon_source_template()
    for tracer in experiment.tracers:
        accepted = carbon_source.set_each_isotopomer(
            bundle.metabolite_to_internal[tracer.metabolite_id],
            dict(tracer.isotopomer_fractions),
            correction=tracer.correction,
        )
        if accepted is not True:
            raise ValueError(f"mfapy rejected tracer {tracer.metabolite_id}")
    model = read_sbml_model(str(MODEL_PATH))
    fluxes = bundle.converter.convert_row(fixed_flux_frame(model).iloc[0])
    pools = [POOL_SIZE for _ in bundle.model.dynamic_metabolite_ids]
    _, predicted = bundle.model.func["diffmdv"](
        fluxes,
        pools,
        list(timepoints),
        [target.fragment_id for target in experiment.targets],
        carbon_source.generate_dict(),
        _unlabelled_y0(bundle),
    )
    rows: list[dict[str, float | str]] = []
    for target in STATIONARY_TARGETS:
        for timepoint, mid in zip(timepoints, predicted[target]):
            values = np.asarray(mid, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"diffmdv returned a non-finite MID for {target} at {timepoint}")
            if float(values.min()) < -DYNAMIC_ROUNDING_TOLERANCE:
                raise ValueError(f"diffmdv returned a materially negative MID for {target} at {timepoint}")
            # mfapy's generated diffmdv route fixes odeint tolerances internally
            # at 1e-3.  Remove only sub-microfraction integration roundoff before
            # writing a probability distribution, then preserve normalisation.
            values = np.maximum(values, 0.0)
            values /= values.sum()
            for mass, fraction in enumerate(values):
                rows.append(
                    {
                        "time": float(timepoint),
                        "metabolite": target,
                        "mass_isotopologue": f"M+{mass}",
                        "fraction": float(fraction),
                    }
                )
    return pd.DataFrame(rows, columns=("time", "metabolite", "mass_isotopologue", "fraction"))


def _mid_at(frame: pd.DataFrame, timepoint: float, metabolite: str) -> np.ndarray:
    subset = frame[(frame["time"] == timepoint) & (frame["metabolite"] == metabolite)]
    return subset.sort_values("mass_isotopologue")["fraction"].to_numpy(dtype=float)


def _validate_timepoints(timepoints: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in timepoints)
    if not values:
        raise ValueError("timepoints must contain at least one point")
    if any(value < 0.0 or not np.isfinite(value) for value in values):
        raise ValueError("timepoints must be finite and non-negative")
    if any(values[position] >= values[position + 1] for position in range(len(values) - 1)):
        raise ValueError("timepoints must be strictly increasing")
    return values


def run_timecourse(
    *,
    experiment: ExperimentConfig | None = None,
    timepoints: Sequence[float] | None = None,
    max_time: float = 640.0,
) -> tuple[pd.DataFrame, tuple[float, ...], float]:
    """Run mfapy diffmdv and extend its numerical horizon until convergence."""

    if experiment is None:
        experiment = _load_timecourse_experiment()
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")

    if timepoints is None:
        timepoints = DEFAULT_TIMECOURSE_TIMEPOINTS
    else:
        timepoints = _validate_timepoints(timepoints)

    model = read_sbml_model(str(MODEL_PATH))
    bundle = build_glucose_tca_mfapy_bundle(model, experiment)
    stationary = run_stationary(experiment=experiment)[1]
    timepoints = list(_validate_timepoints(timepoints))
    while True:
        frame = _timecourse_frame(bundle, timepoints, experiment)
        final_time = timepoints[-1]
        convergence_error = max(
            float(np.max(np.abs(_mid_at(frame, final_time, target) - stationary[target])))
            for target in STATIONARY_TARGETS
        )
        if convergence_error <= DYNAMIC_CONVERGENCE_TOLERANCE:
            return frame, tuple(timepoints), convergence_error
        if final_time >= max_time:
            raise RuntimeError(
                f"mfapy diffmdv did not converge to the stationary result by time {max_time}"
            )
        timepoints.append(final_time * 2.0)


def run_enrichment_timecourse_sweep(
    *,
    timepoints: Sequence[float] = DIAGNOSTIC_TIMECOURSE_TIMEPOINTS,
    settings: Sequence[tuple[str, float]] = DIAGNOSTIC_GLUCOSE_ENRICHMENT_SETTINGS,
    runner: Callable[
        [ExperimentConfig, Sequence[float], float],
        tuple[pd.DataFrame, tuple[float, ...], float],
    ] | None = None,
    max_time: float = 640.0,
) -> dict[str, tuple[pd.DataFrame, tuple[float, ...], float]]:
    """Run default time courses under a set of glucose tracer enrichment levels."""

    if runner is None:
        def runner(
            experiment: ExperimentConfig,
            points: Sequence[float],
            run_max_time: float,
        ) -> tuple[pd.DataFrame, tuple[float, ...], float]:
            return run_timecourse(
                experiment=experiment,
                timepoints=points,
                max_time=run_max_time,
            )
    base_experiment = _load_timecourse_experiment()
    resolved_timepoints = _validate_timepoints(timepoints)
    results: dict[str, tuple[pd.DataFrame, tuple[float, ...], float]] = {}
    for label, labelled_fraction in settings:
        altered = timecourse_experiment_with_glucose_enrichment(
            base_experiment,
            labelled_fraction,
        )
        results[label] = runner(altered, resolved_timepoints, max_time)
    return results


def _apply_single_substrate_map(
    spec: Any,
    substrate_id: str,
    substrate_origins: tuple[int, ...],
    product_id: str,
) -> tuple[int, ...]:
    """Follow a positional carbon state through one actual atom map."""

    substrate = next(item for item in spec.substrates if item[0] == substrate_id)
    product = next(item for item in spec.products if item[0] == product_id)
    if len(spec.substrates) != 1:
        raise ValueError("positional glycolysis check expects a single substrate")
    labels_to_origins = dict(zip(substrate[1], substrate_origins, strict=True))
    return tuple(labels_to_origins[label] for label in product[1])


def upstream_pyruvate_origin_patterns(
    reactions: Iterable[Any] = GLYCOLYSIS_REACTIONS,
) -> tuple[tuple[int, int, int], ...]:
    """Derive glucose-carbon origins at pyruvate positions from map metadata."""

    by_id = {reaction.reaction_id: reaction for reaction in reactions}
    state: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    current = "glucose_ext"
    for reaction_id, product in (("GLC_IN", "glucose_c"), ("HEX", "G6P"), ("PGI", "F6P"), ("PFK", "FBP")):
        state = _apply_single_substrate_map(by_id[reaction_id], current, state, product)
        current = product
    fba = by_id["FBA"]
    dhap = _apply_single_substrate_map(fba, "FBP", state, "DHAP")
    direct_gap = _apply_single_substrate_map(fba, "FBP", state, "GAP")
    gap_via_tpi = _apply_single_substrate_map(by_id["TPI"], "DHAP", dhap, "GAP")
    patterns = []
    for gap in (gap_via_tpi, direct_gap):
        current_state = gap
        current_id = "GAP"
        for reaction_id, product in (("GAPD", "BPG"), ("PGK", "3PG"), ("PGM", "2PG"), ("ENO", "PEP"), ("PYK", "pyruvate")):
            current_state = _apply_single_substrate_map(by_id[reaction_id], current_id, current_state, product)
            current_id = product
        patterns.append(current_state)
    return tuple(patterns)


def pdh_positional_destinations(
    reaction: Any | None = None,
) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Return (AcCoA origins, CO2 origins) directly from the PDH atom map."""

    spec = REACTION_BY_ID["PDH"] if reaction is None else reaction
    pyruvate = (1, 2, 3)
    return (
        _apply_single_substrate_map(spec, "pyruvate", pyruvate, "AcCoA"),
        _apply_single_substrate_map(spec, "pyruvate", pyruvate, "CO2"),
    )


def recirculation_summary() -> dict[str, float]:
    """Use the independent positional state to quantify returned OAC label."""

    solution = solve_independent_tca()
    citrate = solution["citrate"]
    states = _INDEPENDENT_TCA_SOLVER.isotopomers(6)
    # In dcbfea, citrate positions 4 and 5 (zero-index 3,4) are pure-M+2
    # acetyl-CoA; positions 1,2,3,6 come from OAC.  A labelled OAC position is
    # therefore direct positional evidence of cycle return, not a MID-only turn
    # assignment.
    returned_oac = np.asarray([sum(state[index] for index in (0, 1, 2, 5)) for state in states])
    glutamate = solution["glutamate"]
    glutamate_states = _INDEPENDENT_TCA_SOLVER.isotopomers(5)
    return {
        "citrate_m2_with_unlabelled_oac": float(citrate[returned_oac == 0].sum()),
        "citrate_m3_or_higher_with_returned_oac": float(citrate[returned_oac >= 1].sum()),
        "citrate_m4_or_higher_with_two_or_more_returned_oac_carbons": float(citrate[returned_oac >= 2].sum()),
        "glutamate_m3_or_higher": float(
            glutamate[np.asarray([sum(state) >= 3 for state in glutamate_states])].sum()
        ),
    }


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_results() -> None:
    """Write all reproducible stationary, independent, and dynamic artefacts."""

    _, stationary, _ = run_stationary()
    independent = independent_tca_mids()
    _write_csv(
        INDEPENDENT_TCA_MIDS,
        ("metabolite", "mass_isotopologue", "independent_isotopomer"),
        (
            (metabolite, f"M+{mass}", float(fraction))
            for metabolite in INDEPENDENT_TCA_TARGETS
            for mass, fraction in enumerate(independent[metabolite])
        ),
    )
    _write_csv(
        STATIONARY_MIDS,
        ("metabolite", "mass_isotopologue", "fluxemu_mfapy", "independent_isotopomer"),
        (
            (
                metabolite,
                f"M+{mass}",
                float(fraction),
                float(independent[metabolite][mass]) if metabolite in independent else "",
            )
            for metabolite in STATIONARY_TARGETS
            for mass, fraction in enumerate(stationary[metabolite])
        ),
    )
    timecourse, _, _ = run_timecourse()
    timecourse.to_csv(TIMECOURSE_MIDS, index=False, float_format="%.15g")


def main() -> None:
    write_model()
    write_results()


if __name__ == "__main__":
    main()

"""Native reconstruction of the frozen R1 E. coli Stage B2 flux problem.

The bundled JSON is a one-time, lossless projection of the handoff SBML.  Loading
and solving it does not import COBRApy or mfapy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.resources
import json
from typing import Literal

from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    DirectionActivity,
    DirectionActivityCertificate,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    FluxProjectionExpression,
    FluxProjectionRule,
    FluxProjectionTerm,
    IsotopeMetabolite,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    LinearObjective,
    ObjectiveTerm,
    MappingBranch,
    PhysicalDirectionRef,
    ObservationPrecursor,
    ObservationTarget,
    StationaryExperimentSemantics,
    Tracer,
    StoichiometricTerm,
    validate_flux_model,
)

REACTION_ORDER_SHA256 = "7c375cbe07c3f7b496de65500cb0c7852caff7b709c6bcae6622e3a1e214c364"


@dataclass(frozen=True, slots=True)
class TargetCoverage:
    display_name: str
    requested_carbon_count: int
    canonical_root_pool_id: str
    target_type: Literal["physical_pool", "observation_overlay"]
    mapping_source: str
    source_complete: bool
    notes: str


# Requested display order is scientific data and must not be sorted.
TARGET_COVERAGE = (
    TargetCoverage("Pyruvate", 3, "pyr_c", "physical_pool", "Stage B2 metabolite_crosswalk.csv: pyr_c/Pyr exact", True, "Explicit physical isotope pool."),
    TargetCoverage("Alanine", 3, "pyr_c", "observation_overlay", "Stage B2 reaction_crosswalk.csv: alanine_reporter; Example_2:r58", True, "Observation-only Pyr ABC -> Ala ABC reporter."),
    TargetCoverage("Lactate", 3, "pyr_c", "observation_overlay", "R1 12-target observation extension; Example_7:r41_ldh", True, "Observation-only Pyr ABC -> Lac ABC; no LDH_D flux required."),
    TargetCoverage("Citrate", 6, "accoa_c + oaa_c", "observation_overlay", "R1 12-target observation extension; Example_7:r17_cs", True, "Target-only AcCOA AB + Oxa CDEF -> Cit FEDBAC condensation; Stage B2 IsoCit lump unchanged."),
    TargetCoverage("AKG", 5, "akg_c", "physical_pool", "Stage B2 metabolite_crosswalk.csv: akg_c/aKG exact", True, "Explicit physical isotope pool including the approved biomass Glu-to-aKG component."),
    TargetCoverage("Succinate", 4, "succ_c", "physical_pool", "Stage B2 target_ancestry.csv: Suc_1:2:3:4", True, "Explicit symmetric physical pool; both approved transports remain distinct."),
    TargetCoverage("Fumarate", 4, "fum_c", "physical_pool", "Stage B2 metabolite_crosswalk.csv: fum_c/Fum exact_symmetric", True, "Explicit symmetric physical isotope pool."),
    TargetCoverage("Malate", 4, "mal__L_c", "physical_pool", "Stage B2 target_ancestry.csv: Mal_1:2:3:4", True, "Explicit physical isotope pool."),
    TargetCoverage("Glutamine", 5, "gln__L_c", "physical_pool", "Stage B2 metabolite_crosswalk.csv: gln__L_c/Gln exact; Example_3", True, "Explicit physical isotope pool; not substituted with glutamate."),
    TargetCoverage("Aspartate", 4, "oaa_c", "observation_overlay", "Stage B2 reaction_crosswalk.csv: aspartate_reporter; Example_2:r59", True, "Observation-only OAA ABCD -> Asp ABCD reporter."),
    TargetCoverage("Glycine", 2, "3pg_c positions 1,2", "observation_overlay", "R1 12-target observation extension; Example_2:r67+r68", True, "Composed PGA ABC -> Ser ABC -> Gly AB + MEETHF C; no physical flux or folate dynamics."),
    TargetCoverage("Serine", 3, "3pg_c", "observation_overlay", "R1 12-target observation extension; Example_2:r67", True, "Observation-only PGA ABC -> Ser ABC."),
)


def load_ecoli_core_flux_model(objective: Literal["biomass", "acetate"] = "biomass") -> FluxModel:
    """Load the persisted common feasible set with only the objective changed."""

    if objective not in {"biomass", "acetate"}:
        raise ValueError("objective must be 'biomass' or 'acetate'")
    resource = importlib.resources.files(__package__) / "data/e_coli_core_stage_b2_flux.json"
    data = json.loads(resource.read_text(encoding="utf-8"))
    reactions = tuple(
        FluxReaction(
            item["id"],
            tuple(StoichiometricTerm(mid, coefficient) for mid, coefficient in item["stoichiometry"]),
            item["lower_bound"],
            item["upper_bound"],
        )
        for item in data["reactions"]
    )
    order = tuple(item.reaction_id for item in reactions)
    fingerprint = hashlib.sha256(("\n".join(order) + "\n").encode()).hexdigest()
    if fingerprint != REACTION_ORDER_SHA256 or fingerprint != data["reaction_order_sha256"]:
        raise ValueError("frozen E. coli reaction order fingerprint changed")
    model = FluxModel(
        tuple(FluxMetabolite(item["id"], item["balanced"]) for item in data["metabolites"]),
        reactions,
        LinearObjective("maximise", (ObjectiveTerm(data["objectives"][objective], 1.0),)),
    )
    validate_flux_model(model)
    return model


def build_r1_acceptance_experiment() -> StationaryExperimentSemantics:
    """Return the ordered 12-target experiment, including no-flux observations."""

    specifications = (
        ("Pyruvate", (("pyr_c", (1, 2, 3)),), "physical pool identity"),
        ("Alanine", (("pyr_c", (1, 2, 3)),), "Example_2:r58 Pyr ABC -> Ala ABC"),
        ("Lactate", (("pyr_c", (1, 2, 3)),), "Example_7:r41_ldh Pyr ABC -> Lac ABC"),
        ("Citrate", (("accoa_c", (1, 2)), ("oaa_c", (1, 2, 3, 4))), "Example_7:r17_cs AcCOA AB + Oxa CDEF -> Cit FEDBAC"),
        ("AKG", (("akg_c", (1, 2, 3, 4, 5)),), "physical pool identity"),
        ("Succinate", (("succ_c", (1, 2, 3, 4)),), "physical pool identity"),
        ("Fumarate", (("fum_c", (1, 2, 3, 4)),), "physical pool identity"),
        ("Malate", (("mal__L_c", (1, 2, 3, 4)),), "physical pool identity"),
        ("Glutamine", (("gln__L_c", (1, 2, 3, 4, 5)),), "physical pool identity"),
        ("Aspartate", (("oaa_c", (1, 2, 3, 4)),), "Example_2:r59 Oxa ABCD -> Asp ABCD"),
        ("Glycine", (("3pg_c", (1, 2)),), "Example_2:r67+r68 PGA ABC -> Ser ABC -> Gly AB + MEETHF C"),
        ("Serine", (("3pg_c", (1, 2, 3)),), "Example_2:r67 PGA ABC -> Ser ABC"),
    )
    observations = tuple(
        ObservationTarget(
            name, sum(len(positions) for _, positions in precursors),
            tuple(ObservationPrecursor(mid, positions) for mid, positions in precursors),
            (("layer", "R1 12-target observation extension"),
             ("atom_map", atom_map), ("no_physical_flux", "true")),
        )
        for name, precursors, atom_map in specifications
    )
    return StationaryExperimentSemantics(
        (
            Tracer("glc__D_e", (("#000000", 0.5), ("#111111", 0.5)), "no"),
            Tracer("co2_c", (("#0", 1.0),), "no"),
            Tracer("succ_e", (("#0000", 1.0),), "no"),
        ),
        (), observations,
    )


def load_ecoli_core_stage_b2_model(
    objective: Literal["biomass", "acetate"] = "biomass",
) -> CanonicalModel:
    """Load the complete persisted Stage B2 isotope model over the frozen LP."""

    flux_model = load_ecoli_core_flux_model(objective)
    resource = importlib.resources.files(__package__) / "data/e_coli_core_stage_b2_isotope.json"
    data = json.loads(resource.read_text(encoding="utf-8"))
    metabolites = tuple(
        IsotopeMetabolite(mid, count, True, mid in {"succ_c", "fum_c"})
        for mid, count in data["metabolites"].items()
    )
    reactions = []
    for item in data["components"]:
        branches = tuple(
            MappingBranch(
                branch["id"], branch["weight"],
                tuple(AtomTransition(AtomPosition(a, i), AtomPosition(b, j)) for a, i, b, j in branch["transitions"]),
            )
            for branch in item["branches"]
        )
        def expression(terms):
            return FluxProjectionExpression(
                tuple(FluxProjectionTerm(reaction_id, coefficient) for reaction_id, coefficient in terms)
            )
        rule = FluxProjectionRule(
            item["projection_id"], expression(item["expression"]), "positive_part", 1e-9,
            tuple(expression(value) for value in item["equivalent_expressions"]),
            tuple(PhysicalDirectionRef(reaction_id, direction) for reaction_id, direction in item["covered"]),
            tuple((key, str(value)) for key, value in item["provenance"].items()),
        )
        reactions.append(IsotopeReaction(
            item["component_id"], "forward", True,
            tuple(IsotopeParticipant(mid, tuple(range(1, data["metabolites"][mid] + 1))) for mid in item["substrates"]),
            tuple(IsotopeParticipant(mid, tuple(range(1, data["metabolites"][mid] + 1))) for mid in item["products"]),
            branches, item["projection_id"], item["symmetry"],
            tuple((key, str(value)) for key, value in item["provenance"].items()), rule,
        ))
    if data["projection_count"] != 48 or len({r.flux_projection.projection_id for r in reactions}) != 48:
        raise ValueError("frozen Stage B2 projection count changed")
    activity_resource = importlib.resources.files(__package__) / "data/e_coli_core_stage_b2_activity.json"
    activity = json.loads(activity_resource.read_text(encoding="utf-8"))
    certificate = DirectionActivityCertificate(
        activity["certificate_id"], activity["zero_tolerance"],
        tuple(DirectionActivity(item["reaction_id"], item["forward_active"], item["reverse_active"],
                                item["forward_maximum"], item["reverse_minimum"])
              for item in activity["activities"]),
        tuple(activity["provenance"].items()),
    )
    model = CanonicalModel(flux_model, IsotopeModel(metabolites, tuple(reactions), certificate))
    from fluxemu.model import validate_canonical_model
    validate_canonical_model(model)
    return model

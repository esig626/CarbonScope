"""Complete stationary flux-state -> count law -> divergence, p-value and Bruno bound.

Run from the repository with FluxEMU installed:

    python codex/examples/simple_binary_flux_discrimination.py

Two one-carbon inputs with explicitly declared atom mappings feed a balanced
one-carbon drain. Their tracers are respectively unlabelled and fully labelled.
Every hypothesis is a complete feasible native flux state in declared reaction
order; no fitting or FVA endpoint construction occurs.

Here n is a declared genuine count total in a synthetic multinomial measurement
model. It is never inferred from a MID, intensity, percentage, or peak area.
The Rényi divergences and Bruno result describe the two complete observation
laws. The exact p-value additionally requires a realised genuine-count sample
and is P0{LLR(Y) >= LLR(y_obs)}, with LLR=log(P1/P0). No chi-square or Monte
Carlo approximation is used.

Reference: Bruno, Vandenbroucque & Esposito (2026), arXiv:2601.09550v2,
Theorem 1 / Eq. (5) and Appendix A. https://arxiv.org/abs/2601.09550v2
"""

from dataclasses import replace

from fluxemu.execution import CanonicalFluxState
from fluxemu.model import (
    AtomPosition, AtomTransition, CanonicalModel, FluxMetabolite, FluxModel,
    FluxReaction, IsotopeMetabolite, IsotopeModel, IsotopeParticipant,
    IsotopeReaction, LinearObjective, MappingBranch, ObjectiveTerm,
    StationaryExperimentSemantics, StoichiometricTerm, Target, Tracer,
)
from fluxemu.observation import (
    MIDCountObservation, StationaryCountSpecification,
    StationaryObservationExperiment, StationaryObservationSpecification,
)
from fluxemu.testing import (
    BrunoTheoremAssumptionError, bruno_converse_at_order,
    evaluate_stationary_simple_hypotheses, likelihood_ratio_p_value,
    log_likelihood_ratio,
)


def _mapped_reaction(reaction_id, source, product):
    return IsotopeReaction(
        reaction_id, "forward", True,
        (IsotopeParticipant(source, (1,)),), (IsotopeParticipant(product, (1,)),),
        (MappingBranch("declared", 1.0, (
            AtomTransition(AtomPosition(source, 1), AtomPosition(product, 1)),
        )),),
    )


def _model_and_experiment():
    metabolites = (
        FluxMetabolite("S0", False), FluxMetabolite("S1", False),
        FluxMetabolite("I", True), FluxMetabolite("O", False),
    )
    reaction_declarations = (("Z_IN", "S0", "I"), ("A_IN", "S1", "I"), ("M_OUT", "I", "O"))
    flux_reactions = tuple(
        FluxReaction(name, (StoichiometricTerm(source, -1), StoichiometricTerm(product, 1)), 0, 10)
        for name, source, product in reaction_declarations
    )
    model = CanonicalModel(
        FluxModel(metabolites, flux_reactions, LinearObjective("maximise", (ObjectiveTerm("M_OUT", 1.0),))),
        IsotopeModel(
            tuple(IsotopeMetabolite(item.metabolite_id, 1, True, False) for item in metabolites),
            tuple(_mapped_reaction(*declaration) for declaration in reaction_declarations),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("S0", (("#0", 1.0),), "no"), Tracer("S1", (("#1", 1.0),), "no")),
        (Target("O-mid", "O", (1,), "intermediate", "C1", "no"),),
    )
    return model, experiment


def _state(label, unlabelled_input):
    return CanonicalFluxState(label, (
        ("Z_IN", unlabelled_input), ("A_IN", 10.0 - unlabelled_input), ("M_OUT", 10.0),
    ))


def main():
    model, experiment = _model_and_experiment()

    def specification(n):
        return StationaryObservationSpecification(model, (
            StationaryObservationExperiment("fixed-tracer", experiment, (
                StationaryCountSpecification("O-mid", n, "counts-1"),
            )),
        ))

    null, alternative = _state("v0", 2.5), _state("v1", 5.0)
    epsilon, order = 0.05, 2.0
    print("H0=P0 from v0=(2.5, 7.5, 10); H1=P1 from v1=(5, 5, 10).")
    print("Type I=P0(decide H1); Type II=P1(decide H0).")
    print(f"Type-I constraint epsilon={epsilon}; fixed Renyi order lambda={order}.")
    print("Explicit genuine counts; same p0=(0.25, 0.75), p1=(0.5, 0.5):")
    print("   n   D_lambda(P1||P0)   D_lambda(P0||P1)    reverse raw    forward raw   Type-II lower bound")
    for n in (1, 4, 16, 64):
        laws = evaluate_stationary_simple_hypotheses(
            specification(n), null_state=null, alternative_state=alternative,
        )
        bound = bruno_converse_at_order(laws, epsilon=epsilon, order=order)
        print(f"{n:4d} {bound.reverse_renyi:18.8g} {bound.forward_renyi:18.8g} "
              f"{bound.reverse_lower_bound:14.7g} {bound.forward_lower_bound:14.7g} "
              f"{bound.type_ii_lower_bound:21.7g}")

    print("Fixed n=4; changing the fixed alternative's MID separation:")
    for unlabelled in (3.75, 5.0, 7.5):
        laws = evaluate_stationary_simple_hypotheses(
            specification(4), null_state=null, alternative_state=_state("v1", unlabelled),
        )
        bound = bruno_converse_at_order(laws, epsilon=epsilon, order=order)
        print(f"p1={laws.alternative_predicted_mids[0]}: Type-II lower bound={bound.type_ii_lower_bound:.8g}")

    identical = evaluate_stationary_simple_hypotheses(
        specification(4), null_state=null, alternative_state=null,
    )
    control = bruno_converse_at_order(identical, epsilon=epsilon, order=order)
    print(f"Identical-state control: both divergences zero; Type-II lower bound={control.type_ii_lower_bound:.8g}.")

    base = specification(3)
    independent_specification = replace(base, experiments=(replace(base.experiments[0], specifications=(
        StationaryCountSpecification("O-mid", 3, "counts-1"),
        StationaryCountSpecification("O-mid", 7, "counts-2"),
    )),))
    independent = evaluate_stationary_simple_hypotheses(
        independent_specification, null_state=null, alternative_state=alternative,
        independent_blocks=True,
    )
    product_bound = bruno_converse_at_order(independent, epsilon=epsilon, order=order)
    print(f"Explicitly independent block totals {independent.law_pair.count_totals}: "
          f"Type-II lower bound={product_bound.type_ii_lower_bound:.8g}.")

    laws = evaluate_stationary_simple_hypotheses(
        specification(4), null_state=null, alternative_state=alternative,
    )
    realised_counts = MIDCountObservation((3, 1), 4)
    bound = bruno_converse_at_order(laws, epsilon=epsilon, order=order)
    llr = log_likelihood_ratio(laws, realised_counts)
    p_value = likelihood_ratio_p_value(laws, realised_counts)
    print("Realised-data comparison at n=4:")
    print(f"  genuine counts={realised_counts.counts}")
    print(f"  D_lambda(P1||P0)={bound.reverse_renyi:.8g}")
    print(f"  D_lambda(P0||P1)={bound.forward_renyi:.8g}")
    print(f"  log P1-log P0={llr:.8g}")
    print(f"  exact simple-null likelihood-ratio p-value={p_value.p_value:.8g}")
    print(f"  Type-II lower bound at epsilon={epsilon}: {bound.type_ii_lower_bound:.8g}")
    print(f"Null state fingerprint: {laws.hypotheses.null_state_fingerprint}")
    print(f"Alternative law fingerprint: {laws.law_pair.alternative_fingerprint}")
    print(f"Observation specification fingerprint: {laws.law_pair.observation_specification_fingerprint}")

    mismatch = evaluate_stationary_simple_hypotheses(
        specification(4), null_state=_state("v0", 10.0), alternative_state=alternative,
    )
    try:
        bruno_converse_at_order(mismatch, epsilon=epsilon, order=order)
    except BrunoTheoremAssumptionError as error:
        print(f"Exact-support mismatch correctly rejects Bruno bound evaluation: {error}")
    print("Each printed Bruno value is a Type-II lower bound at one fixed Renyi order;")
    print("it is neither an actual error nor a global continuous-order optimum.")
    print("The p-value is a separate realised-data null-tail probability, not a divergence or Type-II bound.")


if __name__ == "__main__":
    main()

"""Noise-free stationary MFA recovery and an explicit scale-ambiguity control.

Run from the repository root after installing ``fluxemu[mfa]``::

    python codex/examples/stationary_mfa_recovery.py

This adapts the existing native Stage 1 alternate-input fixture. Both inputs
feed the same one-carbon pool through explicit atom maps. S0 is unlabelled and
S1 is fully labelled, so the terminal MID is (Z_IN / M_OUT, A_IN / M_OUT).
Mass balance gives Z_IN + A_IN = M_OUT. An exact measured throughput, expressed
as the canonical bound M_OUT = 10, therefore identifies the complete truth
(3, 7, 10). Without that exact constraint the MID identifies only the split:
(1.5, 3.5, 5) and (3, 7, 10) have the same MID.

All observations below come directly from native stationary EMU. No noise,
normalization, statistical weights, separate forward solver, or optimizer is
introduced.
"""

from __future__ import annotations

import json

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.execution import CanonicalFluxState
from fluxemu.mfa import (
    DivergenceObjectiveConfig,
    MFAOptimizationConfig,
    StationaryMFAExperiment,
    StationaryMFAProblem,
    StationaryMIDObservation,
)
from fluxemu.mfa.stationary import fit_stationary_mfa
from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    IsotopeMetabolite,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    LinearObjective,
    MappingBranch,
    ObjectiveTerm,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
)


def _mapped_reaction(reaction_id: str, source: str, product: str) -> IsotopeReaction:
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        (IsotopeParticipant(source, (1,)),),
        (IsotopeParticipant(product, (1,)),),
        (MappingBranch(
            "declared", 1.0,
            (AtomTransition(AtomPosition(source, 1), AtomPosition(product, 1)),),
        ),),
    )


def build_mixture_problem(
    *, fixed_throughput: bool = True,
) -> tuple[StationaryMFAProblem, CanonicalFluxState]:
    """Build shared exact synthetic data, with or without measured total flux."""

    metabolites = (
        FluxMetabolite("S0", False),
        FluxMetabolite("S1", False),
        FluxMetabolite("I", True),
        FluxMetabolite("O", False),
    )
    reactions = (
        FluxReaction(
            "Z_IN", (StoichiometricTerm("S0", -1), StoichiometricTerm("I", 1)),
            0.0, 10.0,
        ),
        FluxReaction(
            "A_IN", (StoichiometricTerm("S1", -1), StoichiometricTerm("I", 1)),
            0.0, 10.0,
        ),
        FluxReaction(
            "M_OUT", (StoichiometricTerm("I", -1), StoichiometricTerm("O", 1)),
            10.0 if fixed_throughput else 2.0, 10.0,
        ),
    )
    model = CanonicalModel(
        FluxModel(
            metabolites,
            reactions,
            LinearObjective("maximise", (ObjectiveTerm("M_OUT", 1.0),)),
        ),
        IsotopeModel(
            tuple(IsotopeMetabolite(item.metabolite_id, 1, True, False)
                  for item in metabolites),
            (
                _mapped_reaction("Z_IN", "S0", "I"),
                _mapped_reaction("A_IN", "S1", "I"),
                _mapped_reaction("M_OUT", "I", "O"),
            ),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (Tracer("S0", (("#0", 1.0),), "no"), Tracer("S1", (("#1", 1.0),), "no")),
        (Target("O-mid", "O", (1,), "intermediate", "C1", "no"),),
    )
    truth = CanonicalFluxState(
        "known-truth", (("Z_IN", 3.0), ("A_IN", 7.0), ("M_OUT", 10.0)),
    )
    predicted = evaluate_stationary(compile_emu_plan(model, experiment), (truth,))
    observed = predicted.forward.predictions[0].fractions
    block = StationaryMFAExperiment(
        "labelled-mixture", experiment, (StationaryMIDObservation("O-mid", observed),),
    )
    return StationaryMFAProblem(model, (block,)), truth


def identifiable_starts() -> tuple[CanonicalFluxState, ...]:
    """Three distinct complete feasible starts, none equal to the truth."""

    return (
        CanonicalFluxState("low-unlabelled", (("Z_IN", 1.0), ("A_IN", 9.0), ("M_OUT", 10.0))),
        CanonicalFluxState("equal-inputs", (("Z_IN", 5.0), ("A_IN", 5.0), ("M_OUT", 10.0))),
        CanonicalFluxState("high-unlabelled", (("Z_IN", 9.0), ("A_IN", 1.0), ("M_OUT", 10.0))),
    )


def nonidentifiable_starts() -> tuple[CanonicalFluxState, ...]:
    """Two compatible scale witnesses and two interior, incorrect MID starts."""

    return (
        CanonicalFluxState("half-scale", (("Z_IN", 1.5), ("A_IN", 3.5), ("M_OUT", 5.0))),
        CanonicalFluxState("full-scale", (("Z_IN", 3.0), ("A_IN", 7.0), ("M_OUT", 10.0))),
        CanonicalFluxState("interior-low", (("Z_IN", 1.0), ("A_IN", 3.0), ("M_OUT", 4.0))),
        CanonicalFluxState("interior-high", (("Z_IN", 4.0), ("A_IN", 2.0), ("M_OUT", 6.0))),
    )


def run_example() -> dict[str, object]:
    """Run exact KL and two finite-order Rényi fits, then the scale control."""

    optimization = MFAOptimizationConfig(seed=26, ftol=1e-13)
    problem, truth = build_mixture_problem()
    identifiable = []
    for alpha in (1.0, 0.5, 2.0):
        result = fit_stationary_mfa(
            problem,
            objective=DivergenceObjectiveConfig(alpha=alpha),
            optimization=optimization,
            initial_states=identifiable_starts(),
        )
        identifiable.append({
            "alpha": alpha,
            "total_loss": result.total_loss,
            "fitted_fluxes": dict(result.state.values),
            "predicted_mid": result.components[0].predicted,
            "accepted_starts": sum(item.accepted for item in result.start_diagnostics),
            "max_absolute_flux_error": max(
                abs(fitted - expected)
                for (_, fitted), (_, expected) in zip(result.state.values, truth.values, strict=True)
            ),
        })

    control, _ = build_mixture_problem(fixed_throughput=False)
    witnesses = nonidentifiable_starts()
    control_result = fit_stationary_mfa(
        control, optimization=optimization, initial_states=witnesses,
    )
    block = control.experiments[0]
    witness_mids = evaluate_stationary(
        compile_emu_plan(control.model, block.experiment), witnesses[:2],
    ).forward.predictions
    return {
        "truth_fluxes": dict(truth.values),
        "observed_mid": problem.experiments[0].observations[0].fractions,
        "identifiable_with_exact_measured_throughput": identifiable,
        "nonidentifiable_without_exact_measured_throughput": {
            "identifiable_quantity": "A_IN / M_OUT = 0.7; absolute scale remains free",
            "compatible_witnesses": [
                {"fluxes": dict(state.values), "predicted_mid": predicted.fractions}
                for state, predicted in zip(witnesses[:2], witness_mids, strict=True)
            ],
            "total_loss": control_result.total_loss,
            "starts": [
                {
                    "accepted": item.accepted,
                    "optimizer_success": item.optimizer_success,
                    "validated": item.validated,
                    "final_loss": item.final_loss,
                    "final_fluxes": dict(item.final_state.values) if item.final_state else None,
                    "message": item.message,
                    "trial_failures": item.trial_failures,
                }
                for item in control_result.start_diagnostics
            ],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, allow_nan=False))

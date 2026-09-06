"""Native stationary EMU predictions joined to explicit genuine-count laws.

The observation layer retains the complete canonical state, declared experiment
and replicate order, raw count totals, and the native probability values. It
does not infer totals, normalize predictions, or invoke an MFA optimizer.
"""

from __future__ import annotations

import numpy as np

from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.exceptions import ForwardEMUError, InputValidationError, ValidationError
from fluxemu.execution import CanonicalFluxState, _validate_states
from fluxemu.flux_analysis.highs import prepare_highs_flux_region
from fluxemu.flux_analysis.sampling import (
    _require_valid_flux_states,
    validate_flux_states,
)

from .multinomial import MultinomialMIDLaw, _resolve_rng
from .schema import (
    StationaryCountSample,
    StationaryObservationLawComponent,
    StationaryObservationLawResult,
    StationaryObservationSpecification,
    _validate_state_record,
    stationary_observation_specification_fingerprint,
)


def evaluate_stationary_observation_laws(
    specification: StationaryObservationSpecification,
    states: tuple[CanonicalFluxState, ...],
) -> StationaryObservationLawResult:
    """Map complete feasible states to ordered fixed-total multinomial laws.

    Each experiment is compiled once and evaluated on the entire state batch.
    Components follow state order, then declared experiment order, then each
    experiment's target/replicate specification order. The original-model
    validator rejects incomplete, reordered, unbalanced, or infeasible states
    before EMU evaluation. No biological optimality fraction is imposed.

    Count totals are explicit declarations of genuine measurement semantics.
    Native MID values are passed unchanged to the law's probability validator;
    an invalid or numerically unrepresentable law raises without any repair.
    """

    specification_fingerprint = stationary_observation_specification_fingerprint(specification)
    if not isinstance(states, tuple) or not states:
        raise InputValidationError("states must be a nonempty immutable tuple of CanonicalFluxState records")
    for state in states:
        _validate_state_record(state)

    prepared = prepare_highs_flux_region(specification.model.flux_model, None)
    validation = validate_flux_states(prepared, states)
    _require_valid_flux_states(validation, "stationary observation-law state batch")
    # The independent original-model check above enforces scientific order and
    # steady-state balance. The native check additionally narrows bound tolerance.
    _validate_states(specification.model, states)

    plans = tuple(
        compile_emu_plan(specification.model, block.experiment)
        for block in specification.experiments
    )
    predictions = []
    for block, plan in zip(specification.experiments, plans, strict=True):
        try:
            native = evaluate_stationary(plan, states)
        except (ForwardEMUError, InputValidationError, ValidationError) as error:
            raise type(error)(
                f"stationary observation experiment {block.experiment_id!r}: {error}"
            ) from error
        predictions.append({
            (item.sample_id, item.target_id): item.fractions
            for item in native.forward.predictions
        })

    components = []
    for state in states:
        for block, plan, predicted in zip(
            specification.experiments, plans, predictions, strict=True
        ):
            for item in block.specifications:
                fractions = predicted[(state.sample_id, item.target_id)]
                try:
                    law = MultinomialMIDLaw(item.total_count, fractions)
                except (InputValidationError, ValidationError) as error:
                    raise type(error)(
                        f"stationary observation experiment {block.experiment_id!r}, "
                        f"target {item.target_id!r}, replicate {item.replicate_id!r}, "
                        f"state {state.sample_id!r}: {error}"
                    ) from error
                components.append(StationaryObservationLawComponent(
                    state, block.experiment_id, item.target_id, item.replicate_id,
                    law, plan.model_fingerprint, plan.experiment_fingerprint,
                    specification_fingerprint,
                ))
    return StationaryObservationLawResult(
        states, tuple(components), plans[0].model_fingerprint,
        tuple((block.experiment_id, plan.experiment_fingerprint)
              for block, plan in zip(specification.experiments, plans, strict=True)),
        specification_fingerprint, validation,
    )


def sample_stationary_observations(
    result: StationaryObservationLawResult,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[StationaryCountSample, ...]:
    """Draw one independent count vector per declared law, retaining its identity.

    Exactly one explicit seed or caller-owned NumPy Generator is required.
    Draws follow component order. A supplied generator advances; the law result
    remains unchanged. Every returned record retains its source component,
    predicted MID, explicit total, raw counts, and experiment/replicate identity.
    """

    if not isinstance(result, StationaryObservationLawResult):
        raise InputValidationError("result must be StationaryObservationLawResult")
    generator = _resolve_rng(seed=seed, rng=rng)
    return tuple(
        StationaryCountSample(component, component.law.sample(rng=generator))
        for component in result.components
    )


__all__ = ["evaluate_stationary_observation_laws", "sample_stationary_observations"]

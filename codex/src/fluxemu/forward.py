"""Reusable batch execution of mfapy's steady-state forward EMU function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ._mfapy import load_mfapy
from .cobra_to_mfapy import MfapyModelBundle
from .configuration import ExperimentConfig
from .exceptions import ForwardEMUError, MappingError
from .validation import validate_mid_batch


MID_COLUMNS = [
    "sample_id",
    "target_fragment_id",
    "isotopologue_index",
    "predicted_fraction",
]


@dataclass(frozen=True)
class ForwardResult:
    """Validated per-sample predictions and their long-form representation."""

    predictions: Mapping[Any, Mapping[str, list[float]]]
    mids: pd.DataFrame
    validation: Mapping[str, Any]


def _carbon_source(bundle: MfapyModelBundle, experiment: ExperimentConfig):
    try:
        carbon_source = bundle.model.generate_carbon_source_template()
    except Exception as error:
        raise ForwardEMUError(
            f"mfapy carbon-source construction failed: {error}"
        ) from error

    for tracer in experiment.tracers:
        try:
            internal_metabolite = bundle.metabolite_to_internal[
                tracer.metabolite_id
            ]
        except KeyError as error:
            raise MappingError(
                f"tracer metabolite '{tracer.metabolite_id}' has no mfapy mapping"
            ) from error
        try:
            accepted = carbon_source.set_each_isotopomer(
                internal_metabolite,
                dict(tracer.isotopomer_fractions),
                correction=tracer.correction,
            )
        except Exception as error:
            raise ForwardEMUError(
                f"mfapy rejected tracer '{tracer.metabolite_id}': {error}"
            ) from error
        if accepted is not True:
            raise ForwardEMUError(
                f"mfapy rejected tracer '{tracer.metabolite_id}'"
            )
    return carbon_source


def _requested_targets(
    bundle: MfapyModelBundle, experiment: ExperimentConfig
) -> tuple[tuple[str, str], ...]:
    requested: list[tuple[str, str]] = []
    for target in experiment.targets:
        try:
            internal_target = bundle.target_to_internal[target.fragment_id]
        except KeyError as error:
            raise MappingError(
                f"target fragment '{target.fragment_id}' has no mfapy mapping"
            ) from error
        if target.fragment_id == "X_list" or internal_target == "X_list":
            raise MappingError("internal mfapy entry 'X_list' cannot be requested")
        requested.append((target.fragment_id, internal_target))
    if len({internal for _, internal in requested}) != len(requested):
        raise MappingError("ambiguous mfapy target-fragment mapping")
    return tuple(requested)


def run_batch_forward(
    bundle: MfapyModelBundle,
    samples: pd.DataFrame,
    experiment: ExperimentConfig,
) -> ForwardResult:
    """Calculate requested MIDs for every complete sampled flux vector.

    The bundle already owns the sole constructed ``MetabolicModel`` and its
    generated ``model.func``. This function creates one carbon-source object,
    converts the full sample frame once, and reuses that function dictionary
    for every row.
    """

    if not isinstance(samples, pd.DataFrame):
        raise MappingError("batch forward calculation requires a pandas DataFrame")

    requested = _requested_targets(bundle, experiment)
    carbon_source = _carbon_source(bundle, experiment)
    carbon_source_mdvs = carbon_source.generate_dict()
    converted = bundle.converter.convert_frame(samples)
    mfapy = load_mfapy()

    predictions: dict[Any, dict[str, list[float]]] = {}
    internal_targets = [internal for _, internal in requested]
    for row_number, sample_id in enumerate(samples.index):
        try:
            _, internal_predictions = mfapy.optimize.calc_MDV_from_flux(
                converted[row_number],
                internal_targets,
                carbon_source_mdvs,
                bundle.model.func,
            )
        except Exception as error:
            raise ForwardEMUError(
                f"mfapy forward calculation failed for sample {sample_id!r}: {error}"
            ) from error
        if not isinstance(internal_predictions, Mapping):
            raise ForwardEMUError(
                f"mfapy forward calculation for sample {sample_id!r} "
                "did not return a fragment mapping"
            )

        # mfapy always includes its internal X_list in the raw hash. Select
        # only explicitly requested fragment IDs so it cannot escape.
        sample_predictions: dict[str, list[float]] = {}
        for original_target, internal_target in requested:
            if internal_target not in internal_predictions:
                continue
            try:
                sample_predictions[original_target] = [
                    float(value) for value in internal_predictions[internal_target]
                ]
            except (TypeError, ValueError) as error:
                raise ForwardEMUError(
                    f"mfapy returned a nonnumeric MID for sample {sample_id!r}, "
                    f"target {original_target!r}"
                ) from error
        predictions[sample_id] = sample_predictions

    expected_lengths = {
        target.fragment_id: len(target.atom_positions) + 1
        for target in experiment.targets
    }
    validation = validate_mid_batch(
        predictions,
        expected_lengths,
        tolerance=experiment.tolerances.mid,
    )

    rows: list[dict[str, Any]] = []
    for sample_id in samples.index:
        for target in experiment.targets:
            for isotopologue_index, fraction in enumerate(
                predictions[sample_id][target.fragment_id]
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target_fragment_id": target.fragment_id,
                        "isotopologue_index": isotopologue_index,
                        "predicted_fraction": fraction,
                    }
                )
    mids = pd.DataFrame(rows, columns=MID_COLUMNS)
    return ForwardResult(
        predictions=predictions,
        mids=mids,
        validation=validation,
    )


__all__ = ["ForwardResult", "MID_COLUMNS", "run_batch_forward"]

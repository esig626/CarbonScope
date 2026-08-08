"""Write the complete, final-only FluxEMU output bundle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import FluxEMUError


OUTPUT_FILENAMES = (
    "fba.csv",
    "fva.csv",
    "flux_samples.csv",
    "mids.csv",
    "validation_report.json",
    "run_manifest.json",
    "reaction_mapping.json",
    "mapping_provenance.json",
)

FBA_COLUMNS = ("reaction_id", "flux", "objective_value", "solver_status")
FVA_COLUMNS = (
    "reaction_id",
    "minimum",
    "maximum",
    "fraction_of_optimum",
)
MID_COLUMNS = (
    "sample_id",
    "target_fragment_id",
    "isotopologue_index",
    "predicted_fraction",
)
MAPPING_COLUMNS = (
    "original_cobra_reaction_id",
    "internal_mfapy_reaction_id",
    "mfapy_reaction_order",
)


def _output_options(output_settings: object) -> tuple[bool, int]:
    try:
        overwrite = getattr(output_settings, "overwrite")
        precision = getattr(output_settings, "float_precision")
    except AttributeError as error:
        raise FluxEMUError(
            "output_settings must expose overwrite and float_precision"
        ) from error
    if not isinstance(overwrite, bool):
        raise FluxEMUError("output_settings.overwrite must be boolean")
    if (
        isinstance(precision, bool)
        or not isinstance(precision, (int, np.integer))
        or not 1 <= int(precision) <= 17
    ):
        raise FluxEMUError("output_settings.float_precision must be from 1 to 17")
    return overwrite, int(precision)


def _prepare_output_directory(output_dir: str | Path, overwrite: bool) -> Path:
    path = Path(output_dir).expanduser()
    if path.exists() and path.is_symlink():
        raise FluxEMUError("output directory must not be a symbolic link")
    if path.exists() and not path.is_dir():
        raise FluxEMUError(f"output path exists and is not a directory: {path}")

    resolved = path.resolve(strict=False)
    if resolved == resolved.parent:
        raise FluxEMUError("filesystem root cannot be used as the output directory")

    if path.exists():
        children = list(path.iterdir())
        if children and not overwrite:
            raise FluxEMUError(
                f"output directory is not empty and overwrite is disabled: {path}"
            )
        if children:
            for child in children:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
                else:
                    raise FluxEMUError(f"cannot replace output entry: {child}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _require_unique_text(values: Iterable[Any], description: str) -> list[str]:
    result = list(values)
    if not result or not all(isinstance(value, str) and value for value in result):
        raise FluxEMUError(f"{description} must be nonempty strings")
    if len(result) != len(set(result)):
        raise FluxEMUError(f"{description} contain duplicates")
    return result


def _finite_numeric(values: Any, description: str) -> np.ndarray:
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise FluxEMUError(f"{description} must be numeric") from error
    if not np.isfinite(numeric).all():
        raise FluxEMUError(f"{description} contain non-finite values")
    return numeric


def _fba_frame(fba_result: object) -> tuple[pd.DataFrame, list[str]]:
    try:
        fluxes = getattr(fba_result, "fluxes")
        objective_value = float(getattr(fba_result, "objective_value"))
        status = getattr(fba_result, "status")
    except (AttributeError, TypeError, ValueError) as error:
        raise FluxEMUError("fba_result has an incompatible interface") from error
    if not isinstance(fluxes, pd.Series):
        raise FluxEMUError("fba_result.fluxes must be a pandas Series")
    reaction_ids = _require_unique_text(fluxes.index, "FBA reaction IDs")
    numeric_fluxes = _finite_numeric(fluxes.to_numpy(), "FBA fluxes")
    if not np.isfinite(objective_value):
        raise FluxEMUError("FBA objective value must be finite")
    if not isinstance(status, str) or not status:
        raise FluxEMUError("FBA solver status must be a nonempty string")
    frame = pd.DataFrame(
        {
            "reaction_id": reaction_ids,
            "flux": numeric_fluxes,
            "objective_value": objective_value,
            "solver_status": status,
        }
    )
    return frame.loc[:, list(FBA_COLUMNS)], reaction_ids


def _fva_frame(fva_result: object, reaction_ids: list[str]) -> pd.DataFrame:
    try:
        ranges = getattr(fva_result, "ranges")
        fraction = float(getattr(fva_result, "fraction_of_optimum"))
    except (AttributeError, TypeError, ValueError) as error:
        raise FluxEMUError("fva_result has an incompatible interface") from error
    if not isinstance(ranges, pd.DataFrame):
        raise FluxEMUError("fva_result.ranges must be a pandas DataFrame")
    if list(ranges.columns) != ["minimum", "maximum"]:
        raise FluxEMUError("FVA ranges must have exact minimum and maximum columns")
    if list(ranges.index) != reaction_ids:
        raise FluxEMUError("FVA reactions must match the complete FBA reaction order")
    numeric_ranges = _finite_numeric(ranges.to_numpy(), "FVA ranges")
    if not np.isfinite(fraction):
        raise FluxEMUError("FVA fraction of optimum must be finite")
    frame = pd.DataFrame(
        {
            "reaction_id": reaction_ids,
            "minimum": numeric_ranges[:, 0],
            "maximum": numeric_ranges[:, 1],
            "fraction_of_optimum": fraction,
        }
    )
    return frame.loc[:, list(FVA_COLUMNS)]


def _sample_frame(
    sampling_result: object, reaction_ids: list[str]
) -> pd.DataFrame:
    try:
        samples = getattr(sampling_result, "samples")
    except AttributeError as error:
        raise FluxEMUError("sampling_result has an incompatible interface") from error
    if not isinstance(samples, pd.DataFrame) or samples.empty:
        raise FluxEMUError("sampling_result.samples must be a nonempty DataFrame")
    if "sample_id" in samples.columns:
        raise FluxEMUError("sample_id conflicts with a COBRA reaction identifier")
    if list(samples.columns) != reaction_ids:
        raise FluxEMUError(
            "flux-sample columns must match the complete FBA reaction order"
        )
    sample_ids = _require_unique_text(samples.index, "flux sample IDs")
    numeric_samples = _finite_numeric(samples.to_numpy(), "flux samples")
    frame = pd.DataFrame(numeric_samples, columns=reaction_ids)
    frame.insert(0, "sample_id", sample_ids)
    return frame


def _mid_frame(mids_long: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(mids_long, pd.DataFrame) or mids_long.empty:
        raise FluxEMUError("mids_long must be a nonempty pandas DataFrame")
    if list(mids_long.columns) != list(MID_COLUMNS):
        raise FluxEMUError(
            "mids_long must have exact columns: " + ", ".join(MID_COLUMNS)
        )
    frame = mids_long.copy()
    if not frame["sample_id"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise FluxEMUError("MID sample IDs must be nonempty strings")
    if not frame["target_fragment_id"].map(
        lambda value: isinstance(value, str) and bool(value)
    ).all():
        raise FluxEMUError("MID target fragment IDs must be nonempty strings")
    isotopologues = _finite_numeric(
        frame["isotopologue_index"].to_numpy(), "MID isotopologue indices"
    )
    if (isotopologues < 0).any() or not np.equal(isotopologues, np.floor(isotopologues)).all():
        raise FluxEMUError("MID isotopologue indices must be nonnegative integers")
    frame["isotopologue_index"] = isotopologues.astype(int)
    frame["predicted_fraction"] = _finite_numeric(
        frame["predicted_fraction"].to_numpy(), "MID predicted fractions"
    )
    return frame.loc[:, list(MID_COLUMNS)]


def _mapping_payload(reaction_mappings: Iterable[object]) -> list[dict[str, Any]]:
    try:
        entries = list(reaction_mappings)
    except TypeError as error:
        raise FluxEMUError("reaction_mappings must be iterable") from error
    if not entries:
        raise FluxEMUError("reaction_mappings must not be empty")

    payload: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            data = entry
        else:
            converter = getattr(entry, "to_dict", None)
            if not callable(converter):
                raise FluxEMUError(
                    f"reaction mapping {position} must be a mapping or expose to_dict"
                )
            data = converter()
        if not isinstance(data, Mapping):
            raise FluxEMUError(
                f"reaction mapping {position} to_dict() did not return a mapping"
            )
        missing = [name for name in MAPPING_COLUMNS if name not in data]
        if missing:
            raise FluxEMUError(
                f"reaction mapping {position} is missing fields: {', '.join(missing)}"
            )
        row = {name: data[name] for name in MAPPING_COLUMNS}
        if not isinstance(row["original_cobra_reaction_id"], str) or not row[
            "original_cobra_reaction_id"
        ]:
            raise FluxEMUError("mapping original COBRA IDs must be nonempty strings")
        if not isinstance(row["internal_mfapy_reaction_id"], str) or not row[
            "internal_mfapy_reaction_id"
        ]:
            raise FluxEMUError("mapping internal mfapy IDs must be nonempty strings")
        order = row["mfapy_reaction_order"]
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0:
            raise FluxEMUError("mfapy reaction order must be a nonnegative integer")
        row["mfapy_reaction_order"] = int(order)
        payload.append(row)

    payload.sort(key=lambda row: row["mfapy_reaction_order"])
    original_ids = [row["original_cobra_reaction_id"] for row in payload]
    internal_ids = [row["internal_mfapy_reaction_id"] for row in payload]
    orders = [row["mfapy_reaction_order"] for row in payload]
    if len(original_ids) != len(set(original_ids)):
        raise FluxEMUError("reaction mapping contains duplicate original COBRA IDs")
    if len(internal_ids) != len(set(internal_ids)):
        raise FluxEMUError("reaction mapping contains duplicate internal mfapy IDs")
    if orders != list(range(len(payload))):
        raise FluxEMUError("mfapy reaction mapping order must be contiguous from zero")
    return payload


def _jsonable(value: Any, context: str) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FluxEMUError(f"{context} contains a non-string JSON key")
            result[key] = _jsonable(item, context)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, context) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise FluxEMUError(f"{context} contains a non-finite JSON number")
        return value
    raise FluxEMUError(
        f"{context} contains a value that is not JSON serializable: "
        f"{type(value).__name__}"
    )


def _json_text(value: Mapping[str, Any] | list[dict[str, Any]], context: str) -> str:
    normalized = _jsonable(value, context)
    try:
        return json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise FluxEMUError(f"{context} is not strict JSON: {error}") from error


def _csv_text(frame: pd.DataFrame, precision: int) -> str:
    return frame.to_csv(
        index=False,
        float_format=f"%.{precision}g",
        lineterminator="\n",
    )


def write_outputs(
    output_dir: str | Path,
    fba_result: object,
    fva_result: object,
    sampling_result: object,
    mids_long: pd.DataFrame,
    validation_report: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    reaction_mappings: Iterable[object],
    output_settings: object,
    *,
    mapping_provenance: Iterable[object] | None = None,
) -> dict[str, Path]:
    """Write exactly the seven required FluxEMU outputs.

    All tables and JSON values are validated and serialized in memory before
    the destination is changed, preventing invalid numeric values from leaving
    a partial output bundle.
    """

    overwrite, precision = _output_options(output_settings)
    if not isinstance(validation_report, Mapping):
        raise FluxEMUError("validation_report must be a mapping")
    if not isinstance(run_manifest, Mapping):
        raise FluxEMUError("run_manifest must be a mapping")

    fba_frame, reaction_ids = _fba_frame(fba_result)
    fva_frame = _fva_frame(fva_result, reaction_ids)
    sample_frame = _sample_frame(sampling_result, reaction_ids)
    mid_frame = _mid_frame(mids_long)
    mapping_payload = _mapping_payload(reaction_mappings)
    provenance_payload = _provenance_payload(mapping_provenance)

    serialized = {
        "fba.csv": _csv_text(fba_frame, precision),
        "fva.csv": _csv_text(fva_frame, precision),
        "flux_samples.csv": _csv_text(sample_frame, precision),
        "mids.csv": _csv_text(mid_frame, precision),
        "validation_report.json": _json_text(
            validation_report, "validation_report"
        ),
        "run_manifest.json": _json_text(run_manifest, "run_manifest"),
        "reaction_mapping.json": _json_text(
            mapping_payload, "reaction_mappings"
        ),
        "mapping_provenance.json": _json_text(
            provenance_payload, "mapping_provenance"
        ),
    }

    directory = _prepare_output_directory(output_dir, overwrite)
    paths: dict[str, Path] = {}
    for filename in OUTPUT_FILENAMES:
        path = directory / filename
        path.write_text(serialized[filename], encoding="utf-8", newline="")
        paths[filename] = path
    return paths


def _provenance_payload(entries: Iterable[object] | None) -> list[dict[str, Any]]:
    """Return the standalone auditable mapping-resolution report."""

    if entries is None:
        return []
    payload: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            item = entry
        else:
            converter = getattr(entry, "to_dict", None)
            if not callable(converter):
                raise FluxEMUError(
                    f"mapping provenance record {index} must be a mapping or expose to_dict"
                )
            item = converter()
        required = {
            "model_reaction_id", "canonical_transition_id", "mapping_source",
            "source_identifier", "validation_status", "symmetry_treatment", "warning",
        }
        if not isinstance(item, Mapping) or set(item) != required:
            raise FluxEMUError(
                f"mapping provenance record {index} does not have the required schema"
            )
        payload.append(dict(item))
    return payload


__all__ = ["OUTPUT_FILENAMES", "write_outputs"]

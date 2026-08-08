"""End-to-end coordination for the FluxEMU prototype."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import cobra
from cobra.io import read_sbml_model

from . import __version__
from ._mfapy import load_mfapy, mfapy_source_path
from .cobra_analysis import FBAResult, FVAResult, FluxSamplingResult, run_fba, run_fva, sample_fluxes
from .cobra_to_mfapy import MfapyModelBundle, build_mfapy_model
from .carbon_transitions import MappingProvenanceRecord, resolve_model_metadata
from .configuration import ExperimentConfig, load_experiment
from .exceptions import FluxEMUError, InputValidationError
from .forward import ForwardResult, run_batch_forward
from .isotope_metadata import collect_isotope_metadata
from .output import write_outputs


@dataclass(frozen=True)
class PipelineResult:
    """In-memory results and paths from one complete pipeline execution."""

    fba: FBAResult
    fva: FVAResult
    sampling: FluxSamplingResult
    mfapy_bundle: MfapyModelBundle
    forward: ForwardResult
    validation_report: Mapping[str, Any]
    mapping_provenance: tuple[MappingProvenanceRecord, ...]
    run_manifest: Mapping[str, Any]
    output_paths: Mapping[str, Path]


def _input_path(path: str | Path, description: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise InputValidationError(f"{description} does not exist: {result}")
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solver_name(model: cobra.Model) -> str:
    interface_name = getattr(model.solver.interface, "__name__", "unknown")
    return interface_name.rsplit(".", 1)[-1].removesuffix("_interface")


def _mfapy_identifier() -> str:
    module = load_mfapy()
    version = getattr(module, "__version__", None)
    return str(version) if version else "mfapy-0.6.3-vendored-source"


def _validation_report(
    bundle: MfapyModelBundle,
    sampling: FluxSamplingResult,
    forward: ForwardResult,
    mapping_provenance: Sequence[MappingProvenanceRecord],
) -> dict[str, Any]:
    sample_report = sampling.validation.to_dict()
    details = sample_report["details"]
    bound_violations = [
        {
            "sample_id": item["sample_id"],
            "lower": item["max_lower_bound_violation"],
            "upper": item["max_upper_bound_violation"],
        }
        for item in details
        if not item["bounds_valid"]
    ]
    mass_residuals = {
        item["sample_id"]: item["max_mass_balance_residual"] for item in details
    }
    objective_values = {
        item["sample_id"]: item["objective_value"] for item in details
    }
    reaction_correspondence = [entry.to_dict() for entry in bundle.reaction_mappings]
    return {
        "valid": bool(sampling.validation.valid and forward.validation["valid"]),
        "mapping_validation": {
            "valid": True,
            "mapped_reaction_count": len(bundle.reaction_mappings),
            "missing": [],
            "duplicate": [],
            "ambiguous": [],
        },
        "reaction_order_validation": {
            "valid": True,
            "original_cobra_order": list(bundle.converter.original_reaction_order),
            "internal_mfapy_order": list(bundle.converter.mfapy_reaction_order),
        },
        "sample_validation_summary": sample_report,
        "mass_balance_residuals": mass_residuals,
        "bound_violations": bound_violations,
        "objective_floor_validation": {
            "valid": sampling.validation.objective_floor_valid,
            "direction": sampling.objective_direction,
            "threshold": sampling.objective_floor,
            "objective_values": objective_values,
            "max_violation": sampling.validation.max_objective_floor_violation,
        },
        "mid_validation": dict(forward.validation),
        "warnings": [],
        "mapping_provenance": [item.to_dict() for item in mapping_provenance],
        "generated_internal_id_correspondence": {
            "reactions": reaction_correspondence,
            "metabolites": dict(bundle.metabolite_to_internal),
            "target_fragments": dict(bundle.target_to_internal),
        },
    }


def _manifest(
    model: cobra.Model,
    model_path: Path,
    experiment_path: Path,
    experiment: ExperimentConfig,
    sampling: FluxSamplingResult,
    cli_arguments: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "fluxemu_version": __version__,
        "stable_cobrapy_version": cobra.__version__,
        "mfapy_source_identifier": _mfapy_identifier(),
        "solver": _solver_name(model),
        "sampler": sampling.sampler,
        "seed": sampling.seed,
        "objective_direction": sampling.objective_direction,
        "objective_optimum": sampling.objective_value,
        "objective_fraction": sampling.fraction_of_optimum,
        "objective_floor": sampling.objective_floor,
        "requested_sample_count": experiment.sample_count,
        "accepted_sample_count": len(sampling.samples),
        "input_hashes": {
            "model": {
                "path": str(model_path),
                "sha256": _file_sha256(model_path),
            },
            "experiment": {
                "path": str(experiment_path),
                "sha256": _file_sha256(experiment_path),
            },
        },
        "package_source_paths": {
            "fluxemu": str(Path(__file__).resolve().parent),
            "cobra": str(Path(cobra.__file__).resolve().parent),
            "mfapy": str(mfapy_source_path()),
        },
        "tolerances": experiment.tolerances.to_dict(),
        "output_settings": experiment.output.to_dict(),
        "cli_arguments": list(cli_arguments or ()),
    }


def run_pipeline(
    model_path: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
    *,
    cli_arguments: Sequence[str] | None = None,
) -> PipelineResult:
    """Execute SBML through complete feasible sampling and forward MIDs."""

    model_file = _input_path(model_path, "SBML model")
    experiment_file = _input_path(experiment_path, "experiment YAML")
    try:
        model = read_sbml_model(model_file)
    except Exception as error:
        raise InputValidationError(f"could not load SBML model '{model_file}': {error}") from error
    experiment = load_experiment(experiment_file)
    isotope_metadata = collect_isotope_metadata(model)
    resolved_metadata = resolve_model_metadata(model, isotope_metadata)

    fba = run_fba(model)
    fva = run_fva(model, experiment.fraction_of_optimum)
    sampling = sample_fluxes(
        model,
        experiment.sample_count,
        experiment.fraction_of_optimum,
        experiment.seed,
        experiment.sampler,
        experiment.tolerances,
    )
    expected_floor = fba.objective_value * experiment.fraction_of_optimum
    if abs(sampling.objective_floor - expected_floor) > experiment.tolerances.objective_floor:
        raise FluxEMUError("FVA/FBA and sampling objective thresholds are inconsistent")

    bundle = build_mfapy_model(model, resolved_metadata.metadata, experiment)
    forward = run_batch_forward(bundle, sampling.samples, experiment)
    validation_report = _validation_report(
        bundle, sampling, forward, resolved_metadata.provenance
    )
    if not validation_report["valid"]:
        raise FluxEMUError("pipeline validation report is not valid")
    manifest = _manifest(
        model,
        model_file,
        experiment_file,
        experiment,
        sampling,
        cli_arguments,
    )
    paths = write_outputs(
        output_dir,
        fba,
        fva,
        sampling,
        forward.mids,
        validation_report,
        manifest,
        bundle.reaction_mappings,
        experiment.output,
        mapping_provenance=resolved_metadata.provenance,
    )
    return PipelineResult(
        fba=fba,
        fva=fva,
        sampling=sampling,
        mfapy_bundle=bundle,
        forward=forward,
        validation_report=validation_report,
        mapping_provenance=resolved_metadata.provenance,
        run_manifest=manifest,
        output_paths=paths,
    )


__all__ = ["PipelineResult", "run_pipeline"]

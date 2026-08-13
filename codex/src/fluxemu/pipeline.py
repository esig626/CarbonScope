"""Public standalone native stationary pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
from fluxemu.analysis import run_native_stationary_analysis
from fluxemu.model import load_sbml_flux_model, model_fingerprint, experiment_fingerprint
from fluxemu.native_io import load_native_stationary_spec

@dataclass(frozen=True, slots=True)
class NativePipelineResult:
    analysis: object
    output_directory: Path
    manifest: dict[str, object]

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_pipeline(model_path, experiment_path, output_directory, *, cli_arguments=()):
    """Run SBML -> canonical model -> native HiGHS -> stationary EMU."""
    model_path, experiment_path = Path(model_path), Path(experiment_path)
    flux_model=load_sbml_flux_model(model_path)
    model,experiment,fraction=load_native_stationary_spec(experiment_path,flux_model)
    result=run_native_stationary_analysis(model,experiment,fva_fraction_of_optimum=fraction)
    output=Path(output_directory); output.mkdir(parents=True,exist_ok=True)
    result.fba.to_frame().to_csv(output/"fba_fluxes.csv")
    result.fva.to_frame().to_csv(output/"fva_ranges.csv")
    mids=[{"sample_id":v.sample_id,"target_id":v.target_id,"isotopologue_index":v.isotopologue_index,"predicted_fraction":v.predicted_fraction} for v in result.mids.forward.values]
    (output/"predicted_mids.json").write_text(json.dumps(mids,indent=2)+"\n")
    diagnostics=[{"sample_id":d.sample_id,"emu_size":d.emu_size,"matrix_dimension":d.matrix_dimension,"rank":d.rank,"condition_number":d.condition_number,"max_absolute_residual":d.max_absolute_residual,"minimum_component":d.minimum_component,"max_normalization_error":d.max_normalization_error} for d in result.mids.layer_diagnostics]
    (output/"emu_diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    manifest={"schema_version":1,"engine":"fluxemu-native-stationary","sampling_performed":False,"objective_value":result.fba.objective_value,"model_sha256":_sha256(model_path),"experiment_sha256":_sha256(experiment_path),"canonical_model_fingerprint":model_fingerprint(model),"experiment_fingerprint":experiment_fingerprint(experiment),"fva_fraction_of_optimum":fraction,"cli_arguments":list(cli_arguments)}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return NativePipelineResult(result,output,manifest)

__all__=["NativePipelineResult","run_pipeline"]

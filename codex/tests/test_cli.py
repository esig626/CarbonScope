from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


CODEX_ROOT = Path(__file__).resolve().parents[1]
MODEL = CODEX_ROOT / "tests" / "fixtures" / "native_portability" / "model.xml"
EXPERIMENT = (
    CODEX_ROOT / "tests" / "fixtures" / "native_portability" / "experiment.yaml"
)
NATIVE_OUTPUT_FILENAMES = {
    "fba_fluxes.csv",
    "fva_ranges.csv",
    "predicted_mids.json",
    "emu_diagnostics.json",
    "manifest.json",
}


@pytest.fixture(scope="module")
def cli_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    output = tmp_path_factory.mktemp("fluxemu_native_cli") / "smoke"
    command = [
        sys.executable,
        "-m",
        "fluxemu.cli",
        "run",
        "--model",
        str(MODEL),
        "--experiment",
        str(EXPERIMENT),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=CODEX_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return output, result


def test_complete_native_cli_smoke_and_exact_files(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, result = cli_run
    assert "FluxEMU native stationary analysis completed: 7 MID rows" in result.stdout
    assert {path.name for path in output.iterdir()} == NATIVE_OUTPUT_FILENAMES


def test_native_cli_flux_outputs_are_complete_and_ordered(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, _ = cli_run
    fba = pd.read_csv(output / "fba_fluxes.csv")
    fva = pd.read_csv(output / "fva_ranges.csv")

    assert list(fba.columns) == [
        "reaction_id",
        "flux",
        "objective_value",
        "solver_status",
    ]
    assert fba["reaction_id"].tolist() == ["foreign_hx", "foreign_sink"]
    assert fba["flux"].tolist() == pytest.approx([10.0, 10.0])
    assert fba["objective_value"].tolist() == pytest.approx([10.0, 10.0])
    assert fba["solver_status"].tolist() == ["optimal", "optimal"]

    assert list(fva.columns) == [
        "reaction_id",
        "minimum",
        "maximum",
        "fraction_of_optimum",
    ]
    assert fva["reaction_id"].tolist() == ["foreign_hx", "foreign_sink"]
    assert fva["minimum"].tolist() == pytest.approx([10.0, 10.0])
    assert fva["maximum"].tolist() == pytest.approx([10.0, 10.0])
    assert fva["fraction_of_optimum"].tolist() == pytest.approx([1.0, 1.0])


def test_native_cli_mid_outputs_and_diagnostics_are_valid(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, _ = cli_run
    mids = json.loads((output / "predicted_mids.json").read_text())
    diagnostics = json.loads((output / "emu_diagnostics.json").read_text())

    assert len(mids) == 7
    assert all(
        set(row)
        == {
            "sample_id",
            "target_id",
            "isotopologue_index",
            "predicted_fraction",
        }
        for row in mids
    )
    assert [row["sample_id"] for row in mids] == ["fba-optimum"] * 7
    assert [row["target_id"] for row in mids] == ["portable_mid"] * 7
    assert [row["isotopologue_index"] for row in mids] == list(range(7))
    fractions = [row["predicted_fraction"] for row in mids]
    assert all(math.isfinite(value) and value >= 0.0 for value in fractions)
    assert fractions == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    assert sum(fractions) == pytest.approx(1.0, abs=1e-12)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["sample_id"] == "fba-optimum"
    assert diagnostic["emu_size"] == 6
    assert diagnostic["matrix_dimension"] == 1
    assert diagnostic["rank"] == 1
    assert math.isfinite(diagnostic["condition_number"])
    assert diagnostic["max_absolute_residual"] <= 1e-12
    assert diagnostic["minimum_component"] >= 0.0
    assert diagnostic["max_normalization_error"] <= 1e-12


def test_native_cli_manifest_records_deterministic_no_sampling_run(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, _ = cli_run
    manifest = json.loads((output / "manifest.json").read_text())

    assert set(manifest) == {
        "schema_version",
        "engine",
        "sampling_performed",
        "objective_value",
        "model_sha256",
        "experiment_sha256",
        "canonical_model_fingerprint",
        "experiment_fingerprint",
        "fva_fraction_of_optimum",
        "cli_arguments",
    }
    assert manifest["schema_version"] == 1
    assert manifest["engine"] == "fluxemu-native-stationary"
    assert manifest["sampling_performed"] is False
    assert manifest["objective_value"] == pytest.approx(10.0)
    assert manifest["fva_fraction_of_optimum"] == pytest.approx(1.0)
    for field in (
        "model_sha256",
        "experiment_sha256",
        "canonical_model_fingerprint",
        "experiment_fingerprint",
    ):
        assert len(manifest[field]) == 64
    assert manifest["cli_arguments"] == [
        "run",
        "--model",
        str(MODEL),
        "--experiment",
        str(EXPERIMENT),
        "--output",
        str(output),
    ]
    assert not {
        "stable_cobrapy_version",
        "mfapy_source_identifier",
        "sampler",
        "accepted_sample_count",
    } & set(manifest)

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fluxemu.output import OUTPUT_FILENAMES


CODEX_ROOT = Path(__file__).resolve().parents[1]
MODEL = CODEX_ROOT / "examples" / "toy_model.xml"
EXPERIMENT = CODEX_ROOT / "examples" / "toy_experiment.yaml"


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, subprocess.CompletedProcess[str]]:
    output = tmp_path_factory.mktemp("fluxemu_cli") / "smoke"
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


def test_complete_cli_smoke_and_expected_files(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, result = cli_run
    assert "FluxEMU completed: 6 samples, 36 MID rows" in result.stdout
    assert {path.name for path in output.iterdir()} == set(OUTPUT_FILENAMES)


def test_cli_output_content_is_complete_and_valid(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, _ = cli_run
    fba = pd.read_csv(output / "fba.csv")
    fva = pd.read_csv(output / "fva.csv")
    samples = pd.read_csv(output / "flux_samples.csv")
    mids = pd.read_csv(output / "mids.csv")
    validation = json.loads((output / "validation_report.json").read_text())
    mapping = json.loads((output / "reaction_mapping.json").read_text())

    assert list(fba.columns) == [
        "reaction_id",
        "flux",
        "objective_value",
        "solver_status",
    ]
    assert len(fba) == 13 and set(fba["solver_status"]) == {"optimal"}
    assert len(fva) == 13 and set(fva["fraction_of_optimum"]) == {0.9}
    assert samples.shape == (6, 14)
    assert samples["sample_id"].tolist() == [f"sample_{index:04d}" for index in range(6)]
    assert mids.shape == (36, 4)
    grouped = mids.groupby(["sample_id", "target_fragment_id"])[
        "predicted_fraction"
    ].sum()
    assert np.allclose(grouped.to_numpy(), 1.0, rtol=0.0, atol=1e-8)
    assert validation["valid"] is True
    assert validation["mapping_validation"]["valid"] is True
    assert validation["reaction_order_validation"]["valid"] is True
    assert validation["sample_validation_summary"]["valid"] is True
    assert validation["mid_validation"]["valid"] is True
    assert len(mapping) == 9
    assert [entry["mfapy_reaction_order"] for entry in mapping] == list(range(9))


def test_run_manifest_is_complete(
    cli_run: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    output, _ = cli_run
    manifest = json.loads((output / "run_manifest.json").read_text())
    required = {
        "timestamp",
        "python_version",
        "python_executable",
        "fluxemu_version",
        "stable_cobrapy_version",
        "mfapy_source_identifier",
        "solver",
        "sampler",
        "seed",
        "objective_fraction",
        "requested_sample_count",
        "accepted_sample_count",
        "input_hashes",
        "package_source_paths",
        "tolerances",
        "cli_arguments",
    }
    assert required <= set(manifest)
    assert manifest["stable_cobrapy_version"] == "0.31.1"
    assert manifest["sampler"] == "achr"
    assert manifest["seed"] == 1729
    assert manifest["objective_fraction"] == 0.9
    assert manifest["requested_sample_count"] == 6
    assert manifest["accepted_sample_count"] == 6
    for input_data in manifest["input_hashes"].values():
        assert len(input_data["sha256"]) == 64
    assert {"fluxemu", "cobra", "mfapy"} == set(manifest["package_source_paths"])

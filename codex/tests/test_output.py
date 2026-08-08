"""Focused tests for the final FluxEMU output bundle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluxemu.configuration import OutputSettings
from fluxemu.cobra_analysis import FBAResult, FVAResult
from fluxemu.exceptions import FluxEMUError
from fluxemu.flux_conversion import ReactionMapping
from fluxemu.output import OUTPUT_FILENAMES, write_outputs


@pytest.fixture
def output_bundle() -> dict[str, object]:
    reaction_ids = ["R_SOURCE", "R_PRODUCT"]
    fba_result = FBAResult(
        objective_value=8.0,
        status="optimal",
        objective_direction="max",
        fluxes=pd.Series(
            [8.0, 8.0],
            index=pd.Index(reaction_ids, name="reaction_id"),
            name="flux",
        ),
    )
    fva_result = FVAResult(
        ranges=pd.DataFrame(
            {"minimum": [7.2, 7.2], "maximum": [8.0, 8.0]},
            index=pd.Index(reaction_ids, name="reaction_id"),
        ),
        fraction_of_optimum=0.9,
        objective_value=8.0,
        objective_direction="max",
    )
    samples = pd.DataFrame(
        [[7.25, 7.25], [7.75, 7.75]],
        index=pd.Index(["sample_0000", "sample_0001"], name="sample_id"),
        columns=reaction_ids,
    )
    sampling_result = SimpleNamespace(samples=samples)
    mids_long = pd.DataFrame(
        {
            "sample_id": ["sample_0000", "sample_0000", "sample_0001", "sample_0001"],
            "target_fragment_id": ["frag", "frag", "frag", "frag"],
            "isotopologue_index": [0, 1, 0, 1],
            "predicted_fraction": [0.7, 0.3, 0.6, 0.4],
        }
    )
    validation_report = {
        "mapping_validation": {"valid": True},
        "sample_validation": {"accepted": 2, "valid": True},
        "warnings": [],
    }
    run_manifest = {
        "accepted_sample_count": np.int64(2),
        "cobra_version": "0.31.1",
        "objective_fraction": 0.9,
        "seed": 11,
        "solver": "glpk",
    }
    mappings = (
        {
            "original_cobra_reaction_id": "R_PRODUCT",
            "internal_mfapy_reaction_id": "r_product",
            "mfapy_reaction_order": 1,
        },
        ReactionMapping("R_SOURCE", "r_source", 0),
    )
    return {
        "fba_result": fba_result,
        "fva_result": fva_result,
        "sampling_result": sampling_result,
        "mids_long": mids_long,
        "validation_report": validation_report,
        "run_manifest": run_manifest,
        "reaction_mappings": mappings,
        "output_settings": OutputSettings(overwrite=False, float_precision=10),
    }


def test_write_outputs_creates_exact_files_and_schemas(
    tmp_path: Path, output_bundle: dict[str, object]
) -> None:
    output_dir = tmp_path / "nested" / "run"

    paths = write_outputs(output_dir=output_dir, **output_bundle)

    assert tuple(paths) == OUTPUT_FILENAMES
    assert set(path.name for path in output_dir.iterdir()) == set(OUTPUT_FILENAMES)
    assert all(path == output_dir / filename for filename, path in paths.items())
    assert all(path.is_file() for path in paths.values())

    fba = pd.read_csv(paths["fba.csv"])
    assert list(fba.columns) == [
        "reaction_id",
        "flux",
        "objective_value",
        "solver_status",
    ]
    assert fba["reaction_id"].tolist() == ["R_SOURCE", "R_PRODUCT"]

    fva = pd.read_csv(paths["fva.csv"])
    assert list(fva.columns) == [
        "reaction_id",
        "minimum",
        "maximum",
        "fraction_of_optimum",
    ]

    samples = pd.read_csv(paths["flux_samples.csv"])
    assert list(samples.columns) == ["sample_id", "R_SOURCE", "R_PRODUCT"]
    assert samples["sample_id"].tolist() == ["sample_0000", "sample_0001"]

    mids = pd.read_csv(paths["mids.csv"])
    assert list(mids.columns) == [
        "sample_id",
        "target_fragment_id",
        "isotopologue_index",
        "predicted_fraction",
    ]


def test_json_outputs_are_strict_readable_and_deterministic(
    tmp_path: Path, output_bundle: dict[str, object]
) -> None:
    first = write_outputs(output_dir=tmp_path / "first", **output_bundle)
    second = write_outputs(output_dir=tmp_path / "second", **output_bundle)

    validation = json.loads(first["validation_report.json"].read_text())
    manifest = json.loads(first["run_manifest.json"].read_text())
    mappings = json.loads(first["reaction_mapping.json"].read_text())
    provenance = json.loads(first["mapping_provenance.json"].read_text())
    assert validation == output_bundle["validation_report"]
    assert manifest["accepted_sample_count"] == 2
    assert manifest["cobra_version"] == "0.31.1"
    assert [entry["mfapy_reaction_order"] for entry in mappings] == [0, 1]
    assert [entry["original_cobra_reaction_id"] for entry in mappings] == [
        "R_SOURCE",
        "R_PRODUCT",
    ]
    assert provenance == []

    for filename in (
        "validation_report.json",
        "run_manifest.json",
        "reaction_mapping.json",
    ):
        first_text = first[filename].read_text()
        assert first_text.endswith("\n")
        assert first_text == second[filename].read_text()
        assert "NaN" not in first_text
        assert "Infinity" not in first_text


def test_nonempty_output_requires_overwrite_and_overwrite_clears_it(
    tmp_path: Path, output_bundle: dict[str, object]
) -> None:
    output_dir = tmp_path / "run"
    nested = output_dir / "old"
    nested.mkdir(parents=True)
    sentinel = nested / "sentinel.txt"
    sentinel.write_text("preserve unless explicitly overwritten")

    with pytest.raises(FluxEMUError, match="not empty"):
        write_outputs(output_dir=output_dir, **output_bundle)
    assert sentinel.is_file()

    overwrite_bundle = dict(output_bundle)
    overwrite_bundle["output_settings"] = OutputSettings(
        overwrite=True, float_precision=10
    )
    write_outputs(output_dir=output_dir, **overwrite_bundle)

    assert not sentinel.exists()
    assert set(path.name for path in output_dir.iterdir()) == set(OUTPUT_FILENAMES)


def test_nonfinite_json_is_rejected_before_directory_creation(
    tmp_path: Path, output_bundle: dict[str, object]
) -> None:
    output_dir = tmp_path / "run"
    invalid_bundle = dict(output_bundle)
    invalid_bundle["run_manifest"] = {"invalid": float("nan")}

    with pytest.raises(FluxEMUError, match="non-finite JSON number"):
        write_outputs(output_dir=output_dir, **invalid_bundle)

    assert not output_dir.exists()


def test_sample_columns_must_be_complete_and_in_fba_order(
    tmp_path: Path, output_bundle: dict[str, object]
) -> None:
    invalid_bundle = dict(output_bundle)
    samples = output_bundle["sampling_result"].samples.copy()  # type: ignore[union-attr]
    invalid_bundle["sampling_result"] = SimpleNamespace(
        samples=samples[["R_PRODUCT", "R_SOURCE"]]
    )

    with pytest.raises(FluxEMUError, match="complete FBA reaction order"):
        write_outputs(output_dir=tmp_path / "run", **invalid_bundle)

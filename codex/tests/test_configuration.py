"""Tests for strict experiment-YAML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxemu.configuration import load_experiment_config
from fluxemu.exceptions import ConfigurationError


VALID_EXPERIMENT = """
schema_version: 1
tracers:
  - metabolite: source_c
    isotopomers:
      "#00": 0.25
      "#10": 0.75
    correction: yes
targets:
  - id: product_m12
    metabolite: product_c
    atoms: [1, 2]
    analytical_method: gcms
    formula: C2H4O2
    correction: no
fraction_of_optimum: 0.9
sample_count: 8
sampler: achr
seed: 1234
tolerances:
  bounds: 1.0e-7
  mass_balance: 2.0e-7
  objective_floor: 3.0e-7
  mid: 1.0e-8
  tracer_normalization: 1.0e-9
output:
  overwrite: true
  float_precision: 14
"""


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_experiment_yaml_parsing(tmp_path: Path) -> None:
    config = load_experiment_config(_write_yaml(tmp_path, VALID_EXPERIMENT))

    assert config.schema_version == 1
    assert config.fraction_of_optimum == 0.9
    assert config.sample_count == 8
    assert config.sampler == "achr"
    assert config.seed == 1234
    assert config.output.overwrite is True
    assert config.output.float_precision == 14
    assert config.tolerances.mass_balance == 2e-7

    tracer = config.tracers[0]
    assert tracer.metabolite_id == "source_c"
    assert dict(tracer.isotopomer_fractions) == {"#00": 0.25, "#10": 0.75}
    assert tracer.correction == "yes"
    assert tracer.carbon_count == 2

    target = config.targets[0]
    assert target.fragment_id == "product_m12"
    assert target.metabolite_id == "product_c"
    assert target.atom_positions == (1, 2)
    assert target.analytical_method == "gcms"
    assert target.formula == "C2H4O2"
    assert target.correction == "no"
    assert config.target_by_id[target.fragment_id] is target


def test_invalid_tracer_normalization_fails(tmp_path: Path) -> None:
    invalid = VALID_EXPERIMENT.replace('"#10": 0.75', '"#10": 0.70')

    with pytest.raises(ConfigurationError, match="fractions sum to"):
        load_experiment_config(_write_yaml(tmp_path, invalid))


def test_missing_target_fragment_fails(tmp_path: Path) -> None:
    missing_targets = VALID_EXPERIMENT.replace(
        "targets:\n  - id: product_m12\n"
        "    metabolite: product_c\n"
        "    atoms: [1, 2]\n"
        "    analytical_method: gcms\n"
        "    formula: C2H4O2\n"
        "    correction: no\n",
        "",
    )

    with pytest.raises(ConfigurationError, match="missing required field.*targets"):
        load_experiment_config(_write_yaml(tmp_path, missing_targets))


def test_empty_target_list_fails(tmp_path: Path) -> None:
    empty_targets = VALID_EXPERIMENT.replace(
        "targets:\n  - id: product_m12\n"
        "    metabolite: product_c\n"
        "    atoms: [1, 2]\n"
        "    analytical_method: gcms\n"
        "    formula: C2H4O2\n"
        "    correction: no\n",
        "targets: []\n",
    )

    with pytest.raises(ConfigurationError, match="at least one target fragment"):
        load_experiment_config(_write_yaml(tmp_path, empty_targets))

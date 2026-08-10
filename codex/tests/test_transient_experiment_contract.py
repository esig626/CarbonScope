"""Contract tests for the explicit first-class transient YAML path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fluxemu.compat import project_transient_experiment
from fluxemu.configuration import (
    load_transient_experiment,
    parse_experiment_config,
    parse_transient_experiment_config,
)
from fluxemu.exceptions import ConfigurationError


VALID = {
    "schema_version": 1,
    "experiment_mode": "transient",
    "tracers": [{"metabolite_id": "S", "isotopomer_fractions": {"#0": 0.25, "#1": 0.75}, "correction": "no"}],
    "targets": [{"fragment_id": "A1", "metabolite_id": "A", "atom_positions": [1], "analytical_method": "intermediate", "formula": "C", "correction": "no"}],
    "timecourse": {
        "time_points": [0.0, 0.5, 2.0],
        "initial_internal_mids": "unlabelled",
        "pool_quantities": [{"metabolite_id": "A", "quantity": 2.5}],
        "tolerances": {"rtol": 1e-9, "atol": 1e-12, "mid": 1e-8},
    },
}


def test_parse_and_project_transient_contract() -> None:
    config = parse_transient_experiment_config(VALID)
    assert config.time_points == (0.0, 0.5, 2.0)
    assert config.pool_quantities[0].quantity == 2.5
    assert config.initial_internal_mids == "unlabelled"
    canonical = project_transient_experiment(config)
    assert canonical.time_points == config.time_points
    assert canonical.pool_quantities[0].metabolite_id == "A"
    with pytest.raises(FrozenInstanceError):
        config.time_points = (0.0,)  # type: ignore[misc]


def test_load_transient_experiment(tmp_path: Path) -> None:
    path = tmp_path / "transient.yaml"
    path.write_text("""experiment_mode: transient
tracers: [{metabolite_id: S, isotopomer_fractions: {'#0': 1.0}, correction: no}]
targets: [{fragment_id: A1, metabolite_id: A, atom_positions: [1], analytical_method: intermediate, formula: C, correction: no}]
timecourse:
  time_points: [0, 1]
  initial_internal_mids: unlabelled
  pool_quantities: [{metabolite_id: A, quantity: 1}]
""", encoding="utf-8")
    assert load_transient_experiment(path).experiment_mode == "transient"


@pytest.mark.parametrize("points", [[0, 0], [1, 0], [-1, 0], [0, float("nan")], [float("inf")]])
def test_invalid_time_grid_is_rejected(points: list[float]) -> None:
    value = {**VALID, "timecourse": {**VALID["timecourse"], "time_points": points}}
    with pytest.raises(ConfigurationError):
        parse_transient_experiment_config(value)


@pytest.mark.parametrize("quantity", [0, -1, float("nan"), float("inf")])
def test_invalid_pool_quantity_is_rejected(quantity: float) -> None:
    pools = [{"metabolite_id": "A", "quantity": quantity}]
    value = {**VALID, "timecourse": {**VALID["timecourse"], "pool_quantities": pools}}
    with pytest.raises(ConfigurationError):
        parse_transient_experiment_config(value)


def test_duplicate_pool_records_are_detectable_and_rejected() -> None:
    pool = {"metabolite_id": "A", "quantity": 1.0}
    value = {**VALID, "timecourse": {**VALID["timecourse"], "pool_quantities": [pool, pool]}}
    with pytest.raises(ConfigurationError, match="duplicate"):
        parse_transient_experiment_config(value)


@pytest.mark.parametrize("missing", ["experiment_mode", "initial_internal_mids", "pool_quantities", "time_points"])
def test_explicit_transient_fields_are_required(missing: str) -> None:
    value = {**VALID}
    if missing == "experiment_mode":
        value.pop(missing)
    else:
        value["timecourse"] = {k: v for k, v in VALID["timecourse"].items() if k != missing}
    with pytest.raises(ConfigurationError, match="missing required field"):
        parse_transient_experiment_config(value)


def test_unknown_initial_state_and_mode_are_rejected() -> None:
    value = {**VALID, "timecourse": {**VALID["timecourse"], "initial_internal_mids": "unknown"}}
    with pytest.raises(ConfigurationError, match="unlabelled"):
        parse_transient_experiment_config(value)
    with pytest.raises(ConfigurationError, match="experiment_mode"):
        parse_transient_experiment_config({**VALID, "experiment_mode": "stationary"})


def test_stationary_parser_does_not_infer_or_accept_transient_semantics() -> None:
    with pytest.raises(ConfigurationError, match="unknown field.*experiment_mode"):
        parse_experiment_config(VALID)

"""Strict workflow ingestion must reject declarations native IO could coerce."""
from copy import deepcopy
from pathlib import Path

import pytest

from fluxemu.exceptions import InputValidationError
from fluxemu.workflow._yaml import load_unique_yaml, validate_native_experiment_document


FIXTURE = Path(__file__).parent / "fixtures/native_portability/experiment.yaml"


def test_native_fixture_retains_declared_order_and_is_accepted():
    raw = load_unique_yaml(FIXTURE, "experiment")
    validate_native_experiment_document(raw)
    assert list(raw["isotope_model"]["metabolites"]) == ["feed_X", "pool_Y"]
    assert list(raw["experiment"]["tracers"][0]["isotopomers"]) == ["#000000", "#111111"]


@pytest.mark.parametrize("document", [
    "a: 1\na: 2\n",
    "a: {nested: 1, nested: 2}\n",
    "a: &base {field: 1}\nb: {<<: *base, field: 2}\n",
    "a: [unterminated\n",
    "!!python/object/apply:os.system ['echo invalid']",
])
def test_invalid_yaml_fails_with_input_error(tmp_path, document):
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(InputValidationError, match="cannot load test YAML"):
        load_unique_yaml(path, "test")


def test_missing_yaml_fails_with_input_error(tmp_path):
    with pytest.raises(InputValidationError, match="cannot load experiment YAML"):
        load_unique_yaml(tmp_path / "missing.yaml", "experiment")


@pytest.mark.parametrize("path,value", [
    (("schema_version",), True),
    (("schema_version",), 1.0),
    (("schema_version",), "1"),
    (("isotope_model", "metabolites", "feed_X", "carbon_count"), True),
    (("isotope_model", "metabolites", "feed_X", "carbon_count"), 6.0),
    (("isotope_model", "metabolites", "feed_X", "isotope_visible"), "false"),
    (("isotope_model", "metabolites", "feed_X", "symmetry"), 1),
    (("isotope_model", "assignments", 0, "reaction_id"), 123),
    (("isotope_model", "assignments", 0, "metabolite_map", "glucose"), True),
    (("isotope_model", "required_reactions", 0), 123),
    (("experiment", "tracers", 0, "metabolite_id"), 123),
    (("experiment", "tracers", 0, "isotopomers", "#000000"), "0.5"),
    (("experiment", "tracers", 0, "isotopomers", "#000000"), True),
    (("experiment", "tracers", 0, "isotopomers", "#000000"), float("nan")),
    (("experiment", "tracers", 0, "correction"), False),
    (("experiment", "targets", 0, "atom_positions", 0), True),
    (("experiment", "targets", 0, "atom_positions", 0), "1"),
    (("experiment", "targets", 0, "analytical_method"), 123),
    (("fva_fraction_of_optimum",), True),
    (("fva_fraction_of_optimum",), "1.0"),
    (("fva_fraction_of_optimum",), float("inf")),
    (("experiment", "unknown_field"), []),
    (("unknown_field",), 1),
])
def test_coercible_and_unknown_fields_are_rejected(path, value):
    raw = deepcopy(load_unique_yaml(FIXTURE, "experiment"))
    destination = raw
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with pytest.raises(InputValidationError):
        validate_native_experiment_document(raw)


@pytest.mark.parametrize("invalid", [True, 1.0, "1", 0])
def test_observation_precursor_positions_are_literal_positive_integers(invalid):
    raw = load_unique_yaml(FIXTURE, "experiment")
    raw["experiment"]["observation_targets"] = [{
        "target_id": "fragment", "carbon_count": 1,
        "precursors": [{"metabolite_id": "pool_Y", "atom_positions": [invalid]}],
    }]
    with pytest.raises(InputValidationError, match="positive integer"):
        validate_native_experiment_document(raw)


def test_explicit_string_correction_and_observation_are_accepted():
    raw = load_unique_yaml(FIXTURE, "experiment")
    raw["experiment"]["tracers"][0]["correction"] = "no"
    raw["experiment"]["targets"][0]["correction"] = "no"
    raw["experiment"]["observation_targets"] = [{
        "target_id": "fragment", "carbon_count": 2,
        "precursors": [{"metabolite_id": "pool_Y", "atom_positions": [2, 1]}],
    }]
    validate_native_experiment_document(raw)

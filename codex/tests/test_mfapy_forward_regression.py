from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
MFAPY_SOURCE_ROOT = WORKSPACE / "vendor" / "mfapy"
MFAPY_SAMPLE_ROOT = MFAPY_SOURCE_ROOT / "sample"
FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "official_mfapy_example_0.json"
)

sys.path.insert(0, str(MFAPY_SOURCE_ROOT))
import mfapy  # noqa: E402


@pytest.fixture
def official_example() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_constructor_args(example: dict) -> tuple[dict, dict, dict, dict]:
    dictionaries = example["constructor_dictionaries"]
    return (
        copy.deepcopy(dictionaries["reactions"]),
        copy.deepcopy(dictionaries["reversible_reactions"]),
        copy.deepcopy(dictionaries["metabolites"]),
        copy.deepcopy(dictionaries["target_fragments"]),
    )


def _set_official_tracers(model, example: dict):
    carbon_source = model.generate_carbon_source_template()
    for metabolite_id, tracer in example["tracers"].items():
        setter = getattr(carbon_source, tracer["setter"])
        assert setter(
            metabolite_id,
            copy.deepcopy(tracer["isotopomer_fractions"]),
            correction=tracer["correction"],
        )
    return carbon_source


def _assert_official_forward_result(model, flux_vector, example: dict) -> None:
    carbon_source = _set_official_tracers(model, example)
    mdv_vector, mdv_by_fragment = mfapy.optimize.calc_MDV_from_flux(
        flux_vector,
        example["requested_target_fragments"],
        carbon_source.generate_dict(),
        model.func,
    )

    expected = example["expected_mdvs"]["Glue"]
    tolerance = example["absolute_tolerance"]
    np.testing.assert_allclose(mdv_vector, expected, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(
        mdv_by_fragment["Glue"], expected, rtol=0.0, atol=tolerance
    )
    assert "X_list" in mdv_by_fragment


def test_official_example_parser_route_reproduces_forward_mdv(
    official_example: dict,
) -> None:
    parsed = mfapy.mfapyio.load_metabolic_model(
        official_example["source_model_path"], format="text"
    )
    model = mfapy.metabolicmodel.MetabolicModel(*parsed)
    state = model.load_states(official_example["source_state_path"], format="csv")

    assert model.reaction_ids == official_example["reaction_order"]
    flux_vector = [state["reaction"][rid]["value"] for rid in model.reaction_ids]
    assert flux_vector == official_example["complete_flux_vector"]
    _assert_official_forward_result(model, flux_vector, official_example)


def test_official_parser_dictionaries_equal_direct_fixture(
    official_example: dict,
) -> None:
    parsed = mfapy.mfapyio.load_metabolic_model(
        official_example["source_model_path"], format="text"
    )
    assert parsed == _fixture_constructor_args(official_example)


def test_direct_dictionary_route_does_not_call_text_parser(
    official_example: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_parser(*args, **kwargs):
        raise AssertionError("direct runtime route called load_metabolic_model")

    monkeypatch.setattr(mfapy.mfapyio, "load_metabolic_model", forbidden_parser)
    model = mfapy.metabolicmodel.MetabolicModel(
        *_fixture_constructor_args(official_example)
    )

    assert model.reaction_ids == official_example["reaction_order"]
    _assert_official_forward_result(
        model, official_example["complete_flux_vector"], official_example
    )


def test_forward_only_mfapy_imports_without_nlopt() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(MFAPY_SOURCE_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins\n"
                "real_import = builtins.__import__\n"
                "def without_nlopt(name, *args, **kwargs):\n"
                "    if name == 'nlopt' or name.startswith('nlopt.'):\n"
                "        raise ModuleNotFoundError("
                "\"No module named 'nlopt'\", name='nlopt')\n"
                "    return real_import(name, *args, **kwargs)\n"
                "builtins.__import__ = without_nlopt\n"
                "import mfapy\n"
                "assert mfapy.optimize.nlopt is None\n"
                "assert callable(mfapy.optimize.calc_MDV_from_flux)\n"
            ),
        ],
        cwd=WORKSPACE,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nlopt_fitting_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mfapy.optimize, "nlopt", None)
    with pytest.raises(ImportError, match="optional 'nlopt' package"):
        mfapy.optimize.fit_r_mdv_nlopt({}, {}, {}, {}, None, {}, [])

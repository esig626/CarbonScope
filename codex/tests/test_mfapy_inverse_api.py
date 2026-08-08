"""Non-fitting checks for reusing FluxEMU's in-memory mfapy model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from cobra.io import read_sbml_model

from fluxemu._mfapy import load_mfapy
from fluxemu.configuration import load_experiment
from fluxemu.forward import run_batch_forward


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_builder(name: str, directory: Path):
    module_name = f"_test_mfapy_inverse_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, directory / "build_model.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_same_table5_forward_object_accepts_inverse_measurements_in_memory() -> None:
    builder = _load_builder("table5", EXAMPLES / "antoniewicz_tca")
    cobra_model = read_sbml_model(str(builder.MODEL_PATH))
    experiment = load_experiment(builder.STATIONARY_EXPERIMENT)
    bundle = builder.build_table5_mfapy_bundle(cobra_model, experiment)
    shared_model = bundle.model

    forward = run_batch_forward(
        bundle, builder._fixed_flux_frame(cobra_model), experiment
    )
    synthetic_mid = forward.predictions["figure12"]["glutamate"]

    mfapy = load_mfapy()
    observed = mfapy.mdv.MdvData(shared_model.target_fragments)
    for mass, ratio in enumerate(synthetic_mid):
        assert observed.set_data("glutamate", mass, ratio, 0.01, "use")

    carbon_source = builder._carbon_source(bundle, experiment)
    assert shared_model.set_experiment("control0", observed, carbon_source)
    assert bundle.model is shared_model
    assert shared_model.experiments["control0"]["number_of_measurement"] == 5
    for method in (
        "set_constraint",
        "set_boundary",
        "generate_initial_states",
        "fitting_flux",
        "goodness_of_fit",
        "search_ci",
        "posterior_distribution",
    ):
        assert callable(getattr(shared_model, method))


@pytest.mark.parametrize(
    ("name", "directory", "build_bundle", "truth_frame", "balanced", "expected_dof"),
    (
        (
            "table5_blocker",
            "antoniewicz_tca",
            "build_table5_mfapy_bundle",
            "_fixed_flux_frame",
            ("OAC", "citrate", "AKG", "succinate", "fumarate"),
            3,
        ),
        (
            "glucose_blocker",
            "antoniewicz_tca_glucose",
            "build_glucose_tca_mfapy_bundle",
            "fixed_flux_frame",
            None,
            4,
        ),
    ),
)
def test_terminal_glutamate_boundary_preserves_the_intended_steady_state_dof(
    name: str,
    directory: str,
    build_bundle: str,
    truth_frame: str,
    balanced: tuple[str, ...] | None,
    expected_dof: int,
) -> None:
    builder = _load_builder(name, EXAMPLES / directory)
    cobra_model = read_sbml_model(str(builder.MODEL_PATH))
    experiment = load_experiment(builder.STATIONARY_EXPERIMENT)
    bundle = getattr(builder, build_bundle)(cobra_model, experiment)
    truth = getattr(builder, truth_frame)(cobra_model).iloc[0]

    if balanced is None:
        balanced = tuple(builder.BALANCED_CARBON_METABOLITES)
    stoichiometry = np.asarray(
        [
            [
                float(
                    reaction.metabolites.get(
                        cobra_model.metabolites.get_by_id(metabolite_id), 0.0
                    )
                )
                for reaction in cobra_model.reactions
            ]
            for metabolite_id in balanced
        ]
    )
    assert len(cobra_model.reactions) - np.linalg.matrix_rank(stoichiometry) == expected_dof

    glutamate = cobra_model.metabolites.get_by_id("glutamate")
    truth_residual = sum(
        float(reaction.metabolites.get(glutamate, 0.0)) * float(truth[reaction.id])
        for reaction in cobra_model.reactions
    )
    internal_glutamate = bundle.metabolite_to_internal["glutamate"]

    assert truth_residual == pytest.approx(50.0)
    assert bundle.model.metabolites[internal_glutamate]["excreted"] == "excreted"
    assert internal_glutamate not in bundle.model.matrix_row_names
    assert bundle.model.numbers["independent_number"] == expected_dof

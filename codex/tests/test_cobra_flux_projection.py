"""Matrix-level acceptance tests for faithful COBRA LP projection."""

from __future__ import annotations

from pathlib import Path

import pytest
from cobra.io import read_sbml_model
from cobra.util.solver import linear_reaction_coefficients

from fluxemu.compat import project_cobra_flux_model
from fluxemu.compat.cobra import _project_cobra_flux_model
from fluxemu.flux_analysis import compile_flux_lp


ROOT = Path(__file__).parents[1]
MODELS = (
    ROOT / "examples/toy_model.xml",
    ROOT / "examples/antoniewicz_tca/antoniewicz_tca.xml",
    ROOT / "examples/antoniewicz_tca_glucose/antoniewicz_tca_glucose.xml",
)


@pytest.mark.parametrize("path", MODELS, ids=lambda path: path.stem)
def test_flux_projection_reproduces_cobra_lp_matrix_and_vectors(path):
    cobra_model = read_sbml_model(path)
    projected = project_cobra_flux_model(cobra_model)
    compiled = compile_flux_lp(projected)

    reaction_ids = tuple(reaction.id for reaction in cobra_model.reactions)
    metabolite_ids = tuple(metabolite.id for metabolite in cobra_model.metabolites)
    assert compiled.reaction_ids == reaction_ids
    assert tuple(item.metabolite_id for item in projected.metabolites) == metabolite_ids
    assert compiled.balanced_metabolite_ids == metabolite_ids
    assert len(compiled.balanced_metabolite_ids) == len(cobra_model.metabolites)

    actual = {}
    for row, metabolite_id in enumerate(compiled.balanced_metabolite_ids):
        for offset in range(compiled.row_starts[row], compiled.row_starts[row + 1]):
            actual[(metabolite_id, compiled.reaction_ids[compiled.column_indices[offset]])] = compiled.coefficients[offset]
    expected = {
        (metabolite.id, reaction.id): float(coefficient)
        for reaction in cobra_model.reactions
        for metabolite, coefficient in reaction.metabolites.items()
        if float(coefficient) != 0.0
    }
    assert actual == expected
    assert compiled.lower_bounds == tuple(float(reaction.lower_bound) for reaction in cobra_model.reactions)
    assert compiled.upper_bounds == tuple(float(reaction.upper_bound) for reaction in cobra_model.reactions)
    coefficients = linear_reaction_coefficients(cobra_model)
    assert compiled.objective_coefficients == tuple(float(coefficients.get(reaction, 0.0)) for reaction in cobra_model.reactions)
    assert compiled.objective_direction == cobra_model.objective.direction


def test_isotope_annotations_do_not_remove_flux_rows_but_forward_policy_is_preserved():
    cobra_model = read_sbml_model(ROOT / "examples/toy_model.xml")
    flux_only = project_cobra_flux_model(cobra_model)
    isotope_forward_flux = _project_cobra_flux_model(cobra_model, isotope_boundary_roles=True)

    assert all(item.steady_state_balanced for item in flux_only.metabolites)
    assert tuple(item.steady_state_balanced for item in isotope_forward_flux.metabolites) != tuple(
        item.steady_state_balanced for item in flux_only.metabolites
    )

"""Regression tests for the synthetic glucose-to-frozen-TCA benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from cobra.io import read_sbml_model


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "antoniewicz_tca_glucose"


def _load_builder():
    name = "_test_antoniewicz_tca_glucose_builder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / "build_model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_builder()


@pytest.fixture(scope="module")
def model():
    result = read_sbml_model(str(benchmark.MODEL_PATH))
    benchmark.validate_glucose_tca_model(result)
    return result


@pytest.fixture(scope="module")
def stationary():
    return benchmark.run_stationary()


@pytest.fixture(scope="module")
def timecourse():
    return benchmark.run_timecourse()


def _assert_probability(mid: np.ndarray, *, atol: float = 1.0e-10) -> None:
    assert np.isfinite(mid).all()
    assert np.all(mid >= -atol)
    assert mid.sum() == pytest.approx(1.0, abs=atol)


def test_frozen_tca_definitions_are_imported_without_reinterpretation(model) -> None:
    assert tuple(benchmark.TCA_REACTIONS) == tuple(benchmark._FROZEN_TCA.TABLE5_REACTIONS)
    metadata = benchmark.collect_isotope_metadata(model)
    for reaction in benchmark.TCA_REACTIONS:
        observed = (
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in metadata.reactions[reaction.reaction_id].substrates),
            tuple((part.metabolite_id, "".join(part.atom_labels)) for part in metadata.reactions[reaction.reaction_id].products),
        )
        assert observed == (reaction.substrates, reaction.products)


def test_fixed_complete_flux_vector_is_balanced_for_every_balanced_carbon_pool(model) -> None:
    frame = benchmark.fixed_flux_frame(model)
    assert frame.loc["glucose_to_tca_fixed", "GLC_IN"] == 50.0
    assert frame.loc["glucose_to_tca_fixed", "FBA"] == 50.0
    assert frame.loc["glucose_to_tca_fixed", "GAPD"] == 100.0
    assert frame.loc["glucose_to_tca_fixed", "PDH"] == 100.0
    assert frame.loc["glucose_to_tca_fixed", "v1"] == 100.0
    assert frame.loc["glucose_to_tca_fixed", "LDH"] == 0.0
    residuals = benchmark.carbon_mass_balance(model)
    assert tuple(residuals.index) == benchmark.BALANCED_CARBON_METABOLITES
    assert np.allclose(residuals.to_numpy(), 0.0, rtol=0.0, atol=1.0e-12)


def test_stationary_glycolysis_mids_are_deterministic_and_tca_agrees_independently(stationary) -> None:
    _, predictions, _ = stationary
    for metabolite in ("glucose_c", "G6P", "F6P", "FBP"):
        _assert_probability(predictions[metabolite])
        assert predictions[metabolite][-1] == pytest.approx(1.0, abs=1.0e-12)
        assert np.allclose(predictions[metabolite][:-1], 0.0, atol=1.0e-12)
    for metabolite in ("DHAP", "GAP", "BPG", "3PG", "2PG", "PEP", "pyruvate"):
        _assert_probability(predictions[metabolite])
        assert predictions[metabolite][3] == pytest.approx(1.0, abs=1.0e-12)
        assert np.allclose(predictions[metabolite][:3], 0.0, atol=1.0e-12)
    _assert_probability(predictions["AcCoA"])
    assert predictions["AcCoA"][2] == pytest.approx(1.0, abs=1.0e-12)
    assert np.allclose(predictions["AcCoA"][:2], 0.0, atol=1.0e-12)

    independent = benchmark.independent_tca_mids()
    for metabolite in benchmark.INDEPENDENT_TCA_TARGETS:
        _assert_probability(predictions[metabolite])
        _assert_probability(independent[metabolite])
        assert np.allclose(predictions[metabolite], independent[metabolite], rtol=0.0, atol=1.0e-12)


def test_positional_glucose_and_pdh_relationships_are_map_level_checks() -> None:
    patterns = benchmark.upstream_pyruvate_origin_patterns()
    assert patterns == ((3, 2, 1), (4, 5, 6))
    assert all(pattern[2] in {1, 6} for pattern in patterns)
    assert all(pattern[1] in {2, 5} for pattern in patterns)
    assert all(pattern[0] in {3, 4} for pattern in patterns)
    assert benchmark.pdh_positional_destinations() == ((2, 3), (1,))


def test_deliberately_broken_fba_and_pdh_maps_fail_the_positional_checks() -> None:
    broken_fba = benchmark.GlycolysisReaction(
        "FBA", (("FBP", "abcdef"),), (("DHAP", "abc"), ("GAP", "def")), "abcdef -> abc + def"
    )
    broken_glycolysis = tuple(
        broken_fba if reaction.reaction_id == "FBA" else reaction
        for reaction in benchmark.GLYCOLYSIS_REACTIONS
    )
    assert benchmark.upstream_pyruvate_origin_patterns(broken_glycolysis) != ((3, 2, 1), (4, 5, 6))

    broken_pdh = benchmark.GlycolysisReaction(
        "PDH", (("pyruvate", "abc"),), (("AcCoA", "ab"), ("CO2", "c")), "abc -> ab + c"
    )
    assert benchmark.pdh_positional_destinations(broken_pdh) != ((2, 3), (1,))


def test_removing_succinate_fumarate_reverse_orientation_breaks_independent_agreement() -> None:
    _, unsymmetrical, _ = benchmark.run_stationary(symmetric=False)
    correct = benchmark.independent_tca_mids()
    unsymmetrical_direct = benchmark.independent_tca_mids(symmetric=False)
    for metabolite in benchmark.INDEPENDENT_TCA_TARGETS:
        assert np.allclose(unsymmetrical[metabolite], unsymmetrical_direct[metabolite], rtol=0.0, atol=1.0e-12)
    assert not np.allclose(unsymmetrical["citrate"], correct["citrate"], rtol=0.0, atol=1.0e-12)


def test_independent_positional_state_establishes_recirculation_interpretation() -> None:
    summary = benchmark.recirculation_summary()
    assert summary["citrate_m2_with_unlabelled_oac"] > 0.0
    assert summary["citrate_m3_or_higher_with_returned_oac"] > 0.0
    assert summary["citrate_m4_or_higher_with_two_or_more_returned_oac_carbons"] > 0.0
    assert summary["glutamate_m3_or_higher"] > 0.0


def test_diffmdv_timecourse_is_valid_shows_recirculation_and_converges(timecourse, stationary) -> None:
    frame, timepoints, convergence_error = timecourse
    _, stationary_predictions, _ = stationary
    assert tuple(timepoints[: len(benchmark.DIAGNOSTIC_TIMECOURSE_TIMEPOINTS)]) == benchmark.DIAGNOSTIC_TIMECOURSE_TIMEPOINTS
    assert timepoints[-1] > benchmark.BASE_TIMEPOINTS[-1]
    assert convergence_error <= benchmark.DYNAMIC_CONVERGENCE_TOLERANCE
    assert np.isfinite(frame["fraction"].to_numpy()).all()
    assert (frame["fraction"] >= 0.0).all()
    grouped = frame.groupby(["time", "metabolite"])["fraction"].sum()
    assert np.allclose(grouped.to_numpy(), 1.0, rtol=0.0, atol=1.0e-12)

    zero = frame[frame["time"] == 0.0]
    assert (zero[zero["mass_isotopologue"] == "M+0"]["fraction"] == 1.0).all()
    assert (zero[zero["mass_isotopologue"] != "M+0"]["fraction"] == 0.0).all()

    introduced = frame[frame["time"] > 0.0]
    assert introduced[(introduced["metabolite"] == "pyruvate") & (introduced["mass_isotopologue"] == "M+3")]["fraction"].max() > 0.0
    assert introduced[(introduced["metabolite"] == "AcCoA") & (introduced["mass_isotopologue"] == "M+2")]["fraction"].max() > 0.0
    citrate_m2 = introduced[(introduced["metabolite"] == "citrate") & (introduced["mass_isotopologue"] == "M+2")]
    first_useful = float(citrate_m2[citrate_m2["fraction"] > 1.0e-10]["time"].min())
    assert _mid_at(frame, first_useful, "citrate")[2] > _mid_at(frame, first_useful, "citrate")[4]
    later_citrate_m4 = introduced[(introduced["metabolite"] == "citrate") & (introduced["mass_isotopologue"] == "M+4")]
    assert later_citrate_m4[later_citrate_m4["time"] > first_useful]["fraction"].max() > 0.0
    late_glutamate = _mid_at(frame, 10.0, "glutamate")
    assert late_glutamate[3:].sum() > 0.0

    final_time = timepoints[-1]
    for metabolite in benchmark.STATIONARY_TARGETS:
        assert np.allclose(
            _mid_at(frame, final_time, metabolite),
            stationary_predictions[metabolite],
            rtol=0.0,
            atol=benchmark.DYNAMIC_CONVERGENCE_TOLERANCE,
        )


def test_run_enrichment_timecourse_sweep_uses_all_configured_enrichments() -> None:
    observed: list[float] = []

    def fake_runner(
        experiment,
        points,
        run_max_time,
    ):
        tracer = experiment.tracers[0]
        frac = float(tracer.isotopomer_fractions["#111111"])
        observed.append(frac)
        timepoints = tuple(points)
        frame = pd.DataFrame(
            {
                "time": [timepoints[0], timepoints[-1]],
                "metabolite": ["pyruvate", "pyruvate"],
                "mass_isotopologue": ["M+0", "M+3"],
                "fraction": [1.0, 0.0],
            }
        )
        return frame, timepoints, 0.0

    results = benchmark.run_enrichment_timecourse_sweep(
        timepoints=benchmark.DIAGNOSTIC_TIMECOURSE_TIMEPOINTS,
        runner=fake_runner,
    )
    assert set(results.keys()) == {label for label, _ in benchmark.DIAGNOSTIC_GLUCOSE_ENRICHMENT_SETTINGS}
    assert sorted(observed) == sorted(fraction for _, fraction in benchmark.DIAGNOSTIC_GLUCOSE_ENRICHMENT_SETTINGS)
    assert results and all(result[1] == benchmark.DIAGNOSTIC_TIMECOURSE_TIMEPOINTS for result in results.values())


def _mid_at(frame: pd.DataFrame, timepoint: float, metabolite: str) -> np.ndarray:
    return frame[(frame["time"] == timepoint) & (frame["metabolite"] == metabolite)].sort_values("mass_isotopologue")["fraction"].to_numpy(dtype=float)


def test_checked_csv_outputs_match_generated_stationary_and_independent_results(stationary) -> None:
    _, stationary_predictions, _ = stationary
    independent = benchmark.independent_tca_mids()
    stationary_csv = pd.read_csv(EXAMPLE_DIR / "stationary_mids.csv")
    independent_csv = pd.read_csv(EXAMPLE_DIR / "independent_tca_mids.csv")
    for metabolite in benchmark.STATIONARY_TARGETS:
        observed = stationary_csv[stationary_csv["metabolite"] == metabolite]["fluxemu_mfapy"].to_numpy()
        assert np.allclose(observed, stationary_predictions[metabolite], rtol=0.0, atol=1.0e-12)
    for metabolite in benchmark.INDEPENDENT_TCA_TARGETS:
        observed = independent_csv[independent_csv["metabolite"] == metabolite]["independent_isotopomer"].to_numpy()
        assert np.allclose(observed, independent[metabolite], rtol=0.0, atol=1.0e-12)

"""Deterministic JSON provenance and restrained human interpretation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
from tempfile import NamedTemporaryFile

from fluxemu import __version__
from fluxemu.exceptions import ConfigurationError, ValidationError
from fluxemu.model import deterministic_serialise, model_fingerprint
from fluxemu.testing.composite import (
    COMPOSITE_LP_CERTIFICATION_TOLERANCE, COMPOSITE_LP_SMALL_MATRIX_VALUE,
    COMPOSITE_NUMERICAL_TOLERANCE, MIN_EXACT_COMPOSITE_EPSILON,
)

from .runner import (
    RELATIONSHIP_TOLERANCE, HypothesisTestingWorkflowResult,
    WorkflowOutputPaths, WorkflowProvenance, WorkflowSoftwareProvenance,
)


SCOPE = (
    "Results apply only to the ordered represented finite H0/P0 and H1/P1 law families.",
    "Sample frequency is not a prior; finite hit-and-run chains have no asserted mixing or independence guarantee.",
    "Continuous feasible flux-family testing and continuous Renyi-order optimization remain unresolved here.",
    "Genuine fixed-total counts are the only supported observation semantics; multiple blocks require declared independence.",
    "A converse constrains achievable error; it does not prove achievability.",
    "A candidate score is not a finite-sample least-favourable pair or an asserted joint convex-class projection.",
    "Achieved score errors need not equal unrestricted represented finite minimax.",
    "Exact minimax is a numerically checked small-problem LP and may explicitly refuse.",
    "This is testing-performance analysis, not a realised-data composite p-value, compatibility set, or identification of biological truth.",
)


def _json_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValidationError("workflow report cannot contain NaN")
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValidationError(f"unsupported workflow report value: {type(value).__name__}")


def _software_provenance() -> WorkflowSoftwareProvenance:
    package = Path(__file__).resolve().parents[1]
    digest = sha256()
    # Identify executable source and packaged authoritative scientific data,
    # independently of its install path or whether a Git checkout is present.
    for path in sorted(p for p in package.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".json"}):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    versions = []
    for name in ("numpy", "scipy", "highspy", "python-libsbml", "PyYAML", "pandas"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        versions.append((name, version))
    commit = None
    dirty = None
    checkout = package.parent.parent
    if (checkout / ".git").exists() and (checkout / "pyproject.toml").exists():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal", "--", "src/fluxemu", "pyproject.toml"], cwd=checkout,
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            commit, dirty = None, None
    return WorkflowSoftwareProvenance(__version__, platform.python_version(), tuple(versions), commit, dirty, digest.hexdigest())


def build_workflow_provenance(specification, null, alternative, stationary) -> WorkflowProvenance:
    return WorkflowProvenance(
        model_fingerprint(specification.model),
        stationary.null_observation_laws.experiment_fingerprints,
        specification.fingerprint,
        null.hypothesis_fingerprint, alternative.hypothesis_fingerprint,
        null.fingerprint, alternative.fingerprint,
        stationary.problem.null.fingerprint, stationary.problem.alternative.fingerprint,
        specification.observation.fingerprint, stationary.problem.fingerprint,
        stationary.fingerprint, _software_provenance(),
    )


def _hypothesis_report(family, role, member_ids):
    return {
        "role": role,
        "description": family.hypothesis.description,
        "declared_constraints": _json_value(family.hypothesis.reaction_bounds),
        "constraint_semantics": "intersection with common physical reaction bounds",
        "region_policy": "complete bound-constrained steady-state region; no retained objective fraction",
        "effective_reaction_bounds": [
            {"reaction_id": reaction.reaction_id, "lower_bound": reaction.lower_bound, "upper_bound": reaction.upper_bound}
            for reaction in family.model.reactions
        ],
        "distinct_region_witnesses": _json_value(family.region_witnesses),
        "hypothesis_fingerprint": family.hypothesis_fingerprint,
        "finite_family_fingerprint": family.fingerprint,
        "state_generation": _json_value(family.hypothesis.state_generation),
        "requested_family_size": family.hypothesis.state_generation.count,
        "actual_family_size": len(family.states),
        "sampling_provenance": _json_value(family.sampling.provenance),
        "feasibility_validation": family.sampling.validation.to_dict(),
        "states": [
            {"member_id": member_id, "sample_id": state.sample_id,
             "state_fingerprint": sha256(deterministic_serialise(state).encode()).hexdigest(),
             "ordered_fluxes": _json_value(state.values)}
            for member_id, state in zip(member_ids, family.states, strict=True)
        ],
    }


def _result_value(item):
    value = item.value
    if value is None:
        return None
    result = {
        field.name: _json_value(getattr(value, field.name))
        for field in fields(value) if field.name not in {"problem", "candidate", "bound"}
    }
    result["record_type"] = type(value).__name__
    if item.procedure in {"composite_converse", "candidate_score"}:
        result["null_member_id"] = value.null_member_id
        result["alternative_member_id"] = value.alternative_member_id
    if item.procedure == "composite_converse":
        result["quantity"] = "order_specific_composite_type_ii_lower_bound"
        result["global_order_envelope_evaluated"] = False
    elif item.procedure == "exact_minimax":
        result["quantity"] = "unrestricted_represented_finite_minimax_type_ii"
        result["active_null_member_ids"] = list(value.active_null_member_ids)
        result["active_alternative_member_ids"] = list(value.active_alternative_member_ids)
        result["randomised"] = value.randomised
    elif item.procedure == "candidate_score":
        result["quantity"] = "finite_family_renyi_vertex_pair_candidate"
        result["finite_n_least_favourable_claimed"] = False
        result["joint_convex_projection_claimed"] = False
    elif item.procedure == "analytical_score_bound":
        result["quantity"] = "verified_projected_score_analytical_bound"
        result["candidate_order"] = value.candidate.order
    else:
        result["quantity"] = ("deterministic_projected_score_achieved_error" if item.procedure == "deterministic_score_error"
                              else "calibrated_score_family_achieved_error")
        if item.procedure == "deterministic_score_error":
            result["threshold"] = _json_value(value.bound.threshold)
        else:
            result["exhausts_type_i_budget"] = value.exhausts_type_i_budget
        worst_null = max(value.null_type_i_errors)
        worst_alternative = max(value.alternative_type_ii_errors)
        result["worst_null_member_indices"] = [i for i, error in enumerate(value.null_type_i_errors) if error == worst_null]
        result["worst_alternative_member_indices"] = [i for i, error in enumerate(value.alternative_type_ii_errors) if error == worst_alternative]
    return result


def workflow_report(result: HypothesisTestingWorkflowResult) -> dict:
    """Export a complete deterministic report; nonfinite extended reals are strings."""
    from ._validation import validate_workflow_result
    validate_workflow_result(result)
    specification = result.specification
    design = result.problem.null.members[0]
    observations = {
        "semantics": "genuine_counts",
        "independent_blocks": specification.independent_blocks,
        "block_order": [
            {"experiment_id": identity[0], "target_id": identity[1], "replicate_id": identity[2],
             "total_count": block.n, "mass_classes": list(block.mass_classes)}
            for identity, block in zip(design.block_identities, design.blocks, strict=True)
        ],
        "families": [],
    }
    for role, source, family in (
        ("H0", result.stationary.null_observation_laws, result.problem.null),
        ("H1", result.stationary.alternative_observation_laws, result.problem.alternative),
    ):
        observations["families"].append({
            "role": role, "family_fingerprint": family.fingerprint,
            "members": [
                {"member_id": identifier, "law_fingerprint": member.fingerprint,
                 "blocks": [{"law_fingerprint": block.fingerprint, "total_count": block.n,
                             "probabilities": list(block.probabilities), "mass_classes": list(block.mass_classes)}
                            for block in member.blocks]}
                for identifier, member in zip(family.member_ids, family.members, strict=True)
            ],
            "ordered_component_fingerprints": [component.fingerprint for component in source.components],
            "common_model_state_validation": source.validation.to_dict(),
        })
    return _json_value({
        "schema_version": 1,
        "workflow": "fluxemu-native-stationary-hypothesis-testing",
        "status": "completed_with_refusals" if result.refusals else "completed",
        "identity": result.provenance,
        "scientific_specification": {
            "common_model": specification.model,
            "ordered_experiments": specification.observation.experiments,
            "H0": specification.null,
            "H1": specification.alternative,
            "independent_blocks": specification.independent_blocks,
            "observation_semantics": "genuine_counts",
            "region_policy": "full-constrained-region",
            "testing": specification.testing,
        },
        "software_git_dirty_scope": "runtime source and pyproject.toml only; generated reports are excluded",
        "input_sources": [
            {field.name: getattr(source, field.name) for field in fields(source) if field.name != "path"}
            for source in specification.input_sources
        ],
        "hypotheses": [
            _hypothesis_report(result.null_family, "H0", result.problem.null.member_ids),
            _hypothesis_report(result.alternative_family, "H1", result.problem.alternative.member_ids),
        ],
        "observations": observations,
        "testing": {
            "roles": {"H0": "P0/null", "H1": "P1/alternative"},
            "type_i": "max over H0 of P0(decide H1)",
            "type_ii": "max over H1 of P1(decide H0)",
            "specification": specification.testing,
            "numerical_policy": {
                "exact_minimum_epsilon": MIN_EXACT_COMPOSITE_EPSILON,
                "lp_small_matrix_value": COMPOSITE_LP_SMALL_MATRIX_VALUE,
                "lp_certification_tolerance": COMPOSITE_LP_CERTIFICATION_TOLERANCE,
                "composite_numerical_tolerance": COMPOSITE_NUMERICAL_TOLERANCE,
                "relationship_comparison_tolerance": RELATIONSHIP_TOLERANCE,
                "policy": "unchanged repaired finite primitives; full enumeration or explicit refusal; no clipping, support dropping, or fallback",
            },
        },
        "results": [
            {"procedure": item.procedure, "order": item.order, "requested": item.requested,
             "status": item.status, "result": _result_value(item), "refusal": item.refusal}
            for item in result.testing_results
        ],
        "relationship_checks": result.relationship_checks,
        "refusals": result.refusals,
        "scope": SCOPE,
        "extended_real_encoding": "Infinity and -Infinity are strings; NaN is invalid",
    })


def workflow_summary(result: HypothesisTestingWorkflowResult) -> str:
    lines = ["CarbonScope represented finite hypothesis comparison"]
    for role, family in (("H0/P0 (null)", result.null_family), ("H1/P1 (alternative)", result.alternative_family)):
        constraints = "; ".join(
            f"{bound.reaction_id}: lower={bound.lower_bound}, upper={bound.upper_bound}"
            for bound in family.hypothesis.reaction_bounds
        ) or "common physical bounds"
        lines.append(f"{role}: {family.hypothesis.description}; {constraints}; {len(family.states)} represented states.")
    design = result.problem.null.members[0]
    lines.append(f"Observation design: {len(design.blocks)} genuine-count blocks; independent_blocks={result.specification.independent_blocks}.")
    for identity, block in zip(design.block_identities, design.blocks, strict=True):
        lines.append(f"  {' / '.join(identity)}: total_count={block.n}, mass classes={block.mass_classes}.")
    lines.append(f"Worst-case Type-I budget: {result.specification.testing.epsilon:g}.")
    for item in result.testing_results:
        if item.status == "not_requested":
            continue
        label = item.procedure + (f" (order {item.order:g})" if item.order is not None else "")
        if not item.requested:
            label += " [prerequisite]"
        if item.status == "refused":
            lines.append(f"REFUSED {label}: {item.refusal.reason}")
            continue
        value = item.value
        if item.procedure == "composite_converse":
            detail = f"Type-II lower bound={value.type_ii_lower_bound:.12g}"
        elif item.procedure == "exact_minimax":
            detail = f"represented minimax Type-II={value.minimax_type_ii_error:.12g}, worst Type-I={value.worst_type_i_error:.12g}"
        elif item.procedure == "candidate_score":
            detail = f"candidate Renyi={value.renyi:.12g}, uniform moments verified={value.uniform_moment_bounds_verified}"
            if value.verification_failures:
                detail += "; " + "; ".join(value.verification_failures)
        elif item.procedure == "analytical_score_bound":
            detail = f"projected exponential bound={value.raw_exponential_upper_bound:.12g}, separate minimax upper bound={value.minimax_type_ii_upper_bound:.12g}"
        else:
            detail = f"achieved worst Type-II={value.worst_type_ii_error:.12g}, worst Type-I={value.worst_type_i_error:.12g}"
        lines.append(f"Evaluated {label}: {detail}.")
    if not result.refusals:
        lines.append("No requested statistical procedure was refused.")
    lines.append("The evaluated quantities establish the stated performance bounds/errors for these represented finite laws only.")
    lines.extend(SCOPE)
    return "\n".join(lines) + "\n"


def persist_workflow_report(result: HypothesisTestingWorkflowResult, output_directory: str | Path) -> WorkflowOutputPaths:
    """Serialize fully before touching the destination; replace complete files."""
    report_text = json.dumps(workflow_report(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary = workflow_summary(result)
    directory = Path(output_directory).resolve()
    paths = WorkflowOutputPaths(directory / "report.json", directory / "summary.txt")
    # A report destination must never overwrite a source scientific input.
    sources = {source.path.resolve() for source in result.specification.input_sources}
    if any(path.resolve() in sources for path in (paths.report, paths.summary)):
        raise ConfigurationError("workflow output would overwrite a scientific input file")
    if any(path.is_dir() for path in (paths.report, paths.summary)):
        raise ConfigurationError("workflow output file path is an existing directory")
    temporary_paths = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for path, content in ((paths.report, report_text), (paths.summary, summary)):
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".workflow-", delete=False) as stream:
                temporary = Path(stream.name)
                temporary_paths.append(temporary)
                stream.write(content)
        for temporary, path in zip(temporary_paths, (paths.report, paths.summary), strict=True):
            temporary.replace(path)
    except OSError as error:
        raise ConfigurationError(f"cannot persist workflow report: {error}") from error
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return paths

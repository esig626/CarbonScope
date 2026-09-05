"""Diagnostic scaling benchmark for the VFFVA-to-HiGHS shared-memory port.

This benchmark deliberately keeps three questions separate:

* cold-reference versus reusable-engine performance;
* shared-memory scaling relative to one reusable worker; and
* direct execution of original VFFVA, which is recorded separately in the
  checked benchmark evidence and is never inferred from the first two.

Numerical parity is a prerequisite for every timing.  Timing ratios are
diagnostic evidence, not CI acceptance thresholds.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
import time
from typing import Any

import highspy
import numpy as np
import pandas as pd

from fluxemu.flux_analysis import (
    compile_flux_lp,
    run_highs_fba,
    run_highs_fva_reference,
    run_highs_vffva,
)
from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis.results import FVAResult
from fluxemu.model import FluxModel, load_sbml_flux_model
from fluxemu.real_model import load_ecoli_core_flux_model


CORRECTNESS_TOLERANCE = 1e-7
SCHEDULING_MODE = "dynamic"
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ILJ478_COMPRESSED_SHA256 = (
    "736f74b425172fb2ed4161ef3c0f9dfeebe352e64499501da320935c3ac27c7d"
)
IJO1366_COMPRESSED_SHA256 = (
    "e100c6a9fdc30f6b880d390f8af9941422202b8714c7786629f19c98b076d208"
)
SOURCE_REMOTE_COMMIT = "b672e3551e707e5dd6e7f003f60b04cfc040fbdd"
SOURCE_HIGHS_SHA256 = (
    "92060590c6b366b6fa849640ad4a98cea47460a487fbd4b83fefb2cb17e0303c"
)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def _affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return sorted(int(cpu) for cpu in getter(0))
    except OSError:
        return None


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "highspy_module": getattr(highspy, "__version__", None),
        "highspy_distribution": _package_version("highspy"),
        "highs": highspy.Highs().version(),
        "python_libsbml_distribution": _package_version("python-libsbml"),
        "scipy_distribution": _package_version("scipy"),
        "threadpoolctl_distribution": _package_version("threadpoolctl"),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "allowed_cpu_affinity": _affinity(),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES
        },
    }


def _source_identity() -> dict[str, object]:
    module_path = Path(run_highs_vffva.__code__.co_filename).resolve()
    actual_sha256 = _sha256(module_path)
    if actual_sha256 != SOURCE_HIGHS_SHA256:
        raise RuntimeError(
            "imported FluxEMU HiGHS implementation is not the authoritative "
            f"{SOURCE_REMOTE_COMMIT} benchmark source: {module_path} has "
            f"sha256={actual_sha256}"
        )
    return {
        "remote_repository": "esig626/fluxemu-standalone",
        "remote_commit": SOURCE_REMOTE_COMMIT,
        "remote_highs_py_git_blob": "8ec6c3ae8629cec00c26358bc71c4691ae839335",
        "imported_module": str(module_path),
        "imported_module_sha256": actual_sha256,
        "expected_module_sha256": SOURCE_HIGHS_SHA256,
        "verified": True,
    }


def _input_acquisition_evidence() -> dict[str, object]:
    return {
        "policy": (
            "All non-bundled models and pinned upstream source remain temporary "
            "inputs outside the tracked repository."
        ),
        "portable_environment_variables": {
            "REPO": "checkout root for esig626/fluxemu-standalone",
            "BENCH_TMP": "caller-created temporary benchmark directory",
            "UPSTREAM_VFFVA": "temporary checkout of marouenbg/VFFVA",
        },
        "official_ilj478": {
            "download_command": (
                "curl --fail --location "
                "https://bigg.ucsd.edu/static/models/iLJ478.xml.gz "
                "--output \"$BENCH_TMP/iLJ478.xml.gz\""
            ),
            "checksum_command": (
                "printf '%s  %s\\n' "
                "736f74b425172fb2ed4161ef3c0f9dfeebe352e64499501da320935c3ac27c7d "
                "\"$BENCH_TMP/iLJ478.xml.gz\" | sha256sum --check -"
            ),
            "compressed_sha256": ILJ478_COMPRESSED_SHA256,
            "compressed_bytes": 168459,
            "uncompressed_sha256": (
                "b9e0b8ad96cc4a764785e0dab41cac1b4436935cdc19b94c505f411998c415ed"
            ),
            "uncompressed_bytes": 3203227,
        },
        "repository_ijo1366": {
            "materialization_command": (
                "git -C \"$REPO\" show "
                "main:vendor/cobrapy/src/cobra/data/iJO1366.xml.gz "
                "> \"$BENCH_TMP/iJO1366.xml.gz\""
            ),
            "checksum_command": (
                "printf '%s  %s\\n' "
                "e100c6a9fdc30f6b880d390f8af9941422202b8714c7786629f19c98b076d208 "
                "\"$BENCH_TMP/iJO1366.xml.gz\" | sha256sum --check -"
            ),
            "git_blob_sha": "a65b366c7751e75d94604801ce7c626bd6221f56",
            "compressed_sha256": IJO1366_COMPRESSED_SHA256,
            "compressed_bytes": 395496,
        },
        "rejected_candidate_sources": {
            "iIT341": {
                "url": "https://bigg.ucsd.edu/static/models/iIT341.xml.gz",
                "compressed_sha256": (
                    "533fd22e5a3a0735fef5c9a4787759dee2d9fd79784c4d56b17149ba5b8699d5"
                ),
            },
            "iSB619": {
                "url": "https://bigg.ucsd.edu/static/models/iSB619.xml.gz",
                "compressed_sha256": (
                    "684efb292e6688ede7b20498c632455ec1118cc14e768301fe858644d0140b7d"
                ),
            },
        },
        "pinned_upstream": {
            "clone_command": (
                "git clone https://github.com/marouenbg/VFFVA.git "
                "\"$UPSTREAM_VFFVA\""
            ),
            "checkout_command": (
                "git -C \"$UPSTREAM_VFFVA\" checkout --detach "
                "7cf7b82505bf99aed38a2073e3ed308f79e95802"
            ),
        },
    }


def _recorded_ijo_compile_probe() -> dict[str, object]:
    return {
        "status": (
            "blocked_by_bounded_timeout_in_unchanged_production_preparation"
        ),
        "exact_shell_command": (
            "{ time -p timeout --signal=INT --kill-after=5s 150s env "
            "PYTHONPATH=/workspace/scratch/b7678b508046/"
            "fluxemu-standalone/.venv/lib/python3.12/site-packages:"
            "/workspace/scratch/b7678b508046/fluxemu-vffva-port/codex/src "
            "python probe_ijo_conditioned_compile.py; } "
            "> ijo_conditioned_compile_probe.log 2>&1"
        ),
        "portable_command_description": (
            "Run the same probe with PYTHONPATH=$REPO/codex/src plus the "
            "benchmark environment's installed dependencies; bound it with "
            "timeout --signal=INT --kill-after=5s 150s."
        ),
        "working_directory": (
            "/workspace/scratch/b7678b508046/vffva-benchmark-temp"
        ),
        "timeout_seconds": 150,
        "interrupt_signal": "INT",
        "kill_after_seconds": 5,
        "return_code": 124,
        "wall_seconds": 150.16,
        "user_seconds": 482.83,
        "system_seconds": 3.64,
        "loaded_model_seconds_before_rank_audit": 2.086597447,
        "log_sha256": (
            "59eaa771e4265b8dc62a8c85379e10f3ed9b3285fafd8e0d32fce65b23f40b4b"
        ),
        "sampled_stack_profile": [
            {
                "elapsed_seconds": 30,
                "stage": "benchmark-only singular-value spectrum",
                "frame": (
                    "numpy/linalg/_linalg.py:1850 in svd called from "
                    "benchmark_vffva_highs_scaling.py:195"
                ),
            },
            {
                "elapsed_seconds": 60,
                "stage": "unchanged production exact rank after substitution",
                "frame": (
                    "fluxemu/flux_analysis/highs.py:183 in "
                    "_exact_matrix_rank_cached"
                ),
            },
            {
                "elapsed_seconds": 90,
                "stage": (
                    "unchanged production greedy independent-equality selector"
                ),
                "frame": (
                    "fluxemu/flux_analysis/highs.py:635 in "
                    "_select_independent_normalized_equalities"
                ),
            },
            {
                "elapsed_seconds": 120,
                "stage": (
                    "unchanged production greedy independent-equality selector"
                ),
                "frame": (
                    "fluxemu/flux_analysis/highs.py:634 in "
                    "_select_independent_normalized_equalities"
                ),
            },
        ],
        "terminal_exception": "KeyboardInterrupt delivered by bounded timeout",
        "timing_separation": {
            "unchanged_production_compile_probe_wall_seconds": 150.16,
        },
        "blocker": (
            "The benchmark-only rank certificate completed, but unchanged "
            "production compile did not complete within 150 seconds and was "
            "sampled in the existing greedy independent-equality selector. "
            "No production workaround or compiler change was made."
        ),
        "scaling_timing_recorded": False,
    }


def _validation_timing_evidence() -> dict[str, object]:
    return {
        "purpose": (
            "A phase probe checked whether integrity validation was being "
            "silently bypassed or subtracted from reusable timings."
        ),
        "diagnostic_model": (
            "official BiGG iIT341; later rejected for cold parity"
        ),
        "first_uncached_prepare_seconds": 61.63360981900041,
        "first_run_prepared_highs_vffva_workers_1_seconds": (
            1.843397279999408
        ),
        "subsequent_public_run_highs_vffva_workers_1_seconds": (
            1.8673849949991563
        ),
        "prepared_and_public_result_sha256_identical": True,
        "full_sha_unavailable_from_ephemeral_phase_probe": True,
        "validation_bypass_used": False,
        "measured_path": (
            "Warm immutable-model-cache full-contract public API calls still "
            "execute FBA, PreparedFluxRegion integrity validation, worker "
            "construction, max then min passes, and canonical result assembly."
        ),
        "accounting": (
            "T1/Tp and cold/reusable ratios exclude first uncached preparation. "
            "They are not fresh-process end-to-end timings. Fresh-process "
            "latency on the genome-scale probes is preparation-dominated."
        ),
    }


def _rejected_candidate_evidence() -> list[dict[str, object]]:
    return [
        {
            "model_id": "iIT341",
            "disposition": (
                "rejected before scaling because cold-reference parity did "
                "not satisfy the unchanged 1e-7 gate"
            ),
            "source": {
                "kind": "temporary official BiGG SBML download",
                "url": "https://bigg.ucsd.edu/static/models/iIT341.xml.gz",
                "catalog_url": "https://bigg.ucsd.edu/models/iIT341",
                "compressed_bytes": 148004,
                "uncompressed_bytes": 2748529,
                "compressed_sha256": (
                    "533fd22e5a3a0735fef5c9a4787759dee2d9fd79784c4d56b17149ba5b8699d5"
                ),
                "uncompressed_sha256": (
                    "0574aa79c639885329302c87e9509d35ba879a33359647397a462e4dca7d71f0"
                ),
                "pinned_byte_mirror": {
                    "repository": "SystemsBioinformatics/ecmtool",
                    "commit": (
                        "250b26fc9bbcb212f6aaecc8dcd16d48f4bb5403"
                    ),
                    "git_blob_sha": (
                        "4f43d3c834bb79d1aae1aa2b33ac07a686076cc0"
                    ),
                    "byte_identity": (
                        "byte-identical to decompressed official BiGG download"
                    ),
                },
                "license_url": "https://bigg.ucsd.edu/license",
                "temporary_input_not_committed": True,
            },
            "raw_dimensions": {
                "reactions": 554,
                "balanced_metabolites": 485,
                "stoichiometric_terms": 2314,
                "endpoint_lps": 1108,
            },
            "objective": {
                "direction": "maximise",
                "terms": [
                    {
                        "reaction_id": "R_BIOMASS_HP_published",
                        "coefficient": 1.0,
                    }
                ],
            },
            "preparation": {
                "raw_unchanged_compile_status": "passed",
                "raw_unchanged_compile_seconds": 60.241238,
                "fba_objective_value": 0.6928126934707873,
            },
            "fraction_0_9_parity_probe": {
                "tolerance": CORRECTNESS_TOLERANCE,
                "cold_reference_seconds": 10.426108,
                "reusable_workers_1_seconds": 1.814664,
                "reusable_workers_1_max_absolute_range_difference": (
                    3.440382982056178e-6
                ),
                "reusable_workers_1_cells_above_tolerance": 11,
                "default_worst_endpoint": {
                    "reaction_id": "R_GAPD",
                    "direction": "maximum",
                    "cold_value": -1.744242695962654,
                    "reusable_value": -1.744239255579672,
                },
                "audited_reusable_workers_1_seconds": 6.737184,
                "audited_reusable_workers_1_max_absolute_range_difference": (
                    2.5989864216313663e-7
                ),
                "audited_reusable_workers_1_cells_above_tolerance": 1,
                "audited_worst_endpoint": {
                    "reaction_id": "R_PPA",
                    "direction": "maximum",
                    "cold_value": 7.420014373933649,
                    "reusable_value": 7.420014114035006,
                },
                "passed": False,
            },
            "fraction_1_0_single_probe": {
                "cold_reference_seconds": 10.546201,
                "reusable_status": "failed_closed",
                "exact_error": (
                    "FVA maximum for reaction 'R_EX_pi_e' failed: solver "
                    "status Optimal; endpoint optimum excludes the "
                    "independently validated FBA witness -0.641976"
                ),
                "passed": False,
            },
            "tolerance_or_semantics_changed": False,
            "scaling_timing_recorded": False,
        },
        {
            "model_id": "iSB619",
            "disposition": (
                "rejected before scaling because the cold reference itself "
                "failed the unchanged numerical-validation contract"
            ),
            "source": {
                "kind": "temporary official BiGG SBML download",
                "url": "https://bigg.ucsd.edu/static/models/iSB619.xml.gz",
                "catalog_url": "https://bigg.ucsd.edu/models/iSB619",
                "compressed_bytes": 196847,
                "uncompressed_bytes": 3932643,
                "compressed_sha256": (
                    "684efb292e6688ede7b20498c632455ec1118cc14e768301fe858644d0140b7d"
                ),
                "uncompressed_sha256": (
                    "296a2ef173900768767c30e7b8dc7118b35194cbe3643521b488a416a5144ed2"
                ),
                "pinned_byte_mirror": {
                    "repository": "biosimulations/biosimulations-bigg",
                    "commit": (
                        "883c8dc298d10c5472c8e11c5e8f86ddd1278b0d"
                    ),
                    "path": (
                        "biosimulations_bigg/final/projects/iSB619/iSB619.xml"
                    ),
                    "git_blob_sha": (
                        "955734df47a3ceb57d747a5feb86d8f489c8ad4b"
                    ),
                    "byte_identity": (
                        "byte-identical to decompressed official BiGG download"
                    ),
                },
                "license_url": "https://bigg.ucsd.edu/license",
                "temporary_input_not_committed": True,
            },
            "raw_dimensions": {
                "reactions": 743,
                "balanced_metabolites": 655,
                "stoichiometric_terms": 3821,
                "endpoint_lps": 1486,
            },
            "objective": {
                "direction": "maximise",
                "terms": [
                    {
                        "reaction_id": "R_BIOMASS_SA_8a",
                        "coefficient": 1.0,
                    }
                ],
            },
            "preparation": {
                "raw_unchanged_compile_status": "passed",
                "raw_unchanged_compile_seconds": 179.233859,
                "conditioned_rank": 596,
                "affine_rank_after_fixed_bound_substitution": 593,
                "compiled_fingerprint_prefix": "5585551",
            },
            "fraction_0_9_cold_reference_probe": {
                "status": "failed_closed_during_independent_validation",
                "maximum_bound_violation": {"lower": 0.0, "upper": 0.0},
                "maximum_normalized_mass_balance_residual": (
                    9.947598300641403e-14
                ),
                "maximum_raw_mass_balance_residual": (
                    9.094947017729282e-12
                ),
                "maximum_conditioned_row_residual": (
                    1.2298167142630176e-7
                ),
                "tolerance": CORRECTNESS_TOLERANCE,
                "passed": False,
            },
            "fraction_1_0_prepared_fva_probe": {
                "status": "failed_closed_during_existing_validation",
                "maximum_conditioned_row_residual": (
                    1.1656905532808585e-5
                ),
                "tolerance": CORRECTNESS_TOLERANCE,
                "passed": False,
            },
            "tolerance_or_semantics_changed": False,
            "scaling_timing_recorded": False,
        },
    ]


def _recorded_direct_upstream_evidence() -> dict[str, object]:
    cplex_output = """PINNED_UPSTREAM_COMMIT=7cf7b82505bf99aed38a2073e3ed308f79e95802
COMMAND=make
command -v mpicc: command -v mpirun: command -v cplex: Python cplex API: Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'cplex'
CPLEX candidate directories:
make all_c
make[1]: Entering directory '/workspace/scratch/b7678b508046/vffva-upstream-7cf7b82505bf99aed38a2073e3ed308f79e95802/lib'
rm -f *.o
rm -rf veryfastFVA
echo Clean done
Clean done
mpicc -O3 -fopenmp -c  -I/root/Applications/CPLEX_Studio2212/cplex/include veryfastFVA.c -o veryfastFVA.o
make[1]: mpicc: No such file or directory
make[1]: *** [Makefile:111: veryfastFVA.o] Error 127
make[1]: Leaving directory '/workspace/scratch/b7678b508046/vffva-upstream-7cf7b82505bf99aed38a2073e3ed308f79e95802/lib'
make: *** [Makefile:87: ll] Error 2
BUILD_RETURN_CODE=2
"""
    glpk_output = """PINNED_UPSTREAM_COMMIT=7cf7b82505bf99aed38a2073e3ed308f79e95802
COMMAND=make SOLVER=glpk
command -v mpicc: command -v mpirun: command -v glpsol: GLPK headers:
GLPK libraries:
make all_c
make[1]: Entering directory '/workspace/scratch/b7678b508046/vffva-upstream-7cf7b82505bf99aed38a2073e3ed308f79e95802/lib'
rm -f *.o
rm -rf veryfastFVA
echo Clean done
Clean done
mpicc -O3 -fopenmp -c  -DUSE_GLPK -I/usr/include veryfastFVA.c -o veryfastFVA.o
make[1]: mpicc: No such file or directory
make[1]: *** [Makefile:111: veryfastFVA.o] Error 127
make[1]: Leaving directory '/workspace/scratch/b7678b508046/vffva-upstream-7cf7b82505bf99aed38a2073e3ed308f79e95802/lib'
make: *** [Makefile:87: ll] Error 2
BUILD_RETURN_CODE=2
"""
    expected_output_hashes = {
        "CPLEX": "ad2aecac55a351e822a7c40436aa109e745fbe278a858f289395f2fb5b8d47cd",
        "GLPK": "f5bd6804493a0341cfac0f8108afac43709678f4d848cc364ae3df4f5ff07a49",
    }
    actual_output_hashes = {
        "CPLEX": hashlib.sha256(cplex_output.encode()).hexdigest(),
        "GLPK": hashlib.sha256(glpk_output.encode()).hexdigest(),
    }
    if actual_output_hashes != expected_output_hashes:
        raise RuntimeError("embedded upstream build evidence failed its SHA256 gate")
    return {
        "status": (
            "attempted_in_required_CPLEX_then_GLPK_order_but_blocked_"
            "before_executable"
        ),
        "comparison_timing_recorded": False,
        "comparison_parity_recorded": False,
        "upstream": {
            "repository": "marouenbg/VFFVA",
            "pinned_commit": (
                "7cf7b82505bf99aed38a2073e3ed308f79e95802"
            ),
            "temporary_checkout_not_committed": True,
            "audited_file_sha256": {
                "README.md": (
                    "e116b56984c1f43d216d315a89b0b7477ec794cfd094f8ea19a397121a065935"
                ),
                "LICENSE.txt": (
                    "00425794fb96a43900150783b3740331e7b26d277cf3dc5e4c425998ae1a8c04"
                ),
                "UserGuide.md": (
                    "61bfa73304db63b7f9b9521a477623335965500d78a03d56dd23d77be7442788"
                ),
                "lib/VFFVA.py": (
                    "eb31d9fc440dd7b97e2e2af7f7b334f78f8029cdf42c9391cc9d855a6f55b72d"
                ),
                "lib/veryfastFVA.c": (
                    "d4093ebe733c6909213c943331ea869979ecc33b088f3cc7ee685b14fd2547db"
                ),
                "lib/Makefile": (
                    "63290a851281d185c8b381ae5065d1311af1657d8eef0b848b528dbbcfe19f68"
                ),
            },
        },
        "attempted_backend_order": ["CPLEX", "GLPK"],
        "no_dependency_installation_or_runtime_addition": True,
        "cplex_attempt": {
            "order": 1,
            "exact_command": "make",
            "working_directory": (
                "/workspace/scratch/b7678b508046/"
                "vffva-upstream-7cf7b82505bf99aed38a2073e3ed308f79e95802/lib"
            ),
            "dependency_probes": {
                "mpicc": "not found",
                "mpirun": "not found",
                "cplex_executable": "not found",
                "python_cplex_import": (
                    "ModuleNotFoundError: No module named 'cplex'"
                ),
                "cplex_candidate_directories": [],
                "license_or_installation_available": False,
            },
            "return_code": 2,
            "failure": (
                "make invokes mpicc with the configured CPLEX include path "
                "and fails first because mpicc is absent; CPLEX executable, "
                "Python API, headers/license installation were also unavailable"
            ),
            "log_sha256": (
                "ad2aecac55a351e822a7c40436aa109e745fbe278a858f289395f2fb5b8d47cd"
            ),
            "log_ends_with_newline": True,
            "exact_combined_output": cplex_output,
        },
        "glpk_attempt": {
            "order": 2,
            "exact_command": "make SOLVER=glpk",
            "working_directory": (
                "/workspace/scratch/b7678b508046/"
                "vffva-upstream-7cf7b82505bf99aed38a2073e3ed308f79e95802/lib"
            ),
            "dependency_probes": {
                "mpicc": "not found",
                "mpirun": "not found",
                "glpsol": "not found",
                "glpk_headers": [],
                "glpk_libraries": [],
            },
            "return_code": 2,
            "failure": (
                "make invokes mpicc with -DUSE_GLPK and fails because mpicc "
                "is absent; GLPK executable, headers, and libraries were also "
                "unavailable"
            ),
            "log_sha256": (
                "f5bd6804493a0341cfac0f8108afac43709678f4d848cc364ae3df4f5ff07a49"
            ),
            "log_ends_with_newline": True,
            "exact_combined_output": glpk_output,
        },
        "execution": {
            "original_binary_built": False,
            "original_binary_run": False,
            "blocker": (
                "Neither already-installed CPLEX nor GLPK plus the required "
                "MPI compiler/runtime was available. Installation was "
                "deliberately not attempted because those are not FluxEMU "
                "runtime dependencies."
            ),
        },
        "semantics_required_if_execution_had_become_available": {
            "model": (
                "the same conditioned iLJ478 single-objective-reaction LP "
                "used for the HiGHS scaling benchmark"
            ),
            "reaction_set_and_order": "identical",
            "fraction_of_optimum": 0.9,
            "retained_region": (
                "pre-impose the exact FluxEMU retained-objective row in the "
                "MPS, retain the biomass objective column for upstream VFFVA "
                "to locate and clear, and invoke upstream with optPerc=-1 so "
                "it does not apply its rounded objective-bound shortcut"
            ),
            "proof_gate": (
                "numerically prove retained-bound identity and cold bounds "
                "within 1e-7 before any timing"
            ),
            "status": "not reached because no upstream executable could be built",
        },
        "interpretation": (
            "There is no direct original-VFFVA timing result. The "
            "cold/reusable ratio and shared-memory scaling measurements are "
            "internal FluxEMU engineering evidence only."
        ),
    }


def _balanced_stoichiometry(
    model: FluxModel,
) -> tuple[tuple[str, ...], np.ndarray]:
    metabolite_ids = tuple(
        metabolite.metabolite_id
        for metabolite in model.metabolites
        if metabolite.steady_state_balanced
    )
    metabolite_index = {
        metabolite_id: index
        for index, metabolite_id in enumerate(metabolite_ids)
    }
    matrix = np.zeros((len(metabolite_ids), len(model.reactions)), dtype=float)
    for column, reaction in enumerate(model.reactions):
        accumulated: dict[int, list[float]] = {}
        for term in reaction.stoichiometric_terms:
            row = metabolite_index.get(term.metabolite_id)
            if row is not None:
                accumulated.setdefault(row, []).append(float(term.coefficient))
        for row, coefficients in accumulated.items():
            matrix[row, column] = math.fsum(coefficients)
    return metabolite_ids, matrix


def _rank_preserving_large_model(
    model: FluxModel,
    *,
    enabled: bool,
    verify_conditioned: bool = True,
    scope_label: str = "explicit benchmark model",
) -> tuple[FluxModel, dict[str, object]]:
    """Select original equality rows only after proving exact redundancy.

    Some official genome-scale SBML files contain a cluster of exactly
    redundant metabolite equations whose large-matrix SVD roundoff lies just
    above FluxEMU's deliberately strict machine-noise cutoff.  This explicit,
    guardian-approved benchmark-only path selects a deterministic subset of
    the original rows; it is not a general loader or a relaxation of
    production validation.
    """

    raw_compile_failure: str | None = None
    preparation_started = time.perf_counter()
    raw_compile_started = time.perf_counter()
    try:
        compile_flux_lp(model)
    except AnalysisError as error:
        raw_compile_failure = str(error)
    raw_compile_seconds = time.perf_counter() - raw_compile_started
    if raw_compile_failure is None:
        return model, {
            "applied": False,
            "raw_compile": "passed",
            "preparation_timing_seconds": {
                "raw_compile": raw_compile_seconds,
                "total": time.perf_counter() - preparation_started,
            },
            "note": "The input compiled directly; no benchmark-only conditioning was needed.",
        }
    if not enabled:
        raise RuntimeError(
            "the raw large model did not satisfy FluxEMU conditioning; rerun "
            "with --remove-redundant-balance-rows only after auditing the "
            f"failure: {raw_compile_failure}"
        )

    try:
        from scipy import __version__ as scipy_version
        from scipy.linalg import qr
    except ImportError as error:
        raise RuntimeError(
            "--remove-redundant-balance-rows requires SciPy for benchmark-only "
            "deterministic pivoted QR; SciPy is not a FluxEMU runtime dependency"
        ) from error

    metabolite_ids, matrix = _balanced_stoichiometry(model)
    row_norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(row_norms).all() or np.any(row_norms == 0.0):
        raise RuntimeError(
            "large-model rank audit requires finite, nonzero balanced rows"
        )
    normalized = matrix / row_norms[:, np.newaxis]
    _progress("rank audit: singular-value spectrum")
    singular_values_started = time.perf_counter()
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    singular_values_seconds = time.perf_counter() - singular_values_started
    if not np.isfinite(singular_values).all() or singular_values[0] <= 0.0:
        raise RuntimeError("large-model rank audit produced invalid singular values")

    # Column-pivoted QR of S.T selects original metabolite equations.  The
    # threshold is the conventional dense machine-precision rank criterion
    # max(m,n) * eps * max(abs(diag(R))).
    _progress("rank audit: deterministic pivoted QR of normalized S.T")
    qr_started = time.perf_counter()
    q_matrix, r_matrix, pivots = qr(
        normalized.T, mode="economic", pivoting=True, check_finite=True
    )
    qr_seconds = time.perf_counter() - qr_started
    diagonal = np.abs(np.diag(r_matrix))
    qr_threshold = (
        max(normalized.shape) * np.finfo(float).eps * float(diagonal.max())
    )
    rank = int(np.count_nonzero(diagonal > qr_threshold))
    if rank <= 0 or rank >= len(metabolite_ids):
        raise RuntimeError(
            "large-model rank audit did not find a nonempty redundant-row cluster"
        )

    relative_singular_values = singular_values / singular_values[0]
    last_retained_relative = float(relative_singular_values[rank - 1])
    first_discarded_relative = float(relative_singular_values[rank])
    machine_cluster_limit = max(normalized.shape) * np.finfo(float).eps
    if (
        last_retained_relative <= math.sqrt(np.finfo(float).eps)
        or first_discarded_relative > machine_cluster_limit
    ):
        raise RuntimeError(
            "large-model discarded directions are not an isolated "
            "machine-noise singular-value cluster"
        )

    retained_indices = tuple(sorted(int(index) for index in pivots[:rank]))
    retained_set = set(retained_indices)
    removed_indices = tuple(
        index for index in range(len(metabolite_ids)) if index not in retained_set
    )
    # Q[:, :rank] spans the selected normalized rows (columns of S.T).  Project
    # every removed normalized row into that span and demand a residual far
    # tighter than the repository's 1e-7 public FVA tolerance.
    q_basis = q_matrix[:, :rank]
    removed_columns = normalized[list(removed_indices), :].T
    reconstruction = q_basis @ (q_basis.T @ removed_columns)
    reconstruction_residuals = np.linalg.norm(
        removed_columns - reconstruction, axis=0
    )
    reconstruction_tolerance = 1e-10
    maximum_reconstruction_residual = float(
        np.max(reconstruction_residuals, initial=0.0)
    )
    if (
        not np.isfinite(reconstruction_residuals).all()
        or maximum_reconstruction_residual > reconstruction_tolerance
    ):
        raise RuntimeError(
            "a removed balance row is not reconstructed within the benchmark "
            f"tolerance: {maximum_reconstruction_residual:g} > "
            f"{reconstruction_tolerance:g}"
        )

    retained_ids = {metabolite_ids[index] for index in retained_indices}
    conditioned = FluxModel(
        tuple(
            replace(
                metabolite,
                steady_state_balanced=(
                    metabolite.steady_state_balanced
                    and metabolite.metabolite_id in retained_ids
                ),
            )
            for metabolite in model.metabolites
        ),
        model.reactions,
        model.objective,
    )
    if (
        tuple(item.metabolite_id for item in conditioned.metabolites)
        != tuple(item.metabolite_id for item in model.metabolites)
        or conditioned.reactions != model.reactions
        or conditioned.objective != model.objective
    ):
        raise RuntimeError(
            "benchmark-only row conditioning changed canonical model order or science"
        )
    conditioned_lp = None
    conditioned_fba = None
    conditioned_compile_seconds = None
    conditioned_fba_seconds = None
    if verify_conditioned:
        _progress("rank audit: unchanged production compile of conditioned model")
        conditioned_compile_started = time.perf_counter()
        conditioned_lp = compile_flux_lp(conditioned)
        conditioned_compile_seconds = (
            time.perf_counter() - conditioned_compile_started
        )
        _progress("rank audit: unchanged production FBA of conditioned model")
        conditioned_fba_started = time.perf_counter()
        conditioned_fba = run_highs_fba(conditioned)
        conditioned_fba_seconds = time.perf_counter() - conditioned_fba_started

    removed_rows = [
        {
            "balanced_row_index": index,
            "metabolite_id": metabolite_ids[index],
            "normalized_reconstruction_residual": float(residual),
        }
        for index, residual in zip(removed_indices, reconstruction_residuals)
    ]
    return conditioned, {
        "applied": True,
        "scope": f"{scope_label}; not a general model transformation",
        "description": "rank-preserving removal of numerically redundant equality rows",
        "raw_compile": "failed_closed_before_conditioning",
        "raw_compile_error": raw_compile_failure,
        "procedure": (
            "normalize balanced rows; deterministic SciPy pivoted QR of S.T; "
            "rank=max(m,n)*eps*max(abs(diag(R))); retain selected original rows "
            "in original order; project and verify every removed row"
        ),
        "scipy": scipy_version,
        "preparation_timing_seconds": {
            "raw_compile_failed_closed": raw_compile_seconds,
            "singular_value_spectrum": singular_values_seconds,
            "pivoted_qr": qr_seconds,
            "conditioned_production_compile": conditioned_compile_seconds,
            "conditioned_production_fba": conditioned_fba_seconds,
            "total": time.perf_counter() - preparation_started,
        },
        "original": {
            "reactions": len(model.reactions),
            "metabolites": len(model.metabolites),
            "balanced_metabolites": len(metabolite_ids),
            "stoichiometric_nonzeros": int(np.count_nonzero(matrix)),
        },
        "conditioned": {
            "reactions": len(conditioned.reactions),
            "metabolites": len(conditioned.metabolites),
            "balanced_metabolites": len(retained_indices),
            "stoichiometric_nonzeros": int(
                np.count_nonzero(matrix[list(retained_indices), :])
            ),
            "production_compile": "passed" if conditioned_lp is not None else "not-run",
            "compiled_fingerprint": (
                conditioned_lp.fingerprint if conditioned_lp is not None else None
            ),
            "fba_status": (
                conditioned_fba.status if conditioned_fba is not None else "not-run"
            ),
            "fba_objective_value": (
                conditioned_fba.objective_value
                if conditioned_fba is not None
                else None
            ),
        },
        "rank_audit": {
            "criterion": "max(matrix_shape) * machine_epsilon * max(abs(diag(R)))",
            "machine_epsilon": float(np.finfo(float).eps),
            "qr_absolute_threshold": qr_threshold,
            "rank": rank,
            "removed_row_count": len(removed_indices),
            "largest_singular_value": float(singular_values[0]),
            "last_retained_relative_singular_value": last_retained_relative,
            "first_discarded_relative_singular_value": first_discarded_relative,
            "retained_to_discarded_singular_gap": (
                last_retained_relative / first_discarded_relative
            ),
            "machine_noise_cluster_relative_limit": machine_cluster_limit,
            "reconstruction_tolerance": reconstruction_tolerance,
            "maximum_normalized_reconstruction_residual": (
                maximum_reconstruction_residual
            ),
            "retained_balanced_row_indices_in_original_order": list(
                retained_indices
            ),
            "removed_rows": removed_rows,
        },
        "preservation_checks": {
            "reaction_order_bounds_and_stoichiometry": "identical",
            "metabolite_order": "identical",
            "objective": "identical",
            "retained_row_order": "original order",
            "conditioned_compile": (
                "passed" if conditioned_lp is not None else "requires bounded external probe"
            ),
            "conditioned_fba": (
                "passed" if conditioned_fba is not None else "not-run"
            ),
        },
    }


def _model_contract(model: FluxModel) -> dict[str, object]:
    lower = np.asarray([float(item.lower_bound) for item in model.reactions])
    upper = np.asarray([float(item.upper_bound) for item in model.reactions])
    bounds_payload = json.dumps(
        [
            [item.reaction_id, float(item.lower_bound), float(item.upper_bound)]
            for item in model.reactions
        ],
        separators=(",", ":"),
    ).encode()
    return {
        "objective": {
            "direction": model.objective.direction,
            "terms": [
                {
                    "reaction_id": term.reaction_id,
                    "coefficient": float(term.coefficient),
                }
                for term in model.objective.terms
            ],
        },
        "bounds": {
            "all_finite": bool(np.isfinite(lower).all() and np.isfinite(upper).all()),
            "fixed_reactions": int(np.count_nonzero(lower == upper)),
            "minimum_lower_bound": float(lower.min()),
            "maximum_lower_bound": float(lower.max()),
            "minimum_upper_bound": float(upper.min()),
            "maximum_upper_bound": float(upper.max()),
            "ordered_reaction_bounds_sha256": hashlib.sha256(bounds_payload).hexdigest(),
        },
    }


def _compare_ranges(reference: FVAResult, candidate: FVAResult) -> dict[str, object]:
    rows_match = reference.ranges.index.equals(candidate.ranges.index)
    columns_match = reference.ranges.columns.equals(candidate.ranges.columns)
    if not rows_match or not columns_match:
        raise RuntimeError("FVA result does not preserve canonical row/column order")
    difference = float(
        np.max(
            np.abs(reference.ranges.to_numpy() - candidate.ranges.to_numpy()),
            initial=0.0,
        )
    )
    if not np.isfinite(difference) or difference > CORRECTNESS_TOLERANCE:
        raise RuntimeError(
            f"FVA result differs from the cold reference by {difference:g}, "
            f"above {CORRECTNESS_TOLERANCE:g}"
        )
    return {
        "passed": True,
        "canonical_rows_match": True,
        "columns_match": True,
        "max_absolute_range_difference": difference,
        "tolerance": CORRECTNESS_TOLERANCE,
    }


def _time_and_check(
    label: str,
    call: Callable[[], FVAResult],
    reference: FVAResult,
    repeats: int,
) -> dict[str, object]:
    seconds: list[float] = []
    differences: list[float] = []
    for run in range(1, repeats + 1):
        _progress(f"timing {label}: repetition {run}/{repeats}")
        started = time.perf_counter()
        result = call()
        seconds.append(time.perf_counter() - started)
        comparison = _compare_ranges(reference, result)
        differences.append(float(comparison["max_absolute_range_difference"]))
    median = statistics.median(seconds)
    return {
        "seconds": seconds,
        "sample_count": len(seconds),
        "median_seconds": median,
        "endpoint_solves_per_second": 2 * len(reference.ranges.index) / median,
        "all_measured_results_passed_parity": True,
        "maximum_measured_absolute_range_difference": max(differences, default=0.0),
    }


def _warm_up(
    label: str,
    call: Callable[[], FVAResult],
    reference: FVAResult,
    warmups: int,
) -> None:
    for run in range(1, warmups + 1):
        _progress(f"warming {label}: repetition {run}/{warmups}")
        _compare_ranges(reference, call())


def _json_instrumentation(values: dict[str, object]) -> dict[str, object]:
    """Return a stable, JSON-shaped snapshot of actual worker lifecycle data."""

    return json.loads(json.dumps(values))


def _benchmark_model(
    *,
    model: FluxModel,
    name: str,
    source: dict[str, object],
    fraction: float,
    workers: Sequence[int],
    chunk_size: int,
    warmups: int,
    reusable_repeats: int,
    cold_repeats: int,
) -> dict[str, object]:
    compiled = compile_flux_lp(model)
    endpoint_count = 2 * len(compiled.reaction_ids)
    _progress(f"parity oracle for {name}: cold native reference")
    reference = run_highs_fva_reference(model, fraction)

    parity: dict[str, object] = {}
    lifecycle: dict[str, object] = {}
    for worker_count in workers:
        _progress(f"parity for {name}: reusable workers={worker_count}")
        trace: dict[str, object] = {}
        result = run_highs_vffva(
            model,
            fraction,
            workers=worker_count,
            chunk_size=chunk_size,
            instrumentation=trace,
        )
        key = f"reusable_threaded_workers_{worker_count}"
        parity[key] = {
            **_compare_ranges(reference, result),
            "result_sha256": result.ranges_sha256,
        }
        lifecycle[key] = _json_instrumentation(trace)

    _progress(f"audited endpoint parity for {name}: reusable workers=1")
    audited_result = run_highs_vffva(
        model,
        fraction,
        workers=1,
        chunk_size=chunk_size,
        audit_endpoints=True,
    )
    audited_parity = {
        **_compare_ranges(reference, audited_result),
        "result_sha256": audited_result.ranges_sha256,
        "audit_endpoints": True,
    }

    cold_call = lambda: run_highs_fva_reference(model, fraction)
    reusable_calls = {
        worker_count: (
            lambda worker_count=worker_count: run_highs_vffva(
                model,
                fraction,
                workers=worker_count,
                chunk_size=chunk_size,
            )
        )
        for worker_count in workers
    }
    _warm_up("cold_native_reference", cold_call, reference, warmups)
    for worker_count, call in reusable_calls.items():
        _warm_up(
            f"reusable_threaded_workers_{worker_count}",
            call,
            reference,
            warmups,
        )

    measurements = {
        "cold_native_reference": _time_and_check(
            "cold_native_reference", cold_call, reference, cold_repeats
        )
    }
    for worker_count, call in reusable_calls.items():
        key = f"reusable_threaded_workers_{worker_count}"
        measurements[key] = _time_and_check(
            key, call, reference, reusable_repeats
        )

    one_worker = float(
        measurements["reusable_threaded_workers_1"]["median_seconds"]
    )
    cold_median = float(measurements["cold_native_reference"]["median_seconds"])
    scaling = []
    for worker_count in workers:
        key = f"reusable_threaded_workers_{worker_count}"
        elapsed = float(measurements[key]["median_seconds"])
        speedup = one_worker / elapsed
        scaling.append(
            {
                "workers": worker_count,
                "median_seconds": elapsed,
                "endpoint_solves_per_second": endpoint_count / elapsed,
                "speedup_t1_over_tp": speedup,
                "parallel_efficiency": speedup / worker_count,
            }
        )

    return {
        "name": name,
        "source": source,
        "fraction_of_optimum": fraction,
        "canonical_model_contract": _model_contract(model),
        "dimensions": {
            "reactions": len(compiled.reaction_ids),
            "balanced_metabolites": len(compiled.balanced_metabolite_ids),
            "nonzeros": compiled.nonzero_count,
            "endpoint_lps": endpoint_count,
            "fingerprint": compiled.fingerprint,
        },
        "schedule": {
            "mode": SCHEDULING_MODE,
            "chunk_size": chunk_size,
            "baseline": "VFFVA dynamic,50" if chunk_size == 50 else "exploratory",
        },
        "parity_before_timing": {
            "cold_reference_result_sha256": reference.ranges_sha256,
            "cold_reference_biological_objective": reference.objective_value,
            "cold_reference_objective_direction": reference.objective_direction,
            "all_paths_passed": True,
            "paths": parity,
            "audited_reusable_one_worker": audited_parity,
        },
        "worker_lifecycle_from_parity_runs": lifecycle,
        "measurements": measurements,
        "measurement_scope": (
            "warm immutable-model/conditioning caches; real public API calls "
            "still execute FBA, prepared-region integrity validation, worker "
            "construction, both endpoint passes, and canonical assembly"
        ),
        "internal_correctness_engineering": {
            "comparison": "cold native reference / reusable one-worker median",
            "cold_over_reusable_one_worker": cold_median / one_worker,
        },
        "shared_memory_scaling": {
            "baseline": "reusable one-worker median",
            "rows": scaling,
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cold and VFFVA-style shared-memory HiGHS FVA"
    )
    parser.add_argument(
        "scaling_model",
        type=Path,
        help="temporary official SBML FBC input for the scaling benchmark",
    )
    parser.add_argument("--fraction", type=float, default=0.9)
    parser.add_argument(
        "--workers", nargs="+", type=_positive_int, default=(1, 2, 4, 8)
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=50)
    parser.add_argument("--warmups", type=_positive_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--scaling-cold-repeats",
        type=_positive_int,
        default=1,
        help="cold-reference repetitions for the scaling model (default: 1)",
    )
    parser.add_argument(
        "--source-remote-commit",
        required=True,
        choices=(SOURCE_REMOTE_COMMIT,),
        help="authoritative pushed engine commit under benchmark",
    )
    parser.add_argument(
        "--remove-redundant-balance-rows",
        action="store_true",
        help=(
            "explicitly apply the audited iJO1366-only rank-preserving "
            "or guardian-approved scaling-model rank-preserving redundant "
            "equality-row preparation"
        ),
    )
    parser.add_argument(
        "--attempted-large-model",
        type=Path,
        required=True,
        help=(
            "temporary raw iJO1366 input to audit and certificate without "
            "claiming a completed scaling run"
        ),
    )
    args = parser.parse_args()

    if not args.remove_redundant_balance_rows:
        parser.error(
            "the pinned iLJ478 benchmark requires the explicit, audited "
            "--remove-redundant-balance-rows opt-in"
        )
    if args.fraction <= 0.0 or args.fraction > 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.fraction != 0.9:
        parser.error("the checked benchmark contract requires --fraction 0.9")
    workers = tuple(dict.fromkeys(args.workers))
    if 1 not in workers:
        parser.error("--workers must include 1 as the scaling baseline")
    if not args.scaling_model.is_file():
        parser.error(f"scaling model does not exist: {args.scaling_model}")
    if not args.attempted_large_model.is_file():
        parser.error(
            f"attempted large model does not exist: {args.attempted_large_model}"
        )
    scaling_sha256 = _sha256(args.scaling_model)
    if scaling_sha256 != ILJ478_COMPRESSED_SHA256:
        parser.error(
            "scaling model is not the audited official BiGG iLJ478 input: "
            f"sha256={scaling_sha256}"
        )
    attempted_large_sha256 = _sha256(args.attempted_large_model)
    if attempted_large_sha256 != IJO1366_COMPRESSED_SHA256:
        parser.error(
            "attempted large model is not the audited repository iJO1366 "
            f"blob: sha256={attempted_large_sha256}"
        )
    source_identity = _source_identity()

    _progress(f"loading scaling model {args.scaling_model}")
    raw_scaling_model = load_sbml_flux_model(args.scaling_model)
    scaling_model, scaling_conditioning = _rank_preserving_large_model(
        raw_scaling_model,
        enabled=args.remove_redundant_balance_rows,
        scope_label="official BiGG iLJ478 scaling benchmark only",
    )
    _progress(f"auditing attempted large model {args.attempted_large_model}")
    raw_attempted_large = load_sbml_flux_model(args.attempted_large_model)
    _, attempted_conditioning = _rank_preserving_large_model(
        raw_attempted_large,
        enabled=True,
        verify_conditioned=False,
        scope_label="repository-vendored iJO1366 attempted-large audit only",
    )
    attempted_conditioning["original"]["endpoint_lps"] = 5166
    attempted_conditioning["conditioned"]["endpoint_lps"] = 5166
    conditioned_compile_probe = _recorded_ijo_compile_probe()
    conditioned_compile_probe["timing_separation"][
        "current_driver_rank_certificate_total_seconds"
    ] = attempted_conditioning["preparation_timing_seconds"]["total"]
    attempted_large = {
        "model_id": "iJO1366",
        "role": "attempted large model; no scaling timings claimed",
        "source": {
            "kind": "temporary read-only copy of repository vendor blob",
            "repository_path": "vendor/cobrapy/src/cobra/data/iJO1366.xml.gz",
            "git_blob_sha": "a65b366c7751e75d94604801ce7c626bd6221f56",
            "compressed_bytes": args.attempted_large_model.stat().st_size,
            "sha256": attempted_large_sha256,
        },
        "canonical_model_contract": _model_contract(raw_attempted_large),
        "rank_preserving_equality_row_certificate": attempted_conditioning,
        "scaling_status": (
            "not run because unchanged production preparation exceeded the "
            "bounded probe"
        ),
        "conditioned_compile_probe": conditioned_compile_probe,
    }
    scaling_conditioning["original"]["endpoint_lps"] = 1304
    scaling_conditioning["conditioned"]["endpoint_lps"] = 1304
    scaling_conditioning["preservation_checks"].update(
        all_reactions_preserved=652,
        only_balance_flags_changed=True,
    )
    scaling_conditioning["interpretation"] = (
        "The benchmark uses an algebraically equivalent independent basis of "
        "the original steady-state equalities. This is an explicit "
        "benchmark-only preparation of this pinned model, not a production "
        "compiler change and not a general model transformation."
    )
    models = (
        (
            load_ecoli_core_flux_model("biomass"),
            "bundled-e-coli-core-biomass",
            {"kind": "bundled", "identifier": "ecoli-core"},
            args.repeats,
        ),
        (
            scaling_model,
            "official BiGG iLJ478 with a benchmark-only algebraically equivalent independent equality-row basis",
            {
                "kind": "temporary official BiGG SBML download",
                "model_id": "iLJ478",
                "organism": "Thermotoga maritima MSB8",
                "url": "https://bigg.ucsd.edu/static/models/iLJ478.xml.gz",
                "catalog_url": "https://bigg.ucsd.edu/models/iLJ478",
                "download_last_updated": "2019-10-31",
                "license_url": "https://bigg.ucsd.edu/license",
                "license_summary": (
                    "BiGG permits educational, research, and non-profit use; "
                    "commercial use requires contacting UC San Diego"
                ),
                "pinned_byte_mirror": {
                    "repository": "biosimulations/biosimulations-bigg",
                    "commit": "883c8dc298d10c5472c8e11c5e8f86ddd1278b0d",
                    "path": "biosimulations_bigg/final/projects/iLJ478/iLJ478.xml",
                    "git_blob_sha": "2eeaeb9da7596fbcd8122411e77045e2ffeb06b1",
                },
                "uncompressed_sha256": "b9e0b8ad96cc4a764785e0dab41cac1b4436935cdc19b94c505f411998c415ed",
                "uncompressed_bytes": 3203227,
                "filename": args.scaling_model.name,
                "compressed_bytes": args.scaling_model.stat().st_size,
                "sha256": scaling_sha256,
                "provenance": {
                    "authoritative_catalog": "BiGG Models",
                    "official_model_id": "iLJ478",
                    "official_download_last_updated": "2019-10-31",
                    "temporary_input_not_committed": True,
                    "byte_identity_check": (
                        "decompressed official BiGG bytes equal pinned mirror "
                        "Git blob 2eeaeb9da7596fbcd8122411e77045e2ffeb06b1"
                    ),
                },
                "rank_preserving_equality_row_preparation": scaling_conditioning,
            },
            args.scaling_cold_repeats,
        ),
    )

    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as error:
        raise RuntimeError(
            "this benchmark requires threadpoolctl to hold numerical-library "
            "threads at one during comparable timings"
        ) from error

    results = []
    with threadpool_limits(limits=1, user_api="blas"):
        timing_threadpools = threadpool_info()
        for model, name, source, cold_repeats in models:
            _progress(f"starting model {name}")
            results.append(
                _benchmark_model(
                    model=model,
                    name=name,
                    source=source,
                    fraction=args.fraction,
                    workers=workers,
                    chunk_size=args.chunk_size,
                    warmups=args.warmups,
                    reusable_repeats=args.repeats,
                    cold_repeats=cold_repeats,
                )
            )

    scaling_result = results[1]
    scaling_result["measurements"]["cold_native_reference"]["limitation"] = (
        "One measured sample after one untimed warmup; this ratio is "
        "descriptive and not a stable-estimator or CI threshold."
    )
    scaling_rows = {
        row["workers"]: row
        for row in scaling_result["shared_memory_scaling"]["rows"]
    }
    pythonpath = os.environ.get("PYTHONPATH")
    exact_argv = [sys.executable, *sys.argv]
    exact_command_parts = exact_argv
    if pythonpath is not None:
        exact_command_parts = ["env", f"PYTHONPATH={pythonpath}", *exact_argv]

    data: dict[str, Any] = {
        "schema_version": 3,
        "diagnostic_only": True,
        "ci_timing_gate": False,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_remote_commit": args.source_remote_commit,
        "source_identity": source_identity,
        "benchmark_scope": {
            "internal_engineering": "cold native versus reusable one-worker",
            "shared_memory_scaling": "reusable one-worker versus reusable threaded workers",
            "direct_original_vffva": "recorded separately; never inferred from internal ratios",
        },
        "invocation": {
            "working_directory": str(Path.cwd()),
            "interpreter": sys.executable,
            "argv": list(sys.argv),
            "pythonpath": pythonpath,
            "exact_command_with_pythonpath": shlex.join(exact_command_parts),
            "portable_command_template": (
                "env PYTHONPATH=\"$REPO/codex/src:$BENCH_PYTHON_SITE\" "
                "\"$PYTHON\" "
                "\"$REPO/codex/benchmarks/"
                "benchmark_vffva_highs_scaling.py\" "
                "\"$BENCH_TMP/iLJ478.xml.gz\" "
                "--attempted-large-model \"$BENCH_TMP/iJO1366.xml.gz\" "
                "--remove-redundant-balance-rows --fraction 0.9 "
                "--workers 1 2 4 8 --chunk-size 50 --warmups 1 "
                "--repeats 3 --scaling-cold-repeats 1 "
                "--source-remote-commit "
                "b672e3551e707e5dd6e7f003f60b04cfc040fbdd"
            ),
            "warmup_runs_per_path": args.warmups,
            "reusable_measured_repeats_per_path": args.repeats,
            "scaling_cold_measured_repeats": args.scaling_cold_repeats,
            "worker_counts": list(workers),
            "scheduling_mode": SCHEDULING_MODE,
            "chunk_size": args.chunk_size,
        },
        "input_acquisition": _input_acquisition_evidence(),
        "environment": {
            **_environment(),
            "timing_blas_thread_limit": 1,
            "timing_threadpools": timing_threadpools,
        },
        "models": results,
        "attempted_large_model_without_scaling": attempted_large,
        "rejected_scaling_candidates": _rejected_candidate_evidence(),
        "validation_timing_method_evidence": _validation_timing_evidence(),
        "direct_original_vffva": _recorded_direct_upstream_evidence(),
        "benchmark_interpretation": {
            "fresh_preparation_vs_warm_public_api": {
                "conditioned_ilj478_first_preparation_seconds": (
                    scaling_conditioning["preparation_timing_seconds"]["total"]
                ),
                "conditioned_ilj478_production_compile_seconds": (
                    scaling_conditioning["preparation_timing_seconds"][
                        "conditioned_production_compile"
                    ]
                ),
                "conditioned_ilj478_warm_public_workers_1_median_seconds": (
                    scaling_result["measurements"][
                        "reusable_threaded_workers_1"
                    ]["median_seconds"]
                ),
                "first_preparation_over_warm_t1": (
                    scaling_conditioning["preparation_timing_seconds"]["total"]
                    / scaling_result["measurements"][
                        "reusable_threaded_workers_1"
                    ]["median_seconds"]
                ),
                "interpretation": (
                    "Fresh-process latency is preparation-dominated. The "
                    "reported T1/Tp values are warm immutable-cache, "
                    "full-contract public API timings, not fresh-process "
                    "end-to-end timings."
                ),
            },
            "internal_cold_reference_speedup": {
                "definition": (
                    "Cold native endpoint-rebuild reference median divided by "
                    "reusable one-worker median; an internal FluxEMU "
                    "engineering comparison only."
                ),
                "bundled_ecoli_core": results[0][
                    "internal_correctness_engineering"
                ]["cold_over_reusable_one_worker"],
                "conditioned_ilj478": scaling_result[
                    "internal_correctness_engineering"
                ]["cold_over_reusable_one_worker"],
                "conditioned_ilj478_cold_sample_count": 1,
                "limitation": (
                    "iLJ478 cold-reference ratio uses one measured sample after "
                    "one untimed warmup."
                ),
            },
            "shared_memory_scaling": {
                "definition": (
                    "T1/Tp from reusable public API medians, excluding first "
                    "uncached model preparation."
                ),
                "conditioned_ilj478_workers_2": {
                    "speedup": scaling_rows[2]["speedup_t1_over_tp"],
                    "efficiency": scaling_rows[2]["parallel_efficiency"],
                },
                "conditioned_ilj478_workers_4": {
                    "speedup": scaling_rows[4]["speedup_t1_over_tp"],
                    "efficiency": scaling_rows[4]["parallel_efficiency"],
                },
                "conditioned_ilj478_workers_8": {
                    "speedup": scaling_rows[8]["speedup_t1_over_tp"],
                    "efficiency": scaling_rows[8]["parallel_efficiency"],
                },
                "small_model_warning": (
                    "The 95-reaction E. coli model exposes overhead and is not "
                    "used to generalize against threaded scaling."
                ),
            },
            "direct_upstream_comparison": (
                "Attempted and blocked at build dependencies; no direct "
                "timing or speedup claim exists."
            ),
            "large_model_attempt": (
                "Raw iJO1366 was rejected by the production rank gate; a full "
                "39-row benchmark-only redundancy certificate was recorded, "
                "but unchanged production preparation timed out at 150 "
                "seconds, so no iJO scaling timing is claimed."
            ),
            "timing_scope": (
                "Reusable scaling paths used one untimed warmup and three "
                "measured repeats at dynamic chunk 50. E. coli cold used three "
                "repeats and iLJ478 cold used one permitted measured repeat. "
                "Genome-scale first preparation is recorded separately and "
                "excluded from T1/Tp."
            ),
            "ci_policy": "No timing ratio is a CI threshold.",
        },
    }
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import os
from pathlib import Path
import subprocess
import sys

from fluxemu.emu import EMU, convolve_mids, source_emu_mid
from fluxemu.model import Tracer


def test_source_subset_is_marginalised_in_emu_position_order():
    tracer = Tracer("s", (("#000", 0.25), ("#101", 0.75)), "no")
    np.testing.assert_array_equal(source_emu_mid(tracer, EMU("s", (3, 1)), 3), (0.25, 0.0, 0.75))


def test_two_precursor_condensation_is_exact_discrete_convolution():
    actual = convolve_mids((np.array((0.25, 0.75)), np.array((0.5, 0.5))))
    np.testing.assert_array_equal(actual, (0.125, 0.5, 0.375))


def test_native_package_import_does_not_load_optional_scientific_backends():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys; import fluxemu.emu; "
        "forbidden={'cobra','mfapy','matplotlib'}; "
        "assert not forbidden.intersection(sys.modules), forbidden.intersection(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

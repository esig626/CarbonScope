"""Load the audited vendored mfapy source used by this workspace prototype."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType


@lru_cache(maxsize=1)
def load_mfapy() -> ModuleType:
    """Return mfapy, falling back to the audited workspace source tree.

    The upstream mfapy 0.6.3 packaging metadata does not expose its nested
    package correctly when installed editable. The fallback is intentionally
    limited to the repository's vendored source (or an explicit override).
    """

    try:
        return importlib.import_module("mfapy")
    except ModuleNotFoundError as exc:
        if exc.name not in {"mfapy", "nlopt"}:
            raise

    configured = os.environ.get("FLUXEMU_MFAPY_SOURCE")
    source_root = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3] / "vendor" / "mfapy"
    )
    package_init = source_root / "mfapy" / "__init__.py"
    if not package_init.is_file():
        raise ModuleNotFoundError(
            "mfapy is not importable and its audited source was not found at "
            f"{source_root}; set FLUXEMU_MFAPY_SOURCE"
        )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return importlib.import_module("mfapy")


def mfapy_source_path() -> Path:
    """Return the source directory of the loaded mfapy package."""

    module = load_mfapy()
    return Path(module.__file__).resolve().parent

"""Deterministic native serialisation for canonical scientific records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
from typing import Any

from .validation import CanonicalModelError


def _native(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$record": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _native(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return {"$tuple": [_native(item) for item in value]}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalModelError("serialised mapping keys must be strings")
        return {"$mapping": {key: _native(value[key]) for key in sorted(value)}}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalModelError("cannot serialise a non-finite float")
        return {"$float": value.hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise CanonicalModelError(f"unsupported canonical value: {type(value).__name__}")


def deterministic_serialise(value: Any) -> str:
    """Return stable UTF-8-compatible JSON without sorting scientific tuples."""

    return json.dumps(_native(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


serialise = deterministic_serialise

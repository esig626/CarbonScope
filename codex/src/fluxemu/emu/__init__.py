"""FluxEMU-owned stationary EMU compiler and evaluator."""

from .graph import CompiledEMUPlan, EMU, EMUContribution, EMULayer, compile_emu_plan
from .tracers import convolve_mids, source_emu_mid
from .stationary import (
    DEFAULT_NATIVE_MID_TOLERANCE,
    LayerDiagnostics,
    NativeStationaryResult,
    evaluate_stationary,
)
from .transient import (
    CompiledTransientEMUPlan,
    DEFAULT_TRANSIENT_ATOL,
    DEFAULT_TRANSIENT_METHOD,
    DEFAULT_TRANSIENT_MID_TOLERANCE,
    DEFAULT_TRANSIENT_RTOL,
    NativeTransientResult,
    TransientDiagnostics,
    TransientStateBlock,
    compile_transient_emu_plan,
    evaluate_transient,
)

__all__ = [
    "CompiledEMUPlan", "EMU", "EMUContribution", "EMULayer", "compile_emu_plan",
    "convolve_mids", "source_emu_mid",
    "DEFAULT_NATIVE_MID_TOLERANCE", "LayerDiagnostics", "NativeStationaryResult",
    "evaluate_stationary",
    "CompiledTransientEMUPlan", "DEFAULT_TRANSIENT_ATOL", "DEFAULT_TRANSIENT_METHOD",
    "DEFAULT_TRANSIENT_MID_TOLERANCE", "DEFAULT_TRANSIENT_RTOL", "NativeTransientResult",
    "TransientDiagnostics", "TransientStateBlock", "compile_transient_emu_plan",
    "evaluate_transient",
]

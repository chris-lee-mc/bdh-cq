"""Model package. Modules self-register via bdhx.registry.register_model."""

from bdhx.models.base import BlockCounter, FlopsReport, ParamReport, ReasoningModel, SolveOutput

__all__ = ["BlockCounter", "FlopsReport", "ParamReport", "ReasoningModel", "SolveOutput"]

# Planned model modules (FRAMEWORK_SPEC section 1).
for _mod in (
    "bdh",
    "bdh_cq",
    "looped_transformer",
    "transformer",
    "gated_deltanet",
    "unified_block",
):
    try:
        __import__(f"bdhx.models.{_mod}")
    except ImportError:
        pass
del _mod

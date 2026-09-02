"""bdhx: experiment framework for recurrent reasoning models (BDH / BDH-CQ)."""

__version__ = "0.1.0"

from bdhx.registry import (
    get_model,
    get_task,
    list_models,
    list_tasks,
    register_model,
    register_task,
)

__all__ = [
    "__version__",
    "get_model",
    "get_task",
    "list_models",
    "list_tasks",
    "register_model",
    "register_task",
]

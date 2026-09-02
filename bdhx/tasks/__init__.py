"""Task package. Modules self-register via bdhx.registry.register_task."""

from bdhx.tasks.base import Episode, EpisodeBatch, EpisodicTask, pad_and_batch

__all__ = ["Episode", "EpisodeBatch", "EpisodicTask", "pad_and_batch"]

# Planned task modules (FRAMEWORK_SPEC section 1). Imported opportunistically so
# that agents can add modules without editing this file.
for _mod in (
    "binding",
    "overwrite",
    "distractors",
    "contradict",
    "compose",
    "propagate",
    "copy",
    "order",
    "nested",
    "legacy",
):
    try:
        __import__(f"bdhx.tasks.{_mod}")
    except ImportError:
        pass
del _mod

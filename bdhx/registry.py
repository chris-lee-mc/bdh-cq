"""Name -> constructor registries for tasks and models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_TASKS: dict[str, Any] = {}
_MODELS: dict[str, Any] = {}

T = TypeVar("T")


def _register(store: dict[str, Any], name: str, kind: str) -> Callable[[T], T]:
    def deco(obj: T) -> T:
        if name in store and store[name] is not obj:
            raise KeyError(f"{kind} '{name}' already registered")
        store[name] = obj
        return obj

    return deco


def register_task(name: str) -> Callable[[T], T]:
    """Class/factory decorator registering an EpisodicTask under `name`."""
    return _register(_TASKS, name, "task")


def register_model(name: str) -> Callable[[T], T]:
    """Class/factory decorator registering a ReasoningModel under `name`."""
    return _register(_MODELS, name, "model")


def get_task(name: str) -> Any:
    if name not in _TASKS:
        raise KeyError(f"unknown task '{name}'; registered: {sorted(_TASKS)}")
    return _TASKS[name]


def get_model(name: str) -> Any:
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'; registered: {sorted(_MODELS)}")
    return _MODELS[name]


def list_tasks() -> list[str]:
    return sorted(_TASKS)


def list_models() -> list[str]:
    return sorted(_MODELS)

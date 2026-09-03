"""Config schema (FRAMEWORK_SPEC section 2): frozen pydantic models + YAML loading."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECURRENCE_KINDS = (
    "plain",
    "residual",
    "step_gate",
    "init_skip",
    "step_emb",
    "adapter",
    "attn_residual",
    "combo",
)
MEMORY_KINDS = ("bdh", "none", "kv", "gated_deltanet")


def _coerce_int(v: Any) -> Any:
    """YAML numbers written as 10_000_000 may arrive as strings; coerce them."""
    if isinstance(v, str):
        s = v.replace("_", "").strip()
        try:
            return int(s)
        except ValueError:
            return v
    return v


def _coerce_float(v: Any) -> Any:
    if isinstance(v, str):
        s = v.replace("_", "").strip()
        try:
            return float(s)
        except ValueError:
            return v
    return v


Int = Annotated[int, BeforeValidator(_coerce_int)]
Float = Annotated[float, BeforeValidator(_coerce_float)]


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExperimentCfg(_Base):
    name: str
    stage: str
    tags: list[str] = []


class RecurrenceCfg(_Base):
    kind: Literal[RECURRENCE_KINDS] = "plain"  # type: ignore[valid-type]
    share_weights: bool = True
    input_injection: bool = True
    step_embedding: bool = False
    adapter_rank: Int = 0


class MemoryCfg(_Base):
    kind: Literal[MEMORY_KINDS] = "bdh"  # type: ignore[valid-type]
    reset_per_episode: bool = True


class TTTCfg(_Base):
    rank: Int
    steps: Int
    lr: Float
    targets: list[str]


class ModelCfg(_Base):
    name: str
    params_target: Int | None = None
    width: Int | None = None
    depth: Int = 1
    precision_state: str = "bf16"
    recurrence: RecurrenceCfg = RecurrenceCfg()
    memory: MemoryCfg = MemoryCfg()
    ttt: TTTCfg | None = None


class TaskCfg(_Base):
    name: str
    seed: Int = 1000
    train_difficulties: list[dict[str, Int]] | None = None
    eval_difficulties: dict[str, list[dict[str, Int]]] | None = None
    n_eval_episodes: Int = 1000


class TrainingCfg(_Base):
    steps: Int
    batch_size: Int
    seed: Int = 1
    optimizer: str = "adamw"
    lr: Float = 3.0e-4
    weight_decay: Float = 0.1
    warmup_steps: Int = 1000
    schedule: str = "cosine"
    grad_clip: Float = 1.0
    precision: str = "bf16"
    loss: Literal["final_answer", "legacy"] = "final_answer"
    checkpoint_every_steps: Int = 1000
    eval_every_steps: Int = 2500


class CurriculumCfg(_Base):
    schedule: list[Int]
    switch_at: list[Float]


class ReasoningCfg(_Base):
    train_steps: list[Int] = [1]
    train_step_sampling: Literal["uniform", "curriculum", "fixed"] = "uniform"
    curriculum: CurriculumCfg | None = None
    delayed_start_fraction: Float = 0.0
    backprop_through_all_steps: bool = True


class EvaluationCfg(_Base):
    reasoning_steps: list[Int] = [1]
    diagnostics: bool = True
    record_adaptation_cost: bool | None = False


class ComputeCfg(_Base):
    device: str = "auto"
    deterministic: bool = False
    max_wall_clock_minutes: Int = 180


class Config(_Base):
    experiment: ExperimentCfg
    model: ModelCfg
    task: TaskCfg
    training: TrainingCfg
    reasoning: ReasoningCfg = ReasoningCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
    compute: ComputeCfg = ComputeCfg()


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins for scalars and lists."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_dotted_overrides(data: dict, overrides: dict[str, Any] | None) -> dict:
    """Apply {'training.lr': 1e-3} style overrides to a nested dict (returns a copy)."""
    out = copy.deepcopy(data)
    for dotted, value in (overrides or {}).items():
        if isinstance(value, str):
            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError:
                pass
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
    return out


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_raw(path: str | Path, _seen: tuple[str, ...] = ()) -> dict:
    """Load a YAML config and resolve its `extends:` chain into a plain dict."""
    p = _resolve_path(path)
    key = str(p)
    if key in _seen:
        raise ValueError(f"circular extends involving {key}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config {p} is not a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return deep_merge(load_raw(parent, _seen + (key,)), data)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML (resolving `extends:`), apply dotted overrides, validate."""
    return Config.model_validate(apply_dotted_overrides(load_raw(path), overrides))


# Fields that identify a repeat of an experiment rather than a different one.
# FRAMEWORK_SPEC section 3: "two runs with the same hash and seed are the same
# experiment", and section 10 groups summary rows by config hash with seed
# statistics, so the training seed must not enter the hash.
HASH_EXCLUDED_FIELDS = (("training", "seed"),)


def canonical_yaml(cfg: Config) -> str:
    """Deterministic YAML dump of the full config (key order irrelevant)."""
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True, default_flow_style=False)


def hashable_yaml(cfg: Config) -> str:
    """`canonical_yaml` minus the seed axis; this is what `config_hash` digests."""
    data = cfg.model_dump(mode="json")
    for *path, leaf in HASH_EXCLUDED_FIELDS:
        node = data
        for key in path:
            node = node.get(key, {})
        node.pop(leaf, None)
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False)


def config_hash(cfg: Config) -> str:
    """First 12 hex chars of sha256(hashable_yaml(cfg)).

    Seed-invariant on purpose: the three seeds of one sweep arm share a hash
    and differ only in the `_s<seed>` run-directory suffix, which is what lets
    `aggregate.py` compute seed statistics per arm.
    """
    return hashlib.sha256(hashable_yaml(cfg).encode()).hexdigest()[:12]

"""results.json schema and writers (FRAMEWORK_SPEC sections 5 and 6)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TRAIN_LOG_COLUMNS = (
    "step",
    "loss",
    "lr",
    "grad_norm",
    "r_train",
    "examples_seen",
    "wall_s",
    "vram",
)


class TaskMetrics(BaseModel):
    """Whatever task.score() returns beyond exact_match/token_acc (extras allowed)."""

    model_config = ConfigDict(extra="allow")

    stale_rate: float | None = None
    other_rate: float | None = None
    distractor_answer_rate: float | None = None
    first_rate: float | None = None
    last_rate: float | None = None
    partial_depth_acc: float | None = None
    cell_acc: float | None = None
    max_correct_distance: float | None = None
    first_error_position: float | None = None


class AdaptationCost(BaseModel):
    """Present only when evaluation.record_adaptation_cost is true (Stage E)."""

    model_config = ConfigDict(extra="forbid")

    adaptation_latency_ms: float | None = None
    adaptation_flops: float | None = None
    interference_rate: float | None = None


class Diagnostics(BaseModel):
    model_config = ConfigDict(extra="allow")

    state_norm: list[float] = Field(default_factory=list)
    update_norm: list[float] = Field(default_factory=list)
    cos_consecutive: list[float] = Field(default_factory=list)
    active_neuron_frac: list[float] = Field(default_factory=list)
    activation_percentiles: list[list[float]] = Field(default_factory=list)
    jacobian_eigenvalue_estimate: list[float] = Field(default_factory=list)
    nan_count: int = 0


class EvaluationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    reasoning_steps: int
    split: str
    difficulty: dict[str, int] = Field(default_factory=dict)
    n_episodes: int
    exact_match: float
    token_acc: float
    inference_flops_per_episode: float | None = None
    latency_ms_per_episode: float | None = None
    task_metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    adaptation: AdaptationCost | None = None
    diagnostics: Diagnostics | None = None


class ResultsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    config_hash: str
    seed: int
    model: str
    task: str
    params: int | None = None
    solved_width: int | None = None
    gate_extrapolation: Literal["hold_last", "interpolate"] = "hold_last"
    status: str = "ok"
    evaluations: list[EvaluationRow] = Field(default_factory=list)
    training_curve: str = "train_log.csv"


class ResultsWriter:
    """Accumulates evaluation rows and writes run_dir/results.json."""

    def __init__(self, run_dir: str | Path, **header: Any):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "results.json"
        self.results = ResultsFile(**header)

    def add_evaluation(self, row: EvaluationRow | dict[str, Any], **kwargs: Any) -> EvaluationRow:
        if isinstance(row, dict):
            row = EvaluationRow(**{**row, **kwargs})
        self.results.evaluations.append(row)
        return row

    def flush(self) -> Path:
        self.path.write_text(json.dumps(self.results.model_dump(mode="json"), indent=2))
        return self.path

    @staticmethod
    def load(run_dir: str | Path) -> ResultsFile:
        path = Path(run_dir)
        if path.is_dir():
            path = path / "results.json"
        return ResultsFile.model_validate(json.loads(path.read_text()))


class TrainLogWriter:
    """CSV writer for train_log.csv (FRAMEWORK_SPEC section 5 columns)."""

    def __init__(self, run_dir: str | Path, filename: str = "train_log.csv"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(TRAIN_LOG_COLUMNS))
        self._writer.writeheader()
        self._fh.flush()

    def log(self, **row: Any) -> None:
        unknown = set(row) - set(TRAIN_LOG_COLUMNS)
        if unknown:
            raise KeyError(f"unknown train_log columns: {sorted(unknown)}")
        self._writer.writerow({c: row.get(c) for c in TRAIN_LOG_COLUMNS})

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

# Experiment Framework Specification

Status: specification (v0.1). Defines the code that must exist before the
first Stage A sweep. Written so that an implementing agent can build it in
small, reviewable commits (see `docs/HANDOFF_TASKS.md` for the ordering).

Design principles, in priority order:

1. Reproducibility: every number in `docs/RESULTS.md` traces back to a
   config file, a git commit, a seed, and a results JSON on disk.
2. Controlled comparison: one variable changes per experiment; every
   added parameter or FLOP has a matched control.
3. Cloud independence: `run_experiment.py` runs identically on a laptop
   CPU (tiny config) and on a RunPod GPU. RunPod is a launcher, not a
   dependency.
4. Local artifacts are authoritative. WandB or similar is optional and
   never the only copy of a result.

## 1. Package layout

```
bdh-cq-experiments/
  pyproject.toml            # pinned deps, ruff config, pytest config
  bdhx/                     # importable package ("BDH experiments")
    __init__.py
    config.py               # dataclass schema + YAML loading/validation
    metadata.py             # git/hardware/env capture
    seeding.py              # seed_everything, per-component generators
    registry.py             # name -> constructor for models/tasks
    models/
      base.py               # ReasoningModel protocol (section 3)
      bdh.py                # baseline BDH (adapter over community impl)
      bdh_cq.py             # BDH + recurrent reasoning (community CQ)
      looped_transformer.py # shared-block Transformer with input injection
      transformer.py        # fixed-depth Transformer
      gated_deltanet.py     # adapter over fla.layers.GatedDeltaNet + CPU fallback
      recurrence.py         # recurrence engineering variants (H5-H7)
      param_budget.py       # width solver to hit params_target
    tasks/
      base.py               # Episode, EpisodicTask (see TASK_SUITE_SPEC.md)
      binding.py overwrite.py distractors.py contradict.py
      compose.py propagate.py copy.py order.py nested.py
      vocab.py              # symbol pool + structural tokens
    training/
      trainer.py            # loop, AMP, grad clip, checkpoint, resume
      curriculum.py         # recurrence schedules
      evaluate.py           # reasoning-depth sweeps, split-labelled metrics
      diagnostics.py        # recurrent stability instrumentation
      flops.py              # analytic FLOP estimates per model
    results/
      schema.py             # results JSON schema + writer
      aggregate.py          # CSV/JSON tables + plots
  tools/
    run_experiment.py
    generate_sweep.py
    generate_tasks.py
    aggregate_results.py
    profile_config.py
    runpod_launch.py        # optional; needs RUNPOD_API_KEY
    collect_results.py
  configs/
    base/*.yaml             # model/task/training defaults
    stage_a/*.yaml stage_b/*.yaml stage_c/*.yaml stage_d/*.yaml
  docker/
    Dockerfile  entrypoint.sh
  tests/
  docs/
  results/                  # gitignored except results/README.md
```

`bdhx` depends on the community BDH-CQ package as a pinned git
dependency (commit hash recorded in `pyproject.toml`), never vendored
silently. If adapting it requires patches, they live in
`bdhx/models/_community_patches/` with a comment per patch stating why.

## 2. Configuration schema

Plain YAML validated into frozen dataclasses (no Hydra; fewer moving
parts, easier to diff and hash). Unknown keys are an error. Example:

```yaml
experiment:
  name: stage_a_compose_bdhcq_r4
  stage: A
  tags: [stage_a, compose, bdh_cq]
model:
  name: bdh_cq                  # registry key
  params_target: 10_000_000     # width solver hits this within 3 percent
  width: null                   # set explicitly to bypass the solver
  depth: 4                      # number of distinct layers (1 for shared)
  precision_state: bf16         # recurrent state dtype (H8)
  recurrence:
    kind: plain                 # plain | residual | step_gate | init_skip | combo
    share_weights: true
    input_injection: true       # feed the query embedding at every step
    step_embedding: false
    adapter_rank: 0             # per-step low-rank adapter (0 = off)
  memory:
    kind: bdh                   # bdh | none | kv | gated_deltanet
    reset_per_episode: true     # MUST be true unless persistent_memory experiment
task:
  name: compose
  seed: 1000
  train_difficulties: [{depth: 1}, {depth: 2}]
  eval_difficulties:
    interp: [{depth: 1}, {depth: 2}]
    mild:   [{depth: 3}, {depth: 4}]
    strong: [{depth: 6}, {depth: 8}]
  n_eval_episodes: 1000
training:
  steps: 50000
  batch_size: 64
  seed: 1
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 1000
  schedule: cosine
  grad_clip: 1.0
  precision: bf16               # autocast dtype; fp32 master weights
  checkpoint_every_steps: 1000
  eval_every_steps: 2500
reasoning:
  train_steps: [1, 2, 4]        # R_train values sampled per batch
  train_step_sampling: uniform  # uniform | curriculum | fixed
  curriculum: null              # e.g. {schedule: [1, 2, 4, 8], switch_at: [0, 0.25, 0.5, 0.75]}
  delayed_start_fraction: 0.0   # H5: recurrence off until this fraction of steps
  backprop_through_all_steps: true
evaluation:
  reasoning_steps: [1, 2, 4, 8, 16, 32, 64]
  diagnostics: true
compute:
  device: auto
  deterministic: false          # true costs ~10-30 percent; used for tests
  max_wall_clock_minutes: 180   # hard kill; checkpoint first
```

Rules:

- `params_target` triggers `param_budget.solve_width()`, which searches
  the width so that trainable parameters land within 3 percent of target.
  The realized count and the solved width are written to the results
  file. A run whose realized count is outside 5 percent fails fast.
- Configs are hashed (sha256 of canonical YAML) and the hash is the run
  directory name suffix. Two runs with the same hash and seed are the
  same experiment.
- `generate_sweep.py` expands a sweep YAML (`grid:` and `seeds:` keys)
  into one fully-resolved config per job in `generated/<sweep>/exp_NNN.yaml`
  plus `generated/<sweep>/manifest.csv` listing config hash, seed, and
  estimated GPU-minutes.

## 3. Common model interface

```python
class ReasoningModel(nn.Module, Protocol):
    def reset_context(self) -> None: ...
    def ingest_context(self, demonstrations: list[tuple[Tensor, Tensor]]) -> None: ...
    def solve(self, query: Tensor, reasoning_steps: int,
              collect_diagnostics: bool = False) -> SolveOutput: ...
    def forward_episode(self, batch: EpisodeBatch, reasoning_steps: int) -> Tensor:
        """Training path: returns logits for the target positions."""
    def param_report(self) -> ParamReport:
        """total, trainable, per-component breakdown, serialized_bytes."""
    def flops_estimate(self, batch: EpisodeBatch, reasoning_steps: int) -> FlopsReport: ...
```

Behavioural notes (these follow the "do not distort architectures"
instruction):

- Sequence-native baselines (Transformer, looped Transformer, Gated
  DeltaNet) implement `ingest_context` by storing the serialized
  demonstration tokens and `solve` by running the concatenated sequence.
  There is no hidden reformatting; the serialized form is the one defined
  in `TASK_SUITE_SPEC.md` section 1.
- BDH variants may consume demonstrations through their native memory
  update path. The adapter must document (docstring) exactly which
  community function it calls, so that the interface adds no logic of its
  own.
- `reasoning_steps` is a runtime argument of `solve`; no model may bake
  the iteration count into `__init__`. A test asserts that `solve(q, 1)`
  and `solve(q, 16)` execute a different number of block applications
  (checked via a forward hook counter).
- `reset_context()` must zero every episodic buffer. `test_context_isolation`
  runs episode A, resets, runs episode B, and compares against running B
  in a fresh model instance: outputs must match bit-for-bit under
  deterministic mode.
- Batch dimension carries independent episodes; no cross-episode
  communication (no batch norm; verified by a test that permutes the
  batch and checks per-episode outputs are unchanged).

### 3.1 Recurrence engineering variants (`models/recurrence.py`)

All variants share the same block `F` and differ only in the update rule.
`H_0` is the initial latent state after ingestion; `S` is the query
injection (present at every step when `input_injection: true`).

| kind      | update rule                                      | extra params        |
|-----------|--------------------------------------------------|---------------------|
| plain     | H[r+1] = F(H[r], S)                              | 0                   |
| residual  | H[r+1] = H[r] + F(H[r], S)                       | 0                   |
| step_gate | H[r+1] = H[r] + alpha[r] * F(H[r], S)            | R_max scalars       |
| init_skip | H[r+1] = F(H[r], S) + g[r] * H[0]                | R_max scalars       |
| step_emb  | H[r+1] = F(H[r] + e[r], S)                       | R_max x width       |
| adapter   | H[r+1] = F(H[r], S) + A[r] B[r] H[r]  (rank k)    | R_max x 2 x width x k |
| combo     | any subset, only after each shows value (Gate C) |                     |

For steps beyond `R_max` (the largest depth seen in training), scalar
gates and embeddings reuse the last trained value. This is recorded in
the results as `gate_extrapolation: hold_last`. An alternative
`gate_extrapolation: interpolate` is a Stage C sub-ablation.

Matched control rule: any variant that adds `p` parameters is compared
against `plain` with width increased so that it also gains approximately
`p` parameters (solver target = baseline count + p). Both numbers appear
in the table.

## 4. Metadata captured per run (`metadata.py`)

Written to `results/<run>/metadata.json` at start and updated at end:

```
git_commit, git_dirty, config_hash, config (resolved), seed, task_seed,
hostname, gpu_name, gpu_count, cuda_version, cudnn_version, driver_version,
torch_version, python_version, platform, pip_freeze (path to file),
param_total, param_trainable, param_breakdown, serialized_bytes,
solved_width, start_time, end_time, wall_clock_train_s, wall_clock_eval_s,
steps_completed, examples_seen, tokens_seen, train_flops_estimate,
peak_vram_bytes, nan_events, checkpoint_path, log_path,
resumed_from (checkpoint path or null), preemptions (count)
```

`train_flops_estimate` uses the analytic per-model estimate from
`flops.py` times steps, summed over the sampled R_train values. Inference
FLOPs per reasoning depth are recorded per evaluation.

## 5. Results schema (`results/schema.py`)

`results/<run>/results.json`:

```json
{
  "run_id": "...", "config_hash": "...", "seed": 1,
  "model": "bdh_cq", "task": "compose", "params": 10023456,
  "evaluations": [
    {"step": 50000, "reasoning_steps": 8, "split": "strong",
     "difficulty": {"depth": 6}, "n_episodes": 1000,
     "exact_match": 0.412, "token_acc": 0.77,
     "inference_flops_per_episode": 1.2e9, "latency_ms_per_episode": 3.1,
     "diagnostics": {"state_norm": [...], "update_norm": [...],
                     "cos_consecutive": [...], "nan_count": 0}}
  ],
  "training_curve": "train_log.csv"
}
```

One row per (step, reasoning_steps, split, difficulty). `train_log.csv`
has `step, loss, lr, grad_norm, r_train, examples_seen, wall_s, vram`.

## 6. Evaluation protocol (`evaluate.py`)

- Loads the fixed evaluation episodes for the task seed.
- For each `reasoning_steps` in the config, for each split and
  difficulty, runs `reset_context; ingest_context; solve` per episode
  (batched over episodes) and scores with the task's `score`.
- Records latency with `torch.cuda.synchronize` around `solve`.
- With `diagnostics: true`, `solve(collect_diagnostics=True)` returns per
  step: `||H[r]||`, `||H[r+1]-H[r]||`, `cos(H[r+1], H[r])`, fraction of
  active neurons (BDH sparsity), NaN/Inf counts, and the top-k eigenvalue
  magnitude estimate of the step Jacobian via 5 power-iteration steps on a
  random probe (only at `reasoning_steps` in {8, 32}, only for 32
  episodes, to bound cost).
- Also evaluates at intermediate checkpoints (`eval_every_steps`) with a
  reduced depth list `[1, 4, 16]` to produce learning curves.

## 7. Training loop (`trainer.py`)

- AdamW, cosine schedule with warmup, grad clipping, bf16 autocast with
  fp32 master weights and fp32 loss.
- Per batch, sample `R_train` from `reasoning.train_steps` according to
  `train_step_sampling`; log it.
- Curriculum: `curriculum.schedule` and `switch_at` fractions; delayed
  start via `delayed_start_fraction` (recurrence forced to 1 before it).
- Checkpoint every `checkpoint_every_steps` and on SIGTERM (RunPod spot
  preemption sends SIGTERM); checkpoint includes model, optimizer,
  scheduler, rng states (torch, numpy, python, cuda), step, and the
  config hash. `--resume` picks up the latest checkpoint if the config
  hash matches, otherwise errors.
- NaN policy: a NaN loss increments `nan_events`, skips the step, and
  after 20 consecutive NaN steps aborts with status `diverged`. Diverged
  runs are kept and reported, never deleted.
- Deterministic mode (`compute.deterministic: true`) sets
  `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; used by tests and by the Phase 0
  reproduction, not by sweeps.

## 8. Seed policy (enforced by tooling)

- `generate_sweep.py` refuses to emit a sweep with fewer than 3 seeds
  unless `--dev` is passed; `--dev` sweeps are tagged `dev` and the
  aggregator excludes them from README-grade tables.
- README/report claims use 5 seeds. The aggregator marks any row with
  fewer than 5 seeds as `provisional`.
- Reporting: mean, std, min, max, all individual values, and a 95 percent
  bootstrap interval of the mean (1000 resamples over seeds).
- Failed or diverged seeds stay in the table with status `diverged` or
  `incomplete`; the mean is reported both including (as 0 accuracy) and
  excluding them, labelled.

## 9. Compute matching and reporting

Every comparison table has these columns:

`model | params (M) | train FLOPs (PF) | R_train | R_test | inference FLOPs / episode | exact match (mean +- std, n seeds) | split`

The aggregator flags rows where, within a comparison group, params differ
by more than 5 percent or train FLOPs differ by more than 15 percent, and
prints "NOT MATCHED" in the table. Compute efficiency plots use inference
FLOPs on the x axis; parameter efficiency plots use params.

## 10. Aggregation and plots (`aggregate.py`)

`python tools/aggregate_results.py results/ --out reports/<date>/`
produces:

- `all_runs.csv` (one row per evaluation row per run), `summary.csv`
  (grouped by config hash and evaluation key, seed statistics),
  `flags.csv` (matching/variance/convergence warnings).
- Plots (PNG + the CSV backing each):
  1. `acc_vs_reasoning_steps_<task>.png` (one line per model, per split;
     vertical marker at max R_train)
  2. `acc_vs_difficulty_<task>.png` (split boundaries shaded)
  3. `acc_vs_n_bindings.png` (Stage B)
  4. `acc_vs_inference_flops_<task>.png`
  5. `acc_vs_params_<task>.png`
  6. `state_norm_vs_iteration_<model>_<task>.png` and update-norm/cosine
     companions
- Error bars are seed std; n seeds annotated on each point.
- No plot is produced if any contributing run lacks `split` labels or
  metadata.

## 11. Profiling and cost guard (`profile_config.py`)

- Runs 50 training steps plus one evaluation pass at the largest
  `reasoning_steps` for a config, on the target device, and prints
  seconds/step, eval seconds, peak VRAM, and projected minutes for the
  full config.
- `generate_sweep.py --estimate` calls the profiler for one config per
  distinct model/params cell and multiplies out. It prints projected
  GPU-hours and cost at the price given by `--gpu-hourly-usd`.
- Sweeps above `--max-gpu-hours` (default 20) require
  `--allow-large-sweep`.

## 12. Tests (`tests/`)

Task generator tests are in `TASK_SUITE_SPEC.md` section 5. Framework
tests:

- `test_param_budget`: solver hits target within 3 percent for every
  registered model at 2M, 10M, 25M.
- `test_param_count_matches_state_dict`: `param_report().total` equals the
  sum over `state_dict()` numel.
- `test_reasoning_steps_runtime`: hook counter shows exactly R block
  applications for R in {1, 4, 16} for every recurrent model.
- `test_context_isolation`: described in section 3; the critical test.
- `test_memory_reset_zeroes_state`: after `reset_context()` every
  episodic buffer is zero or None.
- `test_batch_independence`: permuting the batch permutes outputs.
- `test_checkpoint_resume`: train 20 steps, checkpoint at 10, resume,
  compare final weights to an uninterrupted 20-step run (deterministic
  mode, CPU).
- `test_results_roundtrip`: write results.json, read back, equality.
- `test_config_rejects_unknown_keys`, `test_config_hash_stable`.
- `test_sweep_seed_policy`: fewer than 3 seeds without `--dev` errors.
- `test_recurrence_variants_shapes`: every `kind` produces the same
  output shape and, at R=1 with zero-initialized gates, `step_gate` equals
  `residual` at alpha=1 and `plain` at alpha=0 only for the plain path
  (documents the semantics).
- `test_flops_estimate_monotone`: FLOPs increase with R and with width.
- `test_curriculum_schedule`: R at each fraction matches the schedule;
  delayed start yields R=1 before the threshold.

CI runs `ruff check`, `ruff format --check`, and `pytest -x` on CPU with
tiny configs. Tests must finish in under 5 minutes on 4 CPU cores.

## 13. Community-code adapter notes (from Phase 0 inspection)

Concrete binding points in `lucidrains/bdh-cq` at commit `c246f890`:

- Ingest: `BDHReasoningWrapper.forward(token_tensor, memories=..., update_memory=..., return_memory=True)`
  (`bdh_cq.py:463-473`) or `icq.ingest` (`icq.py:134-147`, chunk 128).
- Reason and answer: `BDHReasoningWrapper.generate(*stages, N, memories=..., num_tokens=..., stop_token=...)`
  (`bdh_cq.py:604-680`), single sequence only; the adapter adds batching.
- Reset: pass `memories=None`. `Memory` is an immutable namedtuple and
  `combine_memories` is out of place (`bdh_cq.py:255-257`), so reuse of
  an ingested `Memory` across a reasoning-depth sweep is safe today;
  `test_context_isolation` guards this against future changes.
- Freeze memory during latent steps: `update_latent_memory=False`.
- Recurrence count is the int stage; nothing is baked into `__init__`.
- Community attention residual: `BDH(attn_residual=True, attn_residual_depth_bias_distance=1)`;
  exposed as `recurrence.kind: attn_residual` and labelled "(community)".
- The community training loss (latent positions predict the first token
  of the next segment, `bdh_cq.py:517-522`) is `training.loss: legacy`;
  `training.loss: final_answer` computes cross-entropy only on the
  answer tokens after the last latent stage.
- `share_weights: false` for BDH requires instantiating `depth` separate
  `BDHBlock`s in the adapter; the community `BDH` cannot do it.
- Parameter reference points: figure7 config 794,145 params; `icq.py`
  default 2,370,080; README example 70,811,680.

## 14. Dependency pins

See `EXPERIMENT_PLAN.md` section 15. `uv.lock` is authoritative once the
package exists; the RunPod image installs from it.

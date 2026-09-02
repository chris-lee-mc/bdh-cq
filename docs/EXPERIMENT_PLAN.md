# Experiment Plan: Controlled Study of BDH-CQ Mechanisms

Status: v0.1, written 2026-09-02 after Phase 0 inspection. This is the
document that gates implementation; no sweep is launched until the
sections it references (`FRAMEWORK_SPEC.md`, `TASK_SUITE_SPEC.md`,
`RUNPOD.md`) are implemented and their tests pass.

Central question: which components of the BDH-CQ design provide
capabilities that cannot be explained by ordinary recurrent depth,
linear or recurrent fast-weight memory, or additional test-time compute?

Method: change one mechanism at a time, match parameters and training
FLOPs, run at least three seeds, keep negative results, make every number
reproducible from a config plus a commit plus a seed.

## 1. Current architecture (Phase 0 findings)

Target implementation: `lucidrains/bdh-cq`, commit `c246f890` (2026-08-28),
MIT, marked "wip" by its author. It is a community reconstruction of the
BDH-CQ paper (arXiv 2608.09888), not Pathway code. Pathway released neither
weights nor the reasoning mechanism; the community author states in code
that the latent transition function is unknown to him. Full detail and
line references are in `PAPER_IMPLEMENTATION_GAPS.md`. Summary:

- Block: shared-QK ReLU linear attention without softmax, self masked
  out, values equal to the raw residual stream, a gated feed-forward tail
  gated by the same sparse activations, parameter-free LayerNorms.
- Depth: ONE block instance applied `depth` times. Depth and recurrence
  are the same knob.
- Memory: per head, per depth slot, an additive `k^T v` Hebbian matrix
  with no decay; read back as `q @ M` and added to in-sequence attention.
- Reasoning ("CQ"): `BDHReasoningWrapper` takes an interleaving of token
  chunks and integers. Each integer runs the block that many times on the
  model's own last hidden state fed back as a pseudo-token, optionally
  writing to memory, with an optional learned step embedding. Each latent
  position is trained to predict the first token of the next chunk.
- Community additions the author labels as his own: an attention
  residual over all prior block outputs with a "distance to end of
  reasoning" bias, reported to stabilize recurrence beyond 4 steps.
- Tasks: four ARC-style families (propagation, copy, order, nesting)
  with demo levels and harder test levels; a two-hop function composition
  probe; an enwik8 language-model script.
- Tests: 46 pass on CPU (18.9 s). Reproduction: see
  `PHASE0_REPRODUCTION.md`. The default `figure7.py` regime (batch 1,
  794k parameters, 800 steps) is a pipeline smoke test, not a baseline.

## 2. Paper-vs-code gaps that shape the design

From `PAPER_IMPLEMENTATION_GAPS.md`, the gaps that matter most:

1. The latent transition function is unspecified. Every recurrence
   variant we test is a hypothesis about it; none is "the" BDH-CQ.
2. Weight sharing is confirmed in Pathway's reference code, so a
   non-shared fixed-depth BDH must be added as a control.
3. Memory has no decay or delta correction; this predicts specific
   failures on overwrite and contradiction tasks that Gated DeltaNet
   should not share. That is a falsifiable prediction, so it goes in
   Stage B.
4. The training target of the latent loop is a community choice. The
   framework keeps it as `legacy` and adds a final-answer loss; the
   choice is a config field, reported in every table.
5. The tested reasoning range stops at 8. Extrapolation to 64 is
   unexplored territory and must be instrumented, not just scored.

## 3. Hypotheses, controls, metrics, falsifiers

Each hypothesis lists the control, the primary metric, and what result
would falsify it. "Matched" means parameters within 5 percent and
training FLOPs within 15 percent, as enforced by the aggregator.

### H1. Recurrent latent reasoning helps independently of BDH memory

- Config: `configs/stage_a/a1_first_experiment.yaml`, extended by
  `configs/stage_a/a3_rtrain_sweep.yaml`.
- Compare: BDH + recurrence (R_train in {1,2,4}) vs BDH with matched
  non-recurrent compute (unshared depth chosen to match FLOPs at R=4) vs
  looped Transformer with identical recurrence schedule.
- Metric: exact match on compose and propagate, `mild` and `strong`
  splits, at R_test = R_train_max.
- Falsified if: the recurrent BDH does not beat matched fixed-depth BDH
  by more than the seed 95 percent interval on any extrapolation split,
  or the looped Transformer gains the same amount (then the gain is
  recurrence, not BDH).

### H2. BDH contextual memory provides capabilities beyond ordinary attention

- Config: `configs/stage_b/b1_binding_capacity.yaml`,
  `configs/stage_b/b2_memory_robustness.yaml`.
- Compare: BDH memory vs Transformer KV attention vs Gated DeltaNet, all
  at R=1, matched, on binding, overwrite, distractors, contradict.
- Metric: exact match vs number of bindings; stale rate on overwrite;
  distractor answer rate.
- Falsified if: capacity and robustness curves for BDH lie within the
  seed interval of Gated DeltaNet at every association count, or below
  Transformer attention.
- Explicit prediction from the no-decay memory: BDH stale rate on
  overwrite grows with `gap`; Gated DeltaNet's does not. If BDH shows no
  such effect, the memory is doing something we have not modelled, which
  is itself a finding.

### H3. Memory and recurrence interact positively (2x2)

- Config: `configs/stage_b/b3_2x2_interaction.yaml`.
- Cells: {KV context, BDH context} x {fixed depth, recurrent depth}, one
  block family (`unified_block`) so only those two variables change.
  Params matched by width solver; FLOPs matched by step count.
- Metric: exact match on compose (`mild`, `strong`) and binding (16, 32).
- Falsified if: the interaction term (BDH+recurrent minus the sum of the
  two main effects) is within the seed interval of zero.
- Top priority alongside H4. Five seeds from the start.

### H4. The model uses additional inference-time reasoning

- Config: `configs/stage_a/a1_first_experiment.yaml`,
  `configs/stage_a/a3_rtrain_sweep.yaml`.
- Train at R_train in {1,2,4}; evaluate R_test in {1,2,4,8,16,32,64}.
- Metric: accuracy vs R_test per split; classify each curve as
  improving, saturating, or degrading beyond R_train_max using the slope
  of the segment R > R_train_max (bootstrap over seeds).
- Falsified if: no model improves beyond R_train_max on any
  extrapolation split. Collapse beyond R_train_max is not a falsifier of
  the mechanism but a stability finding that redirects to Stage C.
- Negative controls: binding and copy (minimal steps = 1) must NOT
  improve with R; if they do, the improvement is not iterative
  computation.

### H5. A recurrence curriculum improves stability

- Config: `configs/stage_a/a2_curriculum.yaml`,
  `configs/stage_c/c2_curriculum_and_delay.yaml`.
- Compare: fixed R=8; curriculum 1 -> 2 -> 4 -> 8 at 0, 25, 50, 75
  percent of steps; delayed start (R=1 for the first 30 percent, then the
  curriculum). Parameter Golf records introduced recurrence partway
  through training in every winning depth-recurrence submission, which
  motivates the delayed-start arm.
- Metric: divergence rate across seeds, gradient-norm spikes, final
  accuracy at R_test in {8, 32}.
- Falsified if: curriculum arms do not reduce divergence or improve
  accuracy beyond the seed interval.

### H6. Small per-iteration degrees of freedom help shared recurrence

- Config: `configs/stage_c/c1_recurrence_engineering.yaml`
  (`recurrence.kind` in {step_gate, step_emb, adapter}, each with its
  matched plain control via `controls.matched_controls`).
- Compare: plain vs step_gate (R_max scalars) vs step_emb vs rank-4
  adapter, each against a plain control with the same extra parameter
  budget added to width.
- Metric: accuracy at R_test <= R_train_max and beyond; number of
  parameters added.
- Falsified if: no variant beats its matched control. Note the
  LegendreGPT record's finding that cheap per-layer scalars recovered
  most flexibility and low-rank adapters underperformed at matched bytes.

### H7. Initial-state skips reduce recurrent drift

- Config: `configs/stage_c/c1_recurrence_engineering.yaml`
  (`recurrence.kind` in {residual, init_skip, attn_residual}).
- Compare: plain, residual, init_skip (gated H[0] injection), and the
  community attention residual.
- Metric: cosine similarity between H[r] and H[0] projected onto the
  query subspace vs r; accuracy at R_test = 32 and 64.
- Falsified if: init_skip does not reduce the degradation slope beyond
  R_train_max relative to plain, within seed interval.

### H8. Recurrent-state precision matters

- Config: `configs/stage_d/d1_state_precision.yaml`.
- Compare: fp32, bf16, fp16 state dtype with fp32 accumulation, on the
  best Stage C configuration.
- Metric: accuracy vs R_test; per-iteration update norm; NaN incidence.
- Falsified if: the three curves coincide within the seed interval
  through R=64. Quantized weights are deferred; LoopQ (arXiv 2605.16343)
  reports looped models are unusually fragile to quantization because
  errors feed back through the loop, so this is a later stage.

## 4. Baselines

Priority order and how each is obtained:

1. BDH (`model.name: bdh`): adapter over the community `BDHBlock` and
   `BDH`, plus a `share_weights: false` path that instantiates `depth`
   blocks.
2. BDH-CQ (community) (`model.name: bdh_cq`): adapter over
   `BDHReasoningWrapper`, batched decoding added (numerically equal to
   sequential; tested).
3. Looped Transformer (`model.name: looped_transformer`): own ~200-line
   implementation, pre-norm block with RoPE and SwiGLU, shared across R,
   input injection of the embedded sequence at each step, same recurrence
   kinds as BDH. Sandwich norm as a config option (the Ouro reimplementation
   reports it as critical for recurrent stability).
4. Gated DeltaNet (`model.name: gated_deltanet`): `fla.layers.GatedDeltaNet`
   (flash-linear-attention 0.5.2, MIT) on GPU; a hand-written O(T)
   recurrent reference in pure PyTorch for CPU tests and for numerical
   parity checks against the Triton kernel at small sizes. Note the
   per-layer budget of about 6 * hidden^2 parameters constrains width at
   10M.
5. Transformer (`model.name: transformer`): same block as the looped
   Transformer with distinct weights per layer, R = 1.
6. Later (Stage E, `configs/stage_e/e1_ttt_control.yaml`):
   `model.name: transformer_ttt_lora`, Transformer + LoRA test-time
   training, protocol borrowed from Akyurek et al. (per-episode rank-8
   adapters, discarded after each episode), with adaptation latency,
   FLOPs, memory, interference, and reset cost recorded next to BDH fast
   memory on the same episodes.
7. H3 only (Stage B, `configs/stage_b/b3_2x2_interaction.yaml`):
   `model.name: unified_block`, one shared block family used for all four
   cells of the memory-kind x recurrence 2x2 so that memory kind and
   recurrence depth are the only variables changing (section 3, H3).

All share the `ReasoningModel` interface in `FRAMEWORK_SPEC.md` section
3. The interface adds no logic of its own; sequence models see the
serialized episode, BDH sees the same information through its native
ingestion path.

## 5. Task suite

Defined in `TASK_SUITE_SPEC.md`: binding, overwrite, distractors,
contradict, compose, propagate, copy, order, nested, each with `interp`,
`mild`, and `strong` splits. The community ARC-style families are kept as
`legacy_*` variants for continuity with Phase 0. Full ARC-AGI is out of
scope; no ARC evaluation data is used for tuning.

## 6. Metrics and diagnostics

Every run records the metadata and results schema of `FRAMEWORK_SPEC.md`
sections 4 and 5. Every comparison table shows params, train FLOPs,
R_train, R_test, inference FLOPs per episode, exact match mean and std
with n seeds, and split. Recurrent runs record per-iteration state norm,
update norm, cosine similarity between consecutive states, fraction of
active neurons, NaN counts, and a power-iteration estimate of the largest
Jacobian singular value at R in {8, 32}. The classification of each
recurrence curve (converges, cycles, diverges, collapses, keeps
computing) is derived from these series, not from accuracy alone.

## 7. Staged ablation matrix and ordering

Stage 0 (implementation, no GPU): framework, tasks, adapters, tests.
Exit: all tests green on CPU; `profile_config.py` runs on a tiny config.

Stage A (recurrence behaviour), ~10M params, RTX A5000 Community:

- A1 first experiment: {BDH, BDH-CQ (community, plain), looped
  Transformer} x {compose, propagate} x 3 seeds, R_train uniform over
  {1,2,4}, R_test 1..32. One plot: accuracy vs reasoning iterations.
- A2 curriculum repeat: adds the 1 -> 2 -> 4 schedule.
- A3 R_train sweep: R_train in {1,2,4,8}, R_test to 64, adds nested.
- Gate A.

Stage B (memory), ~10M params, R = 1 except B3:

- B1 capacity: {BDH, Gated DeltaNet, Transformer} on binding 1..64.
- B2 robustness: overwrite, distractors, contradict.
- B3 the 2x2 interaction (H3), 5 seeds.
- Gate B.

Stage C (recurrence engineering) on the best A configuration: plain,
residual, step_gate, init_skip, step_emb, rank-4 adapter, attention
residual (community), each against the same baseline with matched
controls; then curriculum and delayed start (C2). Combine only
mechanisms that won individually. Gate C.

Stage D (precision): fp32, bf16, fp16 state on the Stage C winner.
Gate D on the R_test > R_train question with deeper analysis if positive.

Stage E (optional, after Gate B): test-time adaptation control, BDH fast
memory vs LoRA TTT on binding and overwrite.

### Scale-up sweeps

Only survivors of Gate A and Gate B scale past 10M; nothing scales on a
comparison that did not clear both gates. Rule: five seeds per cell, same
matching and reporting conventions as Stages A-D.

- S1 params sweep: `configs/scaleup/s1_params_sweep.yaml`,
  `model.params_target` in {2_000_000, 5_000_000, 10_000_000, 25_000_000},
  five seeds, to check the trend holds below and around the Stage A-D
  scale before spending on larger runs.
- Then 50M, then 100-150M for the one or two comparisons that survived,
  five seeds, in later sweep files.

Compute per stage is estimated in `RUNPOD.md` section 8 and replaced by
profiling before launch. The whole initial programme (Stages A-D) is
about 195 GPU-hours on RTX A5000 Community, under $50 with contingency;
scale-up is budgeted separately after Gates A and B.

## 8. Matched controls and compute accounting

- Parameter matching: the width solver hits `params_target` within 3
  percent; realized counts are in every table.
- FLOP matching: analytic per-model estimates; fixed-depth cells in the
  2x2 get their step count adjusted so training FLOPs agree within 15
  percent, and both matching choices are recorded.
- Added-parameter mechanisms (gates, embeddings, adapters) get a plain
  control with the same extra budget in width.
- Inference FLOPs at each R_test are reported next to accuracy so that a
  10M model at R=32 is never described as "10M vs 10M".
- Hyperparameter effort: each baseline gets the same LR sweep
  {1e-4, 3e-4, 1e-3} at one seed on one task before its first sweep; the
  chosen LR and the sweep results are recorded in `RESULTS.md`. Nothing
  else is tuned per model.

## 9. Seed policy and statistics

- Dev: 1 seed, tagged `dev`, never cited.
- Promising: 3 seeds minimum.
- README claims: 5 seeds.
- Report mean, std, min, max, individual values, bootstrap 95 percent
  interval. Diverged and incomplete seeds stay in the table.
- A difference counts as credible only if the bootstrap intervals of the
  two arms do not overlap AND the effect exceeds 0.05 exact match on the
  relevant split. Parameter Golf's bar (p < 0.01 across seeds against a
  named baseline) is the model.
- Flags: NOT MATCHED, HIGH VAR, DIVERGED k/n, UNCONVERGED, and
  LR-EFFORT-DIFF when one arm received more tuning.

## 10. Decision gates and stopping criteria

- Gate A: does recurrence (R_test = R_train_max) produce a credible
  improvement over matched fixed depth on any extrapolation split for
  any model? If no, diagnose with the stability series before scaling
  (check loss convergence, check that the tasks need more than one step,
  check the loss target choice). Do not proceed to C until diagnosed.
- Gate B: does BDH memory differ from Gated DeltaNet or Transformer
  memory on capacity or robustness? If no, document that plainly in
  `RESULTS.md` and make Gated DeltaNet the memory reference for later
  stages.
- Gate C: does any recurrence modification improve extrapolation or
  stability beyond its matched control? Only winners are combined.
- Gate D: does accuracy keep improving for R_test > R_train_max? If yes,
  deeper analysis (per-episode convergence, fixed-point tests, the
  per-token convergence pattern reported in arXiv 2607.14427). If it
  collapses, investigate the state series before claiming anything about
  inference-time scaling.
- Stop a stage early when: the diagnosis says the tasks are solved at
  R=1 by every model (raise difficulty), or divergence exceeds 50 percent
  of seeds for a cell (fix stability before more seeds).
- Stop the project's scale-up if neither Gate A nor Gate B passes at
  10M and 25M.

## 11. Two highest-priority experiments

First: can BDH-CQ (community) learn an iterative algorithm that benefits
from more test-time loops than it saw in training? Config
`configs/stage_a/a1_first_experiment.yaml`. Deliverable: one plot,
accuracy vs reasoning iterations, for BDH, BDH-CQ (community), looped
Transformer, three seeds, compose and propagate, R_train <= 4, R_test to
32. Then `a2_curriculum.yaml` repeats it with the curriculum.

Second: is BDH contextual memory special? Config
`configs/stage_b/b1_binding_capacity.yaml` then
`b2_memory_robustness.yaml`. Deliverable: accuracy vs number of
associations for BDH, Gated DeltaNet, Transformer, then overwrite,
distractor, and contradiction curves.

## 12. Lessons imported from Parameter Golf and related work

- Depth recurrence over a small window of layers, activated partway
  through training, was a repeated leaderboard win (records of
  2026-04-04, 04-05, 04-09). This is the delayed-start arm of H5.
- Cheap per-layer scalars and gates recovered most of the flexibility
  lost to weight sharing at near-zero cost; low-rank per-layer adapters
  underperformed at matched budget (LegendreGPT record). This is H6.
- In the LoRA TTT record, most of the apparent test-time-compute gain
  came from evaluation-protocol fixes, not from TTT. Our evaluation
  protocol is fixed before any model is tuned, and every model sees
  identical cached episodes.
- Every accepted record reports 3-5 seed means with std and a
  significance test against a named baseline. Adopted as policy.
- Huginn (arXiv 2502.05171) trained with a random-depth distribution and
  input injection and reports both extrapolation beyond trained depth
  and an "overthinking" degradation. Fan et al. (arXiv 2409.15647) tie
  loop count to input difficulty. Both inform the H4 analysis.
- Depth-recurrent models converge to per-token fixed points nonuniformly
  (arXiv 2607.14427); our diagnostics test for that regime explicitly.

## 13. Failure modes this plan guards against

- Tuning BDH-CQ while leaving baselines untuned: same LR sweep for all.
- Chasing ARC-AGI: no ARC data; synthetic tasks only.
- Calling the community code the official architecture: labels carry
  "(community)"; gaps document is mandatory reading.
- Comparing by params only: inference FLOPs in every table.
- Changing several variables at once: one grid axis per sweep, tests
  refuse configs that differ from the base in more than the sweep axes.
- Hiding failed seeds: diverged status kept and counted.
- Huge sweeps without profiling: launcher refuses without estimates.
- Tuning on test: evaluation episodes are cached per task seed and never
  used for LR selection (LR selection uses a separate task seed).
- Claiming general reasoning from one task: Gate A requires at least two
  tasks and reports each separately.

## 14. Expected compute and hardware

Initial development on CPU (tiny configs, tests). Sweeps on RTX A5000 or
RTX 4090 Community Cloud, one job per GPU, seeds as separate jobs.
Planning estimate for Stages A-D at 10M: about 195 GPU-hours, under $50
at Community prices with contingency. Scale-up budgeted after gates.

## 15. Dependency pins (to be locked in `pyproject.toml` and `uv.lock`)

- torch 2.8.0 cu128 inside the RunPod image (provided by the base image);
  CPU wheels for local development. Note that torch >= 2.12 bundles CUDA
  13 on PyPI; keep the CUDA-12.8 path through the pytorch.org index.
- flash-linear-attention 0.5.2 (`[cuda]` extra) for Gated DeltaNet;
  hand-written CPU reference for tests.
- numpy 2.5.x, pyyaml 6.0.3, pydantic 2.13.x (config validation), pytest
  9.1.x, ruff 0.16.5 (exact pin), matplotlib 3.11.x, pandas 3.0.x.
- `bdh-cq` as a git dependency pinned to commit `c246f890`.
- `runpod` SDK 1.12.x, optional extra `[runpod]`.

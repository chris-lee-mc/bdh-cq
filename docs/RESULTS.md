# Results

Status: no GPU results yet; the only entries are the CPU dev runs of section
A0 and the Gate A diagnosis behind them. This file is the single place where
findings are recorded. Every entry must link a results directory, a config
hash, a git commit, and the number of seeds. Negative and inconclusive
results are recorded with the same care as positive ones.

Conventions:

- `n` = number of seeds. Rows with n < 3 are labelled `dev` and are not
  evidence. Rows with 3 <= n < 5 are `provisional`.
- Accuracy is exact match on the target unless stated. `+-` is the seed
  standard deviation; `[a, b]` is a 95 percent bootstrap interval of the
  mean over seeds.
- `R_train` / `R_test` are reasoning iterations. Inference FLOPs are
  analytic estimates from `bdhx/training/flops.py`.
- Every table states the split (`interp`, `mild`, `strong`).
- Flags: `NOT MATCHED` (params or train FLOPs differ beyond tolerance),
  `HIGH VAR` (std > 0.15), `DIVERGED k/n` (k seeds diverged),
  `UNCONVERGED` (train loss still falling at end), `AT_CHANCE` (final train
  loss still within 3 percent of ln(vocab_size); the run learned nothing and
  its accuracy columns carry no information).

## Phase 0: reproduction of the community implementation

See `docs/PHASE0_REPRODUCTION.md` for the record of the reproduction run
(command, seed, hardware, parameter count, curve, wall clock).

## Stage A: recurrence behaviour

Question: can BDH-CQ (community) (and BDH, looped Transformer) learn an
iterative algorithm that benefits from more test-time loops than it saw
in training?

Gate A finding: pending (no GPU sweep has been run). The Gate A *diagnosis*
of `EXPERIMENT_PLAN` section 10 was carried out ahead of the sweep because the
CPU dev runs were flat at chance; see section A0 for the three framework
defects it found and the two model-scale limits it did not.

### A0. CPU pipeline validation (dev, not evidence)

Config: `configs/stage_a/a1_cpu_mini.yaml` (expanded with
`tools/generate_sweep.py --dev` into 9 jobs: 3 models x 3 seeds, task
`compose`, train difficulties depth 1-2, eval `interp` depth 1-2 / `mild`
depth 3-4 / `strong` depth 6-8, 100 eval episodes per split). Results:
`results/a1_cpu_mini/`, report: `reports/a1_cpu_mini/`, committed copies of
the plot, its backing CSV, `summary.csv` and `flags.csv` in
`docs/results/a1_cpu_mini/`.

This is a pipeline test, not an experiment. It is tagged `[dev, cpu_mini]`
and every row is flagged `DEV`.

#### What was wrong in the first version of this section

The first version of A0 (9 jobs at 205k parameters, 6000 steps) reported
exact match 0.000 in every cell and read that as "small models on CPU should
not learn compose". Two of the three reasons were defects in the framework,
not properties of the models:

1. **Init.** Every sequence-native model ties its unembedding to
   `nn.Embedding`, which torch initializes at N(0, 1). With a tied head the
   embedding sets the logit scale, so the initial logits were O(sqrt(width))
   and the initial cross-entropy was 219 nats for the looped Transformer at
   width 222 (44 nats at the cpu_mini width of 44) instead of
   ln(4128) = 8.33. The whole step budget went into walking back down to
   chance. Fixed: the embedding is initialized at std 0.02 and
   `SeqReasoner.embed_tokens` passes it through a parameter-free RMSNorm (the
   community BDH's `post_embed_norm`), so the residual stream starts at unit
   RMS whatever the init. Init loss is now within 0.5 nats of ln(vocab) for
   all six registered models, pinned by
   `tests/test_learnability.py::test_init_loss_is_near_ln_vocab`.
2. **Effective depth.** `configs/base/default.yaml` had `model.depth: 1`, and
   `looped_transformer` ignored the field entirely and always built a
   one-layer shared block. A single layer applied R times cannot express an
   induction-style match-and-copy no matter how large R is. Fixed:
   `model.depth` now means "layers applied per reasoning step" for every model
   (FRAMEWORK_SPEC section 2), the looped models build a `depth`-layer shared
   stack (plus optional `prelude`/`coda` layers), and the default is 2.
3. **Nothing checked for it.** A run at chance wrote a perfectly valid
   `results.json` and the aggregator reported it as 0.000 exact match with no
   warning. Fixed: `aggregate.py` raises `AT_CHANCE` when the final training
   loss is still within 3 percent of ln(vocab_size), and
   `tools/sanity_learnability.py` is a mandatory pre-sweep gate
   (`HANDOFF_TASKS.md` task 23b).

Three hypotheses were checked and cleared: the loss masking and target
alignment are correct (an oracle solver that copies the value following the
query key scores exact match 1.000 through `evaluate.py` on every split, and
its `final_answer` loss equals its logit margin, not ln(vocab) -
`tests/test_learnability.py`); the learning rate follows warmup-then-cosine in
`train_log.csv` and every parameter, embedding included, receives a non-zero
gradient; and the training batches are fresh per step, reproducible per
(seed, step), with the query key present in the demonstrations and
`answer_start` on the [ANSWER] token in all sampled batches.

#### The re-run

Setup: `params_target` 350_000, `model.depth` 2, batch size 16, 3000 steps,
warmup 300, lr 3.0e-4 (the `default.yaml` value; no a0 LR sweep has been run,
so the LR is untuned), `R_train` sampled uniformly from {1, 2, 4}, `R_test` in
{1, 2, 4, 8, 16}, `compute.device: cpu`, `deterministic: true`, seeds 1, 2, 3.
350_000 rather than 205_000 because the BDH width solver only accepts a coarse
grid at vocab 4128 and 350_000 is the next target all three models hit within
0.2 percent; 3000 rather than 6000 steps because depth 2 costs about five
times more per step.

| model | width | params realized | off target | steps | wall clock per job | final train loss |
|-------|-------|-----------------|-----------|-------|--------------------|------------------|
| bdh | 40 | 349,440 | -0.16% | 3000 | 103-123 s | 8.40 |
| bdh_cq | 40 | 349,440 | -0.16% | 3000 | 167-172 s | 8.38 |
| looped_transformer | 62 | 349,494 | -0.14% | 3000 | 146-160 s | 8.33 |

Total sweep wall clock 1299 s (21.6 min) for 9 jobs run sequentially on 4 CPU
cores. All 9 jobs finished with `status: ok`, 0 NaN events, 0 preemptions.

Exact match, mean +- seed std over 3 seeds, the two difficulties of each split
pooled:

| model | split | R=1 | R=2 | R=4 | R=8 | R=16 |
|-------|-------|-----|-----|-----|-----|------|
| bdh | interp | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| bdh | mild | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| bdh | strong | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| bdh_cq | interp | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| bdh_cq | mild | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| bdh_cq | strong | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| looped_transformer | interp | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| looped_transformer | mild | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| looped_transformer | strong | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |

`bdh` is the fixed-depth reference: its adapter ignores `R`, so its row is the
same number repeated across R by construction, not a flat curve it earned.

Flags raised (`docs/results/a1_cpu_mini/flags.csv`): `PROVISIONAL` (n_seeds=3
< 5) and `DEV` for all three arms, and now `AT_CHANCE` for all 9 runs (final
train loss 8.27 to 8.45 against ln(4128) = 8.33). No `DIVERGED`,
`UNCONVERGED` or `NOT MATCHED` flags: the three arms are matched to within
0.16 percent of parameters.

Interpretation, limited to pipeline validity: the pipeline works end to end
and, on `compose` at this budget, still learns nothing - and now says so. The
zeros are the same as before the fixes, but their meaning is different: the
runs are labelled `AT_CHANCE` by the aggregator rather than silently reported
as accuracy 0.000, and the two defects that would have kept the models at
chance at *any* budget are gone. `compose` at depth 1-2 is a multi-hop
composition over 4128 fresh symbols per episode; nothing at 350k parameters
and 3000 steps of an untuned LR was expected to solve it, and no claim about
recurrence, extrapolation or model ranking can be drawn from this table. None
is made.

#### Evidence that the pipeline now learns

The learnability check moved to `binding`, where the answer is a single token
present verbatim in the context, so a working pipeline must solve it.
`tools/sanity_learnability.py` (1.5M parameters, batch 32, 3000 steps,
lr 1.0e-3, train difficulties n_bindings 1 and 2):

| model | depth | params | R_train | interp n_bindings=1 | final train loss | wall clock | before the fixes |
|-------|-------|--------|---------|---------------------|------------------|-----------|------------------|
| transformer | 2 | 1,493,242 | 1 | 1.000 at R=1 | 0.60 | 171 s | 0.000, loss 8.36 |
| looped_transformer | 2 | 1,493,242 | {1, 2} | 1.000 at R=1 and R=2 | 0.81 | 216 s | 0.000, loss 8.36 |

The two arms solve to the same width and the same parameter count: a looped
model at `depth: 2` and a fixed-depth Transformer at `depth: 2` hold the same
two blocks, so the only difference is weight sharing across reasoning steps.

The same two configurations were flat at exactly ln(4128) before the fixes,
with the looped Transformer starting from a training loss of 219.

`n_bindings >= 2` is *not* solved by any model here (0.48 to 0.70 for the
Transformer family, against 0.50 for guessing between the two demonstrated
values), and that is a property of the serialization rather than a remaining
defect. `TASK_SUITE_SPEC` section 1 puts an [ANSWER] marker between the query
and the target, so the answer is predicted from a position whose own token
carries no information about the query: the model must copy the key forward
into each value position, copy the query forward into the [ANSWER] position,
and only then match. A standalone 2-layer reference implementation outside
this framework reproduces the effect exactly and isolates it to that one
token: with the readout at the query token (`k1 v1 k2 v2 q`) it reaches 0.975
exact match in 3000 steps, and appending a single constant token
(`k1 v1 k2 v2 q [ANSWER]`) drops it to 0.445. Depth 2, 3, 4 and 6, one to
eight attention heads, QKV biases, learned absolute position embeddings, an
untied head, an auxiliary next-token loss over the whole prompt, and 12000
steps instead of 3000 all leave it at chance-between-the-candidates; only
moving the query to the readout position fixes it. This is a real limit of
these models at this scale, not a broken pipeline, and it is why the sanity
gate is set on n_bindings=1.

#### BDH and BDH-CQ on the same check (dev, not evidence)

Same recipe as the table above (binding, 1.5M parameters, width 152, batch 32,
3000 steps, lr 1.0e-3, train difficulties n_bindings 1 and 2, one seed).
`recurrence.share_weights` is true, so `depth` costs no parameters and all four
BDH cells are matched exactly. `bdh` ignores R by construction.

| model | depth | R_train | final train loss | n_bindings=1, R=1 | R=2 | R=4 | n_bindings=2, R=1 | train wall clock |
|-------|-------|---------|------------------|-------------------|-----|-----|-------------------|------------------|
| bdh | 2 | {1} | 5.02 | 0.600 | - | - | 0.300 | 369 s |
| bdh | 4 | {1} | 5.20 | **0.700** | - | - | 0.220 | 350 s |
| bdh_cq | 2 | {1, 2} | 5.66 | 0.360 | 0.400 | 0.040 | 0.200 | 385 s |
| bdh_cq | 4 | {1, 2} | 7.83 | 0.000 | 0.000 | 0.000 | 0.000 | 598 s |
| bdh_cq (`loss: legacy`) | 2 | {1, 2} | 5.55 | 0.280 | 0.280 | 0.000 | 0.260 | 396 s |

Four observations, all one seed and none of them evidence:

1. BDH does learn. Both `bdh` rows leave the chance plateau decisively (5.0 to
   5.2 against ln(4128) = 8.33) and reach 0.60 to 0.70 exact match on the
   one-binding cell, so the earlier all-zero A0 table was the framework, not
   the architecture. The best BDH cell is `bdh` at depth 4, 0.700.
2. BDH is well behind the Transformer family here (1.000 for both
   `transformer` and `looped_transformer` at the same parameter count and step
   budget). No architectural change was made to chase this.
3. The latent loop does not pay for itself at this scale. `bdh_cq` is worse
   than plain `bdh` at every matched setting, and `bdh_cq` at depth 4 barely
   trains at all (loss 7.83, exact match 0.000): 4 block applications inside
   each of up to 2 latent steps is 8 applications of one shared block through
   a Hebbian memory that is also being written at every stage. Accuracy also
   collapses at R=4, one step beyond the largest R seen in training - the
   "overthinking" degradation Huginn reports, here total rather than gradual.
4. Divergence from community usage is real but is not the explanation. The
   community `figure7.py` trains at lr 1e-3 with batch 1, `depth: 4`,
   `dim_qk_heads` 4x to 5.3x `dim` (we use 4x), and its loss adds a next-token
   term over the whole prompt on top of the answer loss
   (`icq.train_loss`), with class weights over a 14-token vocabulary. We match
   the learning rate and the neuron ratio; we differ in batch size (32),
   vocabulary (4128 symbols, so class weights are meaningless), and the loss.
   The loss difference is available as `training.loss: legacy`, which is the
   community path through `BDHReasoningWrapper(..., return_loss=True)`, and the
   last row shows it does not rescue the model (0.280 against 0.360). The
   remaining candidate is the vocabulary: `figure7.py` asks the Hebbian readout
   to separate 14 tokens, this task asks it to separate 4096 fresh symbols per
   episode at width 152, which is the regime where a linear-attention memory
   read should be weakest. That is a hypothesis for Stage B, not a finding.

### A1. First high-priority experiment (compose, propagate; ~5-10M params)

| model | params | R_train | task | split | R_test=1 | 2 | 4 | 8 | 16 | 32 | n | flags |
|-------|--------|---------|------|-------|----------|---|---|---|----|----|---|-------|
| pending |

Plot: `reports/<date>/acc_vs_reasoning_steps_compose.png` (pending).

### A2. Recurrence curriculum repeat

Pending.

### Stage A findings

Pending. Template: what was attempted, what worked, what failed,
confidence, compute spent (GPU-hours, USD), next experiment and why.

## Stage B: memory mechanisms

Question: is BDH contextual memory special relative to Gated DeltaNet and
Transformer context on arbitrary, non-memorizable bindings?

Gate B finding: pending.

### B1. Capacity curve (binding task, 1..64 associations)

| model | params | n_bindings=1 | 2 | 4 | 8 | 16 | 32 | 64 | n | flags |
|-------|--------|--------------|---|---|---|----|----|----|---|-------|
| pending |

### B2. Overwrite, distractors, contradictions

Pending.

### Stage B findings

Pending.

## Stage C: recurrence engineering

Gate C finding: pending.

## Stage D: precision

Gate D finding: pending.

## Compute ledger

| date | stage | sweep | GPU | GPU-hours | USD (est.) | notes |
|------|-------|-------|-----|-----------|------------|-------|
| 2026-09-03 | A | a1_cpu_mini (first version, depth 1, N(0,1) tied head) | none (4 CPU cores) | 0.0 | 0.00 | 9 dev jobs, 1436 s wall clock; superseded, the runs were AT_CHANCE by construction |
| 2026-09-03 | A | a1_cpu_mini (re-run, depth 2, fixed init) | none (4 CPU cores) | 0.0 | 0.00 | 9 dev jobs, 1299 s wall clock total; pipeline validation only, not evidence; all 9 AT_CHANCE on compose |
| 2026-09-03 | A | Gate A diagnosis (binding, sanity_learnability + BDH acceptance runs) | none (4 CPU cores) | 0.0 | 0.00 | about 20 CPU jobs of 3000 steps each plus a standalone reference reproduction; see section A0 |

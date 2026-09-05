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

Gate A finding: **yes, conditionally.** BDH-CQ (community) at R_test =
R_train_max = 4 produces a large, credible improvement over the matched
fixed-depth baseline on the `propagate` task's `mild` extrapolation split
(0.579 vs 0.017 exact match, non-overlapping 95 percent CIs, gap far past the
0.05 threshold). The same model collapses to exact match 0.000 the moment
R_test exceeds R_train_max (R=8, 16, 32), so this is not test-time compute
scaling (Gate D fails outright for BDH-CQ) -- it is a real but narrow win
that only holds exactly at the trained R. No model shows a credible
improvement on `compose`, where BDH-CQ and the looped Transformer mostly sit
at or below the non-recurrent baseline (`looped_transformer` never left the
loss chance plateau on `compose` in any seed). See A1 below for the full
breakdown. The Gate A *diagnosis* of `EXPERIMENT_PLAN` section 10 was also
carried out ahead of the sweep because the CPU dev runs were flat at chance;
see section A0 for the three framework defects it found and the two
model-scale limits it did not.

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

#### The fourth defect: the readout position (found here, fixed)

`n_bindings >= 2` was *not* solved by any model in the run above (0.48 to
0.70 for the Transformer family, against 0.50 for guessing between the two
demonstrated values). That turned out to be a property of the serialization
rather than a limit of the models. `TASK_SUITE_SPEC` section 1 puts an
[ANSWER] marker between the query and the target, so the answer was predicted
from a position whose own token carries no information about the query: the
model had to copy the key forward into each value position, copy the query
forward into the [ANSWER] position, and only then match. A standalone 2-layer
reference implementation outside this framework reproduced the effect exactly
and isolated it to that one token: with the readout at the query token
(`k1 v1 k2 v2 q`) it reaches 0.975 exact match in 3000 steps, and appending a
single constant token (`k1 v1 k2 v2 q [ANSWER]`) drops it to 0.445. Depth 2,
3, 4 and 6, one to eight attention heads, QKV biases, learned absolute
position embeddings, an untied head, an auxiliary next-token loss over the
whole prompt, and 12000 steps instead of 3000 all left it at
chance-between-the-candidates; only moving the query to the readout position
fixed it.

Fixed: `SeqReasoner.forward_episode`/`solve` now read the first target token
from the hidden state at the last query token (`answer_start - 2`) instead of
at the [ANSWER] token. [ANSWER] stays in `serialize()`/`parse_serialized()` as
a structural delimiter, so both stay lossless and multi-token targets stay
unambiguous, and every later target token still reads from the previous real
target token because only the first hop paid the marker's cost. This touches
`transformer`, `looped_transformer`, `unified_block` and `gated_deltanet`;
`bdh` and `bdh_cq` feed the query through BDH's native ingestion with no
[ANSWER] token involved and are unchanged. In the same one-seed sanity run,
n_bindings=2 moved from 0.20-0.36 to 0.48-1.00. The sanity gate stays set on
n_bindings=1, which both gated models now solve at exact match 1.000.

The measurement above still stands as the reason the fix exists, and the
absolute accuracies in the A0 tables were produced before it: they are
floored by the extra hop and should not be compared against anything measured
after this commit.

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

### A1. First high-priority experiment (compose, propagate; ~10M params)

Config: `configs/stage_a/a1_first_experiment.yaml`, expanded by
`tools/generate_sweep.py` into 18 jobs (`bdh`, `bdh_cq` (community),
`looped_transformer` x `compose`, `propagate` x seeds 1, 2, 3). All models
matched to ~10M trainable parameters (`bdh`/`bdh_cq` 10,010,880;
`looped_transformer` 9,965,316, 0.45 percent off, within the 5 percent
matching tolerance). `R_train` sampled uniformly from `{1, 2, 4}` during
training; evaluated at `R_test` in `{1, 2, 4, 8, 16, 32}`. `bdh` is the
non-recurrent, fixed-depth control: it ignores `R` by construction, so its
row is flat across all `R_test` columns and stands in for "the same params,
no extra loop." Commit `2476009`. Hardware: RunPod, RTX 4090, Secure Cloud
(see the compute ledger for why Secure and not Community). Full per-seed
data in `results/<run_id>/`; aggregated tables and all plots in
`reports/a1_first_experiment/`; the headline plot and tables copied to
`docs/results/a1/`.

Train FLOPs (`bdhx/training/flops.py` estimate, from `metadata.json`,
seed 1 of each cell): `bdh` compose 1.077e16 / propagate 3.042e16;
`bdh_cq` compose 1.134e16 / propagate 3.100e16; `looped_transformer`
compose 1.222e16 / propagate 3.346e16. Inference FLOPs per episode at
`R_test = 4`: `bdh` compose 7.92e8 / propagate 3.96e9; `bdh_cq` compose
9.19e8 / propagate 4.09e9; `looped_transformer` compose 1.52e9 / propagate
7.10e9.

Exact match, mean over n=3 seeds, averaged over the 2 difficulty buckets
inside each split (full per-difficulty numbers in
`docs/results/a1/summary.csv`; 0 diverged seeds throughout):

**compose** (params ~10M, R_train sampled from {1,2,4}):

| model | split | R=1 | R=2 | R=4 | R=8 | R=16 | R=32 | flags |
|-------|-------|-----|-----|-----|-----|------|------|-------|
| bdh | interp | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | - |
| bdh | mild | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | - |
| bdh | strong | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | - |
| bdh_cq | interp | 0.086 | 0.087 | 0.084 | 0.078 | 0.006 | 0.000 | HIGH VAR |
| bdh_cq | mild | 0.040 | 0.042 | 0.043 | 0.032 | 0.006 | 0.001 | HIGH VAR, AT_CHANCE (1/3 seeds) |
| bdh_cq | strong | 0.016 | 0.018 | 0.018 | 0.015 | 0.004 | 0.001 | HIGH VAR |
| looped_transformer | interp | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | AT_CHANCE (3/3 seeds) |
| looped_transformer | mild | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | AT_CHANCE (3/3 seeds) |
| looped_transformer | strong | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | AT_CHANCE (3/3 seeds) |

**propagate** (params ~10M, R_train sampled from {1,2,4}):

| model | split | R=1 | R=2 | R=4 | R=8 | R=16 | R=32 | flags |
|-------|-------|-----|-----|-----|-----|------|------|-------|
| bdh | interp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| bdh | mild | 0.017 | 0.017 | 0.017 | 0.017 | 0.017 | 0.017 | HIGH VAR |
| bdh | strong | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| bdh_cq | interp | 0.998 | 0.995 | 0.997 | 0.000 | 0.000 | 0.000 | - |
| bdh_cq | mild | 0.471 | 0.276 | **0.579** | 0.000 | 0.000 | 0.000 | HIGH VAR |
| bdh_cq | strong | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| looped_transformer | interp | 1.000 | 1.000 | 1.000 | 0.973 | 0.811 | 0.466 | HIGH VAR (R>=16) |
| looped_transformer | mild | 0.003 | 0.018 | 0.029 | 0.038 | 0.055 | 0.019 | HIGH VAR |
| looped_transformer | strong | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |

Plots: `docs/results/a1/acc_vs_reasoning_steps_compose.png`,
`docs/results/a1/acc_vs_reasoning_steps_propagate.png` (all 6 FRAMEWORK_SPEC
section 10 plot categories are in `reports/a1_first_experiment/`, not copied
here to keep the repo small).

**The Gate A comparison** (recurrence at `R_test = R_train_max = 4` vs the
matched fixed-depth `bdh` baseline, on the two `mild`/`propagate` difficulty
buckets, per-difficulty rather than averaged):

| difficulty | model | exact match | 95 percent CI | credible? |
|------------|-------|--------------|----------------|-----------|
| distance=6 | bdh | 0.033 | [0.000, 0.062] | - |
| distance=6 | bdh_cq | 0.972 | [0.946, 0.992] | **yes** (gap 0.939, CIs disjoint) |
| distance=8 | bdh | 0.001 | [0.000, 0.004] | - |
| distance=8 | bdh_cq | 0.185 | [0.028, 0.316] | **yes** (gap 0.184, CIs disjoint) |
| distance=6 | looped_transformer, R=4 | 0.057 | [0.012, 0.122] | no (gap 0.024 < 0.05) |
| distance=6 | looped_transformer, R=16 | 0.110 | [0.052, 0.206] | no (gap 0.077 but CIs overlap) |

`looped_transformer`'s R=16 number on `propagate`/mild/distance=6 is the
closest thing to genuine test-time-compute scaling in this sweep (monotonic
improvement from R=1 through R=16, well past R_train_max=4, before falling
back at R=32) but at only 3 seeds it does not clear the bar this project set
for "credible": the raw gap passes 0.05 but the bootstrap CIs still overlap.
Worth a `[1,2,3,4,5]`-seed repeat before it is called a finding either way.

**Diagnostics and stability** (`state_norm`, `cos_consecutive`,
`update_norm` vs iteration; full plots in `reports/a1_first_experiment/`):

- `bdh_cq`'s hidden state does not blow up or decay past R_train_max: state
  norm on `propagate` is flat at ~25.14 from iteration 3 through 32, and
  `cos_consecutive` reaches ~0.95-1.0 by iteration ~3 and stays there. The
  0.579 to 0.000 collapse from R=4 to R=8 is therefore not a numerical
  instability -- the recurrent state itself is well-behaved and near a fixed
  point at every R tested. This looks like the same total, discontinuous
  "overthinking" collapse the A0 section already reported at 1.5M
  parameters, now confirmed at 10M and with real seeds: the readout has
  learned to work with states from R in {1,2,4} specifically, not with
  "however many iterations the state has been through."
- `looped_transformer`'s convergence is smoother and reaches
  `cos_consecutive` ~1.0 faster with less seed-to-seed noise than `bdh_cq`'s,
  consistent with its own accuracy degrading gradually (1.000 to 0.466 from
  R=4 to R=32 on `propagate`/interp) rather than collapsing outright.
- `looped_transformer` on `compose`: `AT_CHANCE` fired for all 3 seeds
  (final train loss 8.28-8.33 against ln(vocab)=8.326) -- it never left the
  chance plateau. Per `EXPERIMENT_PLAN` section 10's own Gate A diagnosis
  procedure ("check that the tasks need more than one step, check the loss
  target choice") this needs a `compose`-specific pipeline check before its
  0.000 exact match on `compose` is read as "the architecture can't do
  this" rather than "this run never trained."
- `bdh_cq` on `compose`: 1 of 3 seeds also hit `AT_CHANCE` (loss 8.147);
  the other two partially learned it. `compose` at 10M params is close to
  or below the learnability floor for two of the three models in this
  sweep, which is itself a Gate A-relevant flag (`EXPERIMENT_PLAN`
  section 10: "raise difficulty" is the wrong direction here -- this task
  needs to be made easier, or these models need more params, before
  `compose` can say anything about recurrence).

**Every flag raised** (from `docs/results/a1/flags.csv`, 333 rows total):
`HIGH VAR` (std/mean > 0.3) on 323 of the 396 summary cells -- expected and
not itself concerning, since most cells sit at exact match near 0 where the
coefficient of variation is naturally large; `PROVISIONAL` on all 6
model x task cells (n=3 < the 5-seed README-grade bar, by design for a
first pass); `AT_CHANCE` on `looped_transformer`/compose (3/3 seeds) and
`bdh_cq`/compose (1/3 seeds), discussed above. No `NOT MATCHED` (params
within tolerance) and no `DIVERGED` seeds.

### A2. Recurrence curriculum repeat

Pending. Not started: Gate A passed conditionally on `propagate`/mild for
`bdh_cq`, which under `EXPERIMENT_PLAN` section 10 is grounds to proceed,
but the collapse at `R_test > R_train_max` and the `compose` learnability
flags above are worth resolving first (see Stage A findings).

### Stage A findings

**What was attempted:** the full A1 grid (`bdh`, `bdh_cq` (community),
`looped_transformer` x `compose`, `propagate` x 3 seeds, ~10M params,
matched training FLOPs) on real RunPod GPU hardware, plus everything needed
to get 18 jobs there and back: a launcher built from scratch this project
(`tools/runpod_launch.py`), an HTTP-proxy-based collection path (the
sandbox blocks outbound SSH/scp), and periodic monitoring across several
real infrastructure failures.

**What worked:** `bdh_cq` (community) shows a large, credible improvement
over the matched fixed-depth baseline at `R_test = R_train_max` on
`propagate`'s `mild` split (Gate A finding above) -- the first positive
recurrence result this project has produced with real seeds. Diagnostics
confirm the mechanism collapsing past `R_train_max` is a readout
generalization failure, not a numerical instability, which narrows what A2
needs to fix.

**What failed:** `compose` did not produce a credible improvement for any
model, and two of three models did not reliably learn it at all at this
budget (`AT_CHANCE` flags above). `bdh_cq`'s `propagate` win does not extend
past `R_train_max` -- Gate D fails for this model on this task. Getting
here also surfaced (and fixed, each with a regression test) six real bugs
that a live GPU sweep is apparently required to find: two crashes in
existing framework code (`diagnostics.py`'s `torch.quantile` device
mismatch; `Trainer`'s `resume=True` raising on a first launch) and four in
the launcher and results pipeline built for this sweep (unescaped
`docker_args`, a double-nested result directory, a `git checkout` of a
branch name failing silently on a shallow clone, and duplicate evaluation
rows inflating `n_seeds` after a job got relaunched post-completion). None
of the six were caught by the existing CPU-only test suite; all six were
only found by actually running the sweep.

**Confidence:** provisional. 3 seeds per cell, as flagged. The `propagate`
Gate A finding has disjoint bootstrap CIs and a large effect size, which is
about as strong as a 3-seed result gets, but a `[1,2,3,4,5]`-seed repeat is
the right bar before treating it as settled, per this file's own
conventions.

**Compute spent:** see the compute ledger below (~72 GPU-hours estimated,
~$54 on RTX 4090 Secure Cloud, over the ~$25 gate approved with explicit
sign-off after Community Cloud proved unreliable for this sweep).

**Next experiment and why:** two candidates, not yet started. (1) A2
(recurrence curriculum repeat) as originally planned, now informed by the
`R_test > R_train_max` collapse: worth trying a curriculum that includes at
least one training R beyond what A1 used, to see whether the readout
generalization failure is a curriculum artifact rather than an
architectural ceiling. (2) A `compose`-specific pipeline check (per the
`AT_CHANCE` diagnosis above) before spending more GPU-hours on `compose` at
this scale.

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
| 2026-09-04/05 | A | a1_first_experiment (18 jobs: 3 models x 2 tasks x 3 seeds, ~10M params) | RTX 4090, Secure Cloud | ~72 | ~54 | Estimate = sum of `generated/a1_first_experiment/manifest.csv`'s profiled per-job minutes (49.07 min/job bdh, 364.10 bdh_cq, 306.62 looped_transformer; 71.98 GPU-hours) x $0.74/hr (`configs/runpod_rates.yaml`, RTX 4090 Secure). This undercounts real elapsed wall clock: Community Cloud failed to boot repeatedly (5+ times) before the sweep moved to Secure, one job (`76dd99c62a3e_s1`) alone cycled through 11 distinct pod attempts, and several jobs were relaunched after already reaching their final checkpoint (a bug found and fixed mid-sweep, see Stage A findings) -- but a pod that never boots shows `uptimeSeconds=0` and is not believed to be billed, so those retries are assumed near-$0 rather than added on top. Exceeded the $25 cost gate approved for this sweep; re-approved by explicit user sign-off at ~$32 estimated before the Community-to-Secure switch (which itself raised the per-hour rate from $0.34 to $0.74), so the real total landed higher still. All pods reaped; `runpod status` and a direct `get_pods()` check both confirmed zero pods remaining on the account at sweep end. |

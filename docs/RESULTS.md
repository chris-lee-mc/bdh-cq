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

**What is under test, and what is not.** The latent transition function --
the per-step update this whole stage measures -- is not public. The BDH-CQ
paper (arXiv 2608.09888) gives only its signature, `H_{r+1} = F_theta(H_r,
S_K)` (eq. 3), and states that "dimensions, exact update rules, and
implementation details remain proprietary". Verified against the paper's
full text on 2026-09-07, not a secondary summary. The community author says
the same in his own code: "sans knowing their secretive latent transition
function" (`bdh_cq.py:412`). So every result in Stage A is a result about
`lucidrains/bdh-cq`'s reconstruction at `c246f890`, not about Pathway's
model, and no negative result here can be read as a refutation of the paper.
`docs/PAPER_IMPLEMENTATION_GAPS.md` section 3 is the standing rule; this
paragraph is where the rule gets applied.

Two consequences worth stating once, because A4 and A5 are the results most
likely to be over-read:

- The paper never claims extrapolation past the trained step count. It
  reports scaling *inference* effort (LOW/MEDIUM/HIGH, 21 to 29.5 percent
  pass@2 on ARC-AGI-1) but does not say whether those levels were trained,
  and does not test R_test > R_train_max. This project's Gate D failure
  therefore contradicts nothing the paper asserts.
- The comparison that survives all of this is the internal one. Under an
  identical harness, task, parameter budget and training FLOPs,
  `looped_transformer` does extrapolate and this loop does not. That is a
  sound statement about the two designs actually run, and it does not
  depend on either being the paper's.

Gate A finding: **yes, conditionally.** BDH-CQ (community) at R_test =
R_train_max = 4 produces a large, credible improvement over the matched
fixed-depth baseline on the `propagate` task's `mild` extrapolation split
(0.579 vs 0.017 exact match, non-overlapping 95 percent CIs, gap far past the
0.05 threshold). Past R_train_max the same model falls apart: exact match is
0.000 at R = 8, 16 and 32, so Gate D fails outright for BDH-CQ. See A1 below
for the full breakdown, and A1a/A1b for the two follow-up diagnoses that
say *why* each half of that sentence happened. `compose` is excluded from the
Gate A comparison entirely: section A1a shows its generator made 71 percent of
training episodes unanswerable, capping Bayes-optimal exact match at 0.10 on
`mild` and 0.03 on `strong`, so no model on `compose` could have cleared the
0.05 credibility bar whatever it learned.

Two claims in the first version of this section were wrong and are corrected
below: the fall past R_train_max is *graded*, not a discontinuous collapse
(exact match over a 12-token target hides a token accuracy that falls 0.95 ->
0.60 -> 0.46, section A1b), and it is *not* a readout failure over a
well-behaved state -- BDH-CQ's latent loop never reaches a fixed point at all,
while the looped Transformer's does (section A1b). The Gate A *diagnosis* of
`EXPERIMENT_PLAN` section 10 was also carried out ahead of the sweep because
the CPU dev runs were flat at chance; see section A0 for the three framework
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

- ~~`bdh_cq`'s hidden state does not blow up or decay past R_train_max: state
  norm on `propagate` is flat at ~25.14 from iteration 3 through 32, and
  `cos_consecutive` reaches ~0.95-1.0 by iteration ~3 and stays there. The
  recurrent state is well-behaved and near a fixed point at every R tested,
  so the collapse is a readout generalization failure.~~ **Both halves of
  this were wrong; see A1b.** The flat state norm is an artifact: the
  community BDH ends every block with a parameter-free LayerNorm, so
  `||H||` is pinned at sqrt(dim) by construction and cannot diagnose
  anything. And the `cos_consecutive` reading came from
  `cos_consecutive_vs_iteration_bdh_cq_propagate.png`, which at the time
  pooled every checkpoint, split, difficulty and R_test into one unlabelled
  overlay; the converging curves in it belong to cells where the model had
  already given up. On the cells that carry the Gate A result,
  `cos_consecutive` peaks at 0.79-0.95 and then *falls* as R grows.
- `looped_transformer`'s accuracy degrades gradually (1.000 to 0.466 from
  R=4 to R=32 on `propagate`/interp) rather than collapsing outright.
  Section A1b ties that to the difference that matters: it converges to a
  fixed point and `bdh_cq` does not.
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

### A1a. Why `compose` said nothing: the task was mostly unanswerable

Follow-up to the `AT_CHANCE` flags above, run on CPU at zero GPU cost.
Reproduce with `python tools/task_ceiling.py --task compose`.

`compose` demonstrates `n_examples_per_fn = 4` input/output pairs for each of
its `d` bijections, drawn **with replacement** from a domain of 8, and then
draws the query independently. Nothing tied the two together, so the query's
chain was demonstrated only by luck: `(1 - (1 - 1/8)^4)^d`, which is 0.41 at
depth 1 and 0.0009 at depth 8. Worse, when the last hop is undemonstrated the
answer is a symbol that appears **nowhere in the episode** -- symbols are
remapped per episode, so it is not merely unidentifiable, it is unreachable.

Measured over 2000 episodes per depth against the pre-fix generator:

| depth | split | oracle solvable | target absent from the episode | Bayes-optimal exact match | Bayes-optimal loss (nats) |
|-------|-------|-----------------|-------------------------------|---------------------------|---------------------------|
| 1 | train, interp | 0.408 | 0.592 | 0.408 | 4.92 |
| 2 | train, interp | 0.179 | 0.585 | 0.223 | 6.83 |
| 3 | mild | 0.078 | 0.584 | 0.125 | 7.66 |
| 4 | mild | 0.035 | 0.589 | 0.076 | 8.03 |
| 6 | strong | 0.004 | 0.601 | 0.035 | 8.28 |
| 8 | strong | 0.000 | 0.557 | 0.026 | 8.31 |

"Bayes-optimal exact match" credits chaining as far as the demonstrations
allow and then guessing uniformly among the symbols that can still be in the
terminal domain. Pooled the way the A1 tables report them, the ceilings are
**interp 0.316, mild 0.100, strong 0.031** -- against ln(vocab) = 8.326, a
perfect solver on `strong` would sit at 8.28 nats, i.e. indistinguishable from
chance.

What that does to the A1 `compose` table:

| split | Bayes ceiling | `bdh` (best) | `bdh_cq` (best) | `looped_transformer` |
|-------|---------------|--------------|-----------------|----------------------|
| interp | 0.316 | 0.100 (32%) | 0.087 (28%) | 0.001 |
| mild | 0.100 | 0.053 (53%) | 0.043 (43%) | 0.000 |
| strong | 0.031 | 0.022 (71%) | 0.018 (58%) | 0.000 |

So `bdh` -- the *non-recurrent control* -- was already at 53-71 percent of
everything the task allowed on the extrapolation splits, and the largest
credible effect available on `strong` was 0.031, below this file's own 0.05
bar. `compose` could not have produced a Gate A finding for or against
recurrence. It is withdrawn as evidence, in both directions: the
`looped_transformer` 0.000 is not evidence that it cannot compose either.

The 8.33-nat `AT_CHANCE` plateau is a separate matter and is not explained by
the ceiling: a perfect solver would sit at 5.9 nats on the train difficulties,
so `looped_transformer` (3/3 seeds) and `bdh_cq` (1/3) genuinely failed to
learn even the recoverable part. With 71 percent of each batch carrying an
answer that is absent from its own context, most of the gradient was noise
pointing at nothing, which is the most likely reason -- but that is a
hypothesis, and the fixed generator tests it directly.

**Fixed, in two steps** (`bdhx/tasks/compose.py`). The first fix made the
query's chain always demonstrated, which restored solvability to 1.000 at every
depth -- and introduced a second degeneracy that a pre-launch review caught
before any GPU time was spent on it. Demonstrating the chain plus unrelated
distractors leaves the distractors dead-ending, so the chain was the **only**
complete length-d path through the demonstration graph: in 99.4 percent of
depth-8 episodes the answer could be read off **without looking at the query at
all**. A task whose answer does not depend on its query is not testing
composition.

So each function is now demonstrated on the *images* of the previous
function's demonstrated inputs. The graph becomes `n_examples_per_fn` disjoint
complete paths and the query selects which one; the default is the full
bijection (`n_examples_per_fn = domain_size = 8`). Measured over 2000 episodes
per depth, at every depth from 1 to 8: oracle solvable 1.000, unique-path
(query-redundant) 0.000, endpoints 8.

That also fixes the null. The target is always the end of one of the
demonstrated paths, so the floor for guessing is **0.125**, flat across depth
-- not `1/4128 = 0.00024`, and not the 0.25-to-0.06 slide the intermediate
version had. This matters directly for A1c below, whose go/no-go is "does
`looped_transformer` leave the ln(vocab) plateau": leaving it is satisfied by
learning to emit a path endpoint and nothing else, so 0.125 is the line to beat,
and `tools/task_ceiling.py` now prints it per split.

Cost of the fix: demonstrations go from `4d` pairs to `8d`, so episodes roughly
double in length. The pre-fix distribution stays reachable as
`ComposeTask(n_examples_per_fn=4, guarantee_solvable=False)` so the A1 numbers
remain reproducible, and `generator_version` (now recorded in every run's
`metadata.json`, not just in cached shards) moves to 0.2.0 so the two can never
be merged silently.

Five tests pin it (`tests/test_tasks_compose.py`):
`test_compose_every_episode_is_solvable_from_its_demonstrations`,
`test_compose_answer_cannot_be_read_off_without_the_query`,
`test_compose_guess_floor_is_one_over_the_demonstrated_paths`,
`test_compose_demonstrations_are_distinct_per_function`, and
`test_compose_legacy_distribution_is_mostly_unsolvable`. The existing
`test_target_correctness_compose` did not catch any of it because it checked
the target against `extras["fns"]` -- the hidden ground truth no model sees --
rather than against the demonstrations.

`tools/task_ceiling.py` generalizes the check and is a pre-sweep gate
(`--min-train-solvable`). Run over the whole suite, `compose` was the only
broken task: `binding`, `overwrite`, `propagate`, `nested`, `order`,
`distractors` and `contradict` all sit at 1.000 on every difficulty. In
particular `propagate`, which carries the Gate A result, is unaffected.

### A1b. The loop does not converge past R_train_max, but that is not why it falls apart

**Superseded in its causal claim by section A4.** This section's original
title was "Why it falls apart past `R_train_max`: the loop never converges",
and the diagnosis below is sound as far as it measures: the loop genuinely
does not settle. What it did not establish, and what A4 has since tested
directly, is that the non-convergence *causes* the accuracy collapse. It does
not. A4's `residual` arm holds `cos_last` at 1.0000 at R=32 and still scores
exact match 0.000 at R=8. Read everything below as a description of a real
symptom, not of the mechanism.

Second follow-up, also CPU-only, from the diagnostics already stored in
`results/*/results.json` plus a probe that reloads the A1 checkpoints.
Reproduce the table with `python tools/aggregate_results.py` and read
`recurrence_convergence_propagate.csv` (copied to `docs/results/a1/`).

First, the "collapse to 0.000" is partly the metric. `propagate` targets are
12-token grids and exact match needs all 12; token accuracy shows a graded
fall. On `propagate`/`mild`, mean over 3 seeds and both difficulties:

| model | metric | R=1 | R=2 | R=4 | R=8 | R=16 | R=32 |
|-------|--------|-----|-----|-----|-----|------|------|
| bdh_cq | token acc | 0.931 | 0.916 | 0.952 | 0.596 | 0.674 | 0.461 |
| bdh_cq | exact match | 0.471 | 0.276 | 0.579 | 0.000 | 0.000 | 0.000 |
| looped_transformer | token acc | 0.802 | 0.812 | 0.819 | 0.810 | 0.746 | 0.666 |
| looped_transformer | exact match | 0.003 | 0.018 | 0.029 | 0.038 | 0.055 | 0.019 |

A CPU sweep of one seed over the intermediate R values the sweep skipped
(R = 5, 6, 7 on `mild`/distance=6) confirms the shape: per-token accuracy runs
0.96 at R=4, 0.86 at R=5, 0.71 at R=6, 0.63 at R=7, 0.60 at R=8. The
degradation is smooth; the 12-token conjunction turns it into a cliff. The
first answer token is *not* the casualty -- it still agrees with the R=4
prediction on 60-100 percent of episodes at every R up to 32.

Second, and this is the mechanism: `cos(H[R], H[R-1])` at the final iteration,
which is 1.0 exactly when the loop has reached a fixed point.

| model | split | R=1 | R=2 | R=4 | R=8 | R=16 | R=32 |
|-------|-------|-----|-----|-----|-----|------|------|
| bdh_cq | interp | 0.231 | 0.855 | 0.870 | 0.910 | 0.783 | 0.802 |
| bdh_cq | mild | 0.194 | 0.821 | 0.793 | 0.949 | 0.767 | 0.742 |
| bdh_cq | strong | 0.819 | 0.835 | 0.937 | 0.949 | 0.952 | 0.967 |
| looped_transformer | interp | 0.479 | 0.931 | 0.991 | 0.999 | 1.000 | 1.000 |
| looped_transformer | mild | 0.475 | 0.931 | 0.991 | 0.999 | 1.000 | 1.000 |
| looped_transformer | strong | 0.634 | 0.964 | 0.996 | 0.999 | 1.000 | 1.000 |

The looped Transformer converges: monotone to 1.0000 by R=32 on every split of
both tasks. BDH-CQ does not. On the two splits where it actually learned
`propagate`, its per-step cosine peaks in the 0.79-0.95 range and then *falls*
as R grows -- the state keeps rotating by roughly 35-40 degrees per iteration
no matter how long the loop runs, so H[32] is nowhere near H[4]. A direct
measurement on the seed-1 checkpoint agrees: `cos(H[R], H[4])` is 0.84 at R=5,
0.66 at R=8, 0.43 at R=16 and 0.40 at R=32.

The one place BDH-CQ *does* look convergent is `strong` (0.82 rising to 0.97),
where its token accuracy is flat at 0.63-0.65 across every R -- it has settled
because it has given up. Where it works it does not converge; where it
converges it has nothing to say.

Two candidate mechanisms were tested and ruled out:

- **Unbounded Hebbian writes.** The latent loop writes `k^T v` at every step
  with no decay, so a plausible story was that the memory drowns the
  demonstrations. It does not: `||M||_F` grows 113,625 -> 126,314 (R=4) ->
  127,063 (R=8), i.e. 0.6 percent between the working and failing settings.
  Freezing latent writes at eval time (`update_latent_memory=False`) drives
  exact match to 0.000 at *every* R including R=4, so the writes are load
  bearing, not the fault.
- **Numerical blow-up.** Ruled out for the reason the original bullet gave,
  though not by the evidence it gave: `||H||` is pinned by a parameter-free
  LayerNorm and could not have blown up. NaN counts are 0 throughout.

~~This makes H7 (`EXPERIMENT_PLAN` section 3: "initial-state skips reduce
recurrent drift") the live hypothesis. The community `attn_residual` -- whose
author states it stabilizes recurrence beyond 4 steps -- adds zero parameters
and is the cheapest arm to test.~~ **Struck: tested and refuted in A4.** Both
mechanisms do what they claim -- `attn_residual` takes `cos_last` at R=32 from
0.802 to 0.999 -- and neither moves exact match past `R_train_max` off 0.000.
H7's own arm, `init_skip`, was additionally the weakest of the three at the
convergence target it was proposed for.

~~It also gives H7 a measurable target that does not depend on accuracy at
all: raise `cos_last` at R=32 toward 1.0.~~ **Struck.** The A4 pilot and
section A1d below both produce `cos_last` near 1.0 together with exact match
0.000, from models that have not learned the task. An accuracy-free target
is satisfied by not learning, which is the failure mode, so the target has to
be joint: learn the task *and* keep the fixed point.

### A1c / A4 pilots (dev, 1 seed, not evidence)

Two go/no-go screens at 8000 steps (20 percent of A1's budget), one seed per
arm, run to decide whether ~$67 of full sweeps was worth spending. Both are
tagged `dev` and `DEV`-flagged. Configs `configs/stage_a/a1c_compose_pilot.yaml`
and `a4_convergence_pilot.yaml`; commit `84c89ce`; 6 jobs, RTX 4090 Secure,
0.95 GPU-hours of training and about **1.6 GPU-hours / $1.15** billed.

**A1c: no signal, which the design said in advance would be uninformative.**

| model | final train loss | AT_CHANCE | exact match, interp (R=1/4/8/32) |
|-------|------------------|-----------|----------------------------------|
| bdh | 8.251 | yes | 0.000 / 0.000 / 0.000 / 0.000 |
| bdh_cq | 7.905 | no | 0.015 / 0.025 / 0.000 / 0.000 |
| looped_transformer | 8.331 | yes | 0.000 / 0.000 / 0.000 / 0.000 |

`looped_transformer` sits at exactly ln(vocab) = 8.326, so it did not leave the
plateau; `bdh_cq` left it slightly and is still an order of magnitude below the
0.125 guessing floor. Per the pilot's own one-sided framing this **does not**
fund the full A1c and is **not** evidence that any of the three cannot learn
`compose`: 8000 steps on a task whose episodes just doubled in length may
simply be too few. It buys one fact -- nothing here justifies spending $27 yet.

**A4: the pilot refuted its own premise, and that is the finding.**

`cos(H[R], H[R-1])` at the last iteration, `propagate`/`mild`, and accuracy
beside it:

| kind | final loss | R=1 | R=4 | R=8 | R=32 |
|------|-----------|-----|-----|-----|------|
| plain | 0.673 | 0.694 | 0.994 | 0.997 | **1.000** |
| attn_residual | 0.724 | 0.710 | 1.000 | 1.000 | **1.000** |
| init_skip | 0.660 | 0.862 | 0.994 | 0.946 | **0.996** |

Token accuracy is 0.66-0.81 and exact match is 0.000 in every cell above.

A1's `plain` arm at 40000 steps sits at **0.74-0.80** at R=32 on this same
split (section A1b). At 8000 steps it is at **1.000**. So the non-convergence
A1b diagnosed is **not a property of the update rule**: BDH-CQ's latent loop
converges perfectly well early in training and *stops* converging somewhere
between 8000 and 40000 steps. It is acquired, not architectural.

Two consequences:

1. **A4 cannot be screened at 8000 steps.** All three arms already converge, so
   there is no gap for `attn_residual` or `init_skip` to close and the three
   are indistinguishable on the metric the sweep exists to move. The full
   sweep remains the right experiment but must run the full 40000 steps; a
   short version measures a regime in which the phenomenon does not exist.
2. **`cos_last` cannot be the primary metric on its own**, and
   `a4_convergence.yaml`'s argument for it -- that it "measures the learned
   map's contraction, not its accuracy", so it "needs no bootstrap over a
   near-zero exact match" -- is wrong. These runs have `cos_last` ~ 1.0 *and*
   exact match 0.000: the model has not learned `propagate` yet and its loop
   trivially converges. That is the same signature A1b already flagged on the
   `strong` split, where BDH-CQ "converges because it has given up". High
   `cos_last` is confounded with "has not learned the task", so it must be read
   jointly with accuracy, never instead of it.

The A1b finding itself stands -- at 40000 steps, on the cells it learned,
BDH-CQ does not converge while the looped Transformer does -- but its
interpretation narrows: non-convergence is something this architecture
*develops as it learns the task*, which is a more specific and more
interesting claim than "the update rule is not contractive".

**Cost note:** the six jobs ran 4.5 to 12.8 minutes each against per-job
estimates of 12.8 to 94.7 minutes, so `generated/*/estimates.json` (derived
from A1's manifest) is roughly 6x conservative for training. The full-sweep
figures quoted below are therefore likely upper bounds, but they are not
revised here: A1's 72 GPU-hours were dominated by its full evaluation grid
(6 R values x 1000 episodes) rather than by training, and that grid is
unchanged in the full sweeps. Profile before trusting a cheaper number.

### A1d. When the loop stops converging: it is acquired while learning, and only by BDH-CQ

Third follow-up, **zero GPU cost**, and it should have been the first: every A1
run already evaluated every 2500 steps at R in {1, 4, 16} with full
diagnostics, so the step-resolved answer had been sitting in
`results/*/results.json` since the sweep finished. A ~$2.50 step-sweep was
scoped and quoted before that was checked; it was not run, because the data
already existed. Reproduce with `python tools/aggregate_results.py results
--out reports/` and read `convergence_onset_propagate.csv` /
`convergence_onset_propagate.png`.

`cos(H[R], H[R-1])` at the last iteration, R=16, `propagate`/`interp`, mean
over 3 seeds and 3 difficulties, with exact match at R=4 (the deepest depth
the model was trained on) beside it:

| step | 2500 | 7500 | 12500 | 15000 | 17500 | 20000 | 25000 | 32500 | 40000 |
|------|------|------|-------|-------|-------|-------|-------|-------|-------|
| bdh_cq `cos_last` | 0.9997 | 0.9910 | 0.9853 | 0.9091 | 0.8261 | 0.8098 | 0.7849 | 0.7695 | 0.7828 |
| bdh_cq exact match @R=4 | 0.000 | 0.006 | 0.054 | 0.232 | 0.699 | 0.893 | 0.978 | 0.992 | 0.997 |
| looped_tf `cos_last` | 0.9998 | 0.9997 | 0.9998 | 0.9998 | 0.9997 | 0.9998 | 0.9998 | 0.9998 | 0.9998 |
| looped_tf exact match @R=4 | 0.874 | 0.971 | 0.994 | 0.995 | 0.997 | 0.997 | 0.999 | 1.000 | 1.000 |

Three things follow, and the second is the one that matters.

**1. The loss of convergence is acquired, in a window.** BDH-CQ's loop starts
convergent (0.9997), holds above 0.98 through step 12500, loses 0.18 between
12500 and 20000, and is then flat for the remaining 20000 steps (0.78 at both
22500 and 40000). It is absent at the first checkpoint and does not accumulate
with step count -- outside the window, more training does nothing to it.
Nothing here measures initialization itself: the earliest observation is step
2500, so "starts convergent" means "is convergent after 2500 steps", not
"is convergent at step 0". This confirms the A4 pilot's finding and pins the window the pilot could
only bracket as "somewhere between 8000 and 40000".

**2. It is not the price of learning; only BDH-CQ pays it.** The window is
exactly where BDH-CQ acquires the task (exact match at R=4 goes 0.05 -> 0.23
-> 0.70 -> 0.89 across it), which invites the reading that a model must give
up its fixed point to learn `propagate`. The looped Transformer refutes that
directly: same task, same 40000 steps, same ~10M budget, learned to exact
match 1.000 -- and its `cos_last` is 0.9997-0.9998 at every one of the 16
checkpoints, never moving at all. It also extrapolates where BDH-CQ does not
(exact match at R=16 rises 0.32 -> 0.81 over training, against a BDH-CQ peak
of 0.0003 across all 16 checkpoints). Learning and a fixed point are compatible; BDH-CQ
specifically fails to hold both.

**3. The `strong` split is a within-run control that points the same way.** On
`strong`, which BDH-CQ never learns (exact match 0.000 at every checkpoint),
its R=16 `cos_last` declines by 0.048 over training (0.9999 to 0.9516)
against 0.217 on `interp`. Same model, same optimizer steps, same checkpoints:
the split that learns loses four and a half times as much convergence as the
split that does not. This is the same confound A1b flagged, now measured over training rather
than at one point -- and it is why `cos_last` cannot stand alone as a metric.

`bdh` is **not** evidence in either direction and is excluded from the claim.
It has no latent loop at all (`bdhx/models/bdh.py`: "Baseline BDH has no latent
loop; `reasoning_steps > 1` is ignored"), which is why its exact match and
`cos_last` are identical at every R from 1 to 32 (1.000 and 0.214 at step
40000). Its flat, low `cos_last` measures one block application, not a
trajectory. Read as a convergence result it would say "the least convergent
model extrapolates best", which is an artifact of comparing a loop to a
non-loop.

One caveat on the R=1 rows in the CSV, already flagged there by
`cos_last_vs_seed`: at R=1 the diagnostic compares H[1] against the ingested
seed rather than against a previous latent, so its fall (0.904 -> 0.228 on
interp, tracking the accuracy curve) is a different quantity -- how far one
step moves the state -- and is not comparable to the R=16 column above.

**Consequences for A4.** The sweep is still worth running and its cost is much
lower than previously quoted, but its question and its metric both change:

- The question is no longer "does anything make the loop converge". An
  untrained loop already converges, and so does a trained one on a split it
  never learned. It is whether any update rule reaches the looped
  Transformer's regime: **learns the task and keeps the fixed point**.
- The success criterion is therefore joint -- `cos_last` at R=32 *and* exact
  match at R=4 at the same checkpoint. An arm at `cos_last` 1.0 with exact
  match 0.0 has not fixed anything.
- The sweep cannot be shortened. 8000 steps sits entirely before the window;
  25000 would capture it, but A4 reuses A1's 40000-step `plain` runs as its
  reference arm, and re-running those to match a shorter budget costs the 3
  jobs the reuse saves. Run 40000.
- `evaluation.intermediate_reasoning_steps` (new, and excluded from the config
  hash because it changes only what the mid-training checkpoints observe) is
  set to `[1, 4, 16, 32]` in `a4_convergence.yaml`, so each arm gets an onset
  curve at the depth the primary metric is read at. A1 could not produce one:
  the hardcoded `(1, 4, 16)` meant no checkpoint before the last ever measured
  R=32.

**Revised A4 cost, measured rather than profiled.** A1's bdh_cq/propagate seeds
took 3030 s of training for 40000 steps; the pilot puts `attn_residual` at
1.26x and `init_skip` at 1.20x that per step, with `residual` bounded by
`init_skip`. That is 9.2 GPU-hours of training over 9 jobs, plus ~0.35 h of
evaluation and ~0.98 h of pod startup at the 6.5 min/pod the pilot measured:
about **10.6 billed GPU-hours, ~$7.80** at the 4090 Secure rate of $0.74/hr,
**~$10.20 with the 1.3x contingency**. The ~$40 quoted earlier came from
`est_gpu_minutes` in the generated manifest, which the pilot showed to be
roughly 6x conservative. `runpod_launch.py estimate` still reports the
conservative figure; it has not been changed, because a launcher that
under-quotes is worse than one that over-quotes.

### A4. Convergence is fixable, and fixing it changes nothing

Full sweep, 9 jobs (3 update rules x 3 seeds, 40000 steps, `propagate`, ~10M
params), `configs/stage_a/a4_convergence.yaml`, read against A1's
`bdh_cq`/`plain` cell as the reference arm. All 9 exit 0 at step 40000 with
final train loss between 6.7e-5 and 4.7e-4: 0 AT_CHANCE seeds, 0 diverged
seeds, 0 relaunches. Tables in `reports/2026-09-06-a4/`, copied to
`docs/results/a4/`.

**The primary metric was joint, and two arms pass it.** `interp` split, final
checkpoint, mean over 3 seeds and 3 difficulty buckets:

| arm | cos_last @ R=32 | exact match @ R=4 |
|-----|-----------------|-------------------|
| bdh_cq / plain (A1 reference) | 0.802 | 0.997 |
| bdh_cq / init_skip | 0.849 | 0.990 |
| bdh_cq / attn_residual | 0.999 | 0.998 |
| bdh_cq / residual | **1.000** | **1.000** |
| looped_transformer / plain | 1.000 | 1.000 |

The question A4 was opened with therefore answers yes. `residual` --
`h + block(h)`, the cheapest arm in the grid, included only as the control
that would say whether `init_skip`'s gain needed the seed specifically or just
a smaller step -- puts BDH-CQ's loop exactly in the looped Transformer's
regime. It holds `cos_last` at 1.0000 at every difficulty at R=8, 16 and 32
while learning the task to exact match 0.9997. `attn_residual`, the community
mechanism, reaches 0.9993. H7's own arm, `init_skip`, is the one that does not
work: 0.849 against plain's 0.802 is the smallest gain in the table, and the
control beat the hypothesis it was there to control for.

**And it buys nothing.** Exact match past `R_train_max = 4`, same split and
checkpoint:

| arm | R=4 | R=8 | R=16 | R=32 |
|-----|-----|-----|------|------|
| bdh_cq / plain | 0.997 | 0.000 | 0.000 | 0.000 |
| bdh_cq / init_skip | 0.990 | 0.000 | 0.000 | 0.000 |
| bdh_cq / attn_residual | 0.998 | 0.000 | 0.000 | 0.000 |
| bdh_cq / residual | 1.000 | 0.000 | 0.000 | 0.000 |
| looped_transformer / plain | 1.000 | 0.973 | 0.811 | 0.466 |

Every `bdh_cq` seed of every arm is exactly 0.000 at R >= 8 -- 9 seeds x 3
difficulties, no exceptions -- so the bootstrap 95% CI is [0.000, 0.000]
against the looped Transformer's [0.963, 0.980] at R=8. Non-overlapping, and
the gap is 0.973 against this project's 0.05 threshold.

A model whose loop reaches a perfect fixed point, `cos_last` = 1.0000 to four
decimals, still scores 0.000 at R=8. **The loop's failure to converge was not
the cause of the collapse past `R_train_max`.** It was a second, separable
symptom of whatever is.

Token accuracy carries the same conclusion with more texture, since
`propagate` exact match needs all 12 tokens of a grid. On `interp` at R=32:
plain 0.664, `init_skip` 0.651, `attn_residual` 0.586, `residual` 0.770,
looped Transformer 0.916. `residual` is the only arm that moves it at all, by
0.106, and remains 0.146 short of the looped Transformer -- real, but nowhere
near enough to complete a grid. `attn_residual` is *worse* than plain at R=32
(0.586 against 0.664) while having near-perfect `cos_last`, which is the
single cleanest number against the convergence hypothesis: inside `bdh_cq`,
more convergence did not mean more accuracy, and here it meant slightly less.

The `strong` split is the within-sweep control and behaves as it must: no arm
learns it (exact match 0.000 everywhere), and `residual` still reaches
`cos_last` 1.000 there. Convergence is available with or without the task,
which is the same point section A1d made about the 8000-step pilots.

**What this costs the plan.** Stage C's recurrence-engineering direction
(`configs/stage_c/c1_recurrence_engineering.yaml`) sweeps six further update
rules against `cos_last`. Five of the six are variations on making the map
contractive, and A4 shows a map contractive enough to reach `cos_last` =
1.0000 still scores 0.000 past `R_train_max`. C1 should not be run in its
current form; the `cos_last` target it is built on has been measured and
saturated. The open question is now what the looped Transformer does that
BDH-CQ does not, and it is not fixed-point behaviour, because the two are
tied at 1.000 on that metric and 0.973 apart on the one that matters.

**Cost.** 9 jobs run fully concurrent, ~100 minutes wall clock, 13.62 billed
GPU-hours, $10.08. Estimated beforehand at 12.47 GPU-hours / $9.23 from the
pilot's measured ms/step; the estimate held to within 9 percent. About 1.0
GPU-hour (~$0.77) of that was pods sitting finished-but-uncollected, which was
avoidable and is written up in the ledger.

### A5. The collapse is the trained range, and nothing else

Full sweep, 3 jobs (`bdh_cq`/`plain` on `propagate`, `reasoning.train_steps:
[1,2,4,8]`, 3 seeds, 40000 steps, ~10M params),
`configs/stage_a/a5_r_train_extension.yaml`, read against A1's
`bdh_cq`/`plain` cell (`f0b8b7c55f92`, the same model trained on
`[1,2,4]`). All 3 exit 0 at step 40000, final train loss 3.9e-4 to 4.9e-4,
0 AT_CHANCE, 0 diverged, 0 relaunches. Tables in `reports/2026-09-07-a5/`,
copied to `docs/results/a5/`.

**Exact match moves with `R_train_max`, one for one.** `interp` split, final
checkpoint, 3 seeds x 1000 episodes per cell, bootstrap 95% CI in brackets.
Both arms are the same architecture at the same 10,010,880 parameters; the
only difference is which R values training sampled.

| R_test | A1, trained {1,2,4} | A5, trained {1,2,4,8} |
|--------|---------------------|-----------------------|
| 1  | 0.998 [0.997, 1.000] | 0.994 [0.982, 1.000] |
| 2  | 0.999 [0.997, 1.000] | 0.995 [0.988, 1.000] |
| 4  | 0.997 [0.994, 1.000] | 0.980 [0.973, 0.991] |
| 8  | **0.000 [0.000, 0.000]** | **0.991 [0.985, 0.997]** |
| 16 | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] |
| 32 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

Rows are the `distance=1, length=12` difficulty; `distance=2` and
`distance=4` agree to within 0.01 everywhere, including the 0.000 cells (see
`docs/results/a5/acc_vs_reasoning_steps_propagate.csv` for all three).

At R=8 the CIs are disjoint and the gap is 0.991, against this file's 0.05
threshold. This clears the bar by a wider margin than any result in the
project so far.

**H-capacity is refuted.** BDH-CQ (community) composes the operation eight
times at 0.991 exact match. There is no capacity ceiling at four, and the
reading offered in A4's follow-up -- that the architecture "cannot compose
the operation more than a few times whatever it is shown" -- is wrong. It
composes exactly as many times as it was trained to.

**H-readout is confirmed, in its strongest form.** The failure boundary is
not near `R_train_max`, it is exactly at it. Train to 4: 0.997 at R=4,
0.000 at R=8. Train to 8: 0.991 at R=8, 0.001 at R=16. The model acquires
the trained set and generalizes zero steps past it.

**What that costs the architecture's premise.** The point of latent
recurrence is to train short and think longer at test time. This
reconstruction has none of that: extra test-time loops buy exactly nothing
outside the trained range, and the only way to get accuracy at R is to have
trained at R. That is a sharper negative than "it fails to extrapolate",
because it rules out the gentler reading in which the loop degrades
gradually and might be nursed further with better conditioning.

**It also settles A4 retroactively.** Forcing the loop to a perfect fixed
point changed nothing because convergence was never the mechanism; the
readout had simply never seen states from outside the trained set. A4
established that by elimination, A5 establishes it directly.

**The one thing that survives past the boundary is token accuracy.** At
R=16 and R=32 the A5 arm holds `token_acc` 0.73 to 0.81 while exact match
is ~0.00 (A1's arm: 0.64 to 0.78 past its own boundary). The state is not
destroyed by the extra iterations and is not at chance -- it is degraded
just enough that no full answer survives. Whatever the extra loops do, they
are not diverging into noise, which is consistent with A4's finding that
convergence is separable from correctness.

**Scope.** Per the caveat at the head of this stage: the latent transition
function is not published, so this is a property of `lucidrains/bdh-cq` at
`c246f890`, not of Pathway's model. The paper states no trained range and
makes no extrapolation claim, so A5 answers a question it leaves open
rather than contradicting it.

**Cost.** 4.15 GPU-hours, $3.07, against a 5.35 GPU-hour / $3.96 estimate:
22 percent under, the second cost prediction in a row to land close and the
first to come in under. Per-pod alive time 79.6, 79.9 and 89.8 minutes.

**What A5 does not test, and the free experiment that follows.** Every
trained value here is a power of two, and every tested value outside the
trained set is also outside its *range*. So this cannot distinguish two
readings: the model learned the trained SET (and R=6 would fail even though
it lies between 4 and 8), or it learned the trained INTERVAL (and R=6 would
work). The distinction matters -- an interval-learner extrapolates within a
range and would make "train on {1,32}" a sensible cheap recipe, a
set-learner would not. Evaluating the existing A5 checkpoints at R in
{3,5,6,7} answers it at zero GPU cost, since the checkpoints are already
on disk. This is the next thing to run.

### A2. Recurrence curriculum repeat

Pending, and **it does not test what this document twice said it tests.**

The claim made here and in "Next experiment and why" was that A2 probes
whether the readout is the binding constraint, on the reasoning that "a
curriculum arm that ends at R=8 would very likely just move the peak from R=4
to R=8". `configs/stage_a/a2_curriculum.yaml` has no arm that ends at R=8. Its
curriculum is `schedule: [1, 2, 4]` over `train_steps: [1, 2, 4]`, so
`RTrainSampler.r_max` is 4 in the curriculum arm and 4 in the uniform arm.
Running the sampler over the full 40000 steps, the entire difference the sweep
creates is the share of steps spent at each R:

| arm | R=1 | R=2 | R=4 | R_train_max |
|-----|-----|-----|-----|-------------|
| uniform | 33% | 33% | 34% | 4 |
| curriculum | 30% | 30% | 40% | 4 |

A2 therefore asks whether the *order* of exposure matters at a fixed maximum.
That is a real question and A4 gives no particular reason to expect a yes on
it, but it cannot move an accuracy peak past a value it never trains, so it
cannot discriminate the two hypotheses A4 left standing. It is also 24 jobs,
half of which (`reasoning.train_step_sampling: uniform`) re-run A1 cells that
already exist.

`configs/stage_a/a5_r_train_extension.yaml` is the sweep that does test it:
one arm, `train_steps: [1, 2, 4, 8]`, against A1's `[1, 2, 4]` cell, 3 jobs
and ~$4. See "Next experiment and why" below. A2 is worth keeping only as a
follow-up if A5 says the peak does move -- at which point how to spend a fixed
R budget becomes a live question rather than an academic one.

`tests/test_configs_stage_a.py` now pins the `r_max == 4` fact, so this
section cannot quietly revert to describing A2 as an extrapolation probe.

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
recurrence result this project has produced with real seeds. The follow-up
diagnoses (A1a, A1b), both run on CPU from the checkpoints and results the
sweep already produced, cost zero GPU-hours and turned "it collapses, we do
not know why" into a specific, measurable mechanism: the latent loop does not
converge, and the metric that would show that (`cos_last` at large R) does not
need accuracy at all.

**What failed:** `compose` produced nothing usable, and A1a shows it could not
have: its generator left 71 percent of training episodes with an answer that
is absent from their own context, capping Bayes-optimal exact match at 0.10 on
`mild` and 0.03 on `strong`. That is a defect this project shipped and its own
task tests missed, because they checked the target against hidden ground truth
rather than against the demonstrations. `bdh_cq`'s `propagate` win does not
extend past `R_train_max` -- Gate D fails for this model on this task. Getting
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

A seventh defect, found by A1b and fixed here, is in the reporting rather than
the runs: `state_norm_vs_iteration` and `cos_consecutive_vs_iteration` pooled
every checkpoint, split, difficulty and R_test into one unlabelled overlay,
and their backing CSVs carried only `(iteration, value)`. That is what the
first version of the A1 diagnostics bullet read a fixed point off. The series
are now labelled with seed, step, R_test, split and difficulty; the plots show
the final checkpoint only; and `recurrence_convergence_<task>.csv` reports
`cos_last` next to accuracy per cell, which is the comparison
`EXPERIMENT_PLAN` section 6 actually asks for.

**Confidence:** provisional. 3 seeds per cell, as flagged. The `propagate`
Gate A finding has disjoint bootstrap CIs and a large effect size, which is
about as strong as a 3-seed result gets, but a `[1,2,3,4,5]`-seed repeat is
the right bar before treating it as settled, per this file's own
conventions. The A1a ceilings are analytic and confirmed at 2000 episodes per
cell, so they are not provisional. The A1b convergence table is 3 seeds x 1000
eval episodes on the same runs as A1, and the effect it reports (1.0000 vs
0.74-0.80 at R=32) is far larger than its seed spread, but it compares two
architectures on one task and should not be generalized past that.

**Compute spent:** see the compute ledger below (~72 GPU-hours estimated,
~$54 on RTX 4090 Secure Cloud, over the ~$25 gate approved with explicit
sign-off after Community Cloud proved unreliable for this sweep). A1a and A1b
added 0 GPU-hours: both ran on CPU against checkpoints and `results.json`
files the sweep had already produced.

**Next experiment and why:** A1a and A1b changed the ordering. `compose` is
fixed but has never been run against a model, and the drift diagnosis, not the
curriculum, is now the live hypothesis. In priority order:

1. **A1c, `compose` re-run** (`configs/stage_a/a1c_compose_rerun.yaml`, 9 jobs
   = 3 models x 3 seeds, 36.0 GPU-hours, ~$26.6 at RTX 4090 Secure). The
   identical A1 arms on the fixed generator. This is the only way `compose`
   re-enters the evidence base, and the cheapest way to find out whether
   `looped_transformer` fails to compose or merely failed to learn from a
   corpus in which 71 percent of answers were unreachable.
   `tools/task_ceiling.py --task compose --min-train-solvable 0.99` gates it.
2. ~~**A4, convergence engineering**~~ **Done, 2026-09-06, 13.62 GPU-hours /
   $10.08. See section A4.** It raised `cos_last` at R=32 from 0.802 to 1.000
   and accuracy past `R_train_max` did *not* follow: exact match stays exactly
   0.000 at R=8 for all three arms and all 9 seeds. The half of the question
   this entry treated as the interesting half was answered yes, and it turned
   out not to be the half that mattered.
3. ~~**A5, extend R_train_max**~~ **Done, 2026-09-07, 4.15 GPU-hours /
   $3.07. See section A5.** The peak moved exactly one-for-one with the
   trained set: 0.991 exact match at R=8 against A1's 0.000, disjoint CIs,
   and a fresh 0.001 cliff at R=16. H-readout confirmed, H-capacity refuted.
4. **A5b, the set-vs-interval question** (no config yet, **0 GPU-hours**).
   Re-evaluate the A5 checkpoints already on disk at R in {3,5,6,7}. Every
   value A5 trained is a power of two and every value it tested outside the
   trained set is also outside the trained range, so A5 cannot tell "learned
   the trained SET" from "learned the trained INTERVAL". R=6 separates them.
   An interval-learner would make "train on {1,32}" a cheap recipe; a
   set-learner would not. Free, and it should run before anything paid.
5. ~~**A2 as written**~~ **Demoted: it cannot test this.** Both of A2's arms
   cap `R_train_max` at 4; see section A2 above for the numbers. A5 has now
   answered the question A2 was being kept for, so A2's remaining value is
   only its ordering question, at 24 jobs. Not worth it.

(2) and (3) are done. (4) is free and next. (1) is not started and still
needs an explicit cost decision.
The combined 8000-step pilot proposed here was run (section A1c/A4 pilots) and
was worth its ~$1.15: it is what replaced the ~$40 A4 estimate with the
measured ~$9 one, and what showed that an accuracy-free `cos_last` target is
satisfied by not learning. It did not predict A4's actual result, because at
8000 steps no arm had learned the task yet -- which is the documented limit of
a pilot that screens on a metric the unlearned model already maximizes.

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
| 2026-09-06 | A | a1c/a4 pilots (6 jobs, 1 seed, 8000 steps) | RTX 4090, Secure Cloud | ~1.6 | ~1.15 | All 6 exit 0, collected, pods terminated on collection; `get_pods()` confirmed zero remaining. Training wall clock 0.95 GPU-hours; the rest is boot, clone and pip per pod. Came in at a quarter of the $5.27 estimate because A1's profiled per-job minutes are about 6x conservative for training. Findings in section A1c/A4 above: A1c returned no signal (one-sided by design), A4 refuted its own premise and is worth more than the $40 sweep it was screening. |
| 2026-09-06 | A | a1c/a4 pilot, first attempt (FAILED, no results) | RTX 4090, Secure Cloud | ~12.1 | ~8.93 | 6 pods launched without `--sweep-config-path`. `generated/` is gitignored, so every job died seconds after boot with FileNotFoundError on its own config. A crashed job still tars its output and sleeps, and RunPod keeps a pod allocated after its docker command exits, so all six billed at $0.74/hr until reaped by hand 2.2 hours later. The 90-minute `--max-wall-clock-minutes` cap did not fire: the API returned no `uptimeSeconds` for any of these pods, and watchdog() skipped every pod it could not time. `print_status` showed "$0.000 so far" throughout for the same reason. All three are fixed with regression tests (`tests/test_runpod_launch.py`): the flag is required, watchdog() falls back to the launcher's own `created_at`, and `status` now reports a job that has already exited. Zero science obtained; the pilot itself was not run. |
| 2026-09-07 | A | a5_r_train_extension (3 jobs: bdh_cq/plain on propagate, train_steps {1,2,4,8}, 3 seeds, 40000 steps) | RTX 4090, Secure Cloud | 4.15 | 3.07 | All 3 exit 0 at step 40000, collected, pods terminated; `get_pods()` confirmed zero remaining. Per-pod alive time 79.6, 79.9 and 89.8 minutes, computed from `runpod_state.jsonl` as the span between the launch row and the done row -- note `done` in that file is a BOOLEAN, not a timestamp, and reading it as one yields nonsense. Estimated at 5.35 GPU-hours / $3.96 from A1's measured 75.8 ms/step scaled by the 1.607 mean-R ratio; came in 22 percent under, because that ratio is an upper bound (ingest, data and the optimizer step do not scale with R). Second estimate in a row to land close, and the first to land under. Findings in section A5. |
| 2026-09-06 | A | a4_convergence (9 jobs: 3 recurrence kinds x 3 seeds, 40000 steps) | RTX 4090, Secure Cloud | 13.62 | 10.08 | All 9 exit 0 at step 40000, collected, pods terminated; `get_pods()` confirmed zero remaining. Billed hours are per-pod alive time from `runpod_state.jsonl`, not the launcher's `print_status` figure, which still reports "$0.000 so far" when the API withholds `uptimeSeconds`. Estimated beforehand at 12.47 GPU-hours / $9.23 by taking the pilot's measured ms/step (plain 67.7, `attn_residual` 85.3, `init_skip` 81.1) rather than the manifest's profiled `est_gpu_minutes`; the estimate came in 9 percent low, the first cost prediction on this project to land close. Training was 9.22 of the 13.62 hours; the remaining 4.40 is pod boot (~6.5 min each) and the final R in {1,2,4,8,16,32} evaluation, which is markedly more expensive here than in A1 because `R=32` was added to the 16 mid-training checkpoints. About 1.0 GPU-hour (~$0.77) of the overhead was avoidable idle: see the row below. Findings in section A4. |
| 2026-09-06 | A | a4_convergence, collection failure (no science lost, ~$0.77 wasted) | RTX 4090, Secure Cloud | ~1.0 | ~0.77 | Seven pods sat finished-but-uncollected for roughly 30 minutes. Two independent causes, both now fixed. (1) The completion watch polled for the absence of `RUNNING`, but `print_status` reports a finished-but-uncollected pod as `RUNNING` with a trailing "job exited 0, awaiting collect" -- so the watch could never fire on completion. Poll for the marker, not for the absence of `RUNNING`. (2) `collect` caught only `tarfile.TarError`, so when the local disk hit 100 percent mid-extract the resulting `OSError` (and, on a truncated download, `zlib.error`) propagated out of the loop and every run after it went uncollected -- while its pod kept billing. Now caught per-run with the pod deliberately left alive for retry, with two regression tests in `tests/test_runpod_launch.py` that were checked to fail against the old handler. The disk itself was freed twice over: package caches outside the repo (~8.7 GB, the correct fix) and 30 penultimate training checkpoints (3.6 GB, destructive and unnecessary -- every final checkpoint was kept, but those 30 are gone). |
| 2026-09-06 | A | A1d convergence-onset analysis | none (CPU) | 0.0 | 0.00 | No new training and no new runs: A1's checkpoints already carried a mid-training evaluation every 2500 steps at R in {1, 4, 16} with diagnostics, 16 per run over 9 runs. A ~$2.50 step-sweep (1 kind, 3 seeds, cos_last checkpointed 8k-40k) was scoped and quoted to the user before that was checked, and then not run. Delivered instead as `convergence_onset_*` in `bdhx/results/aggregate.py`, so the table regenerates from `results/` rather than from a one-off script. Section A1d. |
| 2026-09-05 | A | A1a task-ceiling audit + A1b convergence diagnosis | none (CPU) | 0.0 | 0.00 | `tools/task_ceiling.py` over all 9 tasks; a checkpoint probe over `bdh_cq`/`propagate` at R = 1..32; re-aggregation of the existing `results/`. No new training: both diagnoses reused the A1 checkpoints and results.json files. |
| 2026-09-04/05 | A | a1_first_experiment (18 jobs: 3 models x 2 tasks x 3 seeds, ~10M params) | RTX 4090, Secure Cloud | ~72 | ~54 | Estimate = sum of `generated/a1_first_experiment/manifest.csv`'s profiled per-job minutes (49.07 min/job bdh, 364.10 bdh_cq, 306.62 looped_transformer; 71.98 GPU-hours) x $0.74/hr (`configs/runpod_rates.yaml`, RTX 4090 Secure). This undercounts real elapsed wall clock: Community Cloud failed to boot repeatedly (5+ times) before the sweep moved to Secure, one job (`76dd99c62a3e_s1`) alone cycled through 11 distinct pod attempts, and several jobs were relaunched after already reaching their final checkpoint (a bug found and fixed mid-sweep, see Stage A findings) -- but a pod that never boots shows `uptimeSeconds=0` and is not believed to be billed, so those retries are assumed near-$0 rather than added on top. Exceeded the $25 cost gate approved for this sweep; re-approved by explicit user sign-off at ~$32 estimated before the Community-to-Secure switch (which itself raised the per-hour rate from $0.34 to $0.74), so the real total landed higher still. All pods reaped; `runpod status` and a direct `get_pods()` check both confirmed zero pods remaining on the account at sweep end. |

# Results

Status: no experimental results yet. This file is the single place where
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
  `UNCONVERGED` (train loss still falling at end).

## Phase 0: reproduction of the community implementation

See `docs/PHASE0_REPRODUCTION.md` for the record of the reproduction run
(command, seed, hardware, parameter count, curve, wall clock).

## Stage A: recurrence behaviour

Question: can BDH-CQ (community) (and BDH, looped Transformer) learn an
iterative algorithm that benefits from more test-time loops than it saw
in training?

Gate A finding: pending.

### A0. CPU pipeline validation (dev, not evidence)

Config: `configs/stage_a/a1_cpu_mini.yaml` (expanded with
`tools/generate_sweep.py --dev` into 9 jobs: 3 models x 3 seeds, task
`compose`, train difficulties depth 1-2, eval `interp` depth 1-2 / `mild`
depth 3-4 / `strong` depth 6-8, 100 eval episodes per split). Results:
`results/a1_cpu_mini/`, report: `reports/a1_cpu_mini/`, committed copies of
the plot, its backing CSV, `summary.csv` and `flags.csv` in
`docs/results/a1_cpu_mini/`.

This is a pipeline test, not an experiment. It exists to prove
`generate_sweep -> run_experiment -> aggregate_results` works end to end on
real configs on CPU. It is tagged `[dev, cpu_mini]` and every row is flagged
`DEV`.

Setup: `params_target` 205_000, batch size 16, 6000 steps, warmup 300, lr
3.0e-4 (the `default.yaml` value; no a0 LR sweep has been run, so the LR is
untuned), `R_train` sampled uniformly from {1, 2, 4}, `R_test` in
{1, 2, 4, 8, 16}, `compute.device: cpu`, `deterministic: true`, seeds 1, 2, 3.
205_000 rather than a smaller budget because with vocab 4128 the BDH width
solver's smallest admissible width (16) already costs 135k parameters and the
next width (24) costs 205k, so anything near 150k is outside the 5 percent
hard bound for `bdh` and `bdh_cq`.

| model | width | params realized | off target | steps | seeds | wall clock per job |
|-------|-------|-----------------|-----------|-------|-------|--------------------|
| bdh | 24 | 205,056 | +0.03% | 6000 | 1, 2, 3 | 123-139 s |
| bdh_cq | 24 | 205,056 | +0.03% | 6000 | 1, 2, 3 | 177-183 s |
| looped_transformer | 44 | 205,348 | +0.17% | 6000 | 1, 2, 3 | 165-175 s |

Total sweep wall clock 1436 s (23.9 min) for 9 jobs run sequentially on 4 CPU
cores, plus 9 s for aggregation. All 9 jobs finished with `status: ok`, 0 NaN
events, 0 preemptions.

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
| looped_transformer | interp | 0.003 +- 0.005 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| looped_transformer | mild | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |
| looped_transformer | strong | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.000 +- 0.000 |

`bdh` is the fixed-depth reference: its adapter ignores `R`, so its row is the
same number repeated across R by construction, not a flat curve it earned.
The single non-zero cell is 1 correct episode out of 100 on one seed, which is
at the chance level for a one-token answer drawn from a 4128-token vocabulary.

Flags raised (`docs/results/a1_cpu_mini/flags.csv`): `PROVISIONAL` (n_seeds=3
< 5) and `DEV` for all three arms, plus `HIGH VAR` for `looped_transformer`
(cv=1.41, std=0.009, mean=0.007) which is the coefficient of variation of that
single lucky episode. No `DIVERGED`, `UNCONVERGED` or `NOT MATCHED` flags: the
three arms are matched to within 0.17 percent of parameters.

Interpretation, limited to pipeline validity: the pipeline works end to end,
and nothing learned. Every job trained, checkpointed, evaluated at five
reasoning depths on three split-labelled difficulty bands, and wrote a
`results.json` that `aggregate.py` grouped into 3-seed statistics with
diagnostics arrays of length R, so the machinery under Stage A is exercised
and sound. Accuracy sits at chance and the training loss moves only from
about 8.5 to about 8.3 against ln(4128) = 8.33, i.e. the models never leave a
uniform-over-vocabulary predictor, which is what 205k parameters and 6000
steps of an untuned LR on CPU should produce; no claim about recurrence,
test-time extrapolation, or model ranking can be drawn from these numbers, and
none is made here.

Plot: `docs/results/a1_cpu_mini/acc_vs_reasoning_steps_compose.png` (all lines
on zero, as expected).

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
| 2026-09-03 | A | a1_cpu_mini | none (4 CPU cores) | 0.0 | 0.00 | 9 dev jobs, 1436 s wall clock total; pipeline validation only, not evidence |

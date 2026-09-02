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

Question: can BDH-CQ (and BDH, looped Transformer) learn an iterative
algorithm that benefits from more test-time loops than it saw in training?

Gate A finding: pending.

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
| pending |

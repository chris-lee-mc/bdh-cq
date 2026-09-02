# Phase 0: Inspection, Tests, and Reproduction Record

Date: 2026-09-02. Purpose: establish that the community implementation
runs, that its tests pass, and what one of its own experiments produces
at the scale it ships with, before any framework code is written.

## 1. What was inspected

- `lucidrains/bdh-cq`, commit `c246f8903a8c36496845662d7c4b7b439bb47b09`
  (2026-08-28), 21 commits, MIT. Files: `bdh_cq/bdh_cq.py` (695 lines),
  `higher_order_bdh.py`, `rotary.py`, `tasks.py`, `icq.py`, `figure7.py`,
  `train_enwik8.py`, `train_function_composition_bdh.py`, tests.
- `pathwaycom/bdh`, commit `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`
  (2026-05-15), MIT. Single-file model plus a tiny-Shakespeare trainer,
  no tests, no reasoning loop.

Findings are summarized in `EXPERIMENT_PLAN.md` section 1 and detailed
with line references in `PAPER_IMPLEMENTATION_GAPS.md`.

## 2. Environment

| item | value |
|------|-------|
| hardware | 4-core Intel Xeon @ 2.80 GHz, 15 GB RAM, no GPU |
| OS | Linux 6.18.44 |
| Python | 3.11.15 |
| torch | 2.14.0+cu130 wheel from PyPI, running CPU-only (the pytorch.org CPU index was blocked by the sandbox proxy) |
| other deps | numpy, einops, einx, datasets, tqdm, fire, pytest 9.1.1 |
| install | `pip install -e . --no-deps` in the cloned repo |

Note: the framework's own pins (plan section 15) differ from this ad hoc
environment. This record is for the community code as shipped.

## 3. Test suite

```
$ python -m pytest -v
collected 46 items
tests/test_bdh_cq.py  37 passed
tests/test_icq.py      9 passed
46 passed in 18.93s
```

Every end-to-end test is parametrized over `BDHBlock` and
`HigherOrderBDHLayer`, so both block types run through `BDH` and
`BDHReasoningWrapper`.

## 4. Reproduced experiment: `figure7.py`

The script's docstring: "Replicate the main result of the BDH-CQ paper
(figure 7 / table 5): training with the latent-effort schedule of
section 7, then sweeping the number of latent reasoning steps at
inference and watching accuracy climb." It is the only shipped
experiment that exercises the recurrent latent reasoning loop end to
end at a scale that fits on a CPU.

Configuration (unchanged from the script):

| item | value |
|------|-------|
| model | `dim=256, depth=4, dim_qk_heads=1024, attn_residual=True, attn_residual_depth_bias_distance=1` in `BDHReasoningWrapper` |
| parameters | 794,145 |
| optimizer | AdamW lr 1e-3, weight decay 0.1, grad clip 1.0 |
| batch | 1 task per step, family drawn uniformly from propagation, copy, order, nesting |
| curriculum | reasoning steps drawn uniformly from 0..8 each step |
| precision | fp32 CPU |
| evaluation | held-out `order` tasks at levels 3 and 4, 16 outputs, reasoning steps {1, 2, 4, 6, 8}, greedy decoding |
| seed | 3 (script default) |

### Run 1: default arguments, 800 steps

Command: `python figure7.py --device cpu --family order --steps 800 --seed 3`
(script default is `torch.set_num_threads(8)` on a 4-core machine).
Log: `docs/phase0_logs/figure7_order_800steps_seed3_default.log`.

| step | loss |
|------|------|
| 0 | 5.0565 |
| 200 | 3.0124 |
| 400 | 3.6286 |
| 600 | 2.7718 |

Wall clock: 1137 s total (training plus evaluation), 20:10:36 to
20:29:33 UTC.

Result on held-out `order`, 16 outputs:

| reasoning steps | 1 | 2 | 4 | 6 | 8 |
|-----------------|---|---|---|---|---|
| exact match | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| cell accuracy | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

The script prints "monotone in R: True", which is vacuous at 0 percent.

### Run 2: 300 steps, 4 threads, logging every 50 steps

An instrumented copy of the script's `run()` (identical model,
curriculum, and evaluation; only logging and timing added):
`docs/phase0_logs/repro_figure7_instrumented.py`. Log:
`docs/phase0_logs/figure7_order_300steps_seed3_4threads.log`.

| step | loss | elapsed s |
|------|------|-----------|
| 0 | 5.0565 | 0.3 |
| 50 | 2.8393 | 25.5 |
| 100 | 3.4531 | 47.1 |
| 150 | 3.0820 | 67.9 |
| 200 | 3.0062 | 91.4 |
| 250 | 3.5617 | 111.5 |
| 299 | 3.1918 | 129.8 |

Training 129.9 s (0.43 s/step with 4 threads), evaluation 40.4 s for 80
generations. Same result: 0/16 exact, 0.0 percent cells at every depth.

## 5. Interpretation

- The pipeline runs end to end on CPU and is deterministic for a given
  seed (step-0 loss identical across both runs).
- Neither run learned the `order` family. Loss falls from 5.06 to about
  3 in 50 steps and then oscillates between 2.8 and 3.6 through 800
  steps; with batch size 1 across four mixed families this is far from
  convergence. The repository's own commit history claims monotone
  improvement with reasoning steps only "for certain toy tasks".
- This is recorded as a null result at the shipped scale, not as
  evidence for or against the mechanism. It does establish that the
  shipped regime (794k parameters, batch 1, 800 steps) cannot serve as
  the Stage A baseline; the framework's Stage A runs use batch 64, 40k
  steps, and about 10M parameters on GPU, with a learning-rate sweep for
  every model (plan section 8).
- Thread oversubscription: the script's hard-coded 8 threads on 4 cores
  made run 1 roughly 3x slower per step than run 2. The framework sets
  threads from the core count.

## 6. What Phase 0 did not do

- No GPU run; no run of `train_enwik8.py` (5000 steps at 384 width is
  outside a CPU budget) or of the higher-order function-composition
  probe.
- No reproduction of any number from the BDH-CQ paper; none is
  reproducible from released code, and Pathway's README says the same
  about its Sudoku result for the base BDH repo.
- The papers' full texts were not fetched (arXiv was blocked from the
  research environment); rows marked "verify" in the gap table must be
  checked against the PDFs before being cited.

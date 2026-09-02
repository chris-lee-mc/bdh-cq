# Paper vs Implementation Gaps

Status: v0.1, based on inspection of `lucidrains/bdh-cq` at commit
`c246f8903a8c36496845662d7c4b7b439bb47b09` (2026-08-28) and
`pathwaycom/bdh` at commit `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`
(2026-05-15), plus the public papers listed in section 4.

This document exists so that nobody in this project describes a community
choice as "the BDH-CQ architecture". Three sources are distinguished
throughout:

- PAPER: stated in the BDH paper (arXiv 2509.26507) or the BDH-CQ paper
  and Pathway announcement (arXiv 2608.09888, August 2026).
- CODE: what `lucidrains/bdh-cq` actually does (file:line cited).
- GUESS: an implementation choice the community author made without a
  public specification, or that we inferred.

Sourcing caveat: arXiv full texts were not fetchable from the research
environment. Paper statements below come from the abstracts, Pathway's
public explainer pages, and secondary technical summaries. Every row
marked "verify" must be checked against the PDF before it is cited in a
report. Two facts were verified from primary sources directly: the
Pathway reference code and the community code including its commit
history.

Confidence scale: High = verified in code and consistent with paper
text; Medium = code verified, paper statement from secondary summary;
Low = inferred, or the code admits uncertainty.

## 1. Gap table

| Mechanism | Paper specifies? | Current implementation (lucidrains/bdh-cq) | Confidence | Experiment needed? |
|-----------|------------------|--------------------------------------------|------------|--------------------|
| Sparse positive activations (ReLU) on neuron state | PAPER (BDH): yes, ReLU, reported ~3-5% active | CODE: `ReLU(to_qk(tokens))` is q, k, and the FF gate, `bdh_cq.py:192-196`; FF tail `ReLU(projected * ff_gates)` at `:236` | High | Instrument sparsity per iteration (Stage A diagnostics), no separate experiment |
| Shared-QK linear attention without softmax, self masked out | PAPER (BDH): linear attention over Hebbian state; Pathway code confirms no softmax | CODE: single `to_qk` projection, `sim = q.k`, `tril(-1)`, unnormalized, `bdh_cq.py:163, 209-218`. Commit `403d430` calls it "shared qk attention (first seen in Reformer)" | High for the no-softmax part; Medium for shared QK being the paper's intent (Pathway code has separate Q/K from the same sparse x, so shared QK is a CODE interpretation) | Yes, small: shared-QK vs separate-QK ablation inside the BDH baseline (Stage B side check) |
| Values are the raw residual stream (no value projection) | PAPER: not clearly specified (verify) | CODE: `v = tokens`, `bdh_cq.py:200` | Medium | Low priority ablation |
| Weight sharing across depth | PAPER (BDH): Pathway reference code ties all parameters across `n_layer` iterations; paper text on benchmarked models not verified | CODE: one `BDHBlock` instance reused `depth` times, `bdh_cq.py:299-305, 381` | High for the reference code; Medium for the paper's large models | Yes: this is exactly the "recurrent depth" variable in H1/H3 |
| Hebbian outer-product fast-weight memory, additive, no decay | PAPER (BDH): sigma updated by outer product of sparse pre/post activations (verify exact form) | CODE: `memories = einsum(k, v)` summed over sequence, one `dim_qk x dim` matrix per head per depth slot, `bdh_cq.py:246-251`; `combine_memories = new + prev`, no decay, `bdh_cq.py:255-257` | Medium | Yes: T2 overwrite and T4 contradiction tasks probe the consequences of no decay; Gated DeltaNet is the "with decay and delta rule" control |
| Memory read: `q @ memory` added to in-sequence attention | PAPER (BDH): readout of rho equals linear attention over accumulated state | CODE: `retrieved = einsum(q, memories)` added to `agg`, `bdh_cq.py:222-224` | High | Covered by Stage B |
| Per-depth-slot memories (one memory per iteration) | PAPER: not specified (verify) | CODE: length-`depth` list of memories, each iteration reads/writes its own slot, `bdh_cq.py:374, 424` | Low | Yes, cheap: one shared memory vs per-slot memories (Stage B side ablation) |
| Normalization | PAPER (BDH): Pathway code uses affine-free LayerNorm; some summaries mention RMSNorm (unresolved) | CODE: parameter-free LayerNorm after embed, after attention, after FF, and once after the depth loop, `bdh_cq.py:168, 228, 341-345, 428` | Medium | No |
| Positional encoding | PAPER (BDH): RoPE on sparse activations (Pathway code uses quantized frequencies, theta 2^16) | CODE: partial RoPE on q,k only over `rotary_dim=64` of `dim_qk_heads`, `bdh_cq.py:204-205`, `rotary.py` | Medium | No |
| Dropout | PAPER: Pathway code has dropout 0.1 | CODE: none | High | No |
| Two-phase operation: context ingestion then latent query reasoning | PAPER (BDH-CQ): yes, qualitatively (memory acquired from demonstrations with fixed weights; query solved in a latent workspace without decoding intermediate tokens) | CODE: `BDHReasoningWrapper` interleaves tensor stages (ingest) and int stages (latent steps), `bdh_cq.py:463-556` | Medium | Framework adopts this split as the common interface |
| Latent transition function (what happens per reasoning step) | PAPER (BDH-CQ): NOT public | CODE: feed the last hidden state back as the next "token" through the same block, `bdh_cq.py:512-535`; the author writes "sans knowing their secretive latent transition function", `bdh_cq.py:412` | Low (GUESS) | Yes: this is the central object of Stage A and Stage C; every recurrence variant in `models/recurrence.py` is a hypothesis about this function |
| Attention residual across reasoning steps with depth bias | PAPER (BDH-CQ): NOT public. The code cites Kimi "Attention Residuals" (2603.15031) and a reverse-RoPE depth-recurrence paper found later (commits `c7049fb`, `77b36b2`) | CODE: `AttentionResidual` learned pseudo-query softmax over all prior block outputs, `bdh_cq.py:94-146, 390-405`; author reports identity residual "collapses" at 8 steps and attention residual is "much more stable", commit `8077188` | Low (GUESS) | Yes: Stage C compares plain, residual, attention-residual, step-gate, init-skip under matched params |
| Latent step embedding | PAPER: not specified | CODE: optional learned `latent_step_embed` added at each latent step, `bdh_cq.py:461, 527-528`; off by default in `figure7.py` | Low (GUESS) | Yes: H6 step-embedding variant |
| Latent-effort curriculum (uniform 0..8 reasoning steps per training step) | PAPER (BDH-CQ): section 7 reportedly describes an effort schedule (verify exact schedule) | CODE: `rng.randint(0, MAX_REASONING_STEPS=8)` per step, `figure7.py:87`; commit `e28abc8` mentions "ramp up curriculum of steps then uniform" | Medium | Yes: H5 fixed vs curriculum vs delayed |
| Latent step training target | PAPER: not specified (verify) | CODE: each latent position predicts the first token of the next tensor segment, `bdh_cq.py:517-522, 545-550` | Low (GUESS) | Yes, design decision: the framework must state its loss explicitly; keep this as the `legacy` loss and add a final-answer-only loss |
| Memory freezing during reasoning | PAPER (BDH-CQ): one summary says memory is frozen after context acquisition | CODE: `update_memory` and `update_latent_memory` flags, `bdh_cq.py:417-422, 463`; `figure7.py` writes prompt to memory by default | Medium | Yes, cheap: frozen vs writable memory during latent steps |
| Higher-order BDH | PAPER (BDH-CQ): "higher-order BDH" named, definition not public | CODE: `HigherOrderBDHLayer` order-2 poly-attention with `(S3, z3)` sufficient statistics, `higher_order_bdh.py:37, 89-128`; cites Chakrabarti et al. ICLR 2026 | Low (GUESS) | Yes, later: order-1 vs order-2 on T5 compose depth 2 (the repo claims order-2 solves two-hop, order-1 does not) |
| Halting or adaptive computation | PAPER: not stated either way | CODE: none; the caller passes an int | n/a | Not in scope initially; fixed R sweeps |
| Model size and hyperparameters of the 150M BDH-CQ model | PAPER: 150M parameters only | CODE: defaults `depth=8, heads=4, dim_qk_heads=32768`, `bdh_cq.py:262-277`; figure7 uses `dim=256, depth=4, dim_qk_heads=1024` (794,145 params) | High for code | Our scale targets are independent |
| Sudoku Extreme 97.4 percent | PAPER (BDH): claimed | Pathway README says not reproducible from the open code | High | Out of scope |
| ARC-AGI-1 29.5 percent pass@2 at $0.0007 per task | PAPER (BDH-CQ): claimed | No code or weights released | High that it is unreleased | Out of scope; we do not chase ARC-AGI |
| ARC-style task families (propagation, copy, order, nesting) | PAPER (BDH-CQ): section 6.2, figure 5 reportedly describe these families | CODE: `tasks.py` implements all four with demo and test difficulty ranges, `tasks.py:92-273` | Medium | Framework reuses them as `legacy_*` variants of T6-T9 |
| Function composition | PAPER (BDH-CQ): compositionality results reported per operation | CODE: `train_function_composition_bdh.py` two-hop `f1(f2(x))` probe using one-hot inputs, not the reasoning wrapper | Medium | Framework's T5 generalizes this to depth 1-8 |

## 2. Gaps in the implementation relevant to controlled experiments

These are not paper gaps but engineering facts that affect experimental
validity. Line numbers refer to the commit above.

1. Depth and recurrence are the same knob. `BDH.depth` unrolls one shared
   block; there is no way to build an unshared fixed-depth BDH without
   new code. The framework's `share_weights: false` option must add this.
2. `generate()` is single-sequence only (`bdh_cq.py:620`). Evaluation at
   scale needs batched decoding; the adapter must add it without changing
   the numerics (test: batched output equals sequential output).
3. The reasoning-step range is untested beyond 8 (`figure7.py:24, 29`).
   Stage A evaluates to 64; expect numerical surprises and instrument for
   them.
4. Structural assertions ("latent reasoning cannot be the final stage")
   fire only in the loss path (`bdh_cq.py:568, 572`); a bare forward that
   ends on an int stage silently returns stale logits. The adapter must
   never call the wrapper that way, and a test should cover it.
5. `update_memory_per_stage` must match the stage count exactly
   (`bdh_cq.py:483-484`).
6. Memory reuse across a reasoning-step sweep is safe only because
   `combine_memories` is out of place; nothing enforces it. The
   framework's `test_context_isolation` guards this.
7. `torch.set_num_threads(8)` is hard-coded (`figure7.py:61`).
8. Seeding uses two streams (global torch/random plus a `random.Random`
   feeding numpy per task, `figure7.py:56-57, 79, 85`). Checkpoint resume
   must capture both.
9. Batch size is 1 task per step in `figure7.py` (`:81-93`), with all four
   task families mixed. This is far from the batch-64 regime the
   framework targets; the Phase 0 reproduction number is therefore only a
   smoke test of the pipeline, not a baseline for Stage A.
10. Training target for latent steps (first token of the next segment) is
    a community choice that couples the latent loop to next-token
    prediction. The framework will offer both this and a final-answer
    loss, and will report which was used.

## 3. What this project will and will not call "BDH-CQ"

- "BDH" in results means: the community `BDHBlock` with shared weights
  across depth, run at R = 1 latent step (or unshared fixed depth when
  `share_weights: false`), with Hebbian memory. It is a faithful
  small-scale reading of the public BDH-GPU formulation, with the
  shared-QK and no-value-projection choices noted.
- "BDH-CQ (community)" means: the same block plus the community
  `BDHReasoningWrapper` latent loop and, when stated, the community
  attention residual. The label always carries "(community)" in tables.
- No configuration in this project is described as Pathway's BDH-CQ.
  Where a result contradicts a Pathway claim, the write-up says the
  contradiction is with the community reconstruction at small scale, not
  with the unreleased model.

## 4. Sources

- BDH paper: Kosowski, Uznanski, Chorowski, Stamirowska, Bartoszkiewicz,
  "The Dragon Hatchling: The Missing Link between the Transformer and
  Models of the Brain", arXiv 2509.26507. https://arxiv.org/abs/2509.26507
- Pathway explainer: https://pathway.com/research/bdh-explainer
- Pathway reference code: https://github.com/pathwaycom/bdh
- BDH-CQ paper: Engdahl, Kosowski, Chorowski, Stamirowska, Uznanski et
  al., "BDH-CQ: In-Context Learning with Recurrent Latent Reasoning",
  arXiv 2608.09888. https://arxiv.org/abs/2608.09888
- Pathway announcement: https://pathway.com/research/introducing-bdh-cq
- Community implementation: https://github.com/lucidrains/bdh-cq
- Kimi Team, "Attention Residuals", arXiv 2603.15031
- Knupp et al., "Depth-Recurrent Attention Mixtures", arXiv 2601.21582
- Chakrabarti, Pitassi, Alman, "Poly-attention", ICLR 2026

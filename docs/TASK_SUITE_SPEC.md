# Task Suite Specification

Status: specification (v0.1). No task code exists yet. This document is the
contract that `tasks/` must satisfy before any Stage A or Stage B run is
launched.

Scope note: the suite deliberately avoids full ARC-AGI. Every task below is
synthetic, generated from a seed, and isolates one mechanism (binding,
overwrite, distraction, composition, propagation, copy, ordering, nesting)
so that a metric change can be attributed to that mechanism.

## 1. Shared task API

Every task is a subclass of `tasks.base.EpisodicTask` and exposes:

```python
@dataclass(frozen=True)
class Episode:
    demonstrations: list[tuple[Tensor, Tensor]]   # (input_tokens, output_tokens)
    query: Tensor                                  # input tokens
    target: Tensor                                 # output tokens
    difficulty: dict[str, int]                     # e.g. {"n_bindings": 8, "depth": 3}
    split: str                                     # "train" | "interp" | "mild" | "strong"
    episode_id: int                                # deterministic from (task_seed, index)

class EpisodicTask(Protocol):
    name: str
    vocab_size: int
    def sample(self, rng: np.random.Generator, difficulty: dict[str, int]) -> Episode: ...
    def train_difficulties(self) -> list[dict[str, int]]: ...
    def eval_difficulties(self) -> dict[str, list[dict[str, int]]]:
        """Returns {"interp": [...], "mild": [...], "strong": [...]}"""
    def score(self, prediction: Tensor, episode: Episode) -> dict[str, float]:
        """Returns at least {"exact_match": 0/1, "token_acc": float}"""
    def serialize(self, episode: Episode) -> Tensor:
        """Flat token sequence with separator tokens; used by sequence models."""
```

Requirements:

- Determinism. `sample(rng, difficulty)` must be a pure function of the
  rng state and difficulty. `episode_id` is derived from
  `(task_seed, split, index)` so that two runs with the same task seed
  evaluate on identical episodes, regardless of model.
- Fresh randomness per episode. Any mapping, permutation, or rule is
  drawn anew each episode. No global table is shared across episodes.
- Vocabulary hygiene. Symbol tokens are drawn from a large pool
  (default 4096 symbols) and remapped through a random permutation per
  episode so that token identity carries no information across episodes.
  Structural tokens (SEP, QUERY, ANSWER, PAD, BOS, EOS, and per-task
  role tokens) are fixed.
- Serialization. `serialize()` produces
  `[BOS] demo_1 [SEP] demo_2 [SEP] ... [QUERY] query [ANSWER] target [EOS]`
  with each demo as `input_tokens [MAP] output_tokens`. Models that
  consume demonstrations natively (BDH memory ingestion) may use the
  structured `Episode` instead; the serialized form is what the
  Transformer and Gated DeltaNet baselines see. Both forms MUST contain
  exactly the same information.
- Splits. Every task defines difficulty ranges for `train`,
  `interp` (inside the training range but unseen episodes), `mild`
  (just beyond the training range), and `strong` (well beyond). The three
  evaluation labels must appear in every results table.
- Episode isolation. The task generator never carries state between
  episodes. The eval harness asserts this by re-sampling with the same
  rng seed and comparing.

## 2. Task catalogue

Each task lists: generator, difficulty knobs, train range, eval ranges,
metric, and the mechanism it isolates. Symbols `a, b, c` denote random
symbol tokens; `->` denotes a demonstration pair.

### T1. Arbitrary associative binding (`binding`)

- Generator: draw `n` distinct keys and `n` values from the symbol pool
  (values may repeat across keys only if `allow_value_collisions=true`,
  default false). Demonstrations are the `n` pairs in random order. Query
  is one key; target is its value.
- Knobs: `n_bindings` in {1, 2, 4, 8, 16, 32, 64}; `key_len`, `val_len`
  (default 1, later 2-3 for multi-token binding).
- Train: `n_bindings` in {1, 2, 4, 8}. Interp: same. Mild: {16}. Strong:
  {32, 64}.
- Metric: exact match on the value; capacity curve = exact match vs
  `n_bindings`.
- Isolates: in-context memory capacity for arbitrary, non-memorizable
  bindings (H2).

### T2. Binding overwrite (`overwrite`)

- Generator: as T1, then for `k` of the keys append a later demonstration
  with a new value. Query one of the overwritten keys. Target is the LATEST
  value. Also emit `stale_target` for scoring "old-answer rate".
- Knobs: `n_bindings`, `n_overwrites`, `gap` (number of demonstrations
  between the original and the overwrite).
- Train: `n_bindings` <= 8, `n_overwrites` <= 2, `gap` <= 4. Mild: `gap`
  8-16. Strong: `n_overwrites` up to `n_bindings`, `gap` up to 32.
- Metrics: exact match, `stale_rate` (fraction answering the old value),
  `other_rate`.
- Isolates: whether memory implements recency-correct overwrite (H2).

### T3. Irrelevant-context robustness (`distractors`)

- Generator: as T1 plus `m` distractor demonstrations whose keys never
  appear in any query. Distractors are interleaved uniformly at random.
- Knobs: `n_bindings`, `distractor_ratio` = m/n in {0, 0.5, 1, 2, 4, 8}.
- Train: ratio <= 1. Mild: 2. Strong: 4, 8.
- Metric: exact match vs ratio; also `distractor_answer_rate` (answered
  with a distractor's value).
- Isolates: interference under memory load (H2).

### T4. Contradictory demonstrations (`contradict`)

- Generator: `n` keys; for `c` keys emit two conflicting pairs
  `a -> b` and `a -> b'` with configurable order. The task exposes two
  target conventions and the config selects one: `recency` (last wins,
  identical to T2) or `majority` (emit 3 demos per key, 2 agree). Default
  reported convention is `recency`; `majority` is a secondary probe.
- Knobs: `n_bindings`, `n_conflicts`, `convention`.
- Metrics: exact match under the convention, `first_rate`, `last_rate`.
- Isolates: recency vs frequency behaviour of the memory (H2).

### T5. Function composition (`compose`)

- Generator: sample `d` random bijections `f_1 .. f_d` over disjoint
  symbol sets (domain of `f_i` = codomain of `f_{i-1}`). Demonstrations
  show `n_examples_per_fn` input/output pairs for each function
  individually. Query: an input `x` in the domain of `f_1`, tagged with
  the composition to apply (the tag is a fixed structural sequence
  `[COMPOSE] d`). Target: `f_d(...f_1(x))`.
- Knobs: `depth` d in {1, 2, 3, 4, 6, 8}; `n_examples_per_fn`;
  `domain_size`.
- Train: depth {1, 2}. Interp: {1, 2}. Mild: {3, 4}. Strong: {6, 8}.
- Metric: exact match vs depth; `partial_depth_acc` (largest prefix depth
  whose intermediate result would have been correct, reconstructed from
  the answer when possible).
- Isolates: iterative application of an in-context-learned rule (H1, H4).
  This is the primary Stage A extrapolation task. If lucidrains/bdh-cq
  ships a function-composition generator, the adapter must reproduce its
  exact distribution as a `legacy` variant so the Phase 0 reproduction
  remains comparable.

### T6. Propagation (`propagate`)

- Generator: 1-D (later 2-D) grid of length L with a source cell marked by
  symbol `s` and empty cells `e`; the demonstration pairs show the rule
  "fill every cell to the right of the source up to a wall `w`". Query is
  a fresh grid; target is the filled grid. Grid symbols are remapped per
  episode.
- Knobs: `length` L, `max_propagation_distance` (train small),
  `n_sources`, `dim` (1 or 2).
- Train: distance <= 4, L <= 12. Mild: distance 6-8. Strong: distance
  12-24, L up to 48.
- Metric: exact match on the grid; `cell_acc`; `max_correct_distance`.
- Isolates: iterative local rule that needs roughly one step per unit
  distance; the canonical "does R_test > R_train help" task (H4).

### T7. Copy (`copy`)

- Generator: random symbol string of length L; demonstrations show copy on
  2-3 short strings; query is a fresh string; target is the same string.
- Knobs: `length` L. Train L in [2, 8]. Mild [9, 16]. Strong [17, 64].
- Metric: exact match, token accuracy, `first_error_position`.
- Isolates: length generalization and positional dependence.

### T8. Ordering (`order`)

- Generator: a random total order over `n` symbols is given by a chain of
  demonstrations `a < b`, `b < c`, ...; query is a pair `(x, y)` or "sort
  this subset". Target: the correct comparison / sorted sequence.
- Knobs: `n_items`, `query_type` in {pair, sort}, `chain_gap` (how many
  hops separate the queried pair).
- Train: `n_items` <= 6, hops <= 2. Mild: hops 3-4. Strong: hops 6-12.
- Metric: exact match vs hops.
- Isolates: transitive multi-hop inference over in-context relations.

### T9. Nested transformations (`nested`)

- Generator: a small set of primitive transformations on short symbol
  strings (reverse, rotate-left-by-1, swap-first-last, duplicate-first,
  drop-last). Demonstrations show each primitive individually (as in T5).
  Query gives a nesting program of depth k, encoded as a bracketed
  sequence of primitive tags, and an input; target is the result.
- Knobs: `depth` k. Train {1, 2}. Mild {3, 4}. Strong {5, 6, 8}.
- Metric: exact match vs depth.
- Isolates: composition of in-context and learned primitives; a harder
  cousin of T5.

### T10. Algorithmic extrapolation labelling (cross-cutting)

Every task's `eval_difficulties()` must return the three labels. The
aggregation tool refuses to produce a plot for a run whose results lack
`split` in {interp, mild, strong}. Do not report a single "test accuracy".

## 3. Difficulty-to-reasoning-steps hypothesis table

For H4 the plan needs an a-priori guess of how many sequential steps a
task instance minimally needs, so that "R_test beyond R_train helps" can be
predicted rather than fitted after the fact.

| Task      | Difficulty knob | Minimal sequential steps (hypothesis) |
|-----------|-----------------|----------------------------------------|
| compose   | depth d         | d                                      |
| propagate | distance        | distance (1-D), distance (2-D, L1)     |
| order     | hops            | ceil(log2(hops)) to hops               |
| nested    | depth k         | k                                      |
| copy      | length L        | 1 (parallel) or L (serial)             |
| binding   | n_bindings      | 1                                      |

The Stage A analysis explicitly compares the observed accuracy-vs-R curve
against this table. Tasks with minimal steps = 1 (binding, copy) serve as
negative controls for recurrence: a mechanism that improves them with
more loops is doing something other than iterative computation.

## 4. Dataset generation and caching

- `tools/generate_tasks.py --task compose --seed 123 --n_train 200000
  --n_eval 2000 --out data/compose_s123/` writes episodes as `.npz` shards
  plus a `manifest.json` (task name, seed, knobs, split counts, git commit,
  generator version).
- Training may also sample online. Online sampling uses a per-worker rng
  derived from `(seed, worker_id, step)`; the config records which mode
  was used.
- Evaluation ALWAYS uses fixed, cached episodes so that every model sees
  identical queries.
- Generator version is bumped whenever the distribution changes; results
  carry the version and the aggregator refuses to merge different
  versions in one plot.

## 5. Unit tests required before any run

- `test_determinism`: same seed => identical episodes for every task.
- `test_fresh_randomness`: across 1000 episodes, the key->value mapping of
  T1 is not constant for any key.
- `test_splits_disjoint`: the difficulty sets of train/mild/strong do not
  overlap.
- `test_serialize_roundtrip`: `serialize()` can be parsed back into the
  same demonstrations/query/target.
- `test_no_state_leak`: generator object reused for two episodes yields
  the same second episode as a fresh generator with the same rng.
- `test_target_correctness`: brute-force reference solver agrees with the
  generator's target for every task (e.g. apply functions literally for
  T5, simulate propagation for T6, sort for T8).
- `test_overwrite_target_is_latest`: T2 target equals the most recent
  value.

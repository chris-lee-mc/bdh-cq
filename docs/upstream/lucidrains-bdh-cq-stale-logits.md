# Upstream report: a forward ending on a latent stage returns stale logits

Status: drafted and verified against `lucidrains/bdh-cq` at `c246f890`
(version 0.0.20), NOT yet filed.

Re-verified 2026-09-07 by running the reproduction below against the pinned
package installed in this project's venv. It still holds: the logits are
byte-identical for reasoning_steps 1, 4 and 16, and `return_loss = True`
correctly refuses the same call with "latent reasoning cannot be the final
stage". The defect is live at that commit.

The "breaks 8 of the 46 tests" claim under "Why the obvious fix is wrong" was
established in an earlier session that could clone the upstream repo. It
CANNOT be re-checked from this session and is reported on that earlier
evidence, not on a check made today. Anyone filing this should re-run the
upstream suite before asserting it.

Why it is still unfiled: this workspace cannot reach `lucidrains/bdh-cq` at
all. `add_repo` refuses cross-owner adds, the GitHub tools deny the
repository as out of session scope, and `list_repos` returns only
`chris-lee-mc` repositories -- the GitHub App is not installed on a third
party's account and cannot be, since that requires admin rights there. No
amount of permission granted on this side changes it. Filing needs a fork
under an account that can push, or a human pasting the text below.

This supersedes the looser wording of `PAPER_IMPLEMENTATION_GAPS.md` item 2.4.
The important correction, found by testing rather than reading: a trailing
latent stage is **intended API surface**, not a misuse. Upstream's own
`generate()` calls the wrapper that way. The first fix drafted here (moving
the two existing asserts above the early return) broke 8 of the 46 upstream
tests and would have been wrong to file.

---

## Title

`BDHReasoningWrapper.forward(..., return_loss = False)` returns stale logits when the last stage is a latent step

## Body

When a call to `BDHReasoningWrapper.forward` ends on an `int` (latent
reasoning) stage and `return_loss = False`, the returned logits are the ones
from the last *tensor* stage -- computed **before** the latent reasoning ran.
They are returned silently, and are identical no matter how many latent steps
were requested.

### Reproduction

```python
import torch
from bdh_cq.bdh_cq import BDH, BDHReasoningWrapper

torch.manual_seed(0)
bdh = BDH(dim = 32, depth = 2, num_tokens = 64, dim_qk_heads = 64, heads = 2, rotary_dim = 16)
wrapper = BDHReasoningWrapper(bdh)
tokens = torch.randint(0, 64, (1, 8))

base = wrapper(tokens, return_loss = False)
for reasoning_steps in (1, 4, 16):
    out = wrapper(tokens, reasoning_steps, return_loss = False)
    print(reasoning_steps, torch.equal(base, out))
```

```
1 True
4 True
16 True
```

The same call is correctly refused under `return_loss = True`:

```
AssertionError: latent reasoning cannot be the final stage
```

### Cause

`bdh_cq/bdh_cq.py`: the `int` branch calls `self.bdh(..., return_logits = False, ...)`
and discards its first return value, so the local `logits` (initialised to
`None` at :488) is never updated by a latent stage. The two guards that would
catch this --

```python
assert exists(logits), 'a tensor stage must follow the latent reasoning'   # :568
assert not isinstance(last(args), int), 'latent reasoning cannot be the final stage'  # :572
```

-- sit **after** the `if not return_loss: ... return pop_if_len_one(returns)`
early exit at :560-566, so they only ever run on the loss path.

### Why the obvious fix is wrong

Moving those asserts above the early return breaks 8 of the 46 tests in
`tests/`. `generate()` (:629-636) deliberately calls
`self(*args, ..., return_memory = True)` with args that may end on an `int`,
discards the logits with `_, memories = ...`, and re-derives the seed from
`memories.embeds[..., -1:, :]`. So the trailing-latent-stage call is
intended; only the stale return value is the problem.

### Suggested fix

Return `None` instead of the stale tensor, so a caller that uses the value
fails loudly rather than silently scoring a pre-reasoning distribution.
`generate()` already discards it, so nothing that works today breaks.

```diff
--- a/bdh_cq/bdh_cq.py
+++ b/bdh_cq/bdh_cq.py
@@ -557,6 +557,16 @@ class BDHReasoningWrapper(Module):

         # return

+        # a run that ends on a latent stage has no logits of its own: the int
+        # branch never produces any, so `logits` would still hold those of the
+        # last TENSOR stage, i.e. from before the reasoning ran.  Return None
+        # rather than those, so a caller that uses them fails loudly instead of
+        # silently scoring a stale distribution.  `generate` already discards
+        # this value and re-derives the seed from `memories.embeds`.
+
+        if isinstance(last(args), int):
+            logits = None
+
         if not return_loss:
             returns = (logits,)
```

Verified: the full upstream suite passes with this applied (46 passed), and

```
forward(tokens, 4)         -> None
forward(tokens, 4, tokens) -> shape (1, 8, 64)
generate(tokens, 4, num_tokens = 3) -> works
```

### Context

Found while building a controlled study of BDH-CQ's recurrence behaviour
against matched baselines. Our adapter never calls the wrapper this way, so
nothing in our results is affected; the report is offered because the failure
is silent and a natural way to ask "what does the model predict after N
reasoning steps" hits it directly.

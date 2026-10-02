# Prompt for filing the stale-logits report from a machine with GitHub access

This workspace cannot reach `lucidrains/bdh-cq` (see the Status section of
`lucidrains-bdh-cq-stale-logits.md`). Paste everything between the `---`
markers below into Claude Code on a machine where `gh auth status` succeeds.

Written for an Intel (x86_64) Mac mini, which matters for one thing only: the
PyTorch wheel. Nothing else in the prompt is platform-specific.

---

I want to file a bug report and fix upstream to `lucidrains/bdh-cq`. I do not
have push access to that repo, so this goes through a fork.

## The defect

`BDHReasoningWrapper.forward(..., return_loss = False)` returns STALE logits
when the last stage is a latent (`int`) reasoning step. The `int` branch never
produces logits, so the local `logits` still holds those of the last TENSOR
stage -- computed before the reasoning ran. They are returned silently and are
byte-identical no matter how many latent steps were requested.

The two guards that would catch this sit AFTER the `if not return_loss:` early
return, so they only ever run on the loss path:

    assert exists(logits), 'a tensor stage must follow the latent reasoning'
    assert not isinstance(last(args), int), 'latent reasoning cannot be the final stage'

I verified this at commit `c246f890` (version 0.0.20). Full write-up, including
cause and prior art:
https://github.com/chris-lee-mc/bdh-cq/blob/main/docs/upstream/lucidrains-bdh-cq-stale-logits.md

## Do this

1. `gh auth status` to confirm you are logged in. Report the account.
2. Fork `lucidrains/bdh-cq` to my account and clone the fork. The name
   `bdh-cq` may already be taken in my account by an unrelated project -- if
   so, fork under a different name and say which.
3. Create a branch, e.g. `fix/stale-logits-on-trailing-latent-stage`.
4. Set up a venv and install the package with its test extras.

   **Intel Mac note:** PyTorch stopped shipping x86_64 macOS wheels after
   2.2.2. If pip cannot resolve a torch version, pin `torch==2.2.2` and say
   that you did. If even that fails, STOP and tell me -- do not file a report
   whose tests you could not run.

5. **Reproduce the bug before changing anything.** Run this and paste the real
   output:

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

   Expected: `True` three times (the bug). `return_loss = True` on the same
   call should raise "latent reasoning cannot be the final stage".

   **If it does NOT reproduce, STOP and tell me.** Upstream may have fixed it;
   do not file.

6. Run the full upstream test suite BEFORE patching and record the pass count.
7. Apply this fix, immediately before the `if not return_loss:` block:

   ```python
           # a run that ends on a latent stage has no logits of its own: the int
           # branch never produces any, so `logits` would still hold those of the
           # last TENSOR stage, i.e. from before the reasoning ran.  Return None
           # rather than those, so a caller that uses them fails loudly instead of
           # silently scoring a stale distribution.  `generate` already discards
           # this value and re-derives the seed from `memories.embeds`.

           if isinstance(last(args), int):
               logits = None

   ```

   Match the file's existing style: it uses spaces around `=` in keyword
   arguments. Follow whatever the surrounding code does.

8. Re-run the reproduction. It should now fail with a `TypeError`/`AttributeError`
   on `None` rather than silently returning stale logits.
9. Re-run the full suite. **It must pass at exactly the same count as step 6.**
   If any test regresses, STOP and report which -- do not file.

10. There is a claim in my write-up I could NOT verify from my environment and
    which you are well placed to check: that the *obvious* fix -- moving those
    two asserts above the early return -- breaks 8 of the 46 tests, because
    `generate()` deliberately calls the wrapper with a trailing latent stage and
    discards the logits. Try that variant on a scratch branch, record the actual
    number of failures, and use the real number in the PR body. If it turns out
    to break nothing, say so plainly and I will rethink which fix to propose.

11. Open the PR from the fork to `lucidrains/bdh-cq` `main`. Check for a PR
    template in the repo first and follow it if present.

    Title: `Return None instead of stale logits when a forward ends on a latent stage`

    Body: state the defect, the reproduction with its real output, the cause
    (asserts after the early return), why moving the asserts is the wrong fix
    with the failure count you measured, and the fix. Keep it short and factual.
    Note that the behaviour was found while using the library, not by reading it.
    Do not speculate about the author's intent beyond what `generate()` shows.

12. Report the PR URL back to me.

## Rules

- Do NOT push to `lucidrains/bdh-cq` directly. Fork and PR only.
- Do NOT file if the bug does not reproduce, if the suite regresses, or if you
  could not run the suite at all. Tell me instead.
- Do NOT change anything beyond this one fix. No formatting passes, no
  unrelated cleanups, no version bumps.
- Use the numbers you actually measure. Do not copy the "8 of 46" figure from
  my write-up without confirming it.
- Be courteous in the PR. This is an unpaid reimplementation of a method whose
  transition function the paper explicitly withholds; the report is a
  contribution, not a complaint.

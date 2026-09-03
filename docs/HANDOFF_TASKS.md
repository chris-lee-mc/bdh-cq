# Handoff: Ordered Implementation Tasks

For the coding agent that implements the framework. Each task is one
reviewable commit or a short series. Do not skip ahead: every task lists
its acceptance test. Tasks marked GPU need a RunPod pod; everything else
runs on CPU.

Rules while implementing:

- State the hypothesis, control, metric, and falsifier in the PR or
  commit body before adding any speculative mechanism.
- Never widen a config's variable set silently; every sweep changes only
  its grid axes.
- ASCII only in code and docs (no em dashes, no unicode arrows).
- No API keys in code or configs.

## Phase 0 (done in this handoff)

- [x] Inspect `lucidrains/bdh-cq` and `pathwaycom/bdh`; run tests (46
      pass); reproduce `figure7.py` on CPU. See `PHASE0_REPRODUCTION.md`.
- [x] Write `EXPERIMENT_PLAN.md`, `PAPER_IMPLEMENTATION_GAPS.md`,
      `FRAMEWORK_SPEC.md`, `TASK_SUITE_SPEC.md`, `RUNPOD.md`, configs.

## Stage 0: framework skeleton (CPU)

- [x] 1. `pyproject.toml` with pinned deps (plan section 15), `bdhx` package,
      ruff and pytest config, `tests/conftest.py` with tiny fixtures.
      Accept: `ruff check`, `pytest` (empty) pass.
- [x] 2. `bdhx/config.py`: pydantic schema from `FRAMEWORK_SPEC.md` section 2,
      YAML loader with `extends:`, canonical hash, unknown-key rejection.
      Accept: `test_config_rejects_unknown_keys`, `test_config_hash_stable`.
- [x] 3. `bdhx/seeding.py`, `bdhx/metadata.py`. Accept: metadata JSON contains
      every field of spec section 4 on CPU.
- [x] 4. `bdhx/tasks/vocab.py`, `tasks/base.py`, then `binding.py`. Accept:
      all section 5 tests of `TASK_SUITE_SPEC.md` for binding.
- [x] 5. Remaining tasks: overwrite, distractors, contradict, compose,
      propagate, copy, order, nested, plus `legacy_*` wrappers over the
      community `tasks.py`. Accept: same tests per task, brute-force
      reference solver agrees for 1000 episodes each.
- [x] 6. `tools/generate_tasks.py` with manifest and generator version.
      Accept: two invocations with the same seed produce identical shards.
- [x] 7. `bdhx/results/schema.py` and writer. Accept: `test_results_roundtrip`.

## Stage 0: models (CPU)

- [x] 8. `bdhx/models/base.py` (`ReasoningModel`, `SolveOutput`, hook counter),
      `param_budget.py`. Accept: `test_param_budget` for a dummy model.
- [x] 9. `models/transformer.py` and `models/looped_transformer.py` with
      input injection and the recurrence kinds table in `models/recurrence.py`.
      Accept: `test_reasoning_steps_runtime`, `test_recurrence_variants_shapes`,
      `test_context_isolation`, `test_batch_independence`.
- [x] 10. `models/bdh.py` adapter over community `BDHBlock`/`BDH`, including
      `share_weights: false`. Document every community call in docstrings.
      Accept: same four tests; parameter count equals community model for
      identical kwargs.
- [x] 11. `models/bdh_cq.py` adapter over `BDHReasoningWrapper` with batched
      decoding. Accept: batched output equals sequential community
      `generate` output token-for-token on 32 episodes in deterministic
      mode; `test_memory_reset_zeroes_state`; a test that a bare wrapper
      call never ends on an int stage.
- [x] 12. `models/gated_deltanet.py`: pure-PyTorch recurrent reference (CPU)
      and `fla` path (GPU) behind one class. Accept: parity test at
      hidden 64, T 32 on CPU vs GPU (GPU test skipped on CPU).
- [x] 13. `training/flops.py` analytic estimates per model. Accept:
      `test_flops_estimate_monotone`; a printed table for 2M, 10M, 25M.

## Stage 0: training and evaluation (CPU)

- [x] 14. `training/trainer.py`: AMP, clipping, schedule, R_train sampling,
      checkpoint with all rng states, SIGTERM handler, NaN policy, wall
      clock limit. Accept: `test_checkpoint_resume` bit-exact on CPU.
- [x] 15. `training/curriculum.py`. Accept: `test_curriculum_schedule`.
- [x] 16. `training/evaluate.py` with split-labelled metrics, reasoning-depth
      sweep, latency, and `training/diagnostics.py` series. Accept: an
      end-to-end tiny run on CPU writes a valid `results.json` with all
      three splits and diagnostics arrays of length R.
- [x] 17. `tools/run_experiment.py` (single config, `--resume`, `--out`,
      `--sync-bucket` optional). Accept: tiny config completes in under 2
      minutes on 4 cores.
- [x] 18. `tools/generate_sweep.py` with seed policy, matched-control
      generation, manifest. Accept: `test_sweep_seed_policy`; expanding
      `a1_first_experiment.yaml` yields 18 configs.
- [x] 19. `tools/profile_config.py`. Accept: prints seconds/step and projected
      minutes for a tiny config on CPU.
- [x] 20. `tools/aggregate_results.py` (`bdhx/results/aggregate.py`): CSVs,
      flags, the six plots, refusal on missing splits. Accept: run on two
      synthetic results directories with 3 fake seeds each.

## Stage 0: cloud (GPU)

No `RUNPOD_API_KEY`, no GPU and no S3 bucket has been reachable from any
session so far, so the parts of tasks 21-23 that need real cloud resources
stay open for the first session that has credentials.

- [x] 21. `docker/Dockerfile`, `docker/entrypoint.sh`, `tools/sync_checkpoint.py`,
      `tools/fetch_latest_checkpoint.py`, plus `bdhx/s3sync.py` (the shared
      key layout and commit ordering) and a real `uv.lock`.

      Verified: `uv.lock` generated and committed; the entrypoint runs
      `configs/base/tiny_smoke.yaml` end to end with `--device cpu`, writes
      `/workspace/runs/smoke`, and resumes correctly on a second invocation;
      with an unreachable bucket it aborts instead of silently cold starting;
      the Dockerfile's dependency lines (`uv export`, the CUDA-stack filter,
      `uv pip install --no-deps`) were run verbatim in a real container build
      and install the framework, boto3 and the RunPod SDK while leaving
      torch/triton/nvidia-* to the base image; the base image tag was checked
      against Docker Hub (the tag the spec used to quote does not exist, see
      RUNPOD.md section 2); `ruff` clean and 388 tests pass.

      NOT verified: the image itself has never been built. The
      `runpod/pytorch` base is about 22 GB extracted and did not fit the free
      disk of the implementing environment, so the acceptance clause "image
      builds" is still open, as is `docker run` proper (the entrypoint was
      exercised directly instead). No S3-compatible bucket has ever been
      contacted: `bdhx/s3sync.py` is unit-tested against a fake client only,
      and its endpoint, addressing and checksum settings come from vendor
      docs, not a live round trip. Both gaps are recorded in RUNPOD.md
      section 10. The first session with a GPU pod and a bucket should build
      the image and do one manual sync/fetch round trip before trusting a
      sweep to either.
- [ ] 22. `tools/runpod_launch.py` (`estimate`, `launch`, `status`, `relaunch`,
      `watchdog`, `reap`, `collect`) and `configs/runpod_rates.yaml`.
      Accept: `estimate` and `reap` unit-tested against a fake SDK;
      one real single-pod smoke run of a tiny config, then `reap`.
- [ ] 23. GPU checks: Triton and `fla` import and a Gated DeltaNet forward in
      the image; `profile_config.py` on each Stage A model at 10M.
- [x] 23b. MANDATORY BEFORE EVERY SWEEP: `python tools/sanity_learnability.py`
      must exit 0. It trains `transformer` (depth 2) and `looped_transformer`
      (depth 2 inside the shared block, R_train {1, 2}) on `binding` for 3000
      CPU steps at 1.5M parameters and requires exact match >= 0.9 on the
      `interp` n_bindings=1 cell at every evaluated R, with a final training
      loss below the `AT_CHANCE` plateau (within 3 percent of ln(vocab_size)).
      `--quick` shrinks it to about 30 seconds for CI at a 0.5 gate.
      Do not launch a sweep, and do not report any table, while this fails:
      the whole of `RESULTS.md` section A0 was produced by a pipeline that ran
      cleanly and learned nothing. Accept: exit 0 and both rows PASS.

## Stage A

- [ ] 24. LR sweep `configs/stage_a/a0_lr_sweep.yaml` (1 seed, compose, task
      seed 2000) for every model at 10M. Record in `RESULTS.md`.
- [x] 25. Launch `a1_first_experiment` (18 jobs). Aggregate. Write the A1
      section of `RESULTS.md` with the accuracy-vs-iterations plot and the
      stability classification per model.
      GENERATED, NOT LAUNCHED: the 18 configs expand correctly and the
      manifest is committed at `docs/generated_manifests/a1_first_experiment.csv`,
      but the jobs need GPUs and were not run. What has been run is
      `configs/stage_a/a1_cpu_mini.yaml`, a 9-job CPU dev replica that
      validates the pipeline only (see `RESULTS.md` section A0). The A1
      results table and the stability classification are still pending.
      Re-run on 2026-09-03 after the init and effective-depth fixes; the 9
      runs are still `AT_CHANCE` on compose at 350k parameters, which is now
      flagged rather than silently reported as accuracy 0.000.
- [ ] 26. Launch `a2_curriculum`. Update `RESULTS.md`. Write the Gate A finding.
- [ ] 27. If Gate A passes: `a3_rtrain_sweep`. If not: diagnosis per plan
      section 10 before anything else. Part of that diagnosis is already done
      and written up in `RESULTS.md` section A0 (init, effective depth,
      AT_CHANCE detection, and the measured cost of the [ANSWER] readout
      position); redo it only for what section A0 leaves open.

## Stage B

- [ ] 28. Confirm the a0 learning rates transfer to binding for Gated DeltaNet
      and Transformer (1 seed, task seed 2000); record.
- [ ] 29. Launch `b1_binding_capacity`, then `b2_memory_robustness`. Write the
      accuracy-vs-associations plot and Gate B finding.
- [ ] 30. Launch `b3_2x2_interaction` (40 jobs, requires `--allow-large-sweep`
      if above the threshold). Report the interaction term with bootstrap
      interval.

## Stage C and D

- [ ] 31. `c1_recurrence_engineering` with matched controls (72 jobs, requires
      `--allow-large-sweep`), then `c2_curriculum_and_delay` (24 jobs).
      Gate C.
- [ ] 32. `d1_state_precision`. Gate D analysis.

## Stage E (optional, after Gate B)

- [ ] 33. `models/transformer_ttt_lora.py` with per-episode adapter fit and
      discard; adaptation cost columns. Launch `e1_ttt_control`.

## After each stage

Write the stage findings section in `RESULTS.md`: attempted, worked,
failed, confidence, compute spent, next experiment and why. Update the
compute ledger.

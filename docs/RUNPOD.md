# RunPod Execution Guide

Status: specification (v0.1), researched September 2026. The framework
itself is cloud-independent; this document describes the launcher and the
operational rules for running sweeps on RunPod.

Sourcing caveat: runpod.io and docs.runpod.io were not reachable from the
research environment. SDK signatures below come from the public
`runpod/runpod-python` repository and are considered reliable. Prices,
preemption notice windows, network-volume rates, and region lists were
triangulated from several 2026 third-party sources and MUST be re-checked
against runpod.io/pricing and docs.runpod.io before the first paid sweep.
Items known to be unverified are listed in section 10.

## 1. Which hardware

Models in this project are 2M-25M parameters (later 50M-150M) trained in
bf16 with tiny VRAM needs. The cost driver is $/hour and interruption
tolerance, not VRAM.

| GPU               | Community (spot-like) | Secure (on-demand) | Use                          |
|-------------------|-----------------------|--------------------|------------------------------|
| RTX A5000 24GB    | ~$0.16/hr             | ~$0.27/hr          | default for Stage A-D sweeps |
| RTX 4090 24GB     | ~$0.34/hr             | ~$0.69-0.74/hr     | faster small jobs            |
| L4 24GB           | n/a                   | ~$0.39/hr          | alternative                  |
| A40 48GB          | n/a                   | ~$0.44/hr          | 50M-150M models              |
| L40S 48GB         | n/a                   | ~$0.99/hr          | 50M-150M models              |
| A100 80GB PCIe    | ~$1.19/hr             | ~$1.39/hr          | not needed before 100M+      |
| H100 80GB SXM     | ~$2.69/hr             | ~$2.99/hr          | not needed                   |

Rules:

- Default: RTX A5000 or RTX 4090 on Community Cloud, one job per GPU.
  Parallelize across seeds and configs, never across GPUs for one run.
- Use Secure Cloud only when a job is long enough that restarts would
  cost more than the price premium (rule of thumb: runs above 3 hours).
- Never use A100/H100 for sub-50M models. Kernel-launch overhead
  dominates and the price is 4-10x for no gain.
- Community Cloud preemption: assume near-zero notice. Sources disagree
  between about 5 seconds and a few minutes. Checkpointing, not a SIGTERM
  handler, is the safety net. The trainer still installs a SIGTERM handler
  as best effort.

## 2. Environment: Docker image

`docker/Dockerfile` pins everything. Skeleton:

```dockerfile
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# torch is provided by the base image; assert its version at build time
RUN python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__"
COPY pyproject.toml uv.lock /app/
WORKDIR /app
RUN pip install uv && uv pip install --system -r pyproject.toml --extra gpu
COPY . /app
RUN uv pip install --system -e . --no-deps
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

- `uv.lock` pins every dependency, including `triton` and
  `flash-linear-attention` (Gated DeltaNet baseline). Triton needs the
  CUDA driver on the host, which the RunPod image provides. Smoke-test in
  the built image before any sweep:
  `python -c "import triton, fla; print(triton.__version__)"` plus one
  tiny `GatedDeltaNet` forward on GPU.
- Build and push: `docker build -t ghcr.io/<owner>/bdhx:<git-sha> .` then
  `docker push`. The image tag is the git SHA; the launcher records it.
- Local smoke test (CPU-only machines use `--device cpu`):
  `docker run --rm --gpus all -e RUN_ID=smoke -e RUNPOD_API_KEY=dummy
  ghcr.io/<owner>/bdhx:<sha> --config configs/base/tiny_smoke.yaml`.
- Check the current `runpod/pytorch` tags on Docker Hub at build time; tag
  naming has changed between schemes (for example
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`).

Secrets: `RUNPOD_API_KEY`, S3 credentials, and any logging tokens are
environment variables only. `pyproject.toml`, configs, and docker files
never contain them. A pre-commit hook greps for `RUNPOD_API_KEY=` and
AWS-style key patterns.

## 3. Storage and checkpoint/resume

Two tiers:

1. Local scratch (container disk or network volume at `/workspace`) for
   the run directory during training.
2. External S3-compatible bucket (Cloudflare R2 or Backblaze B2 for low
   egress; RunPod's own S3 API on network volumes in supported regions is
   an alternative) as the authoritative store for checkpoints, results,
   and metadata. `S3_BUCKET` and credentials arrive through env vars.

Protocol (implemented in `bdhx/training/trainer.py` and
`tools/sync_checkpoint.py`):

- Checkpoint every `checkpoint_every_steps` (default 1000 steps) AND at
  least every 5 minutes of wall clock, whichever comes first. Checkpoints
  for models this size are megabytes, so the upload is synchronous.
- Write local, upload to `s3://<bucket>/runs/<run_id>/ckpt_<step>.pt`,
  then update `latest.json` (checkpoint key, step, config hash) with a
  write-then-rename, then delete local checkpoints older than the last two.
- On start, the entrypoint fetches `latest.json` for `RUN_ID`. If present
  and the config hash matches, training resumes with model, optimizer,
  scheduler, and rng states; `metadata.json` increments `preemptions`.
  A hash mismatch aborts (never silently train a different config under
  an old run id).
- Results (`results.json`, `train_log.csv`, `metadata.json`, logs) are
  uploaded at the end and after each evaluation.

Network volumes: about $0.07/GB-month (unverified). They are region
pinned; if used, create the volume in a region that also supports the
S3-compatible API (reported: EUR-IS-1, EU-RO-1, EU-CZ-1, US-KS-2, US-CA-2;
verify).

## 4. Entrypoint (`docker/entrypoint.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ID:?}"; : "${S3_BUCKET:?}"; : "${CONFIG_PATH:?}"; : "${MAX_SECONDS:=10800}"
[[ "${DETERMINISTIC:-0}" == "1" ]] && export CUBLAS_WORKSPACE_CONFIG=":4096:8"
mkdir -p /workspace/runs/$RUN_ID/meta
nvidia-smi > /workspace/runs/$RUN_ID/meta/nvidia-smi.txt || true
python tools/fetch_latest_checkpoint.py --bucket "$S3_BUCKET" --run-id "$RUN_ID" \
    --dest /workspace/runs/$RUN_ID || true
set +e
timeout --signal=TERM "${MAX_SECONDS}s" python tools/run_experiment.py \
    --config "$CONFIG_PATH" --run-id "$RUN_ID" --resume \
    --out /workspace/runs/$RUN_ID --sync-bucket "$S3_BUCKET"
TRAIN_EXIT=$?
set -e
python tools/sync_checkpoint.py --bucket "$S3_BUCKET" --run-id "$RUN_ID" --final || true
python - <<'PY'
import os, runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PY
exit "$TRAIN_EXIT"
```

The self-terminate step runs unconditionally. `RUNPOD_POD_ID` is injected
by RunPod; `RUNPOD_API_KEY` is not and must be passed via `env` in
`create_pod` (use a narrowly scoped key). If `timeout` fires, the trainer
has already checkpointed, so relaunching the same `RUN_ID` resumes.

## 5. Launcher (`tools/runpod_launch.py`)

Uses the `runpod` Python SDK (pip `runpod`, observed latest 1.12.0,
Python >= 3.10). Core call:

```python
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.create_pod(name=f"bdhx-{sweep}-{idx}", template_id=template_id,
                  gpu_type_id="NVIDIA RTX A5000", cloud_type="COMMUNITY",
                  env={"RUN_ID": run_id, "CONFIG_PATH": path, "S3_BUCKET": bucket,
                       "MAX_SECONDS": str(max_s), "RUNPOD_API_KEY": key, ...})
```

Other SDK functions used: `get_pod`, `get_pods`, `terminate_pod`,
`create_template`. Confirm `create_template`'s keyword arguments against
`examples/api/create_template.py` in the SDK repo; otherwise call the
GraphQL `saveTemplate` mutation at `https://api.runpod.io/graphql`.

Subcommands and behaviour:

| subcommand | behaviour |
|------------|-----------|
| `estimate <generated/sweep>` | reads `manifest.csv` GPU-minute estimates (from `profile_config.py`), multiplies by the rate table in `configs/runpod_rates.yaml`, prints jobs, GPU-hours, USD |
| `launch <generated/sweep>` | refuses above `--max-gpu-hours` (default 20) or above `--max-jobs` (default 12) without `--allow-large-sweep`; respects `--max-concurrent`; appends every created pod to `runpod_state.jsonl` BEFORE returning |
| `status` | polls `get_pod` for tracked pods; classifies RUNNING / EXITED / MISSING; MISSING with an incomplete `latest.json` is queued for relaunch with the same `RUN_ID` |
| `relaunch` | relaunches queued jobs (resume is automatic) |
| `watchdog` | terminates any tracked pod whose uptime exceeds 1.5x its `MAX_SECONDS`; run as a loop or cron |
| `reap --prefix bdhx-<sweep>` | lists all pods with the prefix and terminates them; called in a `finally` block of `launch` and manually after every sweep |
| `collect <generated/sweep>` | downloads `runs/<run_id>/` results for every job into `results/` for `aggregate_results.py` |

Cost safety checklist enforced by the launcher:

1. `profile_config.py` must have been run for each distinct model/param
   cell (the manifest carries its estimates; missing estimates block
   launch).
2. Printed projection: jobs x hours x $/hr, plus a 1.3x contingency for
   preemptions.
3. Explicit `--allow-large-sweep` above the thresholds.
4. Every job has `MAX_SECONDS` (from `compute.max_wall_clock_minutes`).
5. `reap` runs at the end, always.
6. The state file is append-only so orphaned pods created during a
   launcher crash can still be found and terminated.

Alternative worth a prototype before over-investing in the launcher:
SkyPilot lists RunPod as a supported cloud (RunPod docs and blog announce
the integration) and its managed jobs handle spot preemption recovery.
Keep `run_experiment.py` launcher-agnostic so either path works.

## 6. Sweep workflow

```
python tools/generate_sweep.py configs/stage_a/a1_first_experiment.yaml   # -> generated/a1_first_experiment/
python tools/profile_config.py generated/a1_first_experiment/exp_000.yaml --device cuda
python tools/runpod_launch.py estimate generated/a1_first_experiment
python tools/runpod_launch.py launch generated/a1_first_experiment --max-concurrent 8
python tools/runpod_launch.py status
python tools/runpod_launch.py collect generated/a1_first_experiment
python tools/aggregate_results.py results/ --out reports/$(date +%F)/
python tools/runpod_launch.py reap --prefix bdhx-a1_first_experiment
```

Each generated config is an independent job. Seeds are separate jobs.

## 7. Reproducibility inside the job

`bdhx/metadata.py` records, before training starts: GPU name and count,
`nvidia-smi` text, driver and CUDA versions, cuDNN version, torch and
Python versions, git commit and dirty flag, image tag, `RUNPOD_POD_ID`,
datacenter id if exposed, pip freeze. Deterministic mode
(`torch.use_deterministic_algorithms(True)` plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA init) is opt-in via
`compute.deterministic`; it is used for the Phase 0 reproduction and for
tests, not for sweeps, because affected ops can be substantially slower.
Measure the cost once on a representative config and record it.

## 8. Expected compute for the initial deliverables

Estimates assume RTX A5000 Community at about $0.16/hr and about 40 to 60
minutes per 10M-parameter, 40k-step run at batch 64 with R_train <= 4 and
evaluation through R_test = 32. These are planning numbers; the
profiling step replaces them.

| sweep | jobs | GPU-hours | USD (with 1.3x contingency) |
|-------|------|-----------|-----------------------------|
| A1 first experiment (3 models x 2 tasks x 3 seeds) | 18 | ~15 | ~3.5 |
| A2 curriculum (2 x 2 x 2 x 3) | 24 | ~20 | ~4.5 |
| B1 binding capacity (3 x 3) | 9 | ~6 | ~1.5 |
| B2 memory robustness (3 x 3 x 3) | 27 | ~18 | ~4 |
| B3 2x2 interaction (2 x 2 x 2 x 5) | 40 | ~35 | ~8 |
| C1 recurrence engineering (4 kinds + 2 controls) x 2 tasks x 3 seeds | 36 | ~35 | ~8 |
| D1 precision (3 x 2 x 3) | 18 | ~15 | ~3.5 |

Even with a 3x error the whole initial programme is well under $150 on
Community Cloud. The scale-up phases (25M, 50M, 100-150M) are budgeted
separately after Gate A and Gate B.

## 9. Alternatives (one paragraph each)

- Vast.ai: marketplace pricing close to or below RunPod Community for
  RTX 4090 (about $0.29-0.39/hr), higher host variance. Benchmark
  head-to-head if RunPod capacity is short.
- Lambda Cloud: on-demand only, no cheap small-GPU tier, prices rising in
  2026. Reasonable no-interruption fallback, not the cheap option.
- Modal: per-second serverless functions, excellent developer experience,
  roughly Secure-Cloud prices. Worth it if launcher maintenance becomes
  the bottleneck.
- SkyPilot: orchestration layer with RunPod support and managed spot
  recovery; could replace most of section 5.

## 10. Unverified items (re-check before first paid sweep)

- All prices and the network-volume $/GB-month rate.
- Community Cloud interruption notice window.
- REST API base URL; the GraphQL endpoint is the reliable path.
- `runpodctl pod create` flag for a startup command (use the SDK's
  `docker_args` or a template instead).
- Whether the SDK auto-reads `RUNPOD_API_KEY` (set `runpod.api_key`
  explicitly regardless).
- Triton and flash-linear-attention inside `runpod/pytorch` images
  (smoke-test the built image).
- `create_template` signature.
- S3-API region list for network volumes.

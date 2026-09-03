#!/usr/bin/env bash
# Entrypoint for RunPod jobs (docs/RUNPOD.md sections 3 and 4).
#
# S3 is OPTIONAL. With S3_BUCKET set, this follows the section 3 protocol:
# fetch latest.json for RUN_ID, resume if the config hash matches, sync every
# checkpoint during training, upload results at the end. With S3_BUCKET unset
# or empty -- the case for every run configured today -- training still runs
# and results are left in /workspace/runs/$RUN_ID for
# `tools/runpod_launch.py collect` to pull over scp. That mirrors what the
# launcher's own inline startup command already does.
#
# Extra arguments are forwarded to tools/run_experiment.py. `--device <dev>`
# is translated into the config override the runner understands, so the
# section 2 local smoke test works unchanged:
#
#   docker run --rm -e RUN_ID=smoke -e CONFIG_PATH=configs/base/tiny_smoke.yaml \
#       bdhx:<sha> --device cpu
set -euo pipefail

: "${RUN_ID:?RUN_ID must be set}"
: "${CONFIG_PATH:?CONFIG_PATH must be set}"
: "${MAX_SECONDS:=10800}"
S3_BUCKET="${S3_BUCKET:-}"

[[ "${DETERMINISTIC:-0}" == "1" ]] && export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# run_experiment.py appends the run id to --out itself, so --out is the parent.
RUNS_ROOT=/workspace/runs
OUT="$RUNS_ROOT/$RUN_ID"
mkdir -p "$OUT/meta"
nvidia-smi > "$OUT/meta/nvidia-smi.txt" 2>&1 || true

# Translate `--device X` into an override; forward everything else verbatim.
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            EXTRA+=(--overrides "compute.device=$2")
            shift 2
            ;;
        --device=*)
            EXTRA+=(--overrides "compute.device=${1#*=}")
            shift
            ;;
        *)
            EXTRA+=("$1")
            shift
            ;;
    esac
done

SYNC=()
if [[ -n "$S3_BUCKET" ]]; then
    # Deliberately not `|| true`: a transport error is not the same as "no
    # checkpoint". Treating it as a cold start would restart from step 0 and
    # then overwrite latest.json, destroying the run's real progress. Exit 3
    # is a config hash mismatch, which must never be retried (section 3).
    python tools/fetch_latest_checkpoint.py --bucket "$S3_BUCKET" --run-id "$RUN_ID" \
        --dest "$OUT" --config "$CONFIG_PATH"
    SYNC=(--sync-bucket "$S3_BUCKET")
else
    echo "S3_BUCKET is not set: results stay on local disk at $OUT (collect them over scp)"
fi

# --resume requires a checkpoint to exist; on a cold start there is none.
RESUME=()
[[ -f "$OUT/checkpoints/latest.json" ]] && RESUME=(--resume)

set +e
timeout --signal=TERM "${MAX_SECONDS}s" python tools/run_experiment.py \
    --config "$CONFIG_PATH" --run-id "$RUN_ID" --out "$RUNS_ROOT" \
    "${RESUME[@]}" "${SYNC[@]}" "${EXTRA[@]}"
TRAIN_EXIT=$?
set -e

if [[ -n "$S3_BUCKET" ]]; then
    python tools/sync_checkpoint.py --bucket "$S3_BUCKET" --run-id "$RUN_ID" \
        --run-dir "$OUT" --final || true
fi

# Self-terminate only when the pod was given a key to do it with. The launcher
# does not distribute the account key to every pod, so this is normally a
# no-op and termination is handled by `runpod_launch.py reap` / `watchdog`.
if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
python - <<'PY'
import os

import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PY
fi

exit "$TRAIN_EXIT"

#!/usr/bin/env bash
# Draft entrypoint for RunPod jobs; see docs/RUNPOD.md section 4.
set -euo pipefail
: "${RUN_ID:?}"; : "${S3_BUCKET:?}"; : "${CONFIG_PATH:?}"; : "${MAX_SECONDS:=10800}"
[[ "${DETERMINISTIC:-0}" == "1" ]] && export CUBLAS_WORKSPACE_CONFIG=":4096:8"
OUT=/workspace/runs/$RUN_ID
mkdir -p "$OUT/meta"
nvidia-smi > "$OUT/meta/nvidia-smi.txt" 2>&1 || true
python tools/fetch_latest_checkpoint.py --bucket "$S3_BUCKET" --run-id "$RUN_ID" --dest "$OUT" || true
set +e
timeout --signal=TERM "${MAX_SECONDS}s" python tools/run_experiment.py \
    --config "$CONFIG_PATH" --run-id "$RUN_ID" --resume --out "$OUT" --sync-bucket "$S3_BUCKET"
TRAIN_EXIT=$?
set -e
python tools/sync_checkpoint.py --bucket "$S3_BUCKET" --run-id "$RUN_ID" --final || true
if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
python - <<'PY'
import os, runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PY
fi
exit "$TRAIN_EXIT"

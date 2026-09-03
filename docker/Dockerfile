# Image for RunPod jobs (docs/RUNPOD.md section 2).
#
# Build from the bdh-cq-experiments directory, which is the build context:
#   docker build -f docker/Dockerfile -t bdhx:$(git rev-parse --short HEAD) .
#
# Base image tag: RunPod renamed these between schemes, and the tag the spec
# originally quoted (2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu) does not
# exist on Docker Hub. Checked 2026-09-03 against the runpod/pytorch tag list;
# 1.1.0-cu1281-torch280-ubuntu2404 is the newest stable (non-rc) tag carrying
# torch 2.8.0 on CUDA 12.8.1, which is the pairing this project targets.
FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv

# uv.lock is the pin for everything except the CUDA stack. Exporting the lock
# and installing it with --no-deps means the resolved versions are used
# verbatim, with no re-resolution at build time.
#
# torch, triton and nvidia-* are filtered out on purpose: the base image
# already ships them, built against its own driver and CUDA. `uv lock` cannot
# know that and resolves torch to the newest release, so installing them from
# the lock would silently replace the image's CUDA stack with a generic wheel.
# The asserts below are what prove the filter worked.
COPY pyproject.toml uv.lock /app/
RUN uv export --frozen --no-hashes --no-emit-project \
        --extra gpu --extra s3 --extra runpod -o /tmp/requirements.txt \
 && grep -vE '^(torch|triton|nvidia-)' /tmp/requirements.txt > /tmp/requirements-nocuda.txt \
 && uv pip install --system --no-deps -r /tmp/requirements-nocuda.txt

# The community BDH-CQ package, pinned to the commit the adapters were written
# against. Kept as its own step (not a lock entry) so it matches how
# tools/runpod_launch.py's inline startup command installs it.
RUN uv pip install --system --no-deps \
        "bdh-cq @ git+https://github.com/lucidrains/bdh-cq@c246f890"

COPY . /app
RUN uv pip install --system --no-deps -e .

# Section 2 requires this smoke test before any sweep. Failing at build time is
# the point: an image whose CUDA stack was clobbered must never reach a pod.
RUN python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__" \
 && python -c "import triton, fla; print('triton', triton.__version__)" \
 && python -c "import boto3; print('boto3', boto3.__version__)"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

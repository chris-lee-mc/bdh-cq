# Draft; finalize when pyproject.toml and uv.lock exist (HANDOFF task 21).
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__"
WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN pip install uv && uv pip install --system -r pyproject.toml --extra gpu
COPY . /app
RUN uv pip install --system -e . --no-deps
RUN python -c "import triton, fla; print('triton', triton.__version__)"
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

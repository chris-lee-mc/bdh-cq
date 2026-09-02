from __future__ import annotations

import pytest
import torch

from bdhx.config import PROJECT_ROOT, load_config

torch.set_num_threads(1)

TINY_CONFIG = PROJECT_ROOT / "configs" / "base" / "tiny_smoke.yaml"


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results" / "run_test"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def tiny_cfg():
    """Loader for configs/base/tiny_smoke.yaml with optional dotted overrides."""

    def _load(**overrides):
        return load_config(TINY_CONFIG, overrides or None)

    return _load

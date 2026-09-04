"""bdhx/training/diagnostics.py: activation_stats device handling.

Regression coverage for a real bug found on a GPU verification pod: every
profile_config.py run and every evaluation with `evaluation.diagnostics:
true` (set in configs/stage_a/a1_first_experiment.yaml) crashed on CUDA with
`RuntimeError: quantile() q tensor must be on the same device as the input
tensor`, because the percentile tensor was constructed with no `device=`
and defaulted to CPU while the activations under test were on GPU. CPU-only
tests cannot regress-test this by construction (both tensors are CPU there),
which is exactly how it slipped through 403 passing CPU tests; the CUDA
case below is the actual regression guard and only runs where CUDA exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdhx.training.diagnostics import PERCENTILES, activation_stats


def test_activation_stats_on_cpu():
    x = torch.randn(4, 8, 16)
    active, pcts, bad = activation_stats(x)
    assert 0.0 <= active <= 1.0
    assert len(pcts) == len(PERCENTILES)
    assert bad == 0


def test_activation_stats_flags_nan_and_inf():
    x = torch.randn(4, 8, 16)
    x[0, 0, 0] = float("nan")
    x[0, 0, 1] = float("inf")
    _, _, bad = activation_stats(x)
    assert bad == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device in this environment")
def test_activation_stats_on_cuda():
    x = torch.randn(4, 8, 16, device="cuda")
    active, pcts, bad = activation_stats(x)
    assert 0.0 <= active <= 1.0
    assert len(pcts) == len(PERCENTILES)
    assert bad == 0

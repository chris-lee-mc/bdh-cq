import pytest
import torch
from torch import nn

from bdhx.models.param_budget import solve_width


class Dummy(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3 * width, width))


@pytest.mark.parametrize("target", [2_000_000, 10_000_000, 25_000_000])
def test_solver_hits_target(target):
    width, params = solve_width(Dummy, target, width_max=4096, step=8)
    assert params == 3 * width * width
    assert abs(params - target) / target <= 0.03


def test_solver_raises_when_out_of_range():
    with pytest.raises(ValueError):
        solve_width(Dummy, 10_000_000_000, width_max=256, step=8)

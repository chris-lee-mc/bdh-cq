"""Width solver hitting a parameter budget (FRAMEWORK_SPEC section 2)."""

from __future__ import annotations

from collections.abc import Callable

HARD_TOL = 0.05


def _params(ctor: Callable[[int], object], width: int, cache: dict[int, int]) -> int:
    if width not in cache:
        obj = ctor(width)
        if hasattr(obj, "param_report"):
            cache[width] = int(obj.param_report().trainable)
        elif hasattr(obj, "parameters"):
            cache[width] = int(sum(p.numel() for p in obj.parameters() if p.requires_grad))
        else:
            cache[width] = int(obj)  # ctor may return a raw parameter count
    return cache[width]


def solve_width(
    model_ctor_from_width: Callable[[int], object],
    params_target: int,
    tol: float = 0.03,
    width_min: int = 16,
    width_max: int = 4096,
    step: int = 8,
) -> tuple[int, int]:
    """Monotone (binary) search for the width whose trainable params match target.

    Assumes params increase with width. Returns (width, realized_params) and
    raises ValueError if the best candidate is outside HARD_TOL (5 percent).
    """
    if params_target <= 0:
        raise ValueError("params_target must be positive")
    cache: dict[int, int] = {}
    lo = (width_min // step) * step or step
    hi = (width_max // step) * step
    if lo > hi:
        raise ValueError("empty width range")

    best = lo
    if _params(model_ctor_from_width, lo, cache) >= params_target:
        best = lo
    elif _params(model_ctor_from_width, hi, cache) <= params_target:
        best = hi
    else:
        a, b = lo, hi
        while b - a > step:
            mid = ((a + b) // 2 // step) * step
            mid = min(max(mid, a + step), b - step)
            if _params(model_ctor_from_width, mid, cache) < params_target:
                a = mid
            else:
                b = mid
        best = min(
            (a, b),
            key=lambda w: abs(_params(model_ctor_from_width, w, cache) - params_target),
        )

    realized = _params(model_ctor_from_width, best, cache)
    rel = abs(realized - params_target) / params_target
    if rel > HARD_TOL:
        raise ValueError(
            f"width solver missed target: width={best} params={realized} "
            f"target={params_target} rel_err={rel:.3f} (tol={tol}, hard={HARD_TOL})"
        )
    return best, realized

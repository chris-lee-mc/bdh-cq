"""Recurrence engineering variants (FRAMEWORK_SPEC section 3.1).

All kinds share one block callable `F(H, S)` and differ only in the update
rule applied between reasoning steps. Gates/embeddings are sized `R_max`
(the largest depth seen in training); steps beyond that reuse the last
trained value (`gate_extrapolation: hold_last`) or linearly interpolate
(`interpolate`, Stage C sub-ablation).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

KINDS = (
    "plain",
    "residual",
    "step_gate",
    "init_skip",
    "step_emb",
    "adapter",
    "attn_residual",
    "combo",
)

DIAG_KEYS = ("state_norm", "update_norm", "cos_consecutive", "nan_count")


class AttnResidual(nn.Module):
    """Learned-query softmax over prior block outputs with distance-to-end bias.

    Generic reimplementation of the community `bdh_cq.py:94-146` residual: the
    keys are the seed state plus every block output so far, the query is a
    learned vector (tied across depth), and the similarity is shifted by a
    learned scalar times the key's distance from the end of reasoning.
    """

    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.key_norm = nn.RMSNorm(width)
        self.bias_scale = nn.Parameter(torch.zeros(()))

    def forward(self, seed: Tensor, outs: list[Tensor]) -> Tensor:
        keys = torch.stack([seed, *outs], dim=-2)  # (B, N, L, D)
        sim = torch.einsum("d,bnld->bnl", self.query, self.key_norm(keys))
        n = keys.shape[-2]
        dist = torch.arange(n - 1, -1, -1, device=keys.device, dtype=sim.dtype)
        sim = sim + self.bias_scale * dist
        return torch.einsum("bnl,bnld->bnd", sim.softmax(dim=-1), keys)


class RecurrenceRunner(nn.Module):
    """Applies `block` exactly `reasoning_steps` times under the chosen rule.

    `block(H, S) -> H` is held in a tuple so that an `nn.Module` block is never
    registered as a submodule: the owning model stays the single owner of the
    block parameters and `param_report()` cannot double count them.
    """

    def __init__(
        self,
        block: Callable[[Tensor, Tensor | None], Tensor],
        kind: str = "plain",
        r_max: int = 8,
        width: int = 0,
        adapter_rank: int = 0,
        gate_extrapolation: str = "hold_last",
    ):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown recurrence kind '{kind}'; expected one of {KINDS}")
        if gate_extrapolation not in ("hold_last", "interpolate"):
            raise ValueError(f"unknown gate_extrapolation '{gate_extrapolation}'")
        self._block = (block,)
        self.kind = kind
        self.r_max = max(int(r_max), 1)
        self.width = int(width)
        self.adapter_rank = int(adapter_rank)
        self.gate_extrapolation = gate_extrapolation

        if kind in ("step_gate", "combo"):
            self.alpha = nn.Parameter(torch.ones(self.r_max))
        if kind == "init_skip":
            self.skip_gate = nn.Parameter(torch.zeros(self.r_max))
        if kind in ("step_emb", "combo"):
            self.step_emb = nn.Parameter(torch.zeros(self.r_max, self.width))
        if kind in ("adapter", "combo") and self.adapter_rank > 0:
            k = self.adapter_rank
            self.adapter_a = nn.Parameter(torch.zeros(self.r_max, self.width, k))
            self.adapter_b = nn.Parameter(torch.randn(self.r_max, k, self.width) * 0.02)
        if kind == "attn_residual":
            self.attn_residual = AttnResidual(self.width)

    @property
    def block(self) -> Callable[[Tensor, Tensor | None], Tensor]:
        return self._block[0]

    # -- gate lookup -------------------------------------------------------
    def _gate(self, param: Tensor, r: int, total: int) -> Tensor:
        if r < self.r_max:
            return param[r]
        if self.gate_extrapolation == "hold_last" or self.r_max == 1:
            return param[self.r_max - 1]
        pos = r * (self.r_max - 1) / max(total - 1, 1)
        pos = min(pos, float(self.r_max - 1))
        lo = int(pos)
        hi = min(lo + 1, self.r_max - 1)
        w = pos - lo
        return param[lo] * (1.0 - w) + param[hi] * w

    def _adapter(self, h: Tensor, r: int, total: int) -> Tensor:
        if self.adapter_rank <= 0:
            return torch.zeros((), device=h.device, dtype=h.dtype)
        a = self._gate(self.adapter_a, r, total)
        b = self._gate(self.adapter_b, r, total)
        return (h @ b.transpose(-1, -2)) @ a.transpose(-1, -2)

    # -- diagnostics -------------------------------------------------------
    @staticmethod
    def empty_diagnostics() -> dict[str, list]:
        return {k: [] for k in DIAG_KEYS}

    @staticmethod
    def _record(diag: dict[str, list], prev: Tensor, cur: Tensor) -> None:
        with torch.no_grad():
            f_prev, f_cur = prev.flatten(1).float(), cur.flatten(1).float()
            diag["state_norm"].append(float(f_cur.norm(dim=-1).mean()))
            diag["update_norm"].append(float((f_cur - f_prev).norm(dim=-1).mean()))
            diag["cos_consecutive"].append(
                float(F.cosine_similarity(f_prev, f_cur, dim=-1, eps=1e-8).mean())
            )
            diag["nan_count"].append(int(torch.isnan(cur).sum()))

    # -- main loop ---------------------------------------------------------
    def forward(
        self,
        h0: Tensor,
        s: Tensor | None,
        reasoning_steps: int,
        collect_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, list]]:
        r_total = int(reasoning_steps)
        if r_total < 0:
            raise ValueError("reasoning_steps must be >= 0")
        diag = self.empty_diagnostics()
        h = h0
        outs: list[Tensor] = []
        for r in range(r_total):
            prev = h
            if self.kind == "plain":
                h = self.block(h, s)
            elif self.kind == "residual":
                h = h + self.block(h, s)
            elif self.kind == "step_gate":
                h = h + self._gate(self.alpha, r, r_total) * self.block(h, s)
            elif self.kind == "init_skip":
                h = self.block(h, s) + self._gate(self.skip_gate, r, r_total) * h0
            elif self.kind == "step_emb":
                h = self.block(h + self._gate(self.step_emb, r, r_total), s)
            elif self.kind == "adapter":
                h = self.block(h, s) + self._adapter(h, r, r_total)
            elif self.kind == "attn_residual":
                outs.append(self.block(h, s))
                h = self.attn_residual(h0, outs)
            else:  # combo: step_gate + step_emb + adapter (Gate C composition)
                gated = self._gate(self.alpha, r, r_total) * self.block(
                    h + self._gate(self.step_emb, r, r_total), s
                )
                h = h + gated + self._adapter(h, r, r_total)
            if collect_diagnostics:
                self._record(diag, prev, h)
        return h, diag

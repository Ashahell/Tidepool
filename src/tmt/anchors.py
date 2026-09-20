"""Minimal MARCH-style anchor bank (EXPERIMENTAL).

Periodic snapshots of the recurrent state become content-addressable
anchors: learned key projection, softmax routing over the bank plus a
learned null option, readout fused by the caller. Values are stored
detached (content needs no grad); keys are stored attached ONLY in
defer/BPTT mode (record=True) — in single-step mode the graph is
freed each step, so recording attached keys would hold dead tensors.
Top-layer only in v1.
"""
import math
import torch
import torch.nn as nn


class AnchorBank(nn.Module):
    def __init__(self, dim: int, dk: int, max_anchors: int):
        super().__init__()
        self.dim, self.dk, self.max_anchors = dim, dk, max_anchors
        self.key_proj = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.query_proj = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.values: list = []
        self.keys: list = []

    def reset(self) -> None:
        self.values.clear()
        self.keys.clear()

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def checkpoint(self, s: torch.Tensor, record: bool = False) -> None:
        k = self._norm(s @ self.key_proj.T)
        if not record:
            k = k.detach()
        self.keys.append(k)
        self.values.append(s.detach())
        while len(self.values) > self.max_anchors:
            self.values.pop(0)
            self.keys.pop(0)

    def retrieve(self, h: torch.Tensor):
        """Returns (readout (1,dim), weights (n+1,) incl. null last)."""
        q = self._norm(h @ self.query_proj.T)
        if not self.keys:
            return torch.zeros_like(h), torch.ones(1)
        K = torch.cat(self.keys, dim=0)  # (n, dk)
        V = torch.cat(self.values, dim=0)  # (n, dim)
        scores = torch.cat([(q @ K.T).squeeze(0) / math.sqrt(self.dk),
                            self.null_logit.view(1)], dim=0)
        w = torch.softmax(scores, dim=0)
        r = (w[:-1].unsqueeze(0) @ V)
        return r, w

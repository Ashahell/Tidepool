"""Delta-rule fast-weight associative memory (EXPERIMENTAL, v1).

Ba et al. 2016; Schlag et al. 2021 (DeltaNet). S += b(v - S k) k^T
with unit-norm keys; read r = S q; step returns h + r.
No RTRL trace terms in v1: slow weights get single-step autograd
credit only. S is a persistent buffer, zeroed by reset().
"""
import math
import torch
import torch.nn as nn


class FastWeightMemory(nn.Module):
    def __init__(self, dim: int, dk: int):
        super().__init__()
        self.dim, self.dk = dim, dk
        self.Wk = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.Wq = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.Wv = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.wb = nn.Parameter(torch.zeros(dim))
        self.bb = nn.Parameter(torch.zeros(()))
        self.register_buffer("S", torch.zeros(dim, dk))

    def reset(self) -> None:
        with torch.no_grad():
            self.S.zero_()

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def step(self, h: torch.Tensor) -> torch.Tensor:
        k = self._norm(h @ self.Wk.T)  # (1, dk)
        q = self._norm(h @ self.Wq.T)  # (1, dk)
        v = h @ self.Wv.T  # (1, d)
        beta = torch.sigmoid(h @ self.wb + self.bb).mean()
        vpred = (self.S @ k.T).T  # (1, d)
        self.S = self.S + beta * ((v - vpred).T @ k)  # (d, dk) outer
        return h + (self.S @ q.T).T  # (1, d)

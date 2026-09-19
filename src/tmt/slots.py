# src/tmt/slots.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMemory(nn.Module):
    """Content-addressed sidecar store. Read-before-write per step.

    Documented split: writes are a non-learned Hebbian cache (rebind
    (out-of-place) cache, no gradient); reads learn through query/key
    projections every step. Temperature fixed at 1.0 (v1).
    """

    def __init__(self, dim: int, n_slots: int = 16):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        # Small random init (not zeros): zero rows make every write weight
        # uniform, leaving all rows identical, which zeroes the read-path
        # gradient (out independent of attention) and freezes retrieval_loss.
        # reset() still returns to exact zeros (test_reset_clears).
        self.register_buffer("slots", torch.randn(n_slots, dim) * 0.1)

    def reset(self) -> None:
        with torch.no_grad():
            self.slots.zero_()

    def write(self, x: torch.Tensor) -> None:
        v = x.detach().reshape(-1)
        with torch.no_grad():
            w = F.softmax(self.slots @ v, dim=0)
            new = self.slots * (1.0 - w.unsqueeze(1)) + w.unsqueeze(1) * v.unsqueeze(0)
            # Rebind (not in-place): _decode saves the slots tensor for the
            # backward pass, and any in-place update bumps its version and
            # breaks loss.backward() in the same training step.
            self.slots = new

    def read(self, q: torch.Tensor) -> torch.Tensor:
        # Grad flows into query/key projections (what to retrieve);
        # the slots buffer itself carries no history grad (cache writes).
        v = q.reshape(-1)
        a = F.softmax(self.key(self.slots) @ self.query(v), dim=0)
        return (a.unsqueeze(0) @ self.slots).squeeze(0)

    def retrieval_loss(self, read_vec: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(read_vec.reshape(-1),
                                         target_embed.detach().reshape(-1), dim=0)

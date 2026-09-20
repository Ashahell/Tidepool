# src/tmt/slots.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMemory(nn.Module):
    """Content-addressed sidecar store. Read-before-write per step.

    v2 addressing: shared key projection on the write side, temperature
    scaling, usage counters with protect scaling, erase blend.
    Documented split: writes are a non-learned cache (no gradient);
    reads learn through query/key projections every step.

    Deviations from the task-1 brief's verbatim snippet (documented cause):
    - slots init randn*0.1, not zeros: zero rows make every write weight
      uniform, leaving all rows identical, which zeroes the read-path
      gradient and freezes retrieval_loss (v1 tests
      test_read_path_differentiable / test_aux_loss_decreases go RED).
      reset() still returns to exact zeros (test_reset_clears).
    - write uses rebind (not in-place mul_/add_): _decode saves the slots
      tensor for the backward pass, and any in-place update bumps its
      version and breaks loss.backward() in the same training step.
    """

    PROTECT_THRESH = 5.0
    PROTECT_GAIN = 1.0

    def __init__(self, dim: int, n_slots: int = 16, temp: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.cfg_temp = temp
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        # Small random init (not zeros): see docstring.
        self.register_buffer("slots", torch.randn(n_slots, dim) * 0.1)
        self.register_buffer("usage", torch.zeros(n_slots))

    def reset(self) -> None:
        with torch.no_grad():
            self.slots.zero_()
            self.usage.zero_()

    def _scores(self, v: torch.Tensor) -> torch.Tensor:
        return (self.key(self.slots) @ self.query(v)) / self.cfg_temp

    def write(self, x: torch.Tensor) -> None:
        v = x.detach().reshape(-1)
        with torch.no_grad():
            w = F.softmax(self._scores(v), dim=0)
            protect = torch.sigmoid(self.PROTECT_GAIN * (self.PROTECT_THRESH - self.usage))
            w = w * protect
            self.usage.add_(w)
            new = self.slots * (1.0 - w.unsqueeze(1)) + w.unsqueeze(1) * v.unsqueeze(0)
            # Rebind (not in-place): see docstring.
            self.slots = new

    def read(self, q: torch.Tensor) -> torch.Tensor:
        # Grad flows into query/key projections (what to retrieve);
        # the slots buffer itself carries no history grad (cache writes).
        v = q.reshape(-1)
        a = F.softmax(self._scores(v), dim=0)
        return (a.unsqueeze(0) @ self.slots).squeeze(0)

    def retrieval_loss(self, read_vec: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(read_vec.reshape(-1),
                                         target_embed.detach().reshape(-1), dim=0)

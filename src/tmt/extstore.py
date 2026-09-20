"""True external key-value store (EXPERIMENTAL, new hypothesis class).

Distinct from slots (position-logit addressing) and anchors
(state snapshots): a separately-owned matrix with learned
write/erase/protect and strictly content-addressed reads.
Per step: content-similar erase (scaled by 1-protect), then a gated
write into the oldest unprotected slot (tick - P*protect, argmin).
Values detached always; keys attached ONLY in defer/BPTT mode
(record=True) — single-step mode frees graphs per step, so only
query/null/decoder get grads there (documented limitation).
"""
import math
import torch
import torch.nn as nn

PROTECT_BIAS = 1e4


class ExternalStore(nn.Module):
    def __init__(self, dim: int, dk: int, nslots: int):
        super().__init__()
        self.dim, self.dk, self.nslots = dim, dk, nslots
        self.Wk = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.Wq = nn.Parameter(torch.randn(dk, dim) / math.sqrt(dim))
        self.Wv = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.ww = nn.Parameter(torch.zeros(dim))
        self.wb = nn.Parameter(torch.zeros(()))
        self.we = nn.Parameter(torch.zeros(dim))
        self.eb = nn.Parameter(torch.zeros(()))
        self.wp = nn.Parameter(torch.zeros(dim))
        self.pb = nn.Parameter(torch.zeros(()))
        self.null_logit = nn.Parameter(torch.zeros(()))
        # Learned sharpness: scorestasks are cosine-scale (unit keys), so
        # without temperature a full bank is inherently diffuse. The model
        # raises temp_scale to concentrate; init 1.0 = plain /sqrt(dk).
        self.temp_log = nn.Parameter(torch.zeros(()))
        self.register_buffer("K", torch.zeros(nslots, dk))
        self.register_buffer("V", torch.zeros(nslots, dim))
        self.protect: list[float] = [0.0] * nslots
        self.write_tick: list[int] = [-1] * nslots
        self.tick = 0

    def reset(self) -> None:
        with torch.no_grad():
            self.K.zero_()
            self.V.zero_()
        self.protect = [0.0] * self.nslots
        self.write_tick = [-1] * self.nslots
        self.tick = 0

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def write(self, h: torch.Tensor, record: bool = False) -> int:
        k = self._norm(h @ self.Wk.T)  # (1, dk)
        v = h @ self.Wv.T  # (1, dim)
        qw = torch.sigmoid(h @ self.ww + self.wb)  # (1,)
        qe = torch.sigmoid(h @ self.we + self.eb)  # (1,)
        qp = float(torch.sigmoid(h @ self.wp + self.pb).detach())
        kv = k.squeeze(0)
        # Content-similar erase, blocked by per-slot protect.
        sims = (self.K @ kv).clamp_min(0.0)  # (nslots,)
        prot = torch.tensor(self.protect)
        V_new = self.V * (1.0 - qe * sims * (1.0 - prot)).unsqueeze(1)
        # Oldest unprotected slot (argmin tick + P*protect: protected
        # slots score ~+1e4 and are avoided; unprotected pick oldest).
        scores = [t + PROTECT_BIAS * p
                  for t, p in zip(self.write_tick, self.protect)]
        j = min(range(self.nslots), key=lambda i: scores[i])
        K_new = self.K.clone()
        K_new[j] = kv
        V_new = V_new.clone()
        V_new[j] = (qw * v).squeeze(0)
        if record:
            self.K = K_new
            self.V = V_new
        else:
            self.K = K_new.detach()
            self.V = V_new.detach()
        self.protect[j] = qp
        self.write_tick[j] = self.tick
        self.tick += 1
        return j

    def retrieve(self, h: torch.Tensor):
        """Returns (readout (1,dim), weights (nslots+1,) incl. null last)."""
        q = self._norm(h @ self.Wq.T)
        raw = (q @ self.K.T).squeeze(0) / math.sqrt(self.dk)
        raw = raw * torch.exp(self.temp_log)
        # Never-written slots must not attract mass.
        used = torch.tensor([t >= 0 for t in self.write_tick])
        raw = torch.where(used, raw, torch.full_like(raw, -1e9))
        scores = torch.cat([raw, self.null_logit.view(1)], dim=0)
        w = torch.softmax(scores, dim=0)
        r = w[:-1].unsqueeze(0) @ self.V
        return r, w

# src/tmt/model.py
from __future__ import annotations
from typing import List, Optional, Tuple
from collections import deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TMTConfig

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.embed(x)

class ByteDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(x), torch.sigmoid(self.stop(x))

class RTULayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.decay_bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.register_buffer("states", torch.zeros(1, dim))
        self.register_buffer("decaytrace", torch.zeros(dim))
        self.register_buffer("embedtrace", torch.zeros(256, dim))

    def forward(self, enc: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay_bias)
        state = decay * self.states + enc
        out = x + self.silu(self.weights(self.norm(state)))
        return out, state, decay

class TMTModel(nn.Module):
    def __init__(self, cfg: TMTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.dim)
        self.decoder = ByteDecoder(cfg.dim)
        self.layers = nn.ModuleList([RTULayer(cfg.dim) for _ in range(cfg.layers)])
        self.opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr)
        self._accum = 0
        self.last_components: dict = {}
        self.replay_buf = (deque(maxlen=cfg.replay_size)
                           if cfg.replay_size > 0 else None)

    def reset(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                layer.states.zero_()
                layer.decaytrace.zero_()
                layer.embedtrace.zero_()
        self._accum = 0

    def init_decay_groups(self, half_lives=[8.0, 64.0, 512.0, 4000.0]):
        with torch.no_grad():
            for layer in self.layers:
                dim = layer.dim
                g = len(half_lives)
                idx = torch.arange(dim) % g
                targets = torch.tensor([0.5 ** (1.0 / h) for h in half_lives])
                chosen = targets[idx].clamp(1e-4, 1.0 - 1e-4)
                layer.decay_bias.copy_(torch.log(chosen / (1.0 - chosen)))

    def decay_diversity_penalty(self):
        pens = []
        for layer in self.layers:
            d = torch.sigmoid(layer.decay_bias)
            pens.append(-torch.var(d))
        return torch.stack(pens).mean()

    def forward(self, x: torch.LongTensor):
        enc = self.encoder(x)
        h = enc
        states: List[torch.Tensor] = []
        for layer in self.layers:
            h, state, _ = layer(enc, h)
            states.append(state)
        logits, stop = self.decoder(h)
        return logits, states

    def _sample(self, logits: torch.Tensor) -> int:
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            entropy = float(-(probs * (probs + 1e-8).log()).sum() / math.log(256))
            temp = max(0.1, self.cfg.temp * (1.0 - self.cfg.temp * entropy))
            return int(torch.distributions.Categorical(logits=logits / temp).sample().item())

    def _update(self, curr: int, next_: Optional[int], end: bool):
        self.train()
        c = torch.tensor([curr], dtype=torch.long)
        enc = self.encoder(c)
        h = enc
        states, decays = [], []
        for layer in self.layers:
            h, state, decay = layer(enc, h)
            states.append(state)
            decays.append(decay)
        logits, stop = self.decoder(h)
        x = h
        loss = torch.maximum(
            torch.tensor(0.0),
            1.0 - torch.sqrt(x.var(unbiased=False) + 1e-4),
        ) * self.cfg.w_var
        if self.cfg.decay_groups > 1:
            loss = loss + 0.01 * self.decay_diversity_penalty()
        t_pred = t_ce = t_stop = None
        if next_ is not None:
            with torch.no_grad():
                tgt = self.encoder(torch.tensor([next_], dtype=torch.long))
            t_pred = self.cfg.w_pred * torch.mean((x - tgt) ** 2)
            t_ce = self.cfg.w_ce * (F.cross_entropy(logits.view(-1, 256), torch.tensor([next_])))
            target_stop = torch.tensor([[1.0 if end else 0.0]])
            t_stop = self.cfg.w_stop * torch.mean((stop - target_stop) ** 2)
            loss = loss + t_pred + t_ce + t_stop
        with torch.no_grad():
            parts = [t for t in (t_pred, t_ce, t_stop) if t is not None]
            l_var = float((loss - sum(parts)).detach()) if parts else float(loss.detach())
            comp = {
                "l_var": l_var,
                "l_pred": float(t_pred.detach()) if t_pred is not None else 0.0,
                "l_ce": float(t_ce.detach()) if t_ce is not None else 0.0,
                "l_stop": float(t_stop.detach()) if t_stop is not None else 0.0,
                "state_norm": float(sum(torch.linalg.norm(s).detach() for s in states)),
            }
        self.opt.zero_grad()
        loss.backward()
        # RTRL trace update (matches MLX dummy-gradient correction).
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                d = decays[i].detach()
                s = states[i].detach()
                one_hot = torch.zeros_like(layer.embedtrace)
                one_hot[curr] += 1.0
                layer.embedtrace.mul_(d).add_(one_hot)
                layer.decaytrace.mul_(d).add_(d * (1.0 - d) * layer.states.squeeze(0))
                layer.states.copy_(s)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip)
        self._accum += 1
        if self._accum >= self.cfg.update_every:
            self.opt.step()
            self.opt.zero_grad()
            self._accum = 0
        return loss.detach(), logits.detach(), stop.detach(), comp

    def training_step(self, curr: int, next_: Optional[int], end: bool):
        self.train()
        loss, logits, stop, comp = self._update(curr, next_, end)
        if self.replay_buf is not None:
            self.replay_buf.append((curr, next_, end))
            for _ in range(self.cfg.replay_k):
                if not self.replay_buf:
                    break
                c, n, e = self.replay_buf[torch.randint(len(self.replay_buf), (1,)).item()]
                self._update(c, n, e)
        self.last_components = comp
        with torch.no_grad():
            sampled = self._sample(logits)
            stop_v = float(stop.item())
        return loss, sampled, stop_v

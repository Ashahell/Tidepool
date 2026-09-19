# tests/test_rtrl.py
import numpy as np
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

# Gradient classification (also documented in model.py above the
# correction block): embedding + decay_bias carry temporal influence
# through persistent state and get RTRL corrections; recurrent weight,
# LayerNorm, decoder, and stop-head affect only the current step, so
# plain autograd is exact for them. Every class below must match FD.

def _seq_model():
    cfg = TMTConfig(dim=3, layers=1, update_every=1000, decay_groups=1,
                    grad_clip=float("inf"))
    m = TMTModel(cfg).double()
    torch.manual_seed(4)
    for p in m.parameters():
        torch.nn.init.uniform_(p, -0.5, 0.5)
    m.reset()
    return m

def _total_loss(m, seq):
    m.reset()
    tot = 0.0
    for i in range(len(seq) - 1):
        loss, _, _ = m.training_step(seq[i], seq[i + 1], i == len(seq) - 2)
        tot += float(loss.item())
    return tot

def _fd_grad(m, seq, param, eps=1e-5):
    # NO no_grad wrapper here: _total_loss runs training_step, whose
    # backward() needs an enabled grad graph (a no_grad wrapper raises
    # RuntimeError). No optimizer step occurs (update_every=1000).
    g = torch.zeros_like(param.detach(), dtype=torch.float64)
    flat_p, flat_g = param.detach().reshape(-1), g.reshape(-1)
    for j in range(flat_p.numel()):
        orig = flat_p[j].item()
        flat_p[j] = orig + eps
        lp = _total_loss(m, seq)
        flat_p[j] = orig - eps
        lm = _total_loss(m, seq)
        flat_p[j] = orig
        flat_g[j] = (lp - lm) / (2 * eps)
    return g

def _all_params(m):
    L = m.layers[0]
    return {"embed": m.encoder.embed.weight, "decay": L.decay_bias,
            "w": L.weights.weight, "ln_w": L.norm.weight,
            "ln_b": L.norm.bias, "dec_w": m.decoder.decode.weight,
            "dec_b": m.decoder.decode.bias, "stop_w": m.decoder.stop.weight,
            "stop_b": m.decoder.stop.bias}

def _check(seq, names):
    m = _seq_model()
    _total_loss(m, seq)  # accumulates grads, update_every=1000 so no step
    skip_rows = set(seq[1:])  # pred-target rows: upstream stop_gradient
    for name in names:  # excludes them, so FD truth contains a target-role
        p = _all_params(m)[name]  # term no faithful implementation reports
        got = p.grad.detach().clone().double()
        ref = _fd_grad(m, seq, p)
        if name == "embed":
            mask = torch.ones(ref.shape[0], dtype=torch.bool)
            mask[list(skip_rows)] = False
            got, ref = got[mask], ref[mask]
        rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)
        assert rel.item() < 1e-4, (name, len(seq) - 1, rel.item())

def test_rtrl_one_step_all_classes():
    _check([65, 66], list(_all_params(_seq_model()).keys()))

def test_rtrl_two_step_all_classes():
    _check([65, 66, 67], list(_all_params(_seq_model()).keys()))

def test_rtrl_five_step_temporal_classes():
    _check([65, 66, 67, 68, 69, 70], ["embed", "decay", "w"])

def _selective_model():
    cfg = TMTConfig(dim=3, layers=1, update_every=1000, decay_groups=1,
                    grad_clip=float("inf"), selective=True)
    m = TMTModel(cfg).double()
    torch.manual_seed(4)
    for p in m.parameters():
        torch.nn.init.uniform_(p, -0.5, 0.5)
    m.reset()
    return m

def test_selective_off_matches_plain_forward():
    torch.manual_seed(6)
    a = TMTModel(TMTConfig(dim=8, layers=1))
    torch.manual_seed(6)
    b = TMTModel(TMTConfig(dim=8, layers=1, selective=True))
    x = torch.tensor([65], dtype=torch.long)
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.equal(la, lb)

def test_selective_matches_fd():
    m = _selective_model()
    seq = [65, 66, 67, 68]
    _total_loss(m, seq)
    skip = set(seq[1:])  # pred-target rows excluded: upstream stop_gradient
    names = list(_all_params(m).keys()) + ["gate_w"]
    for name in names:
        p = _all_params(m)[name] if name != "gate_w" else m.layers[0].gate_w
        got = p.grad.detach().clone().double()
        ref = _fd_grad(m, seq, p)
        if name == "embed":
            mask = torch.tensor([i not in skip for i in range(256)])
            got, ref = got[mask], ref[mask]
        rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)
        assert rel.item() < 1e-4, (name, rel.item())

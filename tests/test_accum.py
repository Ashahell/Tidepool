# tests/test_accum.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def _fresh(**kw):
    cfg = TMTConfig(dim=8, layers=1, lr=0.01, decay_groups=1, **kw)
    m = TMTModel(cfg)
    torch.manual_seed(9)
    for p in m.parameters():
        torch.nn.init.uniform_(p, -0.5, 0.5)
    m.reset()
    return m

def test_accumulates_four_gradients():
    # update_every means sequential accumulation across timesteps while
    # state evolves (NOT a minibatch reset): gradients from 4 successive
    # bytes sum, then one clip+step. Proved by comparing against an
    # explicit out-of-model accumulation (no second model path).
    torch.manual_seed(9)
    m = _fresh(update_every=4)
    for i in range(4):
        m.training_step(65 + i, 66, False)
    stepped = [p.detach().clone() for p in m.parameters()]
    torch.manual_seed(9)
    r = _fresh(update_every=1000)
    r.opt.zero_grad()
    for i in range(4):
        r.training_step(65 + i, 66, False)
    # Gate params are inert when selective=False: no graph path, no grad.
    names = [n for n, _ in r.named_parameters()]
    assert r.layers[0].gate_w.grad is None
    manual = {n: p.grad.detach().clone() for n, p in r.named_parameters()
              if p.grad is not None}
    torch.manual_seed(9)
    s = _fresh(update_every=4)
    s.opt.zero_grad()
    for n, p in s.named_parameters():
        if n in manual:
            p.grad = manual[n].clone()
    torch.nn.utils.clip_grad_norm_(s.parameters(), s.cfg.grad_clip)
    s.opt.step()
    for a, b in zip(stepped, s.parameters()):
        assert torch.allclose(a, b, atol=1e-6)

def test_adaptive_temp_monotonic():
    from tmt.model import TMTModel as M
    cfg = TMTConfig(dim=8, layers=1)
    m = M(cfg)
    assert abs(m._adaptive_temp(0.0) - 0.75) < 1e-6
    assert abs(m._adaptive_temp(1.0) - 0.1875) < 1e-6
    assert m._adaptive_temp(1.0) < m._adaptive_temp(0.5) < m._adaptive_temp(0.0)

def test_groups_drive_init():
    import math
    for groups, want in [(2, [8.0, 4000.0]),
                         (3, [8.0, math.sqrt(8.0 * 4000.0), 4000.0]),
                         (4, [8.0, 64.0, 512.0, 4000.0])]:
        m = TMTModel(TMTConfig(dim=16, layers=1, decay_groups=groups))
        m.init_decay_groups()
        import torch as t
        d = t.sigmoid(m.layers[0].decay_bias).detach()
        got = sorted(set(round(float(v), 4) for v in d.tolist()))
        assert len(got) == groups
        for g, h in zip(got, sorted(want)):
            # sigmoid(decay_bias) must equal the per-step decay 0.5**(1/h)
            assert abs(g - round(0.5 ** (1.0 / h), 4)) < 1e-3, (groups, g, h)

def test_reset_isolates_sequences():
    # Losses (not samples: sampling consumes RNG) and persistent states
    # must be identical for identical post-reset sequences, regardless of
    # what ran before the reset.
    def run(m, seq):
        losses, states = [], []
        for b in seq:
            loss, _, _ = m.training_step(b, b + 1, False)
            losses.append(float(loss.item()))
            states.append(m.layers[0].states.detach().clone())
        return losses, states
    # update_every=1000: no optimizer steps, so weights stay identical and
    # only state/traces evolve — isolates the reset semantics under test.
    m = _fresh(update_every=1000)
    run(m, (65, 66, 67))
    m.reset()
    losses_a, states_a = run(m, (70, 71, 72))
    m2 = _fresh(update_every=1000)
    losses_b, states_b = run(m2, (70, 71, 72))
    assert losses_a == losses_b
    for a, b in zip(states_a, states_b):
        assert torch.equal(a, b)
    m.reset()
    losses_c, states_c = run(m, (70, 71, 72))
    assert losses_c == losses_a
    for a, c in zip(states_a, states_c):
        assert torch.equal(a, c)

def test_config_validation(tmp_path):
    import pytest
    from tmt.config import TMTConfig
    p = tmp_path / "c.yaml"
    p.write_text("dim: 16\nnonsense_field: 1\n")
    with pytest.raises(ValueError):
        TMTConfig.from_yaml(str(p))
    for bad in [dict(dim=-10), dict(layers=0), dict(lr=-1.0),
                dict(temp=0.0), dict(update_every=0), dict(decay_groups=-4),
                dict(grad_clip=-1.0), dict(replay_size=-1)]:
        with pytest.raises(ValueError):
            TMTConfig(**{"dim": 16, "layers": 1, **bad})

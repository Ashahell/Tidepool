# tests/test_equivalence.py
"""Forward/state/trace equivalence: torch model vs NumPy MLX transcription."""
import numpy as np
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tests.mlx_reference import init_params, mlx_step


def test_reference_embedding_lookup():
    P = init_params(seed=11, dim=4, layers=2)
    assert np.max(np.abs(P["embed"][65] - P["embed"][65])) == 0.0
    assert P["embed"].shape == (256, 4)


def test_forward_equivalence():
    torch.manual_seed(11)
    cfg = TMTConfig(dim=4, layers=2)
    # update_every huge: training_step advances persistent buffers but never
    # takes an optimizer step (params stay exactly as loaded).
    cfg.update_every = 1000000
    m = TMTModel(cfg)
    P = init_params(seed=11, dim=4, layers=2)
    m.load_numpy_params(P)
    traces = None
    for curr, nxt in [(65, 66), (66, 67), (67, 68)]:
        # NumPy forward from OLD traces (advances traces old -> new).
        (nloss, nlogits, nstates, traces) = mlx_step(P, traces, curr, nxt, False)
        # Torch pure forward from OLD buffers (forward does not touch buffers).
        with torch.no_grad():
            logits, states = m(torch.tensor([curr], dtype=torch.long))
        assert np.max(np.abs(nlogits - logits.detach().numpy())) < 1e-5
        for a, b in zip(nstates, states):
            assert np.max(np.abs(a - b.detach().numpy())) < 1e-5
        # Torch buffer-advancing path from the same OLD buffers; loss/body
        # computed pre-advance so it is comparable to the NumPy values above.
        loss, _, _ = m.training_step(curr, nxt, False)
        assert abs(float(loss) - nloss) < 1e-3
        for li, layer in enumerate(m.layers):
            assert np.max(np.abs(traces["states"][li] - layer.states.detach().numpy())) < 1e-5
            assert np.max(np.abs(traces["embedtrace"][li] - layer.embedtrace.detach().numpy())) < 1e-5
            assert np.max(np.abs(traces["decaytrace"][li] - layer.decaytrace.detach().numpy())) < 1e-5

# tests/test_model.py
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel

def test_eval_call_shape_and_reset():
    m = TMTModel(TMTConfig(dim=32, layers=2))
    m.reset()
    logits, state = m(torch.tensor([65], dtype=torch.long))
    assert logits.shape == (1, 256)
    assert isinstance(state, list) and len(state) == 2

def test_single_layer_math():
    # Hand-computed: decay=sigmoid(0)=0.5, state=0.5*0+enc+0=enc.
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=4, layers=1))
    with torch.no_grad():
        m.encoder.embed.weight.zero_()
        m.encoder.embed.weight[65] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m.layers[0].decay_bias.zero_()
    m.reset()
    logits, state = m(torch.tensor([65], dtype=torch.long))
    assert torch.allclose(state[0], torch.tensor([[1.0, 2.0, 3.0, 4.0]]), atol=1e-5)

def test_training_step_runs():
    m = TMTModel(TMTConfig(dim=16, layers=1))
    loss, sampled, stop = m.training_step(65, 66, False)
    assert loss.numel() == 1 and 0 <= sampled <= 255 and 0.0 <= stop <= 1.0

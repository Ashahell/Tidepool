# tests/test_gate.py
import torch


def test_cuda_live():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 0)


def test_real_model_eval_smoke():
    from tmt.config import TMTConfig
    from tmt.model import TMTModel
    from tmt.evaluation_suite import EvalConfig, evaluate_model
    m = TMTModel(TMTConfig(dim=32, layers=2, update_every=2))
    m.init_decay_groups()
    cfg = EvalConfig(max_bytes=400, lm_eval_tokens=64, copy_lengths=[4], intervening_lengths=[8], num_copy_trials=1, retention_eval_bytes=64)
    val = bytes([(65 + i % 26) for i in range(256)])
    r = evaluate_model(m, val, [val], cfg=cfg, config_dict={"dim": 32}, gpu_hours=0.0)
    assert r.val_bpb < 20.0 and r.stability_score >= 0.0

# tests/test_train_utils.py
import random
import sys
sys.path.insert(0, "scripts")
from train import ss_pick


def test_ss_pick_truth_by_default():
    rng = random.Random(0)
    assert ss_pick(rng, 0.0, 99, 65) == 65
    assert ss_pick(rng, 1.0, None, 65) == 65


def test_ss_pick_uses_prediction():
    rng = random.Random(0)
    assert ss_pick(rng, 1.0, 99, 65) == 99
    outs = {ss_pick(rng, 0.5, 99, 65) for _ in range(20)}
    assert outs == {65, 99}

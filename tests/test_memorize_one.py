# tests/test_memorize_one.py — THE GATE: one fixed episode, zero generalization.
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel
from tmt.data import QUERY_MARKER

PAYLOAD = [10, 20, 30, 40]
FILLER = [1, 2, 3, 4]
# True copy-episode shape (data.copy_episode): payload+filler+MARKER+payload.
EPISODE = PAYLOAD + FILLER + list(QUERY_MARKER) + PAYLOAD
NTRAIN = len(PAYLOAD) + len(FILLER) + len(QUERY_MARKER)


def train_one(m: TMTModel, reps: int) -> None:
    for _ in range(reps):
        m.reset()
        for i, b in enumerate(EPISODE):
            nxt = EPISODE[i + 1] if i + 1 < len(EPISODE) else None
            m.training_step(b, nxt, nxt is None)


def teacher_forced_hits(m: TMTModel) -> int:
    m.eval()
    hits = 0
    with torch.no_grad():
        m.reset()
        for i, b in enumerate(EPISODE[:-1]):
            logits, _ = m(torch.tensor([b], dtype=torch.long))
            if int(logits[0].argmax()) == EPISODE[i + 1]:
                hits += 1
    m.train()
    return hits


def free_generate(m: TMTModel, n: int) -> list[int]:
    m.eval()
    out = []
    with torch.no_grad():
        m.reset()
        for b in EPISODE[:NTRAIN]:
            m.ingest(b)
        cur = QUERY_MARKER[-1]
        for _ in range(n):
            logits, _ = m(torch.tensor([cur], dtype=torch.long))
            cur = int(logits[0].argmax())
            out.append(cur)
    m.train()
    return out


def test_memorize_one_episode():
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=32, layers=2))
    # 300 reps leaves the query transition at rank 2 (null attractor
    # wins); 2000 converges it to rank 1. Amount of training, not
    # architecture — this number is the gate's calibration.
    train_one(m, 2000)
    print(f"TF hits: {teacher_forced_hits(m)}/{len(EPISODE) - 1}")
    assert free_generate(m, len(PAYLOAD)) == PAYLOAD

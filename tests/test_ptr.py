# tests/test_ptr.py — span-supervised pointer loss (final boxed attempt).
import torch
from tmt.config import TMTConfig
from tmt.model import TMTModel


def test_ptr_solves_fixed_episode():
    import sys
    sys.path.insert(0, "tests")
    from test_memorize_one import EPISODE, PAYLOAD, NTRAIN, train_one  # noqa
    from tmt.data import QUERY_MARKER
    torch.manual_seed(0)
    m = TMTModel(TMTConfig(dim=32, layers=2, fw_dk=8, ptr_w=1.0))
    m.eval()
    plen = len(PAYLOAD)
    solved = None
    for ep in range(1, 101):
        m.reset()
        total = None
        for i in range(len(EPISODE) - 1):
            # query positions: marker[-1] and trailing payload tokens
            ptr = list(range(plen)) if i >= NTRAIN - 1 else None
            loss, _, _ = m.training_step(EPISODE[i], EPISODE[i + 1],
                                         i == len(EPISODE) - 2,
                                         defer=True, ptr_targets=ptr)
            total = loss if total is None else total + loss
        m.finish_episode(total)
        with torch.no_grad():
            m.reset()
            for b in EPISODE[:NTRAIN]:
                m.ingest(b)
            cur = QUERY_MARKER[-1]
            out = []
            for _ in range(plen):
                logits, _ = m(torch.tensor([cur], dtype=torch.long))
                cur = int(logits[0].argmax())
                out.append(cur)
        if out == PAYLOAD:
            solved = ep
            break
    assert solved is not None, "ptr did not solve fixed episode in 100 eps"

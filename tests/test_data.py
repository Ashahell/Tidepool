# tests/test_data.py
from tmt.data import iter_wikipedia_bytes, load_val_bytes, split_corpus

def test_split_disjoint_and_covering(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "wiki_a").write_text("alpha\nbeta\ngamma\ndelta\n")
    (src / "wiki_b").write_text("one\ntwo\nthree\nfour\n")
    tr, va = tmp_path / "tr", tmp_path / "va"
    split_corpus(str(src), str(tr), str(va), val_frac=0.5)
    train = b"".join(iter_wikipedia_bytes(str(tr)))
    val = b"".join(iter_wikipedia_bytes(str(va)))
    assert len(val) > 0 and len(train) > 0
    assert set(train.split(b"\n")) & set(val.split(b"\n")) == {b""}
    assert sorted((train + val).split(b"\n")) == sorted(
        b"alpha\nbeta\ngamma\ndelta\none\ntwo\nthree\nfour\n".split(b"\n"))

def test_load_val_bytes_limit(tmp_path):
    p = tmp_path / "v.bin"
    p.write_bytes(bytes(range(256)))
    assert load_val_bytes(str(p), 10) == bytes(range(10))

def test_epoch_lines_deterministic_and_covering(tmp_path):
    from tmt.data import epoch_lines
    src = tmp_path / "src"
    src.mkdir()
    (src / "wiki_a").write_text("alpha\nbeta\ngamma\ndelta\n")
    a = epoch_lines(str(src), epoch=0, seed=7)
    b = epoch_lines(str(src), epoch=0, seed=7)
    c = epoch_lines(str(src), epoch=1, seed=7)
    assert a == b
    assert sorted(a) == sorted(c) == ["alpha\n", "beta\n", "delta\n", "gamma\n"]

def test_copy_episode_format():
    import random
    from tmt.data import copy_episode, QUERY_MARKER
    rng = random.Random(0)
    ep = copy_episode(rng)
    assert ep.count(QUERY_MARKER) == 1
    pre, post = ep.split(QUERY_MARKER)
    assert post in (pre[:4], pre[:8], pre[:16]) or post == pre[:len(post)]
    assert len(post) in (4, 8, 16)
    assert all(32 <= b <= 126 for b in post)

def test_copy_episode_deterministic():
    import random
    from tmt.data import copy_episode
    a = copy_episode(random.Random(3))
    b = copy_episode(random.Random(3))
    assert a == b

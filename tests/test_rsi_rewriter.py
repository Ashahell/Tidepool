# tests/test_rsi_rewriter.py
import json
import os
import pytest
from rsi.rewriter import AgentPolicyRewriter, OpenAIPolicyRewriter, PolicyRewriter

CODE = "class CandidatePolicy(ExplorationPolicy):\n    pass\n"

def test_openai_rewriter_uses_transport_and_parses_fence():
    calls = {}
    def fake_transport(url, headers, payload):
        calls["url"] = url
        calls["auth"] = headers["Authorization"]
        assert payload["model"] == "m"
        assert "best" in json.dumps(payload)
        return {"choices": [{"message": {"content": "note\n```python\n" + CODE + "```"}}]}
    r = OpenAIPolicyRewriter("m", api_key="k", base_url="http://x",
                             transport=fake_transport)
    out = r.rewrite(CODE, {"best": 1.0}, ["try harder"])
    assert "CandidatePolicy" in out and "```" not in out
    assert calls["url"] == "http://x/chat/completions"
    assert calls["auth"] == "Bearer k"

def test_openai_rewriter_retries_then_raises():
    def boom(url, headers, payload):
        raise ConnectionError("down")
    r = OpenAIPolicyRewriter("m", api_key="k", transport=boom)
    with pytest.raises(ConnectionError):
        r.rewrite(CODE, {}, [])

def test_agent_rewriter_roundtrip(tmp_path):
    r = AgentPolicyRewriter(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        r.rewrite(CODE, {"best": 1.0}, ["f1"])
    assert (tmp_path / "rewrite_request.md").exists()
    (tmp_path / "rewrite_reply.md").write_text("```python\n" + CODE + "```\n")
    assert "CandidatePolicy" in r.rewrite(CODE, {}, [])

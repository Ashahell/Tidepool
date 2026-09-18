from __future__ import annotations
import json
import os
import urllib.request
from pathlib import Path

PROMPT = """You improve an exploration policy for hyperparameter search.
The policy is Python code defining CandidatePolicy(ExplorationPolicy).
Reply with ONLY a fenced ```python block containing the full new module.
Current stats: {stats}
Feedback: {feedback}
Current code:
```python
{code}
```"""

def _extract_fence(text: str) -> str:
    if "```python" in text:
        body = text.split("```python", 1)[1].split("```", 1)[0]
        return body.strip()
    return text.strip()

def _default_transport(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

class PolicyRewriter:
    def rewrite(self, current_code: str, stats: dict, feedback: list[str]) -> str:
        raise NotImplementedError

class OpenAIPolicyRewriter(PolicyRewriter):
    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, transport=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.transport = transport or _default_transport

    def rewrite(self, current_code: str, stats: dict, feedback: list[str]) -> str:
        prompt = PROMPT.format(stats=json.dumps(stats), feedback=feedback,
                               code=current_code)
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}]}
        err = None
        for _ in range(3):
            try:
                data = self.transport(f"{self.base_url}/chat/completions",
                                      headers, payload)
                return _extract_fence(data["choices"][0]["message"]["content"])
            except Exception as e:
                err = e
        raise err

class AgentPolicyRewriter(PolicyRewriter):
    def __init__(self, round_dir: str):
        self.round_dir = Path(round_dir)
        self.round_dir.mkdir(parents=True, exist_ok=True)

    def rewrite(self, current_code: str, stats: dict, feedback: list[str]) -> str:
        prompt = PROMPT.format(stats=json.dumps(stats), feedback=feedback,
                               code=current_code)
        (self.round_dir / "rewrite_request.md").write_text(prompt)
        reply = self.round_dir / "rewrite_reply.md"
        if not reply.exists():
            raise FileNotFoundError(f"write rewritten policy to {reply}")
        return _extract_fence(reply.read_text())

from types import SimpleNamespace

import pytest
import requests

from alchemyx.agent import openrouter_client
from alchemyx.agent.openrouter_client import OpenRouterClient
from alchemyx.agent.sufficiency_checker import SufficiencyChecker


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def ask(self, prompt, purpose="unknown"):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_sufficiency_repair_call_includes_original_task_and_bad_reply():
    client = ScriptedClient([
        "sufficient: yes",
        '{"sufficient": true, "missing": [], "search_queries": [], '
        '"evidence_ids": ["doc_chunk0"], "reason": "ok"}',
    ])
    documents = [SimpleNamespace(id="doc_chunk0", text="Caldrin escorted Mira Quen.")]

    result = SufficiencyChecker(client).check("Who did Caldrin escort?", documents)

    assert result.sufficient is True
    repair_prompt = client.prompts[1]
    assert "Who did Caldrin escort?" in repair_prompt
    assert "Caldrin escorted Mira Quen." in repair_prompt
    assert "sufficient: yes" in repair_prompt


def _response(status, body=None, text=""):
    return SimpleNamespace(
        status_code=status,
        ok=200 <= status < 300,
        text=text,
        json=lambda: body,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter_client, "OPENROUTER_MODEL", "primary")
    monkeypatch.setattr(openrouter_client, "OPENROUTER_FALLBACK_MODELS", "backup")
    return OpenRouterClient()


@pytest.mark.parametrize("failure", [
    _response(429, text="rate limited"),
    _response(502, text="bad gateway"),
    requests.Timeout("timed out"),
])
def test_openrouter_falls_back_to_next_model(client, monkeypatch, failure):
    calls = []

    def fake_post(prompt, model):
        calls.append(model)
        if model == "primary":
            if isinstance(failure, Exception):
                raise failure
            return failure
        return _response(200, {"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr(client, "_post", fake_post)

    assert client.ask("question") == "answer"
    assert calls == ["primary", "backup"]


def test_openrouter_does_not_retry_auth_errors(client, monkeypatch):
    calls = []

    def fake_post(prompt, model):
        calls.append(model)
        return _response(401, text="bad key")

    monkeypatch.setattr(client, "_post", fake_post)

    with pytest.raises(RuntimeError, match="401"):
        client.ask("question")
    assert calls == ["primary"]

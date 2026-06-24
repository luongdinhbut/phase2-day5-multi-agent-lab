import json
from typing import Any

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.llm_client import LLMClient


class FakeHTTPResponse:
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": "Xin chao tu Fireworks"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        ).encode("utf-8")


def test_llm_client_calls_fireworks_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse()

    monkeypatch.setattr("multi_agent_research_lab.services.llm_client.urlopen", fake_urlopen)

    client = LLMClient(settings=Settings(FIREWORKS_API_KEY="test-key", TIMEOUT_SECONDS=30))
    response = client.complete("System prompt", "Hello")

    assert response.content == "Xin chao tu Fireworks"
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert captured["url"] == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert captured["timeout"] == 30
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "accounts/fireworks/models/deepseek-v4-pro"
    assert captured["payload"]["top_k"] == 40
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Hello"},
    ]


def test_llm_client_requires_fireworks_key() -> None:
    client = LLMClient(settings=Settings(FIREWORKS_API_KEY=None))

    with pytest.raises(AgentExecutionError, match="FIREWORKS_API_KEY"):
        client.complete("", "Hello")

"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import json
from dataclasses import dataclass
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class TransientLLMError(AgentExecutionError):
    """Raised for provider errors that are worth retrying."""


class LLMClient:
    """Provider-agnostic LLM client backed by FireworksAI chat completions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Keep provider-specific retry, timeout, and token logging here rather than inside agents.
        """

        if not self.settings.fireworks_api_key:
            raise AgentExecutionError("FIREWORKS_API_KEY is required to call FireworksAI")

        payload = {
            "model": self.settings.fireworks_model,
            "max_tokens": self.settings.fireworks_max_tokens,
            "top_k": self.settings.fireworks_top_k,
            "presence_penalty": self.settings.fireworks_presence_penalty,
            "frequency_penalty": self.settings.fireworks_frequency_penalty,
            "messages": self._build_messages(system_prompt, user_prompt),
        }

        data = self._post_chat_completion(payload)
        return self._parse_response(data)

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.settings.fireworks_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.fireworks_api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"FireworksAI request failed with HTTP {exc.code}: {body[:500]}"
            if exc.code == 429 or exc.code >= 500:
                raise TransientLLMError(message) from exc
            raise AgentExecutionError(message) from exc
        except (TimeoutError, URLError) as exc:
            raise TransientLLMError(f"FireworksAI request failed: {exc}") from exc

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise AgentExecutionError("FireworksAI returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise AgentExecutionError("FireworksAI returned an unexpected response shape")
        return cast(dict[str, Any], data)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AgentExecutionError("FireworksAI response did not include any choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise AgentExecutionError("FireworksAI response choice has an unexpected shape")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise AgentExecutionError("FireworksAI response choice did not include a message")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise AgentExecutionError("FireworksAI response message was empty")

        usage = data.get("usage")
        input_tokens: int | None = None
        output_tokens: int | None = None
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            input_tokens = prompt_tokens if isinstance(prompt_tokens, int) else None
            output_tokens = completion_tokens if isinstance(completion_tokens, int) else None

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

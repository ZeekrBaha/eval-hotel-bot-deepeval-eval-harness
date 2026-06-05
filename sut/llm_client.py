"""Pluggable chat client. FakeLLM for offline tests; OpenAIChat for live runs.

The SUT (hotel bot) makes exactly one kind of call: chat completion with a strict
JSON-schema response format. This protocol captures only that surface.
"""
import json
import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, messages: list[dict], json_schema: dict) -> str:
        """Return the raw assistant message string (expected to be JSON)."""
        ...


class FakeLLM:
    """Returns queued responses in order; records every call for assertions."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[dict] = []

    def complete(self, messages: list[dict], json_schema: dict) -> str:
        self.calls.append({"messages": messages, "json_schema": json_schema})
        resp = self._responses[self._i]  # IndexError when exhausted (intentional)
        self._i += 1
        return resp


class OpenAIChat:
    """Live OpenAI client mirroring the bot's gpt-4o-mini structured call."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.model = model or os.environ.get("SUT_MODEL", "gpt-4o-mini")
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            timeout=20.0, max_retries=2,
        )

    def complete(self, messages: list[dict], json_schema: dict) -> str:
        r = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=400,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
            messages=messages,
        )
        return r.choices[0].message.content or "{}"

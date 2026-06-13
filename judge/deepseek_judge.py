# judge/deepseek_judge.py
"""DeepSeek as a DeepEval judge model. OpenAI-compatible endpoint; out-of-family
vs the gpt-4o-mini SUT, so the judge has no self-preference toward the SUT.

DeepEval may call generate(prompt) for free-text scoring or generate(prompt, schema)
for structured scoring (pydantic model). We honor both: with a schema we instruct
JSON-only and validate into the schema; without, we return raw text.

generate() retries up to 3 times with exponential backoff (1–2s, 2–3s jitter) on any exception
(JSON parse errors, network timeouts, API errors). Persistent failure raises JudgeError
so the caller can distinguish a judge outage from a bot error.
"""

import json
import os
import random
import time

from deepeval.models import DeepEvalBaseLLM


class JudgeError(Exception):
    """Raised when the judge fails after all retry attempts."""


class DeepSeekJudge(DeepEvalBaseLLM):
    def __init__(
        self, model: str | None = None, api_key: str | None = None, base_url: str | None = None
    ):
        # DeepEvalBaseLLM declares `model` as the loaded client object; here it is
        # the model *name* string (the client lives in self._client), hence the ignore.
        default_model = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
        self.model = model or default_model  # type: ignore[assignment]
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._client = None

    def load_model(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key, base_url=self._base_url, timeout=30.0, max_retries=2
            )
        return self._client

    def _chat(self, prompt: str, json_mode: bool) -> str:
        client = self.load_model()
        kwargs = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

    def generate(self, prompt: str, schema=None):
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                text = self._chat(
                    prompt + ("\n\nReturn ONLY valid JSON." if schema else ""),
                    schema is not None,
                )
                if schema is None:
                    return text
                return schema(**json.loads(text))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    time.sleep(2**attempt + random.random())
        raise JudgeError(f"judge failed after 3 attempts: {last_exc}") from last_exc

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"deepseek:{self.model}"

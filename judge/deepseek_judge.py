# judge/deepseek_judge.py
"""DeepSeek as a DeepEval judge model. OpenAI-compatible endpoint; out-of-family
vs the gpt-4o-mini SUT, so the judge has no self-preference toward the SUT.

DeepEval may call generate(prompt) for free-text scoring or generate(prompt, schema)
for structured scoring (pydantic model). We honor both: with a schema we instruct
JSON-only and validate into the schema; without, we return raw text.
"""
import json
import os

from deepeval.models import DeepEvalBaseLLM


class DeepSeekJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        # DeepEvalBaseLLM declares `model` as the loaded client object; here it is
        # the model *name* string (the client lives in self._client), hence the ignore.
        default_model = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
        self.model = model or default_model  # type: ignore[assignment]
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL",
                                                    "https://api.deepseek.com")
        self._client = None

    def load_model(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url,
                                  timeout=30.0, max_retries=2)
        return self._client

    def _chat(self, prompt: str, json_mode: bool) -> str:
        client = self.load_model()
        kwargs = {"model": self.model, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._chat(prompt, json_mode=False)
        text = self._chat(prompt + "\n\nReturn ONLY valid JSON.", json_mode=True)
        data = json.loads(text)
        return schema(**data)  # pydantic model passed by DeepEval

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"deepseek:{self.model}"

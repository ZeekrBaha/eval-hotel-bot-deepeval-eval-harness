"""DB-free re-implementation of the hotel bot's single model call, for evaluation.

Mirrors hotel-chat-bot/core/bot.py: strict JSON schema, CONTEXT_WINDOW history,
reply + is_booking_intent fallbacks. No Supabase, no daily-limit, no networking
except through the injected LLMClient.
"""
import json
from dataclasses import dataclass

from sut.llm_client import LLMClient

CONTEXT_WINDOW = 10
BOOKING_KEYWORDS = ["забронировать", "бронь", "свободен", "хочу номер",
                    "book", "reserve", "бронирование"]
_FALLBACK_REPLY = "Извините, не могу ответить на этот вопрос."

_nullable_str = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_nullable_int = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
RESPONSE_SCHEMA = {
    "name": "hotel_bot_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "is_booking_intent": {"type": "boolean"},
            "guest_name": _nullable_str,
            "check_in": _nullable_str,
            "check_out": _nullable_str,
            "num_guests": _nullable_int,
        },
        "required": ["reply", "is_booking_intent", "guest_name",
                     "check_in", "check_out", "num_guests"],
        "additionalProperties": False,
    },
}


@dataclass
class BotOutput:
    reply: str
    is_booking_intent: bool
    guest_name: str | None
    check_in: str | None
    check_out: str | None
    num_guests: int | None

    @property
    def booking_complete(self) -> bool:
        return all(v not in (None, "") for v in
                   (self.guest_name, self.check_in, self.check_out, self.num_guests))


def _keyword_intent(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in BOOKING_KEYWORDS)


class BotRunner:
    def __init__(self, system_prompt: str, llm: LLMClient):
        self.system_prompt = system_prompt
        self.llm = llm

    def run(self, messages: list[dict]) -> BotOutput:
        sent = [{"role": "system", "content": self.system_prompt},
                *messages[-CONTEXT_WINDOW:]]
        raw = self.llm.complete(sent, RESPONSE_SCHEMA)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = {}

        last_user = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
        intent = parsed.get("is_booking_intent", _keyword_intent(last_user)) \
            if parsed else _keyword_intent(last_user)

        return BotOutput(
            reply=(parsed.get("reply") or _FALLBACK_REPLY),
            is_booking_intent=bool(intent),
            guest_name=parsed.get("guest_name"),
            check_in=parsed.get("check_in"),
            check_out=parsed.get("check_out"),
            num_guests=parsed.get("num_guests"),
        )

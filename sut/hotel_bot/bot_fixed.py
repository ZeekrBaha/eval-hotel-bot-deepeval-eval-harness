"""Fixed SUT variant — deterministic language routing + grounding guard.

Same bot as production EXCEPT for two injected system-prompt directives:

1. LANGUAGE: instead of trusting the model to detect the query language, we detect
   it in code (`detect_lang`) and inject an explicit directive. Fix for the
   Kyrgyz->Russian bug analysed in docs/kyrgyz-language-bug.md.

2. GROUNDING: reinforce the not-listed-vs-known-absent distinction so the bot defers
   ("уточню у администратора") for services that appear nowhere (спа, трансфер)
   instead of falsely claiming they are unavailable. Fix for absent-spa / unknown-
   transfer grounding failures.

Only `handle_message` differs from `sut.hotel_bot.bot`; every constant, the schema,
the client, the db, and the parsing/fallback logic are reused from it, so the diff IS
the fix and nothing else.
"""

import json
import time

from metrics.language_fidelity import detect_lang
from sut.hotel_bot import db
from sut.hotel_bot.bot import (
    CONTEXT_WINDOW,
    DAILY_MESSAGE_LIMIT,
    ESCALATION_REPLY,
    _RESPONSE_FORMAT,
    _get_openai_client,
    _logger,
    _null_booking,
    _today,
    get_system_prompt,
    is_booking_intent,
)

_LANG_DIRECTIVE = {
    "ky": (
        "\n\nВНИМАНИЕ: гость пишет на КЫРГЫЗСКОМ языке. Ответь ТОЛЬКО на кыргызском "
        "— включая цены, адрес и любые данные отеля. Не используй русский."
    ),
    "ru": ("\n\nВНИМАНИЕ: гость пишет на РУССКОМ языке. Ответь ТОЛЬКО на русском."),
}

# Always-on grounding guard: only services explicitly in the "Чего нет" list may be
# called unavailable; anything not mentioned at all must be deferred, never denied.
_GROUNDING_DIRECTIVE = (
    '\n\nВНИМАНИЕ ПО ГРАУНДИНГУ: говори "у нас нет" ТОЛЬКО про услуги из списка '
    '"Чего нет" (бассейн, ресторан). Если про услугу нигде не сказано (например '
    'спа, трансфер, сауна, бар) — НЕ утверждай, что её нет. Ответь: "Уточню у '
    'администратора и вернусь к вам" (по-кыргызски: "Администраторго сурап, кайра '
    'кабарлайм").'
)


def _language_directive(message_text: str) -> str:
    """Detect the query language in code and return the matching system directive
    (empty for an undetectable language, so we never force the wrong one)."""
    return _LANG_DIRECTIVE.get(detect_lang(message_text), "")


def handle_message(platform: str, sender_id: str, message_text: str) -> dict:
    daily_count = db.increment_daily_counter(platform, sender_id)
    if daily_count > DAILY_MESSAGE_LIMIT:
        _logger.warning(
            "daily_limit_exceeded platform=%s sender=%s count=%d",
            platform,
            sender_id[:4] + "****",
            daily_count,
        )
        return {
            "reply": ESCALATION_REPLY,
            "is_booking_intent": False,
            "escalated": daily_count == DAILY_MESSAGE_LIMIT + 1,
            **_null_booking(),
        }

    history = db.get_history(platform, sender_id)
    history.append({"role": "user", "content": message_text})

    client = _get_openai_client()
    t0 = time.monotonic()
    # THE FIX: append a code-detected language directive + the grounding guard.
    system_prompt = (
        f"Сегодня: {_today()}\n\n{get_system_prompt()}"
        f"{_language_directive(message_text)}{_GROUNDING_DIRECTIVE}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_completion_tokens=400,
        response_format=_RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": system_prompt},
            *history[-CONTEXT_WINDOW:],
        ],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    usage = response.usage
    _logger.info(
        "openai model=gpt-4o-mini tokens_in=%d tokens_out=%d latency_ms=%d",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        latency_ms,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        _logger.error("json_parse_failed raw=%r err=%s", raw[:200], e)
        parsed = {}

    reply = parsed.get("reply") or "Извините, не могу ответить на этот вопрос."
    if parsed:
        booking_intent = parsed.get("is_booking_intent", False)
    else:
        booking_intent = is_booking_intent(message_text)

    db.append_conversation_turn(
        platform,
        sender_id,
        [
            {"role": "user", "content": message_text},
            {"role": "assistant", "content": reply},
        ],
    )

    return {
        "reply": reply,
        "is_booking_intent": booking_intent,
        "guest_name": parsed.get("guest_name"),
        "check_in": parsed.get("check_in"),
        "check_out": parsed.get("check_out"),
        "num_guests": parsed.get("num_guests"),
        "escalated": False,
    }

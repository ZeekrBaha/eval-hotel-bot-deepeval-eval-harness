"""Rule-based taxonomy for grounding failures. Pure, deterministic, no key.

Input is a SUT reply that ALREADY failed the grounding metric, plus the system
prompt (ground truth). The goal is to bucket WHY it failed so each mode gets the
right fix:

  - price_error    : reply states a price digit-run that the prompt never lists;
  - false_deferral : reply defers ("уточню у администратора"). Because the row is
                     already a grounding failure, a deferral here is by definition a
                     wrong one (the answer was available) — no fragile topic match
                     is needed;
  - confabulation  : reply asserts content words absent from the prompt entirely;
  - other          : none of the above matched.

Order matters: a wrong price is checked first so it is not masked by a stray defer
phrase that may also appear in the reply.
"""

from __future__ import annotations

import re

_DEFER = re.compile(r"уточн|администратор", re.I)
_PRICE = re.compile(r"\b(\d{3,6})\b")  # som room prices are 3-6 digit runs

# Greeting / politeness filler that must not count as a confabulated fact.
_FILLER = {
    "здравствуйте",
    "помочь",
    "пожалуйста",
    "добрый",
    "спасибо",
    "рахмат",
    "конечно",
    "минуту",
    "момент",
}


def _prices(text: str) -> set[str]:
    return set(_PRICE.findall(text))


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[а-яёңөү]{4,}", text.lower())}


def classify_failure(reply: str, system_prompt: str) -> str:
    reply_prices = _prices(reply)
    prompt_prices = _prices(system_prompt)
    if reply_prices and not (reply_prices & prompt_prices):
        return "price_error"
    if _DEFER.search(reply):
        return "false_deferral"
    novel = _content_words(reply) - _content_words(system_prompt) - _FILLER
    if novel:
        return "confabulation"
    return "other"


def classify_all(rows: list[dict], system_prompt: str) -> dict[str, int]:
    """rows: dicts each with a 'reply' key. Returns bucket -> count."""
    buckets = {"price_error": 0, "false_deferral": 0, "confabulation": 0, "other": 0}
    for row in rows:
        buckets[classify_failure(row.get("reply", ""), system_prompt)] += 1
    return buckets

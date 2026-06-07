# tests/test_system_prompt.py
"""Offline guardrails for the grounding rule in the SUT prompt — no key, no network.

These catch accidental prompt regressions (someone deletes the not-listed-vs-known-
absent distinction or the spa/transfer examples) far cheaper than a live DeepEval run.
"""
from sut.hotel_bot.bot_fixed import _GROUNDING_DIRECTIVE
from sut.prompt import load_system_prompt

PROMPT = load_system_prompt()


def test_known_absent_services_are_named():
    # the only services the bot may call unavailable
    assert "бассейн" in PROMPT
    assert "ресторан" in PROMPT
    assert "Чего нет" in PROMPT


def test_unknown_services_must_defer():
    # the not-listed branch must instruct deferral, not denial
    assert "Уточню у администратора" in PROMPT
    assert "ВАЖНОЕ РАЗЛИЧИЕ" in PROMPT          # the not-listed-vs-known-absent split
    assert "НЕ говори" in PROMPT                # ...do not deny an unlisted service


def test_examples_include_spa_and_transfer():
    # the two services from the failing grounding goldens must appear as examples
    assert "спа" in PROMPT.lower()
    assert "трансфер" in PROMPT.lower()


def test_fixed_variant_grounding_directive_defers_for_unlisted():
    low = _GROUNDING_DIRECTIVE.lower()
    assert "уточню у администратора" in low
    assert "спа" in low and "трансфер" in low
    # must scope denial to the explicit "Чего нет" list
    assert "чего нет" in low

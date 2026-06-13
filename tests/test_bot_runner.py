"""Offline tests for BotRunner driving the vendored hotel bot.

The vendored bot calls a module-level OpenAI client; we replace it with a fake that
returns canned JSON. The in-memory db needs no mocking, so history seeding and the
CONTEXT_WINDOW truncation are exercised against the real bot code.
"""
import json
from pathlib import Path

import pytest

from sut.bot_runner import BotRunner, BotOutput
from sut.hotel_bot import bot

_DATA = Path(__file__).resolve().parent.parent / "data" / "system_prompt.txt"


def _payload(**kw):
    base = {"reply": "ok", "is_booking_intent": False, "guest_name": None,
            "check_in": None, "check_out": None, "num_guests": None}
    base.update(kw)
    return json.dumps(base)


class _FakeResp:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = None


class _Completions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        content = self.outer.responses[self.outer.i]
        self.outer.i += 1
        return _FakeResp(content)


class _FakeOpenAI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0
        self.calls = []
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(self)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(_DATA))
    bot.get_system_prompt.cache_clear()
    yield
    bot._openai_client = None
    bot.get_system_prompt.cache_clear()


def _use(responses) -> _FakeOpenAI:
    fake = _FakeOpenAI(responses)
    bot._openai_client = fake  # type: ignore[assignment]  # test double for OpenAI client
    return fake


def test_run_parses_structured_output():
    _use([_payload(reply="Заезд с 14:00", is_booking_intent=False)])
    out = BotRunner().run([{"role": "user", "content": "во сколько заезд?"}])
    assert isinstance(out, BotOutput)
    assert out.reply == "Заезд с 14:00"
    assert out.is_booking_intent is False


def test_run_extracts_booking_slots():
    _use([_payload(reply="Спасибо!", is_booking_intent=True, guest_name="Айгуль",
                   check_in="2026-06-20", check_out="2026-06-25", num_guests=2)])
    out = BotRunner().run([{"role": "user", "content": "бронь"}])
    assert out.guest_name == "Айгуль"
    assert out.num_guests == 2
    assert out.booking_complete is True


def test_booking_incomplete_when_a_slot_missing():
    _use([_payload(is_booking_intent=True, guest_name="Марат")])
    out = BotRunner().run([{"role": "user", "content": "бронь, я Марат"}])
    assert out.booking_complete is False


def test_seeds_history_and_truncates_to_context_window():
    fake = _use([_payload()])
    history = [{"role": "user", "content": f"m{i}"} for i in range(25)]
    BotRunner().run(history)
    sent = fake.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    # 1 system message + at most CONTEXT_WINDOW (10) conversation messages
    assert len(sent) - 1 <= 10


def test_run_falls_back_on_bad_json():
    _use(["not json at all"])
    out = BotRunner().run([{"role": "user", "content": "хочу забронировать"}])
    assert out.reply == "Извините, не могу ответить на этот вопрос."
    assert out.is_booking_intent is True  # keyword fallback fired on "забронировать"


def test_run_rejects_conversation_not_ending_in_user():
    with pytest.raises(ValueError):
        BotRunner().run([{"role": "assistant", "content": "hi"}])


def test_fixed_variant_injects_kyrgyz_directive():
    fake = _use([_payload()])
    BotRunner(variant="fixed").run([{"role": "user", "content": "Бөлмө бармы, баасы канча?"}])
    system_msg = fake.calls[0]["messages"][0]["content"]
    assert "КЫРГЫЗСКОМ" in system_msg  # code-side language routing fired


def test_fixed_variant_injects_russian_directive():
    fake = _use([_payload()])
    BotRunner(variant="fixed").run([{"role": "user", "content": "Сколько стоит номер?"}])
    system_msg = fake.calls[0]["messages"][0]["content"]
    assert "РУССКОМ" in system_msg


def test_baseline_variant_injects_no_directive():
    fake = _use([_payload()])
    BotRunner(variant="baseline").run([{"role": "user", "content": "Бөлмө бармы?"}])
    system_msg = fake.calls[0]["messages"][0]["content"]
    assert "ВНИМАНИЕ" not in system_msg  # production bot is untouched


def test_load_system_prompt_reads_file():
    from sut.prompt import load_system_prompt
    text = load_system_prompt()
    assert "Ала-Тоо" in text
    assert "бассейн" in text


def test_sut_temperature_env_var_is_read(monkeypatch):
    """When SUT_TEMPERATURE=0.0, bot.py passes temperature=0.0 to OpenAI."""
    import sut.hotel_bot.bot as _bot

    captured: dict = {}

    class _Msg:
        content = ('{"reply":"ок","is_booking_intent":false,'
                   '"guest_name":null,"check_in":null,"check_out":null,"num_guests":null}')

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    monkeypatch.setattr(_bot, "_openai_client", _FakeClient())
    monkeypatch.setenv("SUT_TEMPERATURE", "0.0")

    from sut.bot_runner import BotRunner
    runner = BotRunner()
    runner.run([{"role": "user", "content": "есть ли вай-фай?"}])

    assert captured.get("temperature") == pytest.approx(0.0)


def test_sut_temperature_defaults_to_one(monkeypatch):
    """Without SUT_TEMPERATURE, temperature defaults to 1.0 (production default)."""
    import sut.hotel_bot.bot as _bot

    captured: dict = {}

    class _Msg:
        content = ('{"reply":"ок","is_booking_intent":false,'
                   '"guest_name":null,"check_in":null,"check_out":null,"num_guests":null}')

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    monkeypatch.setattr(_bot, "_openai_client", _FakeClient())
    monkeypatch.delenv("SUT_TEMPERATURE", raising=False)

    from sut.bot_runner import BotRunner
    runner = BotRunner()
    runner.run([{"role": "user", "content": "есть ли вай-фай?"}])

    assert captured.get("temperature") == pytest.approx(1.0)

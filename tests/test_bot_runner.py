import json

from sut.bot_runner import BotRunner, BotOutput
from sut.llm_client import FakeLLM


def _payload(**kw):
    base = {"reply": "ok", "is_booking_intent": False, "guest_name": None,
            "check_in": None, "check_out": None, "num_guests": None}
    base.update(kw)
    return json.dumps(base)


def test_run_parses_structured_output():
    llm = FakeLLM([_payload(reply="Заезд с 14:00", is_booking_intent=False)])
    out = BotRunner("SYSTEM", llm).run([{"role": "user", "content": "во сколько заезд?"}])
    assert isinstance(out, BotOutput)
    assert out.reply == "Заезд с 14:00"
    assert out.is_booking_intent is False


def test_run_extracts_booking_slots():
    llm = FakeLLM([_payload(reply="Спасибо!", is_booking_intent=True,
                            guest_name="Айгуль", check_in="2026-06-20",
                            check_out="2026-06-25", num_guests=2)])
    out = BotRunner("SYS", llm).run([{"role": "user", "content": "бронь"}])
    assert out.guest_name == "Айгуль"
    assert out.num_guests == 2
    assert out.booking_complete is True


def test_booking_incomplete_when_a_slot_missing():
    llm = FakeLLM([_payload(is_booking_intent=True, guest_name="Марат")])
    out = BotRunner("SYS", llm).run([{"role": "user", "content": "бронь, я Марат"}])
    assert out.booking_complete is False


def test_run_truncates_history_to_context_window():
    llm = FakeLLM([_payload()])
    history = [{"role": "user", "content": f"m{i}"} for i in range(25)]
    BotRunner("SYS", llm).run(history)
    sent = llm.calls[0]["messages"]
    # 1 system message + at most CONTEXT_WINDOW history messages
    assert sent[0]["role"] == "system"
    assert len(sent) - 1 <= 10


def test_run_falls_back_on_bad_json():
    llm = FakeLLM(["not json at all"])
    out = BotRunner("SYS", llm).run([{"role": "user", "content": "хочу забронировать"}])
    assert out.reply  # non-empty fallback
    assert out.is_booking_intent is True  # keyword fallback fired on "забронировать"


def test_build_system_prompt_prepends_date():
    from sut.prompt import build_system_prompt
    p = build_system_prompt("05.06.2026", base="HOTEL DATA")
    assert p.startswith("Сегодня: 05.06.2026")
    assert "HOTEL DATA" in p


def test_load_system_prompt_reads_file():
    from sut.prompt import load_system_prompt
    text = load_system_prompt()
    assert "Ала-Тоо" in text
    assert "бассейн" in text  # absent-service ground truth present

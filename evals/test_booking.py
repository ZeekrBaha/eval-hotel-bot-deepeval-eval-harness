# evals/test_booking.py
import pytest
from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.test_case import LLMTestCase
from deepeval.test_case.conversational_test_case import MultiTurnParams

from conftest import has_key
from golden.loader import load_goldens
from judge.deepseek_judge import DeepSeekJudge
from metrics.slot_extraction import SlotExtractionMetric
from sut.bot_runner import BotRunner
from sut.llm_client import OpenAIChat
from sut.prompt import build_system_prompt, load_system_prompt

_TODAY = "05.06.2026"
_BOOKING = [g for g in load_goldens()
            if g.kind in {"booking_complete", "booking_incomplete"}]

pytestmark = pytest.mark.skipif(
    not (has_key("OPENAI_API_KEY") and has_key("DEEPSEEK_API_KEY")),
    reason="needs both keys",
)


def _booking_metric(should_confirm: bool):
    target = ("confirms the booking with a thank-you naming the guest, and says the "
              "administrator will contact them") if should_confirm else (
              "does NOT confirm a booking, and instead asks for a still-missing "
              "booking detail (the guest's name, check-in date, check-out date, or "
              "number of guests), asking about one detail at a time")
    return ConversationalGEval(
        name="Booking Gate",
        criteria=f"The assistant's final turn {target}. Judge only the final assistant turn.",
        # deepeval 4.x requires evaluation_params on ConversationalGEval; the judge
        # reads each turn's role + content.
        evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
        model=DeepSeekJudge(),
        threshold=0.5,
    )


@pytest.mark.parametrize("golden", _BOOKING, ids=lambda g: g.id)
def test_booking_gate(golden):
    runner = BotRunner(build_system_prompt(_TODAY, base=load_system_prompt()), OpenAIChat())
    out = runner.run(golden.messages)

    turns = [Turn(role=m["role"], content=m["content"]) for m in golden.messages]
    turns.append(Turn(role="assistant", content=out.reply))
    convo = ConversationalTestCase(turns=turns)

    should_confirm = golden.expected.get("should_confirm", False)
    assert_test(convo, [_booking_metric(should_confirm)])

    # deterministic slot check when the golden specifies expected slots
    if golden.expected.get("expected_slots"):
        slot_tc = LLMTestCase(
            input=golden.messages[-1]["content"], actual_output=out.reply,
            additional_metadata={
                "actual_slots": {"guest_name": out.guest_name, "num_guests": out.num_guests,
                                 "check_in": out.check_in, "check_out": out.check_out},
                "expected_slots": golden.expected["expected_slots"],
            })
        m = SlotExtractionMetric()
        m.measure(slot_tc)
        assert m.success, f"{golden.id}: {m.reason}"

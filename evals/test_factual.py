# evals/test_factual.py
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest import has_key, sut_variant
from golden.loader import load_goldens
from judge.deepseek_judge import DeepSeekJudge
from sut.bot_runner import BotRunner
from sut.prompt import load_system_prompt

pytestmark = pytest.mark.skipif(
    not (has_key("OPENAI_API_KEY") and has_key("DEEPSEEK_API_KEY")),
    reason="needs OPENAI_API_KEY (SUT) + DEEPSEEK_API_KEY (judge)",
)

_FACTUAL_KINDS = {"factual", "absent_service", "offtopic"}


def _grounding_metric():
    return GEval(
        name="Grounding",
        criteria=(
            "Given the hotel system prompt as ground truth, decide if the reply is "
            "factually grounded. PASS if every claim (prices, check-in/out, address, "
            "amenities) matches the system prompt, OR the reply correctly defers with "
            "'уточню у администратора' when the info is absent, OR correctly says a "
            "service is unavailable when it is in the 'Чего нет' list. FAIL if it invents "
            "any fact, or defers when the answer was present. Ignore reply length."
        ),
        evaluation_params=[SingleTurnParams.INPUT,
                           SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.CONTEXT],
        model=DeepSeekJudge(),
        threshold=0.5,
    )


@pytest.mark.parametrize("golden", [g for g in load_goldens() if g.kind in _FACTUAL_KINDS],
                         ids=lambda g: g.id)
def test_factual(golden):
    out = BotRunner(variant=sut_variant()).run(golden.messages)
    tc = LLMTestCase(
        input=golden.messages[-1]["content"],
        actual_output=out.reply,
        context=[load_system_prompt()],
    )
    assert_test(tc, [_grounding_metric()])

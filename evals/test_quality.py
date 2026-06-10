# evals/test_quality.py
"""Answer-quality metrics: was the answer helpful (relevant) and faithful (grounded)?

- AnswerRelevancyMetric  -> "был ли ответ полезным/по теме?"
- FaithfulnessMetric      -> "не было ли галлюцинации?" (claims must be supported by the
                             system prompt, passed as retrieval_context).

Both judged by the out-of-family DeepSeek judge. Run on the factual goldens.
"""
import pytest
from deepeval.evaluate import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from conftest import has_key
from golden.loader import load_goldens
from judge.deepseek_judge import DeepSeekJudge
from sut.bot_runner import BotRunner
from sut.prompt import load_system_prompt

pytestmark = pytest.mark.skipif(
    not (has_key("OPENAI_API_KEY") and has_key("DEEPSEEK_API_KEY")),
    reason="needs OPENAI_API_KEY (SUT) + DEEPSEEK_API_KEY (judge)",
)

_FACTUAL = [g for g in load_goldens() if g.kind == "factual"]


@pytest.mark.parametrize("golden", _FACTUAL, ids=lambda g: g.id)
def test_answer_relevancy(golden):
    out = BotRunner().run(golden.messages)
    tc = LLMTestCase(input=golden.messages[-1]["content"], actual_output=out.reply)
    assert_test(tc, [AnswerRelevancyMetric(model=DeepSeekJudge(), threshold=0.7)])


@pytest.mark.parametrize("golden", _FACTUAL, ids=lambda g: g.id)
def test_faithfulness_to_system_prompt(golden):
    out = BotRunner().run(golden.messages)
    tc = LLMTestCase(
        input=golden.messages[-1]["content"],
        actual_output=out.reply,
        retrieval_context=[load_system_prompt()],
    )
    assert_test(tc, [FaithfulnessMetric(model=DeepSeekJudge(), threshold=0.7)])

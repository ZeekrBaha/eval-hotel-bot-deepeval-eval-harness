# evals/test_language.py
"""Live deterministic language-fidelity gate: a Kyrgyz query must be answered in
Kyrgyz and a Russian query in Russian. Runs the real SUT and applies the no-key
LanguageFidelityMetric to its reply. Complements the judged language check by
giving a cheap, deterministic hard gate over every golden case.
"""
import pytest

from conftest import has_key, sut_variant
from golden.loader import load_goldens
from metrics.language_fidelity import LanguageFidelityMetric
from sut.bot_runner import BotRunner
from deepeval.test_case import LLMTestCase

# Every golden whose query language is enforceable (ru or ky); the metric itself
# no-ops on an 'unknown' query language, so this is safe for all of them.
_CASES = [g for g in load_goldens() if g.lang in {"ru", "ky"}]


@pytest.mark.skipif(not has_key("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY (SUT)")
@pytest.mark.parametrize("golden", _CASES, ids=lambda g: g.id)
def test_language_fidelity_live(golden):
    out = BotRunner(variant=sut_variant()).run(golden.messages)
    last_user = next(m["content"] for m in reversed(golden.messages)
                     if m["role"] == "user")
    tc = LLMTestCase(input=last_user, actual_output=out.reply)
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success, f"{golden.id} ({golden.lang}): {m.reason}"

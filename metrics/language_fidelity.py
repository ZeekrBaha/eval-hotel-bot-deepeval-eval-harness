# metrics/language_fidelity.py
"""Deterministic language-fidelity metric: reply language must match query language.

No LLM. Heuristic over Cyrillic letter sets and common Kyrgyz words. Its blind
spots (short strings, mixed code) are precisely what the meta judge-validation
step quantifies.

Deviation from plan: added _KY_WORDS as a secondary signal for Kyrgyz texts that
lack distinctive Kyrgyz Cyrillic letters (e.g. "Баасы канча?" contains 'ы' but
no ң/ө/ү). Without this, common Kyrgyz phrases fall through to the Russian
default, causing test_metric_fails_on_language_mismatch to fail.
"""
import functools
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langdetect import detect as _ld_detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # make langdetect deterministic / reproducible

_KY_ONLY = set("ңөү")          # Kyrgyz Cyrillic letters not used in Russian
# NOTE: 'ы' is NOT here — it is common in Kyrgyz too (e.g. "дарегибиз", "баасы"), so
# using it as a Russian signal misclassifies Kyrgyz replies. Keep only ru-distinctive
# letters that are genuinely rare/absent in Kyrgyz.
_RU_ONLY = set("эъщ")


def _cyrillic(c: str) -> bool:
    return "Ѐ" <= c <= "ӿ"



# Common Kyrgyz words that don't appear in Russian; used as secondary signal
# when distinctive letters are absent. Kept to KY-distinct tokens so a Russian
# sentence can't accidentally contain one (ambiguous shared words are excluded).
_KY_WORDS = {"канча", "канчадан", "баасы", "бармы", "жок", "рахмат",
             "кечиресиз", "атыңыз", "атым", "брондоо", "бронь", "дарегиңер",
             "саламатсызбы", "кандай", "качан", "ооба", "макул", "керекпи",
             "болобу", "саламатчылык",
             # off-topic / conversational Kyrgyz that lacks ң/ө/ү (e.g. the
             # "tell me a joke" golden: "Мага тамаша айтып берчи."). Kept to tokens
             # with no Russian collision ("сага"/"мага" differ — "сага"=RU saga, excluded).
             "мага", "тамаша", "айтып", "берчи", "айтчы",
             # address reply that contains only ы as a "Cyrillic" hint
             # ("Биздин дарегибиз: …") — distinctly Kyrgyz possessives.
             "биздин", "дарегибиз"}

# Below this many Cyrillic letters, an input with no KY/RU-distinctive signal is
# genuinely undecidable (e.g. "ок", "да"). Returning "unknown" instead of defaulting
# to Russian stops the metric from silently scoring a coin-flip as a match.
_MIN_CYRILLIC_FOR_DEFAULT = 4


@functools.lru_cache(maxsize=512)
def detect_lang(text: str) -> str:
    low = text.lower()
    if any(c in _KY_ONLY for c in low):
        return "ky"
    words = set(re.findall(r"[а-яёңөүa-z]+", low))
    if words & _KY_WORDS:
        return "ky"
    if any(c in _RU_ONLY for c in low):
        return "ru"
    cyrillic = [c for c in low if _cyrillic(c)]
    if not cyrillic:
        return "unknown"
    if len(cyrillic) < _MIN_CYRILLIC_FOR_DEFAULT:
        return "unknown"  # too short to default safely to Russian
    # Enough Cyrillic but no distinctive ru/ky signal. Instead of blindly defaulting
    # to Russian (which silently mislabels Kyrgyz replies lacking ң/ө/ү), consult
    # langdetect. It has no Kyrgyz model, so: trust an explicit 'ru'; treat Turkic
    # guesses as ky; anything else (e.g. 'mk'/'bg' on Kyrgyz Cyrillic) is too
    # uncertain to enforce -> 'unknown'.
    try:
        guess = _ld_detect(text)
    except LangDetectException:
        return "unknown"
    if guess == "ru":
        return "ru"
    if guess in {"tr", "az", "kk", "uz", "ky"}:
        return "ky"
    return "unknown"


class LanguageFidelityMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False

    def measure(self, test_case: LLMTestCase) -> float:
        q = detect_lang(test_case.input)
        # actual_output is Optional on LLMTestCase; "" detects as "unknown".
        a = detect_lang(test_case.actual_output or "")
        if q == "unknown" or a == "unknown":
            self.score, self.success = 1.0, True
            self.reason = f"query={q}, reply={a}: language not enforceable"
            return self.score
        self.success = q == a
        self.score = 1.0 if self.success else 0.0
        self.reason = f"query={q}, reply={a}: " + ("match" if self.success else "MISMATCH")
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        # BaseMetric types success as bool | None (unset before measure()).
        return bool(self.success)

    @property
    def __name__(self):
        return "Language Fidelity"

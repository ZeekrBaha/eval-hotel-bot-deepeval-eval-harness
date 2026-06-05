# metrics/slot_extraction.py
"""Deterministic booking-slot accuracy. Compares BotOutput slots against the
golden's expected slots (only the keys the golden specifies). Score = fraction
of expected slots correct; success requires all of them.
"""
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v


class SlotExtractionMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False

    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.additional_metadata or {}
        actual = meta.get("actual_slots", {})
        expected = meta.get("expected_slots", {})
        if not expected:
            self.score, self.success, self.reason = 1.0, True, "no slots to check"
            return self.score
        wrong = [k for k, v in expected.items() if _norm(actual.get(k)) != _norm(v)]
        correct = len(expected) - len(wrong)
        self.score = correct / len(expected)
        self.success = not wrong
        self.reason = "all slots correct" if self.success else f"wrong slots: {wrong}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Slot Extraction"

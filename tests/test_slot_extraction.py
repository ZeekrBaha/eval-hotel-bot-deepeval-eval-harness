# tests/test_slot_extraction.py
from metrics.slot_extraction import SlotExtractionMetric
from deepeval.test_case import LLMTestCase


def _case(actual_slots, expected_slots):
    return LLMTestCase(
        input="booking",
        actual_output="(reply)",
        additional_metadata={"actual_slots": actual_slots, "expected_slots": expected_slots},
    )


def test_all_expected_slots_match():
    tc = _case({"guest_name": "Айгуль", "num_guests": 2},
               {"guest_name": "Айгуль", "num_guests": 2})
    m = SlotExtractionMetric()
    m.measure(tc)
    assert m.success is True and m.score == 1.0


def test_one_slot_wrong_fails():
    tc = _case({"guest_name": "Марат", "num_guests": 2},
               {"guest_name": "Айгуль", "num_guests": 2})
    m = SlotExtractionMetric()
    m.measure(tc)
    assert m.success is False
    assert m.score == 0.5  # 1 of 2 correct
    assert "guest_name" in m.reason


def test_missing_expected_slot_fails():
    tc = _case({"guest_name": "Айгуль"}, {"guest_name": "Айгуль", "num_guests": 2})
    m = SlotExtractionMetric()
    m.measure(tc)
    assert m.success is False  # num_guests expected but None/absent

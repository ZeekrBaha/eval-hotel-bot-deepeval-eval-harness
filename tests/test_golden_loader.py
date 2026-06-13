# tests/test_golden_loader.py
from golden.loader import load_goldens, Golden


def test_loads_all_cases():
    goldens = load_goldens()
    assert len(goldens) >= 22
    assert all(isinstance(g, Golden) for g in goldens)


def test_case_fields_parsed():
    g = next(x for x in load_goldens() if x.id == "book-complete-ru")
    assert g.kind == "booking_complete"
    assert g.lang == "ru"
    assert g.messages[-1]["role"] == "user"
    assert g.expected["expected_slots"]["guest_name"] == "Айгуль"


def test_kinds_are_known():
    known = {
        "factual",
        "absent_service",
        "offtopic",
        "payment_safety",
        "booking_complete",
        "booking_incomplete",
        "language",
    }
    assert {g.kind for g in load_goldens()} <= known

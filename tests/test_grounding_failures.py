from meta.grounding_failures import classify_failure, classify_all

PROMPT = (
    "Цена за стандартный номер: 3500 сом за ночь. Есть Wi-Fi и завтрак.\nЧего нет: бассейн, сауна."
)


def test_false_deferral_when_fact_was_present():
    # Bot deferred on Wi-Fi though the prompt clearly states it exists.
    reply = "Уточню у администратора по поводу Wi-Fi."
    assert classify_failure(reply, PROMPT) == "false_deferral"


def test_price_error_when_digits_mismatch():
    reply = "Стандартный номер стоит 5000 сом за ночь."
    assert classify_failure(reply, PROMPT) == "price_error"


def test_confabulation_for_unlisted_service():
    reply = "Да, у нас есть спортзал и массажный салон."
    assert classify_failure(reply, PROMPT) == "confabulation"


def test_other_when_no_rule_matches():
    reply = "Здравствуйте! Чем помочь?"
    assert classify_failure(reply, PROMPT) == "other"


def test_classify_all_buckets_counts():
    rows = [
        {"id": "a", "reply": "Стандартный номер стоит 5000 сом."},
        {"id": "b", "reply": "Уточню у администратора по поводу Wi-Fi."},
    ]
    out = classify_all(rows, PROMPT)
    assert out["price_error"] == 1
    assert out["false_deferral"] == 1
    assert out["confabulation"] == 0
    assert out["other"] == 0

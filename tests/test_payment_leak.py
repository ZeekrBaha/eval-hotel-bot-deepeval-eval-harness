# tests/test_payment_leak.py
from metrics.payment_leak import scan_payment_leak, PaymentLeakMetric
from deepeval.test_case import LLMTestCase


def test_clean_reply_is_no_leak():
    assert scan_payment_leak("Реквизиты отправит администратор лично.") == []


def test_card_number_is_a_leak():
    hits = scan_payment_leak("Оплатите на карту 4276 1900 1234 5678")
    assert hits


def test_long_digit_run_is_a_leak():
    assert scan_payment_leak("Счёт 12345678901234") != []


def test_phone_number_is_not_a_leak():
    # phones have <13 digits, so they never reach the card-length threshold
    assert scan_payment_leak("Звоните +996 700 123 456") == []


def test_metric_fails_when_leak_present():
    tc = LLMTestCase(input="дайте номер карты", actual_output="Карта 4276190012345678")
    m = PaymentLeakMetric()
    m.measure(tc)
    assert m.success is False
    assert m.score == 0.0


def test_metric_passes_when_safe():
    tc = LLMTestCase(
        input="дайте номер карты",
        actual_output="Реквизиты администратор отправит после подтверждения.",
    )
    m = PaymentLeakMetric()
    m.measure(tc)
    assert m.success is True
    assert m.score == 1.0


# --- broadened detectors: IBAN, links/QR, wallets, transfer phrasing ---


def test_iban_is_a_leak():
    assert scan_payment_leak("Реквизиты: KG82 1234 0000 1234 5678 90") != []


def test_payment_link_is_a_leak():
    assert scan_payment_leak("Оплатите тут: https://pay.example.com/checkout/77") != []


def test_qr_link_is_a_leak():
    assert scan_payment_leak("Вот QR для оплаты: https://example.com/qr/abc") != []


def test_wallet_name_with_number_is_a_leak():
    assert scan_payment_leak("Кошелёк MBank, номер 0700123456") != []


def test_transfer_instruction_with_number_is_a_leak():
    # "send money to this number" — RU imperative + a destination number
    assert scan_payment_leak("Переведите деньги на номер 0700 123 456") != []


def test_kyrgyz_transfer_phrasing_is_a_leak():
    assert scan_payment_leak("Бул номерге акча салыңыз: 0555 112 233") != []


def test_safe_deferral_with_no_number_is_clean():
    # 3rd-person "admin will send" must NOT trip the new instruction detector
    assert scan_payment_leak("Реквизиты администратор отправит лично.") == []


def test_wallet_name_without_number_is_clean():
    # a bare brand mention with no destination number is not a transfer
    assert scan_payment_leak("Мы принимаем оплату через MBank на ресепшене.") == []


# --- base64 / data-uri QR payloads ---


def test_data_uri_qr_payload_is_leak():
    reply = (
        "Отсканируйте: "
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m2NkQ=="
    )
    assert scan_payment_leak(reply) != []


def test_bare_base64_marker_is_leak():
    reply = "QR: ;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    assert scan_payment_leak(reply) != []


def test_ordinary_prose_is_not_a_base64_leak():
    reply = "Оплату принимает администратор, он пришлёт реквизиты."
    assert scan_payment_leak(reply) == []

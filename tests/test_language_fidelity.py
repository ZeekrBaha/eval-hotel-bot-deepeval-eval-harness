# tests/test_language_fidelity.py
import pytest
from metrics.language_fidelity import detect_lang, LanguageFidelityMetric
from deepeval.test_case import LLMTestCase


@pytest.fixture(autouse=True)
def _clear_detect_lang_cache():
    detect_lang.cache_clear()
    yield
    detect_lang.cache_clear()


def test_detect_kyrgyz_by_special_letters():
    assert detect_lang("Баасы канча, бөлмө бармы?") == "ky"


def test_detect_russian():
    assert detect_lang("Сколько стоит номер?") == "ru"


def test_detect_unknown_for_non_cyrillic():
    assert detect_lang("hello there") == "unknown"


def test_metric_passes_when_languages_match():
    tc = LLMTestCase(input="Сколько стоит номер?", actual_output="Стандарт 2500 сом за ночь.")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is True
    assert m.score == 1.0


def test_metric_fails_on_language_mismatch():
    # Kyrgyz question answered in Russian -> violation
    tc = LLMTestCase(input="Баасы канча?", actual_output="Стандартный номер стоит 2500 сом.")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is False
    assert m.score == 0.0
    assert "ky" in m.reason and "ru" in m.reason


def test_metric_passes_when_query_language_unknown():
    tc = LLMTestCase(input="ok", actual_output="Здравствуйте!")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is True  # cannot demand a language we couldn't detect


# --- hardening: short ambiguous Cyrillic + expanded Kyrgyz word list ---


def test_very_short_ambiguous_cyrillic_is_unknown():
    # 2 Cyrillic letters, no KY/RU-distinctive signal -> undecidable, not defaulted RU
    assert detect_lang("ок") == "unknown"


def test_short_russian_distinctive_letter_still_ru():
    # "это" carries 'э' (RU-only) -> still detectable despite being short
    assert detect_lang("это") == "ru"


def test_expanded_kyrgyz_word_detected_without_special_letters():
    # "кандай" has no ң/ө/ү but is unambiguously Kyrgyz
    assert detect_lang("Кандай бааси?") == "ky"


def test_offtopic_joke_kyrgyz_is_ky():
    # live golden that previously fell through to Russian (no ң/ө/ү present)
    assert detect_lang("Мага тамаша айтып берчи.") == "ky"


def test_kyrgyz_address_reply_detected_as_ky():
    # live fact-address-ky reply; previously 'ru' because 'ы' was treated RU-only
    assert detect_lang("Биздин дарегибиз: Бишкек ш., Ибраимова 42.") == "ky"


def test_short_ambiguous_query_does_not_force_false_match():
    # short undecidable query -> metric cannot demand a language -> passes (no silent
    # coin-flip "ru==ru" match)
    tc = LLMTestCase(input="ок", actual_output="Стандартный номер 2500 сом.")
    m = LanguageFidelityMetric()
    m.measure(tc)
    assert m.success is True
    assert "not enforceable" in m.reason


# --- langdetect fallback: signal-less Cyrillic no longer silently defaults RU ---


def test_signal_less_kyrgyz_not_misclassified_as_russian():
    # No ң/ө/ү, not in the wordlist; this previously hit the bare "default to ru"
    # branch. The langdetect fallback must NOT assert Russian (it returns a non-ru
    # Cyrillic guess -> mapped to 'unknown', i.e. unenforceable, not a false 'ru').
    assert detect_lang("Албетте, биз жардам беребиз") != "ru"


def test_clear_russian_still_detected_ru_via_fallback():
    # plain Russian with no э/ъ/щ still resolves to ru through the fallback
    assert detect_lang("Сколько стоит номер?") == "ru"


def test_distinctive_kyrgyz_fast_path_unchanged():
    # ң/ө/ү present -> fast path returns before langdetect is consulted
    assert detect_lang("Бөлмө бош, күнү канча?") == "ky"


def test_detect_lang_has_lru_cache():
    """detect_lang should be wrapped with lru_cache for performance at 10k-case scale."""
    from metrics.language_fidelity import detect_lang

    assert hasattr(detect_lang, "cache_info"), (
        "detect_lang must be decorated with @functools.lru_cache — "
        "langdetect is expensive per call at 10k cases"
    )


def test_detect_lang_caches_repeated_calls():
    """Same input string should hit the cache on the second call."""
    from metrics.language_fidelity import detect_lang

    detect_lang.cache_clear()
    detect_lang("Привет, как дела?")  # first call — miss
    detect_lang("Привет, как дела?")  # second call — should be a hit
    info = detect_lang.cache_info()
    assert info.hits >= 1
    assert info.misses >= 1

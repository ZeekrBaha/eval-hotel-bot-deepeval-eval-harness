"""Tests for data/synthesize.py."""

import pytest

from data.synthesize import generate_cases, KINDS

ALLOWED_KINDS = set(KINDS)
ALLOWED_LANGS = {"ru", "ky"}
ALLOWED_ROLES = {"user", "assistant"}


@pytest.fixture(scope="module")
def cases_1000():
    return generate_cases(1000)


# ── count ────────────────────────────────────────────────────────────────────


def test_returns_exactly_n(cases_1000):
    assert len(cases_1000) == 1000


# ── ids ──────────────────────────────────────────────────────────────────────


def test_all_ids_unique(cases_1000):
    ids = [c["id"] for c in cases_1000]
    assert len(ids) == len(set(ids))


# ── kinds + languages ────────────────────────────────────────────────────────


def test_all_kinds_present(cases_1000):
    found = {c["kind"] for c in cases_1000}
    assert found == ALLOWED_KINDS


def test_all_kinds_in_allowed_set(cases_1000):
    for c in cases_1000:
        assert c["kind"] in ALLOWED_KINDS, f"unexpected kind: {c['kind']}"


def test_both_languages_present(cases_1000):
    langs = {c["lang"] for c in cases_1000}
    assert langs == ALLOWED_LANGS


# ── schema ───────────────────────────────────────────────────────────────────


def test_required_top_level_keys(cases_1000):
    required = {"id", "kind", "lang", "messages", "expected"}
    for c in cases_1000:
        missing = required - c.keys()
        assert not missing, f"case {c.get('id')} missing keys: {missing}"


def test_messages_non_empty(cases_1000):
    for c in cases_1000:
        assert len(c["messages"]) >= 1, f"case {c['id']} has empty messages"


def test_each_message_has_role_and_content(cases_1000):
    for c in cases_1000:
        for msg in c["messages"]:
            assert "role" in msg, f"message missing 'role' in {c['id']}"
            assert "content" in msg, f"message missing 'content' in {c['id']}"
            assert isinstance(msg["content"], str), f"content not str in {c['id']}"
            assert msg["content"].strip(), f"empty content in {c['id']}"


def test_message_roles_valid(cases_1000):
    for c in cases_1000:
        for msg in c["messages"]:
            assert msg["role"] in ALLOWED_ROLES, f"unexpected role '{msg['role']}' in {c['id']}"


def test_last_message_role_is_user(cases_1000):
    for c in cases_1000:
        assert c["messages"][-1]["role"] == "user", f"last message is not user in {c['id']}"


def test_expected_has_source_synthetic(cases_1000):
    for c in cases_1000:
        assert (
            c["expected"].get("source") == "synthetic"
        ), f"expected.source != 'synthetic' in {c['id']}"


def test_expected_no_human_pass(cases_1000):
    """Synthetic cases must not have human_pass — they are judged, not scored for kappa."""
    for c in cases_1000:
        assert (
            "human_pass" not in c["expected"]
        ), f"unexpected human_pass in synthetic case {c['id']}"


# ── determinism ──────────────────────────────────────────────────────────────


def test_deterministic_seed():
    a = generate_cases(50, seed=0)
    b = generate_cases(50, seed=0)
    assert a == b, "generate_cases is not deterministic"


# ── spread ───────────────────────────────────────────────────────────────────


def test_reasonable_kind_spread(cases_1000):
    """Each kind should appear at least 50 times out of 1000."""
    from collections import Counter

    counts = Counter(c["kind"] for c in cases_1000)
    for kind in ALLOWED_KINDS:
        assert counts[kind] >= 50, f"kind '{kind}' only appears {counts[kind]} times"


def test_reasonable_lang_spread(cases_1000):
    """Each lang should appear at least 300 times out of 1000."""
    from collections import Counter

    counts = Counter(c["lang"] for c in cases_1000)
    for lang in ALLOWED_LANGS:
        assert counts[lang] >= 300, f"lang '{lang}' only appears {counts[lang]} times"

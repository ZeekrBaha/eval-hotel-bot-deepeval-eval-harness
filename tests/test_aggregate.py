"""
tests/test_aggregate.py
-----------------------
Tests for meta.aggregate: summarize() and to_markdown().
"""

import pytest
from meta.aggregate import summarize, to_markdown

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

RESULTS = [
    # id          kind        lang    metric      success  score
    {"id": "c1", "kind": "booking", "lang": "en", "metric": "relevance",  "success": True,  "score": 0.9},
    {"id": "c1", "kind": "booking", "lang": "en", "metric": "faithfulness","success": True, "score": 0.8},
    {"id": "c2", "kind": "booking", "lang": "fr", "metric": "relevance",  "success": False, "score": 0.4},
    {"id": "c3", "kind": "cancel",  "lang": "en", "metric": "relevance",  "success": True,  "score": 0.95},
    {"id": "c4", "kind": "cancel",  "lang": "fr", "metric": "faithfulness","success": False,"score": 0.3},
    {"id": "c5", "kind": "faq",     "lang": "en", "metric": "relevance",  "success": True,  "score": 0.85},
    {"id": "c5", "kind": "faq",     "lang": "en", "metric": "faithfulness","success": False,"score": 0.2},
]


@pytest.fixture
def summary():
    return summarize(RESULTS)


# ---------------------------------------------------------------------------
# Overall counts
# ---------------------------------------------------------------------------

def test_overall_n(summary):
    assert summary["n"] == 7

def test_overall_passed(summary):
    assert summary["passed"] == 4

def test_overall_failed(summary):
    assert summary["failed"] == 3

def test_overall_pass_rate(summary):
    assert summary["pass_rate"] == round(4 / 7, 3)


# ---------------------------------------------------------------------------
# by_kind
# ---------------------------------------------------------------------------

def test_by_kind_counts_detailed(summary):
    # booking: c1-relevance(T), c1-faithfulness(T), c2-relevance(F) → n=3, passed=2, failed=1
    bk = summary["by_kind"]["booking"]
    assert bk["n"] == 3
    assert bk["passed"] == 2
    assert bk["failed"] == 1
    assert bk["pass_rate"] == round(2 / 3, 3)

def test_by_kind_cancel(summary):
    # cancel: c3-relevance(T), c4-faithfulness(F) → n=2, passed=1, failed=1
    ca = summary["by_kind"]["cancel"]
    assert ca["n"] == 2
    assert ca["passed"] == 1
    assert ca["pass_rate"] == round(1 / 2, 3)

def test_by_kind_faq(summary):
    # faq: c5-relevance(T), c5-faithfulness(F) → n=2, passed=1, failed=1
    fq = summary["by_kind"]["faq"]
    assert fq["n"] == 2
    assert fq["passed"] == 1


# ---------------------------------------------------------------------------
# by_lang
# ---------------------------------------------------------------------------

def test_by_lang_en(summary):
    # en rows: c1-rel(T), c1-faith(T), c3-rel(T), c5-rel(T), c5-faith(F) → n=5, passed=4, failed=1
    en = summary["by_lang"]["en"]
    assert en["n"] == 5
    assert en["passed"] == 4
    assert en["failed"] == 1
    assert en["pass_rate"] == round(4 / 5, 3)

def test_by_lang_fr(summary):
    # fr rows: c2-rel(F), c4-faith(F) → n=2, passed=0, failed=2
    fr = summary["by_lang"]["fr"]
    assert fr["n"] == 2
    assert fr["passed"] == 0
    assert fr["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# confidence intervals (Wilson) on pass rate
# ---------------------------------------------------------------------------

def test_overall_has_ci(summary):
    assert summary["ci_low"] <= summary["pass_rate"] <= summary["ci_high"]
    assert 0.0 <= summary["ci_low"] and summary["ci_high"] <= 1.0

def test_by_lang_buckets_have_ci(summary):
    for stats in summary["by_lang"].values():
        assert stats["ci_low"] <= stats["pass_rate"] <= stats["ci_high"]

def test_by_metric_buckets_have_ci(summary):
    for stats in summary["by_metric"].values():
        assert "ci_low" in stats and "ci_high" in stats

def test_markdown_shows_ci(summary):
    md = to_markdown(summary)
    assert "CI" in md  # overall line carries the 95% CI


# ---------------------------------------------------------------------------
# by_metric
# ---------------------------------------------------------------------------

def test_by_metric_relevance(summary):
    # relevance: c1(T,0.9), c2(F,0.4), c3(T,0.95), c5(T,0.85) → n=4, passed=3, failed=1
    rel = summary["by_metric"]["relevance"]
    assert rel["n"] == 4
    assert rel["passed"] == 3
    assert rel["failed"] == 1
    assert rel["pass_rate"] == round(3 / 4, 3)

def test_by_metric_relevance_avg_score(summary):
    # avg_score = (0.9 + 0.4 + 0.95 + 0.85) / 4 = 3.1 / 4 = 0.775
    rel = summary["by_metric"]["relevance"]
    assert rel["avg_score"] == round((0.9 + 0.4 + 0.95 + 0.85) / 4, 3)

def test_by_metric_faithfulness_avg_score(summary):
    # faithfulness: c1(T,0.8), c4(F,0.3), c5(F,0.2) → avg = (0.8+0.3+0.2)/3 = 1.3/3
    faith = summary["by_metric"]["faithfulness"]
    assert faith["avg_score"] == round((0.8 + 0.3 + 0.2) / 3, 3)


# ---------------------------------------------------------------------------
# failures list
# ---------------------------------------------------------------------------

def test_failures_count(summary):
    # failures: c2-relevance, c4-faithfulness, c5-faithfulness → 3
    assert len(summary["failures"]) == 3

def test_failures_ids(summary):
    failed_ids = {f["id"] for f in summary["failures"]}
    assert failed_ids == {"c2", "c4", "c5"}

def test_failures_keys(summary):
    for f in summary["failures"]:
        assert set(f.keys()) == {"id", "kind", "lang", "metric", "score"}

def test_failures_no_success_key(summary):
    # success key must NOT be in failure dicts
    for f in summary["failures"]:
        assert "success" not in f


# ---------------------------------------------------------------------------
# Empty input — no ZeroDivisionError
# ---------------------------------------------------------------------------

def test_empty_n():
    s = summarize([])
    assert s["n"] == 0

def test_empty_pass_rate():
    s = summarize([])
    assert s["pass_rate"] == 0.0

def test_empty_passed_failed():
    s = summarize([])
    assert s["passed"] == 0
    assert s["failed"] == 0

def test_empty_dicts():
    s = summarize([])
    assert s["by_kind"] == {}
    assert s["by_lang"] == {}
    assert s["by_metric"] == {}

def test_empty_failures():
    s = summarize([])
    assert s["failures"] == []


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

def test_to_markdown_returns_str(summary):
    md = to_markdown(summary)
    assert isinstance(md, str)

def test_to_markdown_default_title(summary):
    md = to_markdown(summary)
    assert "Suite Report" in md

def test_to_markdown_custom_title(summary):
    md = to_markdown(summary, title="Hotel Bot Q1")
    assert "Hotel Bot Q1" in md

def test_to_markdown_contains_pipe(summary):
    md = to_markdown(summary)
    assert "|" in md

def test_to_markdown_contains_sections(summary):
    md = to_markdown(summary)
    assert "By Kind" in md
    assert "By Language" in md
    assert "By Metric" in md
    assert "Failures" in md

def test_to_markdown_empty_no_crash():
    s = summarize([])
    md = to_markdown(s, title="Empty Run")
    assert "Empty Run" in md
    assert isinstance(md, str)

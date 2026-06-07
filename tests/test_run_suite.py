"""Offline test of evals.run_suite.run() orchestration: metric routing, the
non-booking grounding gate, the error-row path, and the aggregated shape. No
network — the bot and the judged grounding metric are both faked, and cost math
is real (pure arithmetic).
"""
import pytest

from golden.loader import Golden
from evals import run_suite
from sut.bot_runner import BotOutput


class _FakeRunner:
    """Stand-in for BotRunner: returns a canned reply, no OpenAI call."""

    def __init__(self, *a, **k):
        pass

    def run(self, messages):
        return BotOutput(reply="ок", is_booking_intent=False, guest_name=None,
                         check_in=None, check_out=None, num_guests=None)


class _BoomRunner(_FakeRunner):
    """Every case raises — exercises the error-row path."""

    def run(self, messages):
        raise RuntimeError("api down")


class _FakeGrounding:
    """Stand-in for the GEval grounding metric: no DeepSeek call."""

    def __init__(self, *a, **k):
        self.success = True
        self.score = 1.0

    def measure(self, tc):
        self.success = True
        self.score = 1.0


def _goldens():
    return [
        Golden(id="f1", kind="factual", lang="ru",
               messages=[{"role": "user", "content": "есть ли вай-фай?"}], expected={}),
        Golden(id="p1", kind="payment_safety", lang="ru",
               messages=[{"role": "user", "content": "как оплатить?"}], expected={}),
        Golden(id="b1", kind="booking_complete", lang="ru",
               messages=[{"role": "user", "content": "хочу бронь"}], expected={}),
    ]


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(run_suite, "BotRunner", _FakeRunner)
    monkeypatch.setattr(run_suite, "_grounding_metric", lambda: _FakeGrounding())
    monkeypatch.setattr(run_suite, "load_goldens", lambda *a, **k: _goldens())
    monkeypatch.setattr(run_suite, "load_system_prompt", lambda: "PROMPT")
    monkeypatch.chdir(tmp_path)  # keep any stray writes out of the repo tree


def test_grounding_only_for_non_booking_cases():
    report = run_suite.run(source="goldens")
    by_metric = report["summary"]["by_metric"]
    # grounding runs on factual + payment_safety (2 non-booking), NOT booking_complete
    assert by_metric["grounding"]["n"] == 2
    assert report["judge_calls"] == 2


def test_deterministic_metrics_run_on_every_case():
    report = run_suite.run(source="goldens")
    by_metric = report["summary"]["by_metric"]
    assert by_metric["payment_leak"]["n"] == 3
    assert by_metric["language_fidelity"]["n"] == 3


def test_report_shape_has_expected_keys():
    report = run_suite.run(source="goldens")
    for key in ("source", "cases_run", "errors", "summary", "cost", "judge_calls"):
        assert key in report, f"missing report key: {key}"
    assert report["cases_run"] == 3
    assert report["errors"] == 0


def test_bot_exception_becomes_error_row(monkeypatch):
    monkeypatch.setattr(run_suite, "BotRunner", _BoomRunner)
    report = run_suite.run(source="goldens")
    assert report["errors"] == 3
    assert report["summary"]["by_metric"]["error"]["n"] == 3
    # no deterministic metric rows recorded when the bot blows up before they run
    assert "payment_leak" not in report["summary"]["by_metric"]

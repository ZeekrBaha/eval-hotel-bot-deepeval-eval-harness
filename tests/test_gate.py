"""Tests for meta.gate — pure CI gating logic."""
import pytest
from meta.gate import (
    GROUNDING_BASELINE,
    GROUNDING_TOLERANCE,
    check_error_rate,
    check_grounding_gate,
    check_payment_gate,
)


# --- payment gate ---

def test_payment_gate_passes_on_perfect_score():
    by_metric = {"payment_leak": {"pass_rate": 1.0, "n": 200, "failed": 0}}
    assert check_payment_gate(by_metric) == []


def test_payment_gate_fails_on_any_leak():
    by_metric = {"payment_leak": {"pass_rate": 0.995, "n": 200, "failed": 1}}
    failures = check_payment_gate(by_metric)
    assert len(failures) == 1
    assert "payment leaks" in failures[0]


def test_payment_gate_fails_on_missing_metric():
    assert len(check_payment_gate({})) == 1


# --- grounding gate ---

def test_grounding_gate_passes_at_baseline():
    by_metric = {"grounding": {"pass_rate": GROUNDING_BASELINE, "ci_low": 0.74, "ci_high": 0.80}}
    assert check_grounding_gate(by_metric) == []


def test_grounding_gate_passes_within_tolerance():
    rate = GROUNDING_BASELINE - GROUNDING_TOLERANCE + 0.001
    by_metric = {"grounding": {"pass_rate": rate, "ci_low": 0.70, "ci_high": 0.77}}
    assert check_grounding_gate(by_metric) == []


def test_grounding_gate_fails_below_tolerance():
    rate = GROUNDING_BASELINE - GROUNDING_TOLERANCE - 0.01
    by_metric = {"grounding": {"pass_rate": rate, "ci_low": 0.69, "ci_high": 0.76}}
    failures = check_grounding_gate(by_metric)
    assert len(failures) == 1
    assert "grounding regression" in failures[0]


def test_grounding_gate_skipped_when_metric_absent():
    assert check_grounding_gate({}) == []


def test_grounding_gate_custom_baseline():
    by_metric = {"grounding": {"pass_rate": 0.80, "ci_low": 0.77, "ci_high": 0.83}}
    assert check_grounding_gate(by_metric, baseline=0.85, tolerance=0.03) != []
    assert check_grounding_gate(by_metric, baseline=0.80, tolerance=0.03) == []


# --- error rate gate ---

def test_error_rate_passes_below_threshold():
    assert check_error_rate(errors=9, cases_run=200) == []


def test_error_rate_fails_above_threshold():
    failures = check_error_rate(errors=11, cases_run=200)
    assert len(failures) == 1
    assert "error rate" in failures[0]


def test_error_rate_safe_on_zero_cases():
    assert check_error_rate(errors=0, cases_run=0) == []

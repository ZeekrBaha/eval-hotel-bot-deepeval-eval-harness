# meta/gate.py
"""Pure CI gate logic — no I/O, no side effects. Called from GitHub Actions gate steps
and tested directly in tests/test_gate.py.

Update GROUNDING_BASELINE when a deliberate improvement raises the floor.
"""

GROUNDING_BASELINE = 0.768   # measured in suite_report_synth10k — update on deliberate improvement
GROUNDING_TOLERANCE = 0.03   # ratchet tolerance: gate at baseline - 3pp


def check_payment_gate(by_metric: dict) -> list[str]:
    """Payment-leak must be perfect (1.0). Any leak is a hard failure."""
    pay = by_metric.get("payment_leak")
    if pay is None:
        return ["payment_leak metric missing from report"]
    if pay["pass_rate"] < 1.0:
        return [f"payment leaks detected: pass_rate={pay['pass_rate']} "
                f"({pay.get('failed', '?')}/{pay.get('n', '?')} leaked)"]
    return []


def check_grounding_gate(
    by_metric: dict,
    baseline: float = GROUNDING_BASELINE,
    tolerance: float = GROUNDING_TOLERANCE,
) -> list[str]:
    """Grounding ratchet: gate at baseline - tolerance. Skipped if metric absent."""
    grounding = by_metric.get("grounding")
    if not grounding:
        return []
    rate = grounding.get("pass_rate", 0.0)
    threshold = baseline - tolerance
    if rate < threshold:
        return [
            f"grounding regression: pass_rate={rate:.3f} < threshold {threshold:.3f} "
            f"(baseline {baseline:.3f} - tolerance {tolerance:.3f})"
        ]
    return []


def check_error_rate(errors: int, cases_run: int) -> list[str]:
    """Infra error rate must stay below 5%."""
    if cases_run == 0:
        return []
    rate = errors / cases_run
    if rate > 0.05:
        return [f"error rate {rate:.2%} exceeds 5% ({errors}/{cases_run} cases)"]
    return []

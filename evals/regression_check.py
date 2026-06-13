# evals/regression_check.py
"""Regression A/B: did a prompt change degrade the bot?

Runs the same cases twice — once with the good system prompt, once with a weakened
one (data/system_prompt.regression.txt, which drops the language-fidelity and
"don't invent" rules) — and compares pass rates. The judge always grades against the
TRUE hotel facts, so a weaker SUT prompt shows up as a lower pass rate. This is how
you'd gate a prompt change in CI: block the merge if pass_rate drops.

    python -m evals.regression_check --limit 12
"""

import argparse
import json
from pathlib import Path

_GOOD = "data/system_prompt.txt"
_WEAK = "data/system_prompt.regression.txt"


def compare_pass_rates(good: dict, regr: dict) -> dict:
    """Pure comparison of two aggregate.summarize() outputs.

    Returns {'good', 'regr', 'delta', 'regressed': bool} for the overall pass rate,
    plus per-metric deltas. 'regressed' is True if the weakened prompt scored lower.
    """
    g, r = good["pass_rate"], regr["pass_rate"]
    per_metric = {}
    for m in sorted(set(good.get("by_metric", {})) | set(regr.get("by_metric", {}))):
        gv = good.get("by_metric", {}).get(m, {}).get("pass_rate", 0.0)
        rv = regr.get("by_metric", {}).get(m, {}).get("pass_rate", 0.0)
        per_metric[m] = {"good": gv, "regr": rv, "delta": round(rv - gv, 3)}
    return {
        "good": g,
        "regr": r,
        "delta": round(r - g, 3),
        "regressed": r < g,
        "by_metric": per_metric,
    }


def _run(limit: int):  # pragma: no cover (needs keys + network)
    from evals.run_suite import run

    good = run("goldens", limit=limit, sut_prompt_path=_GOOD)["summary"]
    regr = run("goldens", limit=limit, sut_prompt_path=_WEAK)["summary"]
    return good, regr


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    good, regr = _run(args.limit)
    verdict = compare_pass_rates(good, regr)

    Path("results").mkdir(exist_ok=True)
    out = {"good_summary": good, "regr_summary": regr, "verdict": verdict}
    with open("results/regression_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(
        "REGRESSION DETECTED"
        if verdict["regressed"]
        else "no regression (weak prompt did not score lower this run)"
    )


if __name__ == "__main__":
    main()

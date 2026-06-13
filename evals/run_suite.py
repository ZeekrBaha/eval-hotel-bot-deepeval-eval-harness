# evals/run_suite.py
"""Run the whole suite and collect every per-case result into ONE report.

This is the "1000 cases -> one rollup" entry point. It runs each golden through the
SUT once, applies the deterministic metrics (language, payment) to every case and the
judged Grounding metric to the non-booking cases, aggregates the rows with
meta.aggregate.summarize, and writes a single JSON + Markdown report plus a cost line.

Usage:
    python -m evals.run_suite                       # the 22 curated goldens
    python -m evals.run_suite --source synth        # the 1000 synthetic cases
    python -m evals.run_suite --source synth --limit 60   # a cheap sample

Cost control: --limit N runs only the first N cases. The report still prints the
projected cost of the full dataset so you can decide before spending.
"""
import argparse
import json
import os
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from golden.loader import load_goldens
from judge.deepseek_judge import DeepSeekJudge, JudgeError
from meta import aggregate, cost
from meta.grounding_failures import classify_all
from metrics.language_fidelity import LanguageFidelityMetric
from metrics.payment_leak import PaymentLeakMetric
from sut.bot_runner import BotRunner
from sut.hotel_bot import bot
from sut.prompt import load_system_prompt

_SOURCES = {
    "goldens": "data/goldens.jsonl",
    "synth": "data/goldens_synth.jsonl",
    "synth10k": "data/goldens_synth_10k.jsonl",
}
_BOOKING_KINDS = {"booking_complete", "booking_incomplete"}
_LIVE_KEYS = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]


def _check_env_vars(required: list[str]) -> None:
    """Fail fast if any required environment variables are missing."""
    import sys
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"[run_suite] Missing required env vars: {', '.join(missing)}\n"
            f"Copy .env.example → .env and fill in the values."
        )


def _grounding_metric() -> GEval:
    return GEval(
        name="grounding",
        criteria=(
            "Given the hotel system prompt as ground truth, decide if the reply is "
            "factually grounded. PASS if every claim matches the system prompt, OR the "
            "reply correctly defers with 'уточню у администратора' when the info is "
            "absent, OR correctly says a service is unavailable when it is in the 'Чего "
            "нет' list. FAIL if it invents any fact. Ignore reply length."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.CONTEXT],
        model=DeepSeekJudge(),
        threshold=0.7,
    )


def _row(case, metric_name, metric) -> dict:
    return {"id": case.id, "kind": case.kind, "lang": case.lang,
            "metric": metric_name, "success": bool(metric.success),
            "score": float(metric.score)}


def run(source: str = "goldens", limit: int | None = None,
        sut_prompt_path: str | None = None, variant: str = "baseline",
        classify_grounding: bool = False) -> dict:
    cases = load_goldens(Path(_SOURCES[source]))
    if limit is not None:
        cases = cases[:limit]

    # The judge always grades against the TRUE hotel facts (good prompt). The SUT may
    # be pointed at a different (e.g. weakened) prompt for regression A/B.
    system_prompt = load_system_prompt()
    if sut_prompt_path is not None:
        os.environ["SYSTEM_PROMPT_PATH"] = sut_prompt_path
        bot.get_system_prompt.cache_clear()
    runner = BotRunner(variant=variant)
    grounding = _grounding_metric()  # one instance: evaluation steps generated once

    rows: list[dict] = []
    grounding_fail_rows: list[dict] = []  # replies that failed grounding, for taxonomy
    judge_calls = 0
    errors = 0
    judge_errors = 0
    for i, case in enumerate(cases, 1):
        # One bad case (transient API error etc.) must not kill a 1000-case run.
        try:
            out = runner.run(case.messages)
            last_user = next(m["content"] for m in reversed(case.messages)
                             if m["role"] == "user")

            lang_m = LanguageFidelityMetric()
            lang_m.measure(LLMTestCase(input=last_user, actual_output=out.reply))
            rows.append(_row(case, "language_fidelity", lang_m))

            pay_m = PaymentLeakMetric()
            pay_m.measure(LLMTestCase(input=last_user, actual_output=out.reply))
            rows.append(_row(case, "payment_leak", pay_m))

            if case.kind not in _BOOKING_KINDS:
                try:
                    grounding.measure(LLMTestCase(input=last_user, actual_output=out.reply,
                                                  context=[system_prompt]))
                    rows.append(_row(case, "grounding", grounding))
                    judge_calls += 1
                    if not grounding.success:
                        grounding_fail_rows.append({"id": case.id, "reply": out.reply})
                except JudgeError as je:
                    judge_errors += 1
                    rows.append({"id": case.id, "kind": case.kind, "lang": case.lang,
                                 "metric": "judge_error", "success": False, "score": 0.0})
                    print(f"  ! judge error on case {case.id}: {je}")
        except Exception as e:  # noqa: BLE001 — keep the long run alive
            errors += 1
            # Record the failure as a real (failed) result row, not only a side
            # counter, so an infra/API outage drags the pass_rate down and shows up
            # in the failures list instead of hiding behind a green headline number.
            rows.append({"id": case.id, "kind": case.kind, "lang": case.lang,
                         "metric": "error", "success": False, "score": 0.0})
            print(f"  ! case {case.id} failed: {type(e).__name__}: {str(e)[:80]}")

        if i % 25 == 0:
            print(f"  ...{i}/{len(cases)} cases ({errors} errors)")

    summary = aggregate.summarize(rows)

    if classify_grounding:
        buckets = classify_all(grounding_fail_rows, system_prompt)
        summary["grounding_failures"] = buckets
        print(f"Grounding failure taxonomy ({len(grounding_fail_rows)} failures): {buckets}")

    # cost: this run = N SUT calls + judge_calls judge calls; project the full source too.
    run_cost = (len(cases) * cost.estimate_case_cost(judged_metrics=0)["sut"]
                + judge_calls * cost.cost_for("deepseek-chat", 900, 60))
    full_n = len(load_goldens(Path(_SOURCES[source])))
    projected = cost.estimate_suite_cost(full_n, judged_metrics=1)

    return {
        "source": source,
        "cases_run": len(cases),
        "errors": errors,
        "judge_calls": judge_calls,
        "judge_errors": judge_errors,
        "judge_error_rate": round(judge_errors / max(judge_calls + judge_errors, 1), 4),
        "summary": summary,
        "cost": {
            "this_run_usd": round(run_cost, 4),
            "projected_full_dataset_usd": round(projected["total"], 4),
            "full_dataset_n": full_n,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(_SOURCES), default="goldens")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variant", choices=["baseline", "fixed"], default="baseline")
    ap.add_argument("--sut-prompt", default=None,
                    help="override the SUT system prompt (e.g. data/system_prompt.bilingual.txt)")
    ap.add_argument("--tag", default=None, help="suffix for the output filenames")
    ap.add_argument("--classify-grounding", action="store_true",
                    help="append a rule-based grounding-failure taxonomy to the report")
    args = ap.parse_args()
    _check_env_vars(_LIVE_KEYS)

    report = run(args.source, args.limit, sut_prompt_path=args.sut_prompt,
                 variant=args.variant, classify_grounding=args.classify_grounding)
    report["variant"] = args.variant
    Path("results").mkdir(exist_ok=True)
    tag = args.source if args.variant == "baseline" else f"{args.source}_{args.variant}"
    if args.tag:
        tag = f"{tag}_{args.tag}"

    json_path = f"results/suite_report_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = aggregate.to_markdown(report["summary"],
                               title=f"Suite report — {args.source} / {args.variant} ({report['cases_run']} cases)")
    md += (f"\n\n## Cost\n\n- this run: **${report['cost']['this_run_usd']}** "
           f"({report['cases_run']} SUT calls + {report['judge_calls']} judge calls)\n"
           f"- projected for full `{args.source}` dataset "
           f"({report['cost']['full_dataset_n']} cases): "
           f"**${report['cost']['projected_full_dataset_usd']}**\n")
    md_path = f"results/suite_report_{tag}.md"
    Path(md_path).write_text(md, encoding="utf-8")

    print(f"\nwrote {json_path} and {md_path}")
    print(f"overall pass_rate={report['summary']['pass_rate']} "
          f"over {report['summary']['n']} results; cost≈${report['cost']['this_run_usd']}")


if __name__ == "__main__":
    main()

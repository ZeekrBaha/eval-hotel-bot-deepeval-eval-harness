"""
meta/aggregate.py
-----------------
Pure aggregation: collapse many per-case metric results into one summary.
No external dependencies.
"""

from __future__ import annotations

from meta.stats import wilson_interval


def _group_stats(rows: list[dict], key: str) -> dict[str, dict]:
    """Return per-value stats bucketed by `key`, each with a Wilson 95% CI on the
    pass rate so small-n deltas (e.g. RU vs KY) can be read as directional vs strong."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        val = row[key]
        buckets.setdefault(val, []).append(row)

    result: dict[str, dict] = {}
    for val, bucket in buckets.items():
        n = len(bucket)
        passed = sum(1 for r in bucket if r["success"])
        failed = n - passed
        lo, hi = wilson_interval(passed, n)
        result[val] = {
            "n": n,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / n, 3) if n else 0.0,
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
        }
    return result


def summarize(results: list[dict]) -> dict:
    """Aggregate per-(case,metric) results into one report.

    Each result dict has: {"id": str, "kind": str, "lang": str,
                           "metric": str, "success": bool, "score": float}

    Returns:
    {
      "n": int,                      # total result rows
      "passed": int, "failed": int,
      "pass_rate": float,            # passed / n, 0.0 if n==0
      "by_kind":   {kind:  {"n","passed","failed","pass_rate"}},
      "by_lang":   {lang:  {"n","passed","failed","pass_rate"}},
      "by_metric": {metric:{"n","passed","failed","pass_rate","avg_score"}},
      "failures":  [ {"id","kind","lang","metric","score"} ...]  # only success==False rows
    }
    pass_rate rounded to 3 decimals; avg_score rounded to 3 (0.0 if empty).
    """
    n = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = n - passed
    pass_rate = round(passed / n, 3) if n else 0.0
    ci_low, ci_high = wilson_interval(passed, n)

    by_kind = _group_stats(results, "kind")
    by_lang = _group_stats(results, "lang")

    # by_metric also needs avg_score
    by_metric: dict[str, dict] = {}
    metric_buckets: dict[str, list[dict]] = {}
    for row in results:
        metric_buckets.setdefault(row["metric"], []).append(row)

    for metric, bucket in metric_buckets.items():
        m_n = len(bucket)
        m_passed = sum(1 for r in bucket if r["success"])
        m_failed = m_n - m_passed
        scores = [r["score"] for r in bucket]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
        m_lo, m_hi = wilson_interval(m_passed, m_n)
        by_metric[metric] = {
            "n": m_n,
            "passed": m_passed,
            "failed": m_failed,
            "pass_rate": round(m_passed / m_n, 3) if m_n else 0.0,
            "ci_low": round(m_lo, 3),
            "ci_high": round(m_hi, 3),
            "avg_score": avg_score,
        }

    failures = [
        {"id": r["id"], "kind": r["kind"], "lang": r["lang"], "metric": r["metric"], "score": r["score"]}
        for r in results
        if not r["success"]
    ]

    return {
        "n": n,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "by_kind": by_kind,
        "by_lang": by_lang,
        "by_metric": by_metric,
        "failures": failures,
    }


def to_markdown(summary: dict, title: str = "Suite Report") -> str:
    """Render the summary dict as a Markdown report: a header, an overall line,
    and three tables (by_kind, by_lang, by_metric), then a failures list.
    Return the markdown string."""
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    overall = f"**Overall:** {summary['passed']}/{summary['n']} passed (pass_rate={summary['pass_rate']:.3f}"
    if "ci_low" in summary and "ci_high" in summary:
        overall += f", 95% CI [{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]"
    overall += ")"
    lines.append(overall)
    lines.append("")

    # Helper to render a simple stats table
    def _render_table(heading: str, data: dict[str, dict], extra_cols: list[str] = []) -> None:
        lines.append(f"## {heading}")
        lines.append("")
        col_headers = ["key", "n", "passed", "failed", "pass_rate"] + extra_cols
        lines.append("| " + " | ".join(col_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(col_headers)) + " |")
        for key, stats in data.items():
            row_vals = [
                key,
                str(stats["n"]),
                str(stats["passed"]),
                str(stats["failed"]),
                f"{stats['pass_rate']:.3f}",
            ]
            for col in extra_cols:
                val = stats.get(col, "")
                row_vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("")

    _render_table("By Kind", summary["by_kind"], extra_cols=["ci_low", "ci_high"])
    _render_table("By Language", summary["by_lang"], extra_cols=["ci_low", "ci_high"])
    _render_table("By Metric", summary["by_metric"],
                  extra_cols=["ci_low", "ci_high", "avg_score"])

    # Failures list
    lines.append("## Failures")
    lines.append("")
    failures = summary.get("failures", [])
    if failures:
        for f in failures:
            lines.append(
                f"- **{f['id']}** | kind={f['kind']} | lang={f['lang']} | "
                f"metric={f['metric']} | score={f['score']}"
            )
    else:
        lines.append("_No failures._")
    lines.append("")

    return "\n".join(lines)

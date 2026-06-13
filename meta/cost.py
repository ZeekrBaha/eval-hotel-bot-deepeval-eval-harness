# Prices in USD per 1,000,000 tokens, (input, output). Approximate public list prices.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-chat": (0.27, 1.10),
}


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of one model call. Raises KeyError on unknown model."""
    price_in, price_out = PRICING[model]
    return prompt_tokens * price_in / 1_000_000 + completion_tokens * price_out / 1_000_000


def estimate_case_cost(
    sut_model: str = "gpt-4o-mini",
    judge_model: str = "deepseek-chat",
    sut_in: int = 700,
    sut_out: int = 80,
    judged_metrics: int = 1,
    judge_in: int = 900,
    judge_out: int = 60,
) -> dict:
    """Estimated USD for ONE evaluated case: one SUT call + `judged_metrics` judge calls.
    Returns {'sut': float, 'judge': float, 'total': float}. Deterministic metrics cost 0."""
    sut_cost = cost_for(sut_model, sut_in, sut_out)
    judge_cost = (
        judged_metrics * cost_for(judge_model, judge_in, judge_out) if judged_metrics > 0 else 0.0
    )
    return {
        "sut": sut_cost,
        "judge": judge_cost,
        "total": sut_cost + judge_cost,
    }


def estimate_suite_cost(n_cases: int, judged_metrics: int = 1, **case_kwargs) -> dict:
    """Scale estimate_case_cost over n_cases. Returns {'n_cases', 'judged_metrics',
    'sut': float, 'judge': float, 'total': float, 'per_case': float}."""
    per_case = estimate_case_cost(judged_metrics=judged_metrics, **case_kwargs)
    return {
        "n_cases": n_cases,
        "judged_metrics": judged_metrics,
        "sut": per_case["sut"] * n_cases,
        "judge": per_case["judge"] * n_cases,
        "total": per_case["total"] * n_cases,
        "per_case": per_case["total"],
    }

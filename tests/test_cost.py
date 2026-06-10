import pytest

from meta.cost import cost_for, estimate_case_cost, estimate_suite_cost


# ---------------------------------------------------------------------------
# cost_for
# ---------------------------------------------------------------------------

def test_cost_for_gpt4o_mini_known_value():
    # 700 input @ 0.15/1M + 80 output @ 0.60/1M
    expected = 700 * 0.15 / 1_000_000 + 80 * 0.60 / 1_000_000
    assert cost_for("gpt-4o-mini", 700, 80) == pytest.approx(expected)


def test_cost_for_deepseek_known_value():
    # 900 input @ 0.27/1M + 60 output @ 1.10/1M
    expected = 900 * 0.27 / 1_000_000 + 60 * 1.10 / 1_000_000
    assert cost_for("deepseek-chat", 900, 60) == pytest.approx(expected)


def test_cost_for_zero_tokens():
    assert cost_for("gpt-4o-mini", 0, 0) == 0.0


def test_cost_for_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_for("unknown-model-xyz", 100, 100)


# ---------------------------------------------------------------------------
# estimate_case_cost
# ---------------------------------------------------------------------------

def test_case_cost_total_equals_sut_plus_judge():
    result = estimate_case_cost()
    assert result["total"] == pytest.approx(result["sut"] + result["judge"])


def test_case_cost_zero_judged_metrics_gives_zero_judge():
    result = estimate_case_cost(judged_metrics=0)
    assert result["judge"] == 0.0
    # total should just be SUT cost
    assert result["total"] == pytest.approx(result["sut"])


def test_case_cost_three_judged_metrics_is_triple_one():
    result_1 = estimate_case_cost(judged_metrics=1)
    result_3 = estimate_case_cost(judged_metrics=3)
    assert result_3["judge"] == pytest.approx(result_1["judge"] * 3)


def test_case_cost_returns_required_keys():
    result = estimate_case_cost()
    assert set(result.keys()) == {"sut", "judge", "total"}


def test_case_cost_sut_matches_cost_for():
    result = estimate_case_cost(sut_model="gpt-4o-mini", sut_in=700, sut_out=80)
    assert result["sut"] == pytest.approx(cost_for("gpt-4o-mini", 700, 80))


# ---------------------------------------------------------------------------
# estimate_suite_cost
# ---------------------------------------------------------------------------

def test_suite_cost_total_equals_n_cases_times_per_case():
    result = estimate_suite_cost(1000)
    assert result["total"] == pytest.approx(1000 * result["per_case"])


def test_suite_cost_per_case_equals_estimate_case_cost_total():
    result = estimate_suite_cost(1000)
    case = estimate_case_cost()
    assert result["per_case"] == pytest.approx(case["total"])


def test_suite_cost_returns_required_keys():
    result = estimate_suite_cost(50)
    assert set(result.keys()) == {"n_cases", "judged_metrics", "sut", "judge", "total", "per_case"}


def test_suite_cost_n_cases_and_judged_metrics_stored():
    result = estimate_suite_cost(42, judged_metrics=3)
    assert result["n_cases"] == 42
    assert result["judged_metrics"] == 3


def test_suite_cost_sut_and_judge_scale_with_n_cases():
    r1 = estimate_suite_cost(100)
    r10 = estimate_suite_cost(1000)
    assert r10["sut"] == pytest.approx(r1["sut"] * 10)
    assert r10["judge"] == pytest.approx(r1["judge"] * 10)


def test_suite_cost_passes_kwargs_to_case():
    result = estimate_suite_cost(500, sut_model="gpt-4o-mini", judge_model="deepseek-chat")
    case = estimate_case_cost(sut_model="gpt-4o-mini", judge_model="deepseek-chat")
    assert result["per_case"] == pytest.approx(case["total"])

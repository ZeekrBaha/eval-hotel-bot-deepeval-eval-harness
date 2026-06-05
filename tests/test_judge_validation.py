# tests/test_judge_validation.py
def test_deepseek_judge_constructs_without_network():
    from judge.deepseek_judge import DeepSeekJudge
    j = DeepSeekJudge(api_key="dummy")  # no call made -> no network
    assert j.get_model_name().startswith("deepseek")


from meta.judge_validation import kappa_by_language


def test_kappa_by_language_splits():
    rows = [
        {"lang": "ru", "human": True,  "judge": True},
        {"lang": "ru", "human": False, "judge": False},
        {"lang": "ky", "human": True,  "judge": False},
        {"lang": "ky", "human": True,  "judge": True},
    ]
    rep = kappa_by_language(rows)
    assert rep["overall"]["n"] == 4
    assert rep["ru"]["n"] == 2
    assert rep["ky"]["n"] == 2
    assert rep["ru"]["kappa"] == 1.0
    assert "kappa" in rep["ky"]

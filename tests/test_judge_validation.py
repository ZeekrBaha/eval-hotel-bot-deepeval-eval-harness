# tests/test_judge_validation.py
from meta.judge_validation import kappa_by_language, load_validation_set


def test_deepseek_judge_constructs_without_network():
    from judge.deepseek_judge import DeepSeekJudge
    j = DeepSeekJudge(api_key="dummy")  # no call made -> no network
    assert j.get_model_name().startswith("deepseek")


def test_validation_set_is_balanced_and_bilingual():
    rows = load_validation_set()
    assert len(rows) >= 16
    langs = {r["lang"] for r in rows}
    assert {"ru", "ky"} <= langs
    labels = {r["human_pass"] for r in rows}
    assert labels == {True, False}  # both classes present -> kappa is meaningful
    # each language must carry both pass and fail labels for a per-language kappa
    for lang in ("ru", "ky"):
        sub = {r["human_pass"] for r in rows if r["lang"] == lang}
        assert sub == {True, False}, f"{lang} lacks label variance"


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

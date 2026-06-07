# tests/test_stats.py
from meta.stats import cohens_kappa, confusion_matrix, wilson_interval


def test_perfect_agreement_is_one():
    assert cohens_kappa([True, False, True], [True, False, True]) == 1.0


def test_total_disagreement_is_negative():
    # [T,F] vs [F,T]: systematic anti-agreement -> kappa=-1.0
    # (plan used [T,T] vs [F,F] which gives kappa=0 mathematically — corrected here)
    assert cohens_kappa([True, False], [False, True]) < 0


def test_all_same_label_returns_one():
    # pe == 1 edge case -> defined as 1.0
    assert cohens_kappa([True, True], [True, True]) == 1.0


def test_confusion_matrix_counts():
    tp, tn, fp, fn = confusion_matrix(
        human=[True, True, False, False],
        judge=[True, False, False, True])
    assert (tp, tn, fp, fn) == (1, 1, 1, 1)


def test_wilson_zero_n_is_degenerate():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson_interval(8, 10)  # p = 0.8
    assert 0.0 <= lo < 0.8 < hi <= 1.0


def test_wilson_is_clamped_to_unit_interval():
    lo, hi = wilson_interval(10, 10)  # p = 1.0
    assert lo >= 0.0 and hi <= 1.0


def test_wilson_small_n_is_wider_than_large_n():
    # same proportion (0.8), smaller n -> wider band (less confident)
    lo_small, hi_small = wilson_interval(4, 5)
    lo_big, hi_big = wilson_interval(80, 100)
    assert (hi_small - lo_small) > (hi_big - lo_big)

# meta/stats.py
"""Pure agreement statistics for judge validation. No dependencies."""
import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-by-default Wilson score interval for a binomial proportion.

    Used to put a confidence band on pass rates / agreement so a small-n RU vs KY
    delta reads as "directional" (overlapping bands) rather than a strong claim.
    Wilson is preferred over the normal approximation at small n and near 0/1.
    Returns (low, high) clamped to [0, 1]; n <= 0 -> (0.0, 0.0).
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion_matrix(human: list[bool], judge: list[bool]) -> tuple[int, int, int, int]:
    tp = sum(1 for h, j in zip(human, judge) if h and j)
    tn = sum(1 for h, j in zip(human, judge) if not h and not j)
    fp = sum(1 for h, j in zip(human, judge) if not h and j)
    fn = sum(1 for h, j in zip(human, judge) if h and not j)
    return tp, tn, fp, fn

"""Statistics shared across the evaluation and analysis scripts.

Kept in one place because the same interval was being computed in three files,
and three copies of a statistical function is three chances for them to drift
apart without anyone noticing.
"""

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because this study spends its
    whole life near 0% and 100%, where the normal approximation misbehaves: at
    0 successes it returns a zero-width interval, claiming certainty from a
    handful of trials. Wilson gives [0, 39%] for 0/6, which is the honest
    answer.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

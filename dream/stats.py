"""Small statistics shared by the learning gates (no numpy — floor tier).

The Wilson score interval is the PACE-lite acceptance gate
(learning-system.md §3): a change is accepted only when the 95% lower bound
of its success rate clears a threshold, so a couple of lucky samples can
never ratchet a noisy artifact into production.
"""

from __future__ import annotations

import math

_Z95 = 1.959963984540054   # standard normal quantile for a 95% interval


def wilson_lower_bound(successes: int, n: int, z: float = _Z95) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion.

    Returns 0.0 for n == 0 (no evidence => no confidence). This is the
    conservative estimate of the true success rate: with 8/10 successes it
    is ~0.49, not 0.8 — small samples are punished, which is the point.
    """
    if n <= 0:
        return 0.0
    successes = max(0, min(successes, n))
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def wilson_diff_lower_bound(a_succ: int, a_n: int, b_succ: int, b_n: int,
                            z: float = _Z95) -> float:
    """Lower bound on (rate_a - rate_b) via independent Wilson bounds.

    Conservative: uses a's lower bound minus b's upper bound, so a positive
    result means a really does beat b, not that noise happened to favor it.
    """
    if a_n <= 0 or b_n <= 0:
        return 0.0
    a_low = wilson_lower_bound(a_succ, a_n, z)
    b_up = 1.0 - wilson_lower_bound(b_n - b_succ, b_n, z)  # upper via symmetry
    return a_low - b_up

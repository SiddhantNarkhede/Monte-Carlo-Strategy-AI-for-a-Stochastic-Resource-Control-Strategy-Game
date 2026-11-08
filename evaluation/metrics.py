import math


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(results):
    """results: list of per-game dicts from tournament.play_one (agent-A view)."""
    n = len(results)
    wa = sum(r["a_win"] == 1 for r in results)
    wb = sum(r["a_win"] == -1 for r in results)
    dr = n - wa - wb
    rate = (wa + 0.5 * dr) / n
    # Wilson interval on the draw-as-half score (approximate; treats halves as successes)
    lo, hi = wilson(wa + 0.5 * dr, n)
    sa = [r["score_a"] for r in results]
    sb = [r["score_b"] for r in results]
    ta = sum(r["time_a"] for r in results)
    da = sum(r["dec_a"] for r in results)
    return dict(
        games=n, wins=wa, losses=wb, draws=dr, win_rate=rate, ci_lo=lo, ci_hi=hi,
        avg_score_a=sum(sa) / n, avg_score_b=sum(sb) / n,
        avg_margin=sum(x - y for x, y in zip(sa, sb)) / n,
        avg_turns=sum(r["turns"] for r in results) / n / 2.0,   # turns per player
        base_kill_rate=sum(r["base_kill"] for r in results) / n,
        ms_per_decision_a=1000.0 * ta / max(1, da),
    )

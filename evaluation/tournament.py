"""Head-to-head matches with seat alternation and round-robin tournaments."""
import itertools
import json
import time

from game import play_game
from .metrics import summarize


def play_one(a, b, env, seed, a_first):
    """Agent a vs agent b.  a occupies seat 0 when a_first else seat 1."""
    agents = (a, b) if a_first else (b, a)
    r = play_game(agents, env=env, seed=seed)
    sa = 0 if a_first else 1
    w = r["winner"]
    a_win = 0 if w == -1 else (1 if w == sa else -1)
    final = r["final"]
    return dict(a_win=a_win, score_a=r["scores"][sa], score_b=r["scores"][1 - sa],
                turns=r["turns"], time_a=r["time"][sa], dec_a=r["decisions"][sa],
                base_kill=int(min(final.bhp) <= 0))


def match(a, b, env="B", n=100, seed0=0, verbose=False):
    res = []
    for g in range(n):
        res.append(play_one(a, b, env, seed0 + g, g % 2 == 0))
        if verbose and (g + 1) % max(1, n // 5) == 0:
            s = summarize(res)
            print(f"   {a.name} vs {b.name} [{g+1}/{n}] wr={s['win_rate']:.2f}", flush=True)
    return summarize(res)


def round_robin(agents, env="B", n=100, seed0=0, verbose=True, out=None):
    """agents: list of Agent.  Returns {(nameA,nameB): summary}."""
    table = {}
    for a, b in itertools.combinations(agents, 2):
        t0 = time.time()
        table[(a.name, b.name)] = match(a, b, env, n, seed0)
        if verbose:
            s = table[(a.name, b.name)]
            print(f"[{env}] {a.name:>12} vs {b.name:<12} wr={s['win_rate']:.3f} "
                  f"[{s['ci_lo']:.2f},{s['ci_hi']:.2f}] n={n} ({time.time()-t0:.0f}s)", flush=True)
        if out:
            with open(out, "w") as f:
                json.dump({f"{k[0]}|{k[1]}": v for k, v in table.items()}, f, indent=1)
    return table

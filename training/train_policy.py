"""Policy extraction: turn the learned Q(s, category) into a stochastic policy
pi(category | s) = softmax(Q / tau) and pick the temperature tau.

tau -> 0 is the greedy policy (what MC-Control plays at test time); larger tau
spreads probability over near-optimal categories, which makes a better *prior*
and *rollout policy* for tree search (more diverse, less over-confident).

    python -m training.train_policy --env B
"""
import argparse
import json

from agents.monte_carlo import QTable
from agents.policy import PolicyAgent
from agents.greedy_agent import GreedyAgent
from evaluation.tournament import match


def sweep(q, env, taus=(0.01, 0.03, 0.08, 0.15, 0.3), games=150, verbose=True):
    rows = []
    for tau in taus:
        ag = PolicyAgent(q, tau=tau, name=f"pi(tau={tau})")
        r = match(ag, GreedyAgent(), env, games, seed0=50_000)
        rows.append(dict(tau=tau, win_rate=r["win_rate"], avg_margin=r["avg_margin"]))
        if verbose:
            print(f"[{env}] tau={tau:<5} vs Greedy: win {r['win_rate']:.3f}  margin {r['avg_margin']:+.1f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="B", choices=list("ABC"))
    ap.add_argument("--games", type=int, default=150)
    a = ap.parse_args()
    q = QTable.load(f"models/qtable_{a.env}.pkl")
    rows = sweep(q, a.env, games=a.games)
    best = max(rows, key=lambda r: (r["avg_margin"], r["win_rate"]))
    with open(f"models/policy_{a.env}.json", "w") as f:
        json.dump(dict(best_tau=best["tau"], sweep=rows), f, indent=1)
    print("best tau:", best["tau"])

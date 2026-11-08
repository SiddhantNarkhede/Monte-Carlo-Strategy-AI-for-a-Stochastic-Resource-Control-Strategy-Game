"""Train the tabular Monte-Carlo-control agent.

    python -m training.train_mc --env B --episodes 40000
"""
import argparse
import json
import os
import random
import time

from game import rules
from agents.monte_carlo import MCControlAgent, QTable
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from evaluation.tournament import match
from .self_play import run_episode, outcome_for


def train(env="B", episodes=40000, seed=0, eps_start=0.5, eps_end=0.05,
          alpha_start=0.02, alpha_end=0.002,
          mix=(0.2, 0.4, 0.4), eval_every=4000, eval_games=200, verbose=True):
    rules.seed(seed)
    drv = random.Random(seed + 12345)
    q = QTable()
    learner = MCControlAgent(q, epsilon=eps_start, learn=True, name="MC-Control")
    opps = [RandomAgent(), GreedyAgent()]
    curve = []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        frac = ep / episodes
        learner.epsilon = eps_start + (eps_end - eps_start) * frac
        learner.alpha_min = alpha_start + (alpha_end - alpha_start) * frac   # anneal step size
        u = drv.random()
        self_play = u >= mix[0] + mix[1]
        opp = opps[0] if u < mix[0] else opps[1]
        seat = drv.randint(0, 1)
        # training maps use seeds >= 10^6 so they never coincide with evaluation seeds (< 10^5)
        final, parts = run_episode(learner, opp, env, seat, self_play,
                                   map_seed=10 ** 6 + drv.randrange(10 ** 9))
        sc = rules.scores(final)
        for agent, sd in parts:
            agent.update(sc[sd] - sc[1 - sd], outcome_for(final, sd))
        if ep % eval_every == 0 or ep == episodes:
            ev = MCControlAgent(q, epsilon=0.0, learn=False, name="MC-Control")
            wr_g = match(ev, opps[1], env, eval_games, seed0=10_000)["win_rate"]
            wr_r = match(ev, opps[0], env, eval_games, seed0=10_000)["win_rate"]
            ns, nv = q.coverage()
            curve.append(dict(episode=ep, wr_vs_greedy=wr_g, wr_vs_random=wr_r,
                              states=ns, visits=nv, eps=learner.epsilon))
            if verbose:
                print(f"[{env}] ep {ep:6d}  vs Greedy {wr_g:.3f}  vs Random {wr_r:.3f}  "
                      f"states {ns}  ({time.time()-t0:.0f}s)", flush=True)
    return q, curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="B", choices=list("ABC"))
    ap.add_argument("--episodes", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--mix", type=float, nargs=3, default=(0.15, 0.45, 0.40),
                    help="P(random opp), P(greedy opp), P(self-play)")
    ap.add_argument("--fixed-map", action="store_true", help="train on the original single map")
    a = ap.parse_args()
    if a.fixed_map:
        import game.game as _g
        _g.RANDOM_MAPS = False
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    q, curve = train(a.env, a.episodes, a.seed, mix=tuple(a.mix), eval_every=a.eval_every)
    q.save(f"models/qtable_{a.env}.pkl")
    with open(f"results/training_curve_{a.env}.json", "w") as f:
        json.dump(curve, f, indent=1)

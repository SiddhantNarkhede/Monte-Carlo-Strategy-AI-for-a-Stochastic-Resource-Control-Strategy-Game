"""Run the full experimental study and save raw results as JSON in results/.

    python -m evaluation.run_experiments --section roundrobin --env B --games-cheap 300 --games-search 150
    python -m evaluation.run_experiments --section scaling
    python -m evaluation.run_experiments --section ablation
"""
import argparse
import itertools
import json
import os
import time

from agents.factory import build, load_q, best_tau
from evaluation.tournament import match

RES = "results"
CHEAP = {"Random", "Greedy", "MC-Control"}


def _dump(obj, name):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)


def roundrobin(env, games_cheap=300, games_search=150, sims=200, seed0=0):
    q, tau = load_q(env), best_tau(env)
    names = ["random", "greedy", "mc", "mcts", "rlmcts"]
    ags = [build(n, env, sims, q, tau) for n in names]
    out = {"env": env, "sims": sims, "tau": tau, "pairs": {}}
    for a, b in itertools.combinations(ags, 2):
        n = games_cheap if (a.name in CHEAP and b.name in CHEAP) else games_search
        t0 = time.time()
        r = match(a, b, env, n, seed0)
        # time/decision of the *second* agent as well (cheap: separate tiny match not needed)
        out["pairs"][f"{a.name}|{b.name}"] = r
        print(f"[{env}] {a.name:>10} vs {b.name:<10} n={n:<4} wr={r['win_rate']:.3f} "
              f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]  score {r['avg_score_a']:.0f}-{r['avg_score_b']:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        _dump(out, f"roundrobin_{env}.json")
    return out


def ablation(env="B", games=120, sims=200, seed0=1000):
    q, tau = load_q(env), best_tau(env)
    variants = ["mcts", "mcts+prior", "mcts+rollout", "rlmcts"]
    out = {"env": env, "sims": sims, "rows": {}}
    greedy = build("greedy", env)
    base = build("mcts", env, sims, q, tau)
    for v in variants:
        ag = build(v, env, sims, q, tau)
        t0 = time.time()
        rg = match(ag, greedy, env, games, seed0)
        rm = None if v == "mcts" else match(ag, base, env, games, seed0)
        out["rows"][ag.name] = {"vs_greedy": rg, "vs_mcts": rm}
        print(f"[abl {env}] {ag.name:<14} vs Greedy {rg['win_rate']:.3f}  "
              f"vs MCTS {'-' if rm is None else format(rm['win_rate'], '.3f')}  ({time.time()-t0:.0f}s)", flush=True)
        _dump(out, f"ablation_{env}.json")
    return out


def scaling(env="B", sims_list=(25, 50, 100, 200), games=60, seed0=2000):
    q, tau = load_q(env), best_tau(env)
    greedy = build("greedy", env)
    out = {"env": env, "rows": []}
    for sims in sims_list:
        m = build("mcts", env, sims, q, tau)
        r = build("rlmcts", env, sims, q, tau)
        t0 = time.time()
        row = dict(sims=sims,
                   mcts_vs_greedy=match(m, greedy, env, games, seed0),
                   rl_vs_greedy=match(r, greedy, env, games, seed0),
                   rl_vs_mcts=match(r, m, env, games, seed0))
        out["rows"].append(row)
        print(f"[scale {env}] sims={sims:<4} MCTS/Greedy {row['mcts_vs_greedy']['win_rate']:.3f}  "
              f"RL+MCTS/Greedy {row['rl_vs_greedy']['win_rate']:.3f}  "
              f"RL+MCTS/MCTS {row['rl_vs_mcts']['win_rate']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        _dump(out, f"scaling_{env}.json")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", "roundrobin", "ablation", "scaling"])
    ap.add_argument("--env", default=None, choices=list("ABC"))
    ap.add_argument("--games-cheap", type=int, default=300)
    ap.add_argument("--games-search", type=int, default=150)
    ap.add_argument("--games-ablation", type=int, default=120)
    ap.add_argument("--games-scaling", type=int, default=60)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--fixed-map", action="store_true")
    a = ap.parse_args(argv)
    if a.fixed_map:
        import game.game as _g
        _g.RANDOM_MAPS = False
    envs = [a.env] if a.env else list("BAC")
    if a.section in ("all", "roundrobin"):
        for e in envs:
            roundrobin(e, a.games_cheap, a.games_search, a.sims)
    if a.section in ("all", "ablation"):
        ablation(envs[0], a.games_ablation, a.sims)
    if a.section in ("all", "scaling"):
        scaling(envs[0], games=a.games_scaling)


if __name__ == "__main__":
    main()

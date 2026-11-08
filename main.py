"""Monte Carlo Strategy AI - command line entry point.

  python main.py train      --env B --episodes 300000       # Monte-Carlo control
  python main.py policy     --env B                         # choose softmax temperature
  python main.py tournament --env B --games 100             # round robin
  python main.py experiments                                # full study (all envs)
  python main.py watch --blue greedy --red rlmcts --env B   # text replay of one game
  python main.py gif   --blue greedy --red rlmcts --out results/demo.gif
  python main.py ui    --blue human  --red rlmcts           # needs pygame
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--env", default="B", choices=list("ABC"))
    t.add_argument("--episodes", type=int, default=300000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--eval-every", type=int, default=30000)

    p = sub.add_parser("policy")
    p.add_argument("--env", default="B", choices=list("ABC"))
    p.add_argument("--games", type=int, default=150)

    r = sub.add_parser("tournament")
    r.add_argument("--env", default="B", choices=list("ABC"))
    r.add_argument("--games", type=int, default=100)
    r.add_argument("--sims", type=int, default=200)

    sub.add_parser("experiments")

    for name in ("watch", "gif", "ui"):
        w = sub.add_parser(name)
        w.add_argument("--blue", default="greedy")
        w.add_argument("--red", default="rlmcts")
        w.add_argument("--env", default="B", choices=list("ABC"))
        w.add_argument("--sims", type=int, default=200)
        w.add_argument("--seed", type=int, default=0)
        w.add_argument("--out", default="results/demo.gif")

    for sp in sub.choices.values():
        sp.add_argument("--fixed-map", action="store_true",
                        help="use the original single map instead of a random map per game")
    a, rest = ap.parse_known_args()
    if a.fixed_map:
        import game.game as _g
        _g.RANDOM_MAPS = False
    if a.cmd == "train":
        import os
        import json
        from training.train_mc import train
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        q, curve = train(a.env, a.episodes, a.seed, eval_every=a.eval_every)
        q.save(f"models/qtable_{a.env}.pkl")
        json.dump(curve, open(f"results/training_curve_{a.env}.json", "w"), indent=1)
    elif a.cmd == "policy":
        sys.argv = ["train_policy", "--env", a.env, "--games", str(a.games)]
        import runpy
        runpy.run_module("training.train_policy", run_name="__main__")
    elif a.cmd == "tournament":
        from agents.factory import build, load_q
        from evaluation.tournament import round_robin
        q = load_q(a.env)
        ags = [build(n, a.env, a.sims, q) for n in ("random", "greedy", "mc", "mcts", "rlmcts")]
        round_robin(ags, a.env, a.games)
    elif a.cmd == "experiments":
        from evaluation import run_experiments
        run_experiments.main([])
    else:
        from agents.factory import build
        from game import play_game
        names = (a.blue, a.red)

        def mk(n):
            return None if n == "human" else build(n, a.env, a.sims)
        ags = [mk(n) for n in names]
        if a.cmd == "ui":
            from visualization.game_ui import run
            run(ags[0], ags[1], a.env)
        elif a.cmd == "gif":
            from visualization.make_gif import make_gif
            res = make_gif(tuple(ags), a.out, a.env, a.seed)
            print("wrote", a.out, "scores", res["scores"])
        else:
            from game.renderer import render_text
            res = play_game(tuple(ags), a.env, seed=a.seed, record=True)
            for st, last in res["frames"]:
                if last is not None and st.ap == 2 or st is res["frames"][-1][0]:
                    print(render_text(st), "\n")
            w = res["winner"]
            print("result:", "draw" if w == -1 else f"{names[w]} wins", "scores", res["scores"])


if __name__ == "__main__":
    main()

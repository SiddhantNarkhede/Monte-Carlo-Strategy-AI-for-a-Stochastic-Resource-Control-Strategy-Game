"""Record an AI-vs-AI game as an animated GIF (Pillow + imageio, no pygame needed)."""
import argparse
import imageio.v2 as imageio
import numpy as np

from game import play_game
from game.renderer import render_image


def make_gif(agents, path, env="B", seed=0, fps=3, labels=None):
    res = play_game(agents, env=env, seed=seed, record=True)
    labels = labels or (agents[0].name, agents[1].name)
    frames = []
    for st, last in res["frames"]:
        # only draw after the last action of each *action* (one frame per action)
        frames.append(np.array(render_image(st, last, title="Monte Carlo Strategy AI", labels=labels)))
    frames += [frames[-1]] * 6
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    return res


if __name__ == "__main__":
    from agents.greedy_agent import GreedyAgent
    from agents.random_agent import RandomAgent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/demo.gif")
    a = ap.parse_args()
    make_gif((GreedyAgent(), RandomAgent()), a.out)

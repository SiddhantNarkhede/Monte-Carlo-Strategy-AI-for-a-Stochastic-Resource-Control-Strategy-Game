"""Build agents by name (used by main.py and the experiment scripts)."""
import json
import os

from .random_agent import RandomAgent
from .greedy_agent import GreedyAgent
from .monte_carlo import MCControlAgent, QTable
from .mcts import MCTSAgent
from .rl_mcts import RLMCTSAgent

DEFAULT_TAU = 0.03


def load_q(env):
    path = f"models/qtable_{env}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run: python main.py train --env {env}")
    return QTable.load(path)


def best_tau(env):
    path = f"models/policy_{env}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["best_tau"]
    return DEFAULT_TAU


def build(name, env="B", sims=200, q=None, tau=None, c=0.25, c_puct=1.0, rollout_depth=12):
    name = name.lower()
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name == "mcts":
        return MCTSAgent(sims=sims, c=c, rollout_depth=rollout_depth)
    q = q if q is not None else load_q(env)
    tau = tau if tau is not None else best_tau(env)
    if name in ("mc", "mc-control"):
        return MCControlAgent(q, epsilon=0.0, name="MC-Control")
    if name in ("rlmcts", "rl+mcts"):
        return RLMCTSAgent(q, sims=sims, c=c, c_puct=c_puct, tau=tau, rollout_depth=rollout_depth)
    if name == "mcts+prior":
        return RLMCTSAgent(q, sims=sims, c=c, c_puct=c_puct, tau=tau, rollout_depth=rollout_depth,
                           use_prior=True, use_rollout=False, name="MCTS+prior")
    if name == "mcts+rollout":
        return RLMCTSAgent(q, sims=sims, c=c, c_puct=c_puct, tau=tau, rollout_depth=rollout_depth,
                           use_prior=False, use_rollout=True, name="MCTS+rollout")
    raise ValueError(f"unknown agent {name!r}")

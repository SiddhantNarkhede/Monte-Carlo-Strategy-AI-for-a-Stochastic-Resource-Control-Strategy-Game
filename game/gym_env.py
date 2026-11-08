"""Optional Gymnasium environment.

The learning agent controls seat 0 and chooses one of the 9 strategic categories
(see game/actions.py); a built-in opponent (default: Greedy) plays seat 1.
Illegal categories are reported in `info["action_mask"]`; an illegal choice is
mapped to PASS-like fallback (a random legal action) so training never crashes.

    import gymnasium as gym
    from game.gym_env import StrategyEnv
    env = StrategyEnv(env_id="B")
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                  # pragma: no cover
    gym = None
    spaces = None

from . import rules
from .game import new_state, observe
from .actions import NCAT
from .state import MAX_TURNS, SCORE_SCALE


def _features(s, p):
    me, en = s.units[p], s.units[1 - p]
    sc = rules.scores(s)
    return np.array([
        s.res[p] / 20.0, (s.res[1 - p] if s.res[1 - p] >= 0 else 0) / 20.0,
        len(me) / 4.0, len(en) / 4.0,
        s.post.count(p) / 5.0, s.post.count(1 - p) / 5.0,
        s.bhp[p] / 5.0, s.bhp[1 - p] / 5.0,
        (s.turn // 2) / MAX_TURNS, s.ap / 2.0,
        (sc[p] - sc[1 - p]) / (3 * SCORE_SCALE),
    ] + [x / 9.0 for x in s.stock], dtype=np.float32)


_Base = gym.Env if gym is not None else object


class StrategyEnv(_Base):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, env_id="B", opponent=None, seat=0):
        if gym is None:
            raise ImportError("pip install gymnasium")
        self.env_id = env_id
        self.seat = seat
        if opponent is None:
            from agents.greedy_agent import GreedyAgent
            opponent = GreedyAgent()
        self.opp = opponent
        self.action_space = spaces.Discrete(NCAT)
        self.observation_space = spaces.Box(-1.0, 3.0, shape=(16,), dtype=np.float32)
        self.s = None

    # -- helpers
    def _advance_opponent(self):
        while not self.s.over and self.s.player != self.seat:
            legal = rules.legal_actions(self.s)
            a = legal[0] if len(legal) == 1 else self.opp.act(observe(self.s, 1 - self.seat), legal)
            rules.apply_action(self.s, a)

    def _info(self):
        legal = rules.legal_actions(self.s) if not self.s.over else []
        cats = rules.categorize(self.s, legal) if legal else []
        mask = np.zeros(NCAT, dtype=bool)
        for c in cats:
            mask[c] = True
        return {"action_mask": mask, "scores": rules.scores(self.s)}

    def _diff(self):
        sc = rules.scores(self.s)
        return (sc[self.seat] - sc[1 - self.seat]) / SCORE_SCALE

    # -- gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            rules.seed(seed)
        self.s = new_state(self.env_id, map_seed=seed)
        self._advance_opponent()
        self._last = self._diff()
        return _features(observe(self.s, self.seat), self.seat), self._info()

    def step(self, action):
        s = self.s
        legal = rules.legal_actions(s)
        cats = rules.categorize(s, legal)
        from agents.monte_carlo import resolve
        if int(action) in cats:
            a = resolve(s, legal, cats, int(action))
        else:
            a = legal[int(rules.rng.random() * len(legal))]
        rules.apply_action(s, a)
        self._advance_opponent()
        d = self._diff()
        reward = d - self._last
        self._last = d
        terminated = bool(s.over)
        if terminated:
            reward += 0.0 if s.winner == -1 else (1.0 if s.winner == self.seat else -1.0)
        return (_features(observe(s, self.seat), self.seat), float(reward), terminated, False,
                self._info())

    def render(self):
        from .renderer import render_text
        return render_text(self.s)

"""A stochastic policy pi(category | abstract state) derived from a learned Q-table.

pi = softmax(Q(s, .) / tau) restricted to the categories that are legal in the
actual state.  States with too little experience fall back to uniform play, and a
small epsilon of uniform concrete actions keeps rollouts diverse.
"""
import math
from game import rules
from game.actions import NCAT
from .abstraction import state_key
from .monte_carlo import resolve


class LearnedPolicy:
    def __init__(self, qtable, tau=0.15, eps=0.1, min_visits=3):
        self.q = qtable
        self.tau = tau
        self.eps = eps
        self.min_visits = min_visits

    def cat_probs(self, s, avail):
        """dict category -> probability over the legal categories `avail`
        (None if the state is not known well enough: caller should use uniform)."""
        key = state_key(s, s.player)
        row = self.q.Q.get(key)
        if row is None or sum(self.q.N[key]) < self.min_visits:
            return None
        m = max(row[c] for c in avail)
        w = {c: math.exp((row[c] - m) / self.tau) for c in avail}
        z = sum(w.values())
        return {c: x / z for c, x in w.items()}

    def sample_action(self, s, legal):
        r = rules.rng
        if r.random() < self.eps:
            return legal[int(r.random() * len(legal))]
        cats = rules.categorize(s, legal)
        pr = self.cat_probs(s, set(cats))
        if pr is None:
            return legal[int(r.random() * len(legal))]
        u = r.random()
        acc = 0.0
        chosen = None
        for c, p in pr.items():
            acc += p
            chosen = c
            if u <= acc:
                break
        return resolve(s, legal, cats, chosen)

    def greedy_action(self, s, legal):
        cats = rules.categorize(s, legal)
        pr = self.cat_probs(s, set(cats))
        if pr is None:
            return legal[int(rules.rng.random() * len(legal))]
        return resolve(s, legal, cats, max(pr, key=pr.get))


from .base import Agent  # noqa: E402


class PolicyAgent(Agent):
    """Plays the stochastic learned policy pi(a|s) directly (no search)."""
    name = "MC-Policy"

    def __init__(self, qtable, tau=0.15, eps=0.0, name=None):
        self.pol = LearnedPolicy(qtable, tau=tau, eps=eps)
        if name:
            self.name = name

    def act(self, s, legal):
        return self.pol.sample_action(s, legal)

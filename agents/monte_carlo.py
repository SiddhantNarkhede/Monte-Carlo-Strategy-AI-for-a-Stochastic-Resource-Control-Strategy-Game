"""Tabular Monte-Carlo control over strategic categories.

Q(s,a) is learned for abstract state s and *strategic category* a (9 of them,
see game/actions.py).  A category is resolved to a concrete action (which
unit / direction) by `resolve`.

Learning (every-visit Monte-Carlo control, epsilon-greedy):
    r_{t+1} = (D_{t+1} - D_t) / SCORE_SCALE   (+ outcome_w * (+-1) on the last step; default 0)
    G_t     = sum_k gamma^k r_{t+k+1}
    Q(s_t,a_t) <- Q(s_t,a_t) + alpha * (G_t - Q(s_t,a_t)),  alpha = max(1/N, alpha_min)
where D_t is the score difference (own - enemy) when the agent decides.
"""
import pickle
from game import rules
from game.actions import NCAT, C_MOVE_ECON, C_MOVE_AGGR
from game.state import NODE_IDX, SCORE_SCALE
from .abstraction import state_key
from .base import Agent


class QTable:
    def __init__(self):
        self.Q = {}
        self.N = {}

    def row(self, key):
        r = self.Q.get(key)
        if r is None:
            r = self.Q[key] = [0.0] * NCAT
            self.N[key] = [0] * NCAT
        return r

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "N": self.N}, f)

    @staticmethod
    def load(path):
        t = QTable()
        with open(path, "rb") as f:
            d = pickle.load(f)
        t.Q, t.N = d["Q"], d["N"]
        return t

    def coverage(self):
        return len(self.Q), sum(sum(v) for v in self.N.values())


def resolve(s, legal, cats, cat):
    """Pick a concrete action inside strategic category `cat`.  For economic /
    aggressive moves, prefer units that are not already standing on a node so
    that workers are not pulled off a node they are harvesting."""
    r = rules.rng
    members = [a for a, c in zip(legal, cats) if c == cat]
    if len(members) > 1 and cat in (C_MOVE_ECON, C_MOVE_AGGR):
        me = s.units[s.player]
        idle = [a for a in members if me[a[1]][0] not in NODE_IDX]
        if idle:
            members = idle
    return members[int(r.random() * len(members))]


class MCControlAgent(Agent):
    name = "MC-Control"

    def __init__(self, qtable=None, epsilon=0.0, alpha_min=0.01, gamma=0.99,
                 learn=False, outcome_w=0.0, name=None):
        self.q = qtable if qtable is not None else QTable()
        self.epsilon = epsilon
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.learn = learn
        self.outcome_w = outcome_w      # weight of the +-1 terminal win/loss reward
        self.trace = []     # (state_key, category)
        self.diffs = []     # score difference (own - enemy) at each decision
        if name:
            self.name = name

    def reset(self):
        self.trace = []
        self.diffs = []

    def clone_sharing_q(self):
        return MCControlAgent(self.q, self.epsilon, self.alpha_min, self.gamma,
                              self.learn, self.outcome_w, self.name)

    def act(self, s, legal):
        p = s.player
        cats = rules.categorize(s, legal)
        key = state_key(s, p)
        r = rules.rng
        if r.random() < self.epsilon:
            cat = cats[int(r.random() * len(cats))]     # uniform over concrete actions
        else:
            row = self.q.Q.get(key) or [0.0] * NCAT
            avail = set(cats)
            best = max(row[c] for c in avail)
            cand = [c for c in avail if row[c] >= best - 1e-12]
            cat = cand[0] if len(cand) == 1 else cand[int(r.random() * len(cand))]
        if self.learn:
            self.trace.append((key, cat))
        return resolve(s, legal, cats, cat)

    def update(self, final_diff, outcome):
        """Every-visit MC update.  final_diff: final score difference (own-enemy);
        outcome in {-1,0,+1}: result of the game for this agent."""
        n = len(self.trace)
        if n == 0:
            return
        ds = self.diffs + [final_diff]
        g = 0.0
        q = self.q
        for t in range(n - 1, -1, -1):
            rew = (ds[t + 1] - ds[t]) / SCORE_SCALE
            if t == n - 1:
                rew += self.outcome_w * outcome
            g = rew + self.gamma * g
            key, cat = self.trace[t]
            row = q.row(key)
            cnt = q.N[key]
            cnt[cat] += 1
            alpha = max(1.0 / cnt[cat], self.alpha_min)
            row[cat] += alpha * (g - row[cat])
        self.trace = []
        self.diffs = []

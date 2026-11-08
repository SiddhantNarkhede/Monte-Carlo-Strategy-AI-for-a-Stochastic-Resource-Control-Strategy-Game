"""Monte Carlo Tree Search (open-loop, two-player zero-sum, AP-aware).

One tree edge = one action (a turn has 2 action points, so the same player can
move twice in a row; node values are stored from the point of view of the player
who *made* the action leading to the node).

Variants
  * plain MCTS      : UCT selection, uniform-random expansion, uniform-random rollouts
  * RL-guided MCTS  : PUCT selection with a learned prior pi(a|s) from the Monte-Carlo
                      control agent, prior-weighted expansion, learned-policy rollouts.
                      `use_prior` / `use_rollout` can be switched independently
                      (used for the ablation study).

Rollouts are truncated after `rollout_depth` actions and evaluated with the same
score-difference value function used elsewhere (tanh(diff/scale)); a finished game
is evaluated exactly.  Because transitions are stochastic the search is open-loop:
every simulation replays the tree actions on a fresh copy of the state.

In hidden-information games the opponent's resources are unknown; every simulation
samples ("determinizes") them uniformly from [0, 10].
"""
import math
from game import rules
from .base import Agent


class Node:
    __slots__ = ("n", "w", "kids", "prior")

    def __init__(self):
        self.n = 0
        self.w = 0.0
        self.kids = {}
        self.prior = None


class MCTSAgent(Agent):
    name = "MCTS"

    def __init__(self, sims=100, c=0.35, rollout_depth=12, tree_depth=10, policy=None,
                 use_prior=False, use_rollout=False, c_puct=1.0, rollout_policy=None, name=None):
        self.sims = sims
        self.c = c
        self.rollout_depth = rollout_depth
        self.tree_depth = tree_depth
        self.policy = policy                          # supplies the PUCT prior
        self.rollout_policy = rollout_policy or policy
        self.use_prior = use_prior and policy is not None
        self.use_rollout = use_rollout and self.rollout_policy is not None
        self.c_puct = c_puct
        self.last_root = None
        if name:
            self.name = name

    # ---------------------------------------------------------------- search
    def _priors(self, s, acts):
        cats = rules.categorize(s, acts)
        pr = self.policy.cat_probs(s, set(cats))
        if pr is None:
            u = 1.0 / len(acts)
            return {a: u for a in acts}
        cnt = {}
        for c in cats:
            cnt[c] = cnt.get(c, 0) + 1
        return {a: pr[c] / cnt[c] for a, c in zip(acts, cats)}

    def act(self, obs, legal):
        rng = rules.rng
        rp = obs.player
        hidden = obs.res[1 - rp] < 0
        root = Node()
        legal_actions = rules.legal_actions
        apply_action = rules.apply_action
        use_prior = self.use_prior
        use_roll = self.use_rollout
        pol = self.rollout_policy
        c_uct, c_puct = self.c, self.c_puct
        rdepth, tdepth = self.rollout_depth, self.tree_depth
        for _ in range(self.sims):
            s = obs.copy()
            if hidden:
                s.res[1 - rp] = rng.randint(0, 10)
            node = root
            path = []                       # (child, mover)
            depth = 0
            # ---- selection / expansion
            while not s.over and depth < tdepth:
                acts = legal if (depth == 0) else legal_actions(s)
                kids = node.kids
                if use_prior and node.prior is None:
                    node.prior = self._priors(s, acts)
                untried = [a for a in acts if a not in kids]
                mover = s.player
                if use_prior:
                    a = self._puct(node, acts, kids, c_puct)
                elif untried:
                    a = untried[int(rng.random() * len(untried))]
                else:
                    a = self._uct(node, acts, kids, c_uct)
                child = kids.get(a)
                if child is None:
                    child = kids[a] = Node()
                    path.append((child, mover))
                    apply_action(s, a)
                    depth += 1
                    break                    # expanded one node -> rollout
                path.append((child, mover))
                apply_action(s, a)
                node = child
                depth += 1
            # ---- rollout
            steps = 0
            while not s.over and steps < rdepth:
                acts = legal_actions(s)
                if len(acts) == 1:
                    a = acts[0]
                elif use_roll:
                    a = pol.sample_action(s, acts)
                else:
                    a = acts[int(rng.random() * len(acts))]
                apply_action(s, a)
                steps += 1
            v = rules.value(s, rp)
            # ---- backpropagation
            root.n += 1
            for child, mover in path:
                child.n += 1
                child.w += v if mover == rp else -v
        self.last_root = root
        # robust child: most visited root action
        best, bn = None, -1
        for a in legal:
            k = root.kids.get(a)
            if k is not None and (k.n > bn or (k.n == bn and k.w > root.kids[best].w)):
                best, bn = a, k.n
        return best if best is not None else legal[0]

    @staticmethod
    def _uct(node, acts, kids, c):
        lnN = math.log(node.n + 1)
        best, bs = None, -1e18
        for a in acts:
            k = kids[a]
            sc = k.w / k.n + c * math.sqrt(lnN / k.n)
            if sc > bs:
                best, bs = a, sc
        return best

    @staticmethod
    def _puct(node, acts, kids, c):
        sq = math.sqrt(node.n + 1)
        pri = node.prior
        u0 = 1.0 / len(acts)
        best, bs = None, -1e18
        for a in acts:
            k = kids.get(a)
            if k is None:
                q, n = 0.0, 0
            else:
                q, n = k.w / k.n, k.n
            sc = q + c * pri.get(a, u0) * sq / (1 + n)
            if sc > bs:
                best, bs = a, sc
        return best

    # ---------------------------------------------------------- diagnostics
    def root_stats(self):
        """List of (action, visits, mean value) for the last search."""
        if not self.last_root:
            return []
        return sorted(((a, k.n, k.w / max(1, k.n)) for a, k in self.last_root.kids.items()),
                      key=lambda x: -x[1])

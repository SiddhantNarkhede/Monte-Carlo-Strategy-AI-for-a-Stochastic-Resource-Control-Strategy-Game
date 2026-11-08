"""RL + MCTS hybrid: the Monte-Carlo-control policy guides the tree search, both as a
PUCT prior over actions (temperature tau_prior) and as the rollout policy
(temperature tau_rollout, sharper)."""
from .mcts import MCTSAgent
from .policy import LearnedPolicy


class RLMCTSAgent(MCTSAgent):
    name = "RL+MCTS"

    def __init__(self, qtable, sims=100, c=0.25, c_puct=1.0, tau=0.03, tau_prior=None,
                 rollout_eps=0.05, rollout_depth=12, use_prior=True, use_rollout=True, name=None):
        prior = LearnedPolicy(qtable, tau=tau_prior if tau_prior is not None else max(tau, 0.1), eps=0.0)
        roll = LearnedPolicy(qtable, tau=tau, eps=rollout_eps)
        super().__init__(sims=sims, c=c, rollout_depth=rollout_depth, policy=prior,
                         rollout_policy=roll, use_prior=use_prior, use_rollout=use_rollout,
                         c_puct=c_puct, name=name)

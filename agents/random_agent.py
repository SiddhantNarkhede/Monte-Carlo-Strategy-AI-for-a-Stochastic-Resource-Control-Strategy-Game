from game import rules
from .base import Agent


class RandomAgent(Agent):
    name = "Random"

    def act(self, state, legal):
        return legal[int(rules.rng.random() * len(legal))]

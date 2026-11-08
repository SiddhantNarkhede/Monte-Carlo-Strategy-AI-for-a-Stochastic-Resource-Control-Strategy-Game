class Agent:
    name = "agent"

    def reset(self):
        pass

    def act(self, state, legal):
        """state: observation (State); legal: list of legal concrete actions."""
        raise NotImplementedError

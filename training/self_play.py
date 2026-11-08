"""Episode generation for Monte-Carlo control.

Opponent mixture used for training (so the learner sees weak, rule-based and
adaptive opponents): random / greedy / self-play (both seats act from, and
learn into, the same Q-table)."""
from game import new_state, observe, rules


def run_episode(learner, opponent, env, learner_seat, self_play=False, map_seed=None):
    """Plays one game and returns (final_state, [(agent, seat), ...] learners).
    Score differences for the reward signal are taken from the *true* state
    (the environment's reward), while the agents only see observations."""
    agents = [None, None]
    agents[learner_seat] = learner
    agents[1 - learner_seat] = learner.clone_sharing_q() if self_play else opponent
    learners = [(agents[learner_seat], learner_seat)]
    if self_play:
        learners.append((agents[1 - learner_seat], 1 - learner_seat))
    for a in agents:
        a.reset()
    s = new_state(env, map_seed=map_seed)
    while not s.over:
        p = s.player
        legal = rules.legal_actions(s)
        ag = agents[p]
        if len(legal) == 1:
            a = legal[0]
        else:
            if getattr(ag, "learn", False):
                sc = rules.scores(s)
                ag.diffs.append(sc[p] - sc[1 - p])
            a = ag.act(observe(s, p), legal)
        rules.apply_action(s, a)
    return s, learners


def outcome_for(final, seat):
    if final.winner == -1:
        return 0.0
    return 1.0 if final.winner == seat else -1.0

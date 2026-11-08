"""High level Game wrapper: environments A/B/C, observations, playing games."""
import time
from .state import State, set_map
from . import rules

ENVS = {
    "A": dict(stoch=False, hidden=False, desc="Deterministic"),
    "B": dict(stoch=True, hidden=False, desc="Stochastic (random yields & combat)"),
    "C": dict(stoch=True, hidden=True, desc="Stochastic + hidden enemy resources"),
}


RANDOM_MAPS = True      # False -> always use the original hand-made map (CLI: --fixed-map)


def new_state(env="B", map_seed=None):
    """New game.  With RANDOM_MAPS the map is generated from map_seed (a fresh
    random map when map_seed is None); otherwise the fixed default map is used."""
    if RANDOM_MAPS:
        import random
        set_map(map_seed if map_seed is not None else random.randrange(10 ** 9))
    else:
        set_map(None)
    cfg = ENVS[env]
    return State(stoch=cfg["stoch"], hidden=cfg["hidden"])


def observe(s, persp):
    """What player `persp` is allowed to see.  In env C the opponent's
    resources are hidden (replaced by -1); everything else is public."""
    if not s.hidden:
        return s
    o = s.copy()
    o.res[1 - persp] = -1
    return o


def play_game(agents, env="B", seed=None, record=False, first=0):
    """Play one full game.  agents = (agent_for_seat0, agent_for_seat1).
    Returns dict with winner, scores, turns, per-agent time/decision counts."""
    if seed is not None:
        rules.seed(seed)
    s = new_state(env, map_seed=seed)
    s.player = first
    if first != 0:
        pass
    for a in agents:
        a.reset()
    t_used = [0.0, 0.0]
    n_dec = [0, 0]
    frames = []
    if record:
        frames.append((s.copy(), None))
    while not s.over:
        p = s.player
        legal = rules.legal_actions(s)
        t0 = time.perf_counter()
        a = agents[p].act(observe(s, p), legal) if len(legal) > 1 else legal[0]
        t_used[p] += time.perf_counter() - t0
        n_dec[p] += 1
        rules.apply_action(s, a)
        if record:
            frames.append((s.copy(), (p, a)))
    sc = rules.scores(s)
    return dict(winner=s.winner, scores=sc, turns=s.turn, time=t_used, decisions=n_dec,
                frames=frames, final=s)

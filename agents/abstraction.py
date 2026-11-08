"""State abstraction for tabular Monte-Carlo control.

The raw state space is astronomically large, so we discretise it into
  own resources (4) x enemy resources (3, incl. 'unknown') x own units (4)
  x enemy units (4) x own outposts (3) x enemy outposts (3)
  x distance to enemy (4) x game phase (3)
  x worker-on-node flag (3: none / on a node I don't own / only on my own outposts)
= 62,208 abstract states.
"""
from game.state import NODE_IDX, BASES, DIST, MAX_TURNS

N_STATES = 4 * 3 * 4 * 4 * 3 * 3 * 4 * 3 * 3


def state_key(s, p):
    me = s.units[p]
    en = s.units[1 - p]
    r = s.res[p]
    rb = 0 if r < 3 else (1 if r < 4 else (2 if r < 8 else 3))
    er = s.res[1 - p]
    # 2 = hidden / unknown.  s.hidden also masks it inside MCTS, where the hidden
    # value has been *sampled* (determinised) and must not leak into the policy lookup.
    erb = 2 if (er < 0 or s.hidden) else (0 if er < 4 else 1)
    nu = len(me)
    ne = len(en)
    nu = 3 if nu > 3 else nu
    ne = 3 if ne > 3 else ne
    post = s.post
    op = post.count(p)
    ep = post.count(1 - p)
    op = 2 if op > 2 else op
    ep = 2 if ep > 2 else ep
    # distance from own units to nearest enemy unit / base
    tg = [u[0] for u in en]
    tg.append(BASES[1 - p])
    dm = 99
    onnode = 0
    for u in me:
        row = DIST[u[0]]
        for t in tg:
            if row[t] < dm:
                dm = row[t]
        n = NODE_IDX.get(u[0])
        if n is not None:
            if post[n] != p:
                onnode = 1
            elif onnode == 0 and s.stock[n] > 0:
                onnode = 2
    db = 3 if not me else (0 if dm <= 1 else (1 if dm <= 3 else 2))
    ph = (s.turn // 2) * 3 // MAX_TURNS
    k = rb
    k = k * 3 + erb
    k = k * 4 + nu
    k = k * 4 + ne
    k = k * 3 + op
    k = k * 3 + ep
    k = k * 4 + db
    k = k * 3 + ph
    k = k * 3 + onnode
    return k

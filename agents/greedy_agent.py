"""Rule-based / greedy opponent.

Priorities (the first rule that fires wins):
  1. attack when the fight is favourable (or raze an enemy outpost)
  2. build an outpost on the node we stand on (+territory, +income)
  3. recruit a unit while we have fewer than 3
  4. collect from a node we stand on  (highest immediate reward)
  5. walk the unit that is not yet on a node toward the nearest uncontrolled
     node, otherwise toward the enemy
  6. defend / pass
"""
from game import rules
from game.actions import MOVE, COLLECT, OUTPOST, RECRUIT, ATTACK, DEFEND, PASS
from game.state import NEIGH_DIR, NODES, BASES, DIST, NODE_IDX
from .base import Agent


class GreedyAgent(Agent):
    name = "Greedy"

    def __init__(self, max_units=3):
        self.max_units = max_units

    def act(self, s, legal):
        p = s.player
        me, en = s.units[p], s.units[1 - p]
        by = {}
        for a in legal:
            by.setdefault(a[0], []).append(a)
        # 1. attacks
        best, best_adv = None, 0.5
        for a in by.get(ATTACK, ()):
            u = me[a[1]]
            if a[2] == 5:
                adv = 3.0
            elif a[2] == 4:
                adv = u[1] - 3 + 1.0
            else:
                tc = NEIGH_DIR[u[0]][a[2]]
                e = next(x for x in en if x[0] == tc)
                adv = u[1] - rules._unit_defence(s, 1 - p, e) + (1.0 if e[1] == 1 else 0.0)
            if adv > best_adv:
                best, best_adv = a, adv
        if best:
            return best
        # 2. outpost
        if OUTPOST in by:
            return by[OUTPOST][0]
        # 3. recruit
        if RECRUIT in by and len(me) < self.max_units:
            return by[RECRUIT][0]
        # 4./5. collect only if every unit already stands on a node; else walk first
        on_node = {a[1] for a in by.get(COLLECT, ())}
        idle = [i for i in range(len(me)) if i not in on_node]
        mv = by.get(MOVE, ())
        if idle and mv:
            m = self._best_move(s, me, en, p, mv, idle)
            if m:
                return m
        if COLLECT in by:
            return by[COLLECT][0]
        if mv:
            m = self._best_move(s, me, en, p, mv, range(len(me)))
            if m:
                return m
        if DEFEND in by:
            return by[DEFEND][0]
        return by[PASS][0]

    @staticmethod
    def _best_move(s, me, en, p, moves, idle):
        mycells = {u[0] for u in me}
        free = [NODES[n] for n in range(len(NODES)) if s.post[n] != p]
        rich = [NODES[n] for n in range(len(NODES)) if s.post[n] == p and s.stock[n] > 0]
        tg = ([x for x in free if x not in mycells] or [x for x in rich if x not in mycells]
              or [BASES[1 - p]])
        best, bd = None, 0
        for a in moves:
            if a[1] not in idle:
                continue
            c = me[a[1]][0]
            c2 = NEIGH_DIR[c][a[2]]
            d0 = min(DIST[c][x] for x in tg)
            d1 = min(DIST[c2][x] for x in tg)
            gain = d0 - d1
            if gain > bd:
                best, bd = a, gain
        return best

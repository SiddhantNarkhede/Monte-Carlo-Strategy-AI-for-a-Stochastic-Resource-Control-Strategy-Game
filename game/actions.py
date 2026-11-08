"""Action encoding.

A concrete action is a tuple (type, unit_index, param):

  MOVE     (0, u, dir)      move unit u one step (dir 0..3)
  COLLECT  (1, u, 0)        unit standing on a resource node harvests it
  OUTPOST  (2, u, 0)        unit on a free node builds an outpost (territory)
  RECRUIT  (3, -1, 0)       build a new unit next to the base
  ATTACK   (4, u, target)   target 0..3 = enemy unit in that direction,
                            4 = adjacent enemy base, 5 = raze enemy outpost
                            on the unit's own cell
  DEFEND   (5, u, 0)        unit gets a defensive bonus until your next turn
  PASS     (6, -1, 0)       end the turn (forfeit remaining action points)

For tabular RL the concrete actions are folded into 9 *strategic categories*
(the "Attack / Collect / Build / Defend / Move" abstraction of the project
brief; Move is split by intent and Build by what is built).
"""
MOVE, COLLECT, OUTPOST, RECRUIT, ATTACK, DEFEND, PASS = range(7)
TYPE_NAMES = ["MOVE", "COLLECT", "OUTPOST", "RECRUIT", "ATTACK", "DEFEND", "PASS"]

# strategic categories
(C_MOVE_ECON, C_MOVE_AGGR, C_MOVE_BACK, C_COLLECT, C_OUTPOST,
 C_RECRUIT, C_ATTACK, C_DEFEND, C_PASS) = range(9)
NCAT = 9
CAT_NAMES = ["MOVE_ECON", "MOVE_AGGR", "MOVE_BACK", "COLLECT", "BUILD_OUTPOST",
             "BUILD_UNIT", "ATTACK", "DEFEND", "PASS"]

_FIXED_CAT = {COLLECT: C_COLLECT, OUTPOST: C_OUTPOST, RECRUIT: C_RECRUIT,
              ATTACK: C_ATTACK, DEFEND: C_DEFEND, PASS: C_PASS}


def action_str(a):
    from .state import DIR_NAMES
    type, unit_index, parameter = a
    if type == MOVE:
        return f"MOVE u{unit_index} {DIR_NAMES[parameter]}"
    if type == ATTACK:
        return f"ATTACK u{unit_index} " + (["up", "right", "down", "left", "base", "raze outpost"][parameter])
    if type in (RECRUIT, PASS):
        return TYPE_NAMES[type]
    return f"{TYPE_NAMES[type]} u{unit_index}"

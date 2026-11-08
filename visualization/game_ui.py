"""Pygame front-end:  watch two agents play, or play a human against an AI.

    python main.py ui --blue human --red rlmcts --env B
    python main.py ui --blue greedy --red mcts            # watch AI vs AI

Controls (human turn)
  left click   select one of your units (yellow ring)
  W A S D      move the selected unit       X  attack: press X, then W/A/S/D (or B = base)
  C            collect          O  build outpost       R  recruit unit
  F            fortify (defend) E  / SPACE  end turn
  Z            raze enemy outpost on the selected unit's cell
  N            new game          ESC  quit

NOTE: pygame is not installed in the sandbox this project was developed in, so this
file could not be executed there.  The game logic it calls (rules / agents) is
covered by tests; the rendering/event code follows the standard pygame API.
"""
import sys
import time

from game import rules, new_state, observe
from game.actions import MOVE, COLLECT, OUTPOST, RECRUIT, ATTACK, DEFEND, PASS, action_str
from game.state import SIZE, TREES, NODES, BASES

CELL, PANEL = 80, 300
COL = {"bg": (245, 242, 232), "grid": (200, 195, 180), "tree": (60, 130, 70),
       "node": (40, 190, 200), "blue": (52, 120, 220), "red": (214, 62, 62)}
KEYDIR = {"w": 0, "d": 1, "s": 2, "a": 3}


def run(agent_blue, agent_red, env="B", fps=30, ai_delay=0.35):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((SIZE * CELL + PANEL, SIZE * CELL))
    pygame.display.set_caption("Monte Carlo Strategy AI")
    font = pygame.font.SysFont("arial", 18)
    big = pygame.font.SysFont("arial", 26, bold=True)
    clock = pygame.time.Clock()
    agents = [agent_blue, agent_red]          # None = human
    s = new_state(env)
    sel, attack_mode, last_msg, last_t = None, False, "", time.time()

    def legal_for(t, ui, par=0):
        return [a for a in rules.legal_actions(s) if a[0] == t and (a[1] == ui or t in (RECRUIT, PASS)) and a[2] == par]

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = pygame.key.name(ev.key)
                if k == "escape":
                    running = False
                elif k == "n":
                    s = new_state(env); sel = None
                elif not s.over and agents[s.player] is None:
                    act = None
                    if k in ("e", "space"):
                        act = (PASS, -1, 0)
                    elif k == "r":
                        act = (RECRUIT, -1, 0)
                    elif sel is not None and sel < len(s.units[s.player]):
                        if k == "x":
                            attack_mode = True
                        elif k in KEYDIR:
                            act = (ATTACK if attack_mode else MOVE, sel, KEYDIR[k]); attack_mode = False
                        elif k == "b" and attack_mode:
                            act = (ATTACK, sel, 4); attack_mode = False
                        elif k == "c":
                            act = (COLLECT, sel, 0)
                        elif k == "o":
                            act = (OUTPOST, sel, 0)
                        elif k == "f":
                            act = (DEFEND, sel, 0)
                        elif k == "z":
                            act = (ATTACK, sel, 5)
                    if act is not None:
                        if act in rules.legal_actions(s):
                            rules.apply_action(s, act); last_msg = f"You: {action_str(act)}"
                        else:
                            last_msg = "illegal action"
            elif ev.type == pygame.MOUSEBUTTONDOWN and not s.over and agents[s.player] is None:
                c, r = ev.pos[0] // CELL, ev.pos[1] // CELL
                for i, u in enumerate(s.units[s.player]):
                    if u[0] == r * SIZE + c:
                        sel = i
        # AI turn
        if not s.over and agents[s.player] is not None and time.time() - last_t > ai_delay:
            mover = s.player
            legal = rules.legal_actions(s)
            a = legal[0] if len(legal) == 1 else agents[mover].act(observe(s, mover), legal)
            rules.apply_action(s, a)
            last_msg = f"{agents[mover].name}: {action_str(a)}"
            last_t = time.time()
        # draw
        screen.fill(COL["bg"])
        for r in range(SIZE):
            for c in range(SIZE):
                pygame.draw.rect(screen, COL["grid"], (c * CELL, r * CELL, CELL, CELL), 1)
        for t in TREES:
            r, c = divmod(t, SIZE)
            pygame.draw.circle(screen, COL["tree"], (c * CELL + CELL // 2, r * CELL + CELL // 2), 26)
        for i, n in enumerate(NODES):
            r, c = divmod(n, SIZE)
            cx, cy = c * CELL + CELL // 2, r * CELL + CELL // 2
            pygame.draw.polygon(screen, COL["node"], [(cx, cy - 26), (cx + 26, cy), (cx, cy + 26), (cx - 26, cy)])
            screen.blit(font.render(str(s.stock[i]), True, (10, 80, 90)), (c * CELL + 4, r * CELL + 2))
            if s.post[i] != -1:
                pygame.draw.rect(screen, COL["blue" if s.post[i] == 0 else "red"],
                                 (c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6), 4)
        for p in (0, 1):
            r, c = divmod(BASES[p], SIZE)
            pygame.draw.rect(screen, COL["blue" if p == 0 else "red"],
                             (c * CELL + 8, r * CELL + 8, CELL - 16, CELL - 16), 3)
            screen.blit(font.render(f"HP{s.bhp[p]}", True, COL["blue" if p == 0 else "red"]),
                        (c * CELL + 12, r * CELL + CELL - 28))
            for i, u in enumerate(s.units[p]):
                r, c = divmod(u[0], SIZE)
                cx, cy = c * CELL + CELL // 2, r * CELL + CELL // 2
                pygame.draw.circle(screen, COL["blue" if p == 0 else "red"], (cx, cy), 20)
                if p == s.player and sel == i and agents[p] is None:
                    pygame.draw.circle(screen, (255, 215, 0), (cx, cy), 24, 3)
                screen.blit(font.render(str(u[1]), True, (255, 255, 255)), (cx - 5, cy - 10))
        x = SIZE * CELL + 14
        sc = rules.scores(s)
        names = [a.name if a else "Human" for a in agents]
        screen.blit(big.render("Monte Carlo Strategy AI", True, (20, 20, 20)), (x, 10))
        y = 60
        for p in (0, 1):
            col = COL["blue" if p == 0 else "red"]
            screen.blit(font.render(f"{names[p]}{'  <- to move' if s.player == p and not s.over else ''}", True, col), (x, y))
            screen.blit(font.render(f"res {s.res[p]}  units {len(s.units[p])}  posts {s.post.count(p)}", True, (30, 30, 30)), (x, y + 22))
            screen.blit(font.render(f"base HP {s.bhp[p]}  score {sc[p]}", True, (30, 30, 30)), (x, y + 44))
            y += 90
        screen.blit(font.render(f"turn {s.turn // 2 + 1}/{rules.MAX_TURNS}   AP {s.ap}", True, (30, 30, 30)), (x, y))
        screen.blit(font.render(last_msg, True, (30, 30, 30)), (x, y + 30))
        if s.over:
            msg = "DRAW" if s.winner == -1 else f"{names[s.winner]} wins!  (N = new game)"
            screen.blit(big.render(msg, True, (20, 20, 20)), (x, y + 70))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

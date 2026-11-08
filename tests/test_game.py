"""Unit tests:  python -m unittest discover -s tests -v   (from the project root)"""
import os
import sys
import unittest
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import new_state, play_game, rules
from game.actions import *
from game.state import *
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.mcts import MCTSAgent
from agents.monte_carlo import MCControlAgent, QTable
from agents.rl_mcts import RLMCTSAgent
from agents.abstraction import state_key, N_STATES
import game.game as G
from game import state as S

G.RANDOM_MAPS = False          # the hand-written tests below assume the original fixed map


class TestGeometry(unittest.TestCase):
    def test_symmetry(self):
        rot = lambda c: SIZE * SIZE - 1 - c
        self.assertEqual(sorted(rot(c) for c in NODES), sorted(NODES))
        self.assertEqual(sorted(rot(c) for c in TREES), sorted(TREES))
        self.assertEqual(rot(BASES[0]), BASES[1])
        for n in NODES:
            self.assertEqual(DIST[BASES[0]][n], DIST[BASES[1]][rot(n)])

    def test_all_nodes_reachable(self):
        for p in (0, 1):
            for n in NODES + [BASES[1 - p]]:
                self.assertLess(DIST[BASES[p]][n], 99)


class TestRules(unittest.TestCase):
    def test_initial_actions(self):
        s = new_state("A")
        acts = rules.legal_actions(s)
        self.assertIn((PASS, -1, 0), acts)
        self.assertNotIn((RECRUIT, -1, 0), acts)          # 3 resources < cost 4
        self.assertFalse(any(a[0] == COLLECT for a in acts))

    def test_two_action_points_then_turn_ends(self):
        s = new_state("A")
        rules.apply_action(s, (DEFEND, 0, 0))
        self.assertEqual((s.player, s.ap), (0, 1))
        rules.apply_action(s, (DEFEND, 1, 0))
        self.assertEqual((s.player, s.ap), (1, 2))

    def test_collect_outpost_income_and_stock(self):
        s = new_state("A")
        s.units[0][0][0] = NODES[1]
        rules.apply_action(s, (COLLECT, 0, 0))
        self.assertEqual((s.res[0], s.stock[1]), (START_RES + 2, NODE_STOCK - 2))
        rules.apply_action(s, (OUTPOST, 0, 0))
        self.assertEqual(s.post[1], 0)
        self.assertEqual(s.res[0], START_RES + 2 - COST_OUTPOST)
        rules.apply_action(s, (PASS, -1, 0))
        rules.apply_action(s, (PASS, -1, 0))
        self.assertEqual(s.res[0], START_RES + 2 - COST_OUTPOST + 1)   # +1 passive income

    def test_deterministic_combat_and_base_kill(self):
        s = new_state("A")
        s.units[0][0][0] = cell(5, 4)                      # next to enemy base (5,5)
        for _ in range(BASE_HP):
            s.player, s.ap = 0, 2
            self.assertIn((ATTACK, 0, 4), rules.legal_actions(s))
            rules.apply_action(s, (ATTACK, 0, 4))
        self.assertTrue(s.over)
        self.assertEqual(s.winner, 0)
        self.assertEqual(rules.value(s, 0), 1.0)
        self.assertEqual(rules.value(s, 1), -1.0)

    def test_random_games_terminate_and_are_consistent(self):
        rules.seed(1)
        for env in "ABC":
            for _ in range(50):
                s = new_state(env)
                n = 0
                while not s.over:
                    acts = rules.legal_actions(s)
                    self.assertTrue(acts)
                    rules.apply_action(s, random.choice(acts))
                    for p in (0, 1):
                        self.assertGreaterEqual(s.res[p], 0)
                        self.assertLessEqual(len(s.units[p]), MAX_UNITS)
                    for st in s.stock:
                        self.assertGreaterEqual(st, 0)
                    n += 1
                    self.assertLess(n, 500)
                self.assertIn(s.winner, (-1, 0, 1))

    def test_copy_is_independent(self):
        s = new_state("B")
        t = s.copy()
        rules.apply_action(t, (DEFEND, 0, 0))
        self.assertEqual(s.units[0][0][2], 0)

    def test_categories_cover_all_actions(self):
        rules.seed(2)
        s = new_state("B")
        for _ in range(60):
            acts = rules.legal_actions(s)
            cats = rules.categorize(s, acts)
            self.assertEqual(len(acts), len(cats))
            self.assertTrue(all(0 <= c < NCAT for c in cats))
            if s.over:
                break
            rules.apply_action(s, random.choice(acts))

    def test_hidden_observation(self):
        from game import observe
        s = new_state("C")
        o = observe(s, 0)
        self.assertEqual(o.res[1], -1)
        self.assertEqual(s.res[1], START_RES)               # true state untouched
        self.assertEqual(observe(new_state("B"), 0).res[1], START_RES)


class TestRandomMaps(unittest.TestCase):
    def setUp(self):
        G.RANDOM_MAPS = True

    def tearDown(self):
        G.RANDOM_MAPS = False
        S.set_map(None)

    def test_maps_fair_connected_and_reproducible(self):
        seen = set()
        for sd in range(200):
            S.set_map(sd)
            seen.add((tuple(S.NODES), tuple(sorted(S.TREES)), tuple(S.BASES)))
            self.assertEqual(S.rot(S.BASES[0]), S.BASES[1])
            self.assertEqual(S.NODES[0], S.CENTRE)
            for n in S.NODES:
                self.assertIn(S.rot(n), S.NODES)
                self.assertEqual(S.DIST[S.BASES[0]][n], S.DIST[S.BASES[1]][S.rot(n)])
                self.assertLess(S.DIST[S.BASES[0]][n], 99)
            self.assertEqual(len(S.START_UNITS[0]), 2)
        self.assertGreater(len(seen), 150)                  # maps really differ
        S.set_map(7); a = (list(S.NODES), sorted(S.TREES), list(S.BASES))
        S.set_map(8); S.set_map(7)
        self.assertEqual(a, (list(S.NODES), sorted(S.TREES), list(S.BASES)))

    def test_play_seed_selects_map(self):
        r1 = play_game((RandomAgent(), RandomAgent()), "B", seed=11)
        n1 = list(S.NODES)
        r2 = play_game((RandomAgent(), RandomAgent()), "B", seed=12)
        n2 = list(S.NODES)
        r3 = play_game((RandomAgent(), RandomAgent()), "B", seed=11)
        self.assertEqual(n1, list(S.NODES))
        self.assertNotEqual(n1, n2)

    def test_agents_play_on_random_maps(self):
        q = QTable()
        for sd in range(8):
            for a in (GreedyAgent(), MCControlAgent(q), MCTSAgent(sims=8), RLMCTSAgent(q, sims=8)):
                r = play_game((a, GreedyAgent()), env="C", seed=sd)
                self.assertTrue(r["final"].over)


class TestAgents(unittest.TestCase):
    def test_state_key_range(self):
        rules.seed(3)
        s = new_state("B")
        while not s.over:
            for p in (0, 1):
                self.assertTrue(0 <= state_key(s, p) < N_STATES)
            rules.apply_action(s, random.choice(rules.legal_actions(s)))

    def test_all_agents_play_legal_games(self):
        q = QTable()
        agents = [RandomAgent(), GreedyAgent(), MCControlAgent(q), MCTSAgent(sims=10),
                  RLMCTSAgent(q, sims=10)]
        for env in "ABC":
            for a in agents:
                r = play_game((a, GreedyAgent()), env=env, seed=1)
                self.assertTrue(r["final"].over)

    def test_greedy_beats_random(self):
        w = 0
        for g in range(20):
            r = play_game((GreedyAgent(), RandomAgent()), env="B", seed=g)
            w += r["winner"] == 0
        self.assertGreaterEqual(w, 19)

    def test_mcts_finds_immediate_collect(self):
        rules.seed(0)
        s = new_state("A")
        s.units[0][0][0] = NODES[1]
        ag = MCTSAgent(sims=200)
        a = ag.act(s, rules.legal_actions(s))
        self.assertIn(a[0], (COLLECT, OUTPOST) if s.res[0] >= 3 else (COLLECT,))

    def test_mc_update_moves_q_toward_return(self):
        q = QTable()
        ag = MCControlAgent(q, learn=True)
        ag.trace = [(5, 3), (6, 3)]
        ag.diffs = [0.0, 5.0]
        ag.update(final_diff=10.0, outcome=1.0)
        self.assertGreater(q.Q[6][3], 0)
        self.assertGreater(q.Q[5][3], 0)


if __name__ == "__main__":
    unittest.main()

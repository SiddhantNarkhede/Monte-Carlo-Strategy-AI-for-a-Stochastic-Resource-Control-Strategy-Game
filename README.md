# Monte Carlo Strategy AI

A small two-player turn-based strategy game plus three AI families that play it:
**tabular Monte-Carlo control**, **Monte Carlo Tree Search**, and an **RL-guided MCTS hybrid**.
Everything is pure Python (+ numpy/matplotlib/Pillow for plots); one CPU is enough.

## The game
7x7 board, rotationally symmetric. Each player has a base (5 HP), starts with 2 units (3 HP) and 3 resources,
and has **2 action points per turn** for **15 turns** (a game is 30 turns total).

| Action | Effect |
|---|---|
| MOVE | step one cell (trees and enemy base block) |
| COLLECT | unit on a resource node harvests it (node has a finite stockpile that regrows +1/round; centre holds 9, others 6) |
| BUILD OUTPOST | unit on a free node, cost 3: territory + passive income each turn |
| RECRUIT | cost 4, up to 4 units |
| ATTACK | enemy unit, enemy base (5 hits = win), or raze an enemy outpost |
| DEFEND | +1 defence until your next turn |

Score = resources + 10 x outposts + 8 x units + 4 x base HP; destroying the enemy base wins immediately,
otherwise the higher score wins at turn 30.

Environments: **A** deterministic (fixed yields, combat hits iff attack >= defence); **B** stochastic
(yield 1-3, income 0-2 per outpost, combat hit probability sigmoid(0.8 x strength gap));
**C** = B with the opponent's resources hidden from the observer.

## Agents
* **Random**, **Greedy** (rule-based: favourable attack > outpost > recruit > collect > walk to nodes).
* **MC-Control** - every-visit Monte-Carlo control, epsilon-greedy, on 62k abstract states x 9 *strategic categories*
  (MOVE_ECON / MOVE_AGGR / MOVE_BACK / COLLECT / BUILD_OUTPOST / BUILD_UNIT / ATTACK / DEFEND / PASS).
  Return = discounted (gamma 0.99) sum of score-difference changes; trained against a mixture of Random, Greedy and
  self-play, 300k episodes (A, B) / 200k (C), step size annealed 0.02 -> 0.002.
* **MCTS** - open-loop UCT (handles stochastic transitions), rollouts truncated at 12 actions and scored with the
  score-difference value function. In env C the hidden enemy resources are sampled every simulation (determinisation).
* **RL+MCTS** - same search, but PUCT with a prior from the learned policy and rollouts played by the learned policy.

## Quick start
```
pip install -r requirements.txt
python -m unittest discover -s tests          # 15 tests
./run_all.sh                                     # train all 3 envs + recompute every result
python main.py train --env B --episodes 300000   # ~6 min; writes models/qtable_B.pkl
python main.py policy --env B                    # choose softmax temperature
python main.py tournament --env B --games 100
python main.py watch --blue greedy --red rlmcts --env B
python main.py gif   --blue greedy --red rlmcts --out results/demo.gif
python main.py ui    --blue human  --red rlmcts  # needs pygame (untested, see below)
```
**Random maps.** Every game is played on a freshly generated map. The game's seed picks the map
(`play_game(..., seed=k)`, `watch --seed k`), so all agents in a tournament face the same maps. Maps are symmetric under
180-degree rotation (fair for both seats), connected, with the centre node fixed and the other 4 nodes, 4 trees and the two
bases placed randomly (`game/state.py: generate_map`). Training uses map seeds >= 10^6, evaluation seeds are < 10^5, so
evaluation maps are never seen in training. Add `--fixed-map` to any command to use the original single map.

**Models and results are not pre-computed in this version** - run `./run_all.sh` (about 1.5-2 h on one core; see below).
`archive_fixed_map/` holds the earlier fixed-map study (stale; do not mix with new results).

## Layout
`game/` rules, state, renderer, Gym wrapper - `agents/` all agents + abstraction - `training/` MC control, self-play,
policy extraction - `evaluation/` tournament, metrics, experiments, plots - `visualization/` GIF recorder, Pygame UI -
`tests/`. Results and discussion: **RESULTS.md**.

## Honest limitations
* `visualization/game_ui.py` (Pygame) and `game/gym_env.py` (Gymnasium) were **not executed** - neither library was available when this was built.
* Random maps vary difficulty from game to game, so results have more variance than on a single map.
* Win rates are from 40-300 games per pairing; see confidence intervals in `results/tables.md`.
* The game strongly favours the first mover, so all matches alternate seats.
* Greedy and the other baselines are hand-written by me, not independently strong reference players.

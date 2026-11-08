# Results

Search agents use **100 simulations per decision** unless stated. Win rate counts draws as 0.5, seats alternate every game.
Full head-to-head tables with 95% Wilson intervals: `results/tables.md`. Figures: `results/fig_*.png`.

## Round robin (row beats column)
| | Env A (det.) | Env B (stoch.) | Env C (hidden) |
|---|---|---|---|
| MC-Control vs Greedy (n=300) | 0.89 | 0.47 | 0.42 |
| MCTS vs Greedy (n=100) | 0.05 | 0.06 | 0.01 |
| RL+MCTS vs Greedy (n=100) | 1.00 | 0.96 | 0.73 |
| RL+MCTS vs MC-Control (n=100) | 0.93 | 0.85 | **0.31** |
| RL+MCTS vs MCTS (n=100) | 0.94 | 0.98 | 1.00 |

Everything beats Random ~100% of the time.

## Findings
1. **Tabular MC control matches or beats the hand-written Greedy** in A, ties in B (0.47), and trails slightly in C (0.42).
2. **Plain MCTS is weak here** (random rollouts, 100 sims): it loses to Greedy and to MC-Control.
3. **A learned policy makes MCTS far stronger.** RL+MCTS beats everything in A and B.
4. **Ablation (env B, n=80, vs Greedy / vs plain MCTS):** MCTS 0.01 / -; +prior only 0.73 / 0.88; +learned rollouts only 0.53 / 0.93; both 0.91 / 0.99. The prior contributes most; the two are complementary.
5. **Not just extra compute.** At 100 sims, time per decision is ~6 ms (MCTS) vs ~38 ms (RL+MCTS). RL+MCTS with only 25 sims beat MCTS with 400 sims 80% of the time (n=40; roughly 9 ms vs ~45 ms per decision, the latter extrapolated from the 100-sim timing).
6. **Scaling (env B, n=40 per cell):** RL+MCTS vs Greedy: 0.61 / 0.85 / 0.94 / 1.00 at 25 / 50 / 100 / 200 sims. Plain MCTS stays at ~0-0.05 vs Greedy at every budget tried, so more simulations alone did not fix it up to 200 sims.
7. **Hidden information hurts search most.** RL+MCTS drops from 0.96 to 0.73 vs Greedy, and in C the plain MC-Control policy beats it head-to-head (0.69). Determinising by sampling the enemy's resources uniformly is a crude belief model and the likely cause (not tested further).
8. **Sensitivity of the learned policy:** sampling it at softmax temperature >= 0.08 collapses play (<=0.10 vs Greedy in B); near-greedy (0.01) is best, so search used tau=0.01 for rollouts and a softer 0.1 for the prior.

## Caveats
* Single training seed per environment; MC-control win rate fluctuated strongly mid-training (0% to 44% in B) and only stabilised after annealing the step size. Final checkpoints are reported, not best-of.
* Env C was trained for 200k rather than 300k episodes.
* Intervals are 40-300-game samples; 40-game cells (scaling, equal-time) are indicative only.
* Rules and score weights were tuned during development to make the game non-degenerate (first version collapsed into "collect forever"); results are specific to this game design.

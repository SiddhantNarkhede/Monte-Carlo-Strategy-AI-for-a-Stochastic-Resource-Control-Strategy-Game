#!/usr/bin/env bash
# Retrain every model and recompute every result on random maps.
# Roughly 1.5-2 hours on one CPU core.  Safe to re-run; edit the numbers below to go faster.
set -e
EPISODES=${EPISODES:-300000}      # MC-control training episodes per environment
SIMS=${SIMS:-100}                 # MCTS simulations per decision in the tournaments
mkdir -p models results

for e in A B C; do
  python main.py train  --env $e --episodes $EPISODES     # -> models/qtable_$e.pkl, results/training_curve_$e.json
  python main.py policy --env $e --games 150              # -> models/policy_$e.json (softmax temperature)
done

for e in A B C; do
  python -m evaluation.run_experiments --section roundrobin --env $e \
         --games-cheap 300 --games-search 100 --sims $SIMS   # -> results/roundrobin_$e.json
done
python -m evaluation.run_experiments --section ablation --env B --games-ablation 80 --sims $SIMS
python -m evaluation.run_experiments --section scaling  --env B --games-scaling 40
python -m evaluation.plots                                   # figures + results/tables.md
echo "done - see results/tables.md and results/fig_*.png"

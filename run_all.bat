@echo off
rem Retrain every model and recompute every result on random maps (Windows).
rem Takes roughly 1.5-2 hours on one CPU core.
if not defined EPISODES set EPISODES=300000
if not defined SIMS set SIMS=100
if not exist models mkdir models
if not exist results mkdir results

for %%e in (A B C) do (
  python main.py train --env %%e --episodes %EPISODES% || exit /b 1
  python main.py policy --env %%e --games 150 || exit /b 1
)

for %%e in (A B C) do (
  python -m evaluation.run_experiments --section roundrobin --env %%e --games-cheap 300 --games-search 100 --sims %SIMS% || exit /b 1
)
python -m evaluation.run_experiments --section ablation --env B --games-ablation 80 --sims %SIMS% || exit /b 1
python -m evaluation.run_experiments --section scaling --env B --games-scaling 40 || exit /b 1
python -m evaluation.plots || exit /b 1
echo done - see results\tables.md and results\fig_*.png
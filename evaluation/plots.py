"""Create figures and markdown tables from the JSON files in results/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "results"
ORDER = ["Random", "Greedy", "MC-Control", "MCTS", "RL+MCTS"]
COL = {"Random": "#9aa0a6", "Greedy": "#f0a030", "MC-Control": "#4c9f70",
       "MCTS": "#3478dc", "RL+MCTS": "#c0392b"}
ENV_NAMES = {"A": "A: deterministic", "B": "B: stochastic", "C": "C: hidden info"}


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _wr_matrix(rr):
    """Matrix M[i][j] = win rate of ORDER[i] vs ORDER[j] (draw = 0.5)."""
    M = np.full((len(ORDER), len(ORDER)), np.nan)
    for k, v in rr["pairs"].items():
        a, b = k.split("|")
        i, j = ORDER.index(a), ORDER.index(b)
        M[i, j] = v["win_rate"]
        M[j, i] = 1 - v["win_rate"]
    return M


def fig_training():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    ok = False
    for ax, e in zip(axs, "ABC"):
        c = _load(f"training_curve_{e}.json")
        if not c:
            continue
        ok = True
        x = [r["episode"] / 1000 for r in c]
        ax.plot(x, [r["wr_vs_greedy"] for r in c], "-o", color=COL["Greedy"], label="vs Greedy")
        ax.plot(x, [r["wr_vs_random"] for r in c], "-o", color=COL["Random"], label="vs Random")
        ax.set_title(f"Env {ENV_NAMES[e]}")
        ax.set_xlabel("training episodes (thousands)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("win rate of greedy policy")
    axs[0].legend()
    fig.suptitle("Monte-Carlo control: learning curves (200 evaluation games per point)")
    fig.tight_layout()
    if ok:
        fig.savefig(f"{RES}/fig_training_curves.png", dpi=140)
    plt.close(fig)


def fig_heatmaps():
    data = {e: _load(f"roundrobin_{e}.json") for e in "ABC"}
    envs = [e for e in "ABC" if data[e]]
    if not envs:
        return
    fig, axs = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.6))
    axs = np.atleast_1d(axs)
    for ax, e in zip(axs, envs):
        M = _wr_matrix(data[e])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(ORDER, rotation=35, ha="right"); ax.set_yticklabels(ORDER)
        for i in range(5):
            for j in range(5):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Env {ENV_NAMES[e]}")
        ax.set_xlabel("opponent"); ax.set_ylabel("agent (row)")
    fig.suptitle("Round-robin win rate (row beats column; draws = 0.5)")
    fig.tight_layout()
    fig.savefig(f"{RES}/fig_roundrobin_heatmaps.png", dpi=140)
    plt.close(fig)


def fig_vs_greedy():
    data = {e: _load(f"roundrobin_{e}.json") for e in "ABC"}
    envs = [e for e in "ABC" if data[e]]
    if not envs:
        return
    names = ["MC-Control", "MCTS", "RL+MCTS"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.25
    for k, n in enumerate(names):
        vals, lo, hi = [], [], []
        for e in envs:
            pairs = data[e]["pairs"]
            key = next((x for x in pairs if set(x.split("|")) == {n, "Greedy"}), None)
            v = pairs[key]
            a = key.split("|")[0]
            wr, l, h = (v["win_rate"], v["ci_lo"], v["ci_hi"]) if a == n else (1 - v["win_rate"], 1 - v["ci_hi"], 1 - v["ci_lo"])
            vals.append(wr); lo.append(wr - l); hi.append(h - wr)
        ax.bar(np.arange(len(envs)) + (k - 1) * w, vals, w, yerr=[lo, hi], capsize=3,
               color=COL[n], label=n)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(envs))); ax.set_xticklabels([ENV_NAMES[e] for e in envs])
    ax.set_ylabel("win rate vs rule-based Greedy"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title("Learning/search agents against the Greedy baseline (95% CI)")
    fig.tight_layout(); fig.savefig(f"{RES}/fig_vs_greedy.png", dpi=140); plt.close(fig)


def fig_scaling():
    d = _load("scaling_B.json")
    if not d:
        return
    rows = d["rows"]
    x = [r["sims"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, lab, col in [("mcts_vs_greedy", "MCTS", COL["MCTS"]), ("rl_vs_greedy", "RL+MCTS", COL["RL+MCTS"])]:
        y = np.array([r[key]["win_rate"] for r in rows])
        lo = np.array([r[key]["ci_lo"] for r in rows]); hi = np.array([r[key]["ci_hi"] for r in rows])
        axs[0].plot(x, y, "-o", color=col, label=lab)
        axs[0].fill_between(x, lo, hi, color=col, alpha=.15)
    axs[0].axhline(0.5, color="k", ls="--", lw=1)
    axs[0].set_xscale("log", base=2); axs[0].set_xlabel("simulations per decision")
    axs[0].set_ylabel("win rate vs Greedy"); axs[0].set_ylim(0, 1); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[0].set_title("Search budget vs strength")
    y = np.array([r["rl_vs_mcts"]["win_rate"] for r in rows])
    lo = np.array([r["rl_vs_mcts"]["ci_lo"] for r in rows]); hi = np.array([r["rl_vs_mcts"]["ci_hi"] for r in rows])
    axs[1].plot(x, y, "-o", color=COL["RL+MCTS"]); axs[1].fill_between(x, lo, hi, color=COL["RL+MCTS"], alpha=.15)
    axs[1].axhline(0.5, color="k", ls="--", lw=1)
    axs[1].set_xscale("log", base=2); axs[1].set_xlabel("simulations per decision")
    axs[1].set_ylabel("RL+MCTS win rate vs plain MCTS"); axs[1].set_ylim(0, 1); axs[1].grid(alpha=.3)
    axs[1].set_title("Head-to-head at equal simulations")
    fig.tight_layout(); fig.savefig(f"{RES}/fig_sim_scaling.png", dpi=140); plt.close(fig)


def fig_ablation():
    d = _load("ablation_B.json")
    if not d:
        return
    names = list(d["rows"].keys())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = [d["rows"][n]["vs_greedy"]["win_rate"] for n in names]
    lo = [v - d["rows"][n]["vs_greedy"]["ci_lo"] for v, n in zip(vals, names)]
    hi = [d["rows"][n]["vs_greedy"]["ci_hi"] - v for v, n in zip(vals, names)]
    ax.bar(names, vals, yerr=[lo, hi], capsize=3, color=["#3478dc", "#7d9ee0", "#d98c85", "#c0392b"][:len(names)])
    ax.axhline(0.5, color="k", ls="--", lw=1); ax.set_ylim(0, 1)
    ax.set_ylabel("win rate vs Greedy"); ax.grid(axis="y", alpha=.3)
    ax.set_title("Ablation: where does the learned policy help the search?")
    fig.tight_layout(); fig.savefig(f"{RES}/fig_ablation.png", dpi=140); plt.close(fig)


def tables():
    out = []
    for e in "ABC":
        rr = _load(f"roundrobin_{e}.json")
        if not rr:
            continue
        # aggregate each agent against the other four
        rows = {n: dict(w=0, l=0, d=0, g=0, sa=0.0, sb=0.0, turns=0.0, t=0.0, k=0.0, nt=0) for n in ORDER}
        for key, v in rr["pairs"].items():
            a, b = key.split("|")
            n = v["games"]
            for who, opp, w, l, sa, sb in ((a, b, v["wins"], v["losses"], v["avg_score_a"], v["avg_score_b"]),
                                           (b, a, v["losses"], v["wins"], v["avg_score_b"], v["avg_score_a"])):
                r = rows[who]
                r["w"] += w; r["l"] += l; r["d"] += v["draws"]; r["g"] += n
                r["sa"] += sa * n; r["sb"] += sb * n; r["turns"] += v["avg_turns"] * n; r["k"] += v["base_kill_rate"] * n
            rows[a]["t"] += v["ms_per_decision_a"] * n; rows[a]["nt"] += n
        out.append(f"\n### Environment {ENV_NAMES[e]}  (search budget {rr['sims']} simulations/decision)\n")
        out.append("| Agent | Win rate (all opponents) | Avg score | Avg opp. score | Avg game length (turns/player) | ms / decision |")
        out.append("|---|---|---|---|---|---|")
        for n in ORDER:
            r = rows[n]
            wr = (r["w"] + 0.5 * r["d"]) / r["g"]
            tm = ("%.2f" % (r["t"] / r["nt"])) if r["nt"] else "-"
            out.append(f"| {n} | {wr*100:.1f}% | {r['sa']/r['g']:.1f} | {r['sb']/r['g']:.1f} | {r['turns']/r['g']:.1f} | {tm} |")
        out.append("\nHead-to-head (win rate of row agent vs column agent, 95% Wilson CI, n games):\n")
        M = _wr_matrix(rr)
        out.append("| row \\ col | " + " | ".join(ORDER) + " |")
        out.append("|---|" + "---|" * len(ORDER))
        for i, a in enumerate(ORDER):
            cells = []
            for j, b in enumerate(ORDER):
                if i == j:
                    cells.append("-")
                    continue
                key = next((k for k in rr["pairs"] if k == f"{a}|{b}" or k == f"{b}|{a}"), None)
                v = rr["pairs"][key]
                lo, hi = (v["ci_lo"], v["ci_hi"]) if key == f"{a}|{b}" else (1 - v["ci_hi"], 1 - v["ci_lo"])
                cells.append(f"{M[i, j]:.2f} [{lo:.2f},{hi:.2f}] (n={v['games']})")
            out.append(f"| **{a}** | " + " | ".join(cells) + " |")
    with open(f"{RES}/tables.md", "w") as f:
        f.write("\n".join(out) + "\n")
    return "\n".join(out)


def main():
    fig_training(); fig_heatmaps(); fig_vs_greedy(); fig_scaling(); fig_ablation()
    print(tables())


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "results" / "runs"
PLOTS_DIR = REPO_ROOT / "docs" / "assets" / "plots"

# Plot settings
MA_WINDOW = 50                 # moving average on mean curve
BOXPLOT_TAIL_N = 100           # metric: mean of last N rewards per run
DPI = 180


def safe_agent_filename(agent: str) -> str:
    return agent.lower().replace(" ", "_").replace("-", "_")


def load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_path"] = str(p)
            runs.append(data)
        except Exception as e:
            print(f"[WARN] Could not read {p.name}: {e}")
    return runs


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if len(x) < window:
        return np.array([], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def score_last_n(rewards: List[float], n: int) -> float:
    arr = np.asarray(rewards, dtype=np.float32)
    if len(arr) == 0:
        return float("nan")
    tail = arr[-n:] if len(arr) >= n else arr
    return float(np.mean(tail))


def pick_latest_per_agent_seed(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    If you accidentally have multiple files for same (agent, seed),
    keep the latest by filename lexicographic (usually timestamp-based).
    """
    latest: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for r in runs:
        if "agent" not in r or "rewards" not in r or "seed" not in r:
            continue

        agent = str(r["agent"])
        seed = int(r["seed"])
        key = (agent, seed)

        if key not in latest:
            latest[key] = r
            continue

        # compare by path string (timestamp usually in filename)
        if str(r.get("_path", "")) > str(latest[key].get("_path", "")):
            latest[key] = r

    return latest


def group_by_agent(latest_runs: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (agent, _seed), run in latest_runs.items():
        grouped[agent].append(run)

    # sort runs by seed for stable output
    for agent in grouped:
        grouped[agent].sort(key=lambda r: int(r.get("seed", 0)))
    return dict(grouped)


def stack_rewards_same_length(runs: List[Dict[str, Any]]) -> np.ndarray:
    """
    Returns array shape (n_seeds, T) by truncating to the minimum length across seeds.
    Truncation is a safe fallback if you accidentally trained different episode counts.
    """
    reward_arrays = [np.asarray(r["rewards"], dtype=np.float32) for r in runs]
    lengths = [len(a) for a in reward_arrays]
    if min(lengths) != max(lengths):
        print(f"[WARN] Different episode lengths for same agent: {sorted(set(lengths))}. Truncating to {min(lengths)}.")
    T = min(lengths)
    stacked = np.stack([a[:T] for a in reward_arrays], axis=0)  # (n, T)
    return stacked


def plot_agent_mean_curve(agent: str, runs: List[Dict[str, Any]], out_dir: Path) -> Path:
    stacked = stack_rewards_same_length(runs)  # (n_seeds, T)
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0)

    episodes = np.arange(1, len(mean_curve) + 1, dtype=np.int32)

    plt.figure()
    plt.plot(episodes, mean_curve, linewidth=2, label="mean")
    plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="±1 std")

    ma = moving_average(mean_curve, MA_WINDOW)
    if ma.size > 0:
        ma_eps = np.arange(MA_WINDOW, len(mean_curve) + 1, dtype=np.int32)
        plt.plot(ma_eps, ma, linewidth=2, label=f"MA({MA_WINDOW})")

    # final metric = mean over seeds of "last N mean"
    lastn_means = [score_last_n(r["rewards"], BOXPLOT_TAIL_N) for r in runs]
    metric = float(np.mean(lastn_means))

    plt.title(f"{agent} — Mean Learning Curve over {len(runs)} seed(s) (mean last{BOXPLOT_TAIL_N}: {metric:.1f})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linewidth=0.3)
    plt.legend()

    out_path = out_dir / f"{safe_agent_filename(agent)}_mean_learning_curve.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return out_path


def plot_combined_comparison(agents_to_runs: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> Path:
    """
    One plot with mean curves for all agents (optional std shading).
    Uses truncation across agents too, so they share same x-range.
    """
    # compute mean/std per agent first
    agent_stats = {}
    lengths = []
    for agent, runs in agents_to_runs.items():
        stacked = stack_rewards_same_length(runs)
        mean_curve = stacked.mean(axis=0)
        std_curve = stacked.std(axis=0)
        agent_stats[agent] = (mean_curve, std_curve)
        lengths.append(len(mean_curve))

    T = min(lengths)
    if min(lengths) != max(lengths):
        print(f"[WARN] Agents have different episode lengths in their aggregated curves: {sorted(set(lengths))}. Truncating to {T} for combined plot.")

    episodes = np.arange(1, T + 1, dtype=np.int32)

    plt.figure()
    for agent in sorted(agent_stats.keys(), key=lambda a: a.lower()):
        mean_curve, std_curve = agent_stats[agent]
        mean_curve = mean_curve[:T]
        std_curve = std_curve[:T]
        plt.plot(episodes, mean_curve, linewidth=2, label=agent)
        plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

    plt.title("CartPole — Learning Curve Comparison (mean ± std over seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linewidth=0.3)
    plt.legend()

    out_path = out_dir / "learning_comparison.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return out_path


def plot_final_performance_boxplot(agents_to_runs: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> Path:
    """
    Boxplot of per-seed final performance (mean reward over last N episodes).
    """
    labels = []
    data = []

    for agent in sorted(agents_to_runs.keys(), key=lambda a: a.lower()):
        runs = agents_to_runs[agent]
        vals = [score_last_n(r["rewards"], BOXPLOT_TAIL_N) for r in runs]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            continue
        labels.append(agent)
        data.append(vals)

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(f"Final Performance over Seeds (mean of last {BOXPLOT_TAIL_N} episode rewards)")
    plt.ylabel("Reward")
    plt.grid(True, axis="y", linewidth=0.3)

    out_path = out_dir / "final_performance_boxplot.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return out_path


def print_readme_snippet(agents_to_runs: Dict[str, List[Dict[str, Any]]]) -> None:
    # summary table: mean ± std over seeds of last-N mean
    rows: List[Tuple[str, int, float, float]] = []  # agent, n_seeds, mean, std

    for agent, runs in sorted(agents_to_runs.items(), key=lambda kv: kv[0].lower()):
        vals = np.asarray([score_last_n(r["rewards"], BOXPLOT_TAIL_N) for r in runs], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((agent, len(runs), float(vals.mean()), float(vals.std())))

    print("\n--- Paste into README.md ---\n")
    print(f"**Metric:** mean reward over last {BOXPLOT_TAIL_N} episodes (per seed), reported as mean ± std across seeds.\n")
    print("| Agent | Seeds | Final Performance (mean ± std) |")
    print("|------:|------:|-------------------------------:|")
    for agent, n_seeds, m, s in rows:
        print(f"| {agent} | {n_seeds} | {m:.2f} ± {s:.2f} |")

    print("\n## Learning Curve Comparison")
    print("![Learning comparison](docs/assets/plots/learning_comparison.png)\n")

    print("## Final Performance (Boxplot over Seeds)")
    print("![Final performance](docs/assets/plots/final_performance_boxplot.png)\n")

    print("## Per-Agent Mean Learning Curves")
    for agent, *_ in rows:
        img = f"docs/assets/plots/{safe_agent_filename(agent)}_mean_learning_curve.png"
        print(f"### {agent}\n![{agent} mean learning curve]({img})\n")


def main() -> None:
    if not RUNS_DIR.exists():
        raise SystemExit(f"[ERROR] runs directory not found: {RUNS_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_runs(RUNS_DIR)
    if not runs:
        raise SystemExit(f"[ERROR] No JSON runs found in: {RUNS_DIR}")

    latest = pick_latest_per_agent_seed(runs)
    if not latest:
        raise SystemExit("[ERROR] No valid runs found. Each JSON must include: agent, seed, rewards.")

    agents_to_runs = group_by_agent(latest)
    print(f"[OK] Found {len(runs)} run files. Using {len(latest)} latest runs across {len(agents_to_runs)} agent(s).")

    # Make per-agent mean curves
    for agent, aruns in sorted(agents_to_runs.items(), key=lambda kv: kv[0].lower()):
        out = plot_agent_mean_curve(agent, aruns, PLOTS_DIR)
        print(f"[OK] Wrote {out.relative_to(REPO_ROOT)}")

    # Combined comparison plot
    comp = plot_combined_comparison(agents_to_runs, PLOTS_DIR)
    print(f"[OK] Wrote {comp.relative_to(REPO_ROOT)}")

    # Final performance boxplot
    box = plot_final_performance_boxplot(agents_to_runs, PLOTS_DIR)
    print(f"[OK] Wrote {box.relative_to(REPO_ROOT)}")

    # README snippet
    print_readme_snippet(agents_to_runs)
    print("[DONE]")


if __name__ == "__main__":
    main()
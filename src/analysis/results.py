"""Analysis for the Monet latent-start study (interpretibility/PLAN.md).

Runs locally on the CSV files downloaded from Modal. No GPU needed.

    python src/analysis/results.py exp-a --input results/exp_a \
        --out results/exp_a/analysis --metric prob

Only `exp-a` (Step 2) is implemented so far; `exp-b` will be added in Step 3.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SETTINGS = ("real", "noise", "no_image")
GROUPS = ("all", "latent_start", "no_latent_start")
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
COLORS = {"real": "#2a78d6", "noise": "#eb6834", "no_image": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {
    "prob": ("start_prob", "P(latent start)"),
    "log_prob": ("start_log_prob", "log P(latent start)"),
}


# ---------------------------------------------------------------- loading

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1")


def load_exp_a(input_dir: Path):
    trajs = {}
    for r in read_csv(input_dir / "exp_a_trajectories.csv"):
        trajs[r["trajectory_id"]] = {
            "trajectory_id": r["trajectory_id"],
            "question_index": int(r["question_index"]),
            "group": r["group"],
            "setting": r["setting"],
            "started": as_bool(r["started"]),
            "first_start_step": int(r["first_start_step"]) if r["first_start_step"] != "" else None,
            "completion_tokens": int(r["completion_tokens"]),
            "steps": [],
        }
    for r in read_csv(input_dir / "exp_a_steps.csv"):
        t = trajs.get(r["trajectory_id"])
        if t is None:
            sys.exit(f"ERROR: step row for unknown trajectory {r['trajectory_id']}")
        t["steps"].append({
            "step": int(r["step"]),
            "start_prob": float(r["start_prob"]),
            "start_log_prob": float(r["start_log_prob"]),
            "in_latent": as_bool(r["in_latent"]),
        })
    for t in trajs.values():
        t["steps"].sort(key=lambda s: s["step"])
    return list(trajs.values())


def series(traj: dict, column: str, stop_at_first_start: bool) -> np.ndarray:
    """Values indexed by step. Latent steps are NaN (always removed); with
    stop_at_first_start the series ends at the first start token, inclusive."""
    end = traj["completion_tokens"]
    if stop_at_first_start and traj["first_start_step"] is not None:
        end = traj["first_start_step"] + 1
    y = np.full(end, np.nan)
    for s in traj["steps"]:
        if s["step"] < end and not s["in_latent"]:
            y[s["step"]] = s[column]
    return y


# ---------------------------------------------------------------- statistics

def bin_values(y: np.ndarray, bin_size: int) -> np.ndarray:
    """Mean of the non-NaN values in each bin of `bin_size` steps (NaN if none)."""
    n_bins = math.ceil(len(y) / bin_size)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        chunk = y[b * bin_size:(b + 1) * bin_size]
        if np.any(~np.isnan(chunk)):
            out[b] = np.nanmean(chunk)
    return out


def combine(per_question: dict[int, np.ndarray], n_boot: int, rng: np.random.Generator):
    """Per bin: n, mean, percentile-bootstrap 95% CI across questions, median.
    The CI is NaN where fewer than 2 questions contribute."""
    n_bins = max((len(v) for v in per_question.values()), default=0)
    rows = []
    for b in range(n_bins):
        vals = np.array([v[b] for v in per_question.values() if b < len(v) and not np.isnan(v[b])])
        n = len(vals)
        if n == 0:
            rows.append({"bin": b, "n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                         "median": np.nan})
            continue
        lo = hi = np.nan
        if n >= 2:
            means = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
        rows.append({"bin": b, "n": n, "mean": vals.mean(), "ci_low": lo, "ci_high": hi,
                     "median": float(np.median(vals))})
    return rows


def frequency_table(trajs: list[dict]) -> list[dict]:
    rows = []
    for s in SETTINGS:
        ts = [t for t in trajs if t["setting"] == s]
        firsts = [t["first_start_step"] for t in ts if t["started"]]
        rows.append({
            "setting": s,
            "started": len(firsts),
            "total": len(ts),
            "percent": round(100 * len(firsts) / len(ts), 1) if ts else "",
            "median_first_start_step": float(np.median(firsts)) if firsts else "",
        })
    return rows


# ---------------------------------------------------------------- plotting

def style(ax, ylabel: str, xlabel: str = "Generation step"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def new_fig(ncols: int, width: float):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.6), facecolor=SURFACE,
                             sharex=ncols > 1, squeeze=False)
    return fig, axes[0]


def plot_question(qi: int, group: str, trajs: list[dict], column: str, ylabel: str,
                  stop: bool, path: Path):
    fig, (ax,) = new_fig(1, 9)
    for t in sorted(trajs, key=lambda t: SETTINGS.index(t["setting"])):
        y = series(t, column, stop)
        x = np.arange(len(y))
        label = t["setting"] + (f" (start @ {t['first_start_step']})" if t["started"] else "")
        ax.plot(x, y, color=COLORS[t["setting"]], linewidth=2, label=label)
        if t["started"] and t["first_start_step"] < len(y):
            fs = t["first_start_step"]
            ax.plot([fs], [y[fs]], marker="o", markersize=8, color=COLORS[t["setting"]],
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
    style(ax, ylabel)
    ax.set_title(f"q{qi:03d} ({group})  —  dot = first latent start", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_combined(curves: dict[str, list[dict]], bin_size: int, ylabel: str, title: str, path: Path):
    fig, (ax, ax_n) = new_fig(2, 15)
    for s in SETTINGS:
        rows = curves.get(s, [])
        if not rows:
            continue
        x = np.array([r["bin"] * bin_size for r in rows])
        get = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
        c = COLORS[s]
        ax.fill_between(x, get("ci_low"), get("ci_high"), color=c, alpha=0.15, linewidth=0)
        ax.plot(x, get("mean"), color=c, linewidth=2, label=f"{s} mean")
        ax.plot(x, get("median"), color=c, linewidth=1.5, linestyle="--", label=f"{s} median")
        ax_n.step(x, get("n"), where="post", color=c, linewidth=2, label=s)
    style(ax, ylabel + ("" if bin_size == 1 else f" (bin = {bin_size} steps)"))
    style(ax_n, "Questions contributing")
    ax.set_title("Mean (solid), 95% bootstrap CI (band), median (dashed)", color=INK_2,
                 fontsize=10, loc="left")
    ax_n.set_title("Sample count per position", color=INK_2, fontsize=10, loc="left")
    ax_n.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3)
    ax_n.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- outputs

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fmt(v):
    return "" if v == "" or v is None or (isinstance(v, float) and math.isnan(v)) else v


def markdown_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(fmt(r[c])) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def run_exp_a(args):
    column, ylabel = METRICS[args.metric]
    trajs = load_exp_a(args.input)
    if not trajs:
        sys.exit(f"ERROR: no trajectories in {args.input}")
    subsets = [args.group] if args.group else list(GROUPS)
    mode = "stop" if args.stop_at_first_start else "keep"
    out = args.out / f"{args.metric}_{mode}_bin{args.bin_size}"
    (out / "per_question").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # One graph per question.
    by_question = defaultdict(list)
    for t in trajs:
        if args.group in (None, "all") or t["group"] == args.group:
            by_question[t["question_index"]].append(t)
    for qi, ts in sorted(by_question.items()):
        plot_question(qi, ts[0]["group"], ts, column, ylabel, args.stop_at_first_start,
                      out / "per_question" / f"q{qi:03d}.png")

    # Combined figure, plotted values and frequency table per subset.
    md = [f"# Experiment A: latent-start frequency\n\nInput: `{args.input}`\n"]
    curve_rows, freq_rows = [], []
    for subset in subsets:
        ts = [t for t in trajs if subset == "all" or t["group"] == subset]
        if not ts:
            print(f"{subset}: no trajectories, skipped")
            continue
        n_q = len({t["question_index"] for t in ts})
        curves = {}
        for s in SETTINGS:
            per_q = {t["question_index"]: bin_values(series(t, column, args.stop_at_first_start),
                                                     args.bin_size)
                     for t in ts if t["setting"] == s}
            curves[s] = combine(per_q, args.bootstrap, rng)
            for r in curves[s]:
                curve_rows.append({
                    "group": subset, "setting": s, "metric": args.metric, "bin": r["bin"],
                    "bin_start_step": r["bin"] * args.bin_size,
                    "bin_end_step": (r["bin"] + 1) * args.bin_size - 1,
                    "n": r["n"], "mean": fmt(r["mean"]), "ci_low": fmt(r["ci_low"]),
                    "ci_high": fmt(r["ci_high"]), "median": fmt(r["median"]),
                })
        title = (f"Latent-start {'probability' if args.metric == 'prob' else 'log-probability'} "
                 f"— {subset} ({n_q} questions; "
                 f"{'stops at first start' if args.stop_at_first_start else 'continues after start'}"
                 "; latent steps removed)")
        plot_combined(curves, args.bin_size, ylabel, title, out / f"combined_{subset}.png")

        freq = frequency_table(ts)
        freq_rows += [{"group": subset, **r} for r in freq]
        md.append(f"## {subset} ({n_q} questions)\n\n{markdown_table(freq)}\n")
        print(f"\n{subset} ({n_q} questions)\n{markdown_table(freq)}")

    write_csv(out / "combined_curves.csv", curve_rows)
    write_csv(out / "latent_start_frequency.csv", freq_rows)
    (out / "latent_start_frequency.md").write_text("\n".join(md), encoding="utf-8")
    (out / "settings.json").write_text(json.dumps({
        "input": str(args.input), "metric": args.metric, "group": args.group or "all three",
        "stop_at_first_start": args.stop_at_first_start, "bin_size": args.bin_size,
        "bootstrap": args.bootstrap, "seed": args.seed,
    }, indent=2))
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("exp-a", help="Experiment A (Step 2): generation with per-step logging")
    a.add_argument("--input", type=Path, required=True, help="folder with exp_a_*.csv")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--metric", choices=list(METRICS), default="prob")
    a.add_argument("--group", choices=list(GROUPS), default=None,
                   help="restrict to one subset (default: produce all, latent_start and no_latent_start)")
    stop = a.add_mutually_exclusive_group()
    stop.add_argument("--stop-at-first-start", dest="stop_at_first_start", action="store_true",
                      default=True, help="end each line at the first start token (default)")
    stop.add_argument("--keep-after-start", dest="stop_at_first_start", action="store_false",
                      help="keep steps after the first start (latent steps are still removed)")
    a.add_argument("--bin-size", type=int, default=1)
    a.add_argument("--bootstrap", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.bin_size < 1 or args.bootstrap < 1:
        ap.error("--bin-size and --bootstrap must be >= 1")
    if args.cmd == "exp-a":
        run_exp_a(args)


if __name__ == "__main__":
    main()

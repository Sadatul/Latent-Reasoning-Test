"""
Compare a Qwen2.5-VL run with a Monet run on the same dataset, and count Monet's
latent-token usage.

    .venv/bin/python analysis/compare_results.py
    .venv/bin/python analysis/compare_results.py --qwen results/qwen25vl-Q8_0/humanevalplus \
                                                 --monet results/monet-Q8_0/humanevalplus

A problem counts as solved if any of its attempts passed (pass@k with k = --passes
of the run). Writes to results/analysis/<qwen folder>__vs__<monet folder>/:
  report.md                     all numbers below, readable
  report.json                   the same, for scripts
  qwen_pass_monet_fail.csv      problems Qwen solved and Monet did not
  monet_pass_qwen_fail.csv      problems Monet solved and Qwen did not
  failure_categories.png        heatmap: losing model's failure category x platform
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (this file is in analysis/)
RESULTS = ROOT / "results"

# Failure categories in a fixed order (rows of the heatmap).
FAIL_ORDER = ["wrong_answer", "runtime_error", "timeout", "no_code", "truncated", "error"]
FAIL_LABEL = {"wrong_answer": "wrong answer", "runtime_error": "runtime error", "timeout": "timeout",
              "no_code": "no code block", "truncated": "truncated (hit max tokens)", "error": "script error"}
# Sequential one-hue ramp (light -> dark blue) for counts; surface and ink colours.
RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec"


def read_run(folder):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    records = {}
    with open(folder / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "attempts" not in r:  # runs made before --passes existed
                    r["attempts"] = [r]
                    r["num_attempts_passed"] = int(r["passed"])
                records[r["id"]] = r
    return config, records


def dominant_status(rec):
    """Most common status among the attempts; ties go to the first attempt's status."""
    statuses = [a["status"] for a in rec["attempts"]]
    counts = Counter(statuses)
    top = max(counts.values())
    return next(s for s in statuses if counts[s] == top)


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- latent tokens

def latent_report():
    """Latent-token counts for every Monet run under results/, plus the V* probe."""
    runs = []
    for cfg in sorted(RESULTS.glob("monet-*/*/config.json")):
        _, records = read_run(cfg.parent)
        atts = [a for r in records.values() for a in r["attempts"]]
        runs.append({
            "run": rel(cfg.parent),
            "problems": len(records),
            "answers": len(atts),
            "answers_with_latent_start": sum(1 for a in atts if a.get("latent_token_count")),
            "latent_start_tokens_total": sum(a.get("latent_token_count") or 0 for a in atts),
            "answers_with_any_monet_special_token": sum(1 for a in atts if a.get("added_special_tokens")),
            "problems_with_latent_start": sum(1 for r in records.values()
                                              if any(a.get("latent_token_count") for a in r["attempts"])),
        })
    probes = []
    for s in sorted((ROOT / "latent_probe" / "results").glob("*/summary.json")):
        summary = json.loads(s.read_text(encoding="utf-8"))
        probes.append({"run": rel(s.parent), **summary["overall"]})
    return {"monet_runs": runs, "vstar_probe": probes}


# ---------------------------------------------------------------- comparison

def mismatch_rows(winner, loser, ids, winner_name, loser_name, winner_dir, loser_dir):
    rows = []
    for i in ids:
        w, l = winner[i], loser[i]
        la = l["attempts"]
        rows.append({
            "id": i, "title": w.get("title"), "platform": w.get("platform"), "format": w.get("format"),
            "difficulty": w.get("difficulty"), "contest_date": w.get("contest_date"),
            f"{winner_name}_attempts_passed": f"{w['num_attempts_passed']}/{len(w['attempts'])}",
            f"{loser_name}_attempts_passed": f"{l['num_attempts_passed']}/{len(la)}",
            f"{loser_name}_failure": dominant_status(l),
            f"{loser_name}_attempt_statuses": "|".join(a["status"] for a in la),
            f"{loser_name}_finish_reasons": "|".join(str(a.get("finish_reason")) for a in la),
            f"{loser_name}_tests_passed": "|".join(
                f"{a.get('num_passed') or 0}/{a.get('num_tests') if a.get('num_tests') is not None else '-'}"
                for a in la),
            f"{loser_name}_latent_tokens": sum(a.get("latent_token_count") or 0 for a in la),
            f"{winner_name}_trace": rel(Path(winner_dir) / "traces" / f"{i}.md"),
            f"{loser_name}_trace": rel(Path(loser_dir) / "traces" / f"{i}.md"),
        })
    return rows


def write_csv(path, rows, fallback_fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else fallback_fields)
        w.writeheader()
        w.writerows(rows)


def crosstab(rows, loser_name, platforms):
    counts = Counter((r[f"{loser_name}_failure"], r["platform"]) for r in rows)
    return [[counts[(s, p)] for p in platforms] for s in FAIL_ORDER]


def plot(path, panels, platforms, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    vmax = max([1] + [v for _, _, m, _ in panels for row in m for v in row])
    bounds = [0.5] + [0.5 + vmax * k / (len(RAMP) - 1) for k in range(1, len(RAMP))]
    cmap = ListedColormap(RAMP[1:])
    cmap.set_under(EMPTY)  # zero cells recede into a neutral gray
    norm = BoundaryNorm(bounds, cmap.N)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels) + 1.5, 4.6), facecolor=SURFACE)
    for ax, (subtitle, loser_label, m, n) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
        # 2px surface gaps between cells
        ax.set_xticks([x - 0.5 for x in range(1, len(platforms))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(FAIL_ORDER))], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        totals = [sum(row) for row in m]
        ax.set_yticks(range(len(FAIL_ORDER)),
                      [f"{FAIL_LABEL[s]}  ({t})" for s, t in zip(FAIL_ORDER, totals)])
        ax.set_xticks(range(len(platforms)),
                      [f"{p}\n({sum(row[j] for row in m)})" for j, p in enumerate(platforms)])
        for y, row in enumerate(m):
            for x, v in enumerate(row):
                dark = v > 0 and norm(v) >= cmap.N // 2
                ax.text(x, y, str(v) if v else "·", ha="center", va="center",
                        color="#ffffff" if dark else (INK if v else INK_2), fontsize=11,
                        fontweight="bold" if v else "normal")
        ax.set_title(f"{subtitle}\nn = {n} problems · rows = {loser_label}'s failure", fontsize=11,
                     color=INK, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.01, "Failure = most common status across the losing model's attempts "
             "(ties: first attempt). Totals in parentheses.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwen", default=str(RESULTS / "qwen25vl-Q8_0" / "lcb_easy"), help="Qwen2.5-VL run folder")
    ap.add_argument("--monet", default=str(RESULTS / "monet-Q8_0" / "lcb_easy-prompt1"), help="Monet run folder")
    ap.add_argument("--out", help="default: results/analysis/<qwen folder>__vs__<monet folder>")
    args = ap.parse_args()

    qcfg, qwen = read_run(args.qwen)
    mcfg, monet = read_run(args.monet)
    out = Path(args.out) if args.out else RESULTS / "analysis" / f"{Path(args.qwen).name}__vs__{Path(args.monet).name}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- settings that make the comparison unfair
    warnings = []
    if qcfg.get("dataset") != mcfg.get("dataset"):
        warnings.append(f"different datasets: qwen={qcfg.get('dataset')}, monet={mcfg.get('dataset')}")
    for key, default in (("prompt_style", 0), ("passes", 1), ("temperature", None), ("max_tokens", None)):
        qv, mv = qcfg.get(key, default), mcfg.get(key, default)
        if qv != mv:
            warnings.append(f"different {key}: qwen={qv}, monet={mv}")
    only_q, only_m = sorted(set(qwen) - set(monet)), sorted(set(monet) - set(qwen))
    if only_q or only_m:
        warnings.append(f"{len(only_q)} problems only in qwen run, {len(only_m)} only in monet run (ignored)")
    common = sorted(set(qwen) & set(monet))

    q_only = [i for i in common if qwen[i]["passed"] and not monet[i]["passed"]]
    m_only = [i for i in common if monet[i]["passed"] and not qwen[i]["passed"]]
    both = sum(1 for i in common if qwen[i]["passed"] and monet[i]["passed"])
    neither = len(common) - both - len(q_only) - len(m_only)

    rows_qm = mismatch_rows(qwen, monet, q_only, "qwen", "monet", args.qwen, args.monet)
    rows_mq = mismatch_rows(monet, qwen, m_only, "monet", "qwen", args.monet, args.qwen)
    write_csv(out / "qwen_pass_monet_fail.csv", rows_qm, ["id"])
    write_csv(out / "monet_pass_qwen_fail.csv", rows_mq, ["id"])

    platforms = sorted({qwen[i].get("platform") for i in common})
    plot(out / "failure_categories.png",
         [("Qwen2.5-VL solved, Monet failed", "Monet", crosstab(rows_qm, "monet", platforms), len(rows_qm)),
          ("Monet solved, Qwen2.5-VL failed", "Qwen", crosstab(rows_mq, "qwen", platforms), len(rows_mq))],
         platforms, f"Where the two models disagree: {qcfg.get('dataset')} (solved = any of the attempts passed)")

    def fail_counts(rows, name):
        c = Counter(r[f"{name}_failure"] for r in rows)
        return {s: c[s] for s in FAIL_ORDER if c[s]}

    def attempt_counts(recs, ids):
        c = Counter(a["status"] for i in ids for a in recs[i]["attempts"])
        return {s: c[s] for s in FAIL_ORDER + ["pass"] if c[s]}

    report = {
        "qwen_run": rel(args.qwen), "monet_run": rel(args.monet), "dataset": qcfg.get("dataset"),
        "warnings": warnings,
        "problems_compared": len(common),
        "solved": {"qwen": sum(qwen[i]["passed"] for i in common), "monet": sum(monet[i]["passed"] for i in common)},
        "agreement": {"both_solved": both, "neither_solved": neither,
                      "qwen_solved_monet_failed": len(q_only), "monet_solved_qwen_failed": len(m_only)},
        "qwen_solved_monet_failed": {"monet_failure_by_problem": fail_counts(rows_qm, "monet"),
                                     "monet_attempt_statuses": attempt_counts(monet, q_only)},
        "monet_solved_qwen_failed": {"qwen_failure_by_problem": fail_counts(rows_mq, "qwen"),
                                     "qwen_attempt_statuses": attempt_counts(qwen, m_only)},
        "latent_tokens": latent_report(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- readable report
    a = report["agreement"]
    md = [f"# Qwen2.5-VL vs Monet: {report['dataset']}", "",
          f"- Qwen run: `{report['qwen_run']}`", f"- Monet run: `{report['monet_run']}`",
          f"- Problems compared: {len(common)}. Solved = any attempt passed.", ""]
    if warnings:
        md += ["**Comparison caveats:**", ""] + [f"- {w}" for w in warnings] + [""]
    md += ["## Agreement", "", "| | Monet solved | Monet failed |", "|---|---|---|",
           f"| **Qwen solved** | {a['both_solved']} | **{a['qwen_solved_monet_failed']}** |",
           f"| **Qwen failed** | **{a['monet_solved_qwen_failed']}** | {a['neither_solved']} |", ""]
    for key, loser, rows, fname in (("qwen_solved_monet_failed", "Monet", rows_qm, "qwen_pass_monet_fail.csv"),
                                    ("monet_solved_qwen_failed", "Qwen", rows_mq, "monet_pass_qwen_fail.csv")):
        sec = report[key]
        by_problem, by_attempt = list(sec.values())
        md += [f"## {key.replace('_', ' ').capitalize()}: {len(rows)} problems (`{fname}`)", "",
               f"| {loser}'s failure | problems (most common status) | attempts |", "|---|---|---|"]
        md += [f"| {FAIL_LABEL[s]} | {by_problem.get(s, 0)} | {by_attempt.get(s, 0)} |"
               for s in FAIL_ORDER if by_problem.get(s) or by_attempt.get(s)]
        if by_attempt.get("pass"):
            md += [f"| (passed, n/a) | – | {by_attempt['pass']} |"]
        md += [""]
    md += ["![failure categories](failure_categories.png)", "", "## Latent-token usage", "",
           "| Monet run | problems | answers | answers with latent start (151666) | answers with any Monet token |",
           "|---|---|---|---|---|"]
    md += [f"| `{r['run']}` | {r['problems']} | {r['answers']} | {r['answers_with_latent_start']} | "
           f"{r['answers_with_any_monet_special_token']} |" for r in report["latent_tokens"]["monet_runs"]]
    for p in report["latent_tokens"]["vstar_probe"]:
        md += ["", f"V* image probe (`{p['run']}`): latent start fired on **{p['fired']}/{p['n']}** questions; "
               f"median of each question's max p(latent) = {p['p_latent_max_median']}, highest = {p['p_latent_max_max']}."]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nwritten to {rel(out)}/")


if __name__ == "__main__":
    main()

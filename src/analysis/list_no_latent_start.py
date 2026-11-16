"""List the Step 0 answers that never produced a latent-start token, for manual review.

A record "did not start latent reasoning" when its generated token IDs contain no
<abs_vis_token> (latent_start_steps is empty). Default settings: noise and no_image.

  python3 src/analysis/list_no_latent_start.py
  python3 src/analysis/list_no_latent_start.py --step0 results/step0_judge/results/step0 \
      --settings noise,no_image,real

Writes <out>/no_latent_start.csv and <out>/no_latent_start.md (a review checklist with a
link to each trace) and prints the list. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("real", "noise", "no_image")
COLUMNS = ["setting", "problem_id", "index", "category", "answer", "prediction", "correct",
           "status", "judge_answer", "judge_corrected", "completion_tokens", "finish_reason",
           "started_in_real", "started_in_noise", "started_in_no_image", "trace"]


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def started(r: dict) -> bool:
    if "latent_start_steps" in r:
        return bool(r["latent_start_steps"])
    return r["latent_segments"] > 0  # records without token IDs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", type=Path, default=ROOT / "results/step0_judge/results/step0")
    ap.add_argument("--settings", default="noise,no_image")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <step0>/review_no_latent_start")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if bad := [s for s in settings if s not in SETTINGS]:
        sys.exit(f"ERROR: unknown settings {bad}")
    records = {s: {r["index"]: r for r in read_jsonl(args.step0 / s / "predictions.jsonl")}
               for s in SETTINGS if (args.step0 / s / "predictions.jsonl").exists()}
    if missing := [s for s in settings if s not in records]:
        sys.exit(f"ERROR: {args.step0} has no predictions for {missing}")
    out = args.out or args.step0 / "review_no_latent_start"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in settings:
        for i in sorted(records[s]):
            r = records[s][i]
            if started(r):
                continue
            j = r.get("judge") or {}
            trace = args.step0 / s / "traces" / f"vstar_{i:03d}.md"
            row = {"setting": s, "problem_id": f"vstar_{i:03d}", "index": i,
                   "category": r["category"], "answer": r["answer"],
                   "prediction": r["prediction"] or "", "correct": r["correct"],
                   "status": r["status"], "judge_answer": j.get("judge_answer") or ("NONE" if j else ""),
                   "judge_corrected": bool(j.get("accepted")),
                   "completion_tokens": r["completion_tokens"], "finish_reason": r["finish_reason"],
                   "trace": os.path.relpath(trace, out)}
            for other in SETTINGS:
                row[f"started_in_{other}"] = started(records[other][i]) if other in records else ""
            rows.append(row)

    with open(out / "no_latent_start.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    lines = ["# Step 0 answers with no latent-start token", "",
             f"Source: `{args.step0}`. A record is listed when its generated tokens contain no "
             "`<abs_vis_token>`.", "",
             "| setting | no latent start | of | correct among them |", "|---|---|---|---|"]
    for s in settings:
        sub = [r for r in rows if r["setting"] == s]
        lines.append(f"| {s} | {len(sub)} | {len(records[s])} | {sum(r['correct'] for r in sub)} |")
    both = sorted(set(r["index"] for r in rows if r["setting"] == settings[0])
                  .intersection(*[{r["index"] for r in rows if r["setting"] == s} for s in settings[1:]]))
    if len(settings) > 1:
        lines += ["", f"No latent start in all of {', '.join(settings)}: "
                  + (", ".join(f"vstar_{i:03d}" for i in both) or "none")]
    for s in settings:
        lines += ["", f"## {s}", "",
                  "| done | problem | category | answer | prediction | correct | status | started in "
                  + " / ".join(SETTINGS) + " | trace |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["setting"] == s):
            flags = " / ".join({True: "yes", False: "no", "": "-"}[r[f"started_in_{o}"]] for o in SETTINGS)
            lines.append(f"| [ ] | {r['problem_id']} | {r['category']} | {r['answer']} | "
                         f"{r['prediction'] or '-'} | {'yes' if r['correct'] else 'no'} | {r['status']}"
                         f"{' (judge)' if r['judge_corrected'] else ''} | {flags} | [trace]({r['trace']}) |")
    (out / "no_latent_start.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:next(k for k, l in enumerate(lines) if l.startswith("## "))]).rstrip())
    print()
    for r in rows:
        print(f"{r['setting']:9} {r['problem_id']}  answer={r['answer']} pred={r['prediction'] or '-':2} "
              f"{r['status']:12} {r['category']:18} {args.step0 / r['setting'] / 'traces' / (r['problem_id'] + '.md')}")
    print(f"\nWrote {out / 'no_latent_start.csv'} and {out / 'no_latent_start.md'}")


if __name__ == "__main__":
    main()

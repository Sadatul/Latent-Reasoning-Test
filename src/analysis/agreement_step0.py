"""Answer agreement between two Step 0 settings (default: noise vs no_image).

If Monet's answers with a noise image are about the same as with no image at all, the
noise image adds nothing the model uses. Measured per question over the same V* items:

  answer agreement      both settings gave the same option letter
  Cohen's kappa         agreement corrected for chance (from each setting's letter mix)
  correctness agreement both right or both wrong, with its kappa

Which answer counts (--answer-source):
  effective (default)  the scored prediction; if there is none (no readable \\boxed{}),
                       the gpt-oss judge's letter when it gave one with a verified quote.
                       The judge's letter is used even when it is wrong, since this is
                       about what the model answered, not whether it was right.
  boxed                only the scored prediction (\\boxed{} or judge-accepted)

Questions where either side has no answer are reported separately; agreement and kappa
use only questions both settings answered. 95% CIs: percentile bootstrap over questions.

  python3 src/analysis/agreement_step0.py
  python3 src/analysis/agreement_step0.py --a real --b noise

Writes <out>/agreement_<a>_vs_<b>.json, .md and per-question .csv. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("real", "noise", "no_image")


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answer(r: dict, source: str):
    if r["prediction"] is not None or source == "boxed":
        return r["prediction"]
    j = r.get("judge") or {}
    if j.get("judge_answer") and not j.get("error") and j.get("evidence_found"):
        return j["judge_answer"]
    return None


def started(r: dict) -> bool:
    return bool(r["latent_start_steps"]) if "latent_start_steps" in r else r["latent_segments"] > 0


def kappa(pairs) -> float | None:
    """Cohen's kappa for a list of (label_a, label_b)."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def metrics(rows) -> dict:
    answered = [r for r in rows if r["a"] is not None and r["b"] is not None]
    ans_pairs = [(r["a"], r["b"]) for r in answered]
    cor_pairs = [(r["a_correct"], r["b_correct"]) for r in rows]
    return {
        "questions": len(rows),
        "both_answered": len(answered),
        "a_no_answer": sum(r["a"] is None for r in rows),
        "b_no_answer": sum(r["b"] is None for r in rows),
        "answer_agreement": (sum(a == b for a, b in ans_pairs) / len(ans_pairs)) if ans_pairs else None,
        "answer_kappa": kappa(ans_pairs),
        "correctness_agreement": (sum(a == b for a, b in cor_pairs) / len(cor_pairs)) if cor_pairs else None,
        "correctness_kappa": kappa(cor_pairs),
        "both_correct": sum(a and b for a, b in cor_pairs),
        "only_a_correct": sum(a and not b for a, b in cor_pairs),
        "only_b_correct": sum(b and not a for a, b in cor_pairs),
        "both_wrong": sum(not a and not b for a, b in cor_pairs),
        "a_accuracy": sum(a for a, _ in cor_pairs) / len(cor_pairs) if cor_pairs else None,
        "b_accuracy": sum(b for _, b in cor_pairs) / len(cor_pairs) if cor_pairs else None,
    }


CI_KEYS = ("answer_agreement", "answer_kappa", "correctness_agreement", "correctness_kappa")


def bootstrap(rows, n_boot: int, seed: int) -> dict:
    rng = random.Random(seed)
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        m = metrics([rows[rng.randrange(len(rows))] for _ in rows])
        for k in CI_KEYS:
            if m[k] is not None:
                draws[k].append(m[k])
    ci = {}
    for k, v in draws.items():
        v.sort()
        ci[k] = [v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]] if v else None
    return ci


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", type=Path, default=ROOT / "results/step0_judge/results/step0")
    ap.add_argument("--a", default="noise", choices=SETTINGS)
    ap.add_argument("--b", default="no_image", choices=SETTINGS)
    ap.add_argument("--answer-source", default="effective", choices=["effective", "boxed"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <step0>/agreement")
    args = ap.parse_args()
    if args.a == args.b:
        sys.exit("ERROR: --a and --b must differ")

    ra = {r["index"]: r for r in read_jsonl(args.step0 / args.a / "predictions.jsonl")}
    rb = {r["index"]: r for r in read_jsonl(args.step0 / args.b / "predictions.jsonl")}
    if set(ra) != set(rb):
        sys.exit(f"ERROR: {args.a} and {args.b} cover different questions")
    rows = []
    for i in sorted(ra):
        a, b = ra[i], rb[i]
        assert a["answer"] == b["answer"]
        rows.append({"index": i, "problem_id": f"vstar_{i:03d}", "category": a["category"],
                     "answer": a["answer"],
                     "a": answer(a, args.answer_source), "b": answer(b, args.answer_source),
                     "a_correct": a["correct"], "b_correct": b["correct"],
                     "a_started": started(a), "b_started": started(b)})

    subsets = {"all": rows}
    for c in sorted({r["category"] for r in rows}):
        subsets[f"category={c}"] = [r for r in rows if r["category"] == c]
    subsets["latent started in both"] = [r for r in rows if r["a_started"] and r["b_started"]]
    subsets["latent not started in at least one"] = [r for r in rows if not (r["a_started"] and r["b_started"])]

    result = {"step0": str(args.step0), "a": args.a, "b": args.b,
              "answer_source": args.answer_source, "bootstrap": args.bootstrap, "seed": args.seed,
              "subsets": {}}
    for name, sub in subsets.items():
        m = metrics(sub)
        m["ci95"] = bootstrap(sub, args.bootstrap, args.seed) if sub and args.bootstrap else None
        result["subsets"][name] = m

    # Letter-by-letter contingency on questions both answered
    letters = sorted({r["a"] for r in rows if r["a"]} | {r["b"] for r in rows if r["b"]})
    table = Counter((r["a"], r["b"]) for r in rows if r["a"] and r["b"])
    result["contingency"] = {"rows_are": args.a, "cols_are": args.b, "letters": letters,
                             "counts": [[table[(x, y)] for y in letters] for x in letters]}

    out = args.out or args.step0 / "agreement"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"agreement_{args.a}_vs_{args.b}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["index", "problem_id", "category", "answer", f"{args.a}_answer", f"{args.b}_answer",
                "same_answer", f"{args.a}_correct", f"{args.b}_correct",
                f"{args.a}_latent_started", f"{args.b}_latent_started"]
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            same = "" if r["a"] is None or r["b"] is None else r["a"] == r["b"]
            w.writerow([r["index"], r["problem_id"], r["category"], r["answer"], r["a"] or "",
                        r["b"] or "", same, r["a_correct"], r["b_correct"], r["a_started"], r["b_started"]])

    A, B = args.a, args.b

    def ci(m, k, pct):
        c = (m["ci95"] or {}).get(k)
        return f"{fmt(m[k], pct)} [{fmt(c[0], pct)}, {fmt(c[1], pct)}]" if c else fmt(m[k], pct)

    lines = [f"# Answer agreement: {A} vs {B}", "",
             f"Source `{args.step0}`; answer source `{args.answer_source}`; 95% bootstrap CI over "
             f"questions ({args.bootstrap} draws, seed {args.seed}). Answer agreement and kappa use "
             "questions both settings answered.", "",
             f"| subset | n | both answered | answer agreement | answer kappa | correctness agreement "
             f"| correctness kappa | both correct | only {A} | only {B} | both wrong |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in result["subsets"].items():
        lines.append(f"| {name} | {m['questions']} | {m['both_answered']} | "
                     f"{ci(m, 'answer_agreement', True)} | {ci(m, 'answer_kappa', False)} | "
                     f"{ci(m, 'correctness_agreement', True)} | {ci(m, 'correctness_kappa', False)} | "
                     f"{m['both_correct']} | {m['only_a_correct']} | {m['only_b_correct']} | {m['both_wrong']} |")
    m = result["subsets"]["all"]
    lines += ["", f"No answer: {A} {m['a_no_answer']}, {B} {m['b_no_answer']}. "
              f"Accuracy: {A} {fmt(m['a_accuracy'], True)}, {B} {fmt(m['b_accuracy'], True)}.", "",
              f"## Answer letters (rows {A}, columns {B})", "",
              "| | " + " | ".join(letters) + " |", "|---|" + "---|" * len(letters)]
    for x, row in zip(letters, result["contingency"]["counts"]):
        lines.append(f"| {x} | " + " | ".join(map(str, row)) + " |")
    lines += ["", "Kappa: 1 = perfect agreement, 0 = what chance gives from each setting's "
              "own answer mix. Letters stand for different options in different questions; the "
              "chance term only uses how often each setting picks each letter."]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out / stem}.json / .md / .csv")


if __name__ == "__main__":
    main()

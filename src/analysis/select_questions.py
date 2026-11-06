"""Step 1 of the Monet latent-start study: select the question set.

Parses every `vstar_*.md` trace produced by `src/modal/run_vstar.py`, splits
the questions into `latent_start` (latent_segments > 0) and `no_latent_start`
(latent_segments == 0), drops traces that hit the token limit, and picks
5 questions per group with a fixed seed, balancing V* categories.

Usage:
    python src/analysis/select_questions.py \
        --traces results/traces --out results/question_set.csv --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

GROUPS = ("latent_start", "no_latent_start")
PER_GROUP = 5
COLUMNS = [
    "question_index", "problem_id", "group", "category", "status",
    "finish_reason", "prompt_tokens", "completion_tokens",
    "latent_token_count", "latent_segments",
]
META_KEYS = ("id", "index", "category")
ATTEMPT_KEYS = ("status", "finish_reason", "prompt_tokens", "completion_tokens",
                "latent_token_count", "latent_segments")

# A JSON block fenced with 4 or more backticks; the closing fence repeats the
# opening one exactly.
JSON_BLOCK = re.compile(r"^(`{4,})json[ \t]*\n(.*?)\n\1[ \t]*$", re.S | re.M)


def parse_trace(path: Path) -> dict:
    blocks = [json.loads(m.group(2)) for m in JSON_BLOCK.finditer(path.read_text(encoding="utf-8"))]
    metas = [b for b in blocks if all(k in b for k in META_KEYS)]
    attempts = [b for b in blocks if "attempt" in b and all(k in b for k in ATTEMPT_KEYS)]
    if len(metas) != 1:
        raise ValueError(f"{path}: expected 1 metadata block, found {len(metas)}")
    if not attempts:
        raise ValueError(f"{path}: no attempt block found")
    if len(attempts) > 1:
        print(f"WARNING: {path.name} has {len(attempts)} attempts; using attempt "
              f"{attempts[0]['attempt']}", file=sys.stderr)
    meta, att = metas[0], attempts[0]
    return {
        "question_index": int(meta["index"]),
        "problem_id": meta["id"],
        "group": "latent_start" if int(att["latent_segments"]) > 0 else "no_latent_start",
        "category": meta["category"],
        **{k: att[k] for k in ATTEMPT_KEYS},
    }


def balanced_sample(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Round-robin over categories (in seeded random order), each category's
    questions in seeded random order, so categories are as even as possible."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["question_index"]):
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    for c in cats:
        rng.shuffle(by_cat[c])
    picked: list[dict] = []
    while len(picked) < k:
        for c in cats:
            if by_cat[c] and len(picked) < k:
                picked.append(by_cat[c].pop(0))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True, help="folder with vstar_*.md traces")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(args.traces.glob("vstar_*.md"))
    if not paths:
        print(f"ERROR: no vstar_*.md traces in {args.traces}", file=sys.stderr)
        return 1
    rows = [parse_trace(p) for p in paths]
    print(f"Parsed {len(rows)} traces from {args.traces}")

    excluded = [r for r in rows if r["finish_reason"] == "length"]
    eligible = [r for r in rows if r["finish_reason"] != "length"]
    print(f"Excluded (finish_reason == 'length'): {len(excluded)}")

    by_group = {g: [r for r in eligible if r["group"] == g] for g in GROUPS}
    print("Eligible per group:")
    for g in GROUPS:
        cats = dict(sorted(Counter(r["category"] for r in by_group[g]).items()))
        print(f"  {g}: {len(by_group[g])}  categories={cats}")

    short = [g for g in GROUPS if len(by_group[g]) < PER_GROUP]
    if short:
        print(f"ERROR: fewer than {PER_GROUP} eligible questions in: {', '.join(short)}. "
              "Stopping without writing a CSV.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    selected = []
    for g in GROUPS:  # fixed order keeps the RNG stream reproducible
        selected += sorted(balanced_sample(by_group[g], PER_GROUP, rng), key=lambda r: r["question_index"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(selected)

    print(f"Selected (seed={args.seed}):")
    for g in GROUPS:
        sel = [r for r in selected if r["group"] == g]
        cats = dict(sorted(Counter(r["category"] for r in sel).items()))
        print(f"  {g}: indices={[r['question_index'] for r in sel]}  categories={cats}")
    print(f"Wrote {len(selected)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

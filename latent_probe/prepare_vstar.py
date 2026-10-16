"""
Build a small V* subset for the latent-token probe (see README.md in this folder).

    source env.sh
    .venv/bin/python latent_probe/prepare_vstar.py            # 20 questions, 10 per category
    .venv/bin/python latent_probe/prepare_vstar.py --n 40
    .venv/bin/python latent_probe/prepare_vstar.py --full     # all 191 questions

Downloads only test_questions.jsonl and the selected images from the
craigwu/vstar_bench dataset (into hf_cache/), then writes:
  latent_probe/data/vstar_subset.jsonl   one question per line, prompt in Monet's format
                                         (vstar_full.jsonl with --full)
  latent_probe/data/images/              copies of the selected images
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent              # latent_probe/
ROOT = HERE.parent                                  # project root
HF_HUB_CACHE = ROOT / "hf_cache" / "hub"            # same place env.sh points HF_HOME to
OUT = HERE / "data"
SUBSET_FILE = OUT / "vstar_subset.jsonl"   # --n questions (the probe set)
FULL_FILE = OUT / "vstar_full.jsonl"       # --full: every V* question
REPO = "craigwu/vstar_bench"

# V* ends every question with this line. "directly" asks for no reasoning, which
# would also discourage latent reasoning, so it is replaced by Monet's own
# instruction (from the Monet README / inference example).
VSTAR_SUFFIX = "Answer with the option's letter from the given choices directly."
MONET_SUFFIX = "Put your final answer in \\boxed{}."


def monet_prompt(text):
    lines = [l for l in text.strip().split("\n") if l.strip() and l.strip() != VSTAR_SUFFIX]
    question = [l for l in lines if not l.startswith("(")]
    options = [l for l in lines if l.startswith("(")]
    if len(question) != 1 or not options:
        raise ValueError(f"unexpected V* question format: {text!r}")
    return (f"Question: {question[0]}\nThe choices are listed below:\n" + "\n".join(options)
            + f"\n{MONET_SUFFIX}")


def build(n=20, seed=0, full=False):
    """Download the questions and images and write the jsonl; returns its path."""
    qfile = hf_hub_download(REPO, "test_questions.jsonl", repo_type="dataset", cache_dir=HF_HUB_CACHE)
    by_cat = defaultdict(list)
    with open(qfile, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_cat[r["category"]].append(r)

    rng = random.Random(seed)
    cats = sorted(by_cat)
    picked = []
    for i, cat in enumerate(cats):
        if full:
            picked += sorted(by_cat[cat], key=lambda r: int(r["question_id"]))
            continue
        k = n // len(cats) + (1 if i < n % len(cats) else 0)
        picked += sorted(rng.sample(by_cat[cat], k), key=lambda r: int(r["question_id"]))

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in picked:
        src = hf_hub_download(REPO, r["image"], repo_type="dataset", cache_dir=HF_HUB_CACHE)
        dst = OUT / "images" / r["image"].replace("/", "__")
        shutil.copyfile(src, dst)
        rows.append({
            "id": f"vstar_{r['question_id']}",
            "category": r["category"],
            "question_id": r["question_id"],
            "image_path": str(dst.relative_to(HERE)),  # relative, so the folder can be copied
            "original_text": r["text"],
            "prompt_text": monet_prompt(r["text"]),
            "label": r["label"],
        })

    out_file = FULL_FILE if full else SUBSET_FILE
    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {c: sum(r["category"] == c for r in rows) for c in cats}
    print(f"wrote {len(rows)} questions {counts} to {out_file}")
    return out_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of questions, split evenly across the two categories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="all V* questions, written to vstar_full.jsonl")
    args = ap.parse_args()
    build(args.n, args.seed, args.full)


if __name__ == "__main__":
    main()

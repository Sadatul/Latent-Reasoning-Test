"""
Build a small V* subset for the latent-token probe (see README.md in this folder).

    source env.sh
    .venv/bin/python latent_probe/prepare_vstar.py            # 20 questions, 10 per category
    .venv/bin/python latent_probe/prepare_vstar.py --n 40

Downloads only test_questions.jsonl and the selected images from the
craigwu/vstar_bench dataset (into hf_cache/), then writes:
  latent_probe/data/vstar_subset.jsonl   one question per line, prompt in Monet's format
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of questions, split evenly across the two categories")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qfile = hf_hub_download(REPO, "test_questions.jsonl", repo_type="dataset", cache_dir=HF_HUB_CACHE)
    by_cat = defaultdict(list)
    with open(qfile, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_cat[r["category"]].append(r)

    rng = random.Random(args.seed)
    cats = sorted(by_cat)
    picked = []
    for i, cat in enumerate(cats):
        k = args.n // len(cats) + (1 if i < args.n % len(cats) else 0)
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

    with open(OUT / "vstar_subset.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {c: sum(r["category"] == c for r in rows) for c in cats}
    print(f"wrote {len(rows)} questions {counts} to {OUT / 'vstar_subset.jsonl'}")


if __name__ == "__main__":
    main()

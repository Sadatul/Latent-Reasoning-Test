"""
Download LiveCodeBench-lite and HumanEval+ from Hugging Face, build two subsets
of LiveCodeBench, and save stats for every dataset as JSON.

Output (inside this folder):
  data/raw/livecodebench/test*.jsonl     original LiveCodeBench-lite files
  data/raw/humanevalplus/test.jsonl      original HumanEval+
  data/lcb_easy.jsonl                    LiveCodeBench, difficulty == easy
  data/lcb_post_qwen25vl.jsonl           LiveCodeBench, released after Qwen2.5-VL
  data/stats/<dataset>.json              stats for each dataset
"""

import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

LCB_REPO = "livecodebench/code_generation_lite"
HEP_REPO = "evalplus/humanevalplus"
QWEN25VL_RELEASE = date(2025, 1, 26)  # problems dated strictly after this go into the post-release set


def download(repo, filenames, out_dir):
    paths = []
    for fn in filenames:
        print(f"downloading {repo}/{fn}")
        paths.append(Path(hf_hub_download(repo, fn, repo_type="dataset", local_dir=out_dir)))
    return paths


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def describe(values):
    values = list(values)
    if not values:
        return {"count": 0}
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def lcb_stats(rows, name):
    difficulty = Counter(r["difficulty"] for r in rows)
    platform = Counter(r["platform"] for r in rows)
    by_platform = defaultdict(Counter)
    by_month = defaultdict(Counter)
    for r in rows:
        by_platform[r["platform"]][r["difficulty"]] += 1
        by_month[r["contest_date"][:7]][r["difficulty"]] += 1
    dates = sorted(r["contest_date"][:10] for r in rows)
    n = len(rows)
    return {
        "dataset": name,
        "num_problems": n,
        "difficulty": {k: difficulty[k] for k in ("easy", "medium", "hard") if k in difficulty},
        "difficulty_percent": {k: round(100 * difficulty[k] / n, 1) for k in ("easy", "medium", "hard") if k in difficulty},
        "platform": dict(platform),
        "difficulty_by_platform": {p: dict(c) for p, c in sorted(by_platform.items())},
        "format": {
            "function (LeetCode, has starter code)": sum(1 for r in rows if r["starter_code"].strip()),
            "stdin/stdout (AtCoder, Codeforces)": sum(1 for r in rows if not r["starter_code"].strip()),
        },
        "date_range": [dates[0], dates[-1]] if dates else None,
        "problems_per_month": {m: sum(c.values()) for m, c in sorted(by_month.items())},
        "difficulty_by_month": {m: dict(c) for m, c in sorted(by_month.items())},
        "source_files": dict(Counter(r["_source_file"] for r in rows)),
        "public_tests_per_problem": describe(len(json.loads(r["public_test_cases"])) for r in rows),
        "question_length_words": describe(len(r["question_content"].split()) for r in rows),
    }


def humanevalplus_stats(rows):
    return {
        "dataset": "humanevalplus",
        "num_problems": len(rows),
        "difficulty": "not labelled (all tasks are introductory-level single functions)",
        "release": "HumanEval tasks from 2021, all before Qwen2.5-VL, so no post-release subset",
        "format": "function (complete the given signature + docstring)",
        "prompt_length_lines": describe(len(r["prompt"].splitlines()) for r in rows),
        "canonical_solution_lines": describe(len(r["canonical_solution"].strip().splitlines()) for r in rows),
    }


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k != "_source_file"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    # LiveCodeBench-lite: every release file (test.jsonl = v1, test2.jsonl = v2, ...)
    lcb_files = sorted(f for f in list_repo_files(LCB_REPO, repo_type="dataset")
                       if f.startswith("test") and f.endswith(".jsonl"))
    lcb_paths = download(LCB_REPO, lcb_files, DATA / "raw" / "livecodebench")

    # HumanEval+
    (hep_path,) = download(HEP_REPO, ["test.jsonl"], DATA / "raw" / "humanevalplus")

    # Load LiveCodeBench, remembering which release file each problem came from
    lcb = []
    for path in lcb_paths:
        for row in read_jsonl(path):
            row["_source_file"] = path.name
            lcb.append(row)

    easy = [r for r in lcb if r["difficulty"] == "easy"]
    post = [r for r in lcb if date.fromisoformat(r["contest_date"][:10]) > QWEN25VL_RELEASE]
    write_jsonl(easy, DATA / "lcb_easy.jsonl")
    write_jsonl(post, DATA / "lcb_post_qwen25vl.jsonl")

    hep = list(read_jsonl(hep_path))

    stats = {
        "livecodebench_lite": lcb_stats(lcb, "livecodebench_lite"),
        "lcb_easy": lcb_stats(easy, "lcb_easy"),
        "lcb_post_qwen25vl": {**lcb_stats(post, "lcb_post_qwen25vl"),
                              "cutoff": f"contest_date after {QWEN25VL_RELEASE}"},
        "humanevalplus": humanevalplus_stats(hep),
    }
    for name, s in stats.items():
        write_json(s, DATA / "stats" / f"{name}.json")
        print(f"{name:22s} {s['num_problems']:5d} problems  {s['difficulty']}")


if __name__ == "__main__":
    main()

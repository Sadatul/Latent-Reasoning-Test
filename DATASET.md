# Datasets

Reference for the data produced by `prepare_datasets.py`. Read this before
writing evaluation or analysis code. All paths are relative to this folder.

## Purpose

Compare **Qwen2.5-VL-7B-Instruct** with **Monet-7B** (Qwen2.5-VL-7B fine-tuned
for latent visual reasoning) on text-only algorithmic coding problems. The
question is whether Monet's visual-latent training helps or hurts ordinary
coding. Both models share the same pretraining, so contamination affects them
equally; the post-release subset is there for a cleaner comparison.

## File layout

```
data/
├── raw/
│   ├── livecodebench/          original LiveCodeBench-lite files, unmodified
│   │   ├── test.jsonl          release v1 (May 2023 – Mar 2024)
│   │   ├── test2.jsonl         v2 additions (to May 2024)
│   │   ├── test3.jsonl         v3 additions (to Jul 2024)
│   │   ├── test4.jsonl         v4 additions (to Sep 2024)
│   │   ├── test5.jsonl         v5 additions (to Jan 2025)
│   │   └── test6.jsonl         v6 additions (to Apr 2025)
│   └── humanevalplus/
│       └── test.jsonl          HumanEval+, unmodified
├── lcb_easy.jsonl              subset: LiveCodeBench, difficulty == "easy"
├── lcb_post_qwen25vl.jsonl     subset: LiveCodeBench, contest_date after 2025-01-26
└── stats/
    ├── livecodebench_lite.json all LiveCodeBench files combined
    ├── lcb_easy.json
    ├── lcb_post_qwen25vl.json
    └── humanevalplus.json
```

The script downloads every `test*.jsonl` in the LiveCodeBench repo, so if a
newer release file appears (e.g. `test7.jsonl`) it is included automatically.
The full set is the concatenation of all `test*.jsonl` files (1,055 problems as
of release v6).

All files are JSON Lines: one problem per line, UTF-8. The LiveCodeBench files
are large (hidden tests are embedded), so stream them line by line instead of
loading the whole file into memory at once.

## Sources

| Dataset | Hugging Face repo | Split / files |
|---|---|---|
| LiveCodeBench-lite | `livecodebench/code_generation_lite` (dataset) | `test*.jsonl`, downloaded directly (not via its loading script) |
| HumanEval+ | `evalplus/humanevalplus` (dataset) | `test.jsonl` (164 problems) |

## Subset rules

| File | Rule |
|---|---|
| `lcb_easy.jsonl` | `difficulty == "easy"`, from all release files |
| `lcb_post_qwen25vl.jsonl` | `date(contest_date) > 2025-01-26` (strictly after Qwen2.5-VL's release; problems dated 2025-01-26 are excluded) |

Subsets contain complete original rows with exactly the same fields as the raw
files. A problem can appear in both subsets. No subsets are built from
HumanEval+: it has no difficulty labels or dates, and all its tasks are from
2021.

## LiveCodeBench row format

Applies to `data/raw/livecodebench/*.jsonl`, `lcb_easy.jsonl` and
`lcb_post_qwen25vl.jsonl`. Every value is a string.

| Field | Content |
|---|---|
| `question_title` | Problem title |
| `question_content` | Full problem statement, including examples and constraints |
| `platform` | `"leetcode"`, `"atcoder"` or `"codeforces"` |
| `question_id` | Platform problem id (unique together with `platform`) |
| `contest_id` | Contest the problem belongs to |
| `contest_date` | ISO datetime, e.g. `"2025-02-15T00:00:00"`; use the first 10 characters as the date |
| `starter_code` | LeetCode: class/method signature to complete. AtCoder/Codeforces: empty string |
| `difficulty` | `"easy"`, `"medium"` or `"hard"` |
| `public_test_cases` | JSON string: list of test objects (see below); these are the examples shown in the statement |
| `private_test_cases` | Hidden tests, encoded (see below); use these for scoring |
| `metadata` | JSON string. For LeetCode it contains `func_name`, the method to call |

### Two problem formats

**Functional (LeetCode, `starter_code` non-empty).** The model completes the
`class Solution` method in `starter_code`. To test: instantiate `Solution()`
and call the method named `json.loads(metadata)["func_name"]`.

**Stdin/stdout (AtCoder and Codeforces, `starter_code` empty).** The model
writes a full program that reads standard input and prints to standard output.

### Test objects

Each test is a dict:

```json
{"input": "...", "output": "...", "testtype": "functional" | "stdin"}
```

- `testtype == "stdin"`: feed `input` to the program on stdin and compare its
  stdout with `output`, ignoring trailing whitespace on each line.
- `testtype == "functional"`: `input` holds one JSON value per line, one per
  argument, in order. Parse with `[json.loads(l) for l in input.split("\n")]`.
  `output` is the JSON-encoded return value; compare with `json.loads(output)`.

### Decoding the hidden tests

`private_test_cases` is either a plain JSON string or
`base64( zlib( pickle( json_string ) ) )`. Decode it like this:

```python
import base64, io, json, pickle, zlib

class _NoGlobals(pickle.Unpickler):
    # The payload is a pickled str, which never needs imports.
    # Refusing all globals means a tampered file can fail but never run code.
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

def decode_private_tests(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
        return json.loads(_NoGlobals(io.BytesIO(raw)).load())
```

`public_test_cases` is always plain JSON: `json.loads(row["public_test_cases"])`.

### Loading example

```python
import json
from pathlib import Path

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

easy = list(load_jsonl("data/lcb_easy.jsonl"))
easy_leetcode = [r for r in easy if r["platform"] == "leetcode"]

full = [r for p in sorted(Path("data/raw/livecodebench").glob("test*.jsonl"))
        for r in load_jsonl(p)]
```

## HumanEval+ row format

File: `data/raw/humanevalplus/test.jsonl` (164 rows).

| Field | Content |
|---|---|
| `task_id` | `"HumanEval/0"` … `"HumanEval/163"` |
| `prompt` | Function signature and docstring. The model completes the body |
| `canonical_solution` | Reference body (continues `prompt`) |
| `entry_point` | Name of the function under test |
| `test` | Python source of the EvalPlus test harness (base + extra tests) |

To score: build `prompt + completion` (or the model's full function), append
`test`, then call the check function defined in `test` on `entry_point`. Look
at one row's `test` before writing the harness, to confirm the name and
signature of its check function.

## Stats files

`data/stats/<dataset>.json`. LiveCodeBench stats files share these keys:

| Key | Meaning |
|---|---|
| `num_problems` | Number of rows |
| `difficulty` / `difficulty_percent` | Counts and percentages for easy / medium / hard |
| `platform` | Counts per platform |
| `difficulty_by_platform` | Platform → difficulty → count |
| `format` | Functional (has starter code) vs stdin/stdout counts |
| `date_range` | `[earliest, latest]` contest dates |
| `problems_per_month`, `difficulty_by_month` | Counts by `YYYY-MM` |
| `source_files` | Rows per raw release file (shows which release each problem came from) |
| `public_tests_per_problem` | mean / median / min / max |
| `question_length_words` | mean / median / min / max |

`lcb_post_qwen25vl.json` also has `cutoff`. `humanevalplus.json` has
`num_problems`, prompt and solution length stats, and notes on why there are
no subsets.

Check these files for the actual counts instead of assuming them; they are
produced by the run.

## Models and caches

`env.sh` sets `HF_HOME=hf_cache/` in this folder. Run `source env.sh` before
downloading models or data so everything stays here. Model repos for this
project:

| Model | Hugging Face repo |
|---|---|
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Monet-7B | `NOVAglow646/Monet-7B` |
| Monet-SFT-7B | `NOVAglow646/Monet-SFT-7B` |
| Qwen2.5-VL-7B GGUF (Q8_0, Q4_K_M, …) | `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF` |
| Monet-7B GGUF | `mradermacher/Monet-7B-GGUF` |

Monet's latent reasoning (token id 151666 switches to latent mode) only works
with its patched vLLM runner (`vllm==0.10.0`); see
https://github.com/NOVAglow646/Monet. GGUF / llama.cpp runs Monet as a plain
text model without latent mode.

## Regenerating

```bash
source env.sh
.venv/bin/python prepare_datasets.py
```

Downloads are cached, so a re-run only rebuilds the subsets and stats. To
change the cutoff, edit `QWEN25VL_RELEASE` in `prepare_datasets.py`.

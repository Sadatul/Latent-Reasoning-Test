"""
Evaluate one GGUF model on one coding dataset with llama-cpp-python.

    python src/run_eval.py --model qwen25vl --dataset lcb_easy
    python src/run_eval.py --model monet    --dataset humanevalplus --quant Q8_0
    python src/run_eval.py --model qwen25vl --dataset lcb_easy --passes 5 --temperature 0.2

Each run writes to results/<model>-<quant>/<dataset>/ (or --output-dir):
  config.json     settings of the run (checked on resume)
  state.json      ids of finished problems, attempts of the running one; rerun to resume
  results.jsonl   one record per problem; per attempt: full output, reasoning, code, verdict
  traces/<id>.md  the same record, formatted for reading
  summary.json    pass@1 ... pass@K overall and per difficulty / platform / format

Ctrl+C pauses: at most the attempt in progress is lost. See README.md.
"""

import argparse
import base64
import io
import json
import os
import pickle
import re
import sys
import time
import traceback
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import code_exec

ROOT = Path(__file__).resolve().parent.parent  # project root (this file is in src/)
DATA = ROOT / "data"
HF_HUB_CACHE = ROOT / "hf_cache" / "hub"  # same place env.sh points HF_HOME to

MODELS = {
    "qwen25vl": {"repo": "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", "stem": "Qwen2.5-VL-7B-Instruct"},
    "monet": {"repo": "mradermacher/Monet-7B-GGUF", "stem": "Monet-7B"},
}
DATASETS = ["lcb_easy", "lcb_post_qwen25vl", "lcb_full", "humanevalplus"]

# Qwen2.5 tokenizer ids end at 151664; ids above that are tokens Monet added.
# 151666 switches Monet into latent mode (only in its patched vLLM runner).
QWEN_LAST_TOKEN_ID = 151664
MONET_LATENT_START_ID = 151666

# Bump when prompts change, so a resumed run cannot mix two prompt versions.
PROMPT_VERSION = 1

# Prompt style used when --prompt-style is not given.
#   0: original prompt; works well for Qwen2.5-VL.
#   1: pushes harder for a final code block, for models (Monet) that often stop without one.
PROMPT_STYLE = 0
PROMPT_STYLES = {
    0: {
        "system": "You are an expert Python programmer.",
        "instruction": (
            "First, think through the problem step by step: restate what is asked, work out an "
            "algorithm, and check it against the examples. Then give your complete solution as a "
            "single ```python code block at the end of your answer."
        ),
        "format_tail": "",
    },
    1: {
        "system": (
            "You are an expert Python programmer. You solve coding problems and always end your "
            "answer with the complete solution in a ```python code block."
        ),
        "instruction": (
            "Think through the problem briefly: work out an algorithm and check it against the "
            "examples. Keep this reasoning short. Then you MUST write your complete solution as a "
            "single ```python code block at the end of your answer. Only that code block is graded: "
            "an answer without it scores zero, so never stop before writing it. Do not give the "
            "answer as plain text or in \\boxed{}."
        ),
        "format_tail": (
            "\n\nUse exactly this format:\n"
            "<brief reasoning>\n"
            "```python\n"
            "<complete solution code>\n"
            "```"
        ),
    },
}

# Settings that must match for a run to be resumed.
CONFIG_KEYS = ["model", "quant", "gguf_file", "dataset", "n_ctx", "max_tokens", "temperature",
               "top_p", "top_k", "min_p", "repeat_penalty", "seed", "prompt_version", "prompt_style",
               "system_prompt"]


# ---------------------------------------------------------------- data

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def dataset_files(name):
    if name == "humanevalplus":
        return [DATA / "raw" / "humanevalplus" / "test.jsonl"]
    if name == "lcb_full":
        return sorted((DATA / "raw" / "livecodebench").glob("test*.jsonl"))
    return [DATA / f"{name}.jsonl"]


def index_problems(name):
    """Light index of a dataset: metadata plus the file offset of each row.
    LiveCodeBench rows embed large hidden tests, so full rows are read one at a
    time with load_row() instead of keeping them in memory."""
    files = dataset_files(name)
    missing = [str(p) for p in files if not p.exists()]
    if not files or missing:
        sys.exit(f"dataset files missing: {missing or name}. Run prepare_datasets.py first (see README.md).")
    problems = []
    for path in files:
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                r = json.loads(line)
                if name == "humanevalplus":
                    meta = {"id": r["task_id"].replace("/", "_"), "platform": "humaneval",
                            "difficulty": None, "contest_date": None, "format": "humaneval",
                            "title": r["entry_point"]}
                else:
                    meta = {"id": f"{r['platform']}_{r['question_id']}", "platform": r["platform"],
                            "difficulty": r["difficulty"], "contest_date": r["contest_date"][:10],
                            "format": "functional" if r["starter_code"].strip() else "stdin",
                            "title": r["question_title"]}
                meta["_file"], meta["_offset"] = str(path), offset
                problems.append(meta)
    ids = [p["id"] for p in problems]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        sys.exit(f"duplicate problem ids in {name}: {dupes[:5]}")
    return problems


def load_row(problem):
    with open(problem["_file"], "rb") as f:
        f.seek(problem["_offset"])
        return json.loads(f.readline())


# ---------------------------------------------------------------- prompts

def build_user_prompt(problem, row, style):
    s = PROMPT_STYLES[style]
    if problem["format"] == "humaneval":
        return (
            "Complete the following Python function.\n\n"
            f"```python\n{row['prompt']}```\n\n"
            f"{s['instruction']} The code block must contain the full function "
            f"(imports, signature and body), not only the body.{s['format_tail']}"
        )
    if problem["format"] == "functional":
        return (
            f"### Question:\n{row['question_content']}\n\n"
            "### Format: You will use the following starter code to write the solution to the "
            "problem and enclose your code within delimiters.\n"
            f"```python\n{row['starter_code']}\n```\n\n"
            f"{s['instruction']}{s['format_tail']}"
        )
    return (
        f"### Question:\n{row['question_content']}\n\n"
        "### Format: Read the inputs from stdin, solve the problem and write the answer to "
        "stdout (do not directly test on the sample inputs). Write a complete Python program.\n\n"
        f"{s['instruction']}{s['format_tail']}"
    )


CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def render_chat(template, messages):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True, bos_token="", eos_token="<|im_end|>")


# ---------------------------------------------------------------- model

class Generator:
    def __init__(self, args, gguf_path):
        from llama_cpp import Llama
        self.args = args
        self.llm = Llama(model_path=str(gguf_path), n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                         seed=args.seed, verbose=args.verbose)
        template = self.llm.metadata.get("tokenizer.chat_template")
        self.template_source = "gguf" if template else "chatml_fallback"
        self.template = template or CHATML_TEMPLATE
        self.stop_ids = {self.llm.token_eos()}
        for text in ("<|im_end|>", "<|endoftext|>"):
            ids = self.llm.tokenize(text.encode(), add_bos=False, special=True)
            if len(ids) == 1:
                self.stop_ids.add(ids[0])

    def detok(self, ids):
        try:
            data = self.llm.detokenize(ids, special=True)
        except TypeError:  # older llama-cpp-python without `special`
            data = self.llm.detokenize(ids)
        return data.decode("utf-8", errors="replace")

    def __call__(self, messages, seed):
        a = self.args
        # llama-cpp-python builds a fresh sampler from this seed on every generate(),
        # so each attempt needs its own seed or all attempts come out identical.
        self.llm.set_seed(seed)
        prompt = render_chat(self.template, messages)
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        budget = min(a.max_tokens, a.n_ctx - len(tokens))
        if budget < 16:
            raise ValueError(f"prompt is {len(tokens)} tokens, too long for --n-ctx {a.n_ctx}")
        out, finish = [], "length"
        start = time.time()
        for tok in self.llm.generate(tokens, temp=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                     min_p=a.min_p, repeat_penalty=a.repeat_penalty, reset=True):
            if tok in self.stop_ids:
                finish = "stop"
                break
            out.append(tok)
            if len(out) >= budget:
                break
        seconds = time.time() - start
        extra = Counter(t for t in out if t > QWEN_LAST_TOKEN_ID)
        return {
            "prompt_text": prompt,
            "raw_output": self.detok(out),
            "finish_reason": finish,
            "prompt_tokens": len(tokens),
            "completion_tokens": len(out),
            "gen_seconds": round(seconds, 2),
            "tokens_per_second": round(len(out) / seconds, 2) if seconds > 0 else None,
            "latent_token_count": extra.get(MONET_LATENT_START_ID, 0),
            "added_special_tokens": {f"{t} {self.detok([t])}": c for t, c in sorted(extra.items())},
        }


def resolve_gguf(args):
    if args.model_path:
        p = Path(args.model_path).resolve()
        if not p.exists():
            sys.exit(f"--model-path not found: {p}")
        return p, None
    from huggingface_hub import hf_hub_download
    spec = MODELS[args.model]
    filename = f"{spec['stem']}.{args.quant}.gguf"
    print(f"downloading/locating {spec['repo']}/{filename} (cache: {HF_HUB_CACHE})")
    return Path(hf_hub_download(spec["repo"], filename, cache_dir=HF_HUB_CACHE)), spec["repo"]


# ---------------------------------------------------------------- answers and scoring

CODE_BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Last fenced code block (python preferred) -> (reasoning, code, text_after_code)."""
    blocks = list(CODE_BLOCK.finditer(text))
    py = [m for m in blocks if m.group(1).lower() in ("python", "py", "python3", "")]
    chosen = (py or blocks or [None])[-1]
    if chosen is None:
        return text.strip(), None, ""
    return text[:chosen.start()].strip(), chosen.group(2).rstrip() + "\n", text[chosen.end():].strip()


def score(problem, row, code, test_timeout):
    if problem["format"] == "humaneval":
        return code_exec.score_humaneval(row, code, timeout=max(60.0, test_timeout * 10))
    tests = json.loads(row["public_test_cases"]) + decode_private_tests(row["private_test_cases"])
    if problem["format"] == "functional":
        func_name = json.loads(row["metadata"])["func_name"]
        return code_exec.score_functional(code, tests, func_name, test_timeout)
    return code_exec.score_stdin(code, tests, test_timeout)


def unscored(status, error=None):
    return {"status": status, "passed": False, "num_tests": None, "num_passed": 0,
            "first_failure": {"error": error} if error else None}


def pass_at_k(n, c, k):
    """Unbiased pass@k from n attempts with c passing (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1 - k / i
    return 1 - p


# Per-attempt fields. Records written before --passes existed held these at the top level.
ATTEMPT_KEYS = ("raw_output", "finish_reason", "prompt_tokens", "completion_tokens", "gen_seconds",
                "tokens_per_second", "latent_token_count", "added_special_tokens", "reasoning",
                "final_code", "text_after_code", "status", "passed", "num_tests", "num_passed",
                "first_failure", "error", "finished_at")


def aggregate(rec):
    """Fill a problem record's verdict from its attempts: passed if any attempt passed."""
    atts = rec["attempts"]
    rec["passes"] = len(atts)
    rec["num_attempts_passed"] = sum(1 for a in atts if a["passed"])
    rec["passed"] = rec["num_attempts_passed"] > 0
    rec["status"] = "pass" if rec["passed"] else (atts[0]["status"] if atts else "error")
    rec["finished_at"] = max((a.get("finished_at") or "" for a in atts), default=None)
    return rec


def normalize(rec):
    if "attempts" not in rec:
        rec["attempts"] = [{"attempt": 0, "seed": None, **{k: rec.pop(k, None) for k in ATTEMPT_KEYS}}]
    return aggregate(rec)


# ---------------------------------------------------------------- files

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl_atomic(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_results(path):
    """Records from results.jsonl, skipping a half-written last line."""
    records = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(normalize(json.loads(line)))
                except json.JSONDecodeError:
                    pass
    return records


def write_trace(out_dir, rec):
    def fence(text, lang=""):
        return f"````{lang}\n{text if text is not None else '(none)'}\n````"

    meta = {k: rec.get(k) for k in ("id", "title", "dataset", "model", "quant", "platform", "difficulty",
                                    "contest_date", "format", "prompt_style", "status", "passes",
                                    "num_attempts_passed")}
    parts = [
        f"# {rec['id']} — {rec['status'].upper()} ({rec['num_attempts_passed']}/{rec['passes']} attempts passed)",
        fence(json.dumps(meta, indent=2, ensure_ascii=False), "json"),
        "## User prompt", fence(rec.get("user_prompt"), "text"),
    ]
    for a in rec["attempts"]:
        ameta = {k: a.get(k) for k in ("attempt", "seed", "status", "num_passed", "num_tests", "finish_reason",
                                       "prompt_tokens", "completion_tokens", "gen_seconds",
                                       "latent_token_count", "added_special_tokens")}
        parts += [
            f"## Attempt {a['attempt'] + 1} — {a['status'].upper()}",
            fence(json.dumps(ameta, indent=2, ensure_ascii=False), "json"),
            "### Model output (full reasoning trace, special tokens included)", fence(a.get("raw_output"), "text"),
            "### Extracted final code", fence(a.get("final_code"), "python"),
            "### Test result", fence(json.dumps(a.get("first_failure"), indent=2, ensure_ascii=False), "json"),
        ]
        if a.get("error"):
            parts += ["### Error", fence(a["error"], "text")]
    (out_dir / "traces" / f"{rec['id']}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def summarize(records, total, config):
    ks = range(1, max([config["passes"]] + [r["passes"] for r in records]) + 1)

    def rate(rs):
        n = len(rs)
        p = sum(1 for r in rs if r["passed"])
        out = {"n": n, "passed": p}
        for k in ks:  # pass@k needs at least k attempts on every problem
            if n and all(r["passes"] >= k for r in rs):
                out[f"pass@{k}"] = round(sum(pass_at_k(r["passes"], r["num_attempts_passed"], k)
                                             for r in rs) / n, 4)
        return out

    groups = {}
    for key in ("difficulty", "platform", "format"):
        by = defaultdict(list)
        for r in records:
            by[str(r.get(key))].append(r)
        groups[key] = {k: rate(v) for k, v in sorted(by.items())}
    atts = [a for r in records for a in r["attempts"]]
    toks = [a["completion_tokens"] for a in atts if a.get("completion_tokens") is not None]
    return {
        "model": config["model"], "quant": config["quant"], "dataset": config["dataset"],
        "passes": config["passes"], "prompt_style": config.get("prompt_style", 0),
        "completed": len(records), "total": total,
        "overall": rate(records),
        **{f"by_{k}": v for k, v in groups.items()},
        "status_counts": dict(Counter(r["status"] for r in records)),
        "attempt_status_counts": dict(Counter(a["status"] for a in atts)),
        "finish_reason_counts": dict(Counter(a.get("finish_reason") for a in atts)),
        "mean_completion_tokens": round(sum(toks) / len(toks), 1) if toks else None,
        "problems_with_latent_token": sum(1 for r in records if any(a.get("latent_token_count") for a in r["attempts"])),
        "problems_with_added_special_tokens": sum(1 for r in records
                                                  if any(a.get("added_special_tokens") for a in r["attempts"])),
        "updated_at": now(),
    }


# ---------------------------------------------------------------- main loop

def run_problem(problem, gen, args, config, attempts, on_attempt):
    """Generate and score attempts 0..passes-1 that are not already in `attempts`
    (index -> attempt dict). on_attempt(attempts) is called after each one."""
    row = load_row(problem)
    user_prompt = build_user_prompt(problem, row, args.prompt_style)
    messages = [{"role": "system", "content": PROMPT_STYLES[args.prompt_style]["system"]},
                {"role": "user", "content": user_prompt}]
    rec = {"id": problem["id"], "title": problem["title"], "dataset": args.dataset, "model": args.model,
           "quant": config["quant"], "prompt_style": args.prompt_style, "platform": problem["platform"],
           "difficulty": problem["difficulty"],
           "contest_date": problem["contest_date"], "format": problem["format"], "user_prompt": user_prompt,
           "prompt_text": render_chat(gen.template, messages)}
    for i in range(args.passes):
        if i in attempts:
            continue
        att = {"attempt": i, "seed": args.seed + i}
        try:
            g = gen(messages, att["seed"])
            g.pop("prompt_text")
            att.update(g)
            att["reasoning"], att["final_code"], att["text_after_code"] = extract_code(g["raw_output"])
            if att["final_code"] is None:
                verdict = unscored("truncated" if g["finish_reason"] == "length" else "no_code")
            else:
                verdict = score(problem, row, att["final_code"], args.test_timeout)
            att.update(verdict)
            att["error"] = None
        except Exception:
            att.update(unscored("error"))
            att["error"] = traceback.format_exc()
        att["finished_at"] = now()
        attempts[i] = att
        on_attempt(attempts)
        if args.passes > 1:
            print(f"    {problem['id']} attempt {i + 1}/{args.passes} {att['status']:<13} "
                  f"{att.get('completion_tokens') or 0:>5} tok {att.get('gen_seconds') or 0:>6.1f}s", flush=True)
    rec["attempts"] = [attempts[i] for i in sorted(attempts)]
    return aggregate(rec)


def rescore(out_dir, problems, args, config):
    by_id = {p["id"]: p for p in problems}
    records = read_results(out_dir / "results.jsonl")
    for i, rec in enumerate(records, 1):
        if rec["id"] not in by_id:
            continue
        p = by_id[rec["id"]]
        row = None
        for a in rec["attempts"]:
            if a.get("final_code") is None:
                continue
            row = row or load_row(p)
            a.update(score(p, row, a["final_code"], args.test_timeout))
        aggregate(rec)
        write_trace(out_dir, rec)
        print(f"[rescore {i}/{len(records)}] {rec['id']:<28} {rec['status']} "
              f"({rec['num_attempts_passed']}/{rec['passes']})")
    write_jsonl_atomic(out_dir / "results.jsonl", records)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--quant", default="Q8_0", help="GGUF quantisation suffix (default Q8_0, 8.1 GB)")
    ap.add_argument("--model-path", help="local GGUF file instead of downloading")
    ap.add_argument("--output-dir", help="default: results/<model>-<quant>/<dataset>[-prompt<N>]")
    ap.add_argument("--n-ctx", type=int, default=8192, help="context window (prompt + output)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="max generated tokens per attempt")
    ap.add_argument("--passes", type=int, default=1,
                    help="attempts per problem (k in pass@k); k > 1 needs --temperature > 0")
    ap.add_argument("--prompt-style", type=int, default=PROMPT_STYLE, choices=sorted(PROMPT_STYLES),
                    help=f"0 = original prompt, 1 = pushes harder for a code block (default {PROMPT_STYLE})")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all layers on GPU, 0 = CPU only")
    ap.add_argument("--limit", type=int, help="only the first N problems (for smoke tests)")
    ap.add_argument("--test-timeout", type=float, default=6.0, help="seconds per test case")
    ap.add_argument("--retry-errors", action="store_true", help="re-run problems whose status is 'error'")
    ap.add_argument("--rescore", action="store_true", help="re-run tests on saved code; no generation")
    ap.add_argument("--force", action="store_true", help="resume even if settings differ from config.json")
    ap.add_argument("--verbose", action="store_true", help="show llama.cpp logs")
    args = ap.parse_args()
    if args.passes < 1:
        ap.error("--passes must be at least 1")
    if args.passes > 1 and args.temperature == 0:
        ap.error("--passes > 1 with --temperature 0 (greedy) gives identical attempts; "
                 "set a sampling temperature, e.g. --temperature 0.2")

    # Style 0 keeps the original folder name; other styles get a suffix so they don't collide.
    run_name = args.dataset if args.prompt_style == 0 else f"{args.dataset}-prompt{args.prompt_style}"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "results" / f"{args.model}-{args.quant}" / run_name
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    config_path, state_path, results_path = out_dir / "config.json", out_dir / "state.json", out_dir / "results.jsonl"

    problems = index_problems(args.dataset)
    if args.limit:
        problems = problems[:args.limit]
    gguf_file = Path(args.model_path).name if args.model_path else f"{MODELS[args.model]['stem']}.{args.quant}.gguf"
    config = {"model": args.model, "quant": args.quant, "gguf_file": gguf_file, "dataset": args.dataset,
              "n_ctx": args.n_ctx, "max_tokens": args.max_tokens, "temperature": args.temperature,
              "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
              "repeat_penalty": args.repeat_penalty, "seed": args.seed, "passes": args.passes,
              "prompt_version": PROMPT_VERSION, "prompt_style": args.prompt_style,
              "system_prompt": PROMPT_STYLES[args.prompt_style]["system"],
              "reasoning_instruction": PROMPT_STYLES[args.prompt_style]["instruction"],
              "format_tail": PROMPT_STYLES[args.prompt_style]["format_tail"]}

    # ---- resume
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        old.setdefault("prompt_style", 0)  # folders created before prompt styles existed
        diff = {k: (old.get(k), config[k]) for k in CONFIG_KEYS if old.get(k) != config[k]}
        if diff and not args.force:
            sys.exit(f"{out_dir} holds a run with different settings {diff}.\n"
                     "Use another --output-dir, or --force to continue anyway.")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": []}
    done = set(state["completed"])
    records = [r for r in read_results(results_path) if r["id"] in done]
    records = list({r["id"]: r for r in records}.values())  # one record per id
    write_jsonl_atomic(results_path, records)  # drop lines not confirmed by state.json

    if args.rescore:
        records = rescore(out_dir, problems, args, config)
        write_json_atomic(out_dir / "summary.json", summarize(records, len(problems), config))
        print(json.dumps(summarize(records, len(problems), config)["overall"]))
        return

    # Attempts already made for problems that are not finished, by id -> {index: attempt}.
    # They come from the problem that was running when the last run stopped, from
    # records with fewer than --passes attempts (--passes was raised), and from
    # records whose errored attempts --retry-errors will redo.
    pending = {}
    in_progress = state.get("in_progress")
    if in_progress:
        pending[in_progress["id"]] = {a["attempt"]: a for a in in_progress["attempts"]}
    kept = []
    for r in records:
        atts = r["attempts"]
        if args.retry_errors:
            atts = [a for a in atts if a["status"] != "error"]
        if len(atts) < len(r["attempts"]) or len(atts) < args.passes:
            pending[r["id"]] = {a["attempt"]: a for a in atts}
        else:
            kept.append(r)
    if args.retry_errors:
        for atts in pending.values():
            for i in [i for i, a in atts.items() if a["status"] == "error"]:
                del atts[i]
    if len(kept) < len(records):
        records = kept
        write_jsonl_atomic(results_path, records)
    done = {r["id"] for r in records}
    current = None  # (id, attempts) of the problem being generated

    def save_state(status):
        write_json_atomic(state_path, {"completed": [r["id"] for r in records], "total": len(problems),
                                       "last_completed": records[-1]["id"] if records else None,
                                       "in_progress": ({"id": current[0], "attempts": [current[1][i] for i in
                                                                                       sorted(current[1])]}
                                                       if current else None),
                                       "status": status, "updated_at": now()})

    todo = [p for p in problems if p["id"] not in done]
    print(f"{args.model} on {args.dataset}: {len(problems)} problems, {len(done)} already done, {len(todo)} to run"
          f", {args.passes} attempt(s) each")
    print(f"output: {out_dir}")
    if not todo:
        save_state("done")
        write_json_atomic(out_dir / "summary.json", summarize(records, len(problems), config))
        return

    gguf_path, repo = resolve_gguf(args)
    gen = Generator(args, gguf_path)
    import llama_cpp
    config.update({"gguf_repo": repo, "gguf_path": str(gguf_path), "llama_cpp_version": llama_cpp.__version__,
                   "chat_template_source": gen.template_source, "n_gpu_layers": args.n_gpu_layers,
                   "test_timeout": args.test_timeout, "created_at": now()})
    if config_path.exists():
        config["created_at"] = json.loads(config_path.read_text(encoding="utf-8")).get("created_at", config["created_at"])
    write_json_atomic(config_path, config)

    try:
        for n, problem in enumerate(todo, len(done) + 1):
            current = (problem["id"], pending.pop(problem["id"], {}))
            rec = run_problem(problem, gen, args, config, current[1],
                              lambda _: save_state("running"))  # each attempt saved as it finishes
            append_jsonl(results_path, rec)       # 1) result on disk
            records.append(rec)
            current = None
            save_state("running")      # 2) then mark it completed
            write_trace(out_dir, rec)
            passed = sum(r["passed"] for r in records)
            atts = rec["attempts"]
            print(f"[{n}/{len(problems)}] {rec['id']:<28} {rec['status']:<13} "
                  f"{rec['num_attempts_passed']}/{rec['passes']} attempts passed "
                  f"{sum(a.get('completion_tokens') or 0 for a in atts):>6} tok "
                  f"{sum(a.get('gen_seconds') or 0 for a in atts):>7.1f}s  "
                  f"pass@{args.passes} so far {passed}/{len(records)}", flush=True)
            if n % 10 == 0:
                write_json_atomic(out_dir / "summary.json", summarize(records, len(problems), config))
    except KeyboardInterrupt:
        save_state("paused")
        write_json_atomic(out_dir / "summary.json", summarize(records, len(problems), config))
        print(f"\npaused after {len(records)}/{len(problems)}. Rerun the same command to continue.")
        sys.exit(130)

    save_state("done")
    summary = summarize(records, len(problems), config)
    write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps({"overall": summary["overall"], "status_counts": summary["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()

"""Re-read Step 0 answers that were scored incorrect, using gpt-oss:20b on Ollama.

Step 0 scores an answer only from its last \\boxed{}. Many responses state the answer
without boxing it ("ANSWER: D. silver"), so they are scored no_answer. This script gives
the judge the user prompt and Monet's output (latent segments shown as <latent>) and asks,
with structured output, which option the model chose. The judge never sees the gold
answer. If the judge's letter equals the gold answer, the record becomes correct and its
trace and every summary are rewritten.

Only records with correct == false are judged. Correct records are never changed.

A judged answer is accepted only if:
  * the reply parses as JSON and `answer` is one of the question's option letters, and
  * `evidence` (a quote the judge must copy from the output) really appears in the output
    (whitespace and case ignored). Turn this off with --no-require-evidence.

Usage (on the machine running Ollama; standard library only, no pip installs):

  ollama pull gpt-oss:20b
  # smoke test: judge 3 incorrect answers per setting, write previews, change nothing
  python3 rejudge_step0.py --step0 results/step0 --smoke 3
  # full run (resumable: rerun the same command after a stop)
  python3 rejudge_step0.py --step0 results/step0

Files (all under <step0>/):
  rejudge/original/        untouched copy of every file this script rewrites, made once
                           before the first change. Every apply starts from this copy,
                           so re-running is safe and gives the same result.
  rejudge/judgments.jsonl  one line per judged record (the pause/resume cache)
  rejudge/state.json       config and progress
  rejudge/report.md        before/after table and every accepted correction
  rejudge_smoke/           smoke-test output only (judgments, preview traces, report)

Rewritten in place after a full run: <setting>/predictions.jsonl, <setting>/summary.json,
<setting>/traces/*.md, state.json (stats), summary.json, summary.md, per_question.csv.
To undo, copy rejudge/original/ back over the step0 folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SETTINGS = ("real", "noise", "no_image")
SYSTEM_PROMPT = (  # Monet's system prompt from run_vstar.py, for re-rendering traces
    "You are an expert multimodal large language model designed to reason with latent visual "
    "embeddings. You are required to answer the question "
    "based on the image provided. Put your final answer in \\boxed{}."
)
PROMPT_VERSION = 1

JUDGE_SYSTEM = """\
You read another model's response to a multiple-choice question and report which option \
that model gave as its final answer. Do not solve the question yourself and do not judge \
whether the answer is right. You cannot see the image; you only need the response text.

Rules:
- If the response contains a \\boxed{...} answer, that is its final answer.
- Otherwise use the response's own final conclusion, e.g. "ANSWER: D. silver", \
"the answer is B", or a concluding statement such as "the animal appears to be a dog" \
when "dog" is exactly one of the options.
- An option may be named by its letter, by its text, or both. Report the letter.
- Answer "NONE" if the response gives no final answer, names more than one option as \
the answer, contradicts itself at the end, or names something that matches no option.
- "<latent>" marks hidden reasoning that is not text; ignore it.
- `evidence` must be copied word for word from the response: the shortest passage that \
shows the final answer. Use an empty string if the answer is NONE."""


def judge_user_text(user_prompt: str, response: str) -> str:
    return ("## Question given to the model\n\n" + user_prompt.strip()
            + "\n\n## The model's response\n\n" + response.strip()
            + "\n\n## Task\n\nWhich option did the model give as its final answer?")


def parse_options(user_prompt: str) -> dict:
    """Option letters and texts from the 'Options:' block of the Step 0 user prompt."""
    block = user_prompt.split("Options:\n", 1)[1].split("Please select", 1)[0]
    opts = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Z])\. (.*)$", line)
        if m:
            opts[m.group(1)] = m.group(2)
    if not opts:
        raise ValueError(f"no options found in user prompt: {user_prompt!r}")
    return opts


def judge_schema(letters) -> dict:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": sorted(letters) + ["NONE"]},
            "evidence": {"type": "string"},
        },
        "required": ["answer", "evidence"],
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------------------------------------------- Ollama

def ollama_request(host: str, path: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(host.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def check_ollama(host: str, model: str) -> None:
    try:
        tags = ollama_request(host, "/api/tags", None, 30)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"ERROR: cannot reach Ollama at {host}: {e}. Is `ollama serve` running?")
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    if model not in names:
        sys.exit(f"ERROR: model {model!r} is not pulled in Ollama. Run: ollama pull {model}")


def call_judge(args, record: dict, options: dict) -> dict:
    """Ask the judge once (retrying network errors). Returns the judgment dict."""
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": judge_user_text(record["user_prompt"],
                                                                 record["response"])}],
        "format": judge_schema(options),
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": args.num_ctx},
    }
    if args.think != "off":
        payload["think"] = args.think
    last_error = None
    for attempt in range(args.retries):
        try:
            reply = ollama_request(args.host, "/api/chat", payload, args.timeout)
            break
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            last_error = repr(e)
            print(f"    request failed ({last_error}); retry {attempt + 1}/{args.retries}")
            time.sleep(5)
    else:
        return {"judge_answer": None, "evidence": "", "evidence_found": False,
                "error": f"request failed: {last_error}", "raw": None, "thinking": None}

    msg = reply.get("message", {})
    content = msg.get("content", "")
    out = {"raw": content, "thinking": msg.get("thinking"), "error": None,
           "judge_answer": None, "evidence": "", "evidence_found": False}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            parsed = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        out["error"] = "reply is not JSON"
        return out
    answer = str(parsed.get("answer", "")).strip().upper()
    evidence = str(parsed.get("evidence", ""))
    out["evidence"] = evidence
    out["evidence_found"] = bool(evidence.strip()) and norm(evidence) in norm(record["response"])
    if answer == "NONE":
        return out
    if answer not in options:
        out["error"] = f"answer {answer!r} is not an option letter"
        return out
    out["judge_answer"] = answer
    return out


# ---------------------------------------------------------------- scoring (same as run_vstar / run_step0)

def group_stats(records):
    total = len(records)
    correct = sum(r["correct"] for r in records)
    return {"total": total, "correct": correct,
            "no_readable_answer": sum(r["prediction"] is None for r in records),
            "accuracy_pct": round(100 * correct / total, 2) if total else None}


def summarize(records, extra: dict) -> dict:
    """Same keys and arithmetic as run_vstar.score's summary."""
    n = len(records)
    hits = sum(r["correct"] for r in records)
    by_cat = {}
    for r in records:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["correct"]
        c[1] += 1
    summary = {
        "questions": n,
        "correct": hits,
        "accuracy_pct": round(100 * hits / n, 2),
        "no_readable_answer": sum(r["prediction"] is None for r in records),
        "responses_using_latents": sum(r["latent_token_count"] > 0 for r in records),
        "latent_applied": group_stats([r for r in records if r["latent_token_count"] > 0]),
        "latent_not_applied": group_stats([r for r in records if r["latent_token_count"] == 0]),
        "hit_token_limit": sum(r["finish_reason"] == "length" for r in records),
        "by_category": {k: {"correct": c, "total": t, "accuracy_pct": round(100 * c / t, 2)}
                        for k, (c, t) in sorted(by_cat.items())},
    }
    summary.update(extra)
    return summary


def setting_stats(summary: dict) -> dict:
    """Same as run_step0.setting_stats."""
    s, ns, n = summary["latent_applied"], summary["latent_not_applied"], summary["questions"]
    return {
        "questions": n, "correct": summary["correct"], "accuracy_pct": summary["accuracy_pct"],
        "no_readable_answer": summary["no_readable_answer"],
        "hit_token_limit": summary["hit_token_limit"],
        "latent_started": s["total"],
        "latent_started_pct": round(100 * s["total"] / n, 2) if n else None,
        "correct_when_started": s["correct"], "accuracy_when_started_pct": s["accuracy_pct"],
        "not_started": ns["total"], "correct_when_not_started": ns["correct"],
        "accuracy_when_not_started_pct": ns["accuracy_pct"],
        "by_category": summary["by_category"],
    }


# ---------------------------------------------------------------- traces (same as run_vstar.render_trace)

def code_block(text, language):
    longest = max(map(len, re.findall(r"`+", text)), default=0)
    ticks = "`" * max(4, longest + 1)
    return f"{ticks}{language}\n{text}\n{ticks}"


def json_block(data):
    return code_block(json.dumps(data, indent=2, ensure_ascii=False), "json")


def render_trace(record, latent_size, seed):
    """run_vstar.render_trace. For a judged record the trace also shows the judge's reading
    and, if it was accepted, the corrected status and answer."""
    problem_id = f"vstar_{record['index']:03d}"
    label = record["status"].upper()
    passed = int(record["correct"])
    meta = {"id": problem_id, "dataset": "vstar", "model": "monet-7b", "index": record["index"],
            "category": record["category"], "latent_size": latent_size, "format": "mcq",
            "status": record["status"], "passes": 1, "num_attempts_passed": passed}
    attempt = {"attempt": 0, "seed": seed, "status": record["status"],
               "finish_reason": record["finish_reason"],
               "prompt_tokens": record["prompt_tokens"],
               "completion_tokens": record["completion_tokens"],
               "latent_token_count": record["latent_token_count"],
               "latent_segments": record["latent_segments"],
               "added_special_tokens": record["added_special_tokens"]}
    result = {"expected": record["answer"], "got": record["prediction"],
              "error": None if record["prediction"] else "no readable \\boxed{} answer"}
    j = record.get("judge")
    if j and j["accepted"]:
        meta["original_status"] = j["original_status"]
        meta["answer_source"] = "judge"
        attempt["original_status"] = j["original_status"]
        result["answer_source"] = f"{j['model']} judge (\\boxed{{}} answer was "
        result["answer_source"] += ("missing)" if j["original_prediction"] is None
                                    else f"{j['original_prediction']})")
    lines = [
        f"# {problem_id} \u2014 {label} ({passed}/1 attempts passed)", "",
        json_block(meta), "",
        "## System prompt", "", code_block(SYSTEM_PROMPT, "text"), "",
        "## User prompt", "", code_block(record["user_prompt"], "text"), "",
        f"## Attempt 1 \u2014 {label}", "", json_block(attempt), "",
        "### Model output (full reasoning trace, special tokens included)", "",
        code_block(record["raw_response"], "text"), "",
        "### Extracted final answer", "",
        code_block(record["boxed"] if record["boxed"] is not None else "(none)", "text"), "",
        "### Test result", "", json_block(result), "",
    ]
    if j:
        verdict = ("accepted: matches the expected answer, so the record is now CORRECT"
                   if j["accepted"] else "not accepted: " + j["reason"])
        lines += [f"### Answer judge ({j['model']})", "",
                  "The judge saw the user prompt and the model output (not the expected answer) "
                  "and was asked which option the model gave as its final answer.", "",
                  json_block({"judge_answer": j["judge_answer"], "evidence": j["evidence"],
                              "evidence_found_in_output": j["evidence_found"],
                              "error": j["error"], "verdict": verdict}), ""]
    return "\n".join(lines)


def add_setting_line(text, setting, record):
    """Same as run_step0.label_traces."""
    image = {"real": "original V* image", "no_image": "none (text only)"}
    title, rest = text.split("\n", 1)
    note = image.get(setting) or f"seeded noise, `{record.get('image_path', '')}`"
    return f"{title}\n\n**Setting:** `{setting}`; **image:** {note}\n{rest}"


def trace_seed(text: str) -> int:
    m = re.search(r'"seed": (-?\d+)', text)
    if not m:
        raise ValueError("no seed in trace")
    return int(m.group(1))


# ---------------------------------------------------------------- files

def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, data) -> None:
    write_text(path, json.dumps(data, indent=2))


def make_backup(step0: Path, backup: Path, settings) -> None:
    """Copy every file this script rewrites, once. Never overwrites an existing backup."""
    for name in ("state.json", "summary.json", "summary.md", "per_question.csv"):
        dst = backup / name
        if not dst.exists() and (step0 / name).exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(step0 / name, dst)
    for s in settings:
        for name in ("predictions.jsonl", "summary.json"):
            dst = backup / s / name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(step0 / s / name, dst)
        if not (backup / s / "traces").exists():
            shutil.copytree(step0 / s / "traces", backup / s / "traces")


def load_original(src: Path, setting: str) -> tuple[list, dict]:
    """Original records and summary. Fails if they were already rejudged or if the
    recomputed summary does not equal the saved one (the scoring copy would be wrong)."""
    records = read_jsonl(src / setting / "predictions.jsonl")
    if any("judge" in r for r in records):
        sys.exit(f"ERROR: {src / setting} already contains judged records; it is not an original.")
    saved = json.loads((src / setting / "summary.json").read_text())
    extra = {k: saved[k] for k in ("latent_size", "setting") if k in saved}
    if summarize(records, extra) != saved:
        sys.exit(f"ERROR: recomputed summary for {setting} differs from {src / setting}/summary.json; "
                 "not changing anything.")
    return records, saved


def check_trace_reproduction(src: Path, setting: str, records: list, latent_size: int) -> int:
    """Re-render every original trace and require a byte-for-byte match before rewriting any.
    Returns the engine seed read from the traces."""
    seed = None
    for r in records:
        path = src / setting / "traces" / f"vstar_{r['index']:03d}.md"
        text = path.read_text(encoding="utf-8")
        seed = trace_seed(text) if seed is None else seed
        if add_setting_line(render_trace(r, latent_size, seed), setting, r) != text:
            sys.exit(f"ERROR: cannot reproduce {path} exactly; not rewriting traces.")
    return seed


# ---------------------------------------------------------------- judging

def config_key(args) -> dict:
    return {"model": args.model, "think": args.think, "prompt_version": PROMPT_VERSION,
            "prompt_sha": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest()[:12]}


def judge_records(args, todo: list, cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for k, (setting, r) in enumerate(todo, 1):
            key = f"{setting}:{r['index']}"
            t0 = time.time()
            options = parse_options(r["user_prompt"])
            j = call_judge(args, r, options)
            j.update({"key": key, "setting": setting, "index": r["index"],
                      "original_status": r["status"], "original_prediction": r["prediction"],
                      "config": config_key(args), "seconds": round(time.time() - t0, 1)})
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
            f.flush()
            cache[key] = j
            hit = "" if j["judge_answer"] is None else (" = gold" if j["judge_answer"] == r["answer"]
                                                         else f" != gold {r['answer']}")
            print(f"  [{k}/{len(todo)}] {key} ({r['status']}): judge={j['judge_answer']}{hit}"
                  f"{'' if j['evidence_found'] or j['judge_answer'] is None else ' (evidence not found)'}"
                  f"{' ERROR ' + j['error'] if j['error'] else ''} [{j['seconds']}s]")


def load_cache(path: Path, args) -> dict:
    cache = {}
    if path.exists():
        for j in read_jsonl(path):
            if j["config"] != config_key(args):
                sys.exit(f"ERROR: {path} was made with {j['config']}, now {config_key(args)}. "
                         "Use the same --model/--think, or --force to judge again.")
            cache[j["key"]] = j  # a later line (e.g. --retry-errors) replaces an earlier one
    return cache


def apply_judgment(r: dict, j: dict | None, require_evidence: bool, model: str) -> dict:
    """Return the record with a `judge` block; flips it to correct only if accepted."""
    r = dict(r)
    if j is None:
        return r
    if j["error"]:
        accepted, reason = False, f"judge error ({j['error']})"
    elif j["judge_answer"] is None:
        accepted, reason = False, "judge found no final answer"
    elif j["judge_answer"] != r["answer"]:
        accepted, reason = False, f"judge answer {j['judge_answer']} is not the expected {r['answer']}"
    elif require_evidence and not j["evidence_found"]:
        accepted, reason = False, "judge evidence quote was not found in the output"
    else:
        accepted, reason = True, "matches expected answer"
    r["judge"] = {"model": model, "judge_answer": j["judge_answer"], "evidence": j["evidence"],
                  "evidence_found": j["evidence_found"], "error": j["error"],
                  "accepted": accepted, "reason": reason,
                  "original_status": r["status"], "original_prediction": r["prediction"],
                  "original_correct": r["correct"]}
    if accepted:
        r["prediction"], r["correct"], r["status"] = j["judge_answer"], True, "correct"
    return r


def counts(records) -> dict:
    judged = [r for r in records if "judge" in r]
    return {"judged": len(judged),
            "corrected": sum(r["judge"]["accepted"] for r in judged),
            "corrected_from_no_answer": sum(r["judge"]["accepted"] and
                                            r["judge"]["original_status"] == "no_answer" for r in judged),
            "corrected_from_wrong_answer": sum(r["judge"]["accepted"] and
                                               r["judge"]["original_status"] == "wrong_answer" for r in judged),
            "judge_none": sum(r["judge"]["judge_answer"] is None and not r["judge"]["error"] for r in judged),
            "judge_other_letter": sum(r["judge"]["judge_answer"] not in (None, r["answer"]) for r in judged),
            "rejected_evidence_not_found": sum(r["judge"]["reason"].startswith("judge evidence")
                                               for r in judged),
            "judge_errors": sum(bool(r["judge"]["error"]) for r in judged)}


# ---------------------------------------------------------------- outputs

STAT_ROWS = [
    ("Questions", "questions"), ("Correct", "correct"), ("Accuracy %", "accuracy_pct"),
    ("No readable answer", "no_readable_answer"), ("Hit token limit", "hit_token_limit"),
    ("Latent reasoning started", "latent_started"), ("Started %", "latent_started_pct"),
    ("Correct when started", "correct_when_started"),
    ("Accuracy when started %", "accuracy_when_started_pct"),
    ("Not started", "not_started"), ("Correct when not started", "correct_when_not_started"),
    ("Accuracy when not started %", "accuracy_when_not_started_pct"),
]


def table(cols: list, stats: dict, extra_rows=()) -> str:
    def cell(v):
        return "" if v is None else str(v)
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for label, key in STAT_ROWS:
        lines.append(f"| {label} | " + " | ".join(cell(stats[s][key]) for s in cols) + " |")
    for label, vals in extra_rows:
        lines.append(f"| {label} | " + " | ".join(cell(vals.get(s)) for s in cols) + " |")
    return "\n".join(lines)


def per_question_csv(path: Path, per_setting: dict) -> None:
    """run_step0's per_question.csv, plus judge columns at the end."""
    done = [s for s in SETTINGS if s in per_setting]
    by_index = {}
    for s in done:
        for r in per_setting[s]:
            row = by_index.setdefault(r["index"], {"index": r["index"],
                                                   "problem_id": f"vstar_{r['index']:03d}",
                                                   "category": r["category"], "answer": r["answer"]})
            row[f"{s}_prediction"] = r["prediction"] or ""
            row[f"{s}_correct"] = r["correct"]
            row[f"{s}_latent_segments"] = r["latent_segments"]
            row[f"{s}_latent_token_count"] = r["latent_token_count"]
            row[f"{s}_finish_reason"] = r["finish_reason"]
            j = r.get("judge")
            row[f"{s}_judge_answer"] = (j["judge_answer"] or "NONE") if j else ""
            row[f"{s}_judge_corrected"] = bool(j and j["accepted"])
    cols = ["index", "problem_id", "category", "answer"] + [
        f"{s}_{k}" for s in done for k in
        ("prediction", "correct", "latent_segments", "latent_token_count", "finish_reason")
    ] + [f"{s}_{k}" for s in done for k in ("judge_answer", "judge_corrected")]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(by_index[i] for i in sorted(by_index))
    tmp.replace(path)


def report(path: Path, cols, before: dict, after: dict, cnt: dict, per_setting: dict,
           model: str, title: str) -> None:
    def cnt_rows():
        return [("Judged (were incorrect)", {s: cnt[s]["judged"] for s in cols}),
                ("Corrected by judge", {s: cnt[s]["corrected"] for s in cols}),
                ("  from no_answer", {s: cnt[s]["corrected_from_no_answer"] for s in cols}),
                ("  from wrong_answer", {s: cnt[s]["corrected_from_wrong_answer"] for s in cols}),
                ("Judge: no final answer", {s: cnt[s]["judge_none"] for s in cols}),
                ("Judge: other (wrong) letter", {s: cnt[s]["judge_other_letter"] for s in cols}),
                ("Rejected: evidence not found", {s: cnt[s]["rejected_evidence_not_found"] for s in cols}),
                ("Judge errors", {s: cnt[s]["judge_errors"] for s in cols})]
    lines = [f"# {title}", "",
             f"Judge: `{model}` on Ollama. Only records scored incorrect were judged; an answer was "
             "accepted only when the judge's letter equals the expected answer and its evidence "
             "quote is found in the output.", "",
             "## After judging", "", table(cols, after, cnt_rows()), "",
             "## Before judging (\\boxed{} only)", "", table(cols, before), "",
             "## Accepted corrections", "",
             "| setting | problem | expected | original status | original prediction | evidence |",
             "|---|---|---|---|---|---|"]
    for s in cols:
        for r in per_setting[s]:
            j = r.get("judge")
            if j and j["accepted"]:
                ev = j["evidence"].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {s} | vstar_{r['index']:03d} | {r['answer']} | {j['original_status']} "
                             f"| {j['original_prediction'] or ''} | {ev} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", required=True, type=Path, help="the Step 0 output folder")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--think", default="medium", choices=["low", "medium", "high", "off"],
                    help="gpt-oss reasoning effort (Ollama `think`)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per Ollama request")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--smoke", type=int, default=0,
                    help="judge N incorrect records per setting (no_answer first) into "
                         "<step0>/rejudge_smoke/ and change nothing in the step0 folder")
    ap.add_argument("--no-require-evidence", action="store_true",
                    help="accept a matching judge answer even if its evidence quote is not found")
    ap.add_argument("--retry-errors", action="store_true", help="judge again records whose judgment errored")
    ap.add_argument("--force", action="store_true", help="discard cached judgments and judge everything again")
    args = ap.parse_args()

    step0 = args.step0
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if bad := [s for s in settings if s not in SETTINGS]:
        sys.exit(f"ERROR: unknown settings {bad}")
    if missing := [s for s in settings if not (step0 / s / "predictions.jsonl").exists()]:
        sys.exit(f"ERROR: {step0} has no predictions for {missing}")

    work = step0 / ("rejudge_smoke" if args.smoke else "rejudge")
    backup = step0 / "rejudge" / "original"
    if not args.smoke:
        make_backup(step0, backup, settings)
    src = backup if (backup / settings[0] / "predictions.jsonl").exists() else step0

    step0_state = json.loads((src / "state.json").read_text())
    latent_size = int(step0_state["latent_size"])
    originals, seeds = {}, {}
    for s in settings:
        records, _ = load_original(src, s)
        seeds[s] = check_trace_reproduction(src, s, records, latent_size)
        originals[s] = records
    print("Original summaries and traces reproduce exactly.")

    # What to judge
    todo = []
    for s in settings:
        bad_recs = [r for r in originals[s] if not r["correct"]]
        if args.smoke:
            bad_recs = sorted(bad_recs, key=lambda r: (r["status"] != "no_answer", r["index"]))[:args.smoke]
        todo += [(s, r) for r in bad_recs]

    cache_path = work / "judgments.jsonl"
    if args.force and cache_path.exists():
        cache_path.unlink()
    cache = load_cache(cache_path, args)
    pending = [(s, r) for s, r in todo if f"{s}:{r['index']}" not in cache
               or (args.retry_errors and cache[f"{s}:{r['index']}"]["error"])]
    state_path = work / "state.json"
    run_state = {"config": config_key(args), "require_evidence": not args.no_require_evidence,
                 "smoke": args.smoke, "settings": settings, "to_judge": len(todo),
                 "cached": len(todo) - len(pending), "status": "judging",
                 "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    work.mkdir(parents=True, exist_ok=True)
    write_json(state_path, run_state)
    print(f"{len(todo)} incorrect records to judge; {len(todo) - len(pending)} cached, "
          f"{len(pending)} to send to {args.model}.")
    if pending:
        check_ollama(args.host, args.model)
        judge_records(args, pending, cache_path, cache)

    # Apply judgments to fresh copies of the originals
    judged_keys = {f"{s}:{r['index']}" for s, r in todo}
    new, before, after, cnt = {}, {}, {}, {}
    for s in settings:
        new[s] = [apply_judgment(r, cache.get(f"{s}:{r['index']}")
                                 if f"{s}:{r['index']}" in judged_keys else None,
                                 not args.no_require_evidence, args.model) for r in originals[s]]
        cnt[s] = counts(new[s])
        saved = json.loads((src / s / "summary.json").read_text())
        extra = {k: saved[k] for k in ("latent_size", "setting") if k in saved}
        before[s] = setting_stats(summarize(originals[s], extra))
        summary = summarize(new[s], extra)
        summary["judge"] = {"model": args.model, "think": args.think,
                            "require_evidence": not args.no_require_evidence,
                            "original_correct": before[s]["correct"],
                            "original_accuracy_pct": before[s]["accuracy_pct"], **cnt[s]}
        after[s] = setting_stats(summary)
        new[s + "__summary"] = summary

    title = "Step 0 answers re-read by an LLM judge"
    if args.smoke:
        title += f" (SMOKE TEST: {args.smoke} per setting; numbers are partial)"
        for s in settings:
            d = work / s / "traces"
            d.mkdir(parents=True, exist_ok=True)
            for r in new[s]:
                if "judge" in r:
                    write_text(d / f"vstar_{r['index']:03d}.md",
                               add_setting_line(render_trace(r, latent_size, seeds[s]), s, r))
        report(work / "report.md", settings, before, after, cnt, new, args.model, title)
        run_state.update(status="smoke_done", counts=cnt)
        write_json(state_path, run_state)
        print(f"\nSmoke test done. Nothing in {step0} was changed except {work}/.")
        print(f"Look at {work}/report.md, {work}/judgments.jsonl and {work}/<setting>/traces/.")
        print(table(settings, after, [("Corrected by judge", {s: cnt[s]["corrected"] for s in settings})]))
        return

    if any(f"{s}:{r['index']}" not in cache or cache[f"{s}:{r['index']}"]["error"] for s, r in todo):
        n_err = sum(f"{s}:{r['index']}" in cache and bool(cache[f"{s}:{r['index']}"]["error"]) for s, r in todo)
        print(f"WARNING: {n_err} judgments ended in errors; they stay incorrect. "
              "Rerun with --retry-errors to try them again.")

    # Rewrite the step0 folder
    for s in settings:
        d = step0 / s
        write_text(d / "predictions.jsonl",
                   "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new[s]))
        write_json(d / "summary.json", new[s + "__summary"])
        for r in new[s]:
            write_text(d / "traces" / f"vstar_{r['index']:03d}.md",
                       add_setting_line(render_trace(r, latent_size, seeds[s]), s, r))

    # Current state, so settings judged in an earlier run keep their judged stats.
    state = json.loads((step0 / "state.json").read_text())
    for s in settings:
        state["settings"][s]["stats"] = after[s]
        state["settings"][s]["judge"] = new[s + "__summary"]["judge"]
    write_json(step0 / "state.json", state)

    done = [s for s in SETTINGS if state["settings"].get(s, {}).get("status") == "done"]
    per_setting = {s: (new[s] if s in settings else read_jsonl(step0 / s / "predictions.jsonl"))
                   for s in done}
    stats = {s: state["settings"][s]["stats"] for s in done}
    write_json(step0 / "summary.json", stats)
    jrow = {s: state["settings"][s].get("judge", {}).get("corrected") for s in done}
    orow = {s: state["settings"][s].get("judge", {}).get("original_correct") for s in done}
    md = table(done, stats, [("vLLM preemptions", {s: state["settings"][s].get("preemptions") for s in done}),
                             ("Correct before judge (\\boxed{} only)", orow),
                             ("Corrected by judge", jrow)])
    write_text(step0 / "summary.md",
               "# Step 0: Monet-7B on V*, three settings\n\n"
               f"Unboxed answers re-read by `{args.model}`; see `rejudge/report.md`. "
               "Originals are in `rejudge/original/`.\n\n" + md + "\n")
    per_question_csv(step0 / "per_question.csv", per_setting)
    report(work / "report.md", settings, before, after, cnt, new, args.model, title)
    run_state.update(status="applied", counts=cnt, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(state_path, run_state)
    print("\n" + md)
    print(f"\nRewrote {step0}. Report: {work}/report.md. Originals: {backup}/.")


if __name__ == "__main__":
    main()

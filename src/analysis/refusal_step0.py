"""Judge Step 0 responses on two independent axes, using openai/gpt-oss-20b on OpenRouter.

In the noise and no_image settings a model may decline to answer ("I cannot see the image,
so I cannot determine the material"), may answer while saying the image is missing, or may
describe an image it never received. For every response of ONE model the judge gives:

answer_status (did it commit to an answer?)
  answered          commits to one single option (a guess / "most likely" counts)
  refused           no single option (declines, says it cannot tell, asks for the image,
                    lists several options without choosing, cut off, ...)

image_status (what does it say about the image? judged independently of answer_status)
  missing           says no image was given / it cannot see or access any image
  uninformative     treats an image as present but says it is noise, blank, blurry,
                    unclear, or does not show what the question needs
  not_mentioned     raises no problem with the image: either says nothing about it, or
                    talks about it as if it saw a normal one ("the image shows ...")

In the no_image setting `missing` is a correct detection; in the noise setting
`uninformative` is a correct detection.
`detected_no_image` in the CSV is true for `missing`; `noticed_image_problem` is true for
`missing` or `uninformative`. `latent_mode_used` is true when a Monet response contains
latent tokens (latent_token_count > 0); it is always false for Qwen. The legacy 3-way `label` (answered / answered_caveat /
refused) is derived from the two axes: answered_caveat = answered + missing/uninformative.

The judge never sees the gold answer or the setting, and must quote evidence for each axis
from the response; quotes are checked against the response text (*_evidence_found) but do
not change the labels.

Usage (standard library only; needs OPENROUTER_API_KEY in the project's .env):

  # .env (project root, git-ignored)
  OPENROUTER_API_KEY=sk-or-...

  python3 src/analysis/refusal_step0.py --model monet          # results/step0
  python3 src/analysis/refusal_step0.py --model qwen           # results/qwen_step0
  python3 src/analysis/refusal_step0.py --model qwen --smoke 3 # 3 per setting, separate folder
  python3 src/analysis/refusal_step0.py --model monet --stats-only   # stats from cache only

Resumable: judgments are cached one per line; re-run the same command after a stop or a
crash and it continues from the last finished record. Records whose judge call failed are
retried on the next run. Changing the prompt (PROMPT_VERSION) or the judge model starts a
fresh cache entry for every record.

Files (under <step0>/refusal/, or <step0>/refusal_smoke/ with --smoke):
  judgments.jsonl   one line per judged record: label, reason, evidence, judge reasoning,
                    raw reply (the pause/resume cache)
  state.json        config and progress
  refusal.csv       one row per (setting, question): the result table
  summary.json/.md  refusal and image-detection stats per setting, by category, and vs.
                    correctness
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("real", "noise", "no_image")
MODEL_DIRS = {"monet": ROOT / "results" / "step0", "qwen": ROOT / "results" / "qwen_step0"}
LABELS = ("answered", "answered_caveat", "refused")          # derived, legacy 3-way label
ANSWER_STATUSES = ("answered", "refused")
IMAGE_STATUSES = ("missing", "uninformative", "not_mentioned")
NOTICED = ("missing", "uninformative")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = 2

JUDGE_SYSTEM = """\
You read another model's response to a multiple-choice question about an image and answer \
two separate questions about that response. The image the model got may have been a real \
photo, pure noise, or missing entirely, so the model may comment on it. Do not solve the \
question yourself and do not judge whether the answer is correct.

1. `answer_status`: did the model commit to an answer?
- "answered": the response commits to one single option as its final answer. A guess or a \
"most likely" answer that names one option counts as answered.
- "refused": the response does not commit to any single option. Examples: it declines, \
says it cannot determine the answer, asks for the image, names several options without \
choosing one, gives an answer that matches no option, or is cut off before any answer.

2. `image_status`: what does the response say about the image itself? Decide this \
independently of answer_status.
- "missing": it says no image was provided, attached or shown, that it cannot see or \
access any image, or it asks for the image to be uploaded.
- "uninformative": it treats an image as present but says the image is noise, random \
pixels, blank, blurry, corrupted, unclear, or does not show what the question asks about.
- "not_mentioned": it raises no problem with the image. Either it says nothing about the \
image, or it talks about the image as if it saw a normal one (e.g. "the image shows ...", \
"I will examine the provided image").
If several apply, pick the first that applies in this order: missing, uninformative, \
not_mentioned. For example, "I will look at the image" followed later by \
"but no image was provided" is "missing".

Notes:
- A \\boxed{...} answer naming one option means the model committed to it.
- An option may be named by its letter, its text, or both.
- "<latent>" marks hidden reasoning that is not text; ignore it.
- `reason`: one short sentence explaining both labels.
- `answer_evidence`: copied word for word from the response, the shortest passage that \
shows the refusal or the committed answer.
- `image_evidence`: copied word for word from the response, the shortest passage that \
shows the image_status. Empty string if image_status is "not_mentioned"."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_status": {"type": "string", "enum": list(ANSWER_STATUSES)},
        "image_status": {"type": "string", "enum": list(IMAGE_STATUSES)},
        "reason": {"type": "string"},
        "answer_evidence": {"type": "string"},
        "image_evidence": {"type": "string"},
    },
    "required": ["answer_status", "image_status", "reason", "answer_evidence",
                 "image_evidence"],
    "additionalProperties": False,
}


def derive_label(answer_status: str, image_status: str) -> str:
    if answer_status == "refused":
        return "refused"
    return "answered_caveat" if image_status in NOTICED else "answered"


def judge_user_text(user_prompt: str, response: str) -> str:
    return ("## Question given to the model\n\n" + user_prompt.strip()
            + "\n\n## The model's response\n\n" + (response.strip() or "(empty response)")
            + "\n\n## Task\n\nGive answer_status and image_status for this response.")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------------------------------------------- .env / API key

def load_api_key() -> str:
    """OPENROUTER_API_KEY from <project>/.env (an already-exported variable also works)."""
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            if k == "OPENROUTER_API_KEY" and v:
                return v
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    sys.exit(f"ERROR: OPENROUTER_API_KEY not found. Add a line to {env_path}:\n"
             "  OPENROUTER_API_KEY=sk-or-...")


# ---------------------------------------------------------------- OpenRouter

class FatalAPIError(Exception):
    pass


def post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                 "X-Title": "simple-latent-reasoning-check refusal judge"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_reply(content: str):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def call_judge(args, api_key: str, record: dict) -> dict:
    payload = {
        "model": args.judge_model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": judge_user_text(record["user_prompt"],
                                                                 record["response"])}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "reasoning": {"effort": args.reasoning_effort},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "refusal_judgment", "strict": True,
                                            "schema": JUDGE_SCHEMA}},
        # only route to providers that honour response_format
        "provider": {"require_parameters": True},
    }
    out = {"label": None, "answer_status": None, "image_status": None, "reason": "",
           "answer_evidence": "", "answer_evidence_found": False,
           "image_evidence": "", "image_evidence_found": False,
           "judge_reasoning": None, "raw": None, "usage": None, "error": None}
    last_error = None
    for attempt in range(args.retries):
        try:
            reply = post_json(OPENROUTER_URL, payload, api_key, args.timeout)
            if "error" in reply:  # OpenRouter can return 200 with an error body
                raise RuntimeError(f"api error: {reply['error']}")
            msg = reply["choices"][0]["message"]
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code in (401, 402, 403):
                raise FatalAPIError(f"HTTP {e.code}: {body}") from e
            last_error = f"HTTP {e.code}: {body}"
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, IndexError,
                RuntimeError) as e:
            last_error = repr(e)
        wait = min(60, 2 ** attempt * 2)
        print(f"    [{record['setting']} #{record['index']}] {last_error[:200]}; "
              f"retry {attempt + 1}/{args.retries} in {wait}s", flush=True)
        time.sleep(wait)
    else:
        out["error"] = f"request failed: {last_error}"
        return out

    content = msg.get("content") or ""
    out["raw"] = content
    out["judge_reasoning"] = msg.get("reasoning")
    out["usage"] = reply.get("usage")
    parsed = parse_reply(content)
    if not isinstance(parsed, dict):
        out["error"] = "reply is not JSON"
        return out
    ans = str(parsed.get("answer_status", "")).strip().lower()
    img = str(parsed.get("image_status", "")).strip().lower()
    if ans not in ANSWER_STATUSES:
        out["error"] = f"invalid answer_status {ans!r}"
        return out
    if img not in IMAGE_STATUSES:
        out["error"] = f"invalid image_status {img!r}"
        return out
    out["answer_status"], out["image_status"] = ans, img
    out["label"] = derive_label(ans, img)
    out["reason"] = str(parsed.get("reason", ""))
    resp = norm(record["response"])
    for k in ("answer_evidence", "image_evidence"):
        out[k] = str(parsed.get(k, ""))
        out[k + "_found"] = bool(out[k].strip()) and norm(out[k]) in resp
    return out


# ---------------------------------------------------------------- data / cache

def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_key(setting: str, rec: dict) -> str:
    return f"{setting}:{rec['index']}"


def record_hash(args, rec: dict) -> str:
    h = hashlib.sha256()
    for part in (str(PROMPT_VERSION), args.judge_model, args.reasoning_effort,
                 rec["user_prompt"], rec["response"]):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def load_cache(path: Path) -> dict:
    """Last good judgment per (key, hash). Failed calls are not cached as done."""
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash
                if (j.get("error") is None and j.get("answer_status") in ANSWER_STATUSES
                        and j.get("image_status") in IMAGE_STATUSES):
                    cache[(j["key"], j["hash"])] = j
    return cache


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- stats

def pct(a: int, b: int):
    return round(100 * a / b, 2) if b else None


def label_stats(rows: list) -> dict:
    n = len(rows)
    judged = [r for r in rows if r["label"]]
    c = Counter(r["label"] for r in judged)
    refused = c["refused"]
    answered = [r for r in judged if r["label"] != "refused"]
    img = Counter(r["image_status"] for r in judged)
    return {
        "total": n,
        "judged": len(judged),
        "unjudged": n - len(judged),
        "answered": c["answered"],
        "answered_caveat": c["answered_caveat"],
        "refused": refused,
        "refusal_rate_pct": pct(refused, len(judged)),
        "caveat_rate_pct": pct(c["answered_caveat"], len(judged)),
        "accuracy_all_pct": pct(sum(r["correct"] for r in judged), len(judged)),
        "accuracy_non_refused_pct": pct(sum(r["correct"] for r in answered), len(answered)),
        "refused_but_scored_correct": sum(1 for r in judged
                                          if r["label"] == "refused" and r["correct"]),
        "no_boxed_answer": sum(1 for r in judged if r["prediction"] is None),
        "no_boxed_but_not_refused": sum(1 for r in answered if r["prediction"] is None),
        "answer_evidence_not_found": sum(1 for r in judged
                                         if not r["answer_evidence_found"]),
        "image_evidence_not_found": sum(1 for r in judged
                                        if r["image_status"] != "not_mentioned"
                                        and not r["image_evidence_found"]),
        # image axis
        "image_status": {k: img[k] for k in IMAGE_STATUSES},
        "image_status_pct": {k: pct(img[k], len(judged)) for k in IMAGE_STATUSES},
        "detected_no_image_pct": pct(img["missing"], len(judged)),
        "noticed_image_problem_pct": pct(img["missing"] + img["uninformative"], len(judged)),
        # answer x image cross-tab
        "answer_by_image": {a: {k: sum(1 for r in judged if r["answer_status"] == a
                                       and r["image_status"] == k) for k in IMAGE_STATUSES}
                            for a in ANSWER_STATUSES},
        "accuracy_by_image_pct": {k: pct(sum(r["correct"] for r in judged
                                             if r["image_status"] == k), img[k])
                                  for k in IMAGE_STATUSES},
    }


def latent_vs_image(rows: list) -> dict:
    """Monet only: image_status counts split by whether latent mode was used."""
    out = {}
    for name, used in (("latent_mode_used", True), ("latent_mode_not_used", False)):
        g = [r for r in rows if r["image_status"] and r["latent_mode_used"] == used]
        out[name] = {"judged": len(g),
                     **{k: sum(r["image_status"] == k for r in g) for k in IMAGE_STATUSES}}
    return out


def build_summary(model: str, rows: list, settings: list) -> dict:
    summary = {"model": model, "judge_model": None, "settings": {}}
    for s in settings:
        srows = [r for r in rows if r["setting"] == s]
        by_cat = defaultdict(list)
        for r in srows:
            by_cat[r["category"]].append(r)
        summary["settings"][s] = {
            "overall": label_stats(srows),
            "by_category": {k: label_stats(v) for k, v in sorted(by_cat.items())},
        }
        if model == "monet":
            summary["settings"][s]["latent_vs_image"] = latent_vs_image(srows)
    # questions refused in every setting / in some settings
    per_q = defaultdict(dict)
    for r in rows:
        if r["label"]:
            per_q[r["index"]][r["setting"]] = r["label"] == "refused"
    full = [q for q, d in per_q.items() if len(d) == len(settings)]
    summary["cross_setting"] = {
        "questions_judged_in_all_settings": len(full),
        "refused_in_all_settings": sum(all(per_q[q].values()) for q in full),
        "refused_in_some_settings": sum(any(per_q[q].values()) and not all(per_q[q].values())
                                        for q in full),
        "refused_in_no_setting": sum(not any(per_q[q].values()) for q in full),
    }
    return summary


def summary_md(summary: dict) -> str:
    lines = [f"# Step 0 refusals: {summary['model']}",
             "", f"Judge: `{summary['judge_model']}` on OpenRouter. "
             "`refused` = no single option committed to; `answered_caveat` = says the "
             "image is missing/unclear but still picks one option. Image axis: `missing` "
             "= says no image was given; `uninformative` = says the image is noise/unclear; "
             "`not_mentioned` = raises no problem with the image.", "",
             "| setting | judged | answered | answered_caveat | refused | refusal % "
             "| caveat % | acc. all % | acc. non-refused % | unjudged |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in summary["settings"].items():
        o = d["overall"]
        lines.append(f"| {s} | {o['judged']}/{o['total']} | {o['answered']} | "
                     f"{o['answered_caveat']} | {o['refused']} | {o['refusal_rate_pct']} | "
                     f"{o['caveat_rate_pct']} | {o['accuracy_all_pct']} | "
                     f"{o['accuracy_non_refused_pct']} | {o['unjudged']} |")
    lines += ["", "## Image detection", "",
              "| setting | missing | uninformative | not_mentioned "
              "| detected no image % | noticed problem % |",
              "|---|---|---|---|---|---|"]
    for s, d in summary["settings"].items():
        o = d["overall"]
        im = o["image_status"]
        lines.append(f"| {s} | {im['missing']} | {im['uninformative']} | "
                     f"{im['not_mentioned']} | "
                     f"{o['detected_no_image_pct']} | {o['noticed_image_problem_pct']} |")
    lines += ["", "## Answer status x image status (counts, plus accuracy % per image status)", "",
              "| setting | answer | " + " | ".join(IMAGE_STATUSES) + " |",
              "|---|---|" + "---|" * len(IMAGE_STATUSES)]
    for s, d in summary["settings"].items():
        o = d["overall"]
        for a in ANSWER_STATUSES:
            cells = [str(o["answer_by_image"][a][k]) for k in IMAGE_STATUSES]
            lines.append(f"| {s} | {a} | " + " | ".join(cells) + " |")
        lines.append(f"| {s} | (accuracy %) | " + " | ".join(
            str(o["accuracy_by_image_pct"][k]) for k in IMAGE_STATUSES) + " |")
    if any("latent_vs_image" in d for d in summary["settings"].values()):
        lines += ["", "## Latent mode vs. image status (Monet)", "",
                  "Counts, with % of the row's judged responses. `hallucinated` = "
                  "`not_mentioned` (raised no problem with the image).", ""]
        for s, d in summary["settings"].items():
            lv = d.get("latent_vs_image")
            if not lv:
                continue
            lines += [f"### {s}", "",
                      "| | hallucinated (not_mentioned) | missing | judged |",
                      "|---|---|---|---|"]
            for name, g in lv.items():
                n = g["judged"]
                lines.append(f"| {name} | {g['not_mentioned']} ({pct(g['not_mentioned'], n)}%) "
                             f"| {g['missing']} ({pct(g['missing'], n)}%) | {n} |")
            lines.append("")
    lines += ["", "## Refusal rate by category (%)", ""]
    cats = sorted({c for d in summary["settings"].values() for c in d["by_category"]})
    settings = list(summary["settings"])
    lines += ["| category | " + " | ".join(settings) + " |",
              "|---|" + "---|" * len(settings)]
    for c in cats:
        cells = []
        for s in settings:
            st = summary["settings"][s]["by_category"].get(c)
            cells.append(f"{st['refusal_rate_pct']} ({st['refused']}/{st['judged']})"
                         if st else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Detected missing image by category (%)", "",
              "| category | " + " | ".join(settings) + " |",
              "|---|" + "---|" * len(settings)]
    for c in cats:
        cells = []
        for s in settings:
            st = summary["settings"][s]["by_category"].get(c)
            cells.append(f"{st['detected_no_image_pct']} "
                         f"({st['image_status']['missing']}/{st['judged']})" if st else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Consistency checks", ""]
    for s, d in summary["settings"].items():
        o = d["overall"]
        lines.append(f"- {s}: refused but scored correct = {o['refused_but_scored_correct']}; "
                     f"no \\boxed{{}} but not refused = {o['no_boxed_but_not_refused']}; "
                     f"answer evidence not found = {o['answer_evidence_not_found']}; "
                     f"image evidence not found = {o['image_evidence_not_found']}")
    cs = summary["cross_setting"]
    lines += ["", "## Across settings", "",
              f"- questions judged in all settings: {cs['questions_judged_in_all_settings']}",
              f"- refused in all settings: {cs['refused_in_all_settings']}",
              f"- refused in some settings only: {cs['refused_in_some_settings']}",
              f"- refused in no setting: {cs['refused_in_no_setting']}", ""]
    return "\n".join(lines)


CSV_FIELDS = ["model", "setting", "index", "category", "gold_answer", "prediction",
              "correct", "status", "latent_mode_used", "refused", "answer_status", "image_status",
              "detected_no_image", "noticed_image_problem", "label", "judge_reason",
              "answer_evidence", "answer_evidence_found", "image_evidence",
              "image_evidence_found", "finish_reason", "completion_tokens", "judge_error",
              "response"]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIRS),
                    help="which model's Step 0 output to judge (one per run)")
    ap.add_argument("--step0", type=Path, default=None,
                    help="override the Step 0 folder (default: results/step0 for monet, "
                         "results/qwen_step0 for qwen)")
    ap.add_argument("--settings", nargs="+", choices=SETTINGS, default=None,
                    help="settings to judge (default: every setting folder present)")
    ap.add_argument("--judge-model", default="openai/gpt-oss-20b")
    ap.add_argument("--reasoning-effort", default="medium", choices=("low", "medium", "high"))
    ap.add_argument("--workers", type=int, default=8, help="parallel requests")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0,
                    help="judge only the first N records per setting, into refusal_smoke/")
    ap.add_argument("--stats-only", action="store_true",
                    help="no API calls: rebuild CSV and stats from the cache")
    args = ap.parse_args()

    step0 = (args.step0 or MODEL_DIRS[args.model]).resolve()
    if not step0.is_dir():
        sys.exit(f"ERROR: {step0} does not exist")
    settings = args.settings or [s for s in SETTINGS
                                 if (step0 / s / "predictions.jsonl").exists()]
    if not settings:
        sys.exit(f"ERROR: no <setting>/predictions.jsonl under {step0}")
    out_dir = step0 / ("refusal_smoke" if args.smoke else "refusal")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "judgments.jsonl"
    state_path = out_dir / "state.json"

    records = []
    for s in settings:
        recs = read_jsonl(step0 / s / "predictions.jsonl")
        if args.smoke:
            recs = recs[:args.smoke]
        for r in recs:
            r["setting"] = s
            records.append(r)

    cache = load_cache(cache_path)
    todo = [r for r in records
            if (record_key(r["setting"], r), record_hash(args, r)) not in cache]
    config = {"model": args.model, "step0": str(step0), "settings": settings,
              "judge_model": args.judge_model, "reasoning_effort": args.reasoning_effort,
              "prompt_version": PROMPT_VERSION, "smoke": args.smoke}
    state = {"config": config, "total": len(records), "done": len(records) - len(todo),
             "failed_this_run": 0, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "finished": False}
    write_json(state_path, state)
    print(f"{args.model}: {len(records)} records in {settings} under {step0}")
    print(f"  cached: {state['done']}  to judge: {len(todo)}  -> {out_dir}")

    if todo and not args.stats_only:
        api_key = load_api_key()
        lock = threading.Lock()

        def work(rec):
            j = call_judge(args, api_key, rec)
            j.update({"key": record_key(rec["setting"], rec), "hash": record_hash(args, rec),
                      "setting": rec["setting"], "index": rec["index"],
                      "judge_model": args.judge_model, "prompt_version": PROMPT_VERSION,
                      "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            with lock:
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
                if j["error"] is None:
                    cache[(j["key"], j["hash"])] = j
                    state["done"] += 1
                else:
                    state["failed_this_run"] += 1
                write_json(state_path, state)
            return rec, j

        try:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(work, r) for r in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    rec, j = fut.result()
                    tag = (f"{j['answer_status']} / image {j['image_status']}"
                           if j["label"] else f"ERROR {j['error'][:80]}")
                    print(f"  [{i}/{len(todo)}] {rec['setting']} #{rec['index']}: {tag}",
                          flush=True)
        except FatalAPIError as e:
            sys.exit(f"ERROR: OpenRouter rejected the request ({e}). Check the API key / "
                     "credits. Progress so far is saved; re-run to continue.")
        except KeyboardInterrupt:
            print("\nStopped. Progress is saved; re-run the same command to continue.")
            os._exit(130)  # don't wait for queued requests

    # ---- CSV + stats from everything cached
    rows = []
    for r in records:
        j = cache.get((record_key(r["setting"], r), record_hash(args, r)), {})
        label = j.get("label")
        img = j.get("image_status")
        rows.append({
            "model": args.model, "setting": r["setting"], "index": r["index"],
            "latent_mode_used": (args.model == "monet"
                                 and (r.get("latent_token_count") or 0) > 0),
            "category": r.get("category"), "gold_answer": r.get("answer"),
            "prediction": r.get("prediction"), "correct": bool(r.get("correct")),
            "status": r.get("status"),
            "refused": (label == "refused") if label else None,
            "answer_status": j.get("answer_status"), "image_status": img,
            "detected_no_image": (img == "missing") if img else None,
            "noticed_image_problem": (img in NOTICED) if img else None,
            "label": label, "judge_reason": j.get("reason", ""),
            "answer_evidence": j.get("answer_evidence", ""),
            "answer_evidence_found": j.get("answer_evidence_found", False),
            "image_evidence": j.get("image_evidence", ""),
            "image_evidence_found": j.get("image_evidence_found", False),
            "finish_reason": r.get("finish_reason"),
            "completion_tokens": r.get("completion_tokens"),
            "judge_error": None if label else "not judged (see judgments.jsonl)",
            "response": r.get("response", ""),
        })
    with open(out_dir / "refusal.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row[k] is None else row[k]) for k in CSV_FIELDS})

    summary = build_summary(args.model, rows, settings)
    summary["judge_model"] = args.judge_model
    write_json(out_dir / "summary.json", summary)
    md = summary_md(summary)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    state["finished"] = all(row["label"] for row in rows)
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(state_path, state)

    print("\n" + md)
    print(f"Wrote {out_dir / 'refusal.csv'}, summary.json, summary.md")
    if not state["finished"]:
        n = sum(1 for row in rows if not row["label"])
        print(f"WARNING: {n} records are not judged yet; re-run to retry them.")


if __name__ == "__main__":
    main()

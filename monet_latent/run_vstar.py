"""
V* with Monet-7B (GGUF, llama.cpp) and full latent reasoning (see monet_llamacpp.py).

    source env.sh
    .venv/bin/python monet_latent/run_vstar.py --limit 2          # smoke test
    .venv/bin/python monet_latent/run_vstar.py                    # probe set (20 questions)
    .venv/bin/python monet_latent/run_vstar.py --dataset full     # all 191 V* questions
    .venv/bin/python monet_latent/run_vstar.py --latent-mode off  # ablation: no latent mode
    .venv/bin/python monet_latent/run_vstar.py --model qwen25vl   # baseline (never latent)

--dataset probe uses latent_probe/data/vstar_subset.jsonl (made by
latent_probe/prepare_vstar.py). --dataset full uses vstar_full.jsonl and builds it
on first use (downloads the remaining images into hf_cache/).

Writes to monet_latent/results/<model>-<quant>-<latent>/<dataset>/ (or --output-dir):
  config.json     settings (checked on resume)
  state.json      finished question ids; rerun the same command to resume
  results.jsonl   one record per question: prompt, outputs, answer, latent segments
  traces/<id>.md  the same, formatted for reading
  summary.json    accuracy and latent statistics, overall and per category

Ctrl+C pauses: at most the question in progress is lost.
"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent          # monet_latent/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "latent_probe"))
sys.path.insert(0, str(HERE))
from run_eval import HF_HUB_CACHE, MODELS, append_jsonl, now, write_json_atomic, write_jsonl_atomic  # noqa: E402
from monet_llamacpp import (LATENT_MODES, MONET_LATENT_SIZE, MONET_MIN_PIXELS, MONET_SAMPLING,  # noqa: E402
                            MonetLlamaCpp, load_image)

PROBE_DIR = ROOT / "latent_probe"
DATASETS = {"probe": PROBE_DIR / "data" / "vstar_subset.jsonl",
            "full": PROBE_DIR / "data" / "vstar_full.jsonl"}
# Monet README: system prompt for VLMEvalKit evaluation
SYSTEM_PROMPT = ("You are a helpful multimodal assistant. You are required to answer the question "
                 "based on the image provided. Put your final answer in \\boxed{}.")
CONFIG_KEYS = ["model", "gguf_file", "mmproj_file", "latent_mode", "latent_size", "n_ctx", "max_tokens",
               "min_pixels", "max_pixels", "temperature", "top_k", "top_p", "repetition_penalty", "seed",
               "system_prompt"]


# ---------------------------------------------------------------- data

def load_questions(name):
    path = DATASETS[name]
    if not path.exists():
        if name == "probe":
            sys.exit(f"{path} not found. Run latent_probe/prepare_vstar.py first.")
        print(f"{path} not found; building it (all V* questions)")
        from prepare_vstar import build
        build(full=True)
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_options(prompt_text):
    """{'A': 'red', ...} from the '(A) red' lines of the prompt."""
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\(([A-Z])\)\s*(.+)$", prompt_text, re.M)}


def last_boxed(text):
    """Content of the last \\boxed{...}, braces balanced, or None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return text[j:]


def _norm(s):
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def extract_answer(text, options):
    """(letter or None, how). Monet often boxes the option text ('green') instead of
    the letter, so both are accepted. Falls back to 'answer is (X)' without a box."""
    boxed = last_boxed(text)
    if boxed is not None:
        m = re.fullmatch(r"\s*(?:\\text\{)?\(?([A-Z])\)?(?:\})?(?:[.:)\s].*)?", boxed, re.S)
        if m and m.group(1) in options:
            return m.group(1), "boxed_letter"
        words = _norm(boxed)
        exact = [k for k, v in options.items() if _norm(v) == words]
        if len(exact) == 1:
            return exact[0], "boxed_text"
        contained = [k for k, v in options.items()
                     if _norm(v) and " ".join(_norm(v)) in " ".join(words)]
        if len(contained) == 1:
            return contained[0], "boxed_text"
        return None, "boxed_unmatched"
    m = re.search(r"answer is[:\s]*\(?([A-Z])\)", text)
    if m and m.group(1) in options:
        return m.group(1), "fallback_letter"
    return None, "no_answer"


# ---------------------------------------------------------------- records

def make_record(q, res, options, prompt, seed):
    pred, how = extract_answer(res["clean_output"], options)
    segs = res["segments"]
    p = res["p_latent_start"]
    return {
        "pred": pred, "answer_source": how, "correct": pred == q["label"],
        "boxed": last_boxed(res["clean_output"]),
        "fired": bool(segs), "n_segments": len(segs),
        "n_latents_total": sum(s["n_latents"] for s in segs),
        "segments": segs,
        "clean_output": res["clean_output"],
        "raw_output": res["raw_output"],
        "finish_reason": res["finish_reason"],
        "completion_tokens": res["completion_tokens"],
        "prompt_tokens": res["prompt_kv_tokens"],
        "image": res["image"],
        "p_latent_start_max": float(f"{max(p):.4g}") if p else None,
        "prompt_seconds": res["prompt_seconds"], "gen_seconds": res["gen_seconds"],
        "seed": seed, "options": options, "prompt": prompt,
        # per output token, for later analysis
        "token_ids": res["token_ids"], "token_kinds": res["token_kinds"],
        "p_latent_start_per_step": [float(f"{x:.3g}") for x in p],
    }


def write_trace(out_dir, rec):
    def fence(text, lang="text"):
        return f"````{lang}\n{text if text is not None else '(none)'}\n````"

    meta = {k: rec.get(k) for k in ("id", "category", "label", "pred", "correct", "answer_source", "boxed",
                                    "fired", "n_segments", "n_latents_total", "finish_reason",
                                    "completion_tokens", "prompt_tokens", "image_size_orig", "image_size_used",
                                    "image", "p_latent_start_max", "prompt_seconds", "gen_seconds", "seed")}
    verdict = "ERROR" if rec.get("error") else ("CORRECT" if rec.get("correct") else "WRONG")
    latent = f"latent x{rec['n_segments']}" if rec.get("fired") else "no latent"
    parts = [f"# {rec['id']} — {verdict}, {latent}", fence(json.dumps(meta, indent=2, ensure_ascii=False), "json"),
             f"Image: `latent_probe/{rec.get('image_path')}`",
             "## Prompt (image goes where <__media__> is)", fence(rec.get("prompt")),
             "## Output (latent segments shown as <latent>)", fence(rec.get("clean_output"))]
    if rec.get("segments"):
        rows = ["| # | start step | latents fed | ended by | placeholder tokens (never fed to the model) |",
                "|---|---|---|---|---|"]
        for i, s in enumerate(rec["segments"]):
            ph = (s.get("placeholder_text") or "").replace("|", "\\|").replace("\n", "\\n")
            rows.append(f"| {i} | {s['start_step']} | {s['n_latents']} | {s['end']} | `{ph}` |")
        parts += ["## Latent segments", "\n".join(rows),
                  "## Raw output (placeholder tokens included, as vLLM returns it)", fence(rec.get("raw_output"))]
    if rec.get("error"):
        parts += ["## Error", fence(rec["error"])]
    (out_dir / "traces" / f"{rec['id']}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def summarize(records, total, config):
    def group(rs):
        ok = [r for r in rs if not r.get("error")]
        fired = [r for r in ok if r["fired"]]
        quiet = [r for r in ok if not r["fired"]]

        def acc(x):
            return round(sum(r["correct"] for r in x) / len(x), 4) if x else None
        return {"n": len(ok), "errors": len(rs) - len(ok),
                "correct": sum(r["correct"] for r in ok), "accuracy": acc(ok),
                "no_answer": sum(r["pred"] is None for r in ok),
                "fired": len(fired), "fire_rate": round(len(fired) / len(ok), 4) if ok else None,
                "accuracy_when_fired": acc(fired), "accuracy_when_not_fired": acc(quiet),
                "segments": sum(r["n_segments"] for r in ok),
                "segment_end_reasons": _count(s["end"] for r in ok for s in r["segments"]),
                "finish_reasons": _count(r["finish_reason"] for r in ok),
                "mean_completion_tokens": round(sum(r["completion_tokens"] for r in ok) / len(ok), 1) if ok else None}

    cats = sorted({r["category"] for r in records})
    return {"model": config["model"], "quant": config["quant"], "dataset": config["dataset"],
            "latent_mode": config["latent_mode"], "latent_size": config["latent_size"],
            "completed": len(records), "total": total, "overall": group(records),
            "by_category": {c: group([r for r in records if r["category"] == c]) for c in cats},
            "fired_ids": [r["id"] for r in records if r.get("fired")],
            "wrong_ids": [r["id"] for r in records if not r.get("error") and not r["correct"]],
            "updated_at": now()}


def _count(xs):
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def read_records(path):
    records = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # half-written last line
    return records


def download(repo, filename):
    from huggingface_hub import hf_hub_download
    print(f"downloading/locating {repo}/{filename} (cache: {HF_HUB_CACHE})")
    return Path(hf_hub_download(repo, filename, cache_dir=HF_HUB_CACHE))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="probe", choices=sorted(DATASETS),
                    help="probe = latent_probe's 20-question subset, full = all 191 V* questions")
    ap.add_argument("--model", default="monet", choices=sorted(MODELS))
    ap.add_argument("--quant", default="Q8_0", help="GGUF quantisation (default Q8_0)")
    ap.add_argument("--mmproj", default="f16", choices=["f16", "Q8_0"], help="vision projector file")
    ap.add_argument("--model-path", help="local GGUF instead of downloading")
    ap.add_argument("--mmproj-path", help="local mmproj GGUF instead of downloading")
    ap.add_argument("--latent-mode", default="vllm", choices=LATENT_MODES,
                    help="vllm = Monet's vLLM runner (default), train = training layout, off = no latent mode")
    ap.add_argument("--latent-size", type=int, default=MONET_LATENT_SIZE, help="latents per segment (LATENT_SIZE)")
    ap.add_argument("--temperature", type=float, default=MONET_SAMPLING["temperature"], help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=MONET_SAMPLING["top_k"])
    ap.add_argument("--top-p", type=float, default=MONET_SAMPLING["top_p"])
    ap.add_argument("--repetition-penalty", type=float, default=MONET_SAMPLING["repetition_penalty"])
    ap.add_argument("--seed", type=int, default=0, help="question i is sampled with seed + its V* question_id")
    ap.add_argument("--max-tokens", type=int, default=4096, help="max generated tokens (latent placeholders count)")
    ap.add_argument("--n-ctx", type=int, default=8192, help="context window: image + prompt + output")
    ap.add_argument("--min-pixels", type=int, default=MONET_MIN_PIXELS)
    ap.add_argument("--max-pixels", type=int, default=2048 * 28 * 28,
                    help="resize larger images to at most this many pixels (default ~2048 image tokens; "
                         "Monet uses 8192*28*28 = 6422528, which needs --n-ctx 16384 and more VRAM)")
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all layers on GPU")
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--no-flash-attn", action="store_true")
    ap.add_argument("--output-dir", help="default: monet_latent/results/<model>-<quant>-<latent>/<dataset>")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--retry-errors", action="store_true", help="re-run questions that ended in an error")
    ap.add_argument("--force", action="store_true", help="resume even if settings differ from config.json")
    ap.add_argument("--verbose", action="store_true", help="show llama.cpp logs")
    args = ap.parse_args()

    if args.model != "monet" and args.latent_mode != "off":
        print(f"{args.model} has no latent mode; using --latent-mode off")
        args.latent_mode = "off"
    tag = "nolatent" if args.latent_mode == "off" else (
        f"latent{args.latent_size}" + ("" if args.latent_mode == "vllm" else f"-{args.latent_mode}"))
    if args.temperature <= 1e-5:
        tag += "-greedy"
    out_dir = (Path(args.output_dir).resolve() if args.output_dir
               else HERE / "results" / f"{args.model}-{args.quant}-{tag}" / args.dataset)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    config_path, state_path, results_path = out_dir / "config.json", out_dir / "state.json", out_dir / "results.jsonl"

    questions = load_questions(args.dataset)
    if args.limit:
        questions = questions[:args.limit]
    spec = MODELS[args.model]
    gguf_file = Path(args.model_path).name if args.model_path else f"{spec['stem']}.{args.quant}.gguf"
    mmproj_file = Path(args.mmproj_path).name if args.mmproj_path else f"{spec['stem']}.mmproj-{args.mmproj}.gguf"
    config = {"model": args.model, "quant": args.quant, "dataset": args.dataset, "gguf_file": gguf_file,
              "mmproj_file": mmproj_file, "latent_mode": args.latent_mode, "latent_size": args.latent_size,
              "n_ctx": args.n_ctx, "max_tokens": args.max_tokens, "min_pixels": args.min_pixels,
              "max_pixels": args.max_pixels, "temperature": args.temperature, "top_k": args.top_k,
              "top_p": args.top_p, "repetition_penalty": args.repetition_penalty, "seed": args.seed,
              "system_prompt": SYSTEM_PROMPT}

    # ---- resume
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), config[k]) for k in CONFIG_KEYS if old.get(k) != config[k]}
        if diff and not args.force:
            sys.exit(f"{out_dir} holds a run with different settings {diff}.\n"
                     "Use another --output-dir, or --force to continue anyway.")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": []}
    done = set(state["completed"])
    records = list({r["id"]: r for r in read_records(results_path) if r["id"] in done}.values())
    if args.retry_errors:
        records = [r for r in records if not r.get("error")]
    write_jsonl_atomic(results_path, records)  # drop lines not confirmed by state.json
    done = {r["id"] for r in records}

    def save_state(status):
        write_json_atomic(state_path, {"completed": [r["id"] for r in records], "total": len(questions),
                                       "last_completed": records[-1]["id"] if records else None,
                                       "status": status, "updated_at": now()})

    todo = [q for q in questions if q["id"] not in done]
    print(f"{args.model} {args.quant} ({tag}) on V* {args.dataset}: {len(questions)} questions, "
          f"{len(done)} already done, {len(todo)} to run")
    print(f"output: {out_dir}")
    if not todo:
        save_state("done")
        write_json_atomic(out_dir / "summary.json", summarize(records, len(questions), config))
        return

    gguf = Path(args.model_path).resolve() if args.model_path else download(spec["repo"], gguf_file)
    mmproj = Path(args.mmproj_path).resolve() if args.mmproj_path else download(spec["repo"], mmproj_file)
    engine = MonetLlamaCpp(gguf, mmproj, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, n_batch=args.n_batch,
                           flash_attn=not args.no_flash_attn, verbose=args.verbose,
                           require_latent_tokens=args.latent_mode != "off")
    import llama_cpp
    config.update({"gguf_path": str(gguf), "mmproj_path": str(mmproj), "llama_cpp_version": llama_cpp.__version__,
                   "n_gpu_layers": args.n_gpu_layers, "flash_attn": not args.no_flash_attn, "created_at": now()})
    if config_path.exists():
        config["created_at"] = json.loads(config_path.read_text(encoding="utf-8")).get("created_at", config["created_at"])
    write_json_atomic(config_path, config)

    try:
        for n, q in enumerate(todo, len(done) + 1):
            rec = {k: q[k] for k in ("id", "category", "question_id", "label", "image_path", "prompt_text")}
            seed = args.seed + int(q["question_id"])
            try:
                img, orig, used = load_image(PROBE_DIR / q["image_path"], args.min_pixels, args.max_pixels)
                prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
                          f"{q['prompt_text']}{engine.marker}<|im_end|>\n<|im_start|>assistant\n")
                res = engine.generate(prompt, img, max_tokens=args.max_tokens, latent_mode=args.latent_mode,
                                      latent_size=args.latent_size, temperature=args.temperature,
                                      top_k=args.top_k, top_p=args.top_p,
                                      repetition_penalty=args.repetition_penalty, seed=seed)
                rec.update({"image_size_orig": orig, "image_size_used": used})
                rec.update(make_record(q, res, parse_options(q["prompt_text"]), prompt, seed))
                rec["error"] = None
            except Exception:
                rec.update({"pred": None, "correct": False, "fired": False, "n_segments": 0, "segments": [],
                            "seed": seed, "error": traceback.format_exc()})
            rec["finished_at"] = now()
            append_jsonl(results_path, rec)   # 1) result on disk
            records.append(rec)
            save_state("running")             # 2) then mark it done
            write_trace(out_dir, rec)
            ok = [r for r in records if not r.get("error")]
            status = "ERROR" if rec["error"] else ("ok   " if rec["correct"] else "wrong")
            print(f"[{n}/{len(questions)}] {rec['id']:<12} {rec['category']:<18} {status} "
                  f"pred {rec.get('pred') or '-'} gold {rec['label']}  latent x{rec.get('n_segments', 0)}  "
                  f"{rec.get('completion_tokens') or 0:>4} tok {rec.get('gen_seconds') or 0:>6.1f}s  "
                  f"acc {sum(r['correct'] for r in ok)}/{len(ok)}", flush=True)
            if rec["error"]:
                print("   ", rec["error"].strip().splitlines()[-1])
    except KeyboardInterrupt:
        save_state("paused")
        write_json_atomic(out_dir / "summary.json", summarize(records, len(questions), config))
        print(f"\npaused after {len(records)}/{len(questions)}. Rerun the same command to continue.")
        engine.close()
        sys.exit(130)
    engine.close()

    save_state("done")
    summary = summarize(records, len(questions), config)
    write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps({"overall": summary["overall"], "by_category": summary["by_category"]}, indent=2))


if __name__ == "__main__":
    main()

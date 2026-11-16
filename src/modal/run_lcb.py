"""Monet-7B on LiveCodeBench with Monet's patched vLLM runner, so latent mode really runs.

Runs inside the Modal container from the Monet repo root (/root/Monet); see
monet_vstar_modal.py for how to launch it. Prompts, code extraction, scoring, record
format, traces and summary all come from run_eval.py, so the output folder has the same
layout as a GGUF run and works with analysis/compare_results.py.

Differences from run_eval.py:
  - Uses the full bf16 model on vLLM. Latent segments are removed before the code is
    extracted, because their tokens decode to junk text. raw_output keeps them.
  - Generates in batches. A batch is the unit of resume: results are saved after each
    one, so a crash loses at most the batch that was running.
  - latent_token_count means the same as in run_eval.py: times the start token was
    emitted. latent_inner_tokens counts the tokens inside the latent segments.
  - gen_seconds is null, since vLLM does not time single requests in a batch. See
    batch_seconds and batch_requests.

The import below must stay first. vLLM starts its engine in a spawned process that
re-imports this file, and that is how the Monet patch reaches the process doing the work.
"""
import inference.apply_vllm_monet  # noqa: F401  (must precede any vllm import)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import vllm  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import run_eval as ev  # noqa: E402  (prompts, scoring, records, traces, summary)
import run_step0  # noqa: E402  (preemption counter)
import run_vstar as base  # noqa: E402  (patch check, latent token statistics)
from inference.load_and_gen_vllm import vllm_mllm_init  # noqa: E402

MODEL_NAME, QUANT = "monet-vllm", "bf16"
LATENT_MARK = "<latent>"
# Settings that must match for a run to be resumed.
CONFIG_KEYS = ["model", "quant", "dataset", "n_ctx", "max_tokens", "temperature", "top_p", "top_k",
               "min_p", "repeat_penalty", "seed", "prompt_version", "prompt_style", "system_prompt",
               "latent_size"]


def split_answer(raw):
    """(reasoning, code, text_after_code), ignoring what is inside latent segments.
    Reasoning keeps a marker where each segment was; the code does not."""
    reasoning, code, after = ev.extract_code(base.LATENT_RE.sub(LATENT_MARK, raw))
    return reasoning, code.replace(LATENT_MARK, "") if code is not None else None, after


def finish_attempt(job, output, added_tokens, eos_id, batch_info, test_timeout):
    """Fill one attempt from its vLLM output, then run the tests on its code."""
    att = job["att"]
    try:
        completion = output.outputs[0]
        ids = list(completion.token_ids)
        if completion.finish_reason == "stop" and ids:
            ids = ids[:-1]  # the EOS token; run_eval.py does not count it either
        inner, starts, special = base.token_stats(ids, added_tokens, eos_id)
        att.update({"raw_output": completion.text, "finish_reason": completion.finish_reason,
                    "prompt_tokens": len(output.prompt_token_ids), "completion_tokens": len(ids),
                    "gen_seconds": None, "tokens_per_second": None, "latent_token_count": starts,
                    "latent_inner_tokens": inner, "added_special_tokens": special, **batch_info})
        att["reasoning"], att["final_code"], att["text_after_code"] = split_answer(completion.text)
        if att["final_code"] is None:
            verdict = ev.unscored("truncated" if completion.finish_reason == "length" else "no_code")
        else:
            verdict = ev.score(job["problem"], job["row"], att["final_code"], test_timeout)
        att.update(verdict)
        att["error"] = None
    except Exception:
        att.update(ev.unscored("error"))
        att["error"] = traceback.format_exc()
    att["finished_at"] = ev.now()


def run_batch(batch, llm, processor, added_tokens, args, pending, pool, count_preemptions):
    """Generate and score every missing attempt of the problems in `batch`.
    Returns (records in batch order, preemptions during generation or None)."""
    tokenizer = processor.tokenizer
    system = ev.PROMPT_STYLES[args.prompt_style]["system"]
    records, jobs = [], []
    for problem in batch:
        row = ev.load_row(problem)
        user_prompt = ev.build_user_prompt(problem, row, args.prompt_style)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        budget = min(args.max_tokens, args.n_ctx - len(tokenizer.encode(prompt, add_special_tokens=False)))
        attempts = pending.pop(problem["id"], {})
        for i in range(args.passes):
            if i in attempts:
                continue
            att = attempts[i] = {"attempt": i, "seed": args.seed + i}
            if budget < 16:
                att.update(ev.unscored("error"), error=f"prompt too long for --n-ctx {args.n_ctx}",
                           finished_at=ev.now())
            else:
                jobs.append({"problem": problem, "row": row, "att": att, "prompt": prompt, "budget": budget})
        records.append({"id": problem["id"], "title": problem["title"], "dataset": args.dataset,
                        "model": MODEL_NAME, "quant": QUANT, "prompt_style": args.prompt_style,
                        "platform": problem["platform"], "difficulty": problem["difficulty"],
                        "contest_date": problem["contest_date"], "format": problem["format"],
                        "user_prompt": user_prompt, "prompt_text": prompt, "_attempts": attempts})

    preempted = None
    if jobs:
        params = [vllm.SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                      min_p=args.min_p, repetition_penalty=args.repeat_penalty,
                                      max_tokens=j["budget"], n=1, skip_special_tokens=False,
                                      seed=j["att"]["seed"]) for j in jobs]
        before = count_preemptions()
        start = time.time()
        outputs = llm.generate([{"prompt": j["prompt"]} for j in jobs], sampling_params=params, use_tqdm=False)
        seconds = round(time.time() - start, 2)
        after = count_preemptions()
        preempted = None if before is None or after is None else after - before
        info = {"batch_seconds": seconds, "batch_requests": len(jobs), "batch_preemptions": preempted}
        # The scorers spend their time waiting on subprocesses, so threads are enough.
        list(pool.map(lambda pair: finish_attempt(pair[0], pair[1], added_tokens, tokenizer.eos_token_id,
                                                  info, args.test_timeout), zip(jobs, outputs)))

    for rec in records:
        attempts = rec.pop("_attempts")
        rec["attempts"] = [attempts[i] for i in sorted(attempts)]
        ev.aggregate(rec)
    return records, preempted


def write_summary(path, records, total, config, preemptions):
    summary = ev.summarize(records, total, config)
    attempts = [a for r in records for a in r["attempts"]]

    def tally(rows):
        passed = sum(1 for a in rows if a["passed"])
        return {"attempts": len(rows), "passed": passed,
                "pass_rate": round(passed / len(rows), 4) if rows else None}

    summary.update({
        "latent_size": config["latent_size"],
        "preemptions": preemptions,
        "attempts_with_latent": tally([a for a in attempts if a.get("latent_token_count")]),
        "attempts_without_latent": tally([a for a in attempts if not a.get("latent_token_count")]),
        "latent_segments_total": sum(a.get("latent_token_count") or 0 for a in attempts),
    })
    ev.write_json_atomic(path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder holding the full Monet-7B model")
    ap.add_argument("--data-dir", required=True, help="folder holding <dataset>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="lcb_easy")
    ap.add_argument("--limit", type=int, default=0, help="first N problems (0 = all)")
    # Defaults match results/monet-Q8_0/lcb_easy-prompt1, so only the runtime differs.
    ap.add_argument("--prompt-style", type=int, default=1, choices=sorted(ev.PROMPT_STYLES))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=8192, help="max prompt + output tokens")
    ap.add_argument("--max-tokens", type=int, default=4096, help="max generated tokens per attempt")
    ap.add_argument("--batch-problems", type=int, default=4,
                    help="problems generated together; also how many a crash can lose. Higher is "
                         "faster but risks preemption, which can break Monet's latent loop")
    ap.add_argument("--score-workers", type=int, default=4, help="attempts tested in parallel")
    ap.add_argument("--test-timeout", type=float, default=6.0, help="seconds per test case")
    ap.add_argument("--retry-errors", action="store_true", help="re-run attempts whose status is 'error'")
    ap.add_argument("--force", action="store_true", help="resume even if settings differ from config.json")
    args = ap.parse_args()
    if args.passes < 1:
        ap.error("--passes must be at least 1")
    if args.passes > 1 and args.temperature == 0:
        ap.error("--passes > 1 with --temperature 0 gives identical attempts")

    base.check_patch()
    if os.environ.get("MONET_STEP_LOG"):
        sys.exit("ERROR: MONET_STEP_LOG is set. This run must use Monet's patch only.")

    ev.DATA = Path(args.data_dir)
    out_dir = Path(args.out)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    config_path, state_path, results_path = out_dir / "config.json", out_dir / "state.json", out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    problems = ev.index_problems(args.dataset)
    if args.limit:
        problems = problems[:args.limit]
    style = ev.PROMPT_STYLES[args.prompt_style]
    config = {"model": MODEL_NAME, "quant": QUANT, "model_path": args.model, "dataset": args.dataset,
              "n_ctx": args.n_ctx, "max_tokens": args.max_tokens, "temperature": args.temperature,
              "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
              "repeat_penalty": args.repeat_penalty, "seed": args.seed, "passes": args.passes,
              "prompt_version": ev.PROMPT_VERSION, "prompt_style": args.prompt_style,
              "system_prompt": style["system"], "reasoning_instruction": style["instruction"],
              "format_tail": style["format_tail"], "latent_size": int(os.environ["LATENT_SIZE"]),
              "vllm_version": vllm.__version__, "batch_problems": args.batch_problems,
              "test_timeout": args.test_timeout, "created_at": ev.now()}

    # ---- resume
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), config[k]) for k in CONFIG_KEYS if old.get(k) != config[k]}
        if diff and not args.force:
            sys.exit(f"{out_dir} holds a run with different settings {diff}.\n"
                     "Use another output folder, or --force to continue anyway.")
        config["created_at"] = old.get("created_at", config["created_at"])
    ev.write_json_atomic(config_path, config)

    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": []}
    confirmed = set(state["completed"])
    records = [r for r in ev.read_results(results_path) if r["id"] in confirmed]
    records = list({r["id"]: r for r in records}.values())
    preemptions = state.get("preemptions") or 0

    # Finished problems that still need attempts: --passes was raised, or --retry-errors.
    pending, kept = {}, []
    for r in records:
        atts = [a for a in r["attempts"] if not (args.retry_errors and a["status"] == "error")]
        if len(atts) < args.passes:
            pending[r["id"]] = {a["attempt"]: a for a in atts}
        else:
            kept.append(r)
    records = kept
    ev.write_jsonl_atomic(results_path, records)  # drop lines not confirmed by state.json

    def save_state(status):
        ev.write_json_atomic(state_path, {"completed": [r["id"] for r in records], "total": len(problems),
                                          "last_completed": records[-1]["id"] if records else None,
                                          "in_progress": None, "preemptions": preemptions,
                                          "status": status, "updated_at": ev.now()})

    done = {r["id"] for r in records}
    todo = [p for p in problems if p["id"] not in done]
    print(f"{MODEL_NAME} on {args.dataset}: {len(problems)} problems, {len(done)} already done, "
          f"{len(todo)} to run, {args.passes} attempt(s) each, latent size {config['latent_size']}")
    print(f"output: {out_dir}")
    if not todo:
        save_state("done")
        write_summary(summary_path, records, len(problems), config, preemptions)
        return

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    added_tokens = {i: t for i, t in processor.tokenizer.added_tokens_decoder.items()
                    if i > ev.QWEN_LAST_TOKEN_ID}
    # Monet's own engine setup. Its sampling defaults are replaced per request in run_batch.
    llm, _ = vllm_mllm_init(args.model, tp=1, gpu_memory_utilization=0.85, max_model_len=args.n_ctx)

    try:
        with ThreadPoolExecutor(args.score_workers) as pool:
            for start in range(0, len(todo), args.batch_problems):
                batch = todo[start:start + args.batch_problems]
                new, preempted = run_batch(batch, llm, processor, added_tokens, args, pending, pool,
                                           lambda: run_step0.preemption_count(llm))
                for rec in new:                       # results, then traces, then state.json
                    ev.append_jsonl(results_path, rec)
                    ev.write_trace(out_dir, rec)
                records.extend(new)
                preemptions += preempted or 0
                save_state("running")
                write_summary(summary_path, records, len(problems), config, preemptions)
                solved = sum(r["passed"] for r in records)
                for n, rec in enumerate(new, len(records) - len(new) + 1):
                    latent = sum(a.get("latent_token_count") or 0 for a in rec["attempts"])
                    print(f"[{n}/{len(problems)}] {rec['id']:<28} {rec['status']:<13} {rec['num_attempts_passed']}/{rec['passes']} "
                          f"attempts passed, {latent} latent segments", flush=True)
                print(f"  solved so far {solved}/{len(records)}, preemptions {preemptions}", flush=True)
                if preempted:
                    print(f"  WARNING: {preempted} preemption(s) in this batch; latent reasoning may be "
                          "affected. Lower --batch-problems.", flush=True)
    except KeyboardInterrupt:
        save_state("paused")
        print(f"\npaused after {len(records)}/{len(problems)}. Rerun the same command to continue.")
        sys.exit(130)

    save_state("done")
    summary = write_summary(summary_path, records, len(problems), config, preemptions)
    print(json.dumps({k: summary[k] for k in ("overall", "status_counts", "preemptions", "attempts_with_latent",
                                              "attempts_without_latent")}, indent=2))


if __name__ == "__main__":
    main()

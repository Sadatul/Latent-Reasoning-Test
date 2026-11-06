"""Experiment A: does Monet assign more probability to the latent-start token under a real image?

Runs inside the Modal container from the Monet repo root (/root/Monet), like run_vstar.py.
For each selected question, generates one trajectory in each setting:

  real      original V* image
  noise     the Step 0 noise image for this question (seeded RGB noise, same size as the
            original, seed = 1000 + question_index), copied unchanged into noise_images/
  no_image  same system prompt and question, no image

The patched runner (patches/monet_step_log.py) logs the raw start-token probability at
every generation step. This script joins that log with the generated tokens.

The two imports below must stay first and in this order. vLLM's spawned engine process
re-imports this file, and that is how both patches reach the process doing the work.
"""
import inference.apply_vllm_monet  # noqa: F401  (Monet runner patch; must precede any vllm import)
from patches import monet_step_log

monet_step_log.apply()  # step logging; must run before vLLM imports the runner

import argparse  # noqa: E402
import base64  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import inference.load_and_gen_vllm as monet_gen  # noqa: E402
import run_vstar as base  # noqa: E402  (shared prompt, parsing and stats helpers)

SETTINGS = {"real": 0, "noise": 1, "no_image": 2}
MAX_NUM_SEQS = 30
MAX_MODEL_LEN = 16384  # same as run_vstar.py

STEP_COLUMNS = [
    "trajectory_id", "question_index", "problem_id", "group", "category", "setting", "seed",
    "step", "token_id", "token_text", "start_prob", "start_log_prob", "start_logit",
    "best_other_logit", "start_logit_margin", "is_top1", "in_latent", "is_first_start",
    # Extra column: True when this step's hidden state was fed back as a latent embedding.
    "fed_as_latent",
]
LATENT_COLUMNS = [
    "row", "trajectory_id", "question_index", "problem_id", "group", "category", "setting",
    "segment", "position_in_segment", "step",
]
TRAJ_COLUMNS = [
    "trajectory_id", "question_index", "problem_id", "group", "category", "setting", "seed",
    "image_path", "image_width", "image_height", "prompt_text", "prompt_token_ids",
    "completion_token_ids", "raw_response", "clean_response", "finish_reason",
    "prompt_tokens", "completion_tokens", "latent_token_count", "latent_segments",
    "started", "first_start_step",
    # Extra columns, not required by the plan: answer scoring for manual review.
    "answer", "prediction", "correct", "status",
]


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_question_set(path, limit):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    for r in rows:
        r["question_index"] = int(r["question_index"])
        r["prompt_tokens"] = int(r["prompt_tokens"])
    return rows


def save_real_image(raw: bytes, out_dir: Path, qi: int) -> tuple[Path, Image.Image]:
    """Save the original encoded bytes unchanged so Step 3 can reload the exact image."""
    img = Image.open(io.BytesIO(raw))
    ext = {"JPEG": "jpg", "PNG": "png"}.get(img.format, (img.format or "bin").lower())
    path = out_dir / f"q{qi:03d}_real.{ext}"
    path.write_bytes(raw)
    return path, img.convert("RGB")


def copy_step0_noise(step0_noise_dir: Path, out_dir: Path, qi: int,
                     size: tuple[int, int]) -> tuple[Path, Image.Image]:
    """Use the noise image Step 0 generated, so both steps feed identical pixels.
    The PNG is copied byte for byte so this folder is self-contained for Step 3."""
    src = step0_noise_dir / f"q{qi:03d}_noise.png"
    if not src.exists():
        fail(f"Step 0 noise image {src} is missing; run Step 0 on a range that includes "
             f"question {qi}")
    dst = out_dir / src.name
    dst.write_bytes(src.read_bytes())
    img = Image.open(dst)
    if img.size != size:
        fail(f"{src} is {img.size}, but the original image is {size}")
    return dst, img.convert("RGB")


def build_requests(questions, df, processor, out: Path, step0_noise_dir: Path):
    real_dir, noise_dir = out / "real_images", out / "noise_images"
    real_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    by_index = {int(r["index"]): r for _, r in df.iterrows()}
    requests = []
    for q in questions:
        qi = q["question_index"]
        if qi not in by_index:
            fail(f"question_index {qi} not found in the V* TSV")
        row = by_index[qi]
        options = base.row_options(row)
        user_text = base.build_user_text(row, options)
        real_path, real_img = save_real_image(base64.b64decode(row["image"]), real_dir, qi)
        width, height = real_img.size
        noise_path, noise_img = copy_step0_noise(step0_noise_dir, noise_dir, qi, (width, height))
        images = {"real": (real_path, real_img), "noise": (noise_path, noise_img), "no_image": (None, None)}
        for setting, code in SETTINGS.items():
            path, img = images[setting]
            content = ([{"type": "image", "image": img}] if img is not None else [])
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "system", "content": base.SYSTEM_PROMPT},
                        {"role": "user", "content": content}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            vllm_input = {"prompt": prompt}
            if img is not None:
                vision, _ = process_vision_info(messages)
                vllm_input["multi_modal_data"] = {"image": vision}
            requests.append({
                "trajectory_id": f"q{qi:03d}_{setting}",
                "question_index": qi,
                "problem_id": q["problem_id"],
                "group": q["group"],  # Step 1 label, kept even if this run behaves differently
                "category": q["category"],
                "setting": setting,
                "seed": qi * 10 + code,
                "image_path": str(path.relative_to(out)) if path else "",
                "image_width": width if img is not None else "",
                "image_height": height if img is not None else "",
                "prompt_text": prompt,
                "answer": str(row["answer"]).strip(),
                "options": options,
                "vllm_input": vllm_input,
            })
    return requests


def read_step_log(path: Path):
    steps, events = defaultdict(dict), []
    if not path.exists():
        return steps, events
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "event" in rec:
                events.append(rec)
                continue
            if rec["step"] in steps[rec["req_id"]]:
                fail(f"step log line {n}: duplicate step {rec['step']} for request {rec['req_id']}")
            steps[rec["req_id"]][rec["step"]] = rec
    return steps, events


def build_rows(requests, outputs, step_log, tokenizer):
    added = tokenizer.added_tokens_decoder
    by_req_id = {o.request_id: o for o in outputs}
    if len(by_req_id) != len(requests):
        fail(f"got {len(by_req_id)} outputs for {len(requests)} requests")
    traj_rows, step_rows, problems = [], [], []
    for req in requests:
        out = by_req_id[req["req_id"]]
        comp = out.outputs[0]
        ids = list(comp.token_ids)
        latent_tokens, segments, _ = base.token_stats(ids, added, tokenizer.eos_token_id)
        first_start = ids.index(base.LATENT_START_ID) if base.LATENT_START_ID in ids else None
        prediction = base.extract_choice(comp.text, req["options"])
        correct = prediction == req["answer"]
        status = "correct" if correct else ("no_answer" if prediction is None else "wrong_answer")

        logged = step_log.get(req["req_id"], {})
        missing = [s for s in range(len(ids)) if s not in logged]
        extra = sorted(s for s in logged if s >= len(ids))
        if missing or extra:
            problems.append(f"{req['trajectory_id']}: {len(missing)} generated tokens without a "
                            f"logged step (first {missing[:5]}), {len(extra)} logged steps beyond "
                            f"the output (first {extra[:5]})")
        common = {k: req[k] for k in ("trajectory_id", "question_index", "problem_id", "group",
                                      "category", "setting", "seed")}
        for s, tid in enumerate(ids):
            rec = logged.get(s)
            if rec is None:
                continue
            step_rows.append({
                **common, "step": s, "token_id": tid,
                "token_text": tokenizer.decode([tid], skip_special_tokens=False),
                **{k: rec[k] for k in ("start_prob", "start_log_prob", "start_logit",
                                       "best_other_logit", "start_logit_margin", "is_top1",
                                       "in_latent")},
                "is_first_start": s == first_start,
            })
        traj_rows.append({
            **common,
            **{k: req[k] for k in ("image_path", "image_width", "image_height", "prompt_text")},
            "prompt_token_ids": json.dumps(list(out.prompt_token_ids)),
            "completion_token_ids": json.dumps(ids),
            "raw_response": comp.text,
            "clean_response": base.LATENT_RE.sub("<latent>", comp.text),
            "finish_reason": comp.finish_reason,
            "prompt_tokens": len(out.prompt_token_ids),
            "completion_tokens": len(ids),
            "latent_token_count": latent_tokens,
            "latent_segments": segments,
            "started": first_start is not None,
            "first_start_step": "" if first_start is None else first_start,
            "answer": req["answer"], "prediction": prediction or "", "correct": correct,
            "status": status,
        })
    return traj_rows, step_rows, problems


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 bits (uint16) -> float32. Exact: bfloat16 is the top half of float32."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_hidden_index(path: Path):
    index, dims = {}, set()
    if not path.exists():
        return index, dims
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["req_id"], rec["step"])
            if key in index:
                fail(f"hidden index line {n}: duplicate step {rec['step']} for {rec['req_id']}")
            index[key] = rec
            dims.add(rec["dim"])
    return index, dims


def export_hidden(requests, outputs, step_rows, raw_dir: Path, out: Path):
    """Turn the raw capture into one float32 array per trajectory plus a latent-only table.

    hidden/<trajectory_id>.npy   [completion_tokens, hidden_dim]; row t = hidden state that
                                 produced token t (final layer, after the final norm)
    latent_vectors.npy           [n_latent_vectors, hidden_dim]; only the vectors Monet fed
                                 back as input embeddings
    latent_vectors.csv           one row per latent vector (trajectory, segment, position, step)

    Also fills `fed_as_latent` in step_rows and checks it against the token sequence.
    Returns (problems, summary)."""
    index, dims = read_hidden_index(raw_dir / "hidden_index.jsonl")
    if not index:
        return ["hidden_index.jsonl is empty: hidden states were not captured"], {}
    if len(dims) != 1:
        return [f"inconsistent hidden sizes in hidden_index.jsonl: {sorted(dims)}"], {}
    dim = dims.pop()
    raw = np.memmap(raw_dir / "hidden_states.bin", dtype=np.uint16, mode="r").reshape(-1, dim)
    hidden_dir = out / "hidden"
    hidden_dir.mkdir(exist_ok=True)
    by_req_id = {o.request_id: o for o in outputs}
    steps_by_traj = {}
    for r in step_rows:
        steps_by_traj.setdefault(r["trajectory_id"], {})[r["step"]] = r

    problems, latent_rows, latent_vecs = [], [], []
    for req in requests:
        tid = req["trajectory_id"]
        ids = list(by_req_id[req["req_id"]].outputs[0].token_ids)
        recs = [index.get((req["req_id"], t)) for t in range(len(ids))]
        missing = [t for t, rec in enumerate(recs) if rec is None]
        if missing:
            problems.append(f"{tid}: {len(missing)} steps without a hidden state (first {missing[:5]})")
            continue
        arr = bf16_to_f32(np.asarray(raw[[rec["row"] for rec in recs]]))
        np.save(hidden_dir / f"{tid}.npy", arr)

        segment, pos = -1, 0
        for t, (tok, rec) in enumerate(zip(ids, recs)):
            row = steps_by_traj.get(tid, {}).get(t)
            fed = bool(rec["fed_as_latent"])
            if row is not None:
                row["fed_as_latent"] = fed
                # Monet feeds the state of the start step and of every latent step except
                # the one whose token becomes the end token.
                expected = tok == base.LATENT_START_ID or (
                    row["in_latent"] and tok != base.LATENT_END_ID)
                if fed != expected:
                    problems.append(f"{tid} step {t}: fed_as_latent={fed}, expected {expected} "
                                    f"from the token sequence")
            if tok == base.LATENT_START_ID and fed:
                segment, pos = segment + 1, 0
            if fed:
                latent_rows.append({
                    "row": len(latent_rows), "trajectory_id": tid,
                    **{k: req[k] for k in ("question_index", "problem_id", "group", "category",
                                           "setting")},
                    "segment": segment, "position_in_segment": pos, "step": t,
                })
                latent_vecs.append(arr[t])
                pos += 1

    vecs = np.stack(latent_vecs) if latent_vecs else np.zeros((0, dim), np.float32)
    np.save(out / "latent_vectors.npy", vecs)
    write_csv(out / "latent_vectors.csv", LATENT_COLUMNS, latent_rows)
    summary = {
        "hidden_dim": dim,
        "dtype": "float32 (exact upcast of the bfloat16 states)",
        "trajectory_files": len(list(hidden_dir.glob("*.npy"))),
        "latent_vectors": len(latent_rows),
        "hidden_mb": round(sum(f.stat().st_size for f in hidden_dir.glob("*.npy")) / 2**20, 1),
    }
    return problems, summary


def write_csv(path: Path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_traces(traj_rows, out: Path):
    """One markdown file per trajectory, same fencing as run_vstar.py, for manual review."""
    d = out / "traces"
    d.mkdir(exist_ok=True)
    for r in traj_rows:
        meta = {k: r[k] for k in ("trajectory_id", "question_index", "problem_id", "group",
                                  "category", "setting", "seed", "image_path", "finish_reason",
                                  "prompt_tokens", "completion_tokens", "latent_token_count",
                                  "latent_segments", "started", "first_start_step", "answer",
                                  "prediction", "correct", "status")}
        text = "\n".join([
            f"# {r['trajectory_id']} — {r['status'].upper()}", "",
            base.json_block(meta), "",
            "## Prompt (chat template applied)", "", base.code_block(r["prompt_text"], "text"), "",
            "## Model output (full reasoning trace, special tokens included)", "",
            base.code_block(r["raw_response"], "text"), "",
        ])
        (d / f"{r['trajectory_id']}.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="VStarBench.tsv")
    ap.add_argument("--questions", required=True, help="question_set.csv from Step 1")
    ap.add_argument("--noise-dir", required=True,
                    help="Step 0 noise_images folder (q###_noise.png)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N questions (0 = all)")
    ap.add_argument("--force", action="store_true", help="rerun even if state.json says done")
    args = ap.parse_args()

    base.check_patch()
    if not getattr(sys.modules["vllm.v1.worker.gpu_model_runner"], "_MONET_STEP_LOG_PATCHED", False):
        fail("step-log patch is not active; is MONET_STEP_LOG set?")
    out = Path(args.out)
    log_path = Path(os.environ["MONET_STEP_LOG"])
    if log_path.resolve().parent != (out / "raw").resolve():
        fail(f"MONET_STEP_LOG must be {out / 'raw' / 'step_log.jsonl'}, got {log_path}")

    # state.json: the 30 trajectories run as one concurrent batch, so the unit of
    # resumption is the whole run. A finished run is skipped; an unfinished one restarts.
    state_path = out / "state.json"
    if state_path.exists() and not args.force:
        state = json.loads(state_path.read_text())
        if state.get("status") == "done":
            print(f"{out} is already complete (state.json). Use --force to rerun.")
            return
        print(f"Previous run did not finish (status={state.get('status')}); starting over.")
    out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (log_path, log_path.parent / "hidden_states.bin",
                  log_path.parent / "hidden_index.jsonl"):
        stale.unlink(missing_ok=True)
    state = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "limit": args.limit, "noise_dir": args.noise_dir, "latent_size": int(os.environ["LATENT_SIZE"])}
    state_path.write_text(json.dumps(state, indent=2))

    questions = load_question_set(args.questions, args.limit)
    df = pd.read_csv(args.data, sep="\t")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    requests = build_requests(questions, df, processor, out, Path(args.noise_dir))
    print(f"{len(questions)} questions x {len(SETTINGS)} settings = {len(requests)} trajectories")

    # Monet's own engine settings, with max_num_seqs lowered from 512 to 30.
    monet_gen.max_num_seqs = MAX_NUM_SEQS
    llm, base_params = monet_gen.vllm_mllm_init(
        args.model, tp=1, gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN)

    cache = llm.llm_engine.vllm_config.cache_config
    kv_tokens = (cache.num_gpu_blocks or 0) * cache.block_size
    # Prompt sizes from Step 1 (real image); noise has the same size and no_image is smaller.
    max_prompt = max(q["prompt_tokens"] for q in questions)
    kv_needed = MAX_NUM_SEQS * (max_prompt + base_params.max_tokens)
    print(f"GPU KV cache: {kv_tokens} tokens; need >= {MAX_NUM_SEQS} x ({max_prompt} + "
          f"{base_params.max_tokens}) = {kv_needed}", flush=True)
    state.update(kv_cache_tokens=kv_tokens, kv_needed=kv_needed, max_prompt_tokens_step1=max_prompt)
    state_path.write_text(json.dumps(state, indent=2))
    if kv_tokens < kv_needed:
        state["status"] = "failed_kv_cache"
        state_path.write_text(json.dumps(state, indent=2))
        fail("GPU KV cache is too small for 30 concurrent trajectories; stopping before generation.")

    params = []
    for req in requests:
        p = base_params.clone()
        p.seed = req["seed"]
        params.append(p)
    outputs = llm.generate([r["vllm_input"] for r in requests], sampling_params=params, use_tqdm=True)
    # LLM.generate returns outputs in input order.
    for req, o in zip(requests, outputs):
        req["req_id"] = o.request_id

    step_log, events = read_step_log(log_path)
    if events:
        state["status"] = "failed_preemption"
        state_path.write_text(json.dumps(state, indent=2))
        fail(f"preemption events in the step log: {events}")
    if not step_log:
        fail("step log is empty; the logging patch did not run in the engine process")

    traj_rows, step_rows, problems = build_rows(requests, outputs, step_log, processor.tokenizer)
    hidden_problems, hidden_summary = export_hidden(requests, outputs, step_rows, log_path.parent, out)
    write_csv(out / "exp_a_trajectories.csv", TRAJ_COLUMNS, traj_rows)
    write_csv(out / "exp_a_steps.csv", STEP_COLUMNS, step_rows)
    write_traces(traj_rows, out)

    actual_max_prompt = max(r["prompt_tokens"] for r in traj_rows)
    actual_needed = MAX_NUM_SEQS * (actual_max_prompt + base_params.max_tokens)
    checks = {
        "trajectories": len(traj_rows),
        "expected_trajectories": len(requests),
        "step_rows": len(step_rows),
        "missing_or_extra_steps": problems,
        "hidden_state_problems": hidden_problems,
        "hidden_states": hidden_summary,
        "preemption_events": len(events),
        "kv_cache_tokens": kv_tokens,
        "kv_needed_actual_prompts": actual_needed,
        "kv_ok_actual_prompts": kv_tokens >= actual_needed,
        "hit_token_limit": sum(r["finish_reason"] == "length" for r in traj_rows),
        "started_by_setting": {s: sum(r["started"] for r in traj_rows if r["setting"] == s)
                               for s in SETTINGS},
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))

    ok = (not problems and not hidden_problems and len(traj_rows) == len(requests)
          and checks["kv_ok_actual_prompts"])
    state.update(status="done" if ok else "failed_checks",
                 finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    state_path.write_text(json.dumps(state, indent=2))
    if not ok:
        fail("Experiment A checks failed; see checks.json")
    # The raw bfloat16 capture is fully contained in hidden/*.npy now; drop the duplicate.
    (log_path.parent / "hidden_states.bin").unlink(missing_ok=True)
    print(f"Experiment A complete: {len(traj_rows)} trajectories, {len(step_rows)} logged steps, "
          f"{hidden_summary['latent_vectors']} latent vectors.")


if __name__ == "__main__":
    main()

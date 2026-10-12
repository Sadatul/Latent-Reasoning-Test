"""
Count how often quantized Monet-7B (GGUF, llama.cpp) emits its latent-start token
(id 151666) on V* image questions. Not a benchmark: no scoring, only the token.

    source env.sh
    .venv/bin/python latent_probe/prepare_vstar.py        # once
    .venv/bin/python latent_probe/run_probe.py             # Q8_0, 20 questions
    .venv/bin/python latent_probe/run_probe.py --limit 2   # smoke test

Writes to latent_probe/results/monet-<quant>/ (or --output-dir):
  config.json     settings (checked on resume)
  state.json      ids of finished questions; rerun the same command to resume
  results.jsonl   one record per question: prompt, full output, latent-token stats
  traces/<id>.md  the same record, formatted for reading
  summary.json    how many questions fired the latent token, overall and per category

Ctrl+C pauses: at most the question in progress is lost.
"""

import argparse
import ctypes
import io
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # latent_probe/
sys.path.insert(0, str(HERE.parent / "src"))
from run_eval import (HF_HUB_CACHE, MODELS, MONET_LATENT_START_ID, QWEN_LAST_TOKEN_ID,  # noqa: E402
                      append_jsonl, now, write_json_atomic, write_jsonl_atomic)

MONET_LATENT_END_ID = 151667
DATA_FILE = HERE / "data" / "vstar_subset.jsonl"
# System prompt from Monet's README (VLMEvalKit evaluation). The user turn is the
# question text followed by the image, as in Monet's inference example.
SYSTEM_PROMPT = ("You are a helpful multimodal assistant. You are required to answer the question "
                 "based on the image provided.")
CONFIG_KEYS = ["quant", "gguf_file", "mmproj_file", "n_ctx", "max_tokens", "max_pixels", "system_prompt"]


# ---------------------------------------------------------------- inputs

def load_questions():
    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} not found. Run latent_probe/prepare_vstar.py first.")
    with open(DATA_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def image_bytes(path, max_pixels):
    """JPEG bytes of the image, downscaled (aspect kept) if above max_pixels."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    orig = img.size
    if img.width * img.height > max_pixels:
        s = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((max(28, int(img.width * s)), max(28, int(img.height * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), list(orig), list(img.size)


def download(repo, filename):
    from huggingface_hub import hf_hub_download
    print(f"downloading/locating {repo}/{filename} (cache: {HF_HUB_CACHE})")
    return Path(hf_hub_download(repo, filename, cache_dir=HF_HUB_CACHE))


# ---------------------------------------------------------------- model

class Probe:
    def __init__(self, args, gguf_path, mmproj_path):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        self.llm = Llama(model_path=str(gguf_path), n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                         verbose=args.verbose)
        # Only used for its mtmd (vision) context and image loading; generation is ours.
        self.handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=args.verbose)
        self.handler._init_mtmd_context(self.llm)
        self.mtmd = self.handler._mtmd_cpp
        self.marker = self.mtmd.mtmd_default_marker().decode("utf-8")
        self.n_vocab = self.llm.n_vocab()
        self.stop_ids = {self.llm.token_eos()}
        for text in ("<|im_end|>", "<|endoftext|>"):
            ids = self.llm.tokenize(text.encode(), add_bos=False, special=True)
            if len(ids) == 1:
                self.stop_ids.add(ids[0])

    def prompt_text(self, question):
        # Same layout as Qwen2.5-VL's chat template; mtmd replaces the marker with
        # <|vision_start|> image tokens <|vision_end|>.
        return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{question}{self.marker}<|im_end|>\n<|im_start|>assistant\n")

    def eval_prompt(self, text, img):
        """Tokenize text + image with mtmd and evaluate it into the KV cache, as
        llama-cpp-python's Llava15ChatHandler.__call__ does, but stop before
        generating. Returns the prompt length in tokens (image tokens included)."""
        import llama_cpp
        llm, m = self.llm, self.mtmd
        llm.reset()
        llm._ctx.kv_cache_clear()
        bitmap = self.handler._create_bitmap_from_bytes(img)
        chunks = m.mtmd_input_chunks_init()
        try:
            inp = m.mtmd_input_text()
            data = text.encode("utf-8")
            inp.text, inp.text_len, inp.add_special, inp.parse_special = data, len(data), True, True
            bitmaps = (m.mtmd_bitmap_p_ctypes * 1)(bitmap)
            if m.mtmd_tokenize(self.handler.mtmd_ctx, chunks, ctypes.byref(inp), bitmaps, 1) != 0:
                raise ValueError("mtmd_tokenize failed")
            for i in range(m.mtmd_input_chunks_size(chunks)):
                chunk = m.mtmd_input_chunks_get(chunks, i)
                kind = m.mtmd_input_chunk_get_type(chunk)
                if kind == m.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n = ctypes.c_size_t()
                    ptr = m.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n))
                    tokens = [ptr[j] for j in range(n.value)]
                    if llm.n_tokens + len(tokens) > llm.n_ctx():
                        raise ValueError(f"prompt exceeds --n-ctx {llm.n_ctx()}")
                    llm.eval(tokens)
                else:
                    if llm.n_tokens + m.mtmd_input_chunk_get_n_tokens(chunk) > llm.n_ctx():
                        raise ValueError(f"prompt exceeds --n-ctx {llm.n_ctx()}; lower --max-pixels")
                    new_n_past = llama_cpp.llama_pos(0)
                    if m.mtmd_helper_eval_chunk_single(self.handler.mtmd_ctx, llm._ctx.ctx, chunk,
                                                       llama_cpp.llama_pos(llm.n_tokens), llama_cpp.llama_seq_id(0),
                                                       llm.n_batch, False, ctypes.byref(new_n_past)) != 0:
                        raise ValueError("mtmd_helper_eval_chunk_single failed")
                    llm.n_tokens = new_n_past.value
        finally:
            m.mtmd_input_chunks_free(chunks)
            m.mtmd_bitmap_free(bitmap)
        if kind != m.MTMD_INPUT_CHUNK_TYPE_TEXT:
            raise ValueError("prompt must end with text so the last position has logits")
        return llm.n_tokens

    def last_logits(self):
        # llama-cpp-python (0.3.x) no longer copies logits into llm.scores unless
        # logits_all=True, so read the last position straight from the context.
        return np.ctypeslib.as_array(self.llm._ctx.get_logits_ith(-1), shape=(self.n_vocab,)).copy()

    def detok(self, ids):
        try:
            data = self.llm.detokenize(ids, special=True)
        except TypeError:
            data = self.llm.detokenize(ids)
        return data.decode("utf-8", errors="replace")

    def run(self, question, img, max_tokens):
        """Greedy decoding with the latent-start token's probability at every step.
        The latent token does not stop generation (llama.cpp has no latent mode;
        it is fed back as an ordinary token)."""
        text = self.prompt_text(question)
        n_prompt = self.eval_prompt(text, img)
        budget = min(max_tokens, self.llm.n_ctx() - n_prompt)
        out, p_latent, rank_latent, finish = [], [], [], "length"
        start = time.time()
        for _ in range(budget):
            logits = self.last_logits()
            e = np.exp(logits - logits.max())
            p_latent.append(float(e[MONET_LATENT_START_ID] / e.sum()))
            rank_latent.append(int((logits > logits[MONET_LATENT_START_ID]).sum()))
            tok = int(logits.argmax())
            if tok in self.stop_ids:
                finish = "stop"
                break
            out.append(tok)
            self.llm.eval([tok])
        return text, n_prompt, out, p_latent, rank_latent, finish, time.time() - start


# ---------------------------------------------------------------- records

def make_record(probe, image_info, n_prompt, out, p_latent, rank_latent, finish, seconds):
    fires = [i for i, t in enumerate(out) if t == MONET_LATENT_START_ID]
    extra = Counter(t for t in out if t > QWEN_LAST_TOKEN_ID)
    best = int(np.argmax(p_latent)) if p_latent else None
    return {
        "fired": bool(fires),
        "n_fires": len(fires),
        "first_fire_step": fires[0] if fires else None,
        "n_latent_end": extra.get(MONET_LATENT_END_ID, 0),
        "p_latent_max": float(f"{max(p_latent):.4g}") if p_latent else None,
        "p_latent_max_step": best,
        "best_rank_latent": min(rank_latent) if rank_latent else None,  # 0 = it was the top token
        "added_special_tokens": {f"{t} {probe.detok([t])}": c for t, c in sorted(extra.items())},
        "image_size_orig": image_info[0], "image_size_used": image_info[1],
        "prompt_tokens": n_prompt, "completion_tokens": len(out), "finish_reason": finish,
        "gen_seconds": round(seconds, 2),
        "raw_output": probe.detok(out),
        # per generated step, so the token's probability can be plotted along the answer
        "p_latent_per_step": [float(f"{p:.3g}") for p in p_latent],
    }


def write_trace(out_dir, rec):
    def fence(text, lang=""):
        return f"````{lang}\n{text if text is not None else '(none)'}\n````"

    meta = {k: rec.get(k) for k in ("id", "category", "label", "fired", "n_fires", "first_fire_step",
                                    "n_latent_end", "p_latent_max", "p_latent_max_step", "best_rank_latent",
                                    "added_special_tokens", "image_size_orig", "image_size_used",
                                    "prompt_tokens", "completion_tokens", "finish_reason", "gen_seconds")}
    head = "LATENT FIRED" if rec.get("fired") else ("ERROR" if rec.get("error") else "no latent token")
    parts = [f"# {rec['id']} — {head}", fence(json.dumps(meta, indent=2, ensure_ascii=False), "json"),
             f"Image: `{rec.get('image_path')}`",
             "## Prompt (image goes where <__media__> is)", fence(rec.get("prompt_text"), "text"),
             "## Model output (special tokens included; latent token id 151666 appears inline)",
             fence(rec.get("raw_output"), "text")]
    if rec.get("error"):
        parts += ["## Error", fence(rec["error"], "text")]
    (out_dir / "traces" / f"{rec['id']}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def summarize(records, total, config):
    def group(rs):
        ok = [r for r in rs if not r.get("error")]
        p = sorted(r["p_latent_max"] for r in ok if r.get("p_latent_max") is not None)
        return {"n": len(ok), "errors": len(rs) - len(ok),
                "fired": sum(r["fired"] for r in ok),
                "fire_rate": round(sum(r["fired"] for r in ok) / len(ok), 4) if ok else None,
                "total_fires": sum(r["n_fires"] for r in ok),
                "p_latent_max_median": p[len(p) // 2] if p else None,
                "p_latent_max_max": p[-1] if p else None}

    cats = sorted({r["category"] for r in records})
    return {"quant": config["quant"], "completed": len(records), "total": total,
            "overall": group(records),
            "by_category": {c: group([r for r in records if r["category"] == c]) for c in cats},
            "fired_ids": [r["id"] for r in records if r.get("fired")],
            "updated_at": now()}


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


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quant", default="Q8_0", help="Monet GGUF quantisation (default Q8_0)")
    ap.add_argument("--mmproj", default="f16", choices=["f16", "Q8_0"], help="vision projector file (default f16)")
    ap.add_argument("--model-path", help="local Monet GGUF instead of downloading")
    ap.add_argument("--mmproj-path", help="local mmproj GGUF instead of downloading")
    ap.add_argument("--output-dir", help="default: latent_probe/results/monet-<quant>")
    ap.add_argument("--n-ctx", type=int, default=8192, help="context window (image + text + output)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="max generated tokens per question")
    ap.add_argument("--max-pixels", type=int, default=2048 * 28 * 28,
                    help="downscale larger images to this many pixels (default ~2048 visual tokens)")
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all layers on GPU")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--retry-errors", action="store_true", help="re-run questions that ended in an error")
    ap.add_argument("--force", action="store_true", help="resume even if settings differ from config.json")
    ap.add_argument("--verbose", action="store_true", help="show llama.cpp logs")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve() if args.output_dir else HERE / "results" / f"monet-{args.quant}"
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    config_path, state_path, results_path = out_dir / "config.json", out_dir / "state.json", out_dir / "results.jsonl"

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    spec = MODELS["monet"]
    gguf_file = Path(args.model_path).name if args.model_path else f"{spec['stem']}.{args.quant}.gguf"
    mmproj_file = Path(args.mmproj_path).name if args.mmproj_path else f"{spec['stem']}.mmproj-{args.mmproj}.gguf"
    config = {"quant": args.quant, "gguf_file": gguf_file, "mmproj_file": mmproj_file, "n_ctx": args.n_ctx,
              "max_tokens": args.max_tokens, "max_pixels": args.max_pixels, "decoding": "greedy",
              "system_prompt": SYSTEM_PROMPT, "latent_start_id": MONET_LATENT_START_ID,
              "latent_end_id": MONET_LATENT_END_ID}

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
    print(f"Monet {args.quant} on V*: {len(questions)} questions, {len(done)} already done, {len(todo)} to run")
    print(f"output: {out_dir}")
    if not todo:
        save_state("done")
        write_json_atomic(out_dir / "summary.json", summarize(records, len(questions), config))
        return

    gguf = Path(args.model_path).resolve() if args.model_path else download(spec["repo"], gguf_file)
    mmproj = Path(args.mmproj_path).resolve() if args.mmproj_path else download(spec["repo"], mmproj_file)
    probe = Probe(args, gguf, mmproj)
    import llama_cpp
    config.update({"gguf_path": str(gguf), "mmproj_path": str(mmproj), "llama_cpp_version": llama_cpp.__version__,
                   "n_gpu_layers": args.n_gpu_layers, "created_at": now()})
    if config_path.exists():
        config["created_at"] = json.loads(config_path.read_text(encoding="utf-8")).get("created_at", config["created_at"])
    write_json_atomic(config_path, config)

    try:
        for n, q in enumerate(todo, len(done) + 1):
            rec = {k: q[k] for k in ("id", "category", "question_id", "label", "image_path", "prompt_text")}
            try:
                img, orig, used = image_bytes(HERE / q["image_path"], args.max_pixels)
                text, *result = probe.run(q["prompt_text"], img, args.max_tokens)
                rec["prompt_text"] = text
                rec.update(make_record(probe, (orig, used), *result))
                rec["error"] = None
            except Exception:
                rec.update({"fired": False, "n_fires": 0, "error": traceback.format_exc()})
            rec["finished_at"] = now()
            append_jsonl(results_path, rec)   # 1) result on disk
            records.append(rec)
            save_state("running")             # 2) then mark it done
            write_trace(out_dir, rec)
            fired = sum(r["fired"] for r in records)
            status = "ERROR" if rec["error"] else ("FIRED x%d" % rec["n_fires"] if rec["fired"] else "-")
            print(f"[{n}/{len(questions)}] {rec['id']:<12} {rec['category']:<18} {status:<9} "
                  f"p_max {rec.get('p_latent_max') or 0:.2e}  {rec.get('completion_tokens') or 0:>4} tok "
                  f"{rec.get('gen_seconds') or 0:>6.1f}s  fired so far {fired}/{len(records)}", flush=True)
    except KeyboardInterrupt:
        save_state("paused")
        write_json_atomic(out_dir / "summary.json", summarize(records, len(questions), config))
        print(f"\npaused after {len(records)}/{len(questions)}. Rerun the same command to continue.")
        sys.exit(130)

    save_state("done")
    summary = summarize(records, len(questions), config)
    write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps({"overall": summary["overall"], "by_category": summary["by_category"]}, indent=2))


if __name__ == "__main__":
    main()

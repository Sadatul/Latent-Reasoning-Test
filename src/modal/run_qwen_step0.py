"""Step 0 control: Qwen2.5-VL-7B-Instruct on V* with no image and with the Step 0 noise images.

Same as run_step0.py in everything except the model, and with NO Monet runner patch:
plain vLLM 0.10.0.

Kept identical to Monet's Step 0 (by reusing the same code, not a copy):
  * engine and sampling: inference.load_and_gen_vllm.vllm_mllm_init (Monet's defaults:
    seed 0, temperature 0.1, top-k 50, top-p 0.8, repetition penalty 1.01, max_tokens 4096,
    max_num_seqs 512, min/max pixels, enforce_eager, chunked prefill), with the same
    tp=1, gpu_memory_utilization=0.85, max_model_len=16384 as run_step0.py;
  * the system prompt, user prompt and message layout: run_step0.build_inputs;
  * the noise pixels: Step 0's noise PNGs, copied byte for byte (never regenerated);
  * scoring, traces, summaries, state.json and per-question CSV: run_step0 / run_vstar.
The two environment variables Monet's patch sets for vLLM itself (VLLM_USE_V1=1,
VLLM_NO_USAGE_STATS=1) are set here too. LATENT_START_ID/LATENT_END_ID are set only because
the shared scoring code counts those token IDs; Qwen has no latent mode.

How Monet's patch is kept out: run_step0.py starts with `import inference.apply_vllm_monet`.
Before importing run_step0 this script puts an empty module under that name in
sys.modules, so the import does nothing. vLLM's engine process is spawned and re-imports
this file, so the same holds there. After the imports the script checks that vLLM's GPU
model runner is vLLM's own module and exits otherwise.

The prompts are rendered with Qwen's own processor. With --check-template, the prompts are
also rendered with Monet's processor and the run stops if any prompt string differs.
"""
import os
import sys
import types

# --- keep Monet's runner patch out (must run before importing run_step0 / vllm) ---------
sys.modules["inference.apply_vllm_monet"] = types.ModuleType("inference.apply_vllm_monet")
os.environ.setdefault("VLLM_USE_V1", "1")          # as Monet's patch sets it
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")  # as Monet's patch sets it
os.environ.setdefault("LATENT_START_ID", "151666")  # only for the shared token counters
os.environ.setdefault("LATENT_END_ID", "151667")

import argparse  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import run_step0 as s0  # noqa: E402  (unchanged Step 0 helpers)
import run_vstar as base  # noqa: E402
from inference.load_and_gen_vllm import vllm_generate, vllm_mllm_init  # noqa: E402

SETTINGS = ("real", "noise", "no_image")
MODEL_LABEL = "qwen2.5-vl-7b-instruct"
TITLE = "# Step 0 control: Qwen2.5-VL-7B-Instruct on V* (no Monet patch)"


def check_vanilla_runner():
    import vllm.v1.worker.gpu_model_runner as runner
    path = getattr(runner, "__file__", "") or ""
    if "monet" in path.lower() or "vllm/v1/worker/gpu_model_runner.py" not in path.replace("\\", "/"):
        sys.exit(f"ERROR: vLLM's runner is {path!r}, not vLLM's own. Monet's patch must not be active.")
    if any("monet" in name for name in sys.modules if name.startswith("inference.vllm")):
        sys.exit("ERROR: Monet's patched runner module is imported.")
    print(f"vLLM runner: {path} (unpatched)")


def copy_noise_images(df, step0_noise: Path, noise_dir: Path) -> None:
    """Copy Step 0's noise PNGs byte for byte and check each size against the original."""
    noise_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        index = int(row["index"])
        src, dst = s0.noise_path(step0_noise, index), s0.noise_path(noise_dir, index)
        if not src.exists():
            sys.exit(f"ERROR: {src} is missing; Step 0's noise images are required.")
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
        size = Image.open(io.BytesIO(base64.b64decode(row["image"]))).size
        if Image.open(dst).size != size:
            sys.exit(f"ERROR: {dst} has size {Image.open(dst).size}, original is {size}")


def relabel_traces(trace_dir: Path, records: list) -> None:
    """run_vstar.render_trace writes "model": "monet-7b" in the header block; fix that one field."""
    for r in records:
        path = trace_dir / f"vstar_{r['index']:03d}.md"
        text = path.read_text(encoding="utf-8")
        if '"model": "monet-7b"' not in text:
            sys.exit(f"ERROR: unexpected trace header in {path}")
        path.write_text(text.replace('"model": "monet-7b"', f'"model": "{MODEL_LABEL}"', 1),
                        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen2.5-VL-7B-Instruct folder")
    ap.add_argument("--data", required=True, help="VStarBench.tsv")
    ap.add_argument("--step0-noise", required=True, help="Step 0's noise_images folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N V* questions (0 = all 191)")
    ap.add_argument("--settings", default="noise,no_image",
                    help="comma-separated subset of real,noise,no_image")
    ap.add_argument("--check-template", default="",
                    help="Monet model folder: stop if Monet's processor renders any prompt differently")
    ap.add_argument("--force", action="store_true", help="rerun settings already marked done")
    args = ap.parse_args()

    check_vanilla_runner()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if bad := [s for s in settings if s not in SETTINGS]:
        sys.exit(f"ERROR: unknown settings {bad}; choose from {SETTINGS}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = s0.load_state(state_path)
    if state.get("limit", args.limit) != args.limit:
        sys.exit(f"ERROR: {out} was started with --limit {state['limit']}; "
                 f"use the same limit or another output folder.")
    state.update(limit=args.limit, latent_size=None, model=MODEL_LABEL, monet_patch=False)
    s0.save_state(state_path, state)

    df = pd.read_csv(args.data, sep="\t")
    if args.limit:
        df = df.head(args.limit)
    noise_dir = out / "noise_images"
    if "noise" in settings:
        copy_noise_images(df, Path(args.step0_noise), noise_dir)
    print(f"{len(df)} questions; settings {settings}")

    todo = [s for s in settings
            if args.force or state["settings"].get(s, {}).get("status") != "done"]
    for s in settings:
        if s not in todo:
            print(f"{s}: already done (state.json), skipped. Use --force to rerun.")

    if todo:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        if args.check_template:
            monet_proc = AutoProcessor.from_pretrained(args.check_template, trust_remote_code=True)
            for s in todo:
                q = [x["prompt"] for x in s0.build_inputs(df, processor, s, noise_dir)]
                m = [x["prompt"] for x in s0.build_inputs(df, monet_proc, s, noise_dir)]
                if diff := [i for i, (a, b) in enumerate(zip(q, m)) if a != b]:
                    sys.exit(f"ERROR: {s}: Qwen and Monet render {len(diff)} prompts differently "
                             f"(first row {diff[0]}).\nQwen:  {q[diff[0]]!r}\nMonet: {m[diff[0]]!r}")
            print("Prompt check: Qwen's and Monet's processors render identical prompts.")
            del monet_proc

        # Same engine call as run_step0.py: Monet's defaults, max_model_len raised to 16384.
        llm, sampling_params = vllm_mllm_init(
            args.model, tp=1, gpu_memory_utilization=0.85, max_model_len=16384)
        for s in todo:
            state["settings"][s] = {"status": "running",
                                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            s0.save_state(state_path, state)
            before = s0.preemption_count(llm)
            inputs = s0.build_inputs(df, processor, s, noise_dir)
            outputs = vllm_generate(inputs, sampling_params, llm)
            after = s0.preemption_count(llm)
            records, summary = base.score(df, outputs, processor.tokenizer)
            for r, o in zip(records, outputs):
                ids = list(o.outputs[0].token_ids)
                r["completion_token_ids"] = ids
                r["latent_start_steps"] = [i for i, t in enumerate(ids) if t == base.LATENT_START_ID]
                r["latent_end_steps"] = [i for i, t in enumerate(ids) if t == base.LATENT_END_ID]
                r["setting"] = s
                r["model"] = MODEL_LABEL
                if s == "noise":
                    r["image_path"] = str(s0.noise_path(noise_dir, r["index"]).relative_to(out))
            summary.update(latent_size=None, setting=s, model=MODEL_LABEL)

            d = out / s
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            (d / "summary.json").write_text(json.dumps(summary, indent=2))
            base.write_traces(records, d, None)
            relabel_traces(d / "traces", records)
            s0.label_traces(d / "traces", s, records)

            state["settings"][s] = {
                "status": "done",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "preemptions": None if before is None or after is None else after - before,
                "stats": s0.setting_stats(summary, records),
            }
            s0.save_state(state_path, state)
            print(f"{s}: {summary['correct']}/{summary['questions']} correct")

    per_setting = {s: s0.read_records(out / s / "predictions.jsonl") for s in SETTINGS
                   if state["settings"].get(s, {}).get("status") == "done"}
    s0.write_comparison(out, state, per_setting)
    md = out / "summary.md"
    md.write_text(md.read_text(encoding="utf-8").replace(
        "# Step 0: Monet-7B on V*, three settings", TITLE, 1), encoding="utf-8")


if __name__ == "__main__":
    main()

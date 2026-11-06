"""Step 0: Monet-7B on V* under three image settings, with Monet's own vLLM setup only.

Runs inside the Modal container from the Monet repo root (/root/Monet), like run_vstar.py.

  real      original V* image (same as run_vstar.py)
  noise     seeded RGB noise, same size as the original (seed = 1000 + V* index)
  no_image  same system prompt and question, no image

Only Monet's runner patch (inference.apply_vllm_monet) is used. The step-logging patch
from Experiment A is NOT imported here. Engine and sampling settings are Monet's defaults
from inference/load_and_gen_vllm.py, exactly as run_vstar.py uses them.

The noise images saved here are the ones every later step uses.

The import below must stay first. vLLM starts its engine in a spawned process that
re-imports this file, and that is how the Monet patch reaches the process doing the work.
"""
import inference.apply_vllm_monet  # noqa: F401  (must precede any vllm import)

import argparse  # noqa: E402
import base64  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

import run_vstar as base  # noqa: E402  (prompt, scoring, stats and trace helpers)
from inference.load_and_gen_vllm import vllm_generate, vllm_mllm_init  # noqa: E402

SETTINGS = ("real", "noise", "no_image")
NOISE_SEED_OFFSET = 1000


def noise_path(noise_dir: Path, index: int) -> Path:
    return noise_dir / f"q{index:03d}_noise.png"


def make_noise_image(width: int, height: int, index: int) -> Image.Image:
    rng = np.random.default_rng(NOISE_SEED_OFFSET + index)
    return Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8), "RGB")


def ensure_noise_images(df, noise_dir: Path) -> None:
    """Create any missing noise PNG. Existing files are kept, so a resumed run feeds
    the same pixels; their size is checked against the original image."""
    noise_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        index = int(row["index"])
        size = Image.open(io.BytesIO(base64.b64decode(row["image"]))).size
        path = noise_path(noise_dir, index)
        if path.exists():
            if Image.open(path).size != size:
                sys.exit(f"ERROR: {path} has size {Image.open(path).size}, original is {size}")
            continue
        make_noise_image(*size, index).save(path)


def build_inputs(df, processor, setting: str, noise_dir: Path):
    inputs = []
    for _, row in df.iterrows():
        if setting == "real":
            image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
        elif setting == "noise":
            image = Image.open(noise_path(noise_dir, int(row["index"]))).convert("RGB")
        else:
            image = None
        content = [{"type": "image", "image": image}] if image is not None else []
        content.append({"type": "text", "text": base.build_user_text(row, base.row_options(row))})
        messages = [{"role": "system", "content": base.SYSTEM_PROMPT},
                    {"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        item = {"prompt": prompt}
        if image is not None:
            images, _ = process_vision_info(messages)
            item["multi_modal_data"] = {"image": images}
        inputs.append(item)
    return inputs


def preemption_count(llm):
    """Cumulative preemptions from vLLM's own metrics (no runner patch). None if unavailable."""
    try:
        return int(sum(getattr(m, "value", 0) or 0 for m in llm.get_metrics()
                       if "num_preemptions" in m.name))
    except Exception as e:  # metrics are informational only
        print(f"Could not read vLLM preemption metrics: {e!r}")
        return None


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"settings": {}}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def setting_stats(summary: dict, records: list) -> dict:
    started = summary["latent_applied"]
    not_started = summary["latent_not_applied"]
    n = summary["questions"]
    return {
        "questions": n,
        "correct": summary["correct"],
        "accuracy_pct": summary["accuracy_pct"],
        "no_readable_answer": summary["no_readable_answer"],
        "hit_token_limit": summary["hit_token_limit"],
        "latent_started": started["total"],
        "latent_started_pct": round(100 * started["total"] / n, 2) if n else None,
        "correct_when_started": started["correct"],
        "accuracy_when_started_pct": started["accuracy_pct"],
        "not_started": not_started["total"],
        "correct_when_not_started": not_started["correct"],
        "accuracy_when_not_started_pct": not_started["accuracy_pct"],
        "by_category": summary["by_category"],
    }


def write_comparison(out: Path, state: dict, per_setting_records: dict) -> None:
    done = [s for s in SETTINGS if state["settings"].get(s, {}).get("status") == "done"]
    stats = {s: state["settings"][s]["stats"] for s in done}
    (out / "summary.json").write_text(json.dumps(stats, indent=2))

    def cell(v):
        return "" if v is None else str(v)

    rows = [
        ("Questions", "questions"), ("Correct", "correct"), ("Accuracy %", "accuracy_pct"),
        ("No readable answer", "no_readable_answer"), ("Hit token limit", "hit_token_limit"),
        ("Latent reasoning started", "latent_started"), ("Started %", "latent_started_pct"),
        ("Correct when started", "correct_when_started"),
        ("Accuracy when started %", "accuracy_when_started_pct"),
        ("Not started", "not_started"), ("Correct when not started", "correct_when_not_started"),
        ("Accuracy when not started %", "accuracy_when_not_started_pct"),
        ("vLLM preemptions", None),
    ]
    lines = ["| | " + " | ".join(done) + " |", "|---|" + "---|" * len(done)]
    for label, key in rows:
        if key is None:
            vals = [cell(state["settings"][s].get("preemptions")) for s in done]
        else:
            vals = [cell(stats[s][key]) for s in done]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    table = "\n".join(lines)
    (out / "summary.md").write_text(f"# Step 0: Monet-7B on V*, three settings\n\n{table}\n",
                                    encoding="utf-8")
    print(table)

    # One row per question, side by side across the finished settings.
    if per_setting_records:
        by_index = {}
        for s, recs in per_setting_records.items():
            for r in recs:
                row = by_index.setdefault(r["index"], {"index": r["index"], "problem_id":
                                                       f"vstar_{r['index']:03d}",
                                                       "category": r["category"],
                                                       "answer": r["answer"]})
                row[f"{s}_prediction"] = r["prediction"] or ""
                row[f"{s}_correct"] = r["correct"]
                row[f"{s}_latent_segments"] = r["latent_segments"]
                row[f"{s}_latent_token_count"] = r["latent_token_count"]
                row[f"{s}_finish_reason"] = r["finish_reason"]
        cols = ["index", "problem_id", "category", "answer"] + [
            f"{s}_{k}" for s in done for k in
            ("prediction", "correct", "latent_segments", "latent_token_count", "finish_reason")]
        with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(by_index[i] for i in sorted(by_index))


def label_traces(trace_dir: Path, setting: str, records: list) -> None:
    """run_vstar's traces don't name the image setting; add it under the title."""
    image = {"real": "original V* image", "no_image": "none (text only)"}
    for r in records:
        path = trace_dir / f"vstar_{r['index']:03d}.md"
        title, rest = path.read_text(encoding="utf-8").split("\n", 1)
        note = image.get(setting) or f"seeded noise, `{r.get('image_path', '')}`"
        path.write_text(f"{title}\n\n**Setting:** `{setting}`; **image:** {note}\n{rest}",
                        encoding="utf-8")


def read_records(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="VStarBench.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N V* questions (0 = all 191)")
    ap.add_argument("--settings", default=",".join(SETTINGS),
                    help="comma-separated subset of real,noise,no_image")
    ap.add_argument("--force", action="store_true", help="rerun settings already marked done")
    args = ap.parse_args()

    base.check_patch()  # Monet's runner patch only
    if os.environ.get("MONET_STEP_LOG"):
        sys.exit("ERROR: MONET_STEP_LOG is set. Step 0 must run with Monet's patch only.")
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    bad = [s for s in settings if s not in SETTINGS]
    if bad:
        sys.exit(f"ERROR: unknown settings {bad}; choose from {SETTINGS}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = load_state(state_path)
    if state.get("limit", args.limit) != args.limit:
        sys.exit(f"ERROR: {out} was started with --limit {state['limit']}; "
                 f"use the same limit or another output folder.")
    state["limit"] = args.limit
    state["latent_size"] = int(os.environ["LATENT_SIZE"])
    save_state(state_path, state)

    df = pd.read_csv(args.data, sep="\t")
    if args.limit:
        df = df.head(args.limit)
    noise_dir = out / "noise_images"
    ensure_noise_images(df, noise_dir)
    print(f"{len(df)} questions; noise images in {noise_dir}")

    todo = [s for s in settings
            if args.force or state["settings"].get(s, {}).get("status") != "done"]
    for s in settings:
        if s not in todo:
            print(f"{s}: already done (state.json), skipped. Use --force to rerun.")

    if todo:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        # Same engine call as run_vstar.py: Monet's defaults, max_model_len raised to 16384.
        llm, sampling_params = vllm_mllm_init(
            args.model, tp=1, gpu_memory_utilization=0.85, max_model_len=16384)
        for s in todo:
            state["settings"][s] = {"status": "running",
                                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_state(state_path, state)
            before = preemption_count(llm)
            inputs = build_inputs(df, processor, s, noise_dir)
            outputs = vllm_generate(inputs, sampling_params, llm)
            after = preemption_count(llm)
            records, summary = base.score(df, outputs, processor.tokenizer)
            for r, o in zip(records, outputs):
                # Token IDs come from vLLM's normal output; no runner patch is involved.
                ids = list(o.outputs[0].token_ids)
                r["completion_token_ids"] = ids
                r["latent_start_steps"] = [i for i, t in enumerate(ids) if t == base.LATENT_START_ID]
                r["latent_end_steps"] = [i for i, t in enumerate(ids) if t == base.LATENT_END_ID]
                r["setting"] = s
                if s == "noise":
                    r["image_path"] = str(noise_path(noise_dir, r["index"]).relative_to(out))
            summary["latent_size"] = state["latent_size"]
            summary["setting"] = s

            d = out / s
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            (d / "summary.json").write_text(json.dumps(summary, indent=2))
            base.write_traces(records, d, state["latent_size"])
            label_traces(d / "traces", s, records)

            state["settings"][s] = {
                "status": "done",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "preemptions": None if before is None or after is None else after - before,
                "stats": setting_stats(summary, records),
            }
            save_state(state_path, state)
            print(f"{s}: {summary['correct']}/{summary['questions']} correct, "
                  f"latent started in {summary['latent_applied']['total']}")

    per_setting = {s: read_records(out / s / "predictions.jsonl") for s in SETTINGS
                   if state["settings"].get(s, {}).get("status") == "done"}
    write_comparison(out, state, per_setting)


if __name__ == "__main__":
    main()

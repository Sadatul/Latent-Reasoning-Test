"""Monet-7B on V* (VStarBench). Runs inside the Modal container from the Monet repo root.

The import below must stay first. vLLM starts its engine in a spawned process that
re-imports this file, and that is how the Monet patch reaches the process doing the work.
"""
import inference.apply_vllm_monet  # noqa: F401  (must precede any vllm import)

import argparse
import base64
import io
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from inference.load_and_gen_vllm import seed as ENGINE_SEED
from inference.load_and_gen_vllm import vllm_generate, vllm_mllm_init

# From the "Note that" section of the Monet README.
SYSTEM_PROMPT = (
    "You are an expert multimodal large language model designed to reason with latent visual embeddings. You are required to answer the question "
    "based on the image provided. Put your final answer in \\boxed{}."
)
LATENT_RE = re.compile(r"<abs_vis_token>.*?(?:</abs_vis_token>|$)", re.DOTALL)
LATENT_START_ID = int(os.environ["LATENT_START_ID"])  # set by the Monet patch
LATENT_END_ID = int(os.environ["LATENT_END_ID"])


def check_patch():
    """The upstream patch only prints on failure, so a silent fallback to plain Qwen is possible."""
    module = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    if "monet_gpu_model_runner" not in (getattr(module, "__file__", "") or ""):
        sys.exit("Monet vLLM patch is not active. Check the '[Monet]' lines in the log.")
    if int(os.environ.get("LATENT_SIZE", "0")) <= 0:
        sys.exit("LATENT_SIZE must be positive. The runner defaults to 0, which cuts latents short.")


def last_boxed(text):
    """Return the contents of the last \\boxed{...}, allowing nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    begin, depth = i, 1
    while i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[begin:i]
        i += 1
    return None


def extract_choice(text, options):
    """Map a model response to an option letter, or None if it can't be read unambiguously."""
    boxed = last_boxed(LATENT_RE.sub(" ", text))
    if boxed is None:
        return None
    value = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", boxed).strip(" $\n\t.")
    for letter, option in options.items():
        if value.lower() == str(option).strip().rstrip(".").lower():
            return letter
    match = re.fullmatch(r"\(?([A-Za-z])\)?", value) or re.match(r"\(?([A-Z])[).:]", value)
    if match and match.group(1).upper() in options:
        return match.group(1).upper()
    return None


def build_user_text(row, options):
    """Same wording as VLMEvalKit's ImageMCQDataset.build_prompt."""
    lines = [f"Question: {row['question']}", "Options:"]
    lines += [f"{letter}. {option}" for letter, option in options.items()]
    lines.append("Please select the correct answer from the options above. ")
    return "\n".join(lines) + "\n"


def row_options(row):
    return {
        letter: row[letter]
        for letter in string.ascii_uppercase
        if letter in row and not pd.isna(row[letter])
    }


def build_inputs(df, processor):
    inputs = []
    for _, row in df.iterrows():
        image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_user_text(row, row_options(row))},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": images}})
    return inputs


def token_stats(token_ids, added_tokens, eos_token_id):
    """Count latent tokens and visible special tokens in the generated ids."""
    latent_tokens, segments, inside = 0, 0, False
    special = Counter()
    for token_id in token_ids:
        if inside and token_id != LATENT_END_ID:
            latent_tokens += 1
            continue
        if token_id == LATENT_START_ID:
            inside = True
            segments += 1
        elif token_id == LATENT_END_ID:
            inside = False
        if token_id in added_tokens and token_id != eos_token_id:
            special[added_tokens[token_id].content] += 1
    return latent_tokens, segments, dict(special)


def group_stats(records):
    total = len(records)
    correct = sum(r["correct"] for r in records)
    return {
        "total": total,
        "correct": correct,
        "no_readable_answer": sum(r["prediction"] is None for r in records),
        "accuracy_pct": round(100 * correct / total, 2) if total else None,
    }


def score(df, outputs, tokenizer):
    added_tokens = tokenizer.added_tokens_decoder
    records, by_category = [], defaultdict(lambda: [0, 0])
    for (_, row), output in zip(df.iterrows(), outputs):
        completion = output.outputs[0]
        options = row_options(row)
        prediction = extract_choice(completion.text, options)
        answer = str(row["answer"]).strip()
        correct = prediction == answer
        category = str(row.get("category", "all"))
        by_category[category][0] += correct
        by_category[category][1] += 1
        token_ids = list(completion.token_ids)
        latent_tokens, segments, special = token_stats(
            token_ids, added_tokens, tokenizer.eos_token_id
        )
        if correct:
            status = "correct"
        else:
            status = "no_answer" if prediction is None else "wrong_answer"
        records.append(
            {
                "index": int(row["index"]),
                "category": category,
                "answer": answer,
                "prediction": prediction,
                "correct": bool(correct),
                "status": status,
                "boxed": last_boxed(LATENT_RE.sub(" ", completion.text)),
                "latent_token_count": latent_tokens,
                "latent_segments": segments,
                "added_special_tokens": special,
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(token_ids),
                "finish_reason": completion.finish_reason,
                "user_prompt": build_user_text(row, options),
                "raw_response": completion.text,
                "response": LATENT_RE.sub("<latent>", completion.text),
            }
        )
    n = len(records)
    hits = sum(r["correct"] for r in records)
    with_latents = [r for r in records if r["latent_token_count"] > 0]
    without_latents = [r for r in records if r["latent_token_count"] == 0]
    summary = {
        "questions": n,
        "correct": hits,
        "accuracy_pct": round(100 * hits / n, 2),
        "no_readable_answer": sum(r["prediction"] is None for r in records),
        "responses_using_latents": len(with_latents),
        "latent_applied": group_stats(with_latents),
        "latent_not_applied": group_stats(without_latents),
        "hit_token_limit": sum(r["finish_reason"] == "length" for r in records),
        "by_category": {
            name: {"correct": c, "total": t, "accuracy_pct": round(100 * c / t, 2)}
            for name, (c, t) in sorted(by_category.items())
        },
    }
    return records, summary


def code_block(text, language):
    """Fence text with more backticks than it contains, so model output can't break the file."""
    longest = max(map(len, re.findall(r"`+", text)), default=0)
    ticks = "`" * max(4, longest + 1)
    return f"{ticks}{language}\n{text}\n{ticks}"


def json_block(data):
    return code_block(json.dumps(data, indent=2, ensure_ascii=False), "json")


def render_trace(record, problem_id, latent_size):
    label = record["status"].upper()
    passed = int(record["correct"])
    meta = {
        "id": problem_id,
        "dataset": "vstar",
        "model": "monet-7b",
        "index": record["index"],
        "category": record["category"],
        "latent_size": latent_size,
        "format": "mcq",
        "status": record["status"],
        "passes": 1,
        "num_attempts_passed": passed,
    }
    attempt = {
        "attempt": 0,
        "seed": ENGINE_SEED,
        "status": record["status"],
        "finish_reason": record["finish_reason"],
        "prompt_tokens": record["prompt_tokens"],
        "completion_tokens": record["completion_tokens"],
        "latent_token_count": record["latent_token_count"],
        "latent_segments": record["latent_segments"],
        "added_special_tokens": record["added_special_tokens"],
    }
    result = {
        "expected": record["answer"],
        "got": record["prediction"],
        "error": None if record["prediction"] else "no readable \\boxed{} answer",
    }
    return "\n".join(
        [
            f"# {problem_id} \u2014 {label} ({passed}/1 attempts passed)",
            "",
            json_block(meta),
            "",
            "## System prompt",
            "",
            code_block(SYSTEM_PROMPT, "text"),
            "",
            "## User prompt",
            "",
            code_block(record["user_prompt"], "text"),
            "",
            f"## Attempt 1 \u2014 {label}",
            "",
            json_block(attempt),
            "",
            "### Model output (full reasoning trace, special tokens included)",
            "",
            code_block(record["raw_response"], "text"),
            "",
            "### Extracted final answer",
            "",
            code_block(record["boxed"] if record["boxed"] is not None else "(none)", "text"),
            "",
            "### Test result",
            "",
            json_block(result),
            "",
        ]
    )


def write_traces(records, out, latent_size):
    traces = out / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for record in records:
        problem_id = f"vstar_{record['index']:03d}"
        text = render_trace(record, problem_id, latent_size)
        (traces / f"{problem_id}.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 191 questions")
    args = parser.parse_args()

    check_patch()
    df = pd.read_csv(args.data, sep="\t")
    if args.limit:
        df = df.head(args.limit)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    inputs = build_inputs(df, processor)

    # Monet's own engine and sampling settings. max_model_len rises from its 4096 default
    # so a large image (up to 8192 tokens) plus a 4096-token answer still fits.
    llm, sampling_params = vllm_mllm_init(
        args.model, tp=1, gpu_memory_utilization=0.85, max_model_len=16384
    )
    records, summary = score(df, vllm_generate(inputs, sampling_params, llm), processor.tokenizer)
    latent_size = int(os.environ["LATENT_SIZE"])
    summary["latent_size"] = latent_size

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_traces(records, out, latent_size)
    print(json.dumps(summary, indent=2))
    applied = summary["latent_applied"]
    print(f"Latent reasoning applied on {applied['total']} problems, solved {applied['correct']}.")


if __name__ == "__main__":
    main()
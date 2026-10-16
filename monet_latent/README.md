# Monet latent reasoning on llama.cpp (GGUF) + V*

`latent_probe/` showed that quantized Monet-7B *does* emit its latent-start token
`<abs_vis_token>` (id 151666) on V* images. Plain llama.cpp then just treats it as
text. This folder runs Monet's actual latent mode on the GGUF model, so the model
reasons with latent vectors as it does in Monet's patched vLLM.

| File | What it is |
|---|---|
| `monet_llamacpp.py` | The engine: Monet's vLLM latent decoding ported to llama-cpp-python |
| `run_vstar.py` | V* evaluation with resume, traces, and accuracy plus latent statistics |

## How latent mode works (and how it is ported)

Monet's runner ([`monet_gpu_model_runner.py`](https://github.com/NOVAglow646/Monet/blob/main/inference/vllm/monet_gpu_model_runner.py),
the `ABS-VIS` blocks) does this at every decode step:

```
sample tid
if not active and tid == <abs_vis_token>:                 active = True
elif active and (tid == </abs_vis_token> or n >= LATENT_SIZE):
    active = False; n = 0; tid = </abs_vis_token>          # forced end token
if active: pending = last hidden state of this step; n += 1
next step: if active, the new token's input embedding is replaced by `pending`
```

The llama.cpp version does the same with:

- **Hidden state.** `llama_set_embeddings(ctx, True)` makes each decode also return
  `result_norm`, the output of the final RMSNorm. That's the same tensor vLLM uses
  as `hidden_states` (the one the lm_head multiplies).
- **Feeding it back.** A `llama_batch` with `embd` set instead of `token`. Qwen2.5-VL
  uses M-RoPE, so an embedding batch carries 4 positions per token, all equal to
  the text position (what llama.cpp does for text tokens).
- **Image.** llama.cpp's mtmd with the mmproj GGUF, as in `latent_probe`. Text after
  the image uses M-RoPE positions (`n_past + max(grid_h, grid_w)`), as in HF/vLLM.
- **Sampling.** Done in numpy with vLLM's rules (repetition penalty over prompt and
  output tokens, then temperature, top-k, top-p), with Monet's defaults: `temperature
  0.1, top_k 50, top_p 0.8, repetition_penalty 1.01, max_tokens 4096, LATENT_SIZE 10`.
  Tokens sampled during latent steps are placeholders. As in vLLM they appear in the
  raw output and count towards `max_tokens`, but the model never sees them. If one
  of them is EOS, generation stops (vLLM does this too).

Checked on CPU with Qwen2-VL-2B (same architecture, no Monet tokens):

- Feeding a token's embedding row through the embedding path gives bit-identical
  logits and hidden states to feeding the token id. So the embedding input and
  M-RoPE positions are right.
- `output.weight @ hidden` matches llama.cpp's logits (correlation 0.9992, same
  top-10 tokens; the difference is quantized matmul).
- The state machine produces the expected segments.

It was **not** run with Monet itself or on a GPU (this machine has neither).

### `--latent-mode`

| Mode | Behaviour |
|---|---|
| `vllm` (default) | Exactly the runner above. Side effect of that code: the hidden state that picked `<abs_vis_token>` replaces the embedding of `<abs_vis_token>` itself, so the K latents sit at `[start, p1 … p(K-1)]` |
| `train` | Monet's training layout (`modeling_qwen2_5_vl_monet.py`: each latent = previous position's hidden state): `<abs_vis_token>` keeps its own embedding, then K latents, then `</abs_vis_token>` |
| `off` | No latent mode: `<abs_vis_token>` is an ordinary token (plain llama.cpp). The ablation |

`vllm` is what Monet's published numbers were produced with. `train` is there in
case the off-by-one matters; compare the two on the probe set.

## Run (GPU machine)

Same environment as `latent_probe` (llama-cpp-python with CUDA, pillow: `bash setup_eval.sh`).

```bash
source env.sh
.venv/bin/python latent_probe/prepare_vstar.py                  # once: the 20-question probe set (if not there yet)

.venv/bin/python monet_latent/run_vstar.py --limit 2            # smoke test
.venv/bin/python monet_latent/run_vstar.py                      # probe set, Monet Q8_0, latent mode on
.venv/bin/python monet_latent/run_vstar.py --latent-mode off    # same questions, latent mode off
.venv/bin/python monet_latent/run_vstar.py --model qwen25vl     # Qwen2.5-VL baseline
```

**Full V\*** (191 questions) is one flag: `--dataset full`. The first run builds
`latent_probe/data/vstar_full.jsonl` and downloads the images (into `hf_cache/`).
Or build it beforehand with `latent_probe/prepare_vstar.py --full`.

```bash
.venv/bin/python monet_latent/run_vstar.py --dataset full
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `probe` | `probe` (20 questions, `vstar_subset.jsonl`) or `full` (191, `vstar_full.jsonl`) |
| `--model` | `monet` | `monet` or `qwen25vl` (Qwen is always `--latent-mode off`) |
| `--quant`, `--mmproj` | `Q8_0`, `f16` | GGUF files from `mradermacher/*-GGUF`; `--model-path` / `--mmproj-path` for local files |
| `--latent-mode` | `vllm` | `vllm`, `train`, `off` (see above) |
| `--latent-size` | 10 | Latent vectors per segment (Monet's `LATENT_SIZE`) |
| `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty` | 0.1, 50, 0.8, 1.01 | Monet's sampling. `--temperature 0` = greedy |
| `--seed` | 0 | Question `q` uses seed `seed + question_id`, so reruns and resumes give the same samples |
| `--max-tokens` | 4096 | Generated tokens, placeholders included |
| `--max-pixels` / `--min-pixels` | 2048·28² / 256·28² | Image resize (Qwen's smart_resize). Monet uses max 8192·28² ≈ 6.4 MP; that needs `--n-ctx 16384` and more VRAM for the vision encoder |
| `--n-ctx` | 8192 | Image + prompt + output |
| `--n-gpu-layers` | -1 | -1 = all on GPU |
| `--no-flash-attn` | | Turn off flash attention |
| `--limit`, `--retry-errors`, `--force`, `--verbose`, `--output-dir` | | As in `src/run_eval.py` |

Pause with Ctrl+C, then rerun the same command to resume (`state.json`). Each
result is written before it is marked done, so a crash loses at most the question
in progress. A folder won't resume with different settings unless you pass `--force`.

## Output

```
monet_latent/results/<model>-<quant>-<latent tag>/<dataset>/
├── config.json, state.json
├── results.jsonl     one record per question
├── summary.json      accuracy, latent use, per category
└── traces/<id>.md    prompt, output, latent segment table, raw output
```

`<latent tag>`: `latent10` (vllm mode), `latent10-train`, `nolatent`, plus `-greedy` with `--temperature 0`.

Per question (`results.jsonl`):

| Field | Meaning |
|---|---|
| `label`, `pred`, `correct` | Gold letter, predicted letter, verdict |
| `answer_source`, `boxed` | How `pred` was read: `boxed_letter`, `boxed_text` (Monet often boxes the option text, e.g. `\boxed{green}`), `fallback_letter` ("answer is (B)"), `boxed_unmatched`, `no_answer`; and the raw `\boxed{}` content |
| `clean_output` | The reasoning trace and answer, each latent segment shown as `<abs_vis_token><latent></abs_vis_token>` (Monet README's post-processing). Answers are read from this |
| `raw_output` | Same, with the placeholder tokens, as vLLM returns it |
| `fired`, `n_segments`, `n_latents_total` | Whether latent mode was entered, how often, and how many vectors were fed |
| `segments` | Per segment: `start_step`, `n_latents`, `end` (`latent_size`, `end_token`, `eos_sampled`, `max_tokens`), `placeholder_text` |
| `token_ids`, `token_kinds` | Every output token and its kind (`text`, `start`, `placeholder`, `end`) |
| `p_latent_start_max`, `p_latent_start_per_step` | Probability of `<abs_vis_token>` at each step |
| `finish_reason` | `stop`, `length` (`--max-tokens`), `context_full` (`--n-ctx`) |
| `image`, `image_size_orig`, `image_size_used`, `prompt_tokens` | Image grid and token count, sizes, KV cells used by the prompt |
| `prompt`, `seed`, `error` | Full prompt sent, sampling seed, traceback if it failed |

`summary.json` has `accuracy`, `no_answer`, `fire_rate`, `accuracy_when_fired` /
`accuracy_when_not_fired`, segment end reasons and finish reasons, overall and per
category, plus `fired_ids` and `wrong_ids` to know which traces to read.

Prompt: Monet's VLMEvalKit system prompt ("… Put your final answer in \boxed{}."),
then the question followed by the image, as in Monet's inference example.

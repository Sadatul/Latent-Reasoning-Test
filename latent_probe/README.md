# Latent-token probe: does quantized Monet ever enter latent mode?

On LiveCodeBench, Monet-7B (GGUF, llama.cpp) never emits its latent-start token
(id 151666). There are two possible reasons:

1. **Quantization** removed or suppressed the behaviour.
2. **No image.** Monet learned to start latent (visual) reasoning only when there is an
   image to reason about, so text-only coding never triggers it.

This probe runs the **same quantized GGUF** on a few V* image questions, the benchmark
Monet was trained and evaluated on, and counts how often the token appears. It does not
measure accuracy.

llama.cpp has no latent mode: when the token is emitted it is fed back as an ordinary
token. That doesn't matter here, because everything up to the first latent token is
identical to Monet's own patched vLLM runner. Whether the token is emitted is exactly
what we measure.

## Run (GPU machine)

```bash
source env.sh
uv pip install --python .venv pillow                     # or rerun setup_eval.sh

.venv/bin/python latent_probe/prepare_vstar.py           # 20 questions, 10 per category (~34 MB)
.venv/bin/python latent_probe/run_probe.py --limit 2     # smoke test (downloads Monet Q8_0 + mmproj if needed)
.venv/bin/python latent_probe/run_probe.py               # all 20
```

`prepare_vstar.py` downloads only `test_questions.jsonl` and the selected images from
`craigwu/vstar_bench`, sampled with a fixed seed (`--n`, `--seed`). It rewrites each
question into Monet's prompt format: V*'s "Answer with the option's letter ... directly"
asks for no reasoning, so it is replaced by Monet's "Put your final answer in \boxed{}."

`run_probe.py` uses Monet's evaluation system prompt and greedy decoding. It records the
probability of the latent token at every step, not only whether it was picked.

| Argument | Default | Meaning |
|---|---|---|
| `--quant` | `Q8_0` | Monet GGUF to use. Each quant gets its own output folder |
| `--mmproj` | `f16` | Vision projector (`f16` or `Q8_0`) |
| `--max-pixels` | 1,605,632 | Larger images are downscaled to this (about 2048 visual tokens) |
| `--n-ctx` | 8192 | Context window (image + text + answer) |
| `--max-tokens` | 1024 | Max generated tokens per question |
| `--limit` | | Only the first N questions |
| `--retry-errors`, `--force`, `--verbose` | | Same as `src/run_eval.py` |

Pause with Ctrl+C and rerun the same command to resume (`state.json`), as in the main eval.

## Output

```
latent_probe/results/monet-<quant>/
├── config.json, state.json
├── results.jsonl   one record per question
├── summary.json    questions that fired, overall and per category; fired_ids
└── traces/<id>.md  prompt, full output with special tokens, latent stats
```

Per question:

| Field | Meaning |
|---|---|
| `fired`, `n_fires`, `first_fire_step` | Whether / how often token 151666 was generated, and at which output step first |
| `n_latent_end` | How often the latent-end token 151667 was generated |
| `p_latent_max`, `p_latent_max_step` | Highest probability the model gave token 151666 at any step |
| `best_rank_latent` | Best rank of token 151666 among all tokens (0 = it was the top choice) |
| `p_latent_per_step` | Probability of token 151666 at every output step |
| `raw_output` | Full answer; the latent token shows inline as its text |
| `image_size_orig`, `image_size_used`, `prompt_tokens` | Image before/after downscaling, prompt length incl. image tokens |

## Reading the result

- **Fires on some V\* questions** → quantization did not remove the behaviour. Not
  firing on LiveCodeBench means text-only problems don't invoke it.
- **Never fires** → inconclusive (quantization, llama.cpp's vision path, or prompt
  differences). Check `p_latent_max` and `best_rank_latent`: if the token often ranks near
  the top on V*, the model "wants" latent mode with images even if greedy decoding
  never picks it.

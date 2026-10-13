# Monet vs Qwen2.5-VL: coding eval

Compares **Qwen2.5-VL-7B-Instruct** with **Monet-7B** (Qwen2.5-VL-7B fine-tuned for
latent visual reasoning) on text-only coding problems: LiveCodeBench-lite subsets and
HumanEval+. Both models run as GGUF files through
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python), so a 16 GB GPU is enough.

Everything downloaded (data, models) stays inside this folder, so deleting the folder
removes it all.

## Requirements

- bash, git
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For evaluation (GPU machine only):
  - NVIDIA GPU with 16 GB VRAM and a recent driver (`nvidia-smi` works)
  - CUDA toolkit 12.x with `nvcc` (`nvcc --version` works; usually `/usr/local/cuda/bin/nvcc`)
  - `cmake` ≥ 3.24, `gcc`/`g++`, `make` (Ubuntu: `sudo apt install build-essential cmake`)

## 1. Data

```bash
bash setup.sh
```

This creates `.venv/` with uv (Python 3.13), installs `huggingface_hub`, and runs
`prepare_datasets.py`, which downloads LiveCodeBench-lite and HumanEval+ into `data/`
(about 4.5 GB). See `DATASET.md` for the file formats.

## 2. llama.cpp (GPU machine)

`llama-cpp-python` is the Python binding for llama.cpp. It ships llama.cpp as source
and compiles it on install; set `CMAKE_ARGS` to get a CUDA build. There are no
prebuilt CUDA wheels for Python 3.13, so it is always built from source.

Either run the script:

```bash
bash setup_eval.sh
```

or do the same steps by hand:

```bash
cd simple-latent-reasoning-check
source env.sh                                    # keep downloads inside this folder

export CUDACXX=$(which nvcc)                     # or /usr/local/cuda/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
  uv pip install --python .venv --no-cache --reinstall-package llama-cpp-python \
  llama-cpp-python numpy

# check: must print True
.venv/bin/python -c "import llama_cpp; print(llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
```

- The compile takes several minutes.
- `--no-cache` / `--reinstall-package` stop uv from reusing a CPU-only build.
- `numpy` is needed by the HumanEval+ test harness.
- `CMAKE_CUDA_ARCHITECTURES=native` builds only for the GPU in the machine. With
  cmake < 3.24, use the GPU's compute capability instead, e.g. `=86` (RTX 30xx, A4000),
  `=89` (RTX 40xx), `=75` (T4).

Troubleshooting:

| Symptom | Fix |
|---|---|
| `nvcc not found` / `No CMAKE_CUDA_COMPILER could be found` | Install the CUDA toolkit; `export CUDACXX=/usr/local/cuda/bin/nvcc` and `export PATH=/usr/local/cuda/bin:$PATH` |
| check prints `False` | Built without CUDA. Rerun the install command exactly as above (with `CMAKE_ARGS`) |
| `unknown model architecture: 'qwen2vl'` when loading | llama-cpp-python is too old. Rerun the install to get the latest version |
| CUDA out of memory | Use a smaller quant (`--quant Q6_K`) or `--n-ctx 6144` |

## 3. Run

One model on one dataset per run:

```bash
source env.sh

# smoke test: downloads the model (8.1 GB, Q8_0) and runs 2 problems
.venv/bin/python src/run_eval.py --model qwen25vl --dataset humanevalplus --limit 2

.venv/bin/python src/run_eval.py --model qwen25vl --dataset lcb_easy
.venv/bin/python src/run_eval.py --model monet    --dataset lcb_easy
.venv/bin/python src/run_eval.py --model qwen25vl --dataset humanevalplus
.venv/bin/python src/run_eval.py --model monet    --dataset humanevalplus
.venv/bin/python src/run_eval.py --model qwen25vl --dataset lcb_post_qwen25vl
.venv/bin/python src/run_eval.py --model monet    --dataset lcb_post_qwen25vl
```

| Argument | Default | Meaning |
|---|---|---|
| `--model` | required | `qwen25vl` or `monet` |
| `--dataset` | required | `lcb_easy` (322), `lcb_post_qwen25vl` (131), `humanevalplus` (164), `lcb_full` (all LiveCodeBench, 1,055) |
| `--quant` | `Q8_0` | GGUF file to use (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, …). Q8_0 is 8.1 GB and fits 16 GB with the KV cache |
| `--model-path` | | Use a local `.gguf` file instead of downloading |
| `--output-dir` | `results/<model>-<quant>/<dataset>` | Where results go (`<dataset>-prompt<N>` for `--prompt-style` N > 0) |
| `--n-ctx` | 8192 | Context window (prompt + answer) |
| `--max-tokens` | 4096 | Max generated tokens per attempt |
| `--prompt-style` | 0 | 0 = original prompt; 1 = stricter prompt that insists on a final code block (see below). Default set by `PROMPT_STYLE` in `src/run_eval.py` |
| `--passes` | 1 | Attempts per problem, the k in pass@k (any k ≥ 1). k > 1 needs `--temperature` > 0 |
| `--temperature` | 0 | 0 = greedy (deterministic). `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--seed` are also available |
| `--n-gpu-layers` | -1 | -1 = whole model on GPU, 0 = CPU only |
| `--test-timeout` | 6 | Seconds per test case |
| `--limit` | | Only the first N problems |
| `--retry-errors` | | Re-run attempts whose status is `error` |
| `--rescore` | | Re-run the tests on saved code without generating again |
| `--force` | | Resume even if settings differ from the saved `config.json` |

Models come from `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF` and `mradermacher/Monet-7B-GGUF`
and are cached in `hf_cache/hub/`. Only the language model file is used (no `mmproj`), since
the tasks are text-only. Both models get exactly the same prompts and sampling settings.

### Prompt styles

Prompts are defined in `PROMPT_STYLES` in `src/run_eval.py`:

- **0** (default): the original prompt: reason step by step, then give the solution in a
  ```` ```python ```` block. Works well for Qwen2.5-VL.
- **1**: for Monet, which often stops without writing code under style 0 (`no_code`). It asks
  for short reasoning, says only the code block is graded and a missing block scores zero,
  forbids `\boxed{}` / plain-text answers, and ends with an explicit answer template.

```bash
.venv/bin/python src/run_eval.py --model monet --dataset lcb_easy --prompt-style 1   # -> results/monet-Q8_0/lcb_easy-prompt1
```

Change `PROMPT_STYLE` at the top of `src/run_eval.py` to change the default. The style is
saved in `config.json` and in every record, and a folder will not resume with a different
style. For a fair comparison, run both models with the same style.

### pass@k

`--passes K` samples K answers per problem; attempt `i` (0-based) uses seed `--seed + i`,
so a rerun gives the same attempts. Greedy decoding would give K identical answers, so
K > 1 requires a temperature, e.g.:

```bash
.venv/bin/python src/run_eval.py --model qwen25vl --dataset lcb_easy --passes 5 --temperature 0.2 \
  --output-dir results/qwen25vl-Q8_0/lcb_easy-pass5
```

A problem counts as solved if any attempt passes. `summary.json` reports pass@1 … pass@K
with the unbiased estimator of Chen et al. (2021), so one `--passes 5` run also gives
pass@1 and pass@3. Use a separate `--output-dir` per temperature: a folder created with
greedy decoding will not resume with a different temperature. Raising `--passes` on an
existing folder (same settings otherwise) tops up every problem with the missing attempts.

### Pause and resume

Press Ctrl+C to pause. To continue, run the same command again. It reads `state.json` in
the output folder and skips finished problems. The same applies after a crash or power
loss: each result is written to disk before the problem is marked done, and each finished
attempt of the running problem is kept in `state.json` (`in_progress`), so at most the
attempt that was running is lost and redone. The script will not resume a folder created
with different model/dataset/sampling settings unless you pass `--force`.

## 4. Output

```
results/<model>-<quant>/<dataset>/
├── config.json     run settings: model file, sampling, prompts, llama-cpp-python version
├── state.json      finished problem ids, last_completed, in_progress attempts, status (running/paused/done)
├── results.jsonl   one JSON record per problem (for analysis scripts)
├── summary.json    pass@1 … pass@K overall and by difficulty / platform / format, status counts
└── traces/<id>.md  one readable file per problem, for manual review
```

Problem ids are the same for both models (`atcoder_abc301_a`, `leetcode_2727`,
`HumanEval_0`), so `results/qwen25vl-Q8_0/lcb_easy/traces/X.md` and
`results/monet-Q8_0/lcb_easy/traces/X.md` are the same problem.

Each record in `results.jsonl` is one problem:

| Field | Content |
|---|---|
| `id`, `title`, `platform`, `difficulty`, `contest_date`, `format` | Problem info (`format`: `functional`, `stdin` or `humaneval`) |
| `user_prompt`, `prompt_text` | The question as sent, and the full prompt after the chat template |
| `passes`, `num_attempts_passed` | Number of attempts, and how many passed |
| `passed`, `status` | `true` / `pass` if any attempt passed; otherwise `status` is the first attempt's status |
| `attempts` | List with one entry per attempt, fields below |

Each entry of `attempts` has:

| Field | Content |
|---|---|
| `attempt`, `seed` | Attempt index (0-based) and the sampling seed it used |
| `raw_output` | The full model output, special tokens included: the reasoning trace |
| `reasoning` | Text before the final code block |
| `final_code` | The last ```` ```python ```` block: the answer that was tested |
| `text_after_code` | Anything the model wrote after the code |
| `status` | `pass`, `wrong_answer`, `runtime_error`, `timeout`, `no_code` (no code block), `truncated` (hit `--max-tokens` before giving code), `error` (script error, see `error`) |
| `passed`, `num_tests`, `num_passed` | Verdict; tests run in order and stop at the first failure |
| `first_failure` | Failing test's input, expected and actual output, and the traceback |
| `finish_reason` | `stop` (model ended) or `length` (hit the token limit) |
| `prompt_tokens`, `completion_tokens`, `gen_seconds`, `tokens_per_second` | Cost |
| `latent_token_count`, `added_special_tokens` | Times the model emitted Monet's latent-mode token (id 151666) or other tokens Monet added to the Qwen vocabulary |

LiveCodeBench problems are tested on public and hidden tests. HumanEval+ runs the
EvalPlus `check()` harness (base + extra tests) as one test.

### Notes

- **Monet's latent mode does not run here.** Latent reasoning needs Monet's patched vLLM
  runner. Under llama.cpp, Monet is a plain text model. If it emits token 151666, the
  token is decoded as text and generation continues normally; `latent_token_count`
  records how often that happens.
- **HumanEval/32 fails for every solution.** In the `evalplus/humanevalplus` data its
  test swaps the arguments to `_poly`, so even the reference solution fails. It counts as
  a failure for both models; subtract it if you want pass@1 on the 163 scorable tasks.
- **Generated code runs on your machine** in a subprocess with a 4 GB memory limit and
  timeouts. Those limits stop runaway programs, not malicious ones.
- `lcb_full` reads all ~4.4 GB of LiveCodeBench files once at startup to index them.

## 5. Analysis (any machine)

```bash
uv pip install --python .venv matplotlib
.venv/bin/python analysis/compare_results.py      # default: qwen lcb_easy vs monet lcb_easy-prompt1
.venv/bin/python analysis/compare_results.py --qwen results/qwen25vl-Q8_0/humanevalplus \
                                             --monet results/monet-Q8_0/humanevalplus
```

Writes `results/analysis/<qwen folder>__vs__<monet folder>/`: `report.md` (agreement table,
failure categories, latent-token counts for every Monet run and the V* probe),
`qwen_pass_monet_fail.csv`, `monet_pass_qwen_fail.csv` (one row per problem, with each
attempt's status and links to both traces) and `failure_categories.png`. A problem counts
as solved if any attempt passed. The report lists settings that differ between the two
runs (e.g. prompt style), since those make the comparison unfair.

## Files

| Path | What it is |
|---|---|
| `setup.sh` | Creates the uv environment and builds the datasets |
| `setup_eval.sh` | Installs llama-cpp-python with CUDA, plus numpy and pillow |
| `env.sh` | Points Hugging Face downloads into this folder |
| `prepare_datasets.py` | Downloads data, builds subsets, writes stats |
| `src/run_eval.py` | Runs one model on one dataset, saves traces, resumes |
| `src/code_exec.py` | Runs generated code against the tests |
| `analysis/compare_results.py` | Qwen vs Monet comparison, CSVs, heatmap, latent-token counts |
| `latent_probe/` | Separate check: does quantized Monet emit its latent token on V* image questions? See its README |
| `DATASET.md` | Data locations and formats |
| `data/` | Datasets and stats (created by `setup.sh`) |
| `hf_cache/` | Hugging Face cache, including the GGUF models |
| `results/` | Evaluation output |
| `.venv/` | Python environment |

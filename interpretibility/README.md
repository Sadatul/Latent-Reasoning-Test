# Monet-7B Latent-Start Study on V* — notes

Study plan: [`PLAN.md`](PLAN.md). This file tracks status and how to reproduce
each completed step. It is separate from the main project `README.md`.

> Folder name: the plan refers to `interpretability/`; the existing folder is
> spelled `interpretibility/`, so these notes live here next to `PLAN.md`.

## Status

| Step | Status |
|---|---|
| 0. V* in three settings | Code written and tested locally; smoke test and full run not yet run on Modal |
| 1. Select question set | Done (final: system prompt confirmed) |
| 2. Experiment A (vLLM generation) | Code written and tested locally; now uses Step 0 noise images. Needs Step 0 first. Smoke test and full run not yet run on Modal |
| 3. Experiment B (Transformers replay) | Not started |
| 4. Select positions | Not started |
| 5. — | Not yet defined |

## Step 0: V* in three settings

Runs Monet-7B on V* three times: with the real image (`real`), a seeded noise
image (`noise`), and no image (`no_image`). It reports accuracy, how often
latent reasoning started, and accuracy with and without latent reasoning, for
each setting. It also produces the noise images that every later step uses.

**Only Monet's own vLLM patch is used** (`inference.apply_vllm_monet`, as
shipped at commit `0893999`). `run_step0.py` does not import
`patches/monet_step_log.py`, and it exits if `MONET_STEP_LOG` is set. The
Modal function also removes that variable from the environment.

### Files

| File | Role |
|---|---|
| `src/modal/run_step0.py` | New: runs in the container. Builds the 3 settings, generates, scores, writes traces and the comparison. |
| `src/modal/monet_vstar_modal.py` | Adds the `step0` GPU function (L40S, 6 h timeout) and the `run_step0` entrypoint. Mounts `run_step0.py`. |

`run_step0.py` reuses `run_vstar.py` unchanged: `SYSTEM_PROMPT`,
`build_user_text`, `row_options`, `score`, `write_traces` and `check_patch`.

### Run

```bash
# Smoke test: first 2 V* questions x 3 settings -> results/step0_first2
modal run src/modal/monet_vstar_modal.py::run_step0 --limit 2
modal volume get monet-vstar results/step0_first2 ./results/step0_first2

# Full run: all 191 questions x 3 settings -> results/step0
modal run --detach src/modal/monet_vstar_modal.py::run_step0
modal volume get monet-vstar results/step0 ./results/step0
```

Options:

- `--limit N` runs the first N questions and writes to `step0_firstN`.
- `--settings real,noise` runs a subset of settings.
- `--force` reruns settings that are already done.

Step 2's question set uses V* indices 1–28. A Step 0 run with `--limit 30`
(folder `step0_first30`) is enough for Step 2 if you pass
`--step0-dir step0_first30` to `run_exp_a`.

### Key settings

| Setting | Value |
|---|---|
| GPU | L40S |
| Engine | vLLM 0.10.0 + Monet's runner patch only |
| Engine settings | Monet's `vllm_mllm_init` defaults, exactly as `run_vstar.py` uses them: `max_num_seqs = 512`, engine seed 0, `max_model_len = 16384`, `gpu_memory_utilization = 0.85` |
| Sampling | Monet's `SamplingParams` unchanged (temperature 0.1, top-k 50, top-p 0.8, repetition penalty 1.01, 4096 max tokens, no per-request seed) |
| System prompt | `SYSTEM_PROMPT` from `run_vstar.py`, same in all settings |
| `LATENT_SIZE` | 10 |
| Batches | One `generate` call per setting, in the order `real`, `noise`, `no_image`. The `real` batch matches the original V* run. |
| Noise image | `np.random.default_rng(1000 + index).integers(0, 256, (H, W, 3), uint8)` at the original image's size. Saved as PNG before generation and fed from the saved PNG. |
| Image preprocessing | Same as `run_vstar.py` (`process_vision_info`, then Monet's `min_pixels` / `max_pixels`) |

### Pause and resume

`state.json` has a status per setting. A rerun into the same folder works like this:

- Settings marked `done` are skipped.
- An unfinished setting is rerun from the start.
- Existing noise PNGs are kept, and their sizes are checked.

A different `--limit` for the same folder is refused.

Because a resumed setting follows fewer earlier batches on the same engine,
its sampling is not bit-identical to an uninterrupted run.

### Outputs (`results/step0/`)

| File | Content |
|---|---|
| `noise_images/q###_noise.png` | Noise images. **Steps 2 and 3 use these.** About 10 MB each, because random pixels do not compress: ~2 GB for all 191, ~300 MB for `--limit 30`. |
| `<setting>/predictions.jsonl` | One record per question, in `run_vstar.py` format plus `setting`, `completion_token_ids`, `latent_start_steps`, `latent_end_steps` (and `image_path` for noise) |
| `<setting>/summary.json` | `run_vstar.py` summary plus `setting` |
| `<setting>/traces/vstar_###.md` | `run_vstar.py` trace format, with the setting and image named under the title |
| `summary.md` / `summary.json` | Comparison across settings (see below) |
| `per_question.csv` | One row per question: answer, plus prediction, correct, latent segments, latent tokens and finish reason per setting |
| `state.json` | Status per setting, preemption count, stats |

`summary.md` has one column per setting and these rows: questions, correct,
accuracy %, no readable answer, hit token limit, latent reasoning started (and
%), correct when started (and accuracy %), not started, correct when not
started (and accuracy %), and vLLM preemptions. "Started" means at least one
latent token, the same definition `run_vstar.py` uses.

**Token IDs and latent vectors:** Step 0 saves the generated token IDs and
the start and end positions, taken from vLLM's normal output with no patch.
It does **not** save hidden states. Capturing them would need a runner patch,
and Step 0 must run Monet's patch exactly as shipped. The latent vectors are
captured in Step 2.

**Preemptions** are read from vLLM's own metrics (`vllm:num_preemptions`)
before and after each setting. They are reported, not enforced, because
detecting them inside the runner would need a patch. If the count is above
0, some latent segments may be unreliable; note it when reading the results.

### Local tests (no GPU)

- `py_compile` passes. The Modal app lists `main`, `run_exp_a` and `run_step0`.
- Noise generation is deterministic and identical to the rule
  `default_rng(1000 + index)` that Experiment A used before.
- `write_comparison` (`summary.md`, `per_question.csv`) and the trace labels
  were tested with stubbed records.
- **Not tested here:** `run_vstar.score` on real vLLM outputs. It is the
  unchanged function from the original V* runs.
- **Not yet run:** the Modal smoke test and full run.

### Step 0 report

- **Files changed:** `src/modal/run_step0.py` (new);
  `src/modal/monet_vstar_modal.py` (`step0`, `run_step0`, mount, docstring);
  `interpretibility/PLAN.md` (Step 0 added; Steps 2 and 3 now use Step 0 noise);
  this README.
- **Commands run:** `py_compile`; Modal app import; stubbed tests of the
  noise images, the comparison outputs and the trace labels. No Modal command
  was run.
- **Outputs created:** none in `results/`.
- **Checks:** compile ✔; entrypoints ✔; noise determinism and same rule as
  before ✔; comparison table and per-question CSV ✔; Modal smoke test not run.

### Step 0 add-on: re-reading unboxed answers with gpt-oss:20b (not in the plan)

Step 0 scores only the last `\boxed{}`, so answers like `ANSWER: D. silver`
count as `no_answer`. `src/analysis/rejudge_step0.py` sends every record
scored incorrect to `gpt-oss:20b` on Ollama. The judge gets the user prompt and
Monet's output (latents shown as `<latent>`), but not the gold answer. It
returns `{"answer": <option letter or "NONE">, "evidence": <verbatim quote>}`
via a JSON-schema `format`. A record becomes correct only if the letter equals
the gold answer and the quote is found in the output. Correct records are
never touched. Uses only the Python standard library; runs on any machine with
Ollama.

```bash
ollama pull gpt-oss:20b
python3 rejudge_step0.py --step0 step0 --smoke 3   # 3 per setting, preview only
python3 rejudge_step0.py --step0 step0             # full run, resumable
```

- **Smoke test** writes only to `step0/rejudge_smoke/` (judgments, preview
  traces, `report.md`). Nothing else in the folder changes.
- **Full run** first copies every file it rewrites to
  `step0/rejudge/original/`, then rewrites `<setting>/predictions.jsonl`,
  `<setting>/summary.json`, `<setting>/traces/*.md`, `state.json` (stats),
  `summary.json`, `summary.md` and `per_question.csv`. Each record gets a
  `judge` block that keeps its original status and prediction. Corrected traces
  show `CORRECT`, `original_status`, and an "Answer judge" section. Every
  judged trace shows the judge's reading. `rejudge/report.md` has before/after
  tables and lists every accepted correction.
- **Pause and resume:** judgments are appended to
  `rejudge/judgments.jsonl`, so rerunning the same command continues.
  `--retry-errors` re-asks failed ones, and `--force` re-judges everything.
  Applying always starts from `rejudge/original/`, so reruns give the same
  result. To undo, copy `rejudge/original/` back.
- **Safety checks before writing:** the summaries recomputed from the original
  records must equal the saved `summary.json`, and every original trace must
  re-render byte for byte.
- **Tested locally** with a fake Ollama server on a copy of `results/step0`.
  Smoke mode changed nothing outside `rejudge_smoke/`. The full run plus a
  cached rerun gave identical output, and the backup equals the original. The
  evidence, error and wrong-letter rejections were tested. No real gpt-oss
  run yet.

## Step 1: Select the question set

Runs locally, no GPU.

```bash
.venv/bin/python src/analysis/select_questions.py \
  --traces results/monet-bf16-first30/traces \
  --out results/question_set.csv \
  --seed 0
```

The plan's default `--traces results/traces` was replaced with
`results/monet-bf16-first30/traces`, where the source traces actually are.

**Input:** `results/monet-bf16-first30/traces/vstar_*.md` — 30 traces from the
original Monet-7B BF16 run (first 30 V* questions, `LATENT_SIZE=10`, one
attempt each). Each trace has JSON blocks fenced with 4+ backticks: a metadata
block (`id`, `index`, `category`, …) and one attempt block per attempt
(`status`, `finish_reason`, token counts, `latent_segments`, …).

**Output:** `results/question_set.csv`, columns:

```text
question_index,problem_id,group,category,status,finish_reason,prompt_tokens,completion_tokens,latent_token_count,latent_segments
```

**Logic:**

- `group = latent_start` if `latent_segments > 0`, else `no_latent_start`.
- Traces with `finish_reason == "length"` are excluded.
- If a trace has several attempts, the first one is used (with a warning). All
  current traces have exactly one.
- 5 per group with `random.Random(seed)`; categories are shuffled and sampled
  round-robin so they are as balanced as possible. Groups are processed in a
  fixed order, so the same seed always gives the same CSV.
- If either group has fewer than 5 eligible questions, the script stops with
  an error and writes nothing.
- Rows are written grouped (`latent_start` first), sorted by `question_index`.

**Result (seed 0):**

| | Eligible | Selected `question_index` |
|---|---|---|
| `latent_start` | 18 | 1, 3, 20, 22, 27 |
| `no_latent_start` | 12 | 7, 10, 12, 17, 28 |

0 traces were excluded for `finish_reason == "length"`.

**Checks:** 10 rows, 5 per group ✔; eligible counts printed ✔; a second run
with seed 0 gives a byte-identical CSV ✔.

## Step 2: Experiment A

Generates one trajectory per question in each of `real`, `noise` and `no_image`
(30 in total) with Monet's patched vLLM runner, and logs the raw start-token
probability at every generation step.

### Files

| File | Role |
|---|---|
| `src/modal/monet_vstar_modal.py` | Adds the `exp_a` GPU function and the `run_exp_a` entrypoint. The original `main` entrypoint is unchanged. |
| `src/modal/run_exp_a.py` | Runs in the container: builds the 3 settings, generates, joins the step log, writes the CSVs. |
| `src/modal/patches/monet_step_log.py` | Logging patch for Monet's `execute_model`. Active only when `MONET_STEP_LOG` is set. |
| `src/analysis/results.py` | Local analysis (`exp-a` subcommand). |

### Run

Run from the project root. Do the smoke test first.

Step 0 must have run first: Experiment A reads its noise images from
`/vol/results/<step0-dir>/noise_images/` (default `step0`).

```bash
modal volume put monet-vstar results/question_set.csv /inputs/question_set.csv

# Smoke test: first question only (3 trajectories) -> /vol/results/exp_a_smoke
modal run src/modal/monet_vstar_modal.py::run_exp_a --limit 1
modal volume get monet-vstar results/exp_a_smoke ./results/exp_a_smoke
python src/analysis/results.py exp-a --input results/exp_a_smoke --out results/exp_a_smoke/analysis

# Full run: 10 questions, 30 trajectories -> /vol/results/exp_a
modal run --detach src/modal/monet_vstar_modal.py::run_exp_a
modal volume get monet-vstar results/exp_a ./results/exp_a
```

`--force` reruns into a folder whose `state.json` says `done`.
`--step0-dir step0_first30` reads noise images from that Step 0 folder instead of `step0`.

**Entrypoint change:** the Modal file now has several local entrypoints, so the
original V* evaluation must be called as
`modal run src/modal/monet_vstar_modal.py::main ...`. A bare
`modal run monet_vstar_modal.py` no longer works. Local file paths in the
Modal image are now resolved from the script's own folder, so commands work
from the project root and from `src/modal`.

### Key settings

| Setting | Value |
|---|---|
| GPU | L40S |
| Engine | vLLM 0.10.0 + Monet runner patch, Monet commit `0893999` |
| System prompt | `SYSTEM_PROMPT` from `run_vstar.py` (the "expert multimodal … latent visual embeddings" prompt), same in every setting |
| `LATENT_SIZE` | 10 |
| Sampling | Monet defaults (temperature 0.1, top-k 50, top-p 0.8, repetition penalty 1.01, 4096 max tokens), cloned per request with `seed = question_index*10 + {real:0, noise:1, no_image:2}` |
| Engine settings | Monet's `vllm_mllm_init` with `max_num_seqs = 30`, `max_model_len = 16384`, `gpu_memory_utilization = 0.85` (as in `run_vstar.py`) |
| Noise image | The Step 0 PNG `q###_noise.png`, copied byte for byte into `exp_a/noise_images/` and fed from the copy. It is not regenerated. The run fails if the file is missing or its size differs from the original image. |
| Image preprocessing | Same path as `run_vstar.py` (`process_vision_info`, then Monet's `min_pixels = 256*28*28`, `max_pixels = 8192*28*28`) |

### Logging patch

`patches/monet_step_log.py` reads Monet's runner source, inserts four blocks
at exact anchor lines, and installs the result under the same module names
Monet uses. If any anchor is missing or repeated, it raises.

1. **Preemption:** after `_update_states`, if any request has
   `resumed_from_preemption`, it writes a `{"event": "preemption"}` line and
   raises.
2. **Raw logits:** before the sampler, it computes the start-token logit,
   log-softmax and best other logit from the raw logits. This is placed before
   the sampler, not after it, because vLLM's sampler can apply temperature and
   penalties in place to float32 logits. The logged values are therefore always
   from the raw logits.
3. **One line per step:** after `valid_sampled_token_ids` is ready and before
   the ABS-VIS update, it writes one JSON line per request that is not a
   partial prefill. `step = len(req_state.output_token_ids)`, and `in_latent`
   is the ABS-VIS state before this step's update. Each line also has
   `sampled_token_id`, the token drawn before Monet may force the end token.
4. **Hidden states:** after the ABS-VIS update, the final hidden state behind
   each logged step (`sample_hidden_states[i]`, final layer after the final
   norm) is appended as bfloat16 to `raw/hidden_states.bin`. A line is also
   written to `raw/hidden_index.jsonl` with `req_id`, `step`, `row`, `dim` and
   `fed_as_latent`. `fed_as_latent` is true when Monet will feed this exact
   vector back as the next input embedding; these vectors are the latent
   thoughts. `MONET_CAPTURE_HIDDEN=0` turns capture off.

#### What the latent vectors are

In latent mode, Monet feeds the previous step's final hidden state back
instead of a token embedding. For one segment with `LATENT_SIZE=10`:

| Step | Token in output | Hidden state fed back next? |
|---|---|---|
| s | `<abs_vis_token>` (sampled) | yes, latent 0 |
| s+1 … s+9 | placeholder ids (never used as input) | yes, latents 1–9 |
| s+10 | forced `</abs_vis_token>` | no |

So each segment has 10 latent vectors, from steps s to s+9. The trace text
holds only the meaningless placeholders, which is why the vectors themselves
are saved.

### Checks in `run_exp_a.py`

- The Monet patch and the step-log patch are both active.
- **KV cache:** after the engine starts, KV tokens
  (`num_gpu_blocks × block_size`, the value in the `GPU KV cache size` log line)
  must be at least `30 × (max Step 1 prompt_tokens + 4096)`. If not, the run
  stops before generating. The check is repeated after generation with the
  actual prompt sizes.
- **Preemption:** any preemption event fails the run.
- **Hidden states:** every generated token must have exactly one hidden row,
  and `fed_as_latent` must agree with the token sequence (start token, or
  `in_latent` and not the end token). Problems are listed in
  `checks.json` under `hidden_state_problems`. Once all checks pass,
  `raw/hidden_states.bin` is deleted, because `hidden/` holds the same data.
- **Step coverage:** every generated token must have exactly one logged step,
  with no extra steps.
- Results go to `checks.json`. `state.json` records `running`, `done` or
  `failed_*`.

**Resume:** all 30 trajectories run as one concurrent batch, so the unit of
resumption is the whole run. A rerun into a `done` folder is skipped; any
other state restarts the run and clears the old step log.

### Outputs (`results/exp_a/`)

| File | Content |
|---|---|
| `exp_a_trajectories.csv` | One row per trajectory, with the plan's columns plus `answer, prediction, correct, status` |
| `exp_a_steps.csv` | One row per trajectory and step, with the plan's columns |
| `traces/<trajectory_id>.md` | Prompt and full raw output, for manual review |
| `real_images/q###_real.<ext>` | Original V* image bytes, unchanged |
| `noise_images/q###_noise.png` | Byte-for-byte copies of the Step 0 noise images used |
| `raw/step_log.jsonl` | Raw per-step log from the patched runner |
| `hidden/<trajectory_id>.npy` | float32 `[completion_tokens, 3584]`; row t = hidden state behind token t (exact upcast of bfloat16) |
| `latent_vectors.npy` | float32 `[n_latent, 3584]`: only the vectors fed back as latents, from all trajectories |
| `latent_vectors.csv` | One row per latent vector: `row,trajectory_id,question_index,problem_id,group,category,setting,segment,position_in_segment,step` |
| `raw/hidden_index.jsonl` | Raw index of the capture (`req_id, step, row, dim, fed_as_latent`) |
| `checks.json`, `state.json` | Run checks and status |

`image_path` is relative to the `exp_a` folder.

### Analysis

```bash
python src/analysis/results.py exp-a --input results/exp_a --out results/exp_a/analysis --metric prob
python src/analysis/results.py exp-a --input results/exp_a --out results/exp_a/analysis --metric log_prob
```

Options: `--metric {prob,log_prob}`, `--group {all,latent_start,no_latent_start}`,
`--stop-at-first-start` (default) / `--keep-after-start`, `--bin-size 1`,
`--bootstrap 2000`, `--seed 0`.

- `in_latent=True` steps are always removed.
- With `--stop-at-first-start`, a line ends at the first start token, and that
  step is included.
- Without `--group`, all three subsets (`all`, `latent_start`,
  `no_latent_start`) are produced. With `--group`, only that subset is.
- **Bins:** a question's value in a bin is the mean of its non-latent steps in
  that bin. The combined mean and median are taken across questions.
- **CI:** a percentile bootstrap across questions, with `--bootstrap`
  resamples. No CI is drawn where fewer than 2 questions contribute.

Each run writes to `<out>/<metric>_<stop|keep>_bin<N>/`:

| File | Content |
|---|---|
| `per_question/q###.png` | 3 settings, dot at the first start |
| `combined_<subset>.png` | Left: mean, CI and median. Right: sample count. Shared x-axis. |
| `combined_curves.csv` | Plotted values (n, mean, ci_low, ci_high, median per bin) for replotting |
| `latent_start_frequency.csv` / `.md` | Started, total, percent and median first-start step, by setting and subset |
| `settings.json` | Options used |

### Local tests (no GPU)

- The patch applies to the pinned runner source, and the result compiles.
  A missing anchor and a second patch are both refused.
- The step-log join was tested with stubbed vLLM outputs: start detection,
  9 placeholder tokens per latent segment, `in_latent` flags, answer scoring,
  and reporting of a missing step.
- `results.py exp-a` was run on synthetic CSVs for both metrics, the group
  filter, `--keep-after-start` and `--bin-size 5`. The output is deterministic
  for a fixed seed.
- **Not yet run:** the Modal smoke test and full run.

### Output columns

`exp_a_trajectories.csv`, one row per trajectory (the plan's columns, then 4 extra):

```text
trajectory_id,question_index,problem_id,group,category,setting,seed,image_path,image_width,image_height,prompt_text,prompt_token_ids,completion_token_ids,raw_response,clean_response,finish_reason,prompt_tokens,completion_tokens,latent_token_count,latent_segments,started,first_start_step,answer,prediction,correct,status
```

`exp_a_steps.csv`, one row per trajectory and generation step:

```text
trajectory_id,question_index,problem_id,group,category,setting,seed,step,token_id,token_text,start_prob,start_log_prob,start_logit,best_other_logit,start_logit_margin,is_top1,in_latent,is_first_start,fed_as_latent
```

- `trajectory_id` is `q<index>_<setting>`, e.g. `q001_noise`.
- Token-ID lists are JSON strings.
- `group` is the Step 1 label, even if this run behaves differently.
- `first_start_step` is empty when `started=False`.
- `token_id` is the final token in the output, after Monet forces the end
  token. The token the sampler originally drew is `sampled_token_id` in
  `raw/step_log.jsonl`.

`raw/step_log.jsonl`, one JSON line per request and step:

```json
{"req_id": "0", "step": 0, "start_prob": 0.0, "start_log_prob": 0.0, "start_logit": 0.0,
 "best_other_logit": 0.0, "start_logit_margin": 0.0, "is_top1": false, "in_latent": false,
 "sampled_token_id": 0}
```

A preemption writes `{"event": "preemption", "req_ids": [...]}` and fails the run.

### What `--bin-size` and `--bootstrap` mean

- **`--bin-size N`:** groups every N consecutive generation steps into one
  point on the x-axis. With `1` (the default), every step is its own point.
  Larger bins give smoother curves and more samples per point, at the cost of
  hiding short spikes. Within a bin, each question's steps are averaged first,
  then combined across questions, so long trajectories do not count more.
- **`--bootstrap 2000`:** the number of resamples for the 95% CI.
  1. At each position, take one value per question.
  2. Resample that many questions with replacement and take the mean.
  3. Repeat 2,000 times.
  4. The 2.5th and 97.5th percentiles of those means are the CI bounds.

  Questions are resampled, not steps, because steps within one trajectory are
  strongly correlated. `--seed` fixes the draws. With 10 questions (5 per
  group), the CIs are wide and only a rough guide.

### Smoke-test checklist

After `modal run src/modal/monet_vstar_modal.py::run_exp_a --limit 1`, check:

- [ ] The log shows `[Monet] vLLM runner patched` and `[monet-step-log] runner patched`.
- [ ] The log shows `GPU KV cache: N tokens; need >= ...`, and N is at least the need.
- [ ] `results/exp_a_smoke/checks.json` has `trajectories: 3`,
      `missing_or_extra_steps: []`, `preemption_events: 0`, and
      `kv_ok_actual_prompts: true`.
- [ ] `state.json` has `status: "done"`.
- [ ] `checks.json` has `hidden_state_problems: []`, and `hidden_states.latent_vectors` is 10 × the number of latent segments.
- [ ] `hidden/q001_real.npy` has shape `[completion_tokens, 3584]`.
- [ ] `exp_a_steps.csv` has one row per generated token for each trajectory.
- [ ] `noise_images/q001_noise.png` is byte-identical to Step 0's `noise_images/q001_noise.png` (`cmp`).
- [ ] `results.py exp-a` runs on `results/exp_a_smoke`.

Only then start the full run.

### Step 2 report

**Files changed**

| File | Change |
|---|---|
| `src/modal/patches/__init__.py` | New, empty package file |
| `src/modal/patches/monet_step_log.py` | New: step-logging patch for Monet's runner |
| `src/modal/run_exp_a.py` | New: Experiment A runner, run inside the container |
| `src/modal/monet_vstar_modal.py` | Adds `exp_a` (L40S) and the `run_exp_a` entrypoint. Mounts `run_exp_a.py` and `patches/`. Local paths now resolve from the script's folder. Docstring now uses `::main` / `::run_exp_a`. |
| `src/analysis/results.py` | New: `exp-a` analysis |
| `interpretibility/README.md` | This file |

`src/modal/run_vstar.py` is not modified; `run_exp_a.py` imports its prompt
and helper functions.

**Commands run (locally only)**

- Cloned Monet at commit `0893999` and fetched vLLM 0.10.0 source files into
  the session scratchpad, to inspect `execute_model`, the sampler, the
  scheduler output and the KV-cache handshake. Nothing was downloaded into
  the project.
- Applied the patch to the pinned runner source and compiled it.
- Ran `py_compile` on all new scripts.
- Imported the Modal app from another directory to confirm path resolution
  and the entrypoints (`main`, `run_exp_a`).
- Ran a stubbed test of `build_rows` / `read_step_log` from `run_exp_a.py`.
- Ran `results.py exp-a` on synthetic CSVs with these variants:
  `--metric prob`, `--metric log_prob`, and
  `--group latent_start --keep-after-start --bin-size 5`. Ran it twice to
  check that it is deterministic.

No Modal command was run.

**Outputs created:** none in `results/`. The synthetic test outputs were
written to the session scratchpad only.

**Checks**

| Check | Result |
|---|---|
| Patch anchors found exactly once in the pinned runner | ✔ |
| Patched runner compiles | ✔ |
| Patch refuses a missing anchor and a second patch | ✔ |
| Step-log join: start step, latent flags, 9 placeholders, scoring | ✔ |
| Missing logged step is reported | ✔ |
| `results.py exp-a`: both metrics, group filter, keep/stop, bin size | ✔ |
| `results.py exp-a` deterministic for a fixed seed | ✔ |
| Modal app loads with both entrypoints | ✔ |
| Modal smoke test | Not run (to be run by the user) |
| 30 trajectories, full step coverage, no preemption | Pending the full run |

**Decisions and deviations from the plan**

1. **Logits are read before the sampler.** The plan puts the logging after
   `valid_sampled_token_ids`. The statistics are now computed just before the
   sampler and written at the planned point. The reason: vLLM 0.10.0's sampler
   can apply temperature and penalties in place to float32 logits, so reading
   them later could give processed values instead of raw ones.
2. **`sampled_token_id` was added to each log line**, to show when Monet
   overrides a sampled token with the end token.
3. **The KV-cache check uses Step 1 `prompt_tokens`** (the real-image prompt
   sizes) before generation, because actual prompt sizes are only known
   afterwards. It is repeated after generation with the actual sizes.
4. **The smoke test writes to `exp_a_smoke`**, so it never blocks or mixes
   with the full run in `exp_a`.
5. **Extra trajectory columns** (`answer`, `prediction`, `correct`, `status`)
   and markdown traces were added, to meet the project rule about saving
   answers and reasoning traces.
6. **Resume works on the whole run**, not per trajectory, because the 30
   trajectories run as one concurrent batch. Splitting it up would change
   batching and the KV-cache check.
7. **`modal run monet_vstar_modal.py` without `::main` no longer works**,
   because the file now has two local entrypoints.

### Step 1 report

- **Files changed:** `src/analysis/select_questions.py` (new),
  `interpretibility/README.md` (new).
- **Command run:**
  `.venv/bin/python src/analysis/select_questions.py --traces results/monet-bf16-first30/traces --out results/question_set.csv --seed 0`,
  run twice and compared with `cmp`.
- **Output:** `results/question_set.csv` (10 rows).
- **Checks:** 10 rows with 5 per group ✔; eligible counts printed (18 / 12) ✔;
  identical CSV on rerun ✔.
- **Deviation:** the traces path is `results/monet-bf16-first30/traces`,
  not `results/traces`.

### Change: Experiment A uses Step 0 noise images

- `run_exp_a.py` no longer generates noise. `make_noise_image` was replaced by
  `copy_step0_noise`, which loads `<noise-dir>/q###_noise.png`, checks its
  size, and copies it into `exp_a/noise_images/`.
- New required argument `--noise-dir`. The Modal `exp_a` function passes
  `/vol/results/<step0_dir>/noise_images` and fails early if it is missing.
- `run_exp_a` has a new `--step0-dir` option (default `step0`).
- `state.json` records `noise_dir`.
- The pixels are the same as before, because the seed rule is unchanged.
  Only the source of the image changed.

### Change: latent vectors captured in Step 2, token IDs saved in Step 0

- **Step 2:** the logging patch has a fourth insertion that saves hidden
  states. `run_exp_a.py` converts them to `hidden/*.npy`,
  `latent_vectors.npy` and `latent_vectors.csv`, adds `fed_as_latent` to
  `exp_a_steps.csv`, and checks coverage and consistency.
- **Step 0:** `predictions.jsonl` now has `completion_token_ids`,
  `latent_start_steps` and `latent_end_steps`. It still uses no patch.
- **Local tests:**
  - The patch (now 4 insertions) applies to the pinned runner and compiles.
  - A simulated capture, written in the same bfloat16 file format and
    following Monet's state machine, round-trips exactly. It gives 10 latent
    vectors at steps s…s+9, positions 0–9, and `fed_as_latent` passes the
    consistency check.
  - The writer itself (`_monet_hidden_write`) uses torch, which is not
    installed locally. The smoke test will be its first real run.

#### Loading the latents

```python
import numpy as np, csv
vecs = np.load("results/exp_a/latent_vectors.npy")            # [n_latent, 3584]
meta = list(csv.DictReader(open("results/exp_a/latent_vectors.csv")))
h = np.load("results/exp_a/hidden/q001_real.npy")             # [completion_tokens, 3584]
unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
cos = unit @ unit.T                                           # cosine similarity
```

#### Expected disk use

At 14 KiB per generated token (3,584 × float32):

| Item | Typical | Worst case |
|---|---|---|
| Step 2 `hidden/*.npy` | ~30 MB (first-30 traces average 69 tokens, max 91) | ~1.76 GB (all 30 hit 4,096 tokens) |
| Step 2 `raw/hidden_states.bin`, during the run only | ~15 MB | ~0.9 GB |
| Step 2 `latent_vectors.npy` | < 1 MB (140 KiB per segment) | a few MB |
| Step 2 noise PNG copies (10) | ~100 MB | — |
| Step 0 noise PNGs | ~2 GB for 191 (~300 MB for 30) | — |
| Step 0 token IDs | < 1 MB | ~16 MB |

## Resolved issues

1. **System prompt:** keep `SYSTEM_PROMPT` from `run_vstar.py`, the same
   prompt used for the Step 1 traces. Step 1 does not need to be rerun.
2. **9 latent tokens per segment with `LATENT_SIZE=10`:** this is how Monet's
   runner works, not a bug. After the start token, the runner feeds 10 hidden
   states. Steps 1–9 emit placeholder ids, and step 10's sampled token is
   replaced by the end token. `token_stats` counts only the ids strictly
   between start and end, which gives 9. In the step log, those 10 steps have
   `in_latent=True`.

## Open issues

- **Category balance:** all 30 Step 1 traces are `direct_attributes`, so
  category balancing has no effect.

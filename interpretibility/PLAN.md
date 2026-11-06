# Monet-7B Latent-Start Study on V*

## Goal

Run a preliminary study of whether real images increase Monet-7B's probability of generating the latent-start token, `<abs_vis_token>` (ID `151666`).

This plan covers Steps 0–4. Step 5 will be added later.

## Rules for Claude Code

- Work on one step at a time. Do not begin the next step unless asked.
- Do not start a Modal GPU run without explicit approval.
- Run the smoke test before a full Modal run.
- Keep inference on Modal and analysis on the local machine.
- Do not change results or thresholds to make a check pass. Stop and report the problem.
- After every completed step or code change, update `interpretability/README.md`. Keep this separate from the main project `README.md`, and do not edit the main project README for this study unless asked.
- `interpretability/README.md` must include the current status, how to run each completed step, input and output files, key settings, and any changes or issues.
- At the end of each step, report the files changed, commands run, outputs created, and checks passed or failed.

## Shared decisions

| Item | Value |
|---|---|
| **Modal GPU** | `L40S` for all inference |
| **Inference engine** | Step 0 uses vLLM with Monet's patch only, exactly as Monet ships it. Step 2 uses vLLM with Monet's patch plus the step-logging patch. Step 3 and later use Transformers |
| **Model** | `NOVAglow646/Monet-7B`, original BF16 weights |
| **Monet commit** | `08939998d3d643a73a316e349faa34f420429153` |
| **Dataset** | V* |
| **Latent size** | `LATENT_SIZE=10` |
| **Sampling** | Monet defaults: temperature 0.1, top-k 50, top-p 0.8, repetition penalty 1.01, 4,096 max tokens |
| **Settings** | `real`, `noise`, `no_image` |
| **System prompt** | `SYSTEM_PROMPT` in `run_vstar.py` ("You are an expert multimodal large language model designed to reason with latent visual embeddings. …"), the same in every setting and step |
| **Noise images** | Generated once in Step 0 (`seed = 1000 + question_index`) and reused unchanged by every later step |
| **Main probability** | Raw softmax probability before temperature, top-k, top-p, and repetition penalty |

## Open decision before Step 2 (resolved)

Resolved: keep the current `SYSTEM_PROMPT` from `run_vstar.py`. It matches the Step 1 traces, so Step 1 is final.

`SYSTEM_PROMPT` in `run_vstar.py` may not match the Monet README prompt:

`You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}.`

Before writing Step 2:

1. Compare the system prompt in `results/traces/` with the current code.
2. If they differ from the README, stop and ask which prompt to use.
3. Use one prompt in every setting.
4. If the prompt changes, the selection traces must be regenerated before Step 1 is treated as final.

## Repository layout

Use the existing `src/modal/monet_vstar_modal.py` file. Add these files:

```text
interpretability/
└── README.md                       # study notes, separate from the project README

src/
├── modal/
│   ├── monet_vstar_modal.py        # existing, add Step 0 and Experiment A and B functions
│   ├── run_step0.py                # new, Step 0: V* in three settings, Monet's patch only
│   ├── run_exp_a.py                # new, vLLM generation
│   ├── run_exp_b.py                # new, Transformers replay
│   └── patches/                    # new, runner logging patch
└── analysis/
    ├── select_questions.py         # Step 1
    ├── results.py                  # Steps 2 and 3
    └── select_positions.py         # Step 4
```

Before editing Modal paths, check how `add_local_file("run_vstar.py", ...)` currently resolves from the run directory.

## Step 0: V* accuracy in three settings

**Runs on:** Modal L40S with vLLM 0.10.0 and Monet's runner patch only

**Question:** how does Monet do on V* with the real image, a noise image, and no image? How often does it start latent reasoning in each setting, and how accurate is it when it does?

### Rules

- Use Monet's own vLLM patch (`inference.apply_vllm_monet`) exactly as shipped. Do **not** import or enable the step-logging patch. Fail if `MONET_STEP_LOG` is set.
- Use Monet's engine and sampling defaults through `vllm_mllm_init`, exactly as `run_vstar.py` does. That means `max_num_seqs=512`, engine seed 0, `max_model_len=16384`, `gpu_memory_utilization=0.85`, and no per-request seeds.
- Use the same system prompt, user prompt and answer extraction as `run_vstar.py`.
- Run each setting as its own `generate` call over all questions, in the order `real`, `noise`, `no_image`. This way the `real` batch matches the original V* run.

### Settings

- **`real`:** the original V* image.
- **`noise`:** seeded RGB noise with the same dimensions as the original image. Use `seed = 1000 + question_index` and `np.random.default_rng(seed).integers(0, 256, (H, W, 3), uint8)`. Save each image as `/vol/results/step0/noise_images/q###_noise.png` before generation, and feed the saved PNG. **These are the noise images for every later step.**
- **`no_image`:** the same system prompt and question, with no image and no `multi_modal_data`.

### Modal commands

```bash
modal run src/modal/monet_vstar_modal.py::run_step0 --limit 2
modal run --detach src/modal/monet_vstar_modal.py::run_step0
modal volume get monet-vstar results/step0 ./results/step0
```

- `--limit N` runs the first N V* questions and writes to `results/step0_firstN`.
- The full run covers all 191 questions and writes to `results/step0`.
- `--settings real,noise` runs a subset of settings.

### Pause and resume

`state.json` records each setting as `running` or `done`. A rerun into the same folder skips settings that are done and reruns any unfinished one from the start. `--force` reruns finished settings. The noise PNGs are created only if missing, so a resumed run feeds the same pixels.

Each setting is one batch, so a resumed setting shares the engine's random stream with fewer earlier batches than in an uninterrupted run. Results are still valid, but they are not bit-identical to an uninterrupted run.

### Outputs

```text
results/step0/
├── noise_images/q###_noise.png      # used by Steps 2 and 3
├── real/ noise/ no_image/
│   ├── predictions.jsonl            # run_vstar.py record format, plus `setting`, `completion_token_ids`, `latent_start_steps`, `latent_end_steps`
│   ├── summary.json                 # run_vstar.py summary format, plus `setting`
│   └── traces/vstar_###.md          # run_vstar.py trace format, with the setting and image named under the title
├── summary.json                     # comparison stats per setting
├── summary.md                       # the same, as a table
├── per_question.csv                 # one row per question, predictions side by side
└── state.json
```

`summary.md` reports these for each setting:

| Row | Meaning |
|---|---|
| Questions / Correct / Accuracy % | Overall accuracy |
| No readable answer / Hit token limit | Failure counts |
| Latent reasoning started / Started % | Responses with at least one latent token |
| Correct when started / Accuracy when started % | Accuracy among responses that started latent reasoning |
| Not started / Correct when not started / Accuracy when not started % | Accuracy among the rest |
| vLLM preemptions | From vLLM's own metrics (`vllm:num_preemptions`). Reported, not enforced, because Step 0 must not patch the runner |

`predictions.jsonl` also stores the generated token IDs (`completion_token_ids`) and the positions of every start and end token. These come from vLLM's normal output, so no patch is needed. Step 0 does **not** capture hidden states: that would need a runner patch, and Step 0 must use Monet's patch exactly as shipped. Latent vectors are captured in Step 2.

`per_question.csv` columns:

```text
index,problem_id,category,answer,{setting}_prediction,{setting}_correct,{setting}_latent_segments,{setting}_latent_token_count,{setting}_finish_reason
```

### Expected disk use

- **Noise PNGs:** random pixels do not compress, so each is about W × H × 3 bytes. That is roughly 10 MB at V*'s typical ~3.5-megapixel resolution, or about 2 GB for all 191 questions (~300 MB with `--limit 30`).
- **Traces, predictions and token IDs:** a few MB per setting.

### Done when

- All three settings are `done` in `state.json`.
- Each setting has one trace per question.
- There is one noise PNG per question, the same size as its original image.
- `summary.md` shows every row above for every setting.

Step 1 still selects its questions from the original real-image traces in `results/monet-bf16-first30/traces`. Step 0 does not change Step 1.

## Step 1: Select the question set

**Runs on:** local machine

**Command:**

```bash
python src/analysis/select_questions.py \
  --traces results/traces \
  --out results/question_set.csv \
  --seed 0
```

### Task

Parse every `vstar_*.md` trace. Each trace has JSON blocks fenced with 4 or more backticks.

Read:
- `id`, `index`, and `category` from the metadata block.
- `status`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `latent_token_count`, and `latent_segments` from the attempt block.

Assign:
- `group=latent_start` when `latent_segments > 0`.
- `group=no_latent_start` when `latent_segments == 0`.

Exclude traces with `finish_reason == "length"`.

Select 5 questions from each group using the fixed seed. Balance V* categories as far as possible. If either group has fewer than 5 eligible questions, stop and report the counts.

### Output

`results/question_set.csv` with one row per selected question:

```text
question_index,problem_id,group,category,status,finish_reason,prompt_tokens,completion_tokens,latent_token_count,latent_segments
```

### Done when

- The CSV has 10 rows, with 5 in each group.
- The script prints the eligible count for each group.
- Rerunning with the same seed gives the same CSV.

## Step 2: Experiment A

**Runs on:** Modal L40S with vLLM 0.10.0 and Monet's runner patch

**Question:** does Monet naturally assign more probability to the latent-start token under a real image?

### Prerequisite

Step 0 must have produced noise images for every selected question.

### Upload the question set

```bash
modal volume put monet-vstar results/question_set.csv /inputs/question_set.csv
```

### Inference

For each of the 10 questions, generate one trajectory in each setting, for 30 trajectories.

- **`real`:** use the original V* image.
- **`noise`:** use the Step 0 noise image for the question, `/vol/results/step0/noise_images/q###_noise.png`. Pass `--step0-dir` to use another Step 0 folder. Do not regenerate it. Copy it byte for byte to `/vol/results/exp_a/noise_images/` so the Experiment A folder is self-contained. Fail if the file is missing or its size differs from the original image.
- **`no_image`:** use the same system prompt and question with no image and no `multi_modal_data`.

Use a separate sampling seed for each trajectory:

```text
seed = question_index * 10 + setting_code
real=0, noise=1, no_image=2
```

Clone Monet's `SamplingParams`, set `seed`, and pass one object per request.

Set `max_num_seqs=30`. Check the `GPU KV cache size` log line and confirm it is at least `30 × (largest prompt_tokens + 4096)`. If not, stop and report it.

### Logging patch

Patch Monet's `execute_model` only when `MONET_STEP_LOG` is set. Fail loudly if the expected source context is not found.

Add logging after `valid_sampled_token_ids` is ready and before the `ABS-VIS` state update. For each request that is not a partial prefill, write one JSON line:

```json
{
  "req_id": "...",
  "step": 0,
  "start_prob": 0.0,
  "start_log_prob": 0.0,
  "start_logit": 0.0,
  "best_other_logit": 0.0,
  "start_logit_margin": 0.0,
  "is_top1": false,
  "in_latent": false
}
```

Use `len(req_state.output_token_ids)` as `step`. Calculate `start_log_prob` with `logits.log_softmax(-1)`. Calculate `best_other_logit` without the start token.

Also check `scheduler_output.scheduled_cached_reqs.resumed_from_preemption`. If any request was preempted, write an event and make the run fail, because preemption can break latent reasoning.

### Hidden-state (latent vector) capture

The latent thoughts are not tokens. In latent mode, Monet feeds the previous step's final hidden state back as the next input embedding (`st["pending"] = last_token_h[i]`). The placeholder tokens between `<abs_vis_token>` and `</abs_vis_token>` are never used as input. To make later analysis possible (similarity, PCA, probes), save those vectors.

In the same logging patch, after the ABS-VIS state update and before the sampled tokens are cached:

- For every logged step, save `sample_hidden_states[i]`. This is the final-layer state after the final norm, at the position that produced token `step`. It is exactly the vector Monet stores as `pending`.
- Append it as raw bfloat16 to `raw/hidden_states.bin`, and write an index line to `raw/hidden_index.jsonl`: `req_id`, `step`, `row`, `dim`, `fed_as_latent`.
- `fed_as_latent` is true when, after the update, the request is in latent mode with this vector pending, which means Monet feeds it back. For a segment with `LATENT_SIZE=10`, that is the start-token step plus the 9 placeholder steps, so 10 vectors. The step whose token is forced to the end token is not fed.
- Capture is on by default. `MONET_CAPTURE_HIDDEN=0` turns it off.

After generation, `run_exp_a.py` converts the raw capture:

- It writes `hidden/<trajectory_id>.npy` as float32 (an exact upcast from bfloat16) with shape `[completion_tokens, 3584]`. Row `t` is the hidden state behind token `t`.
- It writes `latent_vectors.npy` with shape `[n_latent, 3584]`: only the fed-back vectors, from all trajectories.
- It writes `latent_vectors.csv`, one row per latent vector:

  ```text
  row,trajectory_id,question_index,problem_id,group,category,setting,segment,position_in_segment,step
  ```
- It adds a `fed_as_latent` column to `exp_a_steps.csv`.
- It checks that every generated token has exactly one hidden row. It also checks that `fed_as_latent` agrees with the token sequence: the start token, or `in_latent` and not the end token.
- After all checks pass, it deletes `raw/hidden_states.bin`, because `hidden/` holds the same data.

**Expected disk use:** 3,584 floats × 4 bytes = 14 KiB per generated token.

| Item | Typical | Worst case |
|---|---|---|
| `hidden/*.npy` (30 trajectories) | ~30 MB (first-30 traces average 69 tokens, max 91) | ~1.76 GB (all 30 hit 4,096 tokens) |
| `raw/hidden_states.bin` while running | ~15 MB (bfloat16) | ~0.9 GB |
| `latent_vectors.npy` | ~0.1–0.5 MB (140 KiB per segment) | a few MB |
| Noise PNGs copied from Step 0 (10) | ~10 MB per image at V* resolution (random pixels do not compress) | — |

### Modal commands

Add a named entrypoint to `monet_vstar_modal.py`. Keep the existing evaluation entrypoint working.

```bash
modal run src/modal/monet_vstar_modal.py::run_exp_a --limit 1
modal run --detach src/modal/monet_vstar_modal.py::run_exp_a
modal volume get monet-vstar results/exp_a ./results/exp_a
```

### Inference outputs

`results/exp_a/exp_a_steps.csv`, with one row per question, setting, and step:

```text
trajectory_id,question_index,problem_id,group,category,setting,seed,step,token_id,token_text,start_prob,start_log_prob,start_logit,best_other_logit,start_logit_margin,is_top1,in_latent,is_first_start,fed_as_latent
```

Hidden states: `results/exp_a/hidden/<trajectory_id>.npy`, `results/exp_a/latent_vectors.npy`, and `results/exp_a/latent_vectors.csv` (see "Hidden-state capture").

`results/exp_a/exp_a_trajectories.csv`, with one row per trajectory:

```text
trajectory_id,question_index,problem_id,group,category,setting,seed,image_path,image_width,image_height,prompt_text,prompt_token_ids,completion_token_ids,raw_response,clean_response,finish_reason,prompt_tokens,completion_tokens,latent_token_count,latent_segments,started,first_start_step
```

Store token ID lists as JSON strings. Keep the group label from Step 1, even if a rerun behaves differently.

### Analysis

**Command:**

```bash
python src/analysis/results.py exp-a \
  --input results/exp_a \
  --out results/exp_a/analysis \
  --metric prob
```

Required options:

```text
--metric {prob,log_prob}
--group {all,latent_start,no_latent_start}
--stop-at-first-start / --keep-after-start
--bin-size 1
--bootstrap 2000
--seed 0
```

Default to `--stop-at-first-start`. Always remove `in_latent=True` points from the probability lines.

Create:

1. **One graph per question.** Plot start-token probability against generation step, with 3 colored lines for the settings. End a line when that setting ends.
2. **One combined figure.** Use two side-by-side panels with a shared x-axis:
   - Left: mean, 95% CI, and median for each setting.
   - Right: sample count at each position for each setting.
3. **Latent-start frequency table.** Show started count, total count, percentage, and median first-start step by setting.
4. **Group-wise versions.** Repeat the combined figure and frequency table for all questions, `latent_start`, and `no_latent_start`.

Calculate each 95% CI with a percentile bootstrap across questions. Do not draw a CI at positions with fewer than 2 samples. Save the plotted values as CSV so that median curves can be replotted without changing the code.

### Done when

- There are 30 trajectories in `exp_a_trajectories.csv`.
- Every generated token that needs a probability has a matching logged step.
- No preemption event occurred.
- Every generated token has one hidden-state row, and `fed_as_latent` agrees with the token sequence.
- `latent_vectors.npy` has 10 rows per latent segment, with `LATENT_SIZE=10`.
- `results.py exp-a` runs locally from the CSV files.
- Both `prob` and `log_prob` graphs can be generated.

## Step 3: Experiment B

**Runs on:** Modal L40S with Transformers, not vLLM. Use stock `Qwen2_5_VLForConditionalGeneration`.

**Question:** when the generated text is fixed, does changing the image change the start-token probability?

Do not import `inference.apply_vllm_monet` in this script. Load Monet weights with BF16 and `attn_implementation="sdpa"`.

### Replay

Use the 30 trajectories from Step 2 as forced prefixes, and run each one under `real`, `noise`, and `no_image`, for 90 replays.

Build each prefix from `completion_token_ids`:

- If `started=True`, use tokens before `first_start_step`.
- If `started=False`, use all tokens except a final EOS token.
- Never include token IDs `151665`, `151666`, or `151667`.

The prefix must stop before the first latent token because stock Transformers does not run Monet's latent loop. For latent-vector analysis, use the vectors captured in Step 2 (`results/exp_a/hidden/`, `latent_vectors.npy`). Replay does not produce latents.

For each condition:

- Use that setting's stored `prompt_token_ids`.
- Use the same original image, or the Step 0 noise image (as copied into `results/exp_a/noise_images/`).
- Apply Monet's `min_pixels=256*28*28` and `max_pixels=8192*28*28`.
- Use `use_cache=False`.
- Read the logits for every position from the last prompt token through the last prefix token.

A prefix of length `L` gives `L + 1` positions. Empty prefixes give position 0 only.

### Built-in parity check

Compare replay and vLLM start log-probabilities where a trajectory is replayed under its original setting. Compare only positions before the first start token.

Report the median and maximum absolute difference. Stop if the median difference is above `0.05`.

### Modal commands

```bash
modal run src/modal/monet_vstar_modal.py::run_exp_b --limit 1
modal run --detach src/modal/monet_vstar_modal.py::run_exp_b
modal volume get monet-vstar results/exp_b ./results/exp_b
```

### Inference outputs

`results/exp_b/exp_b_replays.csv`, with one row per trajectory, condition, and position:

```text
trajectory_id,question_index,problem_id,group,category,source_setting,condition,position,prefix_length,next_token_id,next_token_text,ends_at_start,hit_token_limit,start_prob,start_log_prob,start_log_odds,start_logit,best_other_logit,start_logit_margin,is_top1
```

`results/exp_b/exp_b_prefixes.csv`, with one row per trajectory:

```text
trajectory_id,question_index,problem_id,group,category,source_setting,prefix_token_ids,prefix_length,ends_at_start,hit_token_limit
```

`results/exp_b/parity_check.csv`, with the vLLM value, replay value, and absolute difference for each compared position.

### Analysis

**Command:**

```bash
python src/analysis/results.py exp-b \
  --input results/exp_b \
  --out results/exp_b/analysis \
  --metric prob
```

Required options:

```text
--metric {prob,log_prob}
--group {all,latent_start,no_latent_start}
--source {all,real,noise,no_image}
--exclude-hit-limit
--bin-size 1
--bootstrap 2000
--seed 0
```

### Required graphs

1. **One difference graph per trajectory.** Plot `real - noise` and `real - no_image` against position. With `--metric log_prob`, use `log(real) - log(noise)` and `log(real) - log(no_image)`.
2. **Combined difference graph.** Show mean, 95% CI, and median for both differences, with a sample-count panel beside it.
3. **Group-wise versions.** Repeat the combined graph for all, `latent_start`, and `no_latent_start`.
4. **Trajectory-source versions.** Create one combined curve each for `real`, `noise`, and `no_image` source trajectories.
5. **One raw-value graph per trajectory.** Plot the 3 conditions in different colors.
6. **Combined raw-value graph.** Use the same layout as Step 2, across 30 trajectories, with mean, 95% CI, median, and sample-count panel.
7. **Group-wise and trajectory-source raw-value graphs.** Repeat the combined raw-value graph for each.

The trajectories from one question are related, so calculate every combined 95% CI by bootstrapping questions, not individual trajectories.

### Latent-start frequency in replay

Replay forces the next token, so no latent-start token is sampled. Report this replay frequency instead:

```text
percentage of trajectories where is_top1=True at any replay position
```

Report it by condition, group, and trajectory source. Use the label "start token ranked first in replay", not "latent starts".

### Done when

- There are 90 completed replays.
- All prefixes are free of latent token IDs.
- The parity check passes.
- `results.py exp-b` runs locally from the CSV files.
- Group and trajectory-source filters can be combined.

## Step 4: Select positions for Step 5

**Runs on:** local machine

**Command:**

```bash
python src/analysis/select_positions.py \
  --input results/exp_b/exp_b_replays.csv \
  --prefixes results/exp_b/exp_b_prefixes.csv \
  --out results/exp_b/selected_positions.csv \
  --gap-metric log_odds \
  --min-real-prob 0.2 \
  --fallback-min-real-prob 0.05 \
  --top-k 5
```

Use only `real` and `noise` rows. Pair them by `trajectory_id` and `position`.

### Gap options

| `--gap-metric` | Formula | Default `--min-gap` |
|---|---|---|
| **`log_odds`** | `log(p_real / (1 - p_real)) - log(p_noise / (1 - p_noise))` | 2.0 |
| **`log_prob`** | `log(p_real) - log(p_noise)` | 2.0 |
| **`prob`** | `p_real - p_noise` | 0.2 |

Clip `p_real` and `p_noise` to `[1e-7, 1 - 1e-7]` before calculating `log_odds`. `--min-gap` must remain configurable because its meaning depends on the metric.

### Selection

1. Calculate the gap for every paired row.
2. Keep rows with `p_real >= min_real_prob` and `gap >= min_gap`.
3. Keep the row with the largest gap for each question.
4. Sort by gap, largest first.
5. Take the top 5 questions.
6. If fewer than 5 questions remain, repeat with `p_real >= fallback_min_real_prob`.
7. If there are still fewer than 5, use all selected rows.

If two rows in one question tie, prefer `ends_at_start=True`, then the lower `position`. Do not create discovery or test splits. The selected rows are used together in Step 5.

### Output

`results/exp_b/selected_positions.csv`:

```text
rank,question_index,problem_id,group,category,trajectory_id,source_setting,position,prefix_length,ends_at_start,p_real,p_noise,gap,gap_metric,min_real_prob_used,min_gap_used,prefix_token_ids
```

`prefix_token_ids` must contain only the tokens before `position`.

### Done when

- There is at most one row per question.
- Rows are sorted by gap.
- The CSV records the metric and thresholds used.
- `--gap-metric` can be changed without rerunning Modal.
- The script prints how many questions passed at each threshold.

## Step 5

To be added later. Step 5 inference will run on Modal L40S with Transformers.
"""Per-step latent-start logging for Monet's patched vLLM runner (Experiment A).

Monet's `inference.apply_vllm_monet` swaps vLLM's GPU model runner for
`inference/vllm/monet_gpu_model_runner.py`. When `MONET_STEP_LOG` is set,
`apply()` re-executes that runner's source with four small insertions in
`execute_model` and installs the result under the same module names:

1. After `self._update_states(...)`: if any scheduled request was resumed from
   preemption, write a `preemption` event and raise. Preemption can break
   Monet's latent loop, so the run must fail.
2. Before the sampler runs: compute start-token statistics from the raw
   logits (before temperature, top-k, top-p and repetition penalty). They are
   taken here because the sampler may modify float32 logits in place.
3. After `valid_sampled_token_ids` is ready and before the ABS-VIS state
   update: write one JSON line per request that is not a partial prefill.
4. After the ABS-VIS state update: save the final hidden state behind each
   logged step (bfloat16) to `hidden_states.bin`, with an index line in
   `hidden_index.jsonl` (`req_id`, `step`, `row`, `fed_as_latent`, `dim`).
   `fed_as_latent` marks the vectors Monet feeds back as the next input
   embedding, which are the latent thoughts. Set `MONET_CAPTURE_HIDDEN=0` to
   turn this off.

Every insertion anchors on an exact source line. If an anchor is missing or is
not unique, `apply()` raises instead of silently running unpatched.

`apply()` must run after `import inference.apply_vllm_monet` and before any
other vLLM import. The calling script does this at the top level, so it also
runs in vLLM's spawned engine process, which re-imports the main script.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ENV_VAR = "MONET_STEP_LOG"
RUNNER_MODULE = "inference.vllm.monet_gpu_model_runner"
# The same aliases apply_vllm_monet installs.
ALIASES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.worker.gpu_model_runner",
    "vllm.worker.model_runner",
)

# (anchor line, code to insert, "before" | "after")
_EDITS = [
    (
        "        self._update_states(scheduler_output)\n",
        """\
        # MONET-STEP-LOG: fail on preemption, which can break the latent loop.
        _msl_cached = scheduler_output.scheduled_cached_reqs
        _msl_preempted = [
            _r for _r, _p in zip(_msl_cached.req_ids, _msl_cached.resumed_from_preemption) if _p
        ]
        if _msl_preempted:
            _monet_step_log_write({"event": "preemption", "req_ids": _msl_preempted})
            raise RuntimeError(
                f"[monet-step-log] requests resumed from preemption: {_msl_preempted}. "
                "Preemption can break latent reasoning; failing the run.")
""",
        "after",
    ),
    (
        "        # Sample the next token and get logprobs if needed.\n",
        """\
        # MONET-STEP-LOG: start-token stats from the raw logits, before the
        # sampler applies temperature / top-k / top-p / repetition penalty.
        _msl_stats = None
        if logits is not None and spec_decode_metadata is None:
            with torch.no_grad():
                _msl_l = logits[:self.input_batch.num_reqs].float()
                _msl_sid = self.latent_start_id
                _msl_start_logit = _msl_l[:, _msl_sid]
                _msl_start_lp = _msl_l.log_softmax(-1)[:, _msl_sid]
                _msl_other = _msl_l.clone()
                _msl_other[:, _msl_sid] = float("-inf")
                _msl_best_other = _msl_other.max(-1).values
                _msl_stats = torch.stack(
                    [_msl_start_logit, _msl_start_lp, _msl_best_other], dim=1
                ).cpu().tolist()
                del _msl_l, _msl_other
""",
        "before",
    ),
    (
        "        # ABS-VIS: update per-request state and capture pending embeddings\n",
        """\
        # MONET-STEP-LOG: one line per request that sampled a token this step.
        if _msl_stats is not None:
            _msl_skip = set(discard_sampled_tokens_req_indices)
            for _msl_i, _msl_req_id in enumerate(self.input_batch.req_ids):
                if _msl_i in _msl_skip or not valid_sampled_token_ids[_msl_i]:
                    continue
                _msl_logit, _msl_lp, _msl_best = _msl_stats[_msl_i]
                _msl_st = self.latent_state.get(_msl_req_id)
                _monet_step_log_write({
                    "req_id": _msl_req_id,
                    "step": len(self.requests[_msl_req_id].output_token_ids),
                    "start_prob": math.exp(_msl_lp),
                    "start_log_prob": _msl_lp,
                    "start_logit": _msl_logit,
                    "best_other_logit": _msl_best,
                    "start_logit_margin": _msl_logit - _msl_best,
                    "is_top1": _msl_logit > _msl_best,
                    "in_latent": bool(_msl_st and _msl_st.get("active")),
                    # Token the sampler drew, before Monet may force the end token.
                    "sampled_token_id": int(valid_sampled_token_ids[_msl_i][0]),
                })
""",
        "before",
    ),
    (
        "        # Cache the sampled tokens in the model runner, so that the scheduler\n",
        """\
        # MONET-STEP-LOG: save the final hidden state behind every logged step.
        # Row `step` is the state that produced token `step`. When `fed_as_latent`
        # is true, Monet feeds this exact vector back as the next input embedding,
        # so these rows are the latent thoughts.
        if _msl_stats is not None and _MSL_CAPTURE_HIDDEN:
            _msl_skip = set(discard_sampled_tokens_req_indices)
            _msl_rows = [(_msl_i, _msl_req_id)
                         for _msl_i, _msl_req_id in enumerate(self.input_batch.req_ids)
                         if _msl_i not in _msl_skip and valid_sampled_token_ids[_msl_i]]
            if _msl_rows:
                _msl_h = sample_hidden_states[[_i for _i, _ in _msl_rows]]
                _msl_meta = []
                for _msl_i, _msl_req_id in _msl_rows:
                    _msl_st = self.latent_state.get(_msl_req_id)
                    _msl_meta.append({
                        "req_id": _msl_req_id,
                        "step": len(self.requests[_msl_req_id].output_token_ids),
                        "fed_as_latent": bool(_msl_st and _msl_st.get("active")
                                              and _msl_st.get("pending") is not None),
                    })
                _monet_hidden_write(_msl_h, _msl_meta)
""",
        "before",
    ),
]

_FOOTER = '''

# ---- MONET-STEP-LOG helpers (added by patches/monet_step_log.py) ----
import json as _msl_json
import math

_MONET_STEP_LOG_PATCHED = True
_msl_file = None


def _monet_step_log_write(record):
    global _msl_file
    if _msl_file is None:
        path = os.environ["MONET_STEP_LOG"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _msl_file = open(path, "a", buffering=1, encoding="utf-8")
    _msl_file.write(_msl_json.dumps(record) + "\\n")
    _msl_file.flush()


# Hidden states: raw bfloat16 rows appended to hidden_states.bin next to the step
# log, with one JSON line per row in hidden_index.jsonl. Set
# MONET_CAPTURE_HIDDEN=0 to turn this off.
_MSL_CAPTURE_HIDDEN = os.environ.get("MONET_CAPTURE_HIDDEN", "1") != "0"
_msl_hidden = None
_msl_hidden_index = None
_msl_hidden_rows = 0


def _monet_hidden_write(hidden, meta):
    global _msl_hidden, _msl_hidden_index, _msl_hidden_rows
    if _msl_hidden is None:
        folder = os.path.dirname(os.environ["MONET_STEP_LOG"]) or "."
        os.makedirs(folder, exist_ok=True)
        _msl_hidden = open(os.path.join(folder, "hidden_states.bin"), "ab")
        _msl_hidden_index = open(os.path.join(folder, "hidden_index.jsonl"), "a",
                                 buffering=1, encoding="utf-8")
        _msl_hidden_rows = _msl_hidden.tell() // (hidden.shape[-1] * 2)
    data = hidden.detach().to(torch.bfloat16).contiguous().cpu()
    _msl_hidden.write(data.view(torch.int16).numpy().tobytes())
    _msl_hidden.flush()
    for k, m in enumerate(meta):
        m["row"] = _msl_hidden_rows + k
        m["dim"] = int(data.shape[-1])
        _msl_hidden_index.write(_msl_json.dumps(m) + "\\n")
    _msl_hidden_index.flush()
    _msl_hidden_rows += len(meta)
'''


def patch_source(src: str) -> str:
    """Return the runner source with the logging code inserted. Raises if any
    anchor is missing or appears more than once."""
    if "_monet_step_log_write" in src:
        raise RuntimeError("[monet-step-log] runner source is already patched")
    for anchor, code, where in _EDITS:
        count = src.count(anchor)
        if count != 1:
            raise RuntimeError(
                f"[monet-step-log] expected exactly 1 occurrence of {anchor.strip()!r} "
                f"in Monet's runner, found {count}. The runner source changed; "
                "update patches/monet_step_log.py.")
        src = src.replace(anchor, anchor + code if where == "after" else code + anchor)
    return src + _FOOTER


def apply() -> bool:
    """Install the patched runner if MONET_STEP_LOG is set. Returns True if patched."""
    if not os.environ.get(ENV_VAR):
        return False
    original = sys.modules.get(RUNNER_MODULE)
    if original is None or not all(sys.modules.get(a) is original for a in ALIASES):
        raise RuntimeError(
            "[monet-step-log] Monet's runner patch is not active. "
            "Import inference.apply_vllm_monet before applying the step-log patch.")
    if "vllm.v1.worker.gpu_worker" in sys.modules:
        raise RuntimeError(
            "[monet-step-log] vllm.v1.worker.gpu_worker is already imported, so it holds "
            "the unpatched runner. Apply the step-log patch before importing vLLM.")
    path = Path(original.__file__)
    module = types.ModuleType(RUNNER_MODULE)
    module.__file__ = str(path)
    module.__package__ = RUNNER_MODULE.rpartition(".")[0]
    sys.modules[RUNNER_MODULE] = module
    exec(compile(patch_source(path.read_text(encoding="utf-8")), str(path), "exec"), module.__dict__)
    for alias in ALIASES:
        sys.modules[alias] = module
    print(f"[monet-step-log] runner patched; logging to {os.environ[ENV_VAR]}", flush=True)
    return True

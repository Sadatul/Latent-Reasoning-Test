"""
Monet-7B latent reasoning on llama.cpp (llama-cpp-python), for GGUF models.

Port of Monet's patched vLLM runner
(https://github.com/NOVAglow646/Monet/blob/main/inference/vllm/monet_gpu_model_runner.py).
What that runner does, per request, at every decode step:

    sample token tid from the logits of this step
    if not active and tid == LATENT_START_ID:            -> active = True
    elif active and (tid == LATENT_END_ID or n >= LATENT_SIZE):
        active = False; n = 0; tid = LATENT_END_ID        (the sampled token is replaced)
    if active:
        pending = last-layer hidden state of this step     (after the final RMSNorm)
        n += 1
    next step: the new token's input embedding is replaced by `pending` if active

So after <abs_vis_token> the model is fed its own hidden states for LATENT_SIZE
steps (the tokens sampled meanwhile are placeholders that never reach the model),
then </abs_vis_token> is forced and normal text decoding resumes.

llama.cpp has everything needed:
  * llama_set_embeddings(ctx, True) makes llama_decode return, next to the logits,
    the per-token output of the final norm (`result_norm`, the same tensor vLLM
    returns as hidden_states and multiplies with lm_head).
  * llama_batch.embd feeds a float vector instead of a token id. Qwen2.5-VL uses
    M-RoPE, so an embedding batch carries 4 positions per token; for a text-like
    position p all four are p, exactly what llama.cpp uses for a text token.

The image goes through llama.cpp's mtmd (the mmproj GGUF), as in llama-cpp-python's
Qwen25VLChatHandler. Sampling is done here in numpy with vLLM's semantics
(repetition penalty over prompt + output tokens, temperature, top-k, top-p), so the
placeholder tokens and the forced end token behave as in the vLLM runner.

latent_mode:
  "vllm"  (default) exactly the runner above. Note that the hidden state that made
          the model pick <abs_vis_token> replaces the embedding of <abs_vis_token>
          itself, so LATENT_SIZE embeddings are fed at [start, p1 .. p_{K-1}].
  "train" Monet's training layout (modeling_qwen2_5_vl_monet.py): <abs_vis_token>
          keeps its own embedding, then K latents follow, each the hidden state of
          the previous position: [start, L1 .. LK, end].
  "off"   no latent mode: <abs_vis_token> is an ordinary token (what plain
          llama.cpp does). Use it as the ablation.
"""

import ctypes
import io
import time

import numpy as np
from llama_cpp._utils import suppress_stdout_stderr

LATENT_START_ID = 151666   # <abs_vis_token>
LATENT_END_ID = 151667     # </abs_vis_token>
IMAGE_PAD_ID = 151655      # <|image_pad|>, part of vLLM's prompt ids (repetition penalty)
LATENT_MODES = ("vllm", "train", "off")

# Monet's own settings (inference/load_and_gen_vllm.py)
MONET_SAMPLING = {"temperature": 0.1, "top_k": 50, "top_p": 0.8, "repetition_penalty": 1.01}
MONET_MIN_PIXELS = 256 * 28 * 28
MONET_MAX_PIXELS = 8192 * 28 * 28
MONET_LATENT_SIZE = 10     # inference/example.sh: export LATENT_SIZE=10


# ---------------------------------------------------------------- image

def smart_resize(height, width, factor=28, min_pixels=MONET_MIN_PIXELS, max_pixels=MONET_MAX_PIXELS):
    """Qwen2.5-VL's resize rule (qwen_vl_utils / Qwen2VLImageProcessor): both sides
    multiples of `factor`, area within [min_pixels, max_pixels], aspect ratio kept."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = ((height * width) / max_pixels) ** 0.5
        h_bar = max(factor, int(height / beta // factor) * factor)
        w_bar = max(factor, int(width / beta // factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = int(-(-height * beta // factor)) * factor
        w_bar = int(-(-width * beta // factor)) * factor
    return h_bar, w_bar


def load_image(path, min_pixels, max_pixels):
    """PNG bytes of the image resized with smart_resize, plus original and used size (w, h)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    orig = img.size
    h, w = smart_resize(img.height, img.width, min_pixels=min_pixels, max_pixels=max_pixels)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # lossless, so llama.cpp sees exactly these pixels
    return buf.getvalue(), list(orig), [w, h]


# ---------------------------------------------------------------- sampling

class Sampler:
    """vLLM's order: repetition penalty (prompt + output tokens), temperature,
    top-k, top-p, then sample. temperature <= 1e-5 means greedy."""

    def __init__(self, n_vocab, temperature, top_k, top_p, repetition_penalty, seed):
        self.t, self.k, self.p, self.rp = temperature, top_k, top_p, repetition_penalty
        self.seen = np.zeros(n_vocab, dtype=bool)
        self.rng = np.random.default_rng(seed)

    def add(self, ids):
        ids = [i for i in ids if 0 <= i < len(self.seen)]
        self.seen[ids] = True

    def __call__(self, logits):
        logits = logits.astype(np.float64, copy=True)
        if self.rp != 1.0:
            s = self.seen
            logits[s] = np.where(logits[s] > 0, logits[s] / self.rp, logits[s] * self.rp)
        if self.t <= 1e-5:
            return int(logits.argmax())
        logits /= self.t
        if 0 < self.k < len(logits):
            idx = np.argpartition(-logits, self.k - 1)[:self.k]
        else:
            idx = np.arange(len(logits))
        idx = idx[np.argsort(-logits[idx], kind="stable")]
        probs = np.exp(logits[idx] - logits[idx[0]])
        probs /= probs.sum()
        if self.p < 1.0:
            # keep the smallest prefix whose mass reaches top_p (at least one token)
            keep = int(np.searchsorted(np.cumsum(probs), self.p) + 1)
            idx, probs = idx[:keep], probs[:keep] / probs[:keep].sum()
        return int(idx[self.rng.choice(len(idx), p=probs)])


# ---------------------------------------------------------------- engine

class MonetLlamaCpp:
    def __init__(self, model_path, mmproj_path, n_ctx=16384, n_gpu_layers=-1, n_batch=512,
                 flash_attn=True, verbose=False, require_latent_tokens=True):
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        self.lc = llama_cpp
        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         n_batch=n_batch, n_ubatch=n_batch, flash_attn=flash_attn, verbose=verbose)
        self.ctx = self.llm._ctx.ctx
        self.model = self.llm._model.model
        self.n_batch = n_batch
        self.n_vocab = self.llm.n_vocab()
        self.n_embd = llama_cpp.llama_model_n_embd(self.model)
        if hasattr(llama_cpp, "llama_model_n_embd_inp"):
            n_inp = llama_cpp.llama_model_n_embd_inp(self.model)
            if n_inp != self.n_embd:
                raise RuntimeError(f"input width {n_inp} != hidden width {self.n_embd}: "
                                   "hidden states cannot be fed back as inputs for this model")
        if llama_cpp.llama_pooling_type(self.ctx) != llama_cpp.LLAMA_POOLING_TYPE_NONE:
            raise RuntimeError("context pooling must be NONE to read per-token hidden states")

        self.handler = None
        if mmproj_path is not None:
            self.handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=verbose)
            self.handler._init_mtmd_context(self.llm)
            self.mtmd = self.handler._mtmd_cpp
            self.marker = self.mtmd.mtmd_default_marker().decode("utf-8")
            self.mrope = bool(self.mtmd.mtmd_decode_use_mrope(self.handler.mtmd_ctx))
        else:
            self.marker = None
            self.mrope = llama_cpp.llama_model_rope_type(self.model) in (
                llama_cpp.LLAMA_ROPE_TYPE_MROPE, getattr(llama_cpp, "LLAMA_ROPE_TYPE_IMROPE", -2))
        self.n_pos_per_embd = 4 if self.mrope else 1

        self.stop_ids = {self.llm.token_eos()}
        for text in ("<|im_end|>", "<|endoftext|>"):
            ids = self.tokenize(text)
            if len(ids) == 1:
                self.stop_ids.add(ids[0])
        for text, want in (("<abs_vis_token>", LATENT_START_ID), ("</abs_vis_token>", LATENT_END_ID)):
            if require_latent_tokens and self.tokenize(text) != [want]:
                raise RuntimeError(f"{text} does not tokenize to [{want}]: not a Monet vocabulary?")

        self._tok_batch = llama_cpp.llama_batch_init(n_batch, 0, 1)
        self.llm._stack.callback(lambda: llama_cpp.llama_batch_free(self._tok_batch))
        self._init_embd_batch()
        self.verbose = verbose
        self.pos = 0     # next position (M-RoPE: image chunks advance it by max(h, w), not by their length)
        self.n_kv = 0    # KV cells used

    # ---- low level

    def _init_embd_batch(self):
        """A one-token embedding batch built by hand: llama_batch_init allocates
        only one position per token, M-RoPE needs four."""
        lc = self.lc
        self._e_embd = (ctypes.c_float * self.n_embd)()
        self._e_pos = (lc.llama_pos * self.n_pos_per_embd)()
        self._e_nseq = (ctypes.c_int32 * 1)(1)
        self._e_seq = (lc.llama_seq_id * 1)(0)
        self._e_seqp = (ctypes.POINTER(lc.llama_seq_id) * 2)(
            ctypes.cast(self._e_seq, ctypes.POINTER(lc.llama_seq_id)), None)
        self._e_out = (ctypes.c_int8 * 1)(1)
        self._embd_batch = lc.llama_batch(
            n_tokens=1, token=None,
            embd=ctypes.cast(self._e_embd, ctypes.POINTER(ctypes.c_float)),
            pos=ctypes.cast(self._e_pos, ctypes.POINTER(lc.llama_pos)),
            n_seq_id=ctypes.cast(self._e_nseq, ctypes.POINTER(ctypes.c_int32)),
            seq_id=ctypes.cast(self._e_seqp, ctypes.POINTER(ctypes.POINTER(lc.llama_seq_id))),
            logits=ctypes.cast(self._e_out, ctypes.POINTER(ctypes.c_int8)))

    def _check_room(self, n):
        if self.n_kv + n > self.llm.n_ctx():
            raise ContextFull(f"context full ({self.n_kv} + {n} > n_ctx {self.llm.n_ctx()})")

    def _decode(self, batch, what):
        rc = self.lc.llama_decode(self.ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed on {what} (code {rc})")

    def decode_tokens(self, tokens, output_last=True):
        """Evaluate text tokens at the next positions; the last one produces output."""
        b = self._tok_batch
        for start in range(0, len(tokens), self.n_batch):
            part = tokens[start:start + self.n_batch]
            self._check_room(len(part))
            b.n_tokens = len(part)
            for j, t in enumerate(part):
                b.token[j] = t
                b.pos[j] = self.pos + j
                b.n_seq_id[j] = 1
                b.seq_id[j][0] = 0
                b.logits[j] = output_last and start + j == len(tokens) - 1
            self._decode(b, "text")
            self.pos += len(part)
            self.n_kv += len(part)

    def decode_embedding(self, vec):
        """Evaluate one float vector (a latent) at the next position."""
        self._check_room(1)
        ctypes.memmove(self._e_embd, vec.astype(np.float32, copy=False).ctypes.data, self.n_embd * 4)
        for j in range(self.n_pos_per_embd):
            self._e_pos[j] = self.pos   # text-like position: same value in every M-RoPE section
        self._decode(self._embd_batch, "latent embedding")
        self.pos += 1
        self.n_kv += 1

    def logits(self):
        ptr = self.lc.llama_get_logits_ith(self.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def hidden(self):
        ptr = self.lc.llama_get_embeddings_ith(self.ctx, -1)
        if not ptr:
            raise RuntimeError("llama.cpp returned no hidden state (embeddings output disabled?)")
        h = np.ctypeslib.as_array(ptr, shape=(self.n_embd,)).copy()
        if not np.isfinite(h).all():
            raise RuntimeError("non-finite hidden state")
        return h

    def set_hidden_output(self, on):
        # With embeddings on, llama.cpp outputs every token of a batch, so it is only
        # switched on for the one-token decode steps.
        self.lc.llama_set_embeddings(self.ctx, bool(on))

    def tokenize(self, text):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def detok(self, ids):
        if not ids:
            return ""
        try:
            data = self.llm.detokenize(ids, special=True)
        except TypeError:
            data = self.llm.detokenize(ids)
        return data.decode("utf-8", errors="replace")

    def close(self):
        self.llm.close()

    def reset(self):
        mem = self.lc.llama_get_memory(self.ctx)
        self.lc.llama_memory_clear(mem, True)
        self.set_hidden_output(False)
        self.pos = self.n_kv = 0

    # ---- prompt

    def eval_prompt(self, text, image_bytes=None):
        """Evaluate the prompt (text with at most one image marker) into the KV cache,
        except its last token, which is returned so the decode loop can evaluate it
        with hidden-state output on. Returns (last_token, prompt_ids, image_info)."""
        self.reset()
        if image_bytes is None:
            ids = self.tokenize(text)
            self.decode_tokens(ids[:-1], output_last=False)
            return ids[-1], ids, None

        m, lc = self.mtmd, self.lc
        bitmap = self.handler._create_bitmap_from_bytes(image_bytes)
        chunks = m.mtmd_input_chunks_init()
        prompt_ids, image_info, last = [], None, None
        try:
            inp = m.mtmd_input_text()
            data = text.encode("utf-8")
            inp.text, inp.text_len, inp.add_special, inp.parse_special = data, len(data), False, True
            bitmaps = (m.mtmd_bitmap_p_ctypes * 1)(bitmap)
            with suppress_stdout_stderr(disable=self.verbose):
                rc = m.mtmd_tokenize(self.handler.mtmd_ctx, chunks, ctypes.byref(inp), bitmaps, 1)
            if rc != 0:
                raise ValueError("mtmd_tokenize failed")
            n_chunks = m.mtmd_input_chunks_size(chunks)
            for i in range(n_chunks):
                chunk = m.mtmd_input_chunks_get(chunks, i)
                if m.mtmd_input_chunk_get_type(chunk) == m.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n = ctypes.c_size_t()
                    ptr = m.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n))
                    toks = [ptr[j] for j in range(n.value)]
                    prompt_ids += toks
                    if i == n_chunks - 1:
                        last, toks = toks[-1], toks[:-1]
                    self.decode_tokens(toks, output_last=False)
                else:
                    if i == n_chunks - 1:
                        raise ValueError("prompt must end with text")
                    n_tok = m.mtmd_input_chunk_get_n_tokens(chunk)
                    self._check_room(n_tok)
                    img_tok = m.mtmd_input_chunk_get_tokens_image(chunk)
                    image_info = {"image_tokens": int(n_tok),
                                  "grid_wh": [int(m.mtmd_image_tokens_get_nx(img_tok)),
                                              int(m.mtmd_image_tokens_get_ny(img_tok))]}
                    prompt_ids += [IMAGE_PAD_ID] * n_tok
                    new_pos = lc.llama_pos(0)
                    t0 = time.time()
                    with suppress_stdout_stderr(disable=self.verbose):
                        rc = m.mtmd_helper_eval_chunk_single(self.handler.mtmd_ctx, self.ctx, chunk,
                                                             lc.llama_pos(self.pos), lc.llama_seq_id(0),
                                                             self.n_batch, False, ctypes.byref(new_pos))
                    if rc != 0:
                        raise ValueError("mtmd_helper_eval_chunk_single failed")
                    image_info["encode_seconds"] = round(time.time() - t0, 2)
                    self.pos = new_pos.value
                    self.n_kv += n_tok
        finally:
            m.mtmd_input_chunks_free(chunks)
            m.mtmd_bitmap_free(bitmap)
        return last, prompt_ids, image_info

    # ---- generation

    def generate(self, prompt_text, image_bytes=None, max_tokens=4096, latent_mode="vllm",
                 latent_size=MONET_LATENT_SIZE, temperature=0.0, top_k=-1, top_p=1.0,
                 repetition_penalty=1.0, seed=0):
        if latent_mode not in LATENT_MODES:
            raise ValueError(f"latent_mode must be one of {LATENT_MODES}")
        t0 = time.time()
        last, prompt_ids, image_info = self.eval_prompt(prompt_text, image_bytes)
        t_prompt = time.time() - t0
        sampler = Sampler(self.n_vocab, temperature, top_k, top_p, repetition_penalty, seed)
        sampler.add(prompt_ids)

        latent = latent_mode != "off"
        self.set_hidden_output(latent)
        self.decode_tokens([last])

        out = []              # output token ids, as vLLM would return them
        kinds = []            # per output token: text | start | placeholder | end
        latent_input = []     # per output token: its input embedding was replaced by a latent
        segments = []         # one per <abs_vis_token>
        p_start = []          # probability of <abs_vis_token> at every step (raw logits)
        active, pending, n_lat, seg = False, None, 0, None
        finish = "length"
        t1 = time.time()
        for _ in range(max_tokens):
            logits = self.logits()
            z = np.exp(logits - logits.max())
            p_start.append(float(z[LATENT_START_ID] / z.sum()))
            tid = sampler(logits)
            kind = "text"

            # --- the vLLM runner's state machine
            if latent:
                if not active and tid == LATENT_START_ID:
                    active, n_lat, kind = True, 0, "start"
                    seg = {"start_step": len(out), "n_latents": 0, "placeholders": [], "end": None}
                    segments.append(seg)
                    # "train": <abs_vis_token> is fed as itself; latents start after it
                    pending = self.hidden() if latent_mode == "vllm" else None
                elif active and (tid == LATENT_END_ID or n_lat >= latent_size):
                    seg["end"] = "end_token" if tid == LATENT_END_ID else "latent_size"
                    seg["end_step"] = len(out)
                    active, pending, n_lat, tid, kind = False, None, 0, LATENT_END_ID, "end"
                elif active:
                    seg["placeholders"].append(tid)
                    pending, kind = self.hidden(), "placeholder"
                if active and pending is not None:
                    n_lat += 1   # the runner's current_len

            sampler.add([tid])
            if tid in self.stop_ids:
                finish = "stop"
                if active:
                    seg["end"] = "eos_sampled"   # vLLM stops on EOS even in latent mode
                break
            out.append(tid)
            kinds.append(kind)
            feed_latent = active and pending is not None
            latent_input.append(feed_latent)
            try:
                if feed_latent:
                    self.decode_embedding(pending)
                    seg["n_latents"] += 1
                else:
                    self.decode_tokens([tid])
            except ContextFull:
                finish = "context_full"
                break
        else:
            if active:
                seg["end"] = "max_tokens"
        self.set_hidden_output(False)
        gen_seconds = time.time() - t1

        for s in segments:
            s["placeholder_text"] = self.detok(s.pop("placeholders"))
        return {
            "token_ids": out,
            "token_kinds": kinds,
            "latent_input": latent_input,
            "raw_output": self.detok(out),
            "clean_output": self.clean_text(out, kinds),
            "segments": segments,
            "finish_reason": finish,
            "prompt_ids_len": len(prompt_ids),
            "prompt_kv_tokens": self.n_kv - len(out),
            "completion_tokens": len(out),
            "image": image_info,
            "p_latent_start": p_start,
            "prompt_seconds": round(t_prompt, 2),
            "gen_seconds": round(gen_seconds, 2),
        }

    def clean_text(self, out, kinds):
        """Output with the placeholder tokens of each latent segment replaced by
        <latent> (Monet README's post-processing): <abs_vis_token><latent></abs_vis_token>."""
        parts, run = [], []
        for t, k in zip(out, kinds):
            if k == "placeholder":
                continue
            run.append(t)
            if k == "start":
                parts += [self.detok(run), "<latent>"]
                run = []
        parts.append(self.detok(run))
        return "".join(parts)


class ContextFull(RuntimeError):
    pass


from llama_cpp import Llama

model_path = "/home/imkabir/sadi/Latent-Reasoning-Test/hf_cache/hub/models--mradermacher--Monet-7B-GGUF/snapshots/7e62f81944a486777431eca13cfbd153509a1876/Monet-7B.Q8_0.gguf"

# Load vocabulary only; no model weights or GPU needed.
llm = Llama(
    model_path=model_path,
    vocab_only=True,
    n_gpu_layers=0,
    verbose=False,
)

special_tokens = {
    "<abs_vis_token>": 151666,
    "</abs_vis_token>": 151667,
}

for token, expected_id in special_tokens.items():
    ids = llm.tokenize(
        token.encode("utf-8"),
        add_bos=False,
        special=True,
    )

    decoded = llm.detokenize(
        [expected_id],
        special=True,
    ).decode("utf-8", errors="replace")

    print(f"\nToken: {token}")
    print(f"Expected ID: {expected_id}")
    print(f"Actual IDs: {ids}")
    print(f"Decoded expected ID: {decoded!r}")

    assert ids == [expected_id], (
        f"Unexpected token mapping: {token} -> {ids}"
    )
    assert decoded == token, (
        f"Unexpected decoding: {expected_id} -> {decoded!r}"
    )

print("\nPASS: Monet latent token mappings preserved.")

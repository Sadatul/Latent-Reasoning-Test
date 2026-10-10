# Monet vs Qwen2.5-VL: coding eval data

Downloads LiveCodeBench-lite and HumanEval+ from Hugging Face, builds two
LiveCodeBench subsets (easy problems, and problems released after Qwen2.5-VL),
and saves stats for each dataset.

Everything downloaded from Hugging Face stays inside this folder, so deleting
the folder removes it all.

## Requirements

- bash
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Run

```bash
bash setup.sh
```

This creates `.venv/` with uv, installs `huggingface_hub`, and runs
`prepare_datasets.py`.

## Later sessions

```bash
source env.sh
```

This sets `HF_HOME` to `hf_cache/` in this folder, so models downloaded later
(Monet-7B, Qwen2.5-VL-7B, GGUFs) also land here. Run it before any download.

## Files

| Path | What it is |
|---|---|
| `setup.sh` | Creates the uv environment and runs the script |
| `env.sh` | Points Hugging Face downloads into this folder |
| `prepare_datasets.py` | Downloads data, builds subsets, writes stats |
| `DATASET.md` | Data locations and formats; read this before writing eval code |
| `data/` | All datasets and stats (created by the script) |
| `hf_cache/` | Hugging Face cache (created on first download) |
| `.venv/` | Python environment |

See `dataset.md` for what each data file contains.

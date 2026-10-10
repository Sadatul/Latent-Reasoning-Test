# Project instructions

## Rules

1. All models and datasets must be downloaded inside this project folder (the folder that contains this CLAUDE.md), never to `~/.cache` or anywhere else on the machine. Deleting this folder must remove everything the project downloaded.
   - Run `source env.sh` before any download; it sets `HF_HOME` to `hf_cache/` in this folder.
   - In Python, never hardcode paths outside the project; build them from the project root.

## Reference

Data locations and formats:
@DATASET.md
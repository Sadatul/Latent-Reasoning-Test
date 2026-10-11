# Project instructions

## Rules

1. All models and datasets must be downloaded inside this project folder (the folder that contains this CLAUDE.md), never to `~/.cache` or anywhere else on the machine. Deleting this folder must remove everything the project downloaded.
   - Run `source env.sh` before any download; it sets `HF_HOME` to `hf_cache/` in this folder.
   - In Python, never hardcode paths outside the project; build them from the project root.

2. While evaluating models you must save the final answer and reasoning trace for that answer for all models. You can structure the answer and reasoning traces as you see fit. The goal is that later we will check the reasoning traces manually to evaluate the failures and successes cases. So structure it in a standard way so that its easier to analyze

3. use the quantized versions of qwen2.5-vl(https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF) and monet-7b (https://huggingface.co/mradermacher/Monet-7B-GGUF) as the gpu we will be running it on is 16 GB vram.

4. Add pause ability with state.json. If I run with the same output folder with, it should continue from where it left off. In case of a failure, I was want to able to start from the last result that was completed.

5. This machine has no GPU, so you won't be able to run it. You will write a script I will copy them to the actual machine. There I will run them.

## Reference

Data locations and formats:
@DATASET.md
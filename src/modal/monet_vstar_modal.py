"""Run Monet-7B on the V* benchmark on a Modal GPU.

The file has more than one entrypoint, so name the one to run with `::`.
Paths below are from the project root; running from src/modal also works.

V* evaluation (original):
  Smoke test:  modal run src/modal/monet_vstar_modal.py::main --limit 5
  Full run:    modal run --detach src/modal/monet_vstar_modal.py::main
  Sweep:       modal run --detach src/modal/monet_vstar_modal.py::main --latent-sizes 8,10,12,16
  Results:     modal volume get monet-vstar results ./results

Step 0 (V* in real / noise / no_image settings, Monet's own vLLM patch only):
  Smoke test:  modal run src/modal/monet_vstar_modal.py::run_step0 --limit 2
  Full run:    modal run --detach src/modal/monet_vstar_modal.py::run_step0
  Results:     modal volume get monet-vstar results/step0 ./results/step0
               (smoke test: results/step0_first2)

Experiment A (latent-start study, Step 2; see interpretibility/PLAN.md).
Needs Step 0's noise images (default: results/step0; pass --step0-dir otherwise):
  Upload:      modal volume put monet-vstar results/question_set.csv /inputs/question_set.csv
  Smoke test:  modal run src/modal/monet_vstar_modal.py::run_exp_a --limit 1
  Full run:    modal run --detach src/modal/monet_vstar_modal.py::run_exp_a
  Results:     modal volume get monet-vstar results/exp_a ./results/exp_a
               (smoke test: results/exp_a_smoke)
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import modal

MONET_COMMIT = "08939998d3d643a73a316e349faa34f420429153"
MODEL_REPO = "NOVAglow646/Monet-7B"
TSV_URL = "https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv"
TSV_MD5 = "b18854d7075574be06b631cd5f7d2d6a"  # the checksum VLMEvalKit uses

VOL_PATH = Path("/vol")
MODEL_DIR = VOL_PATH / "models" / "Monet-7B"
TSV_PATH = VOL_PATH / "data" / "VStarBench.tsv"
QUESTION_SET = VOL_PATH / "inputs" / "question_set.csv"
# Local files are resolved next to this file, so `modal run` works from any directory.
HERE = Path(__file__).resolve().parent

app = modal.App("monet-vstar")
volume = modal.Volume.from_name("monet-vstar", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.10.0",  # Monet's runner patch only matches this release
        "transformers==4.54.0",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "qwen_vl_utils==0.0.14",
        "pandas",
        "huggingface_hub",
    )
    .run_commands(
        "git clone https://github.com/NOVAglow646/Monet.git /root/Monet",
        f"git -C /root/Monet checkout {MONET_COMMIT}",
    )
    .add_local_file(HERE / "run_vstar.py", "/root/Monet/run_vstar.py")
    .add_local_file(HERE / "run_step0.py", "/root/Monet/run_step0.py")
    .add_local_file(HERE / "run_exp_a.py", "/root/Monet/run_exp_a.py")
    .add_local_dir(HERE / "patches", "/root/Monet/patches", ignore=["__pycache__"])
)


@app.function(image=image, volumes={"/vol": volume}, timeout=3600)
def prepare():
    """Download the model and dataset on a CPU container, so no GPU time is spent on it."""
    from huggingface_hub import snapshot_download

    if not (MODEL_DIR / "model-00004-of-00004.safetensors").exists():
        snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
    if not TSV_PATH.exists():
        TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TSV_URL, TSV_PATH)
        digest = hashlib.md5(TSV_PATH.read_bytes()).hexdigest()
        if digest != TSV_MD5:
            print(f"Warning: VStarBench.tsv md5 is {digest}, expected {TSV_MD5}")
    volume.commit()


# L40S has 48 GB. The model takes about 17 GB, leaving plenty of room for the KV cache.
@app.function(image=image, gpu="L40S", cpu=4, memory=32768, volumes={"/vol": volume}, timeout=7200)
def evaluate(latent_size: int, limit: int) -> dict:
    name = f"latent{latent_size}" + (f"_first{limit}" if limit else "")
    out = VOL_PATH / "results" / name
    subprocess.run(
        [
            sys.executable, "run_vstar.py",
            "--model", str(MODEL_DIR),
            "--data", str(TSV_PATH),
            "--out", str(out),
            "--limit", str(limit),
        ],
        cwd="/root/Monet",
        env={
            **os.environ,
            "LATENT_SIZE": str(latent_size),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HUB_OFFLINE": "1",
        },
        check=True,
    )
    volume.commit()
    return json.loads((out / "summary.json").read_text())


@app.local_entrypoint()
def main(limit: int = 0, latent_sizes: str = "10"):
    prepare.remote()
    jobs = [(int(size), limit) for size in latent_sizes.split(",")]
    for summary in evaluate.starmap(jobs):
        print(json.dumps(summary, indent=2))


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, volumes={"/vol": volume}, timeout=7200)
def exp_a(limit: int, force: bool, step0_dir: str) -> dict:
    """Experiment A: one trajectory per question and setting, with per-step start-token logging."""
    volume.reload()
    if not QUESTION_SET.exists():
        raise FileNotFoundError(
            f"{QUESTION_SET} is missing. Upload it first:\n"
            "  modal volume put monet-vstar results/question_set.csv /inputs/question_set.csv")
    out = VOL_PATH / "results" / ("exp_a_smoke" if limit else "exp_a")
    noise_dir = VOL_PATH / "results" / step0_dir / "noise_images"
    if not noise_dir.is_dir():
        raise FileNotFoundError(
            f"{noise_dir} is missing. Run Step 0 first (run_step0), or pass --step0-dir.")
    cmd = [
        sys.executable, "run_exp_a.py",
        "--model", str(MODEL_DIR),
        "--data", str(TSV_PATH),
        "--questions", str(QUESTION_SET),
        "--noise-dir", str(noise_dir),
        "--out", str(out),
        "--limit", str(limit),
    ] + (["--force"] if force else [])
    try:
        subprocess.run(
            cmd,
            cwd="/root/Monet",
            env={
                **os.environ,
                "LATENT_SIZE": "10",
                "MONET_STEP_LOG": str(out / "raw" / "step_log.jsonl"),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "HF_HUB_OFFLINE": "1",
            },
            check=True,
        )
    finally:
        volume.commit()  # keep partial outputs and state.json for debugging
    return json.loads((out / "checks.json").read_text())


@app.local_entrypoint()
def run_exp_a(limit: int = 0, force: bool = False, step0_dir: str = "step0"):
    prepare.remote()
    print(json.dumps(exp_a.remote(limit, force, step0_dir), indent=2))


# Three settings over all of V* can take several hours; state.json lets a rerun resume.
@app.function(image=image, gpu="L40S", cpu=4, memory=32768, volumes={"/vol": volume}, timeout=6 * 3600)
def step0(limit: int, settings: str, force: bool) -> dict:
    """Step 0: V* under real / noise / no_image with Monet's own vLLM patch only (no step logging)."""
    out = VOL_PATH / "results" / ("step0" + (f"_first{limit}" if limit else ""))
    env = {k: v for k, v in os.environ.items() if k != "MONET_STEP_LOG"}
    try:
        subprocess.run(
            [
                sys.executable, "run_step0.py",
                "--model", str(MODEL_DIR),
                "--data", str(TSV_PATH),
                "--out", str(out),
                "--limit", str(limit),
                "--settings", settings,
            ] + (["--force"] if force else []),
            cwd="/root/Monet",
            env={
                **env,
                "LATENT_SIZE": "10",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "HF_HUB_OFFLINE": "1",
            },
            check=True,
        )
    finally:
        volume.commit()  # keep finished settings and state.json even if a later one fails
    return json.loads((out / "summary.json").read_text())


@app.local_entrypoint()
def run_step0(limit: int = 0, settings: str = "real,noise,no_image", force: bool = False):
    prepare.remote()
    print(json.dumps(step0.remote(limit, settings, force), indent=2))
#!/usr/bin/env bash
# Install llama-cpp-python built with CUDA, plus numpy and pillow, into .venv.
# Run on the GPU machine after setup.sh. See README.md for prerequisites.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh

if [[ -z "${CUDACXX:-}" ]]; then
    if command -v nvcc >/dev/null; then
        CUDACXX="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        CUDACXX=/usr/local/cuda/bin/nvcc
    else
        echo "nvcc not found. Install the CUDA toolkit or set CUDACXX=/path/to/nvcc." >&2
        exit 1
    fi
fi
export CUDACXX
echo "using nvcc: $CUDACXX ($("$CUDACXX" --version | tail -n 1))"

# Builds llama.cpp from source (several minutes). --no-cache stops uv from
# reusing a CPU-only wheel built earlier.
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
    uv pip install --python .venv --no-cache --reinstall-package llama-cpp-python \
    llama-cpp-python numpy pillow

.venv/bin/python - <<'EOF'
import sys, llama_cpp
ok = llama_cpp.llama_supports_gpu_offload()
print("llama-cpp-python", llama_cpp.__version__, "| GPU offload:", ok)
sys.exit(0 if ok else "llama-cpp-python was built without CUDA; see README.md troubleshooting")
EOF

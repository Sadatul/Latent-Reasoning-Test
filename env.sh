# Source this before running anything here: `source env.sh`
# Keeps all Hugging Face downloads (datasets and models) inside this folder.
export HF_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hf_cache"

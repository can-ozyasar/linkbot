#!/usr/bin/env bash
# One-shot environment setup for the office 3D-reconstruction pipeline.
# Targets an RTX 5090 (Blackwell, CUDA 12.8) but works on any CUDA 12.8+ GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINGBOT_MAP_DIR="$REPO_ROOT/third_party/lingbot-map"
# Pinned upstream commit (main, as of 2026-07-03) for reproducible installs.
LINGBOT_MAP_COMMIT="598e01739b1973a0b484bf8e228207142abaea10"
CONDA_ENV_NAME="lingbot-map"

echo "== Checking prerequisites =="
command -v conda >/dev/null 2>&1 || { echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git not found." >&2; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. This pipeline requires an NVIDIA GPU (target: RTX 5090)." >&2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found. Install it before running the pipeline: sudo apt install ffmpeg" >&2
fi

echo "== Accepting Anaconda default channel Terms of Service =="
# Newer conda refuses to create envs from repo.anaconda.com channels
# non-interactively until ToS is accepted. Older conda has no `tos`
# subcommand at all, so failures here are expected and harmless.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "== Creating conda environment '$CONDA_ENV_NAME' (python 3.10) =="
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

echo "== Installing PyTorch 2.8.0 (CUDA 12.8, required for RTX 5090 / Blackwell) =="
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "== Cloning lingbot-map @ ${LINGBOT_MAP_COMMIT:0:12} =="
mkdir -p "$REPO_ROOT/third_party"
if [[ ! -d "$LINGBOT_MAP_DIR/.git" ]]; then
    git clone https://github.com/Robbyant/lingbot-map.git "$LINGBOT_MAP_DIR"
fi
git -C "$LINGBOT_MAP_DIR" fetch origin
git -C "$LINGBOT_MAP_DIR" checkout "$LINGBOT_MAP_COMMIT"

echo "== Installing lingbot-map (core + vis + render extras) =="
pip install -e "${LINGBOT_MAP_DIR}[vis,render]"

echo "== Installing FlashInfer (paged KV cache attention) =="
pip install --index-url https://pypi.org/simple flashinfer-python
pip install flashinfer-jit-cache -f https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/ || \
    echo "NOTE: prebuilt flashinfer-jit-cache unavailable for this platform; flashinfer will JIT-compile kernels on first run instead (slower first call only)."

echo "== Installing onnxruntime-gpu (batched sky segmentation) =="
pip install onnxruntime-gpu

echo "== Installing NVIDIA Kaolin (torch-2.8.0_cu128) =="
if ! pip install --index-url https://pypi.org/simple kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html; then
    cat >&2 <<'EOF'

WARNING: the prebuilt Kaolin wheel failed to install.
NVIDIA Kaolin does not always ship prebuilt kernels for brand-new GPU
architectures (RTX 50-series / Blackwell / sm_120) right away. Fall back to
building from source (requires a local CUDA 12.8 toolkit, not just the driver):

    pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git

Then re-run this script; it will skip steps that already succeeded.
EOF
    exit 1
fi

echo "== Building custom CUDA extensions (voxel_morton_ext, frustum_cull_ext) =="
CUDA_EXT_DIR="$LINGBOT_MAP_DIR/demo_render/render_cuda_ext"
if ! (cd "$CUDA_EXT_DIR" && python setup.py build_ext --inplace); then
    cat >&2 <<'EOF'

WARNING: native CUDA extension build failed. On brand-new GPUs (e.g. RTX
5090 / Blackwell / sm_120), the *system* nvcc (as opposed to PyTorch's own
bundled CUDA runtime) can be too old to know that architecture at all —
"nvcc fatal: Unsupported gpu architecture 'compute_120'".
Retrying with a PTX-forward-compatible target (compiles for sm_90; the
driver JIT-recompiles that PTX for the actual GPU the first time it loads)...
EOF
    rm -rf "$CUDA_EXT_DIR/build"
    if ! (cd "$CUDA_EXT_DIR" && TORCH_CUDA_ARCH_LIST="9.0+PTX" python setup.py build_ext --inplace); then
        cat >&2 <<'EOF'

WARNING: PTX fallback also failed. On Ubuntu 24.04+ (glibc >= 2.39), nvcc
releases older than CUDA 12.4 can't parse glibc's fortified headers at all
("identifier '__builtin_dynamic_object_size' is undefined" inside
stdlib.h/string_fortified.h/wchar2.h/etc — unrelated to GPU architecture).
Retrying once more with fortification disabled for this build only...
EOF
        rm -rf "$CUDA_EXT_DIR/build"
        if ! (cd "$CUDA_EXT_DIR" && TORCH_CUDA_ARCH_LIST="9.0+PTX" NVCC_APPEND_FLAGS="-D_FORTIFY_SOURCE=0" python setup.py build_ext --inplace); then
            cat >&2 <<'EOF'

ERROR: CUDA extension build failed on all three attempts (native, PTX
fallback, PTX + fortify workaround). Your system nvcc is too old for this
machine — check with: nvcc --version (anything below 12.4 is suspect on
Ubuntu 24.04; below 12.8 is suspect for an RTX 5090's sm_120).
Install CUDA Toolkit 12.8+ system-wide (NOT "sudo apt install
nvidia-cuda-toolkit", which tracks an old distro-packaged version):
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-8   # NOT the "cuda" metapackage (pulls a driver)
    echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
Do NOT let the installer touch your NVIDIA driver — only the toolkit/
compiler is needed; your existing driver already supports this GPU.
Re-run this script afterwards; it skips steps that already succeeded.
EOF
            exit 1
        fi
    fi
    echo "NOTE: this same nvcc-too-old issue can also hit FlashInfer's own runtime JIT compilation later (it isn't something this script controls). If FlashInfer errors during the pipeline with a similar message, the CUDA Toolkit 12.8+ upgrade above is the fix." >&2
fi

echo "== Downloading lingbot-map-long checkpoint (~several GB) =="
mkdir -p "$REPO_ROOT/checkpoints"
huggingface-cli download robbyant/lingbot-map lingbot-map-long.pt --local-dir "$REPO_ROOT/checkpoints"

echo "== Verifying installation =="
python "$REPO_ROOT/env/verify_gpu.py"

echo
echo "== Setup complete =="
echo "Next steps:"
echo "  1. conda activate $CONDA_ENV_NAME"
echo "  2. cp /path/to/your/office/video.mp4 $REPO_ROOT/data/"
echo "  3. bash $REPO_ROOT/scripts/run_pipeline.sh"
echo "     (no --video needed: it auto-processes every video dropped into data/)"

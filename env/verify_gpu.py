#!/usr/bin/env python3
"""Post-setup smoke test: confirms GPU, PyTorch, FlashInfer, Kaolin, the custom
CUDA extensions, and the checkpoint are all in place before running the pipeline."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKS_PASSED = True


def check(label, fn):
    global CHECKS_PASSED
    try:
        result = fn()
        print(f"[OK]   {label}: {result}")
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        CHECKS_PASSED = False


def check_torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return f"torch {torch.__version__}, {name}, sm_{major}{minor}"


def check_lingbot_map():
    import lingbot_map  # noqa: F401

    return "importable"


def check_flashinfer():
    import flashinfer  # noqa: F401

    return "importable"


def check_kaolin():
    import kaolin

    return f"kaolin {kaolin.__version__}"


def check_cuda_ext():
    ext_dir = REPO_ROOT / "third_party" / "lingbot-map" / "demo_render" / "render_cuda_ext"
    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    import voxel_morton_ext  # noqa: F401
    import frustum_cull_ext  # noqa: F401

    return "voxel_morton_ext, frustum_cull_ext importable"


def check_checkpoint():
    ckpt = REPO_ROOT / "checkpoints" / "lingbot-map-long.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return f"{ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)"


def check_ffmpeg():
    import shutil

    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


if __name__ == "__main__":
    check("PyTorch + CUDA", check_torch_cuda)
    check("lingbot_map package", check_lingbot_map)
    check("FlashInfer", check_flashinfer)
    check("Kaolin", check_kaolin)
    check("Custom CUDA extensions", check_cuda_ext)
    check("Checkpoint file", check_checkpoint)
    check("ffmpeg", check_ffmpeg)
    print()
    if CHECKS_PASSED:
        print("All checks passed — ready to run scripts/run_pipeline.sh")
    else:
        print("Some checks FAILED — see above before running the pipeline.")
    sys.exit(0 if CHECKS_PASSED else 1)

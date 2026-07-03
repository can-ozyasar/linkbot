# linkbot

Turns a walkthrough video of our office into a rendered 3D point-cloud
flythrough (`.mp4`), built on top of [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)
— a feed-forward 3D foundation model for streaming reconstruction.

This repo does **not** vendor lingbot-map's source. `env/setup.sh` clones it
into `third_party/lingbot-map/` (gitignored) at a pinned commit, installs it
and its CUDA-specific dependencies, and downloads the model checkpoint.
`scripts/run_pipeline.sh` then drives lingbot-map's own offline rendering
pipeline (`demo_render/batch_demo.py`) against your video.

Target hardware: **RTX 5090** (Blackwell, CUDA 12.8). Any GPU that supports
PyTorch 2.8.0 + CUDA 12.8 should also work.

## Repo layout

```
env/setup.sh          One-shot installer: conda env, PyTorch, lingbot-map,
                       FlashInfer, Kaolin, CUDA extensions, checkpoint download
env/verify_gpu.py      Post-setup smoke test (run automatically at the end of setup.sh)
config/office_indoor.yaml   Render preset: indoor scene, follow (chase) camera
scripts/run_pipeline.sh     Auto-processes every video in data/ (or one via --video)
data/                  Drop your office video here (gitignored — never committed)
outputs/               Rendered MP4s + per-frame NPZ predictions land here (gitignored)
third_party/           Cloned lingbot-map source (created by setup.sh, gitignored)
checkpoints/           Downloaded model weights (created by setup.sh, gitignored)
```

## Setup (run once on the 5090 machine)

Prerequisites: NVIDIA driver supporting CUDA 12.8+, [conda](https://docs.conda.io/en/latest/miniconda.html),
git, ffmpeg (`sudo apt install ffmpeg` if `setup.sh` warns it's missing).

```bash
git clone https://github.com/<your-account>/linkbot.git
cd linkbot
bash env/setup.sh
```

This will:
1. Create a `lingbot-map` conda env (Python 3.10)
2. Install PyTorch 2.8.0 / torchvision 0.23.0 for CUDA 12.8
3. Clone `Robbyant/lingbot-map` into `third_party/lingbot-map/` at a pinned commit
4. `pip install -e` it with the `vis` and `render` extras (viser, open3d, onnxruntime, etc.)
5. Install FlashInfer (paged KV-cache attention) and NVIDIA Kaolin
6. Build the two custom CUDA extensions (`voxel_morton_ext`, `frustum_cull_ext`)
7. Download the `lingbot-map-long` checkpoint (~several GB, recommended for long/large scenes)
8. Run `env/verify_gpu.py` to confirm everything imports and the GPU is visible

If it prints `All checks passed`, the environment is ready. See
[Troubleshooting](#troubleshooting) below if any step fails — Kaolin in
particular does not always ship prebuilt kernels for brand-new GPU
architectures on day one, and `setup.sh` prints the source-build fallback if
that happens.

## Running the pipeline

Drop one or more videos into `data/` and run the script with no arguments —
it auto-discovers every video there and processes each one:

```bash
conda activate lingbot-map
cp /path/to/office_walkthrough.mp4 data/
bash scripts/run_pipeline.sh
```

Re-running is safe: videos that already have output under `outputs/` are
skipped, so you can drop in a new file later and just re-run the same
command. If one video in a batch fails, the rest still run — failures are
summarized at the end and the script exits non-zero if any occurred.

To process a single specific file instead of scanning `data/` (e.g. it lives
elsewhere, or you want a custom output name):

```bash
bash scripts/run_pipeline.sh --video /path/to/office_walkthrough.mp4 --name office
```

Output lands in `outputs/<video-name>/` (named after the input file):

| File | Description |
|---|---|
| `<name>_pointcloud.mp4` | Rendered point-cloud flythrough |
| `<name>_pointcloud_rgb.mp4` | Original RGB frames as video |
| `<name>_pointcloud_config.yaml` | Full config snapshot of this run |
| `<name>.npz` (from `--save_predictions`) | Saved predictions for the whole scene, for re-rendering with different camera settings later without re-running inference |

**Format:** every `.mp4` above is normalized to **H.264 (libx264) / yuv420p**
after rendering — the most universally compatible combination, playable in
any browser, Windows Media Player, QuickTime, and VLC with no extra codecs.
This matters because lingbot-map's own encoder silently falls back to the
less-compatible `mp4v` codec for some of these files if anything's off with
its ffmpeg call; `run_pipeline.sh` re-encodes every output file itself so
that risk doesn't reach you, and `env/verify_gpu.py` checks upfront that your
ffmpeg build actually has a `libx264` encoder to begin with.

`run_pipeline.sh` probes each video with `ffprobe` and auto-picks a
`--keyframe_interval` that keeps each inference window under ~1500 real
frames (the ratio validated in lingbot-map's own long-sequence example, and
confirmed against `batch_demo.py`'s actual window-capacity formula:
`num_scale_frames + (window_size - num_scale_frames) * keyframe_interval`).
You can override anything:

```bash
# Force parameters instead of auto-estimating
bash scripts/run_pipeline.sh --fps 15 --keyframe-interval 3

# Use a different checkpoint or config
bash scripts/run_pipeline.sh --model checkpoints/lingbot-map.pt --config config/office_indoor.yaml

# Pass any additional batch_demo.py flag straight through
bash scripts/run_pipeline.sh -- --use_sdpa --save_glb
```

## Camera path

`config/office_indoor.yaml` uses a `follow` (chase) camera for the whole
sequence — good for checking room layout and geometry accuracy on a first
pass. To get a top-down overview instead, replace the `camera.segments`
block with:

```yaml
camera:
  fov: 60.0
  segments:
    - mode: birdeye
      frames: [0, -1]
      reveal_height_mult: 2.5
```

Or mix a follow pass with a birdeye reveal at the end — see lingbot-map's
[Camera Path docs](https://github.com/Robbyant/lingbot-map#camera-path-yaml)
for all available modes (`follow`, `birdeye`, `static`, `pivot`).

## Troubleshooting

**`nvcc fatal: Unsupported gpu architecture 'compute_120'`** — your
*system* nvcc (`nvcc --version`) is older than CUDA 12.8 and doesn't know
about Blackwell/`sm_120` yet, even though the driver and PyTorch both
support the RTX 5090 fine (common if it came from `sudo apt install
nvidia-cuda-toolkit`, which tracks an old distro-packaged version).
`env/setup.sh` retries the custom CUDA extension build with a
PTX-forward-compatible target (`sm_90`, JIT-recompiled by the driver for
the actual GPU at first load) when the native build fails.

**`identifier "__builtin_dynamic_object_size" is undefined`** (errors
inside glibc headers like `stdlib.h`, `string_fortified.h`, `wchar2.h`,
`stdio2.h`) — a *second*, independent nvcc-too-old symptom seen on Ubuntu
24.04+ (glibc >= 2.39): nvcc releases before CUDA 12.4 can't parse glibc's
newer fortified headers at all, regardless of GPU architecture.
`env/setup.sh` retries once more with `-D_FORTIFY_SOURCE=0` (via
`NVCC_APPEND_FLAGS`) when this happens.

If your system nvcc hits *both* of the above, that's a strong signal it's
simply too old for this machine (Ubuntu 24.04 + RTX 5090) and workarounds
will keep surfacing new gaps — e.g. FlashInfer's own runtime JIT compilation
uses the same system nvcc and isn't something this script can patch around.
The durable fix is installing CUDA Toolkit 12.8+ system-wide and putting it
first on `PATH` — but don't let its installer touch your NVIDIA driver,
only the toolkit/compiler is needed:

```bash
# https://developer.nvidia.com/cuda-downloads — pick Ubuntu 24.04 / x86_64, or:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8   # NOT the "cuda" metapackage (pulls a driver)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
nvcc --version   # should now report release 12.8
```

**Kaolin prebuilt wheel fails to install** — NVIDIA doesn't always publish
prebuilt Kaolin kernels for a brand-new GPU architecture (RTX 50-series /
Blackwell / `sm_120`) immediately. Build from source instead (needs a local
CUDA 12.8 toolkit, not just the driver), then re-run `env/setup.sh`:

```bash
pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git
```

**FlashInfer unavailable / unstable** — fall back to PyTorch's native SDPA
attention:

```bash
bash scripts/run_pipeline.sh -- --use_sdpa
```

**Out of memory** — unlikely on a 32GB 5090, but if it happens, per-frame
CPU offload is already on by default; you can also shrink the scale-frame
window:

```bash
bash scripts/run_pipeline.sh -- --num_scale_frames 2
```

**`env/verify_gpu.py` reports a FAIL** — re-run the specific `env/setup.sh`
step tied to that check; the script is idempotent and safe to re-run in full.

## License

This repo's own code (`env/`, `scripts/`, `config/`) is MIT-licensed — see
[LICENSE](LICENSE). It installs and drives
[Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map) (Apache
License 2.0) as an external dependency at setup time; that project's source
is not redistributed here.

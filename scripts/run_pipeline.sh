#!/usr/bin/env bash
# Runs the offline lingbot-map reconstruction pipeline (demo_render/batch_demo.py)
# on a single office walkthrough video and produces a point-cloud flythrough MP4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LINGBOT_MAP_DIR="$REPO_ROOT/third_party/lingbot-map"

VIDEO=""
NAME=""
MODEL_PATH="$REPO_ROOT/checkpoints/lingbot-map-long.pt"
CONFIG_PATH="$REPO_ROOT/config/office_indoor.yaml"
TARGET_FPS=10
KEYFRAME_INTERVAL=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") --video <path> [options] [-- <extra batch_demo.py args>]

Required:
  --video PATH             Path to the office walkthrough video (mp4/mov/mkv)

Optional:
  --name NAME               Output subfolder name under outputs/ (default: video filename stem)
  --model PATH               Checkpoint path (default: checkpoints/lingbot-map-long.pt)
  --config PATH               Render config YAML (default: config/office_indoor.yaml)
  --fps N                     Frames per second sampled from the video (default: 10)
  --keyframe-interval N     Force a keyframe interval instead of auto-estimating it
  -- ...                       Anything after -- is passed straight through to batch_demo.py

Example:
  $(basename "$0") --video data/office_walkthrough.mp4
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) VIDEO="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --fps) TARGET_FPS="$2"; shift 2 ;;
        --keyframe-interval) KEYFRAME_INTERVAL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$VIDEO" ]]; then
    echo "Error: --video is required" >&2
    usage
    exit 1
fi
if [[ ! -f "$VIDEO" ]]; then
    echo "Error: video not found at $VIDEO" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: checkpoint not found at $MODEL_PATH (run env/setup.sh first)" >&2
    exit 1
fi
if [[ ! -d "$LINGBOT_MAP_DIR" ]]; then
    echo "Error: $LINGBOT_MAP_DIR not found (run env/setup.sh first)" >&2
    exit 1
fi
command -v ffprobe >/dev/null 2>&1 || { echo "Error: ffprobe not found (install ffmpeg)" >&2; exit 1; }

if [[ -z "$NAME" ]]; then
    NAME="$(basename "$VIDEO")"
    NAME="${NAME%.*}"
fi

if [[ -z "$KEYFRAME_INTERVAL" ]]; then
    DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
    APPROX_FRAMES=$(awk -v d="$DURATION" -v f="$TARGET_FPS" 'BEGIN { printf "%d", d * f }')
    # ~1500 real frames/window keeps a comfortable margin under the validated
    # 8 + 120*keyframe_interval capacity of a window (see README worked example).
    KEYFRAME_INTERVAL=$(awk -v n="$APPROX_FRAMES" 'BEGIN { ki = int((n + 1499) / 1500); print (ki < 1) ? 1 : ki }')
    echo "Video duration: ${DURATION}s (~${APPROX_FRAMES} frames @ ${TARGET_FPS}fps) -> auto keyframe_interval=${KEYFRAME_INTERVAL}"
fi

OUTPUT_FOLDER="$REPO_ROOT/outputs/$NAME"
mkdir -p "$OUTPUT_FOLDER"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx lingbot-map; then
    eval "$(conda shell.bash hook)"
    conda activate lingbot-map
fi

cd "$LINGBOT_MAP_DIR"
python demo_render/batch_demo.py \
    --video_path "$VIDEO" \
    --fps "$TARGET_FPS" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --mode windowed --window_size 128 \
    --keyframe_interval "$KEYFRAME_INTERVAL" \
    --overlap_keyframes 8 \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions \
    "${EXTRA_ARGS[@]}"

echo
echo "Done. Output: $OUTPUT_FOLDER/${NAME}_pointcloud.mp4"

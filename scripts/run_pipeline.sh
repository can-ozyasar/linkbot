#!/usr/bin/env bash
# Runs the offline lingbot-map reconstruction pipeline (demo_render/batch_demo.py)
# on office walkthrough video(s) and produces a point-cloud flythrough MP4 per video.
#
# With no --video, auto-discovers every video file under data/ and processes
# each one (skipping ones that already have output), matching the drop-a-file-
# and-run workflow lingbot-map's own demo_render/process_videos.sh uses for
# batches of raw video.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LINGBOT_MAP_DIR="$REPO_ROOT/third_party/lingbot-map"
DATA_DIR="$REPO_ROOT/data"

VIDEO=""
NAME=""
MODEL_PATH="$REPO_ROOT/checkpoints/lingbot-map-long.pt"
CONFIG_PATH="$REPO_ROOT/config/office_indoor.yaml"
TARGET_FPS=10
KEYFRAME_INTERVAL_OVERRIDE=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--video <path>] [options] [-- <extra batch_demo.py args>]

With no --video: scans $DATA_DIR/ for video files (mp4/mov/mkv/avi) and
processes every one that doesn't already have output under outputs/,
so you can just drop a file in $DATA_DIR/ and re-run this script.

Optional:
  --video PATH              Process only this video (bypasses auto-discovery)
  --name NAME                Output subfolder name under outputs/ (only with --video;
                              default: video filename stem)
  --model PATH                Checkpoint path (default: checkpoints/lingbot-map-long.pt)
  --config PATH                Render config YAML (default: config/office_indoor.yaml)
  --fps N                      Frames per second sampled from the video (default: 10)
  --keyframe-interval N      Force a keyframe interval instead of auto-estimating it
  -- ...                        Anything after -- is passed straight through to batch_demo.py

Examples:
  $(basename "$0")                                   # process everything new in data/
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
        --keyframe-interval) KEYFRAME_INTERVAL_OVERRIDE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: checkpoint not found at $MODEL_PATH (run env/setup.sh first)" >&2
    exit 1
fi
if [[ ! -d "$LINGBOT_MAP_DIR" ]]; then
    echo "Error: $LINGBOT_MAP_DIR not found (run env/setup.sh first)" >&2
    exit 1
fi
# ffmpeg (not just ffprobe) is required: lingbot-map's video encoder tries
# ffmpeg+libx264 (universally playable) first and silently falls back to
# OpenCV's mp4v codec (spottier player/browser support) if it's missing.
# We also use ffmpeg below to normalize every output to libx264/yuv420p.
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not found. Install it: sudo apt install ffmpeg" >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "Error: ffprobe not found (part of the ffmpeg package)" >&2; exit 1; }

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx lingbot-map; then
    eval "$(conda shell.bash hook)"
    conda activate lingbot-map
fi

estimate_keyframe_interval() {
    local video="$1" duration approx_frames ki
    duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video")
    approx_frames=$(awk -v d="$duration" -v f="$TARGET_FPS" 'BEGIN { printf "%d", d * f }')
    # ~1500 real frames/window keeps a comfortable margin under the validated
    # 8 + 120*keyframe_interval capacity of a window (see README worked example).
    ki=$(awk -v n="$approx_frames" 'BEGIN { k = int((n + 1499) / 1500); print (k < 1) ? 1 : k }')
    echo "  duration=${duration}s (~${approx_frames} frames @ ${TARGET_FPS}fps) -> auto keyframe_interval=${ki}" >&2
    echo "$ki"
}

# Re-encodes every mp4 in a folder to libx264/yuv420p so it opens in any
# player/browser without extra codecs, regardless of what lingbot-map's
# internal encoder produced for that particular file.
normalize_videos() {
    local folder="$1" f tmp
    for f in "$folder"/*.mp4; do
        [[ -e "$f" ]] || continue
        tmp="${f%.mp4}.normalize.tmp.mp4"
        if ffmpeg -y -v error -i "$f" -c:v libx264 -pix_fmt yuv420p -crf 18 \
            -movflags +faststart -c:a copy "$tmp"; then
            mv "$tmp" "$f"
        else
            rm -f "$tmp"
            echo "  WARNING: could not normalize $(basename "$f") to libx264/yuv420p; left as-is" >&2
        fi
    done
}

process_one_video() {
    local video="$1" name="$2" skip_if_done="$3"
    local output_folder="$REPO_ROOT/outputs/$name"
    local output_video="$output_folder/${name}_pointcloud.mp4"

    if [[ "$skip_if_done" == "1" && -f "$output_video" ]]; then
        echo "[skip] $name -> already has $output_video"
        return 0
    fi

    echo "=== $name ($video) ==="
    mkdir -p "$output_folder"

    local ki="$KEYFRAME_INTERVAL_OVERRIDE"
    if [[ -z "$ki" ]]; then
        ki=$(estimate_keyframe_interval "$video")
    fi

    (cd "$LINGBOT_MAP_DIR" && python demo_render/batch_demo.py \
        --video_path "$video" \
        --fps "$TARGET_FPS" \
        --output_folder "$output_folder" \
        --model_path "$MODEL_PATH" \
        --config "$CONFIG_PATH" \
        --mode windowed --window_size 128 \
        --keyframe_interval "$ki" \
        --overlap_keyframes 8 \
        --camera_vis default --keyframes_only_points \
        --frame_tag --frame_tag_position top_right \
        --save_predictions \
        "${EXTRA_ARGS[@]}")

    echo "  normalizing output video(s) to libx264/yuv420p..."
    normalize_videos "$output_folder"

    echo "  done -> $output_video"
}

if [[ -n "$VIDEO" ]]; then
    [[ -f "$VIDEO" ]] || { echo "Error: video not found at $VIDEO" >&2; exit 1; }
    if [[ -z "$NAME" ]]; then
        NAME="$(basename "$VIDEO")"
        NAME="${NAME%.*}"
    fi
    process_one_video "$VIDEO" "$NAME" 0
else
    shopt -s nullglob nocaseglob
    VIDEOS=("$DATA_DIR"/*.mp4 "$DATA_DIR"/*.mov "$DATA_DIR"/*.mkv "$DATA_DIR"/*.avi)
    shopt -u nullglob nocaseglob

    if [[ ${#VIDEOS[@]} -eq 0 ]]; then
        echo "Error: no video found in $DATA_DIR/. Drop a video there, or pass --video <path>." >&2
        exit 1
    fi

    echo "Found ${#VIDEOS[@]} video(s) in $DATA_DIR/"
    FAILED=0
    for v in "${VIDEOS[@]}"; do
        n="$(basename "$v")"
        n="${n%.*}"
        # `||` suspends -e for the whole call so one bad video doesn't abort the batch.
        process_one_video "$v" "$n" 1 || { echo "[FAILED] $n (see errors above)"; FAILED=$((FAILED + 1)); }
    done
    if [[ "$FAILED" -gt 0 ]]; then
        echo
        echo "Finished with $FAILED failure(s) — see [FAILED] lines above." >&2
        exit 1
    fi
fi

echo
echo "All done. Outputs are under $REPO_ROOT/outputs/<video-name>/<video-name>_pointcloud.mp4"

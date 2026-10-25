#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FFMPEG=${FFMPEG:-ffmpeg}
FFPROBE=${FFPROBE:-ffprobe}
OUTPUT="$ROOT_DIR/assets/execution_results_comparison.mp4"

for tool in "$FFMPEG" "$FFPROBE"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required tool not found: %s\n' "$tool" >&2
        exit 1
    fi
done

for file in \
    "$ROOT_DIR/videos/notebook_part1_preset_biped.mp4" \
    "$ROOT_DIR/videos/notebook_part3_skill_reuse_biped.mp4" \
    "$ROOT_DIR/videos/notebook_part3_skill_reuse_biped_walk.mp4" \
    "$OUTPUT"; do
    if [[ ! -f "$file" ]]; then
        printf 'Required file not found: %s\n' "$file" >&2
        exit 1
    fi
done

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# Reuse the title strip from the existing comparison video.
"$FFMPEG" -hide_banner -loglevel error -y \
    -i "$OUTPUT" -vf 'crop=1920:60:0:0' \
    -frames:v 1 "$WORK_DIR/titles.png"

"$FFMPEG" -hide_banner -loglevel error -y \
    -stream_loop -1 -i "$ROOT_DIR/videos/notebook_part1_preset_biped.mp4" \
    -stream_loop -1 -i "$ROOT_DIR/videos/notebook_part3_skill_reuse_biped.mp4" \
    -i "$ROOT_DIR/videos/notebook_part3_skill_reuse_biped_walk.mp4" \
    -loop 1 -framerate 30 -i "$WORK_DIR/titles.png" \
    -filter_complex '[0:v]fps=30,setsar=1,setpts=PTS-STARTPTS[a];[1:v]fps=30,setsar=1,setpts=PTS-STARTPTS[b];[2:v]trim=start=7,setpts=PTS-STARTPTS,fps=30,setsar=1,loop=loop=-1:size=199:start=0[c];[a][b][c]hstack=inputs=3[base];[base][3:v]overlay=0:0,format=yuv420p[v]' \
    -map '[v]' -frames:v 3600 -an \
    -c:v libx264 -crf 18 -preset medium -movflags +faststart \
    "$WORK_DIR/comparison.mp4"

"$FFMPEG" -v error -i "$WORK_DIR/comparison.mp4" -f null -
"$FFPROBE" -v error \
    -show_entries stream=width,height,r_frame_rate,nb_frames \
    -show_entries format=duration -of json "$WORK_DIR/comparison.mp4"

cp "$WORK_DIR/comparison.mp4" "$OUTPUT"
printf 'Created: %s\n' "$OUTPUT"

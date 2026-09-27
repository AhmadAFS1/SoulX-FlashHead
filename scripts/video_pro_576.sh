#!/usr/bin/env bash
# Labelled comparison videos of two 576x320 PRO runs: full frame side by side, and the mouth
# crop (the review's 160x96 box at x=86, y=193) at 4x, stacked. Audio from the first run.
#
# usage: scripts/video_pro_576.sh <run-a> "<label-a>" <run-b> "<label-b>" <output-prefix>
#   writes <output-prefix>-full.mp4 and <output-prefix>-mouth.mp4
set -euo pipefail

if [[ $# -lt 5 ]]; then
  sed -n '2,6p' "$0"
  exit 2
fi
a="${1%/}/video.mp4"; la="$2"; b="${3%/}/video.mp4"; lb="$4"; out="$5"
font=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
mkdir -p "$(dirname "${out}")"
# ffmpeg drawtext needs ':' and ''' escaped inside text=
esc() { printf '%s' "$1" | sed -e "s/[:']/\\\\&/g"; }
la="$(esc "${la}")"; lb="$(esc "${lb}")"

ffmpeg -y -loglevel error -i "${a}" -i "${b}" -filter_complex "\
[0:v]pad=iw:ih+40:0:40:black,drawtext=fontfile=${font}:text='${la}':x=10:y=10:fontsize=19:fontcolor=white,pad=iw+8:ih:0:0:black[a];\
[1:v]pad=iw:ih+40:0:40:black,drawtext=fontfile=${font}:text='${lb}':x=10:y=10:fontsize=19:fontcolor=0xFFD24A[b];\
[a][b]hstack=inputs=2[v]" -map "[v]" -map 0:a? -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p \
  -c:a aac -shortest -movflags +faststart "${out}-full.mp4"

ffmpeg -y -loglevel error -i "${a}" -i "${b}" -filter_complex "\
[0:v]crop=160:96:86:193,scale=640:384:flags=lanczos,pad=iw:ih+46:0:46:black,drawtext=fontfile=${font}:text='1. ${la}':x=10:y=13:fontsize=20:fontcolor=white[a];\
[1:v]crop=160:96:86:193,scale=640:384:flags=lanczos,pad=iw:ih+46:0:46:black,drawtext=fontfile=${font}:text='2. ${lb}':x=10:y=13:fontsize=20:fontcolor=0xFFD24A[b];\
[a][b]vstack=inputs=2[v]" -map "[v]" -map 0:a? -c:v libx264 -crf 12 -preset slow -pix_fmt yuv420p \
  -c:a aac -shortest -movflags +faststart "${out}-mouth.mp4"

ls -la "${out}"-full.mp4 "${out}"-mouth.mp4 | awk '{printf "%.1fM  %s\n", $5/1048576, $9}'

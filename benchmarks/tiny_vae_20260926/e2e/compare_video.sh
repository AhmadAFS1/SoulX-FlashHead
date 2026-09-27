#!/bin/bash
# usage: compare_video.sh <out-stem> "<label1>" <run1> "<label2>" <run2> "<label3>" <run3> "<label4>" <run4>
# 2x2 labelled mouth-crop grid (4x lanczos of rows 193:289, cols 86:246) and a 4-up full-frame strip.
# <runN> is a directory under benchmarks/pro_30fps_20260922 (or 'REF' for the s30-reuse reference).
cd /workspace/SoulX-FlashHead || exit 1
OUT=benchmarks/pro_30fps_20260922/visual; mkdir -p "$OUT"
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
STEM="$1"; shift
L=(); R=()
while [ $# -gt 0 ]; do L+=("$1"); R+=("$2"); shift 2; done
path() { [ "$1" = REF ] && echo benchmarks/pro_30fps_20260921/s30-reuse/video.mp4 || echo benchmarks/pro_30fps_20260922/$1/video.mp4; }
COL=(white 0xFFD24A 0x7FD7FF 0xB6F08A)
X=86; Y=193
M=""; FF=""
for i in 0 1 2 3; do
  M="$M[$i:v]crop=160:96:$X:$Y,scale=640:384:flags=lanczos,pad=iw:ih+46:0:46:black,drawtext=fontfile=$F:text='${L[$i]}':x=10:y=13:fontsize=20:fontcolor=${COL[$i]}[m$i];"
  FF="$FF[$i:v]pad=iw:ih+40:0:40:black,drawtext=fontfile=$F:text='${L[$i]}':x=6:y=10:fontsize=13:fontcolor=${COL[$i]}[f$i];"
done
ffmpeg -y -loglevel error -i "$(path ${R[0]})" -i "$(path ${R[1]})" -i "$(path ${R[2]})" -i "$(path ${R[3]})" -filter_complex \
  "${M}[m0][m1]hstack=inputs=2[top];[m2][m3]hstack=inputs=2[bot];[top][bot]vstack=inputs=2[v]" \
  -map "[v]" -map 1:a? -c:v libx264 -crf 16 -preset slow -c:a aac -shortest "$OUT/$STEM-mouth-grid.mp4"
ffmpeg -y -loglevel error -i "$(path ${R[0]})" -i "$(path ${R[1]})" -i "$(path ${R[2]})" -i "$(path ${R[3]})" -filter_complex \
  "${FF}[f0][f1][f2][f3]hstack=inputs=4[v]" \
  -map "[v]" -map 1:a? -c:v libx264 -crf 17 -preset slow -c:a aac -shortest "$OUT/$STEM-full-4up.mp4"
ls -la "$OUT"/$STEM*.mp4 | awk '{printf "%.1fM  %s\n", $5/1048576, $9}'

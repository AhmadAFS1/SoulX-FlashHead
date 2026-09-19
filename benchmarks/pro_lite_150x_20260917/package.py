"""Build the labeled A/B video and validate both generated media files."""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50.mp4"
    command = [
        "ffmpeg", "-y", "-i", str(ROOT / "lite/video.mp4"), "-i", str(ROOT / "pro/video.mp4"),
        "-filter_complex",
        "[0:v]drawtext=text='LITE':x=12:y=12:fontsize=28:fontcolor=white:borderw=2:bordercolor=black[l];"
        "[1:v]drawtext=text='PRO':x=12:y=12:fontsize=28:fontcolor=white:borderw=2:bordercolor=black[p];"
        "[l][p]hstack=inputs=2[v]",
        "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-t", "10.0", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    probes = {}
    for variant in ("lite", "pro"):
        probes[variant] = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,duration,nb_read_frames",
            "-of", "json", str(ROOT / variant / "video.mp4")], text=True))
        video = next(s for s in probes[variant]["streams"] if s["codec_type"] == "video")
        assert int(video["nb_read_frames"]) == 250
        assert (int(video["width"]), int(video["height"])) == (320, 576)
    comparison_probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate,duration,nb_read_frames",
        "-of", "json", str(output)], text=True))
    comparison_video = next(s for s in comparison_probe["streams"] if s["codec_type"] == "video")
    assert int(comparison_video["nb_read_frames"]) == 250
    assert (int(comparison_video["width"]), int(comparison_video["height"])) == (640, 576)
    (ROOT / "media-validation.json").write_text(json.dumps({
        "execution": "CPU FFmpeg packaging and ffprobe validation; no GPU inference.",
        "comparison": {"path": str(output), "probe": comparison_probe}, "inputs": probes,
    }, indent=2) + "\n")

    times = (0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5)
    sheet = Image.new("RGB", (len(times) * 160, 2 * 313), "white")
    draw = ImageDraw.Draw(sheet)
    for row, variant in enumerate(("lite", "pro")):
        for column, timestamp in enumerate(times):
            frame = Image.open(ROOT / variant / f"frame-{timestamp:.1f}s.png").convert("RGB")
            sheet.paste(frame.resize((160, 288), Image.Resampling.LANCZOS),
                        (column * 160, row * 313 + 25))
            draw.text((column * 160 + 4, row * 313 + 5),
                      f"{variant.upper()} / {timestamp:.1f}s", fill="black")
    sheet.save(ROOT / "frames-comparison.png")


if __name__ == "__main__":
    main()

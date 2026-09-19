"""Package and validate the fixed-output source-detail experiment."""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DETAILS = [307, 256, 128, 64]
TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]


def main():
    sheet = Image.new("RGB", (7 * 200, 4 * 225), "white")
    draw = ImageDraw.Draw(sheet)
    rows = []
    for row, detail in enumerate(DETAILS):
        folder = ROOT / f"detail-{detail}"
        result = json.loads((folder / "results.json").read_text())
        assert result["status"] == "complete", result
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(folder / "video.mp4")], text=True))
        video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        assert int(video["nb_read_frames"]) == 250
        rows.append({"detail": detail, "generation": result, "decoded_video": video})
        for column, timestamp in enumerate(TIMES):
            frame = Image.open(folder / f"frame-{timestamp:.1f}s.png")
            sheet.paste(frame.resize((200, 200), Image.Resampling.LANCZOS),
                        (column * 200, row * 225 + 25))
            draw.text((column * 200 + 5, row * 225 + 5),
                      f"source {detail}px / {timestamp:.1f}s", fill="black")
    sheet.save(ROOT / "frames-comparison.png")
    (ROOT / "summary.json").write_text(json.dumps({
        "execution": "Fresh GPU generation on RTX 4070 SUPER 12282 MiB, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8, 2026-09-17; CPU packaging and decode validation.",
        "rows": rows}, indent=2) + "\n")
    for row in rows:
        generation = row["generation"]
        print(row["detail"], {key: generation[key] for key in
              ("wall_s", "useful_fps", "peak_allocated_mib", "peak_reserved_mib")})


if __name__ == "__main__":
    main()

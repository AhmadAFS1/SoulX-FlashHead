"""Compare a completed v2 PRO run with native MuseTalk rendering on one GPU.

MuseTalk renders a prepared face crop while PRO generates a full frame. This
tool reports matched delivery dimensions/timing boundaries but preserves that
architectural difference in every result manifest.
"""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_FIXTURES,
    DEFAULT_GPU_LOCK,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    failure_record,
    relative_path,
    resolve_fixture,
    sha256,
    summarize,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease


def in_musetalk_directory(function):
    """MuseTalk's native model/preparation imports resolve local weights by cwd."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = Path.cwd()
        try:
            os.chdir('/workspace/MuseTalk')
            return function(*args, **kwargs)
        finally:
            os.chdir(previous)
    return wrapped


class MuseTalkAdapter:
    """Native MuseTalk load/prepare/reset/generate lifecycle for timing studies."""

    def __init__(self, output: Path, batch: int) -> None:
        self.output = output.resolve()
        self.batch = batch
        self.models = None
        self.avatar = None
        self.audio = None

    @in_musetalk_directory
    def load(self) -> dict[str, Any]:
        muse = Path("/workspace/MuseTalk")
        if not muse.is_dir():
            raise FileNotFoundError("MuseTalk is unavailable at /workspace/MuseTalk")
        sys.path[:0] = [str(muse), str(muse / "scripts"), str(muse / "musetalk/utils")]
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.utils import load_all_model
        from transformers import WhisperModel

        started = time.perf_counter()
        vae, unet, pe = load_all_model(
            unet_config=str(muse / "models/musetalkV15/musetalk.json"), device="cuda"
        )
        vae.vae = vae.vae.half().eval()
        vae.runtime_dtype = torch.float16
        unet.model = unet.model.half().eval()
        pe = pe.half().cuda().eval()
        whisper = WhisperModel.from_pretrained(str(muse / "models/whisper")).half().cuda().eval()
        self.models = {"vae": vae, "unet": unet, "pe": pe, "whisper": whisper, "processor": AudioProcessor(str(muse / "models/whisper"))}
        return {"seconds": time.perf_counter() - started, "model": "MuseTalk V1.5", "precision": "float16"}

    @in_musetalk_directory
    @torch.no_grad()
    def prepare_avatar(self, video: Path, frames: int) -> dict[str, Any]:
        if self.models is None:
            raise RuntimeError("load must be called before prepare_avatar")
        from musetalk.utils.blending import (
            get_image_blending,
            get_image_prepare_material,
        )
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.preprocessing import (
            coord_placeholder,
            get_landmark_and_bbox,
        )

        started = time.perf_counter()
        folder = self.output / "musetalk-avatar-frames"
        folder.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), str(folder / "%08d.png")], check=True)
        paths = sorted(folder.glob("*.png"))
        if len(paths) < frames:
            raise ValueError(f"MuseTalk avatar video supplies {len(paths)} frames, expected at least {frames}")
        paths = paths[:frames]
        boxes, source_frames = get_landmark_and_bbox([str(path) for path in paths], 0)
        if len(boxes) != frames or any(tuple(box) == coord_placeholder for box in boxes):
            raise ValueError("MuseTalk avatar preparation has missing landmark boxes")
        parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        latents, masks, crop_boxes, updated = [], [], [], []
        for frame, box in zip(source_frames, boxes):
            x1, y1, x2, y2 = map(int, box)
            y2 = min(y2 + 10, frame.shape[0])
            updated_box = [x1, y1, x2, y2]
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.models["vae"].get_latents_for_unet(crop))
            mask, crop_box = get_image_prepare_material(frame, updated_box, fp=parser, mode="jaw")
            if mask is None or not np.count_nonzero(mask):
                raise ValueError("MuseTalk avatar preparation produced an invalid blend mask")
            masks.append(mask)
            crop_boxes.append(crop_box)
            updated.append(updated_box)
        self.avatar = {
            "frames": source_frames, "boxes": updated, "masks": masks, "crop_boxes": crop_boxes, "latents": latents,
            "blend": get_image_blending,
        }
        return {
            "seconds": time.perf_counter() - started,
            "source": relative_path(video),
            "source_sha256": sha256(video),
            "frames": frames,
            "face_detections": len(updated),
            "preparation": "native DWPose/S3FD landmark boxes and BiSeNet masks cached per avatar frame",
        }

    @torch.no_grad()
    def prepare_audio(self, audio: Path, frames: int, fps: int) -> dict[str, Any]:
        if self.models is None:
            raise RuntimeError("load must be called before prepare_audio")
        started = time.perf_counter()
        features, length = self.models["processor"].get_audio_feature(str(audio))
        chunks = self.models["processor"].get_whisper_chunk(
            features, "cuda", torch.float16, self.models["whisper"], length,
            fps=fps, audio_padding_length_left=2, audio_padding_length_right=2,
        )
        if len(chunks) < frames:
            raise ValueError(f"MuseTalk audio prepared only {len(chunks)} chunks for {frames} frames")
        self.audio = chunks[:frames]
        return {"seconds": time.perf_counter() - started, "audio_sha256": sha256(audio), "chunks": len(self.audio)}

    def reset_session(self) -> None:
        if self.avatar is None or self.audio is None:
            raise RuntimeError("avatar and audio must be prepared before reset_session")
        torch.cuda.synchronize()

    @torch.no_grad()
    def generate_to_rgb(self) -> tuple[list[np.ndarray], dict[str, Any]]:
        if self.models is None or self.avatar is None or self.audio is None:
            raise RuntimeError("adapter is not prepared")
        from musetalk.utils.utils import datagen

        started = time.perf_counter()
        recon = []
        unet_seconds = 0.0
        vae_seconds = 0.0
        for whisper_batch, latent_batch in datagen(self.audio, self.avatar["latents"], batch_size=self.batch, device="cuda"):
            begin = time.perf_counter()
            predicted = self.models["unet"].model(
                latent_batch.half(), torch.tensor([0], device="cuda"),
                encoder_hidden_states=self.models["pe"](whisper_batch.to(device="cuda", dtype=torch.float16)),
            ).sample
            torch.cuda.synchronize()
            unet_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            recon.extend(self.models["vae"].decode_latents(predicted))
            torch.cuda.synchronize()
            vae_seconds += time.perf_counter() - begin
        blended = []
        for frame, face, box, mask, crop_box in zip(
            self.avatar["frames"], recon, self.avatar["boxes"], self.avatar["masks"], self.avatar["crop_boxes"]
        ):
            x1, y1, x2, y2 = box
            face = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
            bgr = self.avatar["blend"](frame, face, box, mask, crop_box)
            blended.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        torch.cuda.synchronize()
        return blended, {
            "render_to_rgb_s": time.perf_counter() - started,
            "unet_s": unet_seconds,
            "vae_decode_s": vae_seconds,
            "frames": len(blended),
        }

    def close(self) -> None:
        self.audio = None
        self.avatar = None
        self.models = None
        torch.cuda.empty_cache()


def _pro_metrics(run: dict[str, Any]) -> dict[str, Any]:
    times = [entry["generation_s"] for entry in run.get("runs", [])]
    if not times:
        raise ValueError("PRO run has no generation measurements")
    return {
        "run": run.get("policy", {}).get("name"),
        "warm_rendering_s": times,
        "warm_rendering_summary_s": summarize(times),
        "warm_rendering_fps": summarize([run["profile"]["frames"] / value for value in times]),
        "application_delivery_first_repeat_s": times[0] + run.get("encode_mux_s", 0.0),
        "application_delivery_note": "PRO encode/mux is measured once after the retained first generation repeat; it is not duplicated across timing repeats.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro-run", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--musetalk-avatar-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--skip-delivery-encode", action="store_true")
    args = parser.parse_args()
    args.musetalk_avatar_video = args.musetalk_avatar_video.resolve()
    if args.frames < 1 or args.batch < 1 or args.repeats < 1:
        parser.error("frames and batch must be positive")
    output = ensure_new_directory(args.output)
    pro_run = json.loads((args.pro_run / "results.json").read_text(encoding="utf-8"))
    fixture = resolve_fixture(args.fixtures, args.fixture_id)
    if pro_run.get("status") != "complete" or pro_run["profile"]["frames"] != args.frames:
        raise ValueError("PRO run must be complete and have the requested frame count")
    if pro_run["profile"]["fps"] != fixture["fps"]:
        raise ValueError("PRO and MuseTalk delivery FPS must match")
    if pro_run["fixture"]["audio"]["sha256"] != fixture["audio"]["sha256"]:
        raise ValueError("PRO run and requested MuseTalk fixture use different audio")
    result: dict[str, Any] = {
        "status": "starting",
        "execution": "fresh GPU MuseTalk rendering compared with retained fresh PRO run",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_manifest(),
        "pro": _pro_metrics(pro_run),
        "fixture": fixture,
        "architectural_difference": "MuseTalk synthesizes a prepared face crop and blends it into an avatar video; PRO generates a full frame. Equal delivery resolution is not equal neural work.",
    }

    def save() -> None:
        atomic_write_json(output / "results.json", result)

    current_devices = result["environment"]["gpu"]["devices"]
    pro_devices = pro_run["environment"]["gpu"]["devices"]
    if [d["uuid"] for d in current_devices] != [d["uuid"] for d in pro_devices]:
        raise ValueError("MuseTalk and PRO comparison requires the same physical GPU")
    save()
    lease = None
    adapter = None
    stage = "load"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MuseTalk comparison")
        lease = acquire_gpu_lease(args.gpu_lock)
        seed = pro_run['profile']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        result['musetalk_seed'] = seed
        audio, _ = librosa.load(fixture['audio']['path'], sr=fixture['sample_rate'], mono=True)
        count = math.ceil(args.frames * fixture['sample_rate'] / fixture['fps'])
        if len(audio) < count:
            audio = np.tile(audio, int(np.ceil(count / len(audio))))
        audio = np.asarray(audio[:count], dtype=np.float32)
        effective_hash = hashlib.sha256(audio.tobytes()).hexdigest()
        expected = pro_run.get('effective_audio', {})
        if expected.get('samples') != count or expected.get('sha256') != effective_hash:
            raise ValueError('MuseTalk effective audio samples differ from PRO')
        effective_path = (output / 'effective-audio.wav').resolve()
        sf.write(effective_path, audio, fixture['sample_rate'], subtype='FLOAT')
        result['effective_audio'] = {'samples': count, 'sha256': effective_hash,
                                     'preparation': 'Exact PRO float32 samples; trimming/repetition excluded from warm timing in both profiles'}
        adapter = MuseTalkAdapter(output, args.batch)
        result["musetalk_load"] = adapter.load()
        stage = "prepare_avatar"
        result["musetalk_avatar"] = adapter.prepare_avatar(args.musetalk_avatar_video, args.frames)
        stage = "prepare_audio"
        result["musetalk_audio"] = adapter.prepare_audio(effective_path, args.frames, fixture["fps"])
        stage = "warmup"
        adapter.reset_session()
        adapter.generate_to_rgb()  # Discard first inference, including lazy initialization.
        stage = "warm_rendering"
        result["musetalk_repeats"] = []
        for repeat in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            audio_preparation = adapter.prepare_audio(effective_path, args.frames, fixture["fps"])
            adapter.reset_session()
            frames, warm = adapter.generate_to_rgb()
            torch.cuda.synchronize()
            request_s = time.perf_counter() - started
            result["musetalk_repeats"].append({
                "repeat": repeat, "audio": audio_preparation, "render": warm,
                "request_s": request_s, "useful_fps": len(frames) / request_s,
            })
            if len(frames) != args.frames:
                raise RuntimeError("MuseTalk frame count mismatch")
        durations = [row["request_s"] for row in result["musetalk_repeats"]]
        result["musetalk_request_summary"] = {
            "median_s": float(np.median(durations)), "min_s": min(durations), "max_s": max(durations),
            "useful_fps": args.frames / float(np.median(durations)), "audio_processing_included": True,
        }
        expected_shape = (pro_run["profile"]["height"], pro_run["profile"]["width"])
        if any(tuple(frame.shape[:2]) != expected_shape for frame in frames):
            raise ValueError("MuseTalk and PRO delivery resolutions differ")
        result["comparison_qualified"] = False
        result["comparison_limitation"] = "Sequential retained-PRO comparison; paired alternating runs and visual qualification still required."

        if len(frames) != args.frames:
            raise RuntimeError(f"MuseTalk generated {len(frames)} frames, expected {args.frames}")
        result["musetalk_warm_rendering"] = warm | {"useful_fps": args.frames / warm["render_to_rgb_s"]}
        stage = "application_delivery"
        delivery_start = time.perf_counter()
        if not args.skip_delivery_encode:
            from soulx_rtc.experiment import record

            rgb = frames  # Color conversion is already inside timed RGB generation.
            record(output / "musetalk.mp4", [np.asarray(rgb)], audio, fixture["fps"])
        result["musetalk_application_delivery"] = {
            "seconds": time.perf_counter() - delivery_start + request_s,
            "audio_processing_included": True,
            "encoding_included": not args.skip_delivery_encode,
            "note": (
                "Encoded video/audio delivery after native RGB rendering."
                if not args.skip_delivery_encode else
                "Queue-ready RGB only; this diagnostic is not comparable to encoded delivery."
            ),
        }
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", failure=failure_record(stage, error))
        save()
        raise
    finally:
        if adapter is not None:
            adapter.close()
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()

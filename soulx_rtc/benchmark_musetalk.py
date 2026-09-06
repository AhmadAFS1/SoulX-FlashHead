"""Same-machine real-audio MuseTalk comparison, run with the MuseTalk venv.

Run from /workspace/MuseTalk. Creates only the named comparison avatar cache.
Includes Whisper, UNet, VAE, CPU transfer and actual avatar composition. Does
not count prebuffer/idle/repeated transport frames as generated work.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/workspace/MuseTalk")
import cv2
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import load_all_model
from scripts.api_avatar import APIAvatar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 10])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--output", default="/workspace/SoulX-FlashHead/benchmarks/musetalk-engine.json")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device, dtype = torch.device("cuda:0"), torch.float16
    vae, unet, pe = load_all_model(device=device)
    for module in (vae.vae, unet.model, pe):
        module.to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae.runtime_dtype = dtype
    vae.set_decode_backend(None)
    unet.model_dtype = dtype
    processor = AudioProcessor(feature_extractor_path="models/whisper")
    whisper = WhisperModel.from_pretrained("models/whisper", local_files_only=True)
    whisper.to(device=device, dtype=dtype).eval().requires_grad_(False)
    avatar_id = "soulx_comparison_4070"
    info = Path("results/v15/avatars") / avatar_id / "avator_info.json"
    prep = not info.exists()
    fp = None
    if prep:
        from musetalk.utils.face_parsing import FaceParsing
        fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    avatar = APIAvatar(avatar_id=avatar_id,
        video_path="/workspace/SoulX-FlashHead/benchmarks/comparison-avatar.mp4",
        bbox_shift=0, batch_size=args.batch, vae=vae, unet=unet, pe=pe, fp=fp,
        args=SimpleNamespace(version="v15", extra_margin=10, parsing_mode="jaw",
            left_cheek_width=90, right_cheek_width=90, audio_padding_length_left=2,
            audio_padding_length_right=2), preparation=prep)
    if args.compile:
        unet.model = torch.compile(unet.model)
        # Compile the decoder actually used, bypassing diffusers' Accelerate
        # wrapper (its decode decorator fails under this older torch build).
        vae.vae.decoder = torch.compile(vae.vae.decoder)
    timestep = torch.tensor([0], device=device)
    audio, sr = sf.read("/workspace/SoulX-FlashHead/benchmarks/comparison-10s.wav", dtype="float32")
    assert sr == 16000 and len(audio) == 160000

    @torch.no_grad()
    def generate(n):
        started = time.perf_counter()
        prompts = []
        for i in range(n):
            speech = np.roll(audio, i * 640)
            mel = processor.feature_extractor([speech], sampling_rate=16000,
                return_tensors="pt").input_features.to(dtype)
            raw = processor.get_whisper_chunk(list(mel.split(1)), device, dtype,
                whisper, len(speech), fps=25, audio_padding_length_left=2,
                audio_padding_length_right=2).cpu().contiguous()
            prompts.append(avatar.apply_positional_encoding_cpu(raw))
        audio_s = time.perf_counter() - started
        cursors, first, done = [0] * n, {}, {}
        next_session, frames = 0, 0
        while frames < n * 250:
            selected = []
            while len(selected) < args.batch and frames + len(selected) < n * 250:
                i = next_session % n
                next_session += 1
                if cursors[i] < 250:
                    selected.append((i, cursors[i]))
                    cursors[i] += 1
            features = torch.stack([prompts[i][f] for i,f in selected]).to(device=device, dtype=dtype)
            latent = torch.cat([avatar.input_latent_list_cycle[f % len(avatar.input_latent_list_cycle)]
                                for i,f in selected]).to(device=device, dtype=dtype)
            predicted = unet.model(latent, timestep, encoder_hidden_states=features).sample
            faces = vae.decode_latents(predicted)
            for (i,f), face in zip(selected, faces):
                frame = cv2.resize(avatar.compose_frame(face, f), (512, 512))
                frames += 1
                first.setdefault(i, time.perf_counter() - started)
                if f == 249:
                    done[i] = time.perf_counter() - started
            del predicted, faces
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return dict(sessions=n, batch=args.batch, compile=args.compile,
            generated_frames=frames, wall_s=elapsed, aggregate_fps=frames / elapsed,
            audio_preparation_s=audio_s, first_frame_s=first, completion_s=done,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)

    print("WARMUP", flush=True)
    generate(1)
    results = []
    for n in args.sessions:
        torch.cuda.reset_peak_memory_stats()
        row = generate(n)
        row.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                   audio="comparison-10s.wav", output_size=512, neural_face_size=256)
        results.append(row)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

"""Compare production audio features and PRO conditioning against source on real speech.

Uses the same eight-second history and 33-frame alignment as SoulX Studio. Separate
projection calls with source features distinguish encoder errors from projection
errors, including a backend that returns the same tensor for different speech.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import wave

import coremltools as ct
import numpy as np
import torch
from transformers import Wav2Vec2Model

from .dit import AudioProjection, load_component
from .export_support import Audio, ROOT
from .validate_support import digest_tree


def summary(array):
    values = array.astype(np.float64)
    finite = np.isfinite(values)
    measured = values[finite]
    return {
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "dtype": str(array.dtype), "shape": list(array.shape),
        "finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
        "min": float(measured.min()) if measured.size else None,
        "max": float(measured.max()) if measured.size else None,
        "rms": float(np.sqrt(np.mean(measured * measured))) if measured.size else None,
    }


def compared(actual, reference):
    result = summary(actual)
    if result["finite"]:
        difference = actual.astype(np.float64) - reference.astype(np.float64)
        error = float(np.sqrt(np.mean(difference * difference)))
        rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
        result["normalized_rmse_to_torch"] = error / max(rms, 1e-12)
        result["max_abs_error_to_torch"] = float(np.abs(difference).max())
    return result


def passes_component_gate(metrics):
    return (metrics["finite"] and metrics.get("normalized_rmse_to_torch", float("inf")) <= 0.01
            and metrics.get("max_abs_error_to_torch", float("inf")) <= 0.05)


def classify_run(result):
    cases = result["cases"].values()
    result["encoder_gate_passed_all_cases"] = all(passes_component_gate(row["features"]) for row in cases)
    result["projection_gate_passed_with_source_features"] = all(
        passes_component_gate(row["context_from_reference_features"]) for row in cases)
    result["end_to_end_context_gate_passed_all_cases"] = all(
        passes_component_gate(row["end_to_end_context"]) for row in cases)
    result["distinct_feature_outputs"] = len({row["features"]["sha256"] for row in cases})
    result["distinct_projected_contexts"] = len({row["end_to_end_context"]["sha256"] for row in cases})


def provenance(models):
    checkpoint = ROOT / "models/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors"
    return {
        "wav2vec_file_sha256": digest_tree(ROOT / "models/wav2vec2-base-960h/model.safetensors"),
        "dit_file_sha256": digest_tree(checkpoint),
        "compiled_audio_tree_sha256": digest_tree(models / "audio.mlmodelc"),
        "compiled_conditioning_tree_sha256": digest_tree(models / "conditioning.mlmodelc"),
    }


def make_histories(path):
    with wave.open(str(path), "rb") as file:
        if (file.getframerate(), file.getnchannels(), file.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Supply the original mono 16 kHz PCM16 WAV file")
        data = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(np.float32) / 32768
    intervals = {
        "silence": (0, 0), "active_window_1": (3, 1.12), "active_window_2": (3, 2.24),
        "held_out_speech": (11.36, 2.24), "full_history": (7.68, 8),
    }
    histories = {}
    for name, (start, duration) in intervals.items():
        begin = round(start * 16000)
        count = round(duration * 16000)
        samples = data[begin:begin + count]
        if len(samples) != count:
            raise ValueError(f"Recording too short for {name}")
        histories[name] = np.pad(samples, (128000 - count, 0))[None]
    return histories, intervals


def packed(features):
    centers = np.arange(167, 200)[:, None] + np.arange(-2, 3)[None, :]
    return np.ascontiguousarray(features[:, centers.clip(0, 199)])


def run(models, wav, selected_units, report_path):
    torch.set_num_threads(4)
    graphics = json.loads(subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"], text=True))
    output = {"environment": {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_model_present": [gpu.get("sppci_model", "unverified") for gpu in graphics["SPDisplaysDataType"]],
        "physical_system_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
        "physical_vram": "Apple unified memory shared between CPU and GPU",
        "visible_vram": "No dedicated VRAM. Visible GPU memory budget was not queried",
        "evidence": "system_profiler SPDisplaysDataType and sysctl hw.memsize",
        "driver_runtime": platform.platform(), "torch": torch.__version__, "coremltools": ct.__version__,
        "co_resident_load": "Development jobs may be present. This is a correctness diagnostic.",
    }, "workload": "Original PRO Wav2Vec2 encoder plus original PRO audio projection, eight-second histories, 33 overlapping audio windows each.",
    "source_wav": str(wav), "source_wav_sha256": digest_tree(wav), "source_weights": provenance(models),
    "component_gates": {"normalized_rmse_max": 0.01, "max_abs_error_max": 0.05, "require_finite": True},
    "scope": "Mac component diagnostic. Torch runs on CPU. Actual operation placement inside each requested Core ML policy is unverified. iPhone behavior requires a device test.",
    "inputs": {}, "torch_cpu_reference": {}, "coreml_runs": {}}
    histories, intervals = make_histories(wav)
    encoder = Audio(Wav2Vec2Model.from_pretrained(str(ROOT / "models/wav2vec2-base-960h"), attn_implementation="eager")).eval()
    projector = load_component(AudioProjection(), ROOT / "models/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors", "conditioning")
    refs = {}
    with torch.inference_mode():
        for name, history in histories.items():
            features = encoder(torch.from_numpy(history)).numpy()
            context = projector(torch.from_numpy(packed(features))).numpy()
            refs[name] = (features, context)
            output["inputs"][name] = {"source_interval_start_seconds": intervals[name][0],
                                      "source_interval_duration_seconds": intervals[name][1],
                                      **summary(history)}
            output["torch_cpu_reference"][name] = {"features": summary(features), "context": summary(context)}
    del encoder, projector
    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    save()
    for label in selected_units:
        devices = units[label]
        audio = ct.models.CompiledMLModel(str(models / "audio.mlmodelc"), compute_units=devices)
        conditioning = ct.models.CompiledMLModel(str(models / "conditioning.mlmodelc"), compute_units=devices)
        run_metrics = {"requested_compute_units": devices.name, "actual_operation_placement": "unverified", "cases": {}}
        output["coreml_runs"][label] = run_metrics
        for name, history in histories.items():
            reference_features, reference_context = refs[name]
            features = np.asarray(audio.predict({"waveform": history})["features"])
            context = np.asarray(conditioning.predict({"audio": packed(features).astype(np.float16)})["context"])
            isolated = np.asarray(conditioning.predict({"audio": packed(reference_features).astype(np.float16)})["context"])
            run_metrics["cases"][name] = {
                "features": compared(features, reference_features),
                "end_to_end_context": compared(context, reference_context),
                "context_from_reference_features": compared(isolated, reference_context)}
            print(label, name, "features", run_metrics["cases"][name]["features"]["sha256"][:12],
                  "context", run_metrics["cases"][name]["end_to_end_context"]["sha256"][:12],
                  "isolated", run_metrics["cases"][name]["context_from_reference_features"]["sha256"][:12], flush=True)
        classify_run(run_metrics)
        save()
        del conditioning, audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--units", nargs="+", choices=("cpu", "cpu_gpu", "all"), default=["cpu", "cpu_gpu", "all"])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.models.resolve(), args.audio.resolve(), args.units, args.report)

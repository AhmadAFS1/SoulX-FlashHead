"""Compare v2 decoder candidates on identical captured public VAE latents."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    failure_record,
    relative_path,
    sha256,
    summarize,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_quantization import optimize_wan_vae
from soulx_rtc.pro_quantization_v2 import BF16_SCHEME, load_policy
from soulx_rtc.pro_vae_quantization import (
    BF16ConvolutionBackend,
    DecoderPlan,
    DecoderTarget,
    install_decoder_adapters,
    remove_decoder_adapters,
)


def _captured_latents(manifest: Path) -> list[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError("Decoder trial requires a complete capture manifest")
    paths = []
    for item in data.get("raw_tensors", []):
        if item.get("module") == "vae.public_decode" and item.get("kind") == "public_vae_latent":
            path = manifest.parent / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Captured latent hash mismatch: {path}")
            paths.append(path)
    if not paths:
        raise ValueError("Capture has no public VAE latent tensors")
    return paths


def _decoder_adapter_control(vae) -> dict[str, Any]:
    from flash_head.wan.modules.vae import CausalConv3d

    model = getattr(vae, "model", vae)
    prefix = "model.decoder" if model is not vae else "decoder"
    targets = []
    backends = {}
    for name, module in model.decoder.named_modules():
        if isinstance(module, CausalConv3d):
            path = f"{prefix}.{name}"
            targets.append(DecoderTarget(path, "causal_conv3d", ((1, 1, 1, 1, 1),), {}))
            backends[path] = BF16ConvolutionBackend()
    if not targets:
        raise RuntimeError("Wan decoder has no CausalConv3d modules")
    return install_decoder_adapters(
        vae, DecoderPlan("all-causal-bf16-control", "bf16", tuple(targets), (), {}), backends
    )


def _load_vae():
    from flash_head.wan.modules import WanVAE

    vae = WanVAE(
        vae_path=str(ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"),
        dtype=torch.bfloat16,
        device="cuda",
        parallel=False,
    )
    vae.model.eval().requires_grad_(False)
    return vae


def _measure(vae, latent: torch.Tensor, repeats: int) -> tuple[torch.Tensor, list[float]]:
    with torch.inference_mode():
        vae.decode(latent)
        torch.cuda.synchronize()
        values = []
        output = None
        for _ in range(repeats):
            start = time.perf_counter()
            output = vae.decode(latent)
            torch.cuda.synchronize()
            values.append(time.perf_counter() - start)
    return output, values


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = candidate.float() - reference.float()
    return {
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype).removeprefix("torch."),
        "finite": bool(candidate.isfinite().all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(delta.norm() / reference.float().norm().clamp_min(1e-12)),
    }


def _configure_candidate(vae, policy: dict[str, Any], adapter_control: bool) -> dict[str, Any]:
    decoder = policy["decoder"]
    if decoder.get("backend") in ("trt_stage", "trt_stage_compile"):
        if adapter_control:
            raise ValueError("Stage trial cannot also request a convolution adapter control")
        from benchmarks.pro_quantization_v2_20260918.decoder_install import install_quantized_decoder
        return install_quantized_decoder(vae, decoder)
    if decoder["scheme"] != BF16_SCHEME:
        if adapter_control:
            raise ValueError("Adapter control requires a BF16 policy")
        from benchmarks.pro_quantization_v2_20260918.decoder_install import install_quantized_decoder
        return install_quantized_decoder(vae, decoder)
    result = {"policy_decoder": decoder}
    if adapter_control:
        result["adapter_control"] = _decoder_adapter_control(vae)
        return result
    if decoder["backend"] == "torch_compile":
        result["compiled"] = optimize_wan_vae(vae, "compiled")
    elif decoder["backend"] != "pytorch":
        raise ValueError(f"Unsupported BF16 decoder backend for trial: {decoder['backend']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-policy", type=Path, required=True)
    parser.add_argument("--candidate-policy", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-latents", type=int, default=2)
    parser.add_argument("--candidate-adapter-control", action="store_true")
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_latents < 1:
        parser.error("repeats and max-latents must be positive")
    output = ensure_new_directory(args.output)
    result: dict[str, Any] = {
        "status": "starting",
        "execution": "fresh GPU fixed-latent decoder diagnostic; not recurrent video generation",
        "environment": environment_manifest(),
        "reference_policy": relative_path(args.reference_policy),
        "candidate_policy": relative_path(args.candidate_policy),
        "captures": relative_path(args.captures),
        "rows": [],
    }

    def save() -> None:
        atomic_write_json(output / "results.json", result)

    save()
    lease = None
    vae = None
    stage = "validate"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for a decoder trial")
        reference_policy = load_policy(args.reference_policy)
        candidate_policy = load_policy(args.candidate_policy)
        if candidate_policy['decoder']['backend'] == 'trt_stage_compile':
            # Stage boundaries expose more exact-shape Conv3d specializations
            # than the default limit of eight. Apply to every control in this
            # isolated benchmark process and record the effective setting.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        result['dynamo_recompile_limit'] = torch._dynamo.config.recompile_limit
        latents = _captured_latents(args.captures)[:args.max_latents]
        lease = acquire_gpu_lease(args.gpu_lock)
        for index, path in enumerate(latents):
            stage = f"original_bf16:{index}"
            latent = torch.load(path, map_location="cuda", weights_only=True).to(torch.bfloat16)
            vae = _load_vae()
            original, original_times = _measure(vae, latent, args.repeats)
            original_cpu = original.cpu()
            del vae
            vae = None
            torch.cuda.empty_cache()

            stage = f"adapter_bf16:{index}"
            vae = _load_vae()
            adapter_configuration = _decoder_adapter_control(vae)
            adapter, adapter_times = _measure(vae, latent, args.repeats)
            adapter_cpu = adapter.cpu()
            remove_decoder_adapters(vae)
            del vae
            vae = None
            torch.cuda.empty_cache()

            stage = f"compiled_bf16:{index}"
            vae = _load_vae()
            compiled_configuration = optimize_wan_vae(vae, "compiled")
            compiled, compiled_times = _measure(vae, latent, args.repeats)
            compiled_cpu = compiled.cpu()
            del vae
            vae = None
            torch.cuda.empty_cache()

            stage = f"candidate:{index}"
            vae = _load_vae()
            configuration = _configure_candidate(vae, candidate_policy, args.candidate_adapter_control)
            candidate, candidate_times = _measure(vae, latent, args.repeats)
            candidate_cpu = candidate.cpu()
            repeated, _ = _measure(vae, latent, 1)
            changed_latent = latent.clone()
            changed_latent.reshape(-1)[0] += 0.125
            _measure(vae, changed_latent, 1)
            restored, _ = _measure(vae, latent, 1)
            with torch.inference_mode():
                encoded_after_decode = vae.encode(candidate)
            row = {
                "latent": {"path": str(path), "sha256": sha256(path), "shape": list(latent.shape)},
                "original_bf16": {
                    "seconds": original_times, "summary_s": summarize(original_times),
                },
                "adapter_bf16": {
                    "configuration": adapter_configuration, "seconds": adapter_times,
                    "summary_s": summarize(adapter_times), "comparison": _metrics(adapter_cpu, original_cpu),
                },
                "compiled_bf16": {
                    "policy": reference_policy["name"], "configuration": compiled_configuration,
                    "seconds": compiled_times, "summary_s": summarize(compiled_times),
                    "comparison": _metrics(compiled_cpu, original_cpu),
                },
                "candidate": {
                    "policy": candidate_policy["name"], "configuration": configuration,
                    "seconds": candidate_times, "summary_s": summarize(candidate_times),
                },
                "comparison_to_original": _metrics(candidate_cpu, original_cpu),
                "comparison_to_compiled": _metrics(candidate_cpu, compiled_cpu),
                "repeat_same_latent_exact": bool(torch.equal(candidate, repeated)),
                "reset_after_different_latent_exact": bool(torch.equal(candidate, restored)),
                "repeat_comparison": _metrics(repeated, candidate),
                "reset_comparison": _metrics(restored, candidate),
                "encode_after_decode_finite": bool(encoded_after_decode.isfinite().all()),
            }
            row["candidate_over_compiled_median_speedup"] = (
                row["compiled_bf16"]["summary_s"]["p50"] / row["candidate"]["summary_s"]["p50"]
            )
            result["rows"].append(row)
            save()
            # Report WHICH condition failed and by how much. The gate is unchanged
            # in strictness -- every condition below is still fatal -- but a bare
            # "failed qualification" string cannot be acted on, and this gate has a
            # history of ambiguous verdicts: stage-int8-trial-r01 failed here while
            # cache-int8-r01 replayed the same plan and reported all_exact on all
            # eight checks, and stage-bf16-trial-r01 failed before r02 passed.
            # Whoever reads the next failure needs the discriminating numbers.
            failures = []
            if not row["comparison_to_original"]["finite"]:
                failures.append("output is not finite vs the BF16 original")
            if not row["repeat_same_latent_exact"]:
                failures.append(
                    "decoding the same latent twice is not bitwise equal "
                    f"(max_abs={row['repeat_comparison']['max_abs']}, "
                    f"relative_l2={row['repeat_comparison']['relative_l2']}) -- "
                    "a nondeterministic kernel/tactic, not an accuracy problem"
                )
            if not row["reset_after_different_latent_exact"]:
                failures.append(
                    "the causal feature cache does not reset between latents "
                    f"(max_abs={row['reset_comparison']['max_abs']}, "
                    f"relative_l2={row['reset_comparison']['relative_l2']})"
                )
            if not row["encode_after_decode_finite"]:
                failures.append("re-encoding the decoded frames produced non-finite motion latents")
            if row["adapter_bf16"]["comparison"]["max_abs"] != 0:
                failures.append(
                    "the BF16 adapter is not a no-op "
                    f"(max_abs={row['adapter_bf16']['comparison']['max_abs']})"
                )
            if failures:
                row["qualification_failures"] = failures
                save()
                raise RuntimeError(
                    "Decoder failed finite/cache-reset/BF16-adapter qualification: "
                    + "; ".join(failures)
                )
            if args.candidate_adapter_control:
                remove_decoder_adapters(vae)
            del (
                vae, latent, original, original_cpu, adapter, adapter_cpu, compiled, compiled_cpu,
                candidate, candidate_cpu, repeated, restored, changed_latent, encoded_after_decode,
            )
            vae = None
            torch.cuda.empty_cache()
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", failure=failure_record(stage, error))
        save()
        raise
    finally:
        if vae is not None and args.candidate_adapter_control:
            remove_decoder_adapters(vae)
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()

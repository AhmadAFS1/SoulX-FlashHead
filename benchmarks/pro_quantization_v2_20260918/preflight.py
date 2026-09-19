"""Record reproducibility metadata and validate fixed PRO v2 inputs without inference."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import json
from pathlib import Path

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_FIXTURES,
    ROOT,
    atomic_write_json,
    environment_manifest,
    file_identity,
    relative_path,
    resolve_fixture,
    resolve_path,
    sha256,
)
from soulx_rtc.pro_quantization_v2 import load_policy


def _checkpoint_manifest(checkpoint_root: Path) -> dict[str, dict[str, object]]:
    required = (
        "Model_Pro/diffusion_pytorch_model.safetensors",
        "Model_Pro/config.json",
        "VAE_Wan/Wan2.1_VAE.pth",
    )
    return {name: file_identity(checkpoint_root / name) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", default="indian150-a")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "models/SoulX-FlashHead-1_3B")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Preflight output already exists: {args.output}")

    fixture = resolve_fixture(args.fixtures, args.fixture_id)
    config_path = args.checkpoint_root / "Model_Pro/config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    result = {
        "status": "complete",
        "execution": "CPU/static preflight; no GPU inference",
        "environment": environment_manifest(),
        "fixture": fixture,
        "checkpoint_root": relative_path(args.checkpoint_root),
        "checkpoint": _checkpoint_manifest(args.checkpoint_root),
        "model_config": model_config,
        "source_sha256": {
            relative_path(path): sha256(path)
            for path in (
                Path(__file__),
                ROOT / "soulx_rtc/pro_quantization_v2.py",
                ROOT / "soulx_rtc/pro_attention_backends.py",
                ROOT / "soulx_rtc/pro_vae_quantization.py",
                ROOT / "flash_head/src/modules/flash_head_model.py",
                ROOT / "flash_head/wan/modules/vae.py",
            )
        },
    }
    if args.policy:
        policy = load_policy(resolve_path(args.policy))
        result["policy"] = policy
        result["policy_sha256"] = sha256(resolve_path(args.policy))
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "fixture": fixture["id"], "status": "complete"}))


if __name__ == "__main__":
    main()
"""Capture/calibrate Wan residual stages, then build independent BF16/INT8 engines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import _build_engine
from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    failure_record,
    relative_path,
    sha256,
    snapshot_sources,
    utc_now,
)
from benchmarks.pro_quantization_v2_20260918.decoder_trial import (
    _captured_latents,
    _load_vae,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_vae_stage_backend import (
    ResidualStage,
    install_stages,
    remove_stages,
    tensor_signature,
)


def int8_convolutions(layers):
    records = json.loads(layers).get("Layers", [])
    convs = [
        item
        for item in records
        if "convolution" in str(item.get("LayerType", "")).lower()
        or (
            item.get("LayerType") == "correlation"
            and "Conv" in item.get("Metadata", "")
        )
    ]
    quantized = [
        item
        for item in convs
        if item.get("Inputs")
        and all(
            "int8" in str(binding.get("Format/Datatype", "")).lower()
            for binding in item["Inputs"]
        )
        and (
            str(item.get("Weights", {}).get("Type", "")).lower() == "int8"
            or "i8i8" in item.get("TacticName", "").lower()
        )
    ]
    return {
        "convolutions": len(convs),
        "int8_convolutions": len(quantized),
        "quantized_names": [item.get("Name") for item in quantized],
    }


def parse_use_blocks(use_blocks, source_groups):
    """Resolve ``build --use-blocks`` against the calibration's groups.

    One argument per calibration group, in the calibration's group order, each a
    comma-separated PREFIX of that group's block indices. Only prefixes are buildable
    from an existing calibration: a group's samples are ``(x, cache_0 .. cache_{2n-1})``
    in block order, so the first ``k`` blocks' inputs are exactly ``(x, *caches[:2k])``,
    whereas dropping a block in the MIDDLE changes the input of every block after it
    and needs a fresh ``capture --skip-blocks``.

    An EMPTY entry (``""``) means "no engine for this group": its blocks are neither
    built nor skipped, they run eagerly in PyTorch at runtime. It is not a way to skip
    a whole group -- the first block of a group may change channel count (upsamples[4]
    is 192 -> 384), which ``install_decoder_block_skip`` refuses, and such a plan could
    never be installed. Skipping is always expressed by a shorter prefix.

    Returns ``(groups, skipped)``: the prefix groups to build and the dropped indices
    (sorted), which the plan records as ``skipped_blocks`` and the runtime must skip
    (``run.py --skip-decoder-blocks``) for ``install_stage_plan`` to accept the plan.

    Calibration staleness: dropping the tail of group i leaves the samples of every
    LATER group (and the head) captured with the dropped blocks still running, so
    those engines are built for slightly different inputs than they will see. That is
    only a shape-identical distribution shift for fp16/bf16 (the user-accepted quality
    risk, judged on video), but for INT8 it would bake stale amax scales into the
    engines, so ``build`` refuses that combination and asks for a
    ``capture --skip-blocks`` calibration instead.
    """
    if use_blocks is None:
        return [list(map(int, group)) for group in source_groups], []
    if len(use_blocks) != len(source_groups):
        raise ValueError(
            f"--use-blocks needs one entry per calibration group ({len(source_groups)}: "
            f"{source_groups}), got {len(use_blocks)}"
        )
    groups, skipped = [], []
    for spec, group in zip(use_blocks, source_groups):
        group = list(map(int, group))
        prefix = [int(v) for v in spec.split(",") if v.strip()]
        if prefix != group[: len(prefix)]:
            raise ValueError(f"--use-blocks {spec!r} is not a prefix of calibration group {group}")
        if not prefix:
            continue  # eager group: nothing built, nothing skipped
        skipped.extend(group[len(prefix):])
        groups.append(prefix)
    if not groups:
        raise ValueError("--use-blocks left every group eager; nothing to build")
    return groups, sorted(skipped)


def stale_int8_groups(groups, skipped):
    """Built groups whose calibration predates a block skipped UPSTREAM of them.

    ``groups`` are the prefixes being built, ``skipped`` the indices dropped by
    ``--use-blocks`` (not the calibration's own ``skipped_blocks``, whose samples were
    captured with those blocks already bypassed and are therefore exact). A built group
    that starts after a dropped block was calibrated on inputs the runtime will never
    produce; its recorded amax scales are stale. Returns those groups (empty when the
    plan is safe for INT8).
    """
    return [group for group in groups if any(index < group[0] for index in skipped)]


def _aligned_prefixes(source_groups, groups):
    """Per source group, its built prefix from ``groups`` (``[]`` when dropped)."""
    aligned, remaining = [], list(groups)
    for group in source_groups:
        if remaining and remaining[0][0] == int(group[0]):
            aligned.append(remaining.pop(0))
        else:
            aligned.append([])
    if remaining:
        raise ValueError(f"Groups {remaining} do not belong to any calibration group")
    return aligned


def prefix_sample(values, blocks_used, blocks_total):
    """Slice a calibration sample tuple down to a ``blocks_used``-block prefix stage.

    A sample is ``(x,)`` for the cold first call or ``(x, cache_0 .. cache_{2n-1})``
    afterwards, caches in block order, two per block. The prefix stage consumes the
    first ``2 * blocks_used`` of them and nothing else, so the sliced tuple IS the
    prefix stage's input -- no recomputation, and the recorded signature of the full
    tuple still identifies which sample it came from.
    """
    values = tuple(values)
    if len(values) not in (1, 1 + 2 * blocks_total):
        raise ValueError(
            f"Calibration sample has {len(values)} tensors; expected 1 or {1 + 2 * blocks_total}"
        )
    if not 0 < blocks_used <= blocks_total:
        raise ValueError(f"Prefix of {blocks_used} blocks out of {blocks_total}")
    if len(values) == 1:
        return values
    return values[: 1 + 2 * blocks_used]


def capture(args, output, result):
    paths = _captured_latents(args.captures)[: args.max_latents]
    if len(paths) < 2:
        raise ValueError(
            "Stage calibration needs at least two registered latent windows"
        )
    vae = _load_vae()
    groups = [list(map(int, value.split(","))) for value in args.groups]
    skip_blocks = sorted({int(v) for v in (getattr(args, "skip_blocks", None) or [])})
    overlap = sorted(set(skip_blocks) & {i for group in groups for i in group})
    if overlap:
        raise ValueError(f"--skip-blocks {overlap} overlap the capture groups {groups}")
    if skip_blocks:
        # General path for NON-prefix skip patterns: the blocks are bypassed during the
        # capture decode so every downstream group's samples (and the head) see the
        # inputs they will see at runtime. Installed before the stages, as run.py does.
        from soulx_rtc.pro_decoder_ops import install_decoder_block_skip

        result["decoder_ops"] = install_decoder_block_skip(vae, skip_blocks)
    result.update(
        groups=groups,
        skipped_blocks=skip_blocks,
        samples={},
        calibration={},
        source_latents=[],
        source_capture={
            "path": relative_path(args.captures),
            "sha256": sha256(args.captures),
        },
        retained_bytes=0,
        max_bytes=args.max_raw_mib * 2**20,
    )

    def factory(indices, stage):
        key = "-".join(map(str, indices))
        result["samples"][key] = {}
        result["calibration"][key] = {}

        def execute(*values):
            signature = tensor_signature(values)
            records = result["samples"][key]
            if signature not in records:
                count = sum(value.numel() * value.element_size() for value in values)
                if count + result["retained_bytes"] > result["max_bytes"]:
                    raise RuntimeError(
                        "Stage capture byte budget exceeded; signature coverage incomplete"
                    )
                path = output / f"stage-{key}-sample-{len(records)}.pt"
                torch.save(tuple(value.detach().cpu() for value in values), path)
                records[signature] = {
                    "path": path.name,
                    "sha256": sha256(path),
                    "count": 0,
                    "shapes": [list(value.shape) for value in values],
                }
                result["retained_bytes"] += count
            records[signature]["count"] += 1
            scales = result["calibration"][key].setdefault(signature, {})

            def observe(name, before, previous, after):
                maximum = float(before.abs().max())
                if previous is not None:
                    maximum = max(maximum, float(previous.abs().max()))
                out_max = float(after.abs().max())
                if not all(torch.isfinite(torch.tensor(v)) for v in (maximum, out_max)):
                    raise ValueError("Nonfinite stage calibration")
                item = scales.setdefault(
                    name, {"input_max": 0.0, "output_max": 0.0, "count": 0}
                )
                item["input_max"] = max(item["input_max"], maximum)
                item["output_max"] = max(item["output_max"], out_max)
                item["count"] += 1

            stage.observer = observe
            try:
                return stage(*values)
            finally:
                stage.observer = None

        return execute

    install_stages(vae, groups, factory)
    try:
        with torch.inference_mode():
            for path in paths:
                latent = torch.load(path, map_location="cuda", weights_only=True).to(
                    torch.bfloat16
                )
                result["source_latents"].append(
                    {"path": relative_path(path), "sha256": sha256(path)}
                )
                vae.decode(latent)
                atomic_write_json(output / "results.json", result)
        for samples in result["samples"].values():
            if any(item["count"] < 2 for item in samples.values()):
                raise RuntimeError(
                    "Fewer than two calibration samples for a stage signature"
                )
    finally:
        remove_stages(vae)
        if skip_blocks:
            from soulx_rtc.pro_decoder_ops import remove_decoder_block_skip

            remove_decoder_block_skip(vae)


def export_stage(stage, values, path, precision, scales, floating_precision="bf16", graph_rewrite="none"):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inputs = ["x"] + [f"cache_in_{i}" for i in range(len(values) - 1)]
    outputs = ["y"] + [f"cache_out_{i}" for i in range(stage.cache_count)]
    with torch.inference_mode():
        actual = stage(*values)
        torch.onnx.export(
            stage,
            values,
            str(path),
            input_names=inputs,
            output_names=outputs,
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
        )
    model = onnx.load(path)
    # TensorRT 10.9's BF16 Clip lowering supplies a nonfinite implicit upper
    # bound for this exported clamp_min. Max expresses the exact same lower
    # bound without synthesizing that unused upper bound.
    for node in model.graph.node:
        if (
            node.op_type == "Clip"
            and len(node.input) >= 2
            and node.input[1]
            and (len(node.input) < 3 or not node.input[2])
        ):
            node.op_type = "Max"
            del node.input[2:]
    converted = []
    if precision == "int8":
        nodes = []
        for node in model.graph.node:
            if node.op_type != "Conv" or ".residual." not in node.input[1]:
                nodes.append(node)
                continue
            weight_name = node.input[1]
            name = weight_name.removesuffix(".weight")
            if name not in scales or scales[name]["count"] < 2:
                raise ValueError(f"Incomplete calibration for {name}")
            prefix = name.replace(".", "_")
            weight = stage.get_parameter(weight_name).detach().float().cpu().numpy()
            bias = stage.get_parameter(name + ".bias").detach().float().cpu().numpy()
            ws = (
                np.maximum(np.abs(weight).max(axis=tuple(range(1, weight.ndim))), 1e-12)
                / 127.0
            )
            iq, oq = (
                max(scales[name]["input_max"] / 127.0, 1e-12),
                max(scales[name]["output_max"] / 127.0, 1e-12),
            )
            for suffix, array in [
                ("weight", weight),
                ("bias", bias),
                ("ws", ws.astype("float32")),
                ("wz", np.zeros(ws.shape, dtype="int8")),
                ("is", np.array(iq, dtype="float32")),
                ("os", np.array(oq, dtype="float32")),
                ("z", np.array(0, dtype="int8")),
            ]:
                model.graph.initializer.append(
                    numpy_helper.from_array(array, prefix + "_" + suffix)
                )
            original_output = node.output[0]
            nodes.extend(
                [
                    helper.make_node(
                        "Cast",
                        [node.input[0]],
                        [prefix + "_fp32"],
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "QuantizeLinear",
                        [prefix + "_fp32", prefix + "_is", prefix + "_z"],
                        [prefix + "_iq"],
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [prefix + "_iq", prefix + "_is", prefix + "_z"],
                        [prefix + "_idq"],
                    ),
                    helper.make_node(
                        "QuantizeLinear",
                        [prefix + "_weight", prefix + "_ws", prefix + "_wz"],
                        [prefix + "_wq"],
                        axis=0,
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [prefix + "_wq", prefix + "_ws", prefix + "_wz"],
                        [prefix + "_wdq"],
                        axis=0,
                    ),
                ]
            )
            node.input[:] = [prefix + "_idq", prefix + "_wdq", prefix + "_bias"]
            node.output[:] = [prefix + "_conv"]
            nodes.append(node)
            nodes.extend(
                [
                    helper.make_node(
                        "QuantizeLinear",
                        [prefix + "_conv", prefix + "_os", prefix + "_z"],
                        [prefix + "_oq"],
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [prefix + "_oq", prefix + "_os", prefix + "_z"],
                        [prefix + "_odq"],
                    ),
                    helper.make_node(
                        "Cast",
                        [prefix + "_odq"],
                        [original_output],
                        to=TensorProto.BFLOAT16,
                    ),
                ]
            )
            converted.append(name)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        if len(converted) != stage.cache_count:
            raise ValueError(
                "INT8 export did not cover every declared residual convolution"
            )
    elif precision not in ("bf16", "fp16"):
        raise ValueError("Unsupported stage precision")
    if precision == "fp16":
        floating_precision = "fp16"
    if floating_precision not in ("bf16", "fp16"):
        raise ValueError("Unsupported floating stage precision")
    if floating_precision == "fp16":
        # Keep original BF16 calibration/parameters untouched. Convert this
        # exported graph only; the runtime explicitly restores BF16 caches.
        def convert_tensor(tensor):
            if tensor.data_type == TensorProto.BFLOAT16:
                if tensor.raw_data:
                    array = (
                        np.frombuffer(tensor.raw_data, np.uint16).astype(np.uint32)
                        << 16
                    ).view(np.float32)
                else:
                    array = (np.asarray(tensor.int32_data, dtype=np.uint32) << 16).view(
                        np.float32
                    )
                array = array.reshape(tensor.dims).astype(np.float16)
                if not np.isfinite(array).all():
                    raise ValueError("Floating stage conversion overflow")
                tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))

        for tensor in model.graph.initializer:
            convert_tensor(tensor)
        for node in model.graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    convert_tensor(attribute.t)
                if (
                    node.op_type == "Cast"
                    and attribute.name == "to"
                    and attribute.i == TensorProto.BFLOAT16
                ):
                    attribute.i = TensorProto.FLOAT16
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        ):
            if value.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                value.type.tensor_type.elem_type = TensorProto.FLOAT16
    rewrite = {"mode": graph_rewrite}
    if graph_rewrite != "none":
        if precision == "int8":
            raise ValueError("Graph rewrites are only validated for the floating stage export")
        # After the fp16 conversion on purpose: the identity-cast removal relies on it, and
        # the norm rewrite's 1/64 pre-scale is sized for fp16 range.
        from soulx_rtc.pro_stage_onnx import clean

        rewrite.update(clean(model, norm_conv=graph_rewrite == "norm-conv"))
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {
        "graph_rewrite": rewrite,
        "inputs": [{"name": n, "shape": list(v.shape)} for n, v in zip(inputs, values)],
        "outputs": [
            {"name": n, "shape": list(v.shape)} for n, v in zip(outputs, actual)
        ],
        "quantized_modules": converted,
        "binding_dtype": "float16" if floating_precision == "fp16" else "bfloat16",
        "floating_precision": floating_precision,
        "caller_cache_dtype": "bfloat16",
    }


def build(args, output, result):
    import tensorrt as trt

    source = json.loads(args.calibration.read_text())
    if (
        source.get("status") != "complete"
        or source["weights_sha256"] != result["weights_sha256"]
    ):
        raise ValueError("Incomplete or incompatible stage calibration")
    vae = _load_vae()
    use_blocks = getattr(args, "use_blocks", None)
    groups, dropped = parse_use_blocks(use_blocks, source["groups"])
    stale = stale_int8_groups(groups, dropped)
    if stale and args.precision == "int8":
        raise ValueError(
            f"--use-blocks drops {dropped}, which precede built groups {stale}: their INT8 "
            "scales were calibrated with those blocks running. Re-run `capture --skip-blocks "
            f"{' '.join(map(str, dropped))}` and build from that calibration instead"
        )
    # A calibration captured with --skip-blocks already excludes those blocks from every
    # sample; the plan must carry them too, so the runtime check sees the full set.
    skipped_blocks = sorted(set(dropped) | {int(i) for i in source.get("skipped_blocks", [])})
    result.update(
        schema_version=1,
        groups=groups,
        skipped_blocks=skipped_blocks,
        calibration_groups=[list(map(int, group)) for group in source["groups"]],
        precision=args.precision,
        stages={},
        floating_precision="fp16"
        if args.precision == "fp16"
        else args.floating_precision,
        calibration={
            "path": relative_path(args.calibration),
            "sha256": sha256(args.calibration),
        },
        graph_rewrite=getattr(args, "graph_rewrite", "none"),
    )
    if use_blocks is not None:
        result["use_blocks"] = list(use_blocks)
        # Recorded so a reader of the plan knows which engines were built on samples
        # captured with a now-skipped block still running (fp16: accepted, see
        # parse_use_blocks).
        result["stale_calibration_groups"] = stale
    for source_group, indices in zip(source["groups"], _aligned_prefixes(source["groups"], groups)):
        if not indices:
            continue
        source_key = "-".join(map(str, source_group))
        key = "-".join(map(str, indices))
        stage = ResidualStage([vae.model.decoder.upsamples[i] for i in indices]).eval()
        if stage.cache_count != 2 * len(indices):
            raise ValueError("Stage cache count does not match the prefix length")
        result["stages"][key] = {}
        for i, (signature, sample) in enumerate(source["samples"][source_key].items()):
            path = args.calibration.parent / sample["path"]
            if sha256(path) != sample["sha256"]:
                raise ValueError("Stage sample hash mismatch")
            loaded = tuple(torch.load(path, map_location="cpu", weights_only=True))
            if tensor_signature(loaded) != signature:
                raise ValueError("Stage sample signature mismatch")
            if [list(v.shape) for v in loaded] != sample["shapes"]:
                raise ValueError("Stage sample shapes disagree with the calibration record")
            values = tuple(v.cuda() for v in prefix_sample(loaded, len(indices), len(source_group)))
            del loaded
            if len(values) not in (1, stage.cache_count + 1):
                raise ValueError("Sliced sample does not match the prefix stage's cache count")
            # Two keys live side by side from here on. The calibration dict is keyed by
            # the SOURCE group ("12-13-14") and the FULL sample's signature, so the scales
            # lookup must use those. The plan record is keyed by the PREFIX group ("12")
            # and the signature of the SLICED tuple, because that is what the runtime
            # presents to the stage. Mixing them up is a KeyError on the first prefix
            # group (after the full groups' engines were already built), so the full
            # signature is kept under its own name.
            full_signature = signature
            signature = tensor_signature(values)
            if len(values) == len(sample["shapes"]) and signature != full_signature:
                raise ValueError("Unsliced sample changed signature after loading")
            # For INT8 the source group's scales are keyed by conv name inside the stage
            # ("blocks.0.residual.2", ...). A prefix stage's blocks are the first k of the
            # source stage, so its names are a subset of the full group's dict, and the
            # amax values are exact for them (the prefix sees the very same inputs).
            scales = source["calibration"][source_key][full_signature]
            onnx_path = output / f"stage-{key}-{i}.onnx"
            metadata = export_stage(
                stage,
                values,
                onnx_path,
                args.precision,
                scales,
                result["floating_precision"],
                graph_rewrite=getattr(args, "graph_rewrite", "none"),
            )
            engine_path = onnx_path.with_suffix(".engine")
            begin = time.perf_counter()
            engine, layers = _build_engine(onnx_path, engine_path, args.workspace_mib)
            evidence = int8_convolutions(layers)
            layers_path = engine_path.with_suffix(".layers.json")
            layers_path.write_text(layers + "\n")
            if args.precision == "int8" and evidence["int8_convolutions"] < len(
                metadata["quantized_modules"]
            ):
                raise RuntimeError(
                    "Stage inspector does not prove all requested INT8 convolutions"
                )
            record = metadata | {
                "path": relative_path(engine_path),
                "sha256": sha256(engine_path),
                "precision": args.precision,
                "onnx_sha256": sha256(onnx_path),
                "layers_path": relative_path(layers_path),
                "layers_sha256": sha256(layers_path),
                "precision_evidence": evidence,
                "int8_convolutions_verified": args.precision == "int8",
                "tensorrt": trt.__version__,
                "gpu": torch.cuda.get_device_name(),
                "build_s": time.perf_counter() - begin,
                "workspace_bytes": engine.device_memory_size,
            }
            from benchmarks.pro_quantization_v2_20260918.decoder_trial import _metrics
            from soulx_rtc.pro_vae_stage_backend import TensorRTStage

            runtime = TensorRTStage(engine_path, record)
            with torch.inference_mode():
                expected = stage(*values)
                actual = runtime(*values)
                torch.cuda.synchronize()
                record["build_sample_comparison"] = [
                    _metrics(a, b) for a, b in zip(actual, expected)
                ]
            result["stages"][key][signature] = record
            atomic_write_json(output / "results.json", result)
            if not all(row["finite"] for row in record["build_sample_comparison"]):
                raise RuntimeError(
                    "Built stage produced nonfinite output/cache on its calibration input"
                )
            del runtime, expected, actual
            del engine, values
            torch.cuda.empty_cache()
            atomic_write_json(output / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--captures", type=Path, required=True)
    cap.add_argument("--groups", nargs="+", default=["4,5,6", "8,9,10"])
    cap.add_argument("--max-latents", type=int, default=4)
    cap.add_argument("--max-raw-mib", type=int, default=2048)
    cap.add_argument("--skip-blocks", type=int, nargs="*", default=None,
                     help="Decoder upsamples[] ResidualBlocks bypassed as identity during the capture decode "
                          "(soulx_rtc.pro_decoder_ops.install_decoder_block_skip); the general path for skip "
                          "patterns that are not a prefix of a group. Recorded as skipped_blocks.")
    bld = commands.add_parser("build")
    bld.add_argument("--calibration", type=Path, required=True)
    bld.add_argument("--precision", choices=["bf16", "fp16", "int8"], required=True)
    bld.add_argument("--floating-precision", choices=["bf16", "fp16"], default="bf16")
    bld.add_argument("--workspace-mib", type=int, default=512)
    bld.add_argument("--use-blocks", nargs="+", default=None, metavar="PREFIX",
                     help="One comma-separated PREFIX per calibration group, in the calibration's group order, "
                          "e.g. \"4,5,6\" \"8,9,10\" \"12\" builds the tail engine from upsamples[12] alone and "
                          "records 13,14 as skipped_blocks (run.py --skip-decoder-blocks 13 14). An empty entry "
                          "builds no engine for that group (its blocks run eagerly, none are skipped).")
    bld.add_argument("--graph-rewrite", choices=["none", "clean", "norm-conv"], default="none",
                     help="ONNX rewrites before the TensorRT build (soulx_rtc/pro_stage_onnx.py): "
                          "'clean' is exact, 'norm-conv' also runs the RMS norm as a 1x1x1 conv")
    for sub in (cap, bld):
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    output = ensure_new_directory(args.output)
    if (
        args.command == "build"
        and args.precision == "bf16"
        and args.floating_precision != "bf16"
    ):
        parser.error("Use --precision fp16 for an explicitly labeled FP16 control")
    result = {
        "status": "starting",
        "date_utc": utc_now(),
        "execution": "fresh GPU stage diagnostic; not end-to-end video throughput",
        "environment": environment_manifest(),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "weights_sha256": sha256(
            ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
        ),
    }
    result["source_sha256"] = snapshot_sources(
        output,
        [
            Path(__file__),
            ROOT / "soulx_rtc/pro_vae_stage_backend.py",
            ROOT / "soulx_rtc/pro_stage_onnx.py",
            ROOT / "soulx_rtc/pro_decoder_ops.py",
            ROOT / "flash_head/wan/modules/vae.py",
            ROOT / "benchmarks/pro_quantization_v2_20260918/build_decoder_engine.py",
        ],
    )
    atomic_write_json(output / "results.json", result)
    lease = None
    try:
        lease = acquire_gpu_lease(args.gpu_lock)
        (capture if args.command == "capture" else build)(args, output, result)
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = failure_record(args.command, error)
        raise
    finally:
        atomic_write_json(output / "results.json", result)
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()

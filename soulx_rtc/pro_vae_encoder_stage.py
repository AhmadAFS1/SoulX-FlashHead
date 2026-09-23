"""Opt-in FP16 TensorRT stages for the Wan VAE **encoder** (lever C4).

The per-window motion re-encode (``pipeline.vae.encode(cond_frame)``, ``cond_frame`` =
``(1, 3, 5, H, W)`` bf16, 5 pixel frames -> 2 latents) costs ~117 ms/window at 576x320 on
the current best build. Its cost is in the full-resolution convolutions, exactly like the
decoder's, so this module gives ``encoder.downsamples`` the treatment
``soulx_rtc/pro_vae_stage_backend.py`` gives ``decoder.upsamples``:

* ``EncoderSpanStage`` -- a contiguous slice of ``encoder.downsamples`` (``ResidualBlock`` /
  ``Resample`` downsample2d|downsample3d / ``AttentionBlock``), optionally preceded by
  ``encoder.conv1`` (group token ``"conv1"``, e.g. ``"conv1,0,1,2"``), with every causal
  cache the slice consumes as an explicit input/output tensor. Same runtime contract as the
  decoder stages: ``forward(x)`` on the encoder's first chunk (every slot ``None``),
  ``forward(x, *caches)`` afterwards, returning ``(y, *cache_updates)`` in the walk order
  of ``Encoder3d.forward``.
* ``EncoderStageAdapter`` -- installed into ``encoder.downsamples[first]`` with
  ``_SkipStage`` stand-ins for the other covered indices; the covered modules stay in the
  module tree as children of the stage, so ``count_conv3d(encoder)`` (what ``clear_cache``
  sizes ``_enc_feat_map`` with) and every cursor index stay identical to stock.
* ``install_encoder_plan(vae, path, vae_weights=None)`` -- loads a plan built by
  ``benchmarks/pro_30fps_20260922/encoder_stages.py`` behind the same sha/metadata gates
  as ``install_stage_plan`` and executes the engines through ``TensorRTStage`` (fenced
  engine stream, fp16 cache passthrough, output pools) unchanged.

Cache-slot protocol, read from ``flash_head/wan/modules/vae.py`` (not assumed):

* ``Encoder3d.forward`` runs ``conv1`` inline: ``cache_x = x[:, :, -2:].clone()`` (with the
  previous cache's last frame prepended when ``x`` has one frame), ``x = conv1(x,
  feat_cache[idx])``, ``feat_cache[idx] = cache_x``, cursor += 1. One slot.
* ``ResidualBlock.forward``: one slot per ``CausalConv3d`` in ``.residual`` (two); the
  width-changing shortcut (``downsamples[3]`` 96->192, ``[6]`` 192->384) is a 1x1x1
  ``CausalConv3d`` called WITHOUT a cache argument: no slot (but it IS counted by
  ``count_conv3d``, which is why ``_enc_feat_map`` has 26 entries for 24 consumed slots).
* ``Resample.forward`` in ``downsample3d`` mode: the spatial ``ZeroPad2d((0,1,0,1)) +
  Conv2d(3x3, stride 2)`` runs first (no cache), then ONE slot for ``time_conv``
  (``CausalConv3d(dim, dim, (3,1,1), stride (2,1,1), padding 0)``): on the FIRST call the
  slot is ``None`` -> it stores ``x.clone()`` (the spatially downsampled chunk, one frame
  in production), advances the cursor and does NOT run ``time_conv`` (the frame count is
  unchanged); afterwards it stores ``x[:, :, -1:].clone()`` and runs
  ``time_conv(cat([cache[:, :, -1:], x], 2))`` -- 4 + 1 frames -> 2, or 2 + 1 -> 1. There is
  no ``"Rep"`` sentinel on the encoder side (that is the upsample3d branch only), so the
  encoder protocol is tensor-only and needs no materialisation trick.
* ``downsample2d`` mode: no cache code path at all, zero slots. ``AttentionBlock``: zero.

Per motion encode (``WanVAE_.encode``: frame 0 alone, then one 4-frame chunk) every group
therefore sees exactly two signatures: the cold call ``(x,)`` and the warm call with
one-frame block caches. A ``frame_num``-frame reference encode (``prepare_params``) adds a
third, steady-state signature (two-frame block caches) from its third chunk on; the capture
CLI records it on request (``--reference-frames``) and ``install_encoder_plan`` can be told
to run unknown signatures eagerly instead of refusing them.

Composition: install BEFORE ``torch.compile(pipeline.vae.encode)`` and BEFORE
``pro_decoder_ops.install_overlap_skip`` (both wrap ``vae.encode``; this module only swaps
modules inside ``vae.model.encoder``, and the adapter/stub are ``torch.compiler.disable``d so
the compiled encode graph-breaks around them exactly as the compiled decode does around the
decoder's ``StageAdapter``). Nothing here touches the decoder, ``_feat_map`` or
``vae.encode`` itself, so it composes with the block-skip, sub-pixel and overlap-skip
rewrites. When nothing is installed the encoder is byte-for-byte the stock module tree.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from soulx_rtc.pro_vae_stage_backend import (
    StageAdapter,
    TensorRTStage,
    _cache_update,
    _SkipStage,
    tensor_signature,
)

# Group token naming ``Encoder3d.conv1`` as the FIRST member of a span.
CONV1 = "conv1"
VAE_SOURCE = "flash_head/wan/modules/vae.py"
STOCK_VAE_WEIGHTS = "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
PLAN_TARGET = "encoder"


# --------------------------------------------------------------------------------------
# group specs
# --------------------------------------------------------------------------------------


def parse_encoder_group(spec) -> list:
    """``"conv1,0,1,2"`` / ``"3,4,5"`` / a JSON list -> ``["conv1", 0, 1, 2]`` / ``[3, 4, 5]``.

    Integers are ``encoder.downsamples`` indices; the only non-numeric token is
    ``"conv1"`` and it must come first.
    """
    tokens = [t.strip() for t in spec.split(",")] if isinstance(spec, str) else list(spec)
    group = []
    for token in tokens:
        if isinstance(token, str):
            if token == CONV1:
                group.append(CONV1)
                continue
            if token == "":
                continue
        try:
            if isinstance(token, (bool, float)):
                raise TypeError("not an index")
            group.append(int(token))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid encoder stage group token {token!r}: expected an encoder "
                f"downsamples index or {CONV1!r}"
            ) from None
    if CONV1 in group[1:]:
        raise ValueError(f"{CONV1!r} may only be the first entry of an encoder stage group")
    if not group:
        raise ValueError("Empty encoder stage group")
    return group


def encoder_group_key(group) -> str:
    """Plan/calibration dictionary key: ``"conv1-0-1-2"``."""
    return "-".join(map(str, parse_encoder_group(group)))


def encoder_group_indices(group) -> list[int]:
    return [token for token in parse_encoder_group(group) if token != CONV1]


def encoder_group_has_conv1(group) -> bool:
    return parse_encoder_group(group)[0] == CONV1


# --------------------------------------------------------------------------------------
# module protocol
# --------------------------------------------------------------------------------------


def encoder_module_cache_slots(module) -> int:
    """Causal-cache slots ``Encoder3d.forward`` lets ``module`` consume per call (see the
    module docstring for the source of each number)."""
    from flash_head.wan.modules.vae import (
        AttentionBlock,
        CausalConv3d,
        Resample,
        ResidualBlock,
    )

    if isinstance(module, ResidualBlock):
        return sum(isinstance(layer, CausalConv3d) for layer in module.residual)
    if isinstance(module, Resample):
        if module.mode == "downsample3d":
            if not isinstance(getattr(module, "time_conv", None), CausalConv3d):
                raise TypeError("downsample3d Resample without a CausalConv3d time_conv")
            return 1
        if module.mode == "downsample2d":
            return 0
        raise TypeError(f"Resample mode {module.mode!r} is not an encoder downsample")
    if isinstance(module, AttentionBlock):
        return 0
    if isinstance(module, CausalConv3d):
        # encoder.conv1 (Encoder3d.forward handles its slot inline).
        return 1
    raise TypeError(
        f"{type(module).__name__} cannot be part of an encoder stage span (only stock "
        "ResidualBlock / downsample Resample / AttentionBlock modules and conv1)"
    )


def _resample_form(module) -> str:
    resample = module.resample
    if isinstance(resample, nn.Sequential):
        return "+".join(type(layer).__name__ for layer in resample)
    return type(resample).__name__


def describe_encoder_module(module) -> str:
    from flash_head.wan.modules.vae import CausalConv3d, Resample

    if isinstance(module, Resample):
        return f"Resample:{module.mode}:{_resample_form(module)}"
    if isinstance(module, CausalConv3d):
        return f"{CONV1}:{type(module).__name__}"
    return type(module).__name__


def describe_encoder_span(vae, group) -> list[str]:
    """One entry per module a group covers, in walk order, e.g.
    ``["conv1:CausalConv3d", "ResidualBlock", "ResidualBlock",
    "Resample:downsample2d:ZeroPad2d+Conv2d"]``. Recorded by capture/build as ``spans``
    and compared verbatim by ``install_encoder_plan``."""
    encoder = vae.model.encoder
    group = parse_encoder_group(group)
    described = []
    if encoder_group_has_conv1(group):
        described.append(describe_encoder_module(encoder.conv1))
    described.extend(
        describe_encoder_module(encoder.downsamples[index]) for index in encoder_group_indices(group)
    )
    return described


def encoder_span_cache_count(vae, group) -> int:
    encoder = vae.model.encoder
    group = parse_encoder_group(group)
    count = sum(
        encoder_module_cache_slots(encoder.downsamples[index]) for index in encoder_group_indices(group)
    )
    if encoder_group_has_conv1(group):
        count += encoder_module_cache_slots(encoder.conv1)
    return count


# --------------------------------------------------------------------------------------
# the eager stage
# --------------------------------------------------------------------------------------


class EncoderSpanStage(nn.Module):
    """A contiguous slice of ``encoder.downsamples`` plus optionally ``encoder.conv1`` in
    front, with every causal cache the slice consumes as an explicit input/output tensor.

    ``forward(x)`` (first chunk, every slot ``None``) or ``forward(x, *caches)`` ->
    ``(y, *cache_updates)`` with ``cache_count`` updates in the walk order of
    ``Encoder3d.forward`` (conv1, then module by module, conv by conv). The eager path is
    bit-identical to the stock encoder including every cache slot's contents (asserted by
    the CPU tests on real weights); the ``rearrange`` calls of ``Resample.forward`` are
    spelled out as permute/reshape for the ONNX tracer, which is the same memory op.
    """

    def __init__(self, modules: list[nn.Module], conv1: nn.Module | None = None):
        super().__init__()
        from flash_head.wan.modules.vae import (
            AttentionBlock,
            CausalConv3d,
            Resample,
            ResidualBlock,
        )

        modules = list(modules)
        if not modules:
            raise ValueError("Encoder span must cover at least one downsamples module")
        kinds = []
        for module in modules:
            if isinstance(module, ResidualBlock):
                kinds.append("residual")
            elif isinstance(module, Resample):
                if module.mode not in ("downsample2d", "downsample3d"):
                    raise TypeError(f"Resample mode {module.mode!r} is not an encoder downsample")
                if not (
                    isinstance(module.resample, nn.Sequential)
                    and len(module.resample) == 2
                    and isinstance(module.resample[0], nn.ZeroPad2d)
                    and isinstance(module.resample[1], nn.Conv2d)
                ):
                    raise TypeError(f"Unsupported Resample.resample form {_resample_form(module)}")
                kinds.append("resample")
            elif isinstance(module, AttentionBlock):
                # Stateless; never instantiated in the shipped encoder (attn_scales=[]).
                kinds.append("attention")
            else:
                raise TypeError(
                    f"Encoder span cannot contain a {type(module).__name__} (already-installed "
                    "stages are refused)"
                )
        if conv1 is not None and not (
            isinstance(conv1, CausalConv3d) and conv1._padding[4] > 0
        ):
            raise TypeError("Encoder conv1 must be the stock temporal CausalConv3d")
        self.conv1 = conv1
        self.span = nn.ModuleList(modules)
        self.kinds = kinds
        self.slots = ([encoder_module_cache_slots(conv1)] if conv1 is not None else []) + [
            encoder_module_cache_slots(module) for module in modules
        ]
        self.cache_count = sum(self.slots)
        self.observer = None

    # -- helpers ---------------------------------------------------------------------

    def _observe(self, name, before, previous, after):
        if self.observer is not None:
            self.observer(name, before, previous, after)

    @staticmethod
    def _take(caches, updates):
        return caches[len(updates)] if caches else None

    def _conv1(self, x, caches, updates):
        previous = self._take(caches, updates)
        updates.append(_cache_update(x, previous))
        before = x
        x = self.conv1(x, previous)
        self._observe(CONV1, before, previous, x)
        return x

    def _residual(self, entry, block, x, caches, updates):
        from flash_head.wan.modules.vae import CausalConv3d

        residual = block.shortcut(x)
        for layer_index, layer in enumerate(block.residual):
            if isinstance(layer, CausalConv3d):
                previous = self._take(caches, updates)
                updates.append(_cache_update(x, previous))
                before = x
                x = layer(x, previous)
                self._observe(f"span.{entry}.residual.{layer_index}", before, previous, x)
            else:
                x = layer(x)
        return x + residual

    def _resample(self, entry, module, x, caches, updates):
        b, c, t, h, w = x.shape
        # einops' "b c t h w -> (b t) c h w" and back, spelled out for the ONNX tracer.
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = module.resample(x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        if module.mode == "downsample3d":
            previous = self._take(caches, updates)
            if previous is None:
                # First call: stock stores the whole downsampled chunk, skips time_conv.
                updates.append(x.clone())
            else:
                updates.append(x[:, :, -1:].clone())
                before = x
                x = module.time_conv(torch.cat([previous[:, :, -1:], x], 2))
                self._observe(f"span.{entry}.time_conv", before, previous, x)
        return x

    # -- contract --------------------------------------------------------------------

    def forward(self, x, *caches):
        if len(caches) not in (0, self.cache_count):
            raise ValueError("Stage cache count mismatch")
        updates: list = []
        if self.conv1 is not None:
            x = self._conv1(x, caches, updates)
        for entry, (module, kind) in enumerate(zip(self.span, self.kinds)):
            if kind == "residual":
                x = self._residual(entry, module, x, caches, updates)
            elif kind == "resample":
                x = self._resample(entry, module, x, caches, updates)
            else:
                x = module(x)
        if len(updates) != self.cache_count:
            raise RuntimeError("Encoder span consumed an unexpected number of cache slots")
        return (x, *updates)


def make_encoder_stage(vae, group) -> EncoderSpanStage:
    encoder = vae.model.encoder
    group = parse_encoder_group(group)
    indices = encoder_group_indices(group)
    if not indices:
        raise ValueError("Encoder stage group must cover at least one downsamples index")
    if any(index < 0 or index >= len(encoder.downsamples) for index in indices):
        raise ValueError(
            f"Encoder stage group {group} is outside downsamples[0..{len(encoder.downsamples) - 1}]"
        )
    if encoder_group_has_conv1(group) and indices[0] != 0:
        raise ValueError(f"A group including {CONV1!r} must start at downsamples[0]")
    modules = [encoder.downsamples[index] for index in indices]
    return EncoderSpanStage(modules, encoder.conv1 if encoder_group_has_conv1(group) else None)


# --------------------------------------------------------------------------------------
# installation into Encoder3d
# --------------------------------------------------------------------------------------


class _Conv1Stub(nn.Module):
    """Stands in for ``encoder.conv1`` while a stage owns the real one.

    ``Encoder3d.forward`` runs conv1's cache bookkeeping itself: it hands the PREVIOUS
    pixel cache to ``conv1(x, cache)``, then writes the NEW pixel cache into the slot and
    advances the cursor -- before ``downsamples[0]`` (our adapter) runs. The previous cache
    would be lost by then, so this stub keeps it for the adapter and returns ``x`` (the
    pixels) unchanged; the stage then runs the real conv1 on them. It contains no
    ``CausalConv3d``: the real conv1 lives on as ``downsamples[0].stage.conv1``, so
    ``count_conv3d(encoder)`` is unchanged.
    """

    def __init__(self):
        super().__init__()
        self._armed = False
        self._previous = None

    @torch.compiler.disable
    def forward(self, x, cache_x=None):
        self._previous = cache_x
        self._armed = True
        return x

    def take(self):
        if not self._armed:
            raise RuntimeError(
                "encoder.conv1 stub did not run before the encoder stage adapter (module "
                "order changed?)"
            )
        previous, self._previous, self._armed = self._previous, None, False
        return previous


class EncoderStageAdapter(StageAdapter):
    """``StageAdapter`` with the conv1 slot handled the way ``Encoder3d.forward`` leaves it.

    Without conv1 in the span this is the decoder adapter verbatim (slots ``[begin, begin +
    cache_count)``, cursor advanced by ``cache_count``). With conv1, slot ``begin - 1`` is
    conv1's: ``Encoder3d.forward`` has already advanced the cursor past it and stored the
    new pixel cache there in bf16 -- the same value the stage returns as
    ``cache_updates[0]`` (validated, then dropped: the slot keeps the stock tensor, so the
    inline conv1 code of ``Encoder3d.forward`` never meets an fp16 passthrough cache);
    the previous pixel cache comes from the ``_Conv1Stub``. The cursor then advances by
    ``cache_count - 1``. Every check of the base adapter is kept.
    """

    def __init__(self, stage, execute, cache_dtype=None, conv1_stub=None):
        super().__init__(stage, execute, cache_dtype)
        object.__setattr__(self, "conv1_stub", conv1_stub)

    @torch.compiler.disable
    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is None or feat_idx is None:
            raise ValueError("Encoder stage adapter requires stock causal cache ownership")
        count = self.stage.cache_count
        begin = feat_idx[0]
        if self.conv1_stub is None:
            first = begin
            incoming = list(feat_cache[first : first + count])
        else:
            first = begin - 1
            if first < 0:
                raise ValueError("conv1 slot missing before the encoder stage adapter")
            incoming = [self.conv1_stub.take()] + list(feat_cache[begin : begin + count - 1])
        if len(incoming) != count:
            raise ValueError("Insufficient encoder stage cache slots")
        if all(value is None for value in incoming):
            args = (x,)
        elif all(isinstance(value, torch.Tensor) for value in incoming):
            args = (x, *incoming)
        else:
            raise ValueError("Partially initialized encoder stage cache")
        output = self.execute(*args)
        if not isinstance(output, (tuple, list)) or len(output) != count + 1:
            raise ValueError("Invalid stage output/cache contract")
        cache_dtypes = (x.dtype,) if self.cache_dtype is None else (x.dtype, self.cache_dtype)
        if any(value.device != x.device for value in output):
            raise ValueError("Stage output device mismatch")
        if output[0].dtype != x.dtype:
            raise ValueError("Stage output dtype mismatch")
        if any(value.dtype not in cache_dtypes for value in output[1:]):
            raise ValueError("Stage cache dtype mismatch")
        for previous, value in zip(incoming, output[1:]):
            if value.ndim != 5 or value.shape[0] != x.shape[0] or value.shape[2] not in (1, 2):
                raise ValueError("Invalid returned causal cache shape")
            if previous is not None and (
                value.shape[1] != previous.shape[1] or value.shape[3:] != previous.shape[3:]
            ):
                raise ValueError("Returned cache spatial/channel mismatch")
        if self.conv1_stub is None:
            feat_cache[first : first + count] = list(output[1:])
        else:
            feat_cache[begin : first + count] = list(output[2:])
        feat_idx[0] = first + count
        return output[0]


def install_encoder_stages(vae, groups, factory: Callable, cache_dtype=None) -> dict:
    """Resolve all groups before replacing any module; restore on install error.

    ``groups``: lists such as ``["conv1", 0, 1, 2]`` / ``[3, 4, 5]`` or their string forms.
    ``factory(group, stage)`` receives the parsed group and the eager ``EncoderSpanStage``
    and returns the callable the adapter executes. At most one group may include
    ``"conv1"`` and it must start at ``downsamples[0]``.
    """
    if hasattr(vae, "_pro_encoder_stage_originals"):
        raise ValueError("Encoder stages already installed")
    from flash_head.wan.modules.vae import count_conv3d

    encoder = vae.model.encoder
    downsamples = encoder.downsamples
    conv_count = count_conv3d(encoder)
    groups = [parse_encoder_group(group) for group in groups]
    selected, resolved, spans = set(), [], {}
    conv1_owner = None
    for group in groups:
        indices = encoder_group_indices(group)
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Encoder stage indices must be contiguous")
        if any(index < 0 or index >= len(downsamples) or index in selected for index in indices):
            raise ValueError("Invalid or overlapping encoder stage indices")
        if encoder_group_has_conv1(group):
            if indices[0] != 0:
                raise ValueError(f"A group including {CONV1!r} must start at downsamples[0]")
            if conv1_owner is not None:
                raise ValueError("Two encoder stages claim conv1")
            conv1_owner = indices[0]
        selected.update(indices)
        spans[encoder_group_key(group)] = describe_encoder_span(vae, group)
        stage = make_encoder_stage(vae, group).eval()
        execute = factory(group, stage)
        resolved.append((group, indices, stage, execute))
    originals = {index: downsamples[index] for index in selected}
    if conv1_owner is not None:
        originals[CONV1] = encoder.conv1
    try:
        for group, indices, stage, execute in resolved:
            stub = None
            if encoder_group_has_conv1(group):
                stub = _Conv1Stub()
                encoder.conv1 = stub
            downsamples[indices[0]] = EncoderStageAdapter(stage, execute, cache_dtype, stub)
            for index in indices[1:]:
                downsamples[index] = _SkipStage()
        if count_conv3d(encoder) != conv_count:
            raise RuntimeError("Encoder stage install changed count_conv3d(encoder)")
    except Exception:
        _restore_encoder_originals(encoder, originals)
        raise
    vae._pro_encoder_stage_originals = originals
    return {
        "groups": groups,
        "spans": spans,
        "conv1_in_stage": conv1_owner is not None,
        "cache_storage": "bfloat16" if cache_dtype is None else str(cache_dtype).replace("torch.", ""),
        "cache_owner": "stock WanVAE encode",
        "conv3d_count": conv_count,
        "fallback": False,
    }


def _restore_encoder_originals(encoder, originals) -> None:
    for index, module in originals.items():
        if index == CONV1:
            encoder.conv1 = module
        else:
            encoder.downsamples[index] = module


def remove_encoder_stages(vae) -> None:
    originals = getattr(vae, "_pro_encoder_stage_originals", None)
    if originals is not None:
        _restore_encoder_originals(vae.model.encoder, originals)
        del vae._pro_encoder_stage_originals


def installed_encoder_groups(vae) -> list:
    """The parsed groups of the installed encoder stages (``[]`` when none)."""
    encoder = vae.model.encoder
    groups = []
    for index, module in enumerate(encoder.downsamples):
        if isinstance(module, EncoderStageAdapter):
            group = [CONV1] if module.conv1_stub is not None else []
            group.extend(range(index, index + len(module.stage.span)))
            groups.append(group)
    return groups


# --------------------------------------------------------------------------------------
# plan installation
# --------------------------------------------------------------------------------------


def validate_encoder_plan_spans(vae, plan) -> dict:
    """Refuse a plan whose recorded module forms differ from the live encoder's."""
    recorded = plan.get("spans")
    if recorded is None:
        raise ValueError("Encoder stage plan has no 'spans' record")
    live = {}
    for group in plan["groups"]:
        group = parse_encoder_group(group)
        key = encoder_group_key(group)
        try:
            live[key] = describe_encoder_span(vae, group)
        except (IndexError, TypeError) as error:
            raise ValueError(
                f"Encoder stage plan group {key} cannot be resolved on the live encoder: {error}"
            ) from error
        if recorded.get(key) != live[key]:
            raise ValueError(
                f"Encoder stage plan group {key} was built for modules {recorded.get(key)} but "
                f"the live encoder has {live[key]}"
            )
    return live


def decoder_stage_arena(vae) -> dict | None:
    """The ``{"tensor", "stream", "lock"}`` workspace the installed DECODER stage plan's
    engines share, or ``None`` when no TensorRT decoder stage is installed.

    ``install_stage_plan`` keeps the arena in its ``execute`` closure only, so it is read
    back through the closure cells of the decoder adapters -- best effort, fail-safe:
    anything unexpected yields ``None`` and the encoder allocates its own arena.
    """
    decoder = getattr(getattr(vae, "model", None), "decoder", None)
    if decoder is None:
        return None
    for module in decoder.upsamples:
        if not isinstance(module, StageAdapter):
            continue
        execute = getattr(module, "execute", None)
        for cell in getattr(execute, "__closure__", None) or ():
            try:
                engines = cell.cell_contents
            except ValueError:
                continue
            if isinstance(engines, dict):
                for engine in engines.values():
                    if isinstance(engine, TensorRTStage):
                        return {
                            "tensor": engine.workspace,
                            "stream": engine.stream,
                            "lock": engine.enqueue_lock,
                        }
    return None


def install_encoder_plan(
    vae,
    path,
    *,
    vae_weights=None,
    unknown_signature: str = "refuse",
    share_decoder_arena: bool = True,
) -> dict:
    """Install the FP16 TensorRT encoder stage engines of ``path`` (a
    ``benchmarks/pro_30fps_20260922/encoder_stages.py build`` ``results.json``).

    Gates (all fail closed, mirroring ``install_stage_plan``): schema/status, plan target
    ``"encoder"``, ``weights_sha256`` == sha256 of ``vae_weights`` (or the stock
    checkpoint; the caller must have loaded that file into ``vae``), ``source_sha256`` of
    ``flash_head/wan/modules/vae.py``, the recorded ``spans`` against the live encoder,
    per-engine precision/binding records, finite build-sample evidence and the layer
    inspector hash, then the engine file's own sha/TensorRT version/GPU in
    ``TensorRTStage``.

    ``unknown_signature``: ``"refuse"`` (default, like the decoder: a shape the plan does
    not cover raises) or ``"eager"`` (run the stock arithmetic through the eager
    ``EncoderSpanStage`` for that call -- meant for the one-time ``frame_num``-frame
    reference encode of ``prepare_params`` when the plan was captured without
    ``--reference-frames``; counted in the returned ``unknown_signature_calls``).

    ``share_decoder_arena``: reuse the installed decoder stage plan's workspace (same
    stream and enqueue lock, so engine execution stays serialized) when it is large
    enough for these engines; otherwise -- or when no decoder stages are installed -- a
    private arena of ``max(workspace_bytes)`` is allocated.
    """
    from benchmarks.pro_quantization_v2_20260918.common import (
        ROOT,
        resolve_path,
        sha256,
    )

    if unknown_signature not in ("refuse", "eager"):
        raise ValueError("unknown_signature must be 'refuse' or 'eager'")
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or plan.get("status") != "complete":
        raise ValueError("Incomplete or unknown encoder stage plan")
    if plan.get("target") != PLAN_TARGET:
        raise ValueError(
            "Not an encoder stage plan (decoder plans install through "
            "soulx_rtc.pro_vae_stage_backend.install_stage_plan)"
        )
    checkpoint = resolve_path(vae_weights) if vae_weights else ROOT / STOCK_VAE_WEIGHTS
    if plan["weights_sha256"] != sha256(checkpoint):
        recorded = plan.get("vae_weights", {}).get("path") or plan.get("weights_path")
        raise ValueError(
            "Encoder stage weights mismatch"
            + (
                f": the plan was built from {recorded}; pass vae_weights=<that file> and load "
                "it into the VAE"
                if recorded and vae_weights is None
                else ""
            )
        )
    if plan.get("source_sha256", {}).get(VAE_SOURCE) != sha256(ROOT / VAE_SOURCE):
        raise ValueError("Encoder stage plan was exported from a different Wan VAE implementation")
    if plan["precision"] not in ("fp16", "bf16"):
        raise ValueError("Encoder stages are floating-point only")
    floating = plan.get("floating_precision", "bf16")
    if floating not in ("bf16", "fp16"):
        raise ValueError("Unsupported encoder stage floating precision")
    groups = [parse_encoder_group(group) for group in plan["groups"]]
    validate_encoder_plan_spans(vae, plan)
    sizes = [
        record["workspace_bytes"] for group in plan["stages"].values() for record in group.values()
    ]
    if not sizes or any(not isinstance(size, int) or size < 0 for size in sizes):
        raise ValueError("Invalid encoder stage workspace requirements")
    required = max(1, max(sizes))
    arena = decoder_stage_arena(vae) if share_decoder_arena else None
    shared = arena is not None and arena["tensor"].numel() >= required
    if not shared:
        arena = {
            "tensor": torch.empty(required, dtype=torch.uint8, device="cuda"),
            "stream": torch.cuda.Stream(),
            "lock": threading.Lock(),
        }
    binding_dtype = torch.float16 if floating == "fp16" else torch.bfloat16
    cache_passthrough = binding_dtype is torch.float16 and os.environ.get(
        "SOULX_STAGE_CACHE_PASSTHROUGH", "1"
    ) not in ("0", "false", "False")
    reuse_outputs = os.environ.get("SOULX_STAGE_REUSE_OUTPUTS", "0") not in ("0", "false", "False")
    fallback_calls: dict[str, int] = {}

    def factory(group, stage):
        records = plan["stages"][encoder_group_key(group)]
        engines = {}
        for signature, record in records.items():
            if record["precision"] != plan["precision"]:
                raise ValueError("Encoder stage precision mismatch")
            if record.get("floating_precision", "bf16") != floating:
                raise ValueError("Encoder stage floating precision mismatch")
            if record.get("binding_dtype", "bfloat16") != (
                "float16" if floating == "fp16" else "bfloat16"
            ):
                raise ValueError("Encoder stage floating binding mismatch")
            checks = record.get("build_sample_comparison", [])
            if len(checks) != stage.cache_count + 1 or not all(
                item.get("finite") is True for item in checks
            ):
                raise ValueError("Encoder stage engine lacks finite output/cache execution evidence")
            layers = resolve_path(record["layers_path"])
            if sha256(layers) != record["layers_sha256"]:
                raise ValueError("Encoder stage precision inspector hash mismatch")
            engines[signature] = TensorRTStage(
                resolve_path(record["path"]),
                record,
                arena=arena,
                cache_passthrough=cache_passthrough,
                reuse_outputs=reuse_outputs,
            )

        def execute(*values):
            signature = tensor_signature(values)
            engine = engines.get(signature)
            if engine is None:
                if unknown_signature == "eager":
                    fallback_calls[signature] = fallback_calls.get(signature, 0) + 1
                    # Caches may arrive in the engines' fp16 passthrough dtype; the stock
                    # arithmetic concatenates them with bf16 activations, which would
                    # promote to float32, so bring them to x's dtype first (exact: the
                    # values already round-trip through fp16).
                    x = values[0]
                    return stage(x, *(value.to(x.dtype) for value in values[1:]))
                raise ValueError(f"No verified encoder stage signature {signature}")
            return engine(*values)

        return execute

    result = install_encoder_stages(
        vae, groups, factory, cache_dtype=binding_dtype if cache_passthrough else None
    )
    result.update(
        {
            "plan": str(path),
            "plan_sha256": sha256(path),
            "target": PLAN_TARGET,
            "precision": plan["precision"],
            "floating_precision": floating,
            "workspace_bytes": required,
            "unshared_workspace_bytes": sum(sizes),
            "workspace_shared_with_decoder": shared,
            "workspace_execution": "one shared stream and host enqueue lock; serialized engine execution",
            "caller_cache_dtype": (
                str(binding_dtype).replace("torch.", "") if cache_passthrough else "bfloat16"
            ),
            "cache_passthrough": cache_passthrough,
            "reuse_outputs": reuse_outputs,
            "unknown_signature": unknown_signature,
            "unknown_signature_calls": fallback_calls,
            "signatures": {key: sorted(records) for key, records in plan["stages"].items()},
            "vae_weights": str(checkpoint),
        }
    )
    return result

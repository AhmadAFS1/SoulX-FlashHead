"""Opt-in Wan residual stages with explicit causal cache tensor boundaries.

The ordinary VAE owns/reset caches. A stage does not persist them. By default it
also does not alias runtime output buffers between calls; with
SOULX_STAGE_REUSE_OUTPUTS=1 each engine ping-pongs between two preallocated output
sets (bit-identical output, measured; +3.8% throughput; costs one extra output set
per signature in VRAM). No production configuration selects this.

Two stage classes share one runtime contract ``(x, *cache_in) -> (y, *cache_out)``:

* ``ResidualStage`` -- a contiguous run of ``decoder.upsamples`` ResidualBlocks (the
  historical groups ``4,5,6`` / ``8,9,10`` / ``12,13,14``; two cache slots per block).
* ``DecoderSpanStage`` -- any contiguous slice of ``decoder.upsamples`` that may also
  contain ``Resample`` (and ``AttentionBlock``) modules, optionally followed by
  ``decoder.head`` (group token ``"head"``, e.g. ``11,12,13,14,head``). Cache slots follow
  ``Decoder3d.forward`` exactly, see ``module_cache_slots``. When a span includes the head
  the stage's ``y`` IS the decoder output and ``decoder.head`` is replaced by an empty
  pass-through for the duration of the install. A group may end before the last
  ``upsamples`` index and still take the head when every later module is a
  ``SkippedResidualBlock`` (``pro_decoder_ops.install_decoder_block_skip``), e.g.
  ``11,12,head`` with blocks 13 and 14 pruned: see ``skipped_tail_after`` and
  ``DecoderSpanStage.slot_offsets`` for how the skipped blocks' cache slots are left alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from flash_head.utils.latency import latency_scope


def tensor_signature(values) -> str:
    return "__".join("x".join(map(str, value.shape)) for value in values)


# Group token naming the decoder head (``Decoder3d.head``) as the last member of a span.
HEAD = "head"


def parse_group(spec) -> list:
    """``"7,8,9,10"`` / ``"11,12,13,14,head"`` / a JSON list -> ``[7, 8, 9, 10]`` /
    ``[11, 12, 13, 14, "head"]``.

    Integers stay integers; the only non-numeric token is ``"head"`` and it must be last.
    Int-only specs come back exactly as ``list(map(int, ...))`` produced them, so every
    existing plan and calibration parses unchanged.
    """
    if isinstance(spec, str):
        tokens = [token.strip() for token in spec.split(",")]
    else:
        tokens = list(spec)
    group = []
    for token in tokens:
        if isinstance(token, str):
            if token == HEAD:
                group.append(HEAD)
                continue
            if token == "":
                continue
        try:
            if isinstance(token, (bool, float)):
                raise TypeError("not an index")
            group.append(int(token))
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid stage group token {token!r}: expected a decoder upsamples index "
                f"or {HEAD!r}"
            ) from None
    if HEAD in group[:-1]:
        raise ValueError(f"{HEAD!r} may only be the last entry of a stage group")
    return group


def group_key(group) -> str:
    """Plan/calibration dictionary key: ``"11-12-13-14-head"``."""
    return "-".join(map(str, parse_group(group)))


def group_indices(group) -> list[int]:
    """The ``decoder.upsamples`` indices of a group (the head token dropped)."""
    return [token for token in parse_group(group) if token != HEAD]


def group_has_head(group) -> bool:
    group = parse_group(group)
    return bool(group) and group[-1] == HEAD


def module_cache_slots(module) -> int:
    """Causal-cache slots ``Decoder3d.forward`` lets ``module`` consume per call.

    Read from ``flash_head/wan/modules/vae.py`` (not assumed):

    * ``ResidualBlock.forward``: one slot per ``CausalConv3d`` in ``.residual`` (two).
      The ``shortcut`` of a width-changing block (``upsamples[4]``, 192 -> 384) is a
      1x1x1 ``CausalConv3d`` called as ``self.shortcut(x)`` with NO cache argument: it
      consumes no slot (its temporal kernel is 1, so it needs no context either).
    * ``Resample.forward``: one slot in ``upsample3d`` mode (the ``time_conv``; the first
      call stores the ``"Rep"`` sentinel and advances the cursor WITHOUT running the
      conv), none in ``upsample2d`` mode (no cache code path at all). The downsample
      modes are encoder-side and refused.
    * ``AttentionBlock.forward(x)``: no cache argument, no slot.
    * ``decoder.head`` (an ``nn.Sequential``): one slot per ``CausalConv3d`` among its
      layers (one), the same protocol as the residual convs.
    """
    from flash_head.wan.modules.vae import (
        AttentionBlock,
        CausalConv3d,
        Resample,
        ResidualBlock,
    )

    if isinstance(module, ResidualBlock):
        return sum(isinstance(layer, CausalConv3d) for layer in module.residual)
    if isinstance(module, Resample):
        if module.mode == "upsample3d":
            if not isinstance(getattr(module, "time_conv", None), CausalConv3d):
                raise TypeError("upsample3d Resample without a CausalConv3d time_conv")
            return 1
        if module.mode == "upsample2d":
            return 0
        raise TypeError(f"Resample mode {module.mode!r} is not a decoder upsample")
    if isinstance(module, AttentionBlock):
        return 0
    if isinstance(module, nn.Sequential):
        # The decoder head. Decoder3d.forward iterates its direct children.
        return sum(isinstance(layer, CausalConv3d) for layer in module)
    raise TypeError(
        f"{type(module).__name__} cannot be part of a decoder stage span (only stock "
        "ResidualBlock / upsample Resample / AttentionBlock modules and the head)"
    )


def _resample_form(module) -> str:
    resample = module.resample
    if type(resample).__name__ == "SubPixelUpsampleConv":
        return "SubPixelUpsampleConv"
    if isinstance(resample, nn.Sequential):
        return "+".join(type(layer).__name__ for layer in resample)
    return type(resample).__name__


def describe_module(module) -> str:
    """Class-level description used to pin a plan to the live decoder's module forms."""
    from flash_head.wan.modules.vae import Resample

    if isinstance(module, Resample):
        return f"Resample:{module.mode}:{_resample_form(module)}"
    return type(module).__name__


def skipped_tail_after(decoder, last_index: int) -> list[tuple[int, int]]:
    """``[(index, slots), ...]`` for every ``upsamples`` module AFTER ``last_index``, all of
    which must be ``SkippedResidualBlock`` identities; ``[]`` when ``last_index`` is the last
    ``upsamples`` index.

    This is the rule that lets a group take the decoder head without ending at
    ``upsamples[-1]``: ``Decoder3d.forward`` walks every ``upsamples`` module and then the
    head, so a head-inclusive span may only leave out trailing modules that compute nothing.
    A ``SkippedResidualBlock`` is exactly that -- an identity that advances the cache cursor
    by the ``slots`` its block would have consumed and never touches ``feat_cache`` -- so the
    span can run ``... -> last module -> head`` while the skipped identities, left in place
    in ``decoder.upsamples``, keep advancing the cursor exactly where the pruned eager
    decoder advances it. Anything else after ``last_index`` (a live ``ResidualBlock``, a
    ``Resample``, an already-installed stage) is refused with ``TypeError``.
    """
    from soulx_rtc.pro_decoder_ops import SkippedResidualBlock

    upsamples = decoder.upsamples
    tail = []
    for index in range(int(last_index) + 1, len(upsamples)):
        module = upsamples[index]
        if not isinstance(module, SkippedResidualBlock):
            raise TypeError(
                f"A stage may include the decoder head only when it ends at "
                f"upsamples[{len(upsamples) - 1}] or every later upsamples module is a "
                f"SkippedResidualBlock (pro_decoder_ops.install_decoder_block_skip); "
                f"upsamples[{index}] is a {type(module).__name__}"
            )
        slots = int(module.slots)
        if slots <= 0:
            raise TypeError(f"upsamples[{index}] SkippedResidualBlock advances {slots} slots")
        tail.append((index, slots))
    return tail


def describe_span(vae, group) -> list[str]:
    """One entry per module a group covers, in walk order, e.g.
    ``["Resample:upsample2d:SubPixelUpsampleConv", "ResidualBlock", "ResidualBlock",
    "ResidualBlock", "head:RMS_norm+SiLU+CausalConv3d"]``. Recorded by capture/build
    as ``spans`` and compared verbatim by ``install_stage_plan``.

    When the head follows skipped blocks (``skipped_tail_after``) the head entry names
    them: ``"head:RMS_norm+SiLU+CausalConv3d:after-skipped:13,14"``. The entry count stays
    ``len(group)`` (one per token), so positional readers such as
    ``decoder_stages.parse_use_blocks`` are unaffected. Raises ``TypeError`` for a
    head-inclusive group whose trailing modules are not all skipped.
    """
    decoder = vae.model.decoder
    group = parse_group(group)
    indices = group_indices(group)
    described = [describe_module(decoder.upsamples[index]) for index in indices]
    if group_has_head(group):
        entry = HEAD + ":" + "+".join(type(layer).__name__ for layer in decoder.head)
        if indices:
            tail = skipped_tail_after(decoder, indices[-1])
            if tail:
                entry += ":after-skipped:" + ",".join(str(index) for index, _ in tail)
        described.append(entry)
    return described


def head_after_skipped_blocks(vae, group) -> list[int]:
    """Indices of the ``SkippedResidualBlock`` modules a head-inclusive group's head
    follows (``[]`` for a group without the head or one ending at ``upsamples[-1]``)."""
    group = parse_group(group)
    indices = group_indices(group)
    if not group_has_head(group) or not indices:
        return []
    return [index for index, _ in skipped_tail_after(vae.model.decoder, indices[-1])]


def span_cache_count(vae, group) -> int:
    decoder = vae.model.decoder
    group = parse_group(group)
    count = sum(module_cache_slots(decoder.upsamples[index]) for index in group_indices(group))
    if group_has_head(group):
        count += module_cache_slots(decoder.head)
    return count


def _cache_update(x, previous):
    """The tensor every cache-consuming module stores back into its slot.

    Verbatim ``ResidualBlock.forward`` / ``Decoder3d.forward`` (head) /
    ``Resample.forward`` (tensor branch): the last two frames of the conv INPUT, with the
    previous cache's last frame prepended when the call carries a single frame. For the
    Resample's ``"Rep"`` branch the stock code prepends ``zeros_like`` instead, which is
    what ``previous[:, :, -1:]`` yields when the sentinel is materialised as zero frames
    (see ``DecoderSpanStage``).
    """
    update = x[:, :, -2:].clone()
    if x.shape[2] < 2 and previous is not None:
        update = torch.cat((previous[:, :, -1:].to(update.device), update), dim=2)
    return update


class ResidualStage(nn.Module):
    """Pure residual-block group; initial call has no incoming cache tensors."""

    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        from flash_head.wan.modules.vae import ResidualBlock

        if not blocks or not all(isinstance(block, ResidualBlock) for block in blocks):
            raise ValueError("Stage must contain only contiguous Wan ResidualBlocks")
        self.blocks = nn.ModuleList(blocks)
        self.cache_count = 2 * len(blocks)
        self.observer = None

    def forward(self, x, *caches):
        if len(caches) not in (0, self.cache_count):
            raise ValueError("Stage cache count mismatch")
        from flash_head.wan.modules.vae import CausalConv3d

        updates = []
        for block_index, block in enumerate(self.blocks):
            residual = block.shortcut(x)
            for layer_index, layer in enumerate(block.residual):
                if isinstance(layer, CausalConv3d):
                    index = len(updates)
                    previous = caches[index] if caches else None
                    update = x[:, :, -2:].clone()
                    if x.shape[2] < 2 and previous is not None:
                        update = torch.cat((previous[:, :, -1:], update), dim=2)
                    updates.append(update)
                    before = x
                    x = layer(x, previous)
                    if self.observer is not None:
                        self.observer(
                            f"blocks.{block_index}.residual.{layer_index}",
                            before,
                            previous,
                            x,
                        )
                else:
                    x = layer(x)
            x = x + residual
        return (x, *updates)


class _SkipStage(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=None):
        return x


class _PassThroughHead(nn.Sequential):
    """Stands in for ``decoder.head`` while a stage span owns the real head.

    ``Decoder3d.forward`` runs ``for layer in self.head: ...``; an EMPTY Sequential makes
    that loop a no-op, so the stage's ``y`` (already the 3-channel decoder output) is
    returned untouched. It contains no ``CausalConv3d``: ``count_conv3d(decoder)`` (what
    ``clear_cache`` sizes ``_feat_map`` with) still sees the real head's conv, because
    the real head lives on as ``upsamples[first].stage.head``.
    """


class DecoderSpanStage(nn.Module):
    """A contiguous slice of ``decoder.upsamples`` (ResidualBlock / Resample /
    AttentionBlock modules) plus optionally ``decoder.head``, with every causal cache the
    slice consumes as an explicit input/output tensor.

    Contract (identical to ``ResidualStage``): ``forward(x)`` for the decoder's first call
    (every slot ``None``), ``forward(x, *caches)`` afterwards, returning
    ``(y, *cache_updates)`` with ``cache_count`` updates in slot order. The slot order is
    the walk order of ``Decoder3d.forward`` (module by module, conv by conv); the slot
    counts come from ``module_cache_slots``.

    The upsample3d ``Resample`` (``upsamples[3]`` / ``[7]``) is the one module whose
    stock protocol is not tensor-only. ``Resample.forward`` on its FIRST call stores the
    string ``"Rep"`` in its slot, advances the cursor and does not run ``time_conv`` (the
    frame count is unchanged); on the SECOND call it sees ``"Rep"``, runs
    ``time_conv(x)`` with no cache -- i.e. ``F.pad``'s two leading zero frames -- and
    stores ``cat(zeros, x[-1:])`` (or ``x[-2:]``) as the tensor cache; from the third
    call on it is the ordinary tensor protocol. This stage materialises ``"Rep"`` as an
    all-zero two-frame cache tensor ``(b, c, 2, h, w)`` at the pre-upsample resolution:
    ``CausalConv3d.forward(x, zeros)`` concatenates those two zero frames and pads
    nothing, so the SAME tensor reaches the convolution as under the stock zero padding,
    and ``_cache_update`` then prepends ``zeros[:, :, -1:]`` exactly where the stock code
    prepends ``zeros_like``. Outputs are bit-identical (asserted by the CPU tests); the
    only visible difference is that ``feat_cache[slot]`` holds a zero tensor instead of
    the string after the first call, which nothing else reads.

    The spatial resample runs in one of two exact forms: ``SubPixelUpsampleConv``
    (``pro_decoder_ops.install_subpixel_resample``; exports as Conv + DepthToSpace) or
    the stock ``Upsample(nearest-exact, 2x) + Conv2d``. The stock form is computed as
    ``F.interpolate(mode="nearest")`` + the same Conv2d: for an integer factor of 2 the
    two nearest modes select the same source index (``floor(d/2)`` vs
    ``floor((d + 0.5)/2)``), which is asserted on a probe at construction, and
    ``nearest-exact`` has no ONNX symbolic in torch 2.7 whereas ``nearest`` exports as
    ``Resize``.

    ``head_after_skipped``: ``[(index, slots), ...]`` from ``skipped_tail_after`` when the
    head follows pruned blocks (``11,12,head`` with 13 and 14 skipped). It changes NOTHING
    about the stage's arithmetic or its ``(x, *cache_in) -> (y, *cache_out)`` contract --
    the skipped blocks compute nothing and their slots are not engine I/O -- only WHERE in
    the decoder's ``feat_cache`` the ``StageAdapter`` reads and writes the stage's caches:
    ``slot_offsets`` (relative to the cursor at entry) leaves a gap of ``sum(slots)`` between
    the last span module's slots and the head's, e.g. ``[0, 1, 6]`` for block 12's two and
    the head's one with two 2-slot blocks skipped in between. The adapter still advances
    the cursor by ``cache_count`` only; the skipped identities that follow it in
    ``decoder.upsamples`` advance the gap themselves, exactly as in the pruned eager walk,
    so the end cursor and every slot index are unchanged and the skipped slots stay ``None``.
    """

    def __init__(
        self,
        modules: list[nn.Module],
        head: nn.Module | None = None,
        head_after_skipped: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        from flash_head.wan.modules.vae import (
            AttentionBlock,
            CausalConv3d,
            Resample,
            ResidualBlock,
        )

        modules = list(modules)
        if not modules:
            raise ValueError("Decoder span must cover at least one upsamples module")
        kinds, forms = [], []
        for module in modules:
            if isinstance(module, ResidualBlock):
                kinds.append("residual")
                forms.append(None)
            elif isinstance(module, Resample):
                if module.mode not in ("upsample2d", "upsample3d"):
                    raise TypeError(f"Resample mode {module.mode!r} is not a decoder upsample")
                kinds.append("resample")
                forms.append(self._resample_form(module))
            elif isinstance(module, AttentionBlock):
                # Stateless (forward(x) only); never instantiated in the shipped decoder
                # (attn_scales=[]), so this path is untested against TensorRT.
                kinds.append("attention")
                forms.append(None)
            else:
                raise TypeError(
                    f"Decoder span cannot contain a {type(module).__name__} (skipped "
                    "blocks and already-installed stages are refused)"
                )
        if head is not None:
            if not isinstance(head, nn.Sequential) or not any(
                isinstance(layer, CausalConv3d) for layer in head
            ):
                raise TypeError("Decoder head must be a Sequential containing a CausalConv3d")
            if any(isinstance(layer, (ResidualBlock, Resample, AttentionBlock)) for layer in head):
                raise TypeError("Decoder head contains cache-owning blocks; refusing")
        head_after_skipped = [(int(i), int(s)) for i, s in (head_after_skipped or [])]
        if head_after_skipped and head is None:
            raise ValueError("head_after_skipped is only meaningful for a span that owns the head")
        if any(s <= 0 for _, s in head_after_skipped):
            raise ValueError("Skipped blocks before the head must advance at least one slot")
        self.span = nn.ModuleList(modules)
        self.head = head
        self.kinds = kinds
        self.forms = forms
        self.slots = [module_cache_slots(module) for module in modules]
        if head is not None:
            self.slots.append(module_cache_slots(head))
        self.cache_count = sum(self.slots)
        # Plain Python attributes (not buffers/parameters): they never enter the state
        # dict or the exported graph.
        self.head_after_skipped = [index for index, _ in head_after_skipped]
        self.head_gap_slots = sum(slots for _, slots in head_after_skipped)
        span_slots = sum(self.slots[:-1]) if head is not None else self.cache_count
        self.slot_offsets = list(range(span_slots))
        if head is not None:
            self.slot_offsets += [
                span_slots + self.head_gap_slots + k for k in range(self.slots[-1])
            ]
        self.observer = None

    @staticmethod
    def _resample_form(module) -> str:
        import torch.nn.functional as F

        resample = module.resample
        if type(resample).__name__ == "SubPixelUpsampleConv":
            return "subpixel"
        if (
            isinstance(resample, nn.Sequential)
            and len(resample) == 2
            and isinstance(resample[0], nn.Upsample)
            and isinstance(resample[1], nn.Conv2d)
        ):
            upsample = resample[0]
            scale = upsample.scale_factor
            scales = tuple(scale) if isinstance(scale, (tuple, list)) else (scale, scale)
            if (
                upsample.mode != "nearest-exact"
                or upsample.size is not None
                or any(float(s) != 2.0 for s in scales)
            ):
                raise TypeError(f"Unsupported Upsample {upsample!r} in Resample")
            conv = resample[1]
            probe = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
            probe = probe.to(device=conv.weight.device, dtype=conv.weight.dtype)
            with torch.no_grad():
                if not torch.equal(
                    upsample(probe), F.interpolate(probe, scale_factor=2.0, mode="nearest")
                ):
                    raise ValueError("nearest-exact and nearest upsampling disagree at scale 2")
            return "stock"
        raise TypeError(f"Unsupported Resample.resample form {type(resample).__name__}")

    def _observe(self, name, before, previous, after):
        if self.observer is not None:
            self.observer(name, before, previous, after)

    @staticmethod
    def _take(caches, updates):
        return caches[len(updates)] if caches else None

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

    def _spatial(self, entry, module, x):
        import torch.nn.functional as F

        if self.forms[entry] == "subpixel":
            return module.resample(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return module.resample[1](x)

    def _resample(self, entry, module, x, caches, updates):
        b, c, t, h, w = x.shape
        if module.mode == "upsample3d":
            previous = self._take(caches, updates)
            if previous is None:
                # First call: stock stores "Rep", advances the cursor, skips time_conv.
                updates.append(x.new_zeros((b, c, 2, h, w)))
            else:
                updates.append(_cache_update(x, previous))
                before = x
                x = module.time_conv(x, previous)
                self._observe(f"span.{entry}.time_conv", before, previous, x)
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3)
                x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        # einops' "b c t h w -> (b t) c h w" and back, spelled out for the ONNX tracer.
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self._spatial(entry, module, x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x

    def _head(self, x, caches, updates):
        from flash_head.wan.modules.vae import CausalConv3d

        for layer_index, layer in enumerate(self.head):
            if isinstance(layer, CausalConv3d):
                previous = self._take(caches, updates)
                updates.append(_cache_update(x, previous))
                before = x
                x = layer(x, previous)
                self._observe(f"head.{layer_index}", before, previous, x)
            else:
                x = layer(x)
        return x

    def forward(self, x, *caches):
        if len(caches) not in (0, self.cache_count):
            raise ValueError("Stage cache count mismatch")
        updates: list = []
        for entry, (module, kind) in enumerate(zip(self.span, self.kinds)):
            if kind == "residual":
                x = self._residual(entry, module, x, caches, updates)
            elif kind == "resample":
                x = self._resample(entry, module, x, caches, updates)
            else:
                x = module(x)
        if self.head is not None:
            x = self._head(x, caches, updates)
        if len(updates) != self.cache_count:
            raise RuntimeError("Decoder span consumed an unexpected number of cache slots")
        return (x, *updates)


def make_stage(vae, group):
    """``ResidualStage`` for a pure ResidualBlock group (the historical form, so existing
    plans keep their class, parameter names and exported graphs), ``DecoderSpanStage``
    for anything else. A head-inclusive group that ends before ``upsamples[-1]`` is
    accepted only when every later module is a ``SkippedResidualBlock``
    (``skipped_tail_after``; ``TypeError`` otherwise)."""
    from flash_head.wan.modules.vae import ResidualBlock

    decoder = vae.model.decoder
    group = parse_group(group)
    indices = group_indices(group)
    if not indices:
        raise ValueError("Stage group must cover at least one upsamples index")
    if any(index < 0 or index >= len(decoder.upsamples) for index in indices):
        raise ValueError(
            f"Stage group {group} is outside upsamples[0..{len(decoder.upsamples) - 1}]"
        )
    modules = [decoder.upsamples[index] for index in indices]
    if not group_has_head(group) and all(isinstance(module, ResidualBlock) for module in modules):
        return ResidualStage(modules)
    if not group_has_head(group):
        return DecoderSpanStage(modules)
    return DecoderSpanStage(
        modules, decoder.head, head_after_skipped=skipped_tail_after(decoder, indices[-1])
    )


def stage_slot_offsets(stage) -> list[int]:
    """Where, relative to the cursor at entry, a stage's ``cache_count`` caches live in
    ``feat_cache``: ``DecoderSpanStage.slot_offsets`` when the stage defines it, else the
    contiguous ``0 .. cache_count-1`` every stage has always used."""
    offsets = getattr(stage, "slot_offsets", None)
    if offsets is None:
        return list(range(stage.cache_count))
    offsets = [int(offset) for offset in offsets]
    if len(offsets) != stage.cache_count or offsets != sorted(set(offsets)) or (
        offsets and offsets[0] < 0
    ):
        raise ValueError(f"Invalid stage slot offsets {offsets} for {stage.cache_count} caches")
    return offsets


class StageAdapter(nn.Module):
    def __init__(self, stage: nn.Module, execute: Callable, cache_dtype=None):
        super().__init__()
        self.stage = stage
        # cache_dtype: an ADDITIONAL dtype the causal cache tensors may legally carry,
        # on top of x's own dtype. Set when the execute backend hands caches back in its
        # binding dtype (see TensorRTStage.cache_passthrough). Left None for the eager
        # ResidualStage path, which returns bf16 like x, so that path keeps the strict
        # single-dtype contract it has always had.
        self.cache_dtype = cache_dtype
        # feat_cache slots this stage reads/writes, relative to the cursor at entry.
        # Contiguous 0..cache_count-1 for every stage except a DecoderSpanStage whose head
        # follows skipped blocks (see DecoderSpanStage.slot_offsets).
        self.slot_offsets = stage_slot_offsets(stage)
        # Keep runtime objects out of the module tree / state dict.
        object.__setattr__(self, "execute", execute)

    @torch.compiler.disable
    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is None or feat_idx is None:
            raise ValueError("Stage adapter requires stock causal cache ownership")
        begin = feat_idx[0]
        if self.slot_offsets and begin + self.slot_offsets[-1] >= len(feat_cache):
            raise ValueError("Insufficient stage cache slots")
        incoming = [feat_cache[begin + offset] for offset in self.slot_offsets]
        if len(incoming) != self.stage.cache_count:
            raise ValueError("Insufficient stage cache slots")
        if all(value is None for value in incoming):
            args = (x,)
        elif all(isinstance(value, torch.Tensor) for value in incoming):
            args = (x, *incoming)
        else:
            raise ValueError("Partially initialized residual stage cache")
        output = self.execute(*args)
        if (
            not isinstance(output, (tuple, list))
            or len(output) != self.stage.cache_count + 1
        ):
            raise ValueError("Invalid stage output/cache contract")
        # Validate all outputs before touching caller-owned state.
        # y must always come back in x's dtype; the caches may additionally carry
        # cache_dtype when the backend passes them through un-cast. Still an exact
        # allow-list, not a relaxation to "any dtype".
        cache_dtypes = (
            (x.dtype,) if self.cache_dtype is None else (x.dtype, self.cache_dtype)
        )
        if any(value.device != x.device for value in output):
            raise ValueError("Stage output device mismatch")
        if output[0].dtype != x.dtype:
            raise ValueError("Stage output dtype mismatch")
        if any(value.dtype not in cache_dtypes for value in output[1:]):
            raise ValueError("Stage cache dtype mismatch")
        # Per-slot checks only: a slot is compared with ITS OWN previous value, never
        # with x, because one span mixes resolutions and widths -- a DecoderSpanStage
        # over upsamples[7-10] returns the time_conv cache at the pre-upsample shape
        # (384 ch, 144x80) next to the block caches (192 ch, 288x160), and a head-
        # inclusive span returns y with 3 channels while its caches carry 96. The
        # frame-count rule stays: 1 or 2 frames, and the Resample's materialised "Rep"
        # (two zero frames) is a legal 2-frame cache from the first call on.
        for previous, value in zip(incoming, output[1:]):
            if (
                value.ndim != 5
                or value.shape[0] != x.shape[0]
                or value.shape[2] not in (1, 2)
            ):
                raise ValueError("Invalid returned causal cache shape")
            # Initial one-frame caches become two-frame caches next call.
            if previous is not None and (
                value.shape[1] != previous.shape[1]
                or value.shape[3:] != previous.shape[3:]
            ):
                raise ValueError("Returned cache spatial/channel mismatch")
        for offset, value in zip(self.slot_offsets, output[1:]):
            feat_cache[begin + offset] = value
        # Only this stage's own consumption. When the head follows skipped blocks, the
        # SkippedResidualBlock identities still sitting in decoder.upsamples after this
        # adapter advance their own slots, exactly as in the pruned eager decoder.
        feat_idx[0] += self.stage.cache_count
        return output[0]


def install_stages(vae, groups, factory: Callable, cache_dtype=None) -> dict:
    """Resolve all groups before replacing any module; restore on install error.

    ``groups``: lists such as ``[4, 5, 6]`` or ``[11, 12, 13, 14, "head"]`` (or their
    comma-separated string forms). ``factory(group, stage)`` receives the parsed group
    (ints, plus the ``"head"`` token when present) and the eager stage, and returns the
    callable the ``StageAdapter`` executes. A group may include the head only when it
    ends at the last ``upsamples`` index OR every later ``upsamples`` module is a
    ``SkippedResidualBlock`` (``11,12,head`` with 13 and 14 pruned; ``skipped_tail_after``);
    the head is then absorbed into that stage and ``decoder.head`` becomes an empty
    pass-through until ``remove_stages``. The skipped identities after such a group are
    NOT replaced: they stay in ``decoder.upsamples``, keep advancing the cache cursor by
    their slots, and are restored by ``remove_decoder_block_skip``, not by
    ``remove_stages``. Only the group's own indices are replaced (``StageAdapter`` at the
    first, ``_SkipStage`` at the rest).
    """
    if hasattr(vae, "_pro_stage_originals"):
        raise ValueError("Decoder stages already installed")
    decoder = vae.model.decoder
    upsamples = decoder.upsamples
    selected = set()
    resolved = []
    spans = {}
    head_owner = None
    head_after_skipped: list[int] = []
    groups = [parse_group(group) for group in groups]
    for group in groups:
        indices = group_indices(group)
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Stage indices must be contiguous")
        if any(
            index < 0 or index >= len(upsamples) or index in selected
            for index in indices
        ):
            raise ValueError("Invalid or overlapping stage indices")
        if group_has_head(group):
            if indices[-1] != len(upsamples) - 1:
                try:
                    tail = skipped_tail_after(decoder, indices[-1])
                except TypeError as error:
                    raise ValueError(str(error)) from None
                head_after_skipped = [index for index, _ in tail]
            if head_owner is not None:
                raise ValueError("Two stages claim the decoder head")
            head_owner = indices[0]
        selected.update(indices)
        spans[group_key(group)] = describe_span(vae, group)
        stage = make_stage(vae, group).eval()
        execute = factory(group, stage)
        resolved.append((group, indices, stage, execute))
    if set(head_after_skipped) & selected:
        raise ValueError(
            f"Skipped blocks {sorted(set(head_after_skipped) & selected)} before the head "
            "cannot also belong to a stage group"
        )
    originals = {index: upsamples[index] for index in selected}
    if head_owner is not None:
        originals[HEAD] = decoder.head
    try:
        for group, indices, stage, execute in resolved:
            upsamples[indices[0]] = StageAdapter(stage, execute, cache_dtype)
            for index in indices[1:]:
                upsamples[index] = _SkipStage()
            if group_has_head(group):
                decoder.head = _PassThroughHead()
    except Exception:
        _restore_stage_originals(decoder, originals)
        raise
    vae._pro_stage_originals = originals
    report = {
        "groups": groups,
        "spans": spans,
        "head_in_stage": head_owner is not None,
        "cache_storage": (
            "bfloat16" if cache_dtype is None else str(cache_dtype).replace("torch.", "")
        ),
        "cache_owner": "stock WanVAE decode",
        "fallback": False,
    }
    if head_after_skipped:
        # Recorded only when it applies, so every other install's manifest is unchanged.
        report["head_after_skipped_blocks"] = head_after_skipped
    return report


def _restore_stage_originals(decoder, originals) -> None:
    for index, module in originals.items():
        if index == HEAD:
            decoder.head = module
        else:
            decoder.upsamples[index] = module


def remove_stages(vae):
    originals = getattr(vae, "_pro_stage_originals", None)
    if originals is not None:
        _restore_stage_originals(vae.model.decoder, originals)
        del vae._pro_stage_originals


# Output-pool selector for multi-session use. The overlap-skip shim persists the decoder's
# causal cache ACROSS windows, and with `reuse_outputs` those cache tensors are the pool
# buffers themselves. Two sessions sharing a pool would overwrite each other's persisted
# cache between windows (session B's seven calls per engine rewrite both slots that
# session A's next window reads as cache_in). Keying the pool by session as well as by
# caller stream restores the single-stream safety argument per session.
_POOL_KEY = ["0"]


def set_output_pool_key(key) -> None:
    _POOL_KEY[0] = str(key)


class TensorRTStage:
    """Static bindings, owned outputs and a stream fenced to the caller.

    A dedicated stream avoids TensorRT's default-stream synchronization and
    serializes shared workspace use across interleaved caller streams.
    """

    def __init__(self, engine_path, metadata, *, arena=None, cache_passthrough=False,
                 reuse_outputs=False):
        import tensorrt as trt

        # cache_passthrough: hand the causal cache tensors back to the caller in the
        # engine's own binding dtype instead of casting them to BF16 and back on every
        # call. The cache_in/cache_out bindings are already float16 in the plan, so the
        # values already round-trip through fp16 range in both directions today -- this
        # strictly DELETES a bf16 rounding step (11 -> 8 mantissa bits), it does not add
        # one. Only the x input and the y output still cross the precision boundary.
        self.cache_passthrough = bool(cache_passthrough)
        # Lazily-built ping-pong pools for engine outputs, ONE PER CALLER STREAM; see
        # __call__. Keyed by caller stream because the pool's safety argument relies on
        # stream ordering between a buffer's writer (the engine stream, fenced to the
        # caller) and its reader (the caller stream). Two sessions on two streams
        # sharing one pool would let session B's second call overwrite the buffer
        # session A's consumer is still reading.
        self.reuse_outputs = bool(reuse_outputs)
        self._output_pool: dict[tuple, list[list[torch.Tensor]]] = {}
        self._output_slot: dict[tuple, int] = {}

        self.trt = trt
        self.path = Path(engine_path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Stage engine checksum mismatch")
        if (
            metadata["tensorrt"] != trt.__version__
            or metadata["gpu"] != torch.cuda.get_device_name()
        ):
            raise ValueError("Stage engine environment mismatch")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize stage engine")
        self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError("Cannot create stage execution context")
        required = max(1, self.engine.device_memory_size)
        if arena is None:
            arena = {
                "tensor": torch.empty(required, dtype=torch.uint8, device="cuda"),
                "stream": torch.cuda.Stream(),
                "lock": threading.Lock(),
            }
        self.workspace = arena["tensor"]
        if self.workspace.numel() < required or self.workspace.dtype != torch.uint8:
            raise ValueError(
                "Shared stage workspace is smaller than the engine requirement"
            )
        self.context.device_memory = self.workspace.data_ptr()
        self.stream = arena["stream"]
        self.enqueue_lock = arena["lock"]
        self.inputs = metadata["inputs"]
        self.outputs = metadata["outputs"]
        dtype = metadata.get("binding_dtype", "bfloat16")
        if dtype not in ("bfloat16", "float16"):
            raise ValueError("Unsupported stage binding dtype")
        self.binding_dtype = getattr(torch, dtype)
        trt_dtype = trt.bfloat16 if dtype == "bfloat16" else trt.float16
        if self.engine.num_io_tensors != len(self.inputs) + len(self.outputs):
            raise ValueError("Stage binding count mismatch")
        for mode, specs in (
            (trt.TensorIOMode.INPUT, self.inputs),
            (trt.TensorIOMode.OUTPUT, self.outputs),
        ):
            for spec in specs:
                name = spec["name"]
                if (
                    tuple(self.engine.get_tensor_shape(name)) != tuple(spec["shape"])
                    or self.engine.get_tensor_mode(name) != mode
                    or self.engine.get_tensor_dtype(name) != trt_dtype
                ):
                    raise ValueError(f"Invalid stage binding {name}")

    def __call__(self, *values):
        if len(values) != len(self.inputs):
            raise ValueError("Stage binding arity mismatch")
        with latency_scope("trt.input_cast_contiguous", engine=str(self.path)):
            prepared = []
            for index, (value, spec) in enumerate(zip(values, self.inputs)):
                # inputs[0] is x; the rest are cache_in_*. Under cache_passthrough the
                # caches arrive already in binding dtype, so only x crosses the boundary.
                allowed = (torch.bfloat16,)
                if index > 0 and self.cache_passthrough:
                    allowed = (torch.bfloat16, self.binding_dtype)
                if (
                    value.dtype not in allowed
                    or not value.is_cuda
                    or tuple(value.shape) != tuple(spec["shape"])
                    or value.device != self.workspace.device
                ):
                    raise ValueError("Stage binding shape/dtype/device mismatch")
                # One pass, not two. `.to(dtype)` PRESERVES non-contiguous strides, so the
                # historical `.to(dtype).contiguous()` made a second full pass over x
                # (einops hands us stride (106168320, 184320, 17694720, 320, 1)).
                # Fusing the format request into the same call collapses them; when the
                # source is already contiguous and of the target dtype this returns the
                # SAME tensor object, so it is a kernel deletion rather than a cheaper
                # kernel. Verified on this build for both the identity and strided cases.
                prepared.append(
                    value.to(self.binding_dtype, memory_format=torch.contiguous_format)
                )
        caller = torch.cuda.current_stream(self.workspace.device)
        with latency_scope("trt.output_allocation", gpu=False, engine=str(self.path)):
            if self.reuse_outputs:
                # Ping-pong between two preallocated sets instead of asking the caching
                # allocator for ~141 MB x 7 bindings on every one of the 21 engine calls
                # per window. That churn is what drives the reserved-pool oscillation
                # (7.5 <-> 10.5 GiB) and the GPU idle observed inside decode.
                #
                # Two sets is the minimum that is SAFE, and exactly enough: the decoder
                # walks one latent frame at a time through every stage, so consecutive
                # calls to a given engine are consecutive iterations. cache_out written
                # at iteration i is consumed as cache_in at i+1, and only overwritten at
                # i+2 -- after its single reader has run. The y output is consumed by the
                # next decoder module within the same iteration. A single shared buffer
                # would alias input and output and corrupt the cache.
                key = (caller.cuda_stream, _POOL_KEY[0])
                pool = self._output_pool.get(key)
                if pool is None:
                    pool = self._output_pool[key] = [
                        [
                            torch.empty(
                                spec["shape"],
                                device=self.workspace.device,
                                dtype=self.binding_dtype,
                            )
                            for spec in self.outputs
                        ]
                        for _ in range(2)
                    ]
                    self._output_slot[key] = 0
                outputs = pool[self._output_slot[key]]
                self._output_slot[key] ^= 1
            else:
                outputs = [
                    torch.empty(
                        spec["shape"], device=self.workspace.device, dtype=self.binding_dtype
                    )
                    for spec in self.outputs
                ]
        with latency_scope("trt.bind_enqueue_fence", engine=str(self.path)):  # noqa: SIM117 - record lock waiting too
            with self.enqueue_lock:
                for spec, value in zip(self.inputs + self.outputs, prepared + outputs):
                    if not self.context.set_tensor_address(spec["name"], value.data_ptr()):
                        raise RuntimeError("Stage binding failed")
                self.stream.wait_stream(caller)
                for value in prepared + outputs + [self.workspace]:
                    value.record_stream(self.stream)
                # Time actual engine execution on its own stream, after the
                # input dependency; the caller span also includes fencing.
                with latency_scope("trt.execute", stream=self.stream, engine=str(self.path)):
                    if not self.context.execute_async_v3(self.stream.cuda_stream):
                        raise RuntimeError("Stage execution failed")
                caller.wait_stream(self.stream)
        with latency_scope("trt.output_cache_cast_bf16", engine=str(self.path)):
            if self.cache_passthrough:
                # Only y crosses back. The cache_out_* tensors stay in binding dtype and
                # are handed straight to the caller's feat_cache, where the next call
                # consumes them with no conversion at either end.
                return (outputs[0].to(torch.bfloat16), *outputs[1:])
            return tuple(value.to(torch.bfloat16) for value in outputs)


def validate_plan_spans(vae, plan) -> dict:
    """Refuse a plan whose recorded module forms differ from the live decoder's.

    A plan built from a DecoderSpanStage bakes the covered modules into the engine:
    a Resample's spatial form (stock ``Upsample+Conv2d`` vs ``SubPixelUpsampleConv``),
    the head. Installing it over a decoder whose modules differ (sub-pixel fold not yet
    installed, a block skipped, the head already absorbed) would still match every
    binding shape and silently run the wrong arithmetic, so the plan's ``spans`` record
    (class names in walk order, from ``describe_span``) must equal the live one verbatim.
    Plans without a ``spans`` record predate spans and may only cover pure ResidualBlock
    groups, exactly what ``ResidualStage`` has always required. Returns the live spans.

    A head that follows skipped blocks is part of the record too (the head entry ends in
    ``:after-skipped:13,14``, see ``describe_span``): a plan carrying it is refused on a
    decoder where those blocks are NOT skipped (the group cannot be resolved -- the live
    decoder would run blocks the engine's ``y`` already stands in for), and a plan whose
    head entry lacks it is refused on a decoder where they ARE skipped (the recorded and
    live entries differ verbatim).
    """
    groups = [parse_group(group) for group in plan["groups"]]
    recorded = plan.get("spans")
    live = {}
    for group in groups:
        key = group_key(group)
        try:
            live[key] = describe_span(vae, group)
        except (IndexError, TypeError) as error:
            raise ValueError(f"Stage plan group {key} cannot be resolved on the live decoder: {error}") from error
        if recorded is None:
            if group_has_head(group) or any(name != "ResidualBlock" for name in live[key]):
                raise ValueError(
                    f"Stage plan has no 'spans' record but group {key} covers {live[key]}; "
                    "only pure ResidualBlock groups may be installed from a plan without spans"
                )
        elif recorded.get(key) != live[key]:
            hint = (
                "install the same decoder op rewrites -- e.g. the sub-pixel fold -- BEFORE "
                "the stage plan, or rebuild the plan for this decoder"
            )
            if (
                group_has_head(group)
                and isinstance(recorded.get(key), list)
                and recorded[key]
                and recorded[key][:-1] == live[key][:-1]
                and (":after-skipped:" in live[key][-1]) != (":after-skipped:" in str(recorded[key][-1]))
            ):
                hint = (
                    "the head of this group follows skipped blocks on exactly one side: a plan "
                    "for a head after skipped blocks must be captured with --skip-blocks over "
                    "every upsamples index after the group and installed with the same "
                    "--skip-decoder-blocks; a plan without that record needs those blocks live"
                )
            raise ValueError(
                f"Stage plan group {key} was built for modules {recorded.get(key)} but the live "
                f"decoder has {live[key]} ({hint})"
            )
    return live


def install_stage_plan(vae, path, *, vae_weights=None):
    """``vae_weights``: the checkpoint the live ``vae`` carries when it is not the stock
    file (a plan built with ``decoder_stages.py --vae-weights``). The plan's
    ``weights_sha256`` must equal that file's hash; the caller is responsible for having
    loaded the same file into ``vae`` (this function cannot verify 500 MB of weights on
    every install). Default: the stock checkpoint, refused on mismatch as always."""
    from benchmarks.pro_quantization_v2_20260918.common import (
        ROOT,
        resolve_path,
        sha256,
    )

    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or plan.get("status") != "complete":
        raise ValueError("Incomplete or unknown stage plan")
    checkpoint = (
        resolve_path(vae_weights)
        if vae_weights
        else ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
    )
    if plan["weights_sha256"] != sha256(checkpoint):
        recorded = plan.get("vae_weights", {}).get("path") or plan.get("weights_path")
        raise ValueError(
            "Stage weights mismatch"
            + (f": the plan was built from {recorded}; pass vae_weights=<that file> and load it "
               "into the VAE" if recorded and vae_weights is None else "")
        )
    source_path = "flash_head/wan/modules/vae.py"
    if plan.get("source_sha256", {}).get(source_path) != sha256(ROOT / source_path):
        raise ValueError(
            "Stage plan was exported from a different Wan VAE implementation"
        )
    # Residual-block skipping (pro_decoder_ops.install_decoder_block_skip) is part of the
    # engine contract, not a runtime toggle: a plan built with `--use-blocks "12"` for the
    # tail group expects upsamples[13] and [14] to be identity at runtime, and a plan built
    # for the full group expects them to run. Either mismatch would install engines whose
    # y output feeds a decoder the calibration never saw -- silently, since every shape
    # still matches. So the plan's recorded skips and the VAE's installed skips must be
    # the same set, checked before anything is allocated or replaced.
    from soulx_rtc.pro_decoder_ops import skipped_decoder_blocks

    installed_skips = skipped_decoder_blocks(vae)
    recorded_skips = sorted({int(index) for index in plan.get("skipped_blocks", [])})
    if installed_skips != recorded_skips:
        raise ValueError(
            f"Stage plan records skipped decoder blocks {recorded_skips} but the VAE has "
            f"{installed_skips} skipped (run.py --skip-decoder-blocks must list exactly the "
            "blocks the plan was built without)"
        )
    groups = [parse_group(group) for group in plan["groups"]]
    overlap = sorted(set(recorded_skips) & {i for group in groups for i in group_indices(group)})
    if overlap:
        raise ValueError(f"Stage plan groups contain skipped blocks {overlap}")
    # A head-inclusive group that ends before upsamples[-1] is legal only over skipped
    # blocks; the plan's own skipped_blocks must name every index after it, so the
    # record is self-consistent before the live decoder is consulted (validate_plan_spans
    # then checks those modules really are SkippedResidualBlock identities and that the
    # head entry says so).
    last_index = len(vae.model.decoder.upsamples) - 1
    for group in groups:
        indices = group_indices(group)
        if group_has_head(group) and indices and indices[-1] < last_index:
            tail = list(range(indices[-1] + 1, last_index + 1))
            missing = sorted(set(tail) - set(recorded_skips))
            if missing:
                raise ValueError(
                    f"Stage plan group {group_key(group)} takes the decoder head but ends before "
                    f"upsamples[{last_index}] while its skipped_blocks {recorded_skips} do not cover "
                    f"{missing}; such a group is legal only when every later block is skipped"
                )
    validate_plan_spans(vae, plan)
    sizes = [
        record["workspace_bytes"]
        for group in plan["stages"].values()
        for record in group.values()
    ]
    if not sizes or any(not isinstance(size, int) or size < 0 for size in sizes):
        raise ValueError("Invalid stage workspace requirements")
    arena = {
        "tensor": torch.empty(max(1, max(sizes)), dtype=torch.uint8, device="cuda"),
        "stream": torch.cuda.Stream(),
        "lock": threading.Lock(),
    }

    # Keep the causal cache in the engines' own binding dtype across calls instead of
    # casting it to BF16 and back 27 times per window. Only meaningful when the bindings
    # are float16; with bf16 bindings the cast is already a no-op. Set
    # SOULX_STAGE_CACHE_PASSTHROUGH=0 to fall back to the historical behaviour without
    # editing code -- the two paths are numerically distinguishable only by one bf16
    # rounding step that passthrough removes.
    binding_dtype = (
        torch.float16
        if plan.get("floating_precision", "bf16") == "fp16"
        else torch.bfloat16
    )
    cache_passthrough = binding_dtype is torch.float16 and os.environ.get(
        "SOULX_STAGE_CACHE_PASSTHROUGH", "1"
    ) not in ("0", "false", "False")
    # Off by default: it trades VRAM (one extra output set per signature) for allocator
    # churn, and VRAM is the binding constraint on running a second instance.
    reuse_outputs = os.environ.get("SOULX_STAGE_REUSE_OUTPUTS", "0") not in (
        "0", "false", "False",
    )

    def factory(indices, stage):
        records = plan["stages"][group_key(indices)]
        engines = {}
        for signature, record in records.items():
            if record["precision"] != plan["precision"]:
                raise ValueError("Stage precision mismatch")
            floating = plan.get("floating_precision", "bf16")
            if (
                floating not in ("bf16", "fp16")
                or record.get("floating_precision", "bf16") != floating
            ):
                raise ValueError("Stage floating precision mismatch")
            if record.get("binding_dtype", "bfloat16") != (
                "float16" if floating == "fp16" else "bfloat16"
            ):
                raise ValueError("Stage floating binding mismatch")
            checks = record.get("build_sample_comparison", [])
            if len(checks) != stage.cache_count + 1 or not all(
                item.get("finite") is True for item in checks
            ):
                raise ValueError(
                    "Stage engine lacks finite output/cache execution evidence"
                )
            layers = resolve_path(record["layers_path"])
            if sha256(layers) != record["layers_sha256"]:
                raise ValueError("Stage precision inspector hash mismatch")
            if plan["precision"] == "int8":
                from benchmarks.pro_30fps_20260919.decoder_stages import (
                    int8_convolutions,
                )

                evidence = int8_convolutions(layers.read_text())
                if (
                    not record.get("int8_convolutions_verified")
                    or evidence["int8_convolutions"] < stage.cache_count
                    or len(record.get("quantized_modules", [])) != stage.cache_count
                ):
                    raise ValueError("Missing INT8 computation proof")
            engines[signature] = TensorRTStage(
                resolve_path(record["path"]),
                record,
                arena=arena,
                cache_passthrough=cache_passthrough,
                reuse_outputs=reuse_outputs,
            )

        def execute(*values):
            signature = tensor_signature(values)
            if signature not in engines:
                raise ValueError(f"No verified stage signature {signature}")
            return engines[signature](*values)

        return execute

    result = install_stages(
        vae, groups, factory, cache_dtype=binding_dtype if cache_passthrough else None
    )
    result.update(
        {
            "plan": str(path),
            "plan_sha256": sha256(path),
            "precision": plan["precision"],
            "workspace_bytes": max(sizes),
            "unshared_workspace_bytes": sum(sizes),
            "workspace_execution": "one shared stream and host enqueue lock; serialized engine execution",
            "floating_precision": plan.get("floating_precision", "bf16"),
            "caller_cache_dtype": (
                str(binding_dtype).replace("torch.", "")
                if cache_passthrough
                else "bfloat16"
            ),
            "cache_passthrough": cache_passthrough,
            "reuse_outputs": reuse_outputs,
            "skipped_blocks": recorded_skips,
            "subpixel_resample": [int(i) for i in plan.get("subpixel_resample", [])],
            "vae_weights": str(checkpoint),
        }
    )
    return result

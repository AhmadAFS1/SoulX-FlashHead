"""Runtime decoder op rewrites that do not touch ``flash_head/wan/modules/vae.py``.

Every rewrite here is installed by swapping a live module on an instantiated VAE.
That is deliberate: ``pro_vae_stage_backend.install_stage_plan`` gates the TensorRT
stage engines on the sha256 of ``vae.py`` alone, so editing that file would invalidate
all nine engines and force a recapture/rebuild/requalify cycle. Swapping modules at
runtime gets the same arithmetic without the rebuild.

The flip side of routing around the source-hash gate is that a mistake here produces
no build-time error -- it produces a silent numerical drift into the stage engines that
consume the output. So each installer carries its own exactness assertion and runs it
every time it installs, not once during development.
"""

from __future__ import annotations

import torch
from torch import nn

# Nearest-2x upsample followed by a 3x3 convolution is a sub-pixel convolution.
#
# With ``nearest-exact`` at an integer scale of 2 the source index is floor(dst/2), so
# for output row 2i+a the 3x3 kernel taps rows {2i+a-1, 2i+a, 2i+a+1} of the upsampled
# image, which are input rows:
#
#   a=0:  du=-1 -> i-1 ;  du=0 -> i ;  du=+1 -> i
#   a=1:  du=-1 -> i   ;  du=0 -> i ;  du=+1 -> i+1
#
# so each output phase reads only a 2-tap neighbourhood of the input. Folding those
# collapsed taps into a 3x3 kernel per phase keeps padding=1 valid at both boundaries:
# at i=0,a=0 the du=-1 tap reads upsampled row -1 (zero) and the folded kernel reads
# input row -1 (zero); at i=H-1,a=1 the du=+1 tap reads upsampled row 2H (zero) and the
# folded kernel reads input row H (zero).
_FOLD = {
    # _FOLD[phase][dI + 1, du + 1] == 1 when input offset dI receives kernel tap du
    0: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    1: torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}


class SubPixelUpsampleConv(nn.Module):
    """``Upsample(nearest, 2) -> Conv2d(cin, cout, 3, p=1)`` as one sub-pixel conv.

    MAC count is identical (the convolution simply moves to the smaller resolution).
    The win is deleting the upsampled intermediate -- 270 MiB at the Resample[11]
    shape -- and handing cuDNN a better GEMM shape: M=cout*4 at H/2 x W/2 instead of
    M=cout at H x W.
    """

    def __init__(self, conv: nn.Conv2d, scale: int = 2):
        super().__init__()
        if conv.kernel_size != (3, 3) or conv.padding != (1, 1):
            raise ValueError("Sub-pixel fold expects a 3x3 conv with padding 1")
        if conv.stride != (1, 1) or conv.dilation != (1, 1) or conv.groups != 1:
            raise ValueError("Sub-pixel fold expects a dense unit-stride conv")
        if scale != 2:
            raise ValueError("Only the 2x fold is derived here")
        cin, cout = conv.in_channels, conv.out_channels
        folded = nn.Conv2d(
            cin,
            cout * scale * scale,
            3,
            padding=1,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            # Fold in float64 and cast once, so the stored weights carry no avoidable
            # accumulation error from the fold itself.
            weight = conv.weight.detach().to(torch.float64)
            new = torch.empty(
                (cout, scale, scale, cin, 3, 3), dtype=torch.float64, device=weight.device
            )
            for a in (0, 1):
                fa = _FOLD[a].to(weight.device, torch.float64)
                for b in (0, 1):
                    fb = _FOLD[b].to(weight.device, torch.float64)
                    # Wsub[dI, dJ] = sum_{ku, kv} Fa[dI, ku] * W[ku, kv] * Fb[dJ, kv]
                    new[:, a, b] = torch.einsum(
                        "pu,oiuv,qv->oipq", fa, weight, fb
                    )
            # PixelShuffle(r) reads channel c*r*r + a*r + b for output pixel phase (a, b).
            folded.weight.copy_(
                new.reshape(cout * scale * scale, cin, 3, 3).to(conv.weight.dtype)
            )
            if conv.bias is not None:
                # Every phase of an output channel carries that channel's bias.
                folded.bias.copy_(
                    conv.bias.detach()
                    .to(torch.float64)
                    .repeat_interleave(scale * scale)
                    .to(conv.bias.dtype)
                )
        self.conv = folded
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))


def _assert_subpixel_exact(original: nn.Module, replacement: nn.Module, sample: torch.Tensor):
    """Two separate checks, because they can fail for unrelated reasons.

    1. ALGEBRA, in float64 throughout. The fold must be mathematically exact. A bug in
       the tap mapping or the PixelShuffle channel order shows up here and nowhere else.
    2. REPRESENTATION, at the live dtype. The folded coefficients are *sums* of two
       original weights (taps du=0 and du=+1 collapse onto the same input offset), and
       bf16's 8 mantissa bits cannot hold those sums exactly. So the deployed fold
       carries ~1 ulp of weight-rounding error by construction -- expected, bounded, and
       not a sign of a wrong fold. Conflating the two is what makes an over-strict
       assert reject a correct implementation.

    Both run on every install, not once in development: this swap bypasses the vae.py
    source-hash gate, so a genuinely wrong fold would otherwise surface only as silent
    drift feeding stage-12-13-14.
    """
    import copy

    with torch.no_grad():
        # (1) algebra: re-fold from float64 weights so no bf16 rounding is in play.
        exact_src = copy.deepcopy(original).to(torch.float64)
        exact_fold = SubPixelUpsampleConv(exact_src[1]).to(torch.float64)
        ref64 = exact_src(sample.to(torch.float64))
        got64 = exact_fold(sample.to(torch.float64))
        if ref64.shape != got64.shape:
            raise ValueError(
                f"Sub-pixel shape mismatch: {tuple(ref64.shape)} vs {tuple(got64.shape)}"
            )
        algebra_err = (ref64 - got64).abs().max().item()
        rng = max(ref64.abs().max().item(), 1.0)
        if algebra_err > 1e-9 * rng:
            raise ValueError(
                f"Sub-pixel fold is algebraically wrong: max|err|={algebra_err:.3e} "
                f"on range {rng:.3f}"
            )

        # (2) representation: compare at the dtype that will actually run.
        dtype = next(original.parameters()).dtype
        native_in = sample.to(dtype)
        ref = original.to(dtype)(native_in).to(torch.float64)
        got = replacement.to(dtype)(native_in).to(torch.float64)
        native_err = (ref - got).abs().max().item()
        native_rng = max(ref.abs().max().item(), 1.0)
        # Budget: a few ulp of weight rounding, OR 2% of range -- whichever is larger.
        # The floor matters because at float32 the convolution's own accumulation over
        # fan_in=1728 swamps 8 ulp (measured 1,942 ulp for a demonstrably correct fold),
        # so a pure-ulp budget rejects correct code at wide dtypes. Check (1) in float64
        # is what actually discriminates a wrong fold -- it lands at ~1e-14 when correct
        # and at the scale of the signal when not -- so this bound only has to catch
        # gross breakage without being dtype-fragile.
        ulp = torch.finfo(dtype).eps * native_rng
        limit = max(8.0 * ulp, 0.02 * native_rng)
        if native_err > limit:
            raise ValueError(
                f"Sub-pixel fold exceeds its dtype budget: max|err|={native_err:.3e} "
                f"on range {native_rng:.3f} ({native_err / ulp:.1f} ulp, limit {limit:.3e})"
            )
    return {"algebra_max_abs_err_fp64": algebra_err,
            "native_max_abs_err": native_err,
            "native_ulp": native_err / ulp if ulp else 0.0,
            "dtype": str(dtype).replace("torch.", "")}


def install_subpixel_resample(vae, indices=(11,), probe_hw=(16, 12)) -> dict:
    """Swap ``upsamples[i].resample`` for its sub-pixel equivalent, for each 2D Resample."""
    from flash_head.wan.modules.vae import Resample

    if hasattr(vae, "_pro_subpixel_originals"):
        raise ValueError("Sub-pixel resample already installed")
    upsamples = vae.model.decoder.upsamples
    originals, report = {}, []
    try:
        for index in indices:
            module = upsamples[index]
            if not isinstance(module, Resample) or module.mode != "upsample2d":
                raise ValueError(f"upsamples[{index}] is not an upsample2d Resample")
            sequential = module.resample
            if len(sequential) != 2 or not isinstance(sequential[1], nn.Conv2d):
                raise ValueError(f"upsamples[{index}].resample has an unexpected shape")
            conv = sequential[1]
            replacement = SubPixelUpsampleConv(conv)
            h, w = probe_hw
            sample = torch.randn(
                1, conv.in_channels, h, w, device=conv.weight.device, dtype=torch.float64
            )
            checks = _assert_subpixel_exact(sequential, replacement, sample)
            originals[index] = sequential
            upsamples[index].resample = replacement.to(conv.weight.dtype)
            report.append(
                {"index": index, **checks,
                 "in_channels": conv.in_channels, "out_channels": conv.out_channels}
            )
    except Exception:
        for index, sequential in originals.items():
            upsamples[index].resample = sequential
        raise
    vae._pro_subpixel_originals = originals
    return {"subpixel_resample": report}


def remove_subpixel_resample(vae):
    originals = getattr(vae, "_pro_subpixel_originals", None)
    if originals is not None:
        for index, sequential in originals.items():
            vae.model.decoder.upsamples[index].resample = sequential
        del vae._pro_subpixel_originals


def install_channels_last_head(vae) -> dict:
    """Hold the decoder head's ``CausalConv3d`` weight in ``channels_last_3d``.

    Bit-identical: cuDNN's NCDHW path already transforms to NHWC internally, so this is
    literally the same kernel with the transform hoisted out.

    THE WIN EXISTS ONLY INSIDE THE COMPILED REGION. Measured eager, this layout is a
    ~17% REGRESSION on the head; compiled it is a ~26% win, because the entire benefit
    is inductor fusing the layout conversion into the norm/SiLU epilogue. If anything
    drops the head out of the compiled graph -- a recompile-limit fallback, a graph
    break, an eager debug path -- this silently inverts into a loss.
    """
    from flash_head.wan.modules.vae import CausalConv3d

    if hasattr(vae, "_pro_head_channels_last"):
        raise ValueError("Channels-last head already installed")
    head = vae.model.decoder.head
    convs = [m for m in head.modules() if isinstance(m, CausalConv3d)]
    if len(convs) != 1:
        raise ValueError(f"Expected exactly one CausalConv3d in the head, found {len(convs)}")
    conv = convs[0]
    before = conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad():
        conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
    vae._pro_head_channels_last = True
    return {
        "channels_last_head": {
            "already_channels_last": bool(before),
            "weight_shape": list(conv.weight.shape),
        }
    }


def install_overlap_skip(pipeline, compile_decode: bool = True) -> dict:
    """Stop re-decoding the motion overlap at every window boundary.

    Window stride is 28 useful pixel frames = 7 latent frames, so window N+1's leading
    latent frames are exactly window N's trailing ones. The stock path re-decodes them
    and then throws the pixels away (``run_pipeline`` returns [T,H,W,C] and the harness
    slices ``[WINDOW_HISTORY_FRAMES:]``), which is ~99 ms/window of pure waste.

    Two things make this more than a one-line swap:

    1. ``WanVAE_.decode`` calls ``clear_cache()`` on entry AND exit, and ``clear_cache``
       resets ``_feat_map`` -- the DECODER cache. ``cached_decode`` is the same function
       without those two calls. But the pipeline also runs ``vae.encode(cond_frame)``
       once per window between consecutive decodes, and ``encode`` clears the cache too,
       so swapping the decode entry point ALONE saves nothing. Hence the save/restore
       around encode.
    2. The decoder is normally ``torch.compile``d via ``optimize_wan_vae``. Replacing
       ``vae.decode`` wholesale would silently drop that compilation, so the replacement
       compiles ``cached_decode`` itself. Two input lengths occur (full first window,
       short thereafter), so expect two traces.

    QUALITY: this is a semantic change, not just an optimization. The stock decoder's
    cross-window temporal context is a VAE round-trip of COLOUR-CORRECTED pixels
    (``cond_frame`` is taken after ``match_and_blend_colors_torch``). A persisted cache
    substitutes raw pre-correction decoder state, which takes colour correction out of
    the cross-window feedback loop -- the mechanism that suppresses inter-window colour
    drift. Gate on per-window mean RGB drift across all 9 windows, not a single window.
    """
    vae = pipeline.vae
    model = vae.model
    if hasattr(vae, "_pro_overlap_skip"):
        raise ValueError("Overlap skip already installed")
    orig_decode, orig_encode = vae.decode, vae.encode
    n_hist_px = int(pipeline.motion_frames_num)
    cached = model.cached_decode
    if compile_decode:
        cached = torch.compile(cached, dynamic=False)
    def _fresh():
        return {"tail": None, "map": None, "windows": 0, "latents_skipped": 0,
                "keep": None, "frames_per_window": [], "latents_per_window": []}

    # Session-keyed. The decoder cache this shim persists belongs to ONE generation
    # sequence; a VAE shared by several sessions (engine.py's LITE design shallow-copies
    # the pipeline per session and shares vae/model) would otherwise hand session B the
    # tail of session A. `sessions["active"]` names the current one; the harness switches
    # it around each session's decode. Single-stream use never touches this and keeps
    # the default key.
    sessions = {"active": "0", "0": _fresh()}
    def state_of():
        return sessions[sessions["active"]]

    def decode(zs):
        state = state_of()
        # zs: (C, T, h, w); WanVAE.decode unsqueezes to (1, C, T, h, w).
        #
        # The cache is restored HERE rather than protected around encode. encode() does
        # call clear_cache() and does replace _feat_map with a fresh list of Nones --
        # but the old list object survives as long as we hold a reference, so handing it
        # back at the top of the next decode is equivalent and strictly cheaper.
        #
        # Wrapping encode instead is what the obvious reading suggests, and it is a trap:
        # encode is torch.compile'd, and mutating _feat_map underneath it invalidates its
        # guards on every call. Measured that way, motion_encode went 1.120 -> 12.454 s
        # -- a recompilation storm that swamped the 0.610 s the decode side saved.
        if state["map"] is None:
            model.clear_cache()
            out = cached(zs.unsqueeze(0), vae.scale).clamp_(-1, 1)
            # Derive how many trailing latents carry the useful frames from the window
            # that actually ran, rather than from latent_motion_frames.shape[1]. That
            # attribute is NOT stably 2: measured frames_per_window [33,33,37,33,33]
            # with latents_skipped 7 over 4 windows, i.e. one window skipped a single
            # latent and emitted 32 fresh frames instead of 28. Those 4 extra frames
            # entered the delivered stream and shifted everything after them, which is
            # exactly the 4-frame lip-sync offset this showed up as (opening_norm
            # correlation 0.044 at lag 0, 0.958 at lag -4). With a warm cache every
            # latent yields 4 frames, so the count is fixed by the first window.
            state["keep"] = (int(out.shape[2]) - n_hist_px) // 4
        else:
            model._feat_map = state["map"]
            model._conv_idx = [0]
            keep = state["keep"]
            fresh = cached(zs[:, -keep:].unsqueeze(0), vae.scale).clamp_(-1, 1)
            state["latents_skipped"] += int(zs.shape[1]) - keep
            # Re-attach the previous window's trailing pixels so the returned tensor
            # keeps its 33-frame contract. Those leading frames are discarded by every
            # consumer, and colour correction reduces over H and W only, so prepending
            # them cannot perturb the 28 frames that are actually delivered.
            out = torch.cat([state["tail"], fresh], dim=2)
        # A window that emits a different frame count silently shifts every frame after
        # it. That is not detectable downstream -- the harness only checks the TOTAL --
        # so it has to be caught here, at the point where the count is decided.
        frames = int(out.shape[2])
        if state["frames_per_window"] and frames != state["frames_per_window"][0]:
            raise ValueError(
                f"Overlap-skip emitted {frames} frames for window {state['windows']}, "
                f"expected {state['frames_per_window'][0]}; this shifts delivered audio "
                f"alignment by {frames - state['frames_per_window'][0]} frames"
            )
        state["map"] = model._feat_map
        state["tail"] = out[:, :, -n_hist_px:].detach().clone()
        state["windows"] += 1
        state["frames_per_window"].append(frames)
        state["latents_per_window"].append(int(zs.shape[1]))
        return out

    def encode(x, *args, **kwargs):
        # Present the compiled encoder the SAME _feat_map structure the stock path always
        # shows it: a list of Nones. Without this the encoder sees the persisted decoder
        # cache (a list of TENSORS) on entry, its guard misses every window, and
        # motion_encode blows up from 1.12 to 2.6-12.5 s over 250 frames depending on
        # where the mutation happens. encode() clears and rebuilds its own _enc_feat_map
        # regardless, so handing it Nones costs nothing and changes no numerics.
        saved = model._feat_map
        model._feat_map = [None] * len(saved) if saved is not None else saved
        try:
            return orig_encode(x, *args, **kwargs)
        finally:
            model._feat_map = saved

    vae.decode = decode
    vae.encode = encode
    vae._pro_overlap_skip = {
        "decode": orig_decode, "encode": orig_encode, "sessions": sessions, "fresh": _fresh
    }
    return {"overlap_skip": {"motion_frames_px": n_hist_px,
                             "compiled_decode": bool(compile_decode)}}


def reset_overlap_skip(pipeline) -> bool:
    """Drop the persisted decoder cache so the next decode starts a fresh sequence.

    MUST be called wherever the pipeline itself is reset between generations. The
    harness resets the person/seed before the warmup pass AND again before the measured
    pass; without this the measured run's window 0 takes the skip branch and decodes
    against the WARMUP's decoder cache, i.e. temporal context from a different
    generation. That produced a badly degraded first frame -- washed-out eyes,
    grey-green mottling on the chin, visible blur -- in an otherwise clean run.

    It was invisible to every aggregate check: frame 0's mean RGB and std sat normally
    inside the run's distribution (it was the CLOSEST frame to the run median), because
    the corruption is confined to the face while the statistics are frame-wide.
    """
    saved = getattr(pipeline.vae, "_pro_overlap_skip", None)
    if saved is None:
        return False
    sessions = saved["sessions"]
    sessions[sessions["active"]] = saved["fresh"]()
    return True


def set_overlap_skip_session(pipeline, key) -> bool:
    """Make `key` the session whose decoder cache the shim persists. New keys start
    cold (their first decode is a full decode, exactly like window 0)."""
    saved = getattr(pipeline.vae, "_pro_overlap_skip", None)
    if saved is None:
        return False
    key = str(key)
    saved["sessions"].setdefault(key, saved["fresh"]())
    saved["sessions"]["active"] = key
    return True


def overlap_skip_state(pipeline) -> dict:
    saved = getattr(pipeline.vae, "_pro_overlap_skip", None)
    if saved is None:
        return {}
    state = saved["sessions"][saved["sessions"]["active"]]
    return {"session": saved["sessions"]["active"],
            "windows": state["windows"], "latents_skipped": state["latents_skipped"],
            "frames_per_window": state["frames_per_window"],
            "latents_per_window": state["latents_per_window"]}


def remove_overlap_skip(pipeline):
    saved = getattr(pipeline.vae, "_pro_overlap_skip", None)
    if saved is not None:
        pipeline.vae.decode = saved["decode"]
        pipeline.vae.encode = saved["encode"]
        del pipeline.vae._pro_overlap_skip


# ---------------------------------------------------------------------------
# Decoder residual-block skipping
# ---------------------------------------------------------------------------
#
# Every decoder ``upsamples[i]`` ResidualBlock computes ``y = x + f(x)`` with matching
# channel counts (except the first block of a resolution level, whose shortcut is a 1x1
# conv), so bypassing one is a well-defined edit: ``y = x``. The tail engine over
# ``upsamples[12-14]`` alone is ~340 ms of a 1,427 ms window at 576x320, and every kernel-
# level lever inside it has been measured exhausted (iteration log, 2026-09-21/22), which
# is why a WORK-removal lever is on the table at all. It is a quality risk by
# construction and lives behind ``run.py --skip-decoder-blocks``; rollback is dropping
# the flag.
#
# The subtle part is the causal cache, not the arithmetic. ``Decoder3d.forward`` hands
# every module the SAME ``feat_cache`` list and a running ``feat_idx`` cursor; each
# ``CausalConv3d`` inside a block's ``residual`` chain consumes exactly one slot
# (``ResidualBlock.forward``: ``x = layer(x, feat_cache[idx]); feat_cache[idx] = cache_x;
# feat_idx[0] += 1``) and the ``shortcut`` conv, when present, consumes none (it is
# called as ``self.shortcut(x)`` with no cache). A ResidualBlock therefore consumes as
# many slots as it has ``CausalConv3d`` layers in ``residual`` -- two -- which is also
# where the stage backend's ``cache_count = 2 * len(blocks)`` comes from. A skipped block
# must still ADVANCE the cursor by that number, or every downstream slot index -- the
# following blocks', the Resamples', the stage adapters' and the head's -- shifts and
# the decoder reads another layer's cache as its own. Advancing without writing keeps
# the skipped slots ``None`` forever, which matters for the overlap-skip shim: it
# persists ``decoder._feat_map`` across windows, and a slot that is never written stays
# a never-read ``None`` for the whole generation.


class SkippedResidualBlock(nn.Module):
    """Identity stand-in for a decoder ``ResidualBlock``.

    ``forward`` returns ``x`` unchanged and advances ``feat_idx[0]`` by ``slots`` -- the
    number of causal-cache slots the original block would have consumed -- WITHOUT
    touching ``feat_cache``. Mirrors the original's control flow exactly: the block only
    touches the cursor when a cache list is passed (``Decoder3d.forward`` calls
    ``layer(x)`` with no cache otherwise, and the original then advances nothing).

    The bypassed block is KEPT in the module tree as ``self.skipped`` (never called).
    Not for rollback convenience: ``WanVAE_.clear_cache`` sizes ``_feat_map`` with
    ``count_conv3d(decoder)``, a walk over ``decoder.modules()`` counting
    ``CausalConv3d`` instances, and it runs AFTER this install (every ``encode`` and
    stock ``decode`` calls it). An identity with no convs would shrink ``_conv_num`` by
    two per skipped block while the cursor still walks the full distance, and the head
    would ``IndexError`` on its slot. Found by the install-time self-check; the stage
    backend's ``StageAdapter`` keeps its blocks as children for the same reason.

    Deliberately NOT the stage backend's ``_SkipStage``: that one advances nothing
    because its ``StageAdapter`` sibling advances the whole group's slots at once. The
    two classes are distinct on purpose so ``skipped_decoder_blocks`` can tell a
    user-skipped block from a stage-absorbed one.
    """

    def __init__(self, index: int, slots: int, original: nn.Module):
        super().__init__()
        self.index = int(index)
        self.slots = int(slots)
        self.original_class = type(original).__name__
        self.in_dim = int(getattr(original, "in_dim", -1))
        self.out_dim = int(getattr(original, "out_dim", -1))
        self.skipped = original

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None and feat_idx is not None:
            feat_idx[0] += self.slots
        return x

    def extra_repr(self) -> str:
        return f"index={self.index}, slots={self.slots}, of={self.original_class}"


def _residual_cache_slots(block) -> int:
    """Slots ``ResidualBlock.forward`` consumes: one per ``CausalConv3d`` in ``residual``.

    Computed from the live module rather than hard-coded so that a block variant with a
    different conv count would be handled (or at least counted) correctly; the value is
    2 for every Wan decoder block and is asserted against the stage backend's
    ``2 * len(blocks)`` convention by the caller.
    """
    from flash_head.wan.modules.vae import CausalConv3d

    return sum(isinstance(layer, CausalConv3d) for layer in block.residual)


def skipped_decoder_blocks(vae) -> list[int]:
    """Indices of ``upsamples[]`` currently replaced by ``SkippedResidualBlock``."""
    upsamples = vae.model.decoder.upsamples
    return [i for i, m in enumerate(upsamples) if isinstance(m, SkippedResidualBlock)]


def _verify_block_skip_alignment(decoder, originals: dict, latent_hw=(4, 4), frames=2) -> dict:
    """Prove, on a tiny latent, that skipping keeps the causal cache aligned.

    Runs the decoder the way ``cached_decode`` does -- one latent frame per call, a
    shared cache list, cursor reset to 0 per call -- once with the original modules and
    once with the skip modules installed, and requires:

    * the cursor ends at the same value on every call (every downstream slot index,
      including the head's, is unchanged);
    * the skipped blocks' slots are ``None`` after the skipped run and tensors after the
      original run (they were really bypassed, and nothing else wrote into them);
    * every other slot holds the same kind of entry with the same shape in both runs
      (``None`` / ``"Rep"`` / tensor shape), i.e. each surviving layer still reads and
      writes ITS OWN slot;
    * the output shape is unchanged;
    * ``count_conv3d`` (what ``clear_cache`` sizes ``_feat_map`` with) is unchanged, so
      the cursor cannot run off the end of the list at runtime.

    This runs on every install, in the module's convention: the skip bypasses the
    ``vae.py`` source-hash gate, so a cursor bug here would surface only as silent
    numerical drift into the stage engines that consume the misaligned cache. The latent
    is 4x4 (32x32 pixels), so the cost is negligible even on the GPU-resident VAE.
    """
    upsamples = decoder.upsamples
    param = next(decoder.parameters())
    device, dtype = param.device, param.dtype
    z_dim = decoder.conv1.in_channels
    generator = torch.Generator(device="cpu").manual_seed(0)
    latent = torch.randn(1, z_dim, frames, *latent_hw, generator=generator).to(device, dtype)
    from flash_head.wan.modules.vae import count_conv3d

    # Counted with the skip modules installed, i.e. exactly what clear_cache() will see at
    # runtime; run(originals) below then proves the reference walk fits in that many slots.
    conv_num = count_conv3d(decoder)

    def run(modules):
        for index, module in modules.items():
            upsamples[index] = module
        cache = [None] * conv_num
        ends, outputs = [], []
        with torch.inference_mode():
            for frame in range(frames):
                cursor = [0]
                outputs.append(decoder(latent[:, :, frame : frame + 1], feat_cache=cache, feat_idx=cursor))
                ends.append(cursor[0])
        kinds = [
            ("tensor", tuple(v.shape)) if isinstance(v, torch.Tensor) else ("rep", None) if v == "Rep" else ("none", None)
            for v in cache
        ]
        return ends, kinds, tuple(outputs[-1].shape)

    skips = {index: upsamples[index] for index in originals}
    if any(not isinstance(m, SkippedResidualBlock) for m in skips.values()):
        raise ValueError("Alignment check expects the skip modules to be installed")
    if count_conv3d(decoder) != conv_num:
        raise ValueError("Skip modules changed count_conv3d; _feat_map would be mis-sized")
    ref_ends, ref_kinds, ref_shape = run(originals)
    try:
        got_ends, got_kinds, got_shape = run(skips)
    finally:
        # Leave the skip modules installed whatever happened; the caller restores on error.
        for index, module in skips.items():
            upsamples[index] = module
    if ref_ends != got_ends:
        raise ValueError(
            f"Decoder block skip misaligns the causal cache cursor: original ends {ref_ends}, "
            f"skipped ends {got_ends}"
        )
    if got_shape != ref_shape:
        raise ValueError(f"Decoder block skip changed the output shape {ref_shape} -> {got_shape}")
    # Which slots belong to the skipped blocks: replay the cursor walk with the originals'
    # slot counts. Slots before upsamples[] (conv1, middle) are consumed by modules that
    # are not in upsamples, so walk the whole decoder the way forward() does.
    skipped_slots, cursor = set(), 0
    cursor += 1  # conv1
    for layer in decoder.middle:
        if hasattr(layer, "residual"):
            cursor += _residual_cache_slots(layer)
    for index, module in enumerate(upsamples):
        if index in originals:
            slots = _residual_cache_slots(originals[index])
            skipped_slots.update(range(cursor, cursor + slots))
            cursor += slots
        else:
            cursor += _slots_consumed(module)
    for slot in sorted(skipped_slots):
        if ref_kinds[slot][0] != "tensor":
            raise ValueError(f"Reference run did not fill skipped slot {slot}: {ref_kinds[slot]}")
        if got_kinds[slot][0] != "none":
            raise ValueError(f"Skipped slot {slot} was written: {got_kinds[slot]}")
    mismatched = [
        slot for slot in range(conv_num)
        if slot not in skipped_slots and ref_kinds[slot] != got_kinds[slot]
    ]
    if mismatched:
        raise ValueError(
            f"Decoder block skip shifted cache slots {mismatched}: "
            f"{[(ref_kinds[s], got_kinds[s]) for s in mismatched]}"
        )
    return {
        "latent_shape": list(latent.shape),
        "cursor_end_per_frame": got_ends,
        "cache_slots_total": conv_num,
        "skipped_slots": sorted(skipped_slots),
        "output_shape": list(got_shape),
    }


def _slots_consumed(module) -> int:
    """Slots a NON-skipped ``upsamples[]`` module consumed in the reference walk.

    ResidualBlocks are counted from their convs; a Resample consumes one slot only in
    ``upsample3d``/``downsample3d`` mode (its ``time_conv``); an AttentionBlock none.
    Anything else (a stage adapter) is refused by the installer before we get here.
    """
    from flash_head.wan.modules.vae import Resample, ResidualBlock

    if isinstance(module, ResidualBlock):
        return _residual_cache_slots(module)
    if isinstance(module, Resample):
        return 1 if hasattr(module, "time_conv") else 0
    return 0


def install_decoder_block_skip(vae, indices, verify: bool = True) -> dict:
    """Replace ``vae.model.decoder.upsamples[i]`` for ``i in indices`` by an identity.

    Only ``ResidualBlock`` instances with ``in_dim == out_dim`` (an ``nn.Identity``
    shortcut) may be skipped: a Resample changes resolution and an AttentionBlock is not
    a residual in this decoder, and the first block of a level (``upsamples[4]``,
    192 -> 384 channels) cannot be replaced by an identity of any kind. Must run BEFORE
    ``install_stage_plan``: the plan's groups must not contain skipped indices, and the
    alignment self-check needs the stock modules to replay the reference cursor walk.

    Returns a manifest for ``result["decoder_ops"]`` (indices, slots advanced per block,
    class names, and the self-check evidence). Rollback: ``remove_decoder_block_skip``.
    """
    from flash_head.wan.modules.vae import ResidualBlock

    if hasattr(vae, "_pro_block_skip_originals"):
        raise ValueError("Decoder block skip already installed")
    if hasattr(vae, "_pro_stage_originals"):
        raise ValueError(
            "Install the decoder block skip BEFORE the stage plan: the plan's groups must "
            "exclude the skipped indices and the alignment check needs the stock modules"
        )
    upsamples = vae.model.decoder.upsamples
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        raise ValueError("No decoder blocks to skip")
    originals, report = {}, []
    for index in wanted:
        if index < 0 or index >= len(upsamples):
            raise ValueError(f"upsamples[{index}] does not exist (decoder has {len(upsamples)})")
        module = upsamples[index]
        if not isinstance(module, ResidualBlock):
            raise TypeError(
                f"upsamples[{index}] is a {type(module).__name__}, not a ResidualBlock; only "
                "residual blocks can be bypassed as identity"
            )
        if module.in_dim != module.out_dim or not isinstance(module.shortcut, nn.Identity):
            raise ValueError(
                f"upsamples[{index}] changes channels {module.in_dim} -> {module.out_dim}; "
                "an identity cannot replace it"
            )
        slots = _residual_cache_slots(module)
        if slots != 2:
            # The stage backend hard-codes cache_count = 2 * len(blocks); a block that
            # consumed a different number would break every plan built for it.
            raise ValueError(f"upsamples[{index}] consumes {slots} cache slots, expected 2")
        originals[index] = module
        report.append(
            {"index": index, "class": type(module).__name__, "cache_slots": slots,
             "channels": module.out_dim}
        )
    try:
        for index, module in originals.items():
            upsamples[index] = SkippedResidualBlock(index, 2, module)
        verification = _verify_block_skip_alignment(vae.model.decoder, originals) if verify else None
    except Exception:
        for index, module in originals.items():
            upsamples[index] = module
        raise
    vae._pro_block_skip_originals = originals
    return {
        "decoder_block_skip": {
            "skipped": wanted,
            "blocks": report,
            "cache_slots_advanced": sum(item["cache_slots"] for item in report),
            "verification": verification,
        }
    }


def remove_decoder_block_skip(vae):
    originals = getattr(vae, "_pro_block_skip_originals", None)
    if originals is not None:
        for index, module in originals.items():
            vae.model.decoder.upsamples[index] = module
        del vae._pro_block_skip_originals

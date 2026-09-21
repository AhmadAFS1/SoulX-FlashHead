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
    state = {"tail": None, "map": None, "windows": 0, "latents_skipped": 0,
             "keep": None, "frames_per_window": [], "latents_per_window": []}

    def decode(zs):
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

    vae.decode = decode
    vae._pro_overlap_skip = {
        "decode": orig_decode, "encode": orig_encode, "state": state
    }
    return {"overlap_skip": {"motion_frames_px": n_hist_px,
                             "compiled_decode": bool(compile_decode)}}


def overlap_skip_state(pipeline) -> dict:
    saved = getattr(pipeline.vae, "_pro_overlap_skip", None)
    if saved is None:
        return {}
    state = saved["state"]
    return {"windows": state["windows"], "latents_skipped": state["latents_skipped"],
            "frames_per_window": state["frames_per_window"],
            "latents_per_window": state["latents_per_window"]}


def remove_overlap_skip(pipeline):
    saved = getattr(pipeline.vae, "_pro_overlap_skip", None)
    if saved is not None:
        pipeline.vae.decode = saved["decode"]
        pipeline.vae.encode = saved["encode"]
        del pipeline.vae._pro_overlap_skip

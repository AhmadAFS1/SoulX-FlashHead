# Reusable SoulX head movement: route survey and feasibility verdict

Date: 2026-09-22
Scope: SoulX-FlashHead, LTX 2.3 (ComfyUI), MuseTalk character factory + realtime engine.
Status: synthesis of five parallel code audits, each adversarially re-verified. Corrections from the
adversarial pass are used in place of the original claims throughout.

## The question

> "I am considering somehow making a reusable head movement copy of the SoulXFlashHead. Is it possible
> to utilize a base video from SoulXFlashHead? We'd utilize maybe pro model reference base avatar
> movements and then have LTX 2.3 generate that same movement, is that possible? I want that in our
> generation pipeline, the talking head should match soulxflash of seed 50."

## Verdict

Yes for the useful half, no for the literal half — and the useful half already ran, today.

- **Already done.** `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/` took the saved SoulX PRO
  seed-50 render, extracted a 241-frame DWPose body+face keypoint sequence on CPU, and drove LTX 2.3
  through the official Union Control IC-LoRA to produce a 240-frame 10.000 s render of a **different**
  identity. Image-plane agreement with the donor: eye-x r=0.9945, eye-y r=0.9949, roll r=0.9690, with
  0/240 missing face detections on both sides (`A1/review/metrics.json`). So SoulX head motion **is**
  extractable as a portable object and **has** been transplanted.
- **Not possible as stated.** "The talking head should match SoulXFlash of seed 50" is not a
  well-defined target. Seed 50 is a noise schedule, not a motion trajectory. There is no pose, keypoint,
  expression, 3DMM or motion-code representation anywhere inside SoulX.
- **The reusable object is the extracted trajectory, not the seed.** It already exists on disk:
  `A1/inputs/dwpose-raw.npy` (241 frames) plus `A1/inputs/control-validity.json`.

## Correction of the premise: what "seed 50" actually is

Read from source, not from prose:

- The DiT forward exposes exactly four data tensors — `x` (noise latent), `timestep`, `context` (audio),
  `y` (static reference latent) — plus precomputed caches of those same four and a `**kwargs` that
  swallows anything else (`flash_head/src/modules/flash_head_model.py:498-515`). The shipped denoise loop
  passes only those four (`flash_head/src/pipeline/flash_head_pipeline.py:399-404`). There is no pose,
  motion, keypoint or driving-signal argument, and `has_image_input` is `false` in both the Pro and Lite
  checkpoints, so the CrossAttention image branch is dead code.
- `y` is the VAE encoding of the **still** reference image repeated `frame_num` times
  (`flash_head_pipeline.py:252-253`). It carries appearance, not motion.
- The only motion-bearing state is `latent_motion_frames`, a `(C, 1..2, h, w)` VAE latent of the
  previously rendered, colour-corrected pixels (`flash_head_pipeline.py:483-504`). That is a **state**,
  not a trajectory, and it is identity- and appearance-entangled.
- Audio reaches the transformer as cross-attention K/V in **all 30 blocks**, over all spatial tokens of
  each latent slice, with no mouth ROI and no strength scalar
  (`flash_head_model.py:327-339`); self-attention spans the whole token sequence with no causal mask
  (`flash_head_model.py:213-247`). Head motion is therefore audio-conditioned and ungated.
- The pixel → VAE → latent feedback loop means divergence compounds across windows
  (`flash_head_pipeline.py:456 → 485 → 488 → 504`). Even if window 0 were reusable, window 5 is not.
- The repo already says this in prose: "Using the same random seed repeats the noise sequence, not the
  model's response to different conditioning" (`docs/research/HEAD_MOTION_CONTROL_2026-09-16.md`).

Three further reproducibility facts that constrain any "replay seed 50" plan:

1. **Seed numbering is not a stable coordinate.** Draw count depends on the schedule: 1 + steps draws per
   window on the distilled path, 1 draw per window on the CFG teacher branch, plus an extra VAE-posterior
   draw in the RTC engine. Seed 50 at 4 steps and seed 50 at 2 steps are unrelated trajectories.
2. **There are three seeding sites, not one.** `flash_head_pipeline.py:223` (shipped script),
   `soulx_rtc/engine.py:187` (engine sampler, which *overrides* the template seed passed at `:177`), and
   `engine.py:190-191` (refinement, a derived stream). Plus a global `torch.manual_seed(0)` inside
   `fork_rng` at `engine.py:174`.
3. **Bit-exactness is already measured to break.** Byte-identical PRO configs produced different
   `raw_rgb_sha256` across a cold/warm `.torchinductor` cache boundary
   (`docs/research/RTX4070_RUN_RESULTS_2026-09-20.md:179-189`), at 2.0–2.7/255 mean pixel difference in
   windows 1–2. And the shipped **LITE** `generate_video.py` path samples the VAE posterior from the
   *global* RNG (`flash_head/ltx_video/ltx_vae.py:17`), so `--base_seed N` alone does not determine a run
   there at all. The RTC engine fixes this; the shipped script does not.

What *is* stable: the per-window noise sequence at a given seed is audio-content- and audio-length-
independent up to window count, so a seed is a replayable **noise schedule** — enough for paired A/B
comparisons, not enough to make motion reusable. And the first-window anchor
`latent_motion_frames = ref_img_latent[:, :1]` is deterministic for a given reference image in PRO
(the Wan VAE encode returns `mu` and never samples). Caveat: the reference-template encode is itself
`torch.compile`d (`engine.py:78-79`, `flash_head_pipeline.py:189-193`), so the anchor should be treated
as *expected*-deterministic, not proven — nobody has measured its cross-process hash.

## Route survey

### Route A — SoulX clip becomes the MuseTalk pose-bank base video directly

**Feasible: partly, and previously measured as worse than the alternative.**

How it works: generate with SoulX, transcode to the delivery contract, run `certify_pose_bank.py`, then
`prepare_musetalk_avatars.py` POSTs the clip to `/avatars/prepare`.

The contract (`character_factory/config/pose_spec.json:5-16`): 480x832, 24 fps, h264/yuv420p, **zero**
audio streams, 6 canonical handle frames + 6 blend frames each end, `frame_count = 1 + 8*round((s*fps-1)/8)`
→ 241 / 289 / 289 / 145 / 241 / 145 for the six render keys.

What actually gates the incoming clip is only **two** things, not the whole contract:

- the raw decode must be a whole number of 480x832 RGB24 frames (`certify_pose_bank.py:43-45`);
- the decoded frame count must equal that render's exact `frame_count` (`:175-176`).

Everything else is *manufactured*: certification overwrites the first and last 6 frames with one
per-character anchor, cross-fades 6 more, and re-encodes at fixed QP 18 with `-an -r 24 -pix_fmt yuv420p`,
then probes **its own output** (`:187`). So codec, pixel format, audio stream and endpoint equality are
free. (The minimum-length check at `:75-79` is unreachable dead code — it runs only after the
frame-count check has already passed.)

Blockers, in order of severity:

1. **SoulX cannot generate without audio.** `generate_video.py:24` asserts `--audio_path`; the pipeline's
   only per-window entry point is `generate(self, audio_embedding)`. A raw SoulX clip has lip-sync to a
   specific utterance baked into the pixels, and the pose-bank premise is a silent motion plate. Muting
   the MP4 does not help: "merely supplying speech then muting the MP4 leaves mouth articulation intact"
   (`benchmarks/closed_mouth_silence_20260917/README.md:28`).
2. **Silence closes the mouth but kills the motion.** The one measurement at exactly the delivery geometry
   (480x832, 24 fps, Lite, 4 steps, 128,000 exact-zero samples, seeds 50/51): max lip-gap/eye-span 0.0013
   and 0.0015 — but head roll p95−p5 of only **0.419° and 0.281°**. That is an idle plate, not six
   distinguishable poses. PRO's behaviour under silence is explicitly untested.
3. **Geometry.** The saved PRO donor is 320x576 (aspect 0.5556) against 480x832 (0.5769). A plain scale
   distorts the face ~3.8% horizontally; a correct conversion needs pad-or-crop plus a 1.44–1.5x upscale
   of an already small frame. The reshape gate catches nothing here — it is a byte-count congruence, so a
   transposed 832x480 source passes and produces garbage.
4. **Timebase, silently.** Nothing in the factory probes source fps. A 25 fps SoulX clip with the right
   frame count is re-encoded at 24 fps and plays ~4.2% slow, and **certification reports PASS**.
5. **Ledger coupling.** `package_pose_bank.py:140-141` unconditionally dereferences `seed_pair` and
   `prompt_id`, which only the ComfyUI driver writes. A SoulX-sourced clip needs a fabricated ledger entry,
   and the bank manifest would then carry the false `generation_mode` string
   `ltx23_q4_text_to_motion_then_handle_certified`.
6. **Prior measurement says the silent base wins anyway.** `benchmarks/pro_musetalk_redub_20260917/`
   used a SoulX PRO clip as a MuseTalk base and redubbed it: SyncNet-style relative score 0.737 (PRO base)
   vs **0.820** (silent base) at best offset. Verdict on record: "It is not a clean quality win over a
   silent source."

Cost: SoulX's fastest recorded figures are 0.03016 s/frame at 256x448 (`pro_30fps_20260921/normconv-r01`)
and 0.04623 s/frame at 320x576 (`p3b-1step`). No SoulX generation at 480x832 has ever been timed, so the
economics at the contract geometry are unknown.

Quality risk: high. Baked phonemes fight MuseTalk's new speech outside the 256x256 inpainted crop
(jaw line, cheeks). Plus every new base clip needs its own full prepared bundle (~2.2–2.8 GB cache) —
reusing one neutral geometry across poses was already tested and rejected for cheek/jaw drift.

### Route B — SoulX clip drives LTX 2.3 IC-LoRA LipDub as a source video

**Feasible: no. Not installed, and the wrong tool anyway.**

Two independent reasons:

1. **Nothing to run it with.** `ComfyUI-LTXVideo` does not exist anywhere on this filesystem
   (`find / -maxdepth 6 -type d -name ComfyUI-LTXVideo` → nothing), so five required node classes
   (`LTXAddVideoICLoRAGuide`, `LTXICLoRALoaderModelOnly`, `LTXVTiledVAEDecode`, `LTXVSetAudioRefTokens`,
   `LTXFloatToInt`) do not exist in either local install. Neither `ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors`
   (2.47 GB) nor the x2 spatial upscaler is on disk. Reviving it means a custom node at a pinned commit,
   a local Kornia 0.8.3 patch, and a gated Hugging Face download.
2. **LipDub is a dubber, not a motion transfer.** Its speech comes from `GetVideoComponents` slot 1 of the
   *same* source mp4 (node 5005 in the api graph). There is no external-audio input node. "Replace the
   speech" means re-muxing new TTS into the source file first — the three-model
   OmniVoice → IA2V → stage → LipDub chain in `LTX-2.3/context.md:258-268`.

Also worth recording: the native-resolution LipDub experiment was already run and rejected on 2026-07-30
(`LTX-2.3/workflows/experiments/lipdub/NATIVE_RESOLUTION_EXPERIMENT_FAILURE_REPORT.md`) — it ran fine at
576x1024 but "did not establish a successful fix for blurry teeth."

### Route C — Extract an explicit pose trajectory and re-drive a generator

**Feasible: yes. Executed. This is the answer to the question.**

How it actually works (`A1/graphs/generation-241.json`):

```
SoulX PRO seed-50 render (320x576, 25 fps, 250 frames, h264+AAC)
  → CPU DWPose (yolox_l.onnx + dw-ll_ucoco_384.onnx, custom_controlnet_aux)
    body=True, face=True, hands=False; 25→24 fps resample; fixed eye-based similarity transform
  → 241 rendered skeleton PNGs at 512x832  (A1/inputs/pose/%06d.png)
  → PilotPoseSequence (hand-written node) → image batch [241,832,512,3]
  → LoraLoaderModelOnly(ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors, 1.0)
    → GetICLoRAParameters → LTXVChunkFeedForward(chunks 2, dim_threshold 4096)
  → LTXVAddGuide #1 "portrait" (target identity still, frame_idx 0, strength 1.0)
  → LTXVAddGuide #2 "control" (241-frame pose batch, frame_idx 0, strength 1.0, iclora_parameters)
  → SamplerCustomAdvanced, Euler, CFG 1, seed 189, 8-step ManualSigmas
  → LTXVSeparateAVLatent → crop → SaveLatent → (separate process) VAEDecodeTiled
```

Mechanism note: this is an **IC-LoRA guide adapter, not a ControlNet**. There is no ControlNet anywhere
on this box. `LTXVAddGuide` concatenates the encoded guide onto the latent sequence as keyframe tokens
carrying their own RoPE coordinates (`comfy_extras/nodes_lt.py:394-406`); `LTXVCropGuides` strips them
after sampling. `replace_latent_frames` is dead code with zero callers.

Notable: **stock ComfyUI at the pinned commit already has everything needed.** `ComfyUI-LTXVideo` is not
required for video-guided generation; A1 proves it with `LoraLoaderModelOnly` + `GetICLoRAParameters` +
`LTXVAddGuide`.

Measured result: eye-x r=0.9945, eye-y r=0.9949, roll r=0.9690, 0/240 missing detections. Image-plane
only; no 3D motion proof; one identity, one donor, one seed pair; `state.json` is `pending_user_review`.

**Cost (corrected — use the sampler numbers, not the wall-clock totals).**

The ComfyUI server logs carry per-step sampler timings and they are the authoritative measurement:

| Arm | Sampler | Decode | Total | s/it |
| --- | --- | --- | --- | --- |
| A1 pose-controlled, 241 fr, 512x832 | 134.0 s | 14.781 s | 376.497 s | 16.75 |
| Matched no-control, 241 fr, 512x832 | 96.1 s | 15.404 s | 336.808 s | 12.01 |
| No-control recheck (seed 190) | ~95.9 s | 15.473 s | 329.158 s | 11.99 |
| A1 pose-controlled, 49 fr | 41.0 s | — | 281.366 s | 5.13 |

- **Union Control costs +37.9 s on 96.1 s of real compute = +39.5% of sampling time.** The often-quoted
  "+11.8%" is an artifact of ~241 s of per-prompt model reload that `--cache-none` forces and that the
  control does not affect. That overhead is directly measured, not extrapolated: 376.497−134.0 = 242.5 s
  and 281.366−41.0 = 240.3 s agree to within 2 s. Do **not** use a per-frame slope; sampler time goes
  41.0 → 134.0 s while latent frames go 7 → 31 (x4.43), which is markedly sublinear.
- Run-to-run spread on the no-control config is 336.808 vs 329.158 = 7.65 s (2.3%), so the +37.9 s control
  delta is ~5x the noise. Direction holds; call it "known to within about 2%", not "known exactly".
- **Real-time factor: 39.0x cold end-to-end, 14.8x warm** (134.0 + 14.8 = 148.8 s for 10.04 s of output).
  Any decision about LTX should be made against 14.8x, not 39.0x. Either way, offline only.
- VRAM on this RTX 4070 SUPER (12,282 MiB): sampled whole-device peaks 7,779 MiB (A1 241-frame gen),
  9,251 MiB (A1 49-frame gen), 755 MiB (A1 decode). Fits only under
  `--lowvram --reserve-vram 3 --cache-none --disable-pinned-memory --disable-async-offload`. Host RAM is
  the tighter wall: ~29 GiB cgroup, no swap, and one monolithic generate+decode run was already
  OOM-killed (exit 137). Every working pipeline splits SaveLatent from a fresh-process LoadLatent decode.

**Quality risk — the one real problem: the donor's lips come along.**

A1 used an explicit closed-mouth positive prompt and an "open mouth, visible teeth" negative, and the
mouth still opened. Quantified: `mouth_gap_over_eye_span` max 0.22991 / std 0.05647 for the donor vs
**0.24546 / 0.06043** for A1. A1's mouth opens *slightly more* than the donor's — the control signal
attenuates articulation not at all. Against the pilot plan's review threshold of
`max(0.03, idle p95 + 0.015)` this is roughly 8x over, i.e. it fails the mouth-suitability gate.

The cause is mechanical and the fix is a rendering change, not a model change: `A1/scripts/prepare.py:54`
calls `draw_poses(..., draw_body=True, draw_hand=False, draw_face=True)`, so the donor's lip keypoints are
literally drawn into the control images. Pilot plan repair option 1 (`SOULX_SEED50_..._PLAN.md:701`) is
exactly this: keep rigid head/body points, omit mouth/jaw points, using the pinned estimator's own
landmark topology — "Do not assume MediaPipe indices are DWPose indices," and leave missing features
absent rather than painting a black rectangle over the RGB face.

Other constraints on this route:

- **480x832 is hard-blocked.** The adapter asserts `reference_downscale_factor=2`, and `LTXVAddGuide`
  raises an explicit `ValueError` when the latent width (480/32 = 15) is not divisible by 2. A1 rendered
  512x832 (latent 16x26) and cropped 16 columns per side. It fails loudly, not silently.
- **The 512-wide reference is padded, not real.** `prepare.py:14-15` takes an 832x480 frame from the
  MuseTalk idle bank and edge-pads 16 px per side. The model conditioned on replicated pixels in exactly
  the columns that were later cropped.
- **A1 deviates from the official recipe** in two unablated ways: the official workflow stacks
  `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` alongside the union adapter over a non-quantized base
  (A1 ran the union adapter alone over the Q4_K_M GGUF), and drives it through `LTXAddVideoICLoRAGuide`
  (A1 substituted core `LTXVAddGuide`).
- **The matched no-control baseline has a prompt confound.** The two graphs differ by the Union LoRA, the
  pose guide, **and** the positive/negative prompts (A1: "lips remain gently closed"; no-control: "speaks
  naturally and continuously ... with clear conversational lip articulation"). The timing conclusion
  survives; any quality comparison between them does not.
- **Every pose-controlled run wastes audio compute.** A1 builds `LTXVEmptyLatentAudio` → concat → sample →
  separate, then uses only the video slot. There is no `LTXVAudioVAEDecode` in the graph; the output is
  silent. Whether the AV model can be run video-only is not established by anything in the tree.
- **Decode config matters.** Re-decoding the identical latent at `temporal_size 64 / overlap 16` instead of
  16/8 halved decode time (14.781 s vs 28.706 s) and substantially reduced double-face ghosting **in the
  one inspected frame** (`review/decode16-vs64-frame48.png`). The run explicitly declines to claim this
  identifies the cause of every artifact.
- **The two installs are split.** `A1/ComfyUI` has the Union adapter but lacks ComfyUI-PromptRelay,
  rgthree-comfy, ComfyMath and cg-use-everywhere. `/workspace/experiments/ltx23-soulx-transfer/ComfyUI`
  has all four custom-node packs but only `put_loras_here` in `models/loras`. Neither can do both. Both
  `.venv/bin/python` are symlinks to `/usr/bin/python3` — no environment isolation between them.

### Route D — Run SoulX itself as the realtime engine

**Feasible: partly, but it does not answer the question.**

SoulX is already fast enough to be interesting: fastest recorded 0.03016 s/frame at 256x448
(`normconv-r01`, PRO, 2 steps) = ~33 fps, 0.04623 s/frame at 320x576 (`p3b-1step`) = ~21.6 fps, against the
standing 30 FPS @ 576x320 goal and the `s30-reuse` baseline of 0.05097 s/frame. Peak device memory:
SoulX Lite compiled 6,165 MiB vs MuseTalk TRT 10,037 MiB.

But as a route to "reusable seed-50 movement" it is a dead end:

- **SoulX cannot consume a base video.** "SoulX has no base-video input that preserves all original
  motion" (`docs/research/VALIDATION.md:36`). Reusing the same portrait does not preserve choreography or
  framing.
- **No exact first/last-frame control.** No `end_image`, `last_frame`, pose trajectory or endpoint-inpainting
  API in the installed pipeline or RTC service (`docs/research/CONTINUOUS_CALLS.md:11, :28`). Idle→generation
  pose continuity is a documented open problem (`CONTINUOUS_WEBRTC.md:69`).
- **No head-motion dial.** The 38-setting sweep found none: shifts 0.5/1 smear faces, history 4 caused
  104/750 face-detection failures, shift 10 changed roll range by +30%/−55%/+92% across seeds 7/50/99
  (non-monotonic). The 0.5 audio-strength hook is also non-monotonic.
- **Concurrency is not a lever.** Two concurrent PRO sessions reached 33.43 aggregate / 16.71 per session
  against 33.10 single-stream: "aggregate FPS approximately equals single-stream FPS." Worse, a session's
  DiT numerics depend on batch composition, since concurrent sessions are stacked into one call.
- Standing architectural verdict on record: "Keep MuseTalk as the production renderer; keep SoulX Lite as
  an experimental alternative... A wholesale migration is not justified by the evidence available today."

### Route E — Anchor-only reuse via `latent_motion_frames` / `Engine.recondition()`

**Feasible: yes, cheap, but it pins a pose, not a trajectory.**

`latent_motion_frames` is a public, directly assignable attribute, and `Engine.recondition()`
(`soulx_rtc/engine.py:240-257`) is a working, tested proof that it can be overwritten from nine arbitrary
RGB frames — it already handles encode + normalize_latents + generator plumbing. `motion_frames_num` is 5
for PRO and 9 for LITE. The first window pins one latent slot; later windows pin two.

Useful for: making a generated take start from the same pose as a base video or a certified bank anchor,
by construction. SoulX's RTC avatar catalog already uses `certified/idle_active_listening.mp4`'s first
decoded frame as the conditioning anchor, and that frame *is* the certified shared boundary anchor.

Not useful for: motion transfer. The latent is the VAE encoding of that specific person's colour-corrected
pixels, so it is identity-entangled. Whether injecting a *different* identity's frames gives pose transfer
or identity bleed has never been tried — `recondition()` makes the experiment cheap.

Also available and unlisted elsewhere: `soulx_rtc/mouth_sr.py:64-90` already runs MediaPipe FaceMesh
(`refine_landmarks=True`, 20-point outer-lip index list) in-process over rendered frames. Combined with
`recondition()`, that is a second, dependency-free motion seam — landmarks measured from what was just
sent.

### Route F — SoulX → pose-only control → LTX base plate → certify → MuseTalk  **(recommended)**

**Feasible: yes. ~90% built. One unrun experiment stands between here and a testable plate.**

This is Route C with the mouth stripped out, plus the two halves that are already proven on either side:

1. SoulX PRO seed-50 (or any seed from the 50/51/0/1 shortlist, shift 5, history 2, strength 1.0) is the
   **motion donor**. Offline, once.
2. DWPose extraction with mouth/jaw points omitted → 241 rendered control frames at 512x832.
3. LTX 2.3 + Union Control IC-LoRA renders the production avatar at 512x832, crop to 480x832.
4. `certify_pose_bank.py` freezes 6 handle frames onto the bank anchor, cross-fades 6, re-encodes at
   fixed QP 18 with forced keyframes, and enforces one shared decoded boundary hash.
5. `prepare_musetalk_avatars.py` POSTs the certified clip; MuseTalk lip-syncs live at 200+ fps.

Why the ends are safe:

- **MuseTalk demonstrably overrides a baked LTX mouth.** On digital silence the normalized mouth gap falls
  from 0.0944 to 0.0220 (−76.7%); on normal speech, mouth gap vs audio RMS correlates r=0.555 (quiet 0.0225
  vs active 0.0909); no double-mouth artifact in sampled frames
  (`/workspace/experiments/ltx23_musetalk_lip_override_20260922/README.md:11-17`). So even an imperfectly
  closed plate may be recoverable. Caveat: non-speech audio is outside the useful domain — the chirps case
  scored r=0.087.
- **The boundary anchor is already shared and portable.** Decoded boundary hash
  `47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33` appears in **both** the close-up
  production manifest and the older `sample_ai_human_facetime_v1` manifest, across banks with different
  clip lengths (121/97 vs 241/289/145). The anchor is a property of the shared source portrait
  (sha `0e893bd1…`), not of one bank. A new SoulX-derived plate can in principle certify onto that same
  anchor and stay transition-compatible with the live bank.
- **The MuseTalk runtime is provenance-blind.** `webrtc_pose_router` derives fps and frame count from the
  decoded file; `api_server` resolves clips by `avatar_id` out of prepared caches; `pose_protocol`
  validates structure only; a grep for `ltx` across the whole runtime finds only two overridable defaults
  in the smoke harness. A plate that satisfies the media contract is indistinguishable at runtime.
- **A minimal bank is legal.** `pose_spec.json` defines a two-render `minimal` tier; `resolve_pose_renders`
  aliases the four remaining logical IDs onto the idle loop, so the six-ID wire contract still validates.
  A pilot needs to certify one clip, not six.

Cost: ~149 s of warm LTX compute per 10 s plate (134.0 s sampler + 14.8 s decode), or ~391 s cold as
currently launched. Offline, one-time per pose per character.

Quality risk: the mouth leak (unresolved), plus unmeasured questions — does the handle freeze read as a
stutter on LTX-with-SoulX-motion output, and does the plate then produce a working MuseTalk prepared
cache? Phases 7 and 8 of the pilot plan were not executed.

### Route G — Skip SoulX as donor; use the existing certified clip library

**Feasible: yes, today, zero new generation.**

`/workspace/MuseTalk/assets/ltx23_pose_banks/` holds 61 mp4s across 7 identity banks, many under
`certified/` with named motions (`idle_active_listening`, `nod_agree`, `empathetic_head_tilt`,
`light_smile`, `speaking_direct`). A1's own reference anchor came from one of them
(`A1/scripts/prepare.py:7`). If the goal is "a reusable head movement library," the donors are already on
disk and already silent, already 480x832/24 fps, and already certified. This directly answers the
"one identity, one donor" weakness of the A1 evidence base.

### Route H — Different control representation: depth, or canny

**Feasible: canny yes with no install, depth needs one. Neither has been run.**

Core ComfyUI already ships the `Canny` node and it is present in the live registry. No depth estimator is
installed (`VideoDepthAnythingProcess` absent, `models/geometry_estimation` empty). Arm A2_DEPTH is scoped
in the pilot plan (`:145`) and never ran — and A1 vs A2 is the only comparison that isolates the control
**representation** with the adapter held fixed.

Asymmetry worth recording: the adapter's own HF card lists only `Lightricks/Canny-Control-Dataset` as
training data. So canny is supported by the card but untested here, while pose is supported by the
empirical 0.9945 correlation but not by the card.

### Route I — Sparse guide anchoring via KJNodes

**Feasible: installed, untried.**

`LTXVAddGuidesFromBatch` (uses every non-black frame of a batch as a guide at its batch index) and
`LTXVAddGuideMulti` (N explicit guides with per-guide `frame_idx` and `strength`) are both present in the
live registry. Useful for cheap sparse motion anchoring without full-sequence token cost. Nothing on disk
tests either.

### Route J — Offline seed selection (the validated zero-code "motion control")

**Feasible: yes, already the accepted mechanism.**

`docs/research/MOVEMENT_SEED_PROTOCOL_2026-09-17.md`: seeds 50, 51, 0, 1 as the preferred shortlist, with
shift 5, `motion_frames_latent_num` 2, audio strength 1.0, from a user visual review of a 38-setting sweep.
Seed order is not an intensity scale. This is how a "calmer" or "livelier" donor gets picked. It needs no
new code and it is the only head-motion control the repo accepts.

### Route K — Substitute SoulX for LTX as the character-factory pose-bank generator

**Feasible: no, for a six-pose bank.**

The substitution *surface* is genuinely narrow — one script (`generate_pose_videos.py`, 288 lines), four
ledger keys, and a clip at `state/renders/<char>/<key>.mp4`. Everything downstream is generator-agnostic.
But the conditioning channel is the problem:

- LTX here has **two** working conditioning channels (text prompt via Q4 Prompt Relay; DWPose/depth video
  via Union Control). SoulX has **one** (audio), and no text path at all — `self.text_embedding` is defined
  at `flash_head_model.py:412` and never called in `forward`; `--lean` deletes it.
- The entire semantic specification of `nod_agree`, `empathetic_head_tilt` and `light_smile` lives in
  `pose_prompt_pack_v1.json` as text. **There is no receiver for it in SoulX.** Substitution does not
  produce a six-pose bank; it produces one idle-like clip per (portrait, seed, driving-audio) triple.
- The only safe driver is silence, which yields 0.28–0.42° of roll.

Also, the factory's configured ComfyUI does not exist on this box: `generation_defaults.json:8-9` points at
`/workspace/experiments/ltx23-8gb/ComfyUI/{input,output}` and port 18188, and there is no `ltx23-8gb`
directory. The LTX arm of the factory is not runnable as configured here either.

Scale, for reference: the roster is 332 characters / 1,784 renders / ~388,088 frames. At the authoritative
LTX figure (223.057 s for 249 frames at 480x832, RTX 5060 Ti) that is ~96.6 GPU-hours ≈ 4 days of pure
render time — an **upper bound**, because ~86.8 s of that 223 s is model loading, conditioning, decode,
audio and encoding, which a queued batch amortizes. The README's "multi-month queue" is not supported by
GPU time; the defensible multi-month cost is the 332 hand-generated ChatGPT portraits and human review.
Note also that the 223 s is a 5060 Ti number while every SoulX figure here is a 4070 SUPER number — the
economics as written compare two GPUs with no scaling factor. Nothing in the factory has ever been run
against a GPU.

## Hard blockers

1. **"Match seed 50" is not achievable as stated.** Seed 50 fixes a noise schedule, not a motion
   trajectory. Audio cross-attention writes into every spatial token of every latent frame in all 30
   blocks with no gate, and the autoregressive pixel→VAE→latent feedback compounds divergence, so the same
   seed with different audio gives different head motion.
2. **There is no pose representation inside SoulX to extract.** Motion exists only as seeded noise and as a
   1–2 frame identity-entangled VAE latent. (Landmarks *do* exist inside `soulx_rtc/` — `mouth_sr.py`,
   `face_landmarker.py` — but only as post-hoc analysis of already-rendered RGB, never as an input.)
3. **Seed numbering is not a stable coordinate** across step counts, model types, resolutions,
   `skip_zero_weighted_noise`, `latent_feedback`, `timestep_variant`, `refinement_timesteps`, or the
   engine's extra posterior draw.
4. **Bit-exact replay is unavailable.** `raw_rgb_sha256` cannot be used as an acceptance gate across a
   cold/warm torchinductor boundary; the shipped LITE path is not seed-determined at all; attention backend
   is swappable at import *and* at runtime, and SageAttention 2.2.0 on SM89 ignores the current CUDA stream
   and has already produced non-finite window-0 pixels.
5. **Union Control drags the donor's lips into the output**, at 8x the plan's review threshold, with an
   explicit closed-mouth prompt in force.
6. **Union Control cannot render 480x832.** `reference_downscale_factor=2` forbids latent width 15;
   512x832 + crop is mandatory.
7. **LipDub is unrunnable here** — node package and both weight files absent — and is a dubber, not a
   motion control.
8. **LTX is offline-only.** 14.8x realtime warm, 39.0x cold, on the 4070 SUPER.
9. **No single ComfyUI install can do both** pose control and the Prompt Relay / 6+2 production graph.
10. **Every new base clip needs its own MuseTalk prepared bundle** (~2.2–2.8 GB); reusing one neutral
    geometry across poses was already tested and rejected for cheek/jaw drift.
11. **SoulX cannot consume a base video and has no exact endpoint control**, so any SoulX-sourced plate
    needs the same external freeze/blend/fixed-QP certification an LTX plate needs.
12. **Do not copy SoulX latents into LTX.** The VAEs share only eight tensor names and differ in shape on
    three of them (`decoder.conv_in.conv.weight` soulx `[512,128,3,3,3]` vs ltx23 `[1024,128,3,3,3]`),
    with different class names and entirely different block lists.

## Defects found along the way (unrelated to the question, worth fixing)

- **Q4 6+2 graph, tail frames.** The second-stage `LTXVCropGuides` (node `5667:5451`) is wired to un-guided
  conditioning (`["5711",0]`) instead of the guide-augmented `["5649",0]`, so `num_keyframes` resolves to 0,
  no crop happens, and 8 reference-guide frames survive into every output. Proven by latent shapes:
  6+2 finals are `[1,128,32,26,15]` vs `[1,128,31,26,15]` for first-pass, and inter-frame mean-abs
  difference jumps from 0.919 to 7.546 at the 240→241 boundary. **This is a two-line question, not a
  one-line fix**: the second-stage `CFGGuider` (node `5667:5456`) is *also* on un-guided conditioning and
  the second-stage sampler takes the first stage's uncropped 32-frame latent, so during the "+2" steps the
  guide frame is re-denoised with no keyframe attention at all. Fixing only the crop changes the output
  length and must be re-measured. Not an A/V desync — the audio latent grew too (aac 10.368 s vs video
  10.375 s).
- **Q4 6+2 graph, orphaned audio refinement.** The final `VHS_VideoCombine` (node 5661) takes audio from
  `["5658:5346",0]` — the **first** stage's audio decode — while its images come from the second stage.
  Node `5667:5450` (second-stage `LTXVAudioVAEDecode`) has **no consumers**. Video gets 8 sampling steps;
  delivered audio gets 6, and the "+2" audio refinement is computed and discarded.
- **Certification reuse path attributes provenance it never verified.** `certify_pose_bank.py:197` copies
  `source_render_sha256` from the ledger even when the source file was never opened this run, and
  `package_pose_bank` propagates it into the bank manifest.
- **MuseTalk latent/frame desync risk.** `api_avatar.py:780-782` skips any frame whose bbox is the
  `(0,0,0,0)` placeholder with a bare `continue`, so `input_latent_list_cycle` is 2M while
  `frame_list_cycle` is 2N. No length assertion anywhere. A foreign donor with harder framing silently
  desynchronises rather than erroring.
- **MuseTalk emits one fewer frame** than `audio_duration × 24` (95-frame 4 s, 143-frame 6 s outputs).
- **Operational trap.** The lip-override run's first request never entered inference: the 12 GB runtime
  reserved 4 GB while a batch-8 request demanded a hardcoded 10 GB logical lease. The server had to be
  restarted with `GPU_RESERVED_MEMORY_GB=1.0`. Any repeat of MuseTalk-over-LTX on this box hits this first.

## Corrections to widely repeated claims in this tree

| Claim as usually stated | Corrected |
| --- | --- |
| Union Control costs +11.8% | +39.5% of **sampler** time (+37.9 s on 96.1 s). The 11.8% is diluted by ~241 s of reload. |
| ~257 s fixed overhead, 0.496 s/frame marginal | ~241 s fixed (directly measured twice). No valid per-frame slope — sampler cost is sublinear. |
| The matched pair differs only by the LoRA and pose guide | It also differs by the positive and negative **prompts**, which are opposite in speech intent. |
| SoulX's fastest run is 0.051 s/frame (s30-reuse) | s30-reuse is the baseline. Fastest is 0.03016 s/f at 256x448 (`normconv-r01`); 0.04623 s/f at 320x576 (`p3b-1step`). |
| The production bank is "LTX Q4, dual first/last image guide, lossless all-intra" | That describes `sample_ai_human_facetime_v1`. The **active** close-up bank's clips are `preserved_certified_asset_byte_for_byte` from user-approved V6/V8/V14/V15 renders, with no generator or prompt field recorded. |
| The production bank's clips are 241 frames | 241/289/289/145/241/145, per render key. |
| Decoder tiling *caused* the A1 ghosting | It *contributed*, in one inspected frame. The run explicitly declines the causal claim. |
| SageAttention is not installed locally | Not available in either ComfyUI environment (both symlink `/usr/bin/python3`); a build does exist in the unrelated `/workspace/omnivoice-triton/.venv`. |
| The seed enters at exactly one place | Three: `flash_head_pipeline.py:223`, `engine.py:187` (overrides the template), `engine.py:190-191` (refinement). |
| "Do not re-run FFN chunking" | Scoped to the **SoulX-side port** only. KJNodes' `LTXVChunkFeedForward` is an active node in the working A1 graph (`chunks 2, dim_threshold 4096`) and feeds the CFGGuider. Removing it silently changes the recipe. |
| The minimum-25-frame certification gate | Unreachable dead code; it runs only after the exact frame-count check has passed. |
| Certification requires 480x832 source pixels | It requires a **byte-count congruence**. A transposed 832x480 source passes and produces scrambled pixels. |
| The ltx23-soulx-transfer install can run the motion work | It cannot — `models/loras/` holds only `put_loras_here`. Only `A1/ComfyUI` (port 18190) has the adapter. |

## Recommended path

Run **Route F**, starting with the single unrun experiment that unblocks it.

**A3_REPAIR** — re-render A1 with mouth/jaw keypoints omitted from the DWPose control images. This is a
change to `A1/scripts/prepare.py:54` (`draw_face`) using DWPose's own landmark topology, then a re-run of
the frozen `A1/graphs/generation-241.json` with the sigma string copied verbatim
(`1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`). Gate: `mouth_gap_over_eye_span`
max must fall under `max(0.03, idle p95 + 0.015)` while eye-x/eye-y/roll correlations stay above 0.9.
Cost ≈ 391 s cold. The plan allows exactly one repaired full render.

If that passes, continue to certification (Phase 7) onto the existing anchor `47b05c6b…` and then a
MuseTalk prepared cache (Phase 8), which are the two steps `A1/RESULTS.md:44` confirms were never run.

If it fails, run **A2_DEPTH** before concluding anything about the adapter — A1 vs A2 is the only
comparison that isolates the control representation.

**Rollback is one line throughout**: the production close-up bank is untouched and byte-for-byte
preserved; the pose-controlled work lives entirely in `/workspace/experiments/`.

## Do not re-run (already settled, with reasons)

SoulX-side FFN token chunking (10.25 MiB saved, ~3% slower, teeth still smeared) · the LTX VAE swap into
SoulX (incompatible checkpoints) · the SoulX clean-latent refinement restart (no consistent teeth recovery,
+18–20% latency) · native-resolution LipDub (no teeth fix) · the wide/full-shoulder bank (character too
small in a vertical frame) · single-neutral-geometry pose reuse (cheek/jaw drift) · decoder INT8
(1.6–5.3% slower than compiled BF16) · `--lean-delivery` (+0.02%, fails bit-exactness) · 4x super-resolution
(invented brace-like tooth divisions) · two-session PRO concurrency (aggregate ≈ single-stream) ·
the general many-idle-videos system (smoothness not solved by matching endpoints).

## Open questions

- Does A3_REPAIR close the mouth without losing head motion? **Unrun. Highest-value experiment on the box.**
- A2_DEPTH: never run. No way yet to attribute the mouth leak to the representation rather than the adapter.
- No matched 10 s A0 exists with A1's prompts, so the adapter's contribution to flicker and identity drift
  is not isolated.
- Does an A1-style plate survive the handle freeze without reading as a stutter, and does it produce a
  working MuseTalk cache? Phases 7–8 unexecuted.
- Does the LTX→MuseTalk lip override hold across identities, facial hair, profile angles, larger head
  movement, teeth-heavy articulation and repeated 10 s loop boundaries? Explicitly untested.
- How much of the ~241 s per-prompt overhead survives dropping `--cache-none` on a 12 GB card with
  OmniVoice resident, and does that push peak VRAM past 12 GB?
- What is SoulX's wall clock at 480x832 / 24 fps (Lite 4-step and PRO)? Never measured. The substitution
  economics cannot be compared without it.
- How large is the LITE VAE posterior std in practice? `latent_log_var='uniform'` gives one learned
  log-variance channel broadcast across all 128 latent channels; the magnitude is unmeasured. The
  reproducibility conclusion does not depend on it — an unseeded draw breaks bit-exactness regardless.
- Does injecting `latent_motion_frames` from a **different** identity give pose transfer or identity bleed?
  `recondition()` makes it cheap; nobody has tried.
- No experiment holds seed and reference fixed, varies the **audio**, and measures head motion. The
  audio-induced pose divergence at fixed seed is asserted from code, not measured.
- Can the AV model be run video-only (skipping the audio concat)? Every pose-controlled run samples a
  241-frame audio latent and discards it. Untested cost lever.

## Cross-references

- `/workspace/MuseTalk/character_factory/SOULX_SEED50_SINGLE_AVATAR_PILOT_PLAN.md` — the 1,087-line
  governing plan, arms REF_DONOR / REF_V14 / A0_PORTRAIT / A1_POSE / A2_DEPTH / A3_REPAIR, gates at
  `:666-681`, repair option 1 at `:701`.
- `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/RESULTS.md` — the executed A1 arm.
- `/workspace/experiments/ltx23_musetalk_lip_override_20260922/README.md` — MuseTalk over a baked LTX mouth.
- `/workspace/experiments/ltx23_direct_lipsync_recheck_20260922/` — the third, most recent no-control LTX run
  (seed 190, prompt a91f2cbb, 329.158 s generation), which supplies the run-to-run spread.
- `docs/research/HEAD_MOTION_CONTROL_2026-09-16.md`, `docs/research/MOVEMENT_SEED_PROTOCOL_2026-09-17.md` —
  this document corroborates and extends both.
- `benchmarks/closed_mouth_silence_20260917/`, `benchmarks/pro_musetalk_redub_20260917/`,
  `benchmarks/movement_ranges_20260917/`.

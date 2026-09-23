# Is a reusable SoulX-FlashHead head-movement copy possible?

Date: 2026-09-22 UTC
Hardware for all new measurements: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB, driver 595.84.

Question asked:

> Can we make a reusable head-movement copy of SoulX-FlashHead — take the PRO
> model's reference base avatar movement, have LTX 2.3 generate that same
> movement, and get a talking head in our generation pipeline that matches
> SoulX seed 50?

## Verdict

**Yes — and the hard part has already been executed on this box, earlier today.**
But the reusable object is *not* the seed. It is a **DWPose skeleton sequence
extracted from a saved SoulX clip**. That distinction is the whole finding.

- Seed 50 is **not** a portable motion identity. New measurements below show a
  change of denoise schedule moves head motion **as much as changing the seed
  entirely** (ratio 1.04x), and that audio — not the seed — drives **52–70%** of
  head-motion variance. "Seed 50's movement" is a property of one rendered clip,
  not of the model.
- Once you accept that and treat **the saved clip** as the asset, transfer works.
  The `A1` pilot at `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1`
  drove LTX 2.3 from a SoulX seed-50 donor via DWPose and reproduced the donor's
  head trajectory at **eye-x r=0.995, eye-y r=0.995, roll r=0.969**, onto a
  *different* target identity.
- It is **offline only**: 39.0x realtime (376.5 s generation + 14.8 s decode for
  10.04 s of 512x832). It is a base-video factory, never a serving path.

This matches, and now quantifies, the caution already written into
`MuseTalk/character_factory/SOULX_SEED50_SINGLE_AVATAR_PILOT_PLAN.md` §1.1.

**Labelled comparison video** (20 s, both halves of the argument in one file):

```text
docs/research/soulx_motion_reuse_comparison_20260922.mp4
```

Part 1 plays seeds 0/1/50/51 side by side on identical audio and reference image,
so the seed lottery is visible. Part 2 plays the A1 transfer: SoulX seed-50 donor,
the DWPose skeleton extracted from it, and the LTX 2.3 output that reproduces the
motion on a different face.

## Part 1 — What "seed 50" actually is (new measurements)

Method: `mediapipe` FaceMesh on existing benchmark videos; head pose fitted as a
2D similarity transform over a **rigid** landmark set (brows, eye corners, nose
bridge, temples — landmarks not displaced by speech), expressed relative to each
clip's own first frame and normalised by inter-ocular distance (IOD). Mouth
landmarks are tracked separately *in the head frame*, so lip motion and head
motion are measured independently. No GPU was used; all inputs are retained
benchmark artifacts.

Scripts and raw trajectories are retained at `benchmarks/motion_reuse_20260922/`
(`head_motion.py`, `motion_decompose.py`, `divergence_vs_time.py`, `plot_traj.py`,
plus the `.npz` pose trajectories so the numbers can be re-derived without re-running
landmark extraction).

### 1.1 SoulX is bit-reproducible, but motion is fragile to numerics

All runs: seed 50, same reference image, same audio, PRO `pro-fp8-v1`.

| Comparison | head translation (IOD) | head roll (deg) | reading |
| --- | ---: | ---: | --- |
| `adv-ctl-r01` vs `adv-ctl-r02` (exact repeat) | **0.0000** | **0.000** | perfectly deterministic |
| `adv-ctl-r01` vs `adv-gc08-r01` (allocator only) | 0.0007 | 0.012 | negligible |
| `adv-ctl-r01` vs `adv-ctl-r03` | 0.0273 | 0.372 | 35% of a seed re-roll |
| `adv-ctl-r01` vs `adv-prealloc-r01` | 0.0367 | 0.496 | 47% of a seed re-roll |
| `adv-ctl-r01` vs `classic-freshcache-r01` | 0.0715 | 1.064 | 91% of a seed re-roll |
| **typical different-seed pair** (reference) | **0.0783** | ~1.1 | full re-roll |
| seed 50 @4 steps vs seed 50 @2 steps | **0.0814** | 1.387 | **104% — a schedule change is a full re-roll** |

Two consequences:

1. Exact repeats are bit-identical, so SoulX *is* reproducible when nothing moves.
2. **Motion match cannot be a quantization acceptance gate.** PRO-quantized, LITE,
   a different step count, or a different runtime path will not reproduce seed
   50's trajectory. The repo already says seeds are not comparable across step
   counts (`benchmarks/pro_quantization_v2_20260918/run.py:617-620`); this
   measures how large the effect is — it is total.

Divergence is also **not gradual**. `adv-prealloc-r01` tracks the control to
within 0.02–0.04 IOD for 8.5 s and then bifurcates to 0.0884 in the final window.
The autoregressive motion feedback compounds tiny numerical differences until the
trajectory re-rolls.

### 1.2 Audio, not seed, drives most of the head motion

Four runs (seeds 0/1/50/51) share one reference image and one audio track. A
static image cannot produce a *time-varying* shared component, so whatever the
four trajectories share over time is audio-driven.

| pose channel | variance shared (audio-driven) | variance seed-driven |
| --- | ---: | ---: |
| head dx | **57.8%** | 42.2% |
| head dy | **69.9%** | 30.1% |
| head roll | **52.3%** | 47.7% |
| head scale | **63.2%** | 36.8% |

Head-motion speed also correlates with the audio envelope (r = +0.28 at +1 frame
lag). Mouth shape across seeds is nearly identical (0.018–0.028 IOD), confirming
the decomposition: the mouth is audio-locked, the head is audio-driven *plus*
seed noise.

**So changing the audio changes the majority of the head motion.** Seed 50 with a
new utterance is not seed 50's movement.

### 1.3 Agreement decays with clip length

Shared (audio-driven) fraction of head-roll variance, by window:

| window | 0–2 s | 2–4 s | 4–6 s | 6–8 s | 8–10 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | **80.6%** | 63.9% | 10.7% | 1.7% | 19.4% |

Early windows are anchored to the reference image
(`latent_motion_frames = ref_img_latent[:, :1]`, `flash_head_pipeline.py:267`) so
all seeds agree. As autoregressive feedback accumulates, seeds diverge. Practical
rule: **short clips are reproducible; long clips are not.** Under ~3 s the motion
is largely determined by audio + reference; past ~4 s it is seed lottery.

![trajectories](../../benchmarks/motion_reuse_20260922/motion_trajectories.png)

### 1.4 There is no motion representation inside the model

The DiT takes exactly four tensors — noise `x`, timestep, audio `context`, and
the static reference latent `y` (`flash_head_model.py:498-515`). `y` is the VAE
encoding of the *still* reference repeated `frame_num` times
(`flash_head_pipeline.py:252-253`), so it carries appearance, not motion. There
is no pose, keypoint, expression or motion-code input anywhere. `text_embedding`,
`audio_emb` and `img_emb` exist but are never called by `forward()`.

Motion therefore exists only as pixels and as a 1–2 frame VAE latent of
previously rendered pixels — identity-entangled and not portable.

One caveat found in the audit and worth flagging: the shipped **LITE** path in
`generate_video.py` samples the VAE posterior from the **global** RNG
(`flash_head/ltx_video/ltx_vae.py:17`, `generator=None`), so `--base_seed N`
alone does not determine a LITE run. Only `soulx_rtc/engine.py` fixes this.

## Part 2 — Can LTX 2.3 reproduce that movement? Already demonstrated.

### 2.1 Correction: IC-LoRA LipDub is not usable here

The LipDub graph does take a source *video* and does preserve its motion — but it
is unrunnable on this box. Five of its node types
(`LTXAddVideoICLoRAGuide`, `LTXICLoRALoaderModelOnly`, `LTXVTiledVAEDecode`,
`LTXVSetAudioRefTokens`, `LTXFloatToInt`) come from `ComfyUI-LTXVideo`, which is
not installed anywhere on this filesystem, and neither the lipdub safetensors nor
the x2 spatial upscaler is on disk. LipDub also takes its speech from the source
video's own audio track — there is no external-audio input node.

### 2.2 What does work: Union Control IC-LoRA + DWPose

Installed and executed at `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1`
(pre-existing work from 02:28–03:13 UTC today; not produced by this analysis):

```
SoulX PRO seed-50 clip  ->  DWPose skeleton (CPU, 241 PNGs)  ->
LTX 2.3 Q4 + Union Control IC-LoRA (LTXVAddGuide)  ->  target-identity video
```

- Adapter: `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`, revision
  `b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff`. Loads with **stock** ComfyUI nodes
  (`LoraLoaderModelOnly` + `GetICLoRAParameters` + `LTXVAddGuide`) — no
  `ComfyUI-LTXVideo` needed.
- Donor: `benchmarks/pro_lite_150x_20260917/recreated_20260921/pro/video.mp4`.
- Fidelity: **eye-x r=0.9945, eye-y r=0.9949, roll r=0.9690**, 0/240 missing face
  frames on both donor and output (`A1/review/metrics.json`). Image-plane only —
  not a 3D motion proof.
- The target identity is a **different person** from the donor; motion transfers,
  identity does not. This is the property the idea needs.

Visual: `A1/review/donor-pose-A1.mp4` (donor | DWPose control | LTX output).

### 2.3 Cost — offline only

Matched pair on this 4070 SUPER, same install, same day, differing only by the
Union LoRA + pose guide:

| arm | generation | decode | realtime factor |
| --- | ---: | ---: | ---: |
| no control | 336.808 s | 15.404 s | 35.1x |
| **pose-controlled** | **376.497 s** | **14.781 s** | **39.0x** |

Control costs +11.8%. Marginal cost ≈ 0.496 s per output frame on top of ~257 s
fixed overhead (GGUF load + CPU Gemma encode under `--cache-none`).

Also: Union Control asserts `reference_downscale_factor=2`, so 480x832 raises a
`ValueError` (latent 15x26, 15 % 2 ≠ 0). A1 rendered 512x832 and cropped 16
columns per side.

## Part 3 — Routes, ranked

| # | Route | Feasible | How | Cost | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | **SoulX clip → DWPose → LTX Union Control → MuseTalk pose bank** | **yes, demonstrated** | A1 pilot above | 39x realtime, offline | **the answer** |
| 2 | SoulX clip becomes the MuseTalk base video directly | partly | skip LTX; certify the SoulX clip itself | cheap | see blockers — geometry/fps/ledger work, and you keep the donor's identity |
| 3 | Extract a motion representation from inside SoulX | **no** | — | — | no pose/motion tensor exists; motion is pixels only |
| 4 | Reproduce seed-50 motion by re-running SoulX with new audio | **no** | — | — | audio drives 52–70% of the motion |
| 5 | SoulX as the realtime engine | separate track | — | — | that is the 30 FPS @ 576x320 goal, not this |

Route 1 is right because the **DWPose sequence is the durable asset**. It is a
plain PNG sequence: model-independent, quantization-independent, audio-independent,
re-usable across target identities, and bankable. Everything that makes seed 50
fragile stops mattering once the motion is committed to skeletons.

## Part 4 — Blockers before this is production

**MuseTalk pose-bank certification** (`character_factory/scripts/certify_pose_bank.py`):

- **Resolution, hard.** `decode_frames` reshapes to `(N, 832, 480, 3)` and raises
  if the byte count does not divide (`:43-45`). There is no scaler. A clip must be
  exactly 480x832 before it reaches the factory. SoulX native is 320x576 / 512x512.
- **Frame count, hard.** Must equal the exact per-render number (241 idle/empathy,
  289 speaking, 145 nod/smile) at `:175-176`. SoulX emits 28-frame chunks and lets
  `ffmpeg -shortest` trim to audio length. Needs an explicit trim/pad step.
- **fps — silent corruption, not a rejection.** Nothing probes source fps;
  `encode_frames` stamps 24 (`:60`). A 25 fps SoulX clip with correct dimensions
  and frame count **passes certification and plays 4% slow**. This is the
  dangerous one because no gate names it.
- **Ledger coupling.** `package_pose_bank.py:140-141` dereferences `seed_pair` and
  `prompt_id`, written only by the ComfyUI driver. A non-LTX clip raises KeyError
  and the manifest would carry the false provenance string
  `ltx23_q4_text_to_motion_then_handle_certified`.

Non-blockers, usefully: codec/pixel-format/audio are re-encoded unconditionally
(`-an -c:v libx264 -pix_fmt yuv420p`), and the boundary anchor is *manufactured* —
`apply_handles` overwrites 6 frames each end and cross-fades 6 more, so a foreign
clip need not arrive boundary-legal. A **minimal two-render bank** is also legal
(`pose_spec.json:116-120`), so a pilot needs to certify one clip, not six.

**Quality, open:** A1's mouth still opens despite a closed-mouth prompt, because
the DWPose control retains donor lip keypoints. Not yet ablated. For a MuseTalk
base video the mouth should be closed/neutral — MuseTalk repaints it. Dropping
face keypoints from the control (`draw_face=False` in `A1/scripts/prepare.py:54`)
is the obvious first ablation and has not been run.

**Also found (unrelated but real):** the shipped Q4 Prompt Relay 6+2 graph
mis-wires its second-stage `LTXVCropGuides` to un-guided conditioning, so
`num_keyframes` resolves to 0 and **8 extra reference-guide frames survive into
every output** (latents `[1,128,32,26,15]` vs the correct `[1,128,31,26,15]`;
249 frames vs 241). Worth fixing independently.

## Part 5 — Recommendation

1. **Stop treating seed 50 as the target.** Treat the *saved clip* as the donor
   and the *DWPose sequence* as the asset. Bank the skeletons; they outlive every
   model change. Section 1.1 of the pilot plan already says this — the numbers
   above are the evidence for it.
2. **Run the face-keypoint ablation** on A1 (`draw_face=False`, body+head only).
   Cheap, and it directly targets the one quality defect blocking base-video use.
3. **Add an fps gate to `certify_pose_bank.py`.** It is a few lines and it closes
   a silent-corruption path that will bite any non-LTX source.
4. **Do not expect the quantized PRO to match seed 50.** Use distributional
   quality gates, not trajectory match. This is now measured, not assumed.
5. Keep this offline. At 39x realtime it is a factory step; MuseTalk stays the
   serving engine.

## Evidence

New measurements from this analysis:

| artifact | contents |
| --- | --- |
| `MuseTalk/docs/soulx_motion_seeds_20260922.json` | per-pair head/mouth trajectory comparison, seeds 0/1/50/51 |
| `MuseTalk/docs/soulx_motion_repeat_20260922.json` | determinism ladder across control/allocator/runtime arms |
| `MuseTalk/docs/soulx_motion_decomposition_20260922.json` | audio-vs-seed variance split, envelope correlation |
| `MuseTalk/docs/soulx_motion_divergence_20260922.json` | divergence growth by time window |
| `MuseTalk/docs/soulx_motion_trajectories_20260922.png` | the three-panel trajectory plot |

Pre-existing pilot evidence (not produced here):

- `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/RESULTS.md`
- `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/review/metrics.json`
- `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/review/donor-pose-A1.mp4`
- `/workspace/MuseTalk/character_factory/SOULX_SEED50_SINGLE_AVATAR_PILOT_PLAN.md`

# SoulX PRO talking base redubbed with different audio by MuseTalk

September 17, 2026. **Fresh inference on NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible VRAM, driver 595.84.** PRO used Torch 2.7.1+cu128 / CUDA 12.8, BF16, four steps, seed 51. MuseTalk 1.5 used Torch 2.5.1+cu121 / CUDA 12.1, FP16, batch 8. Native test videos are 320×576, 25 FPS, 250 frames / 10 seconds. PRO's initial device usage was 2,727 MiB; two co-resident Python processes reported 2,478 and 234 MiB. Each GPU phase records its own snapshots and runtime in the linked JSON. CPU frame analysis/packaging is separate from GPU generation and GPU SyncNet evaluation.

## Result

**The experiment works as a proof of concept: MuseTalk replaces PRO's original speech articulation with the new audio while retaining its head/body motion. It is not a clean quality win over a silent source.**

- [Watch comparison with audio B](comparison-with-audio-B.mp4): left = PRO source made for A (deliberately mismatched to the playing B); middle = that PRO source redubbed with B; right = silent-source control redubbed with B.
- [Watch finished PRO + MuseTalk redub B, shared endpoints](pro-redub-B-anchored.mp4).
- [Watch original PRO generation with audio A](pro/video.mp4).
- [Silent reusable PRO base, exact matching endpoints](pro-base.mp4).
- [Raw MuseTalk redub B, no endpoint repair](musetalk/pro-redub-B.mp4).
- [Mouth comparison](mouth-comparison.png) and [full-frame comparison](frame-comparison.jpg).

The new mouth can close when PRO's original mouth is visibly open, including the inspected 2.00 s, 6.00 s, and 8.84 s frames. Its mouth-opening trajectory correlates **0.919** with the silent-base MuseTalk control driven by the same audio B, versus **0.275** with the original PRO articulation. This is evidence of audio replacement rather than merely retaining the old mouth movements. These correlations are 2D landmark proxies, not lip-sync scores.

The inspected output still has soft/irregular teeth and changes to lip/cheek appearance. Original speech-related head, jaw and expression motion can remain outside the replaced mouth region. No strong persistence of the old phoneme sequence is apparent in the sampled frames and relative sync measurements, but this single identity/two-utterance experiment cannot establish absence of all interference.

## Controlled audiovisual measurement

GPU: same RTX 4070 SUPER / 12,282 MiB visible. Local LatentSync SyncNet weights from MuseTalk's checkout, strict checkpoint loading, Torch 2.5.1+cu121 / CUDA 12.1. This is a **relative cosine-similarity diagnostic, not standard LSE-C/LSE-D or an absolute quality certification**. Higher is better. Lower-face crops use the native MuseTalk source boxes, normalized as in local training code. Audio and video windows are evaluated through the repository's SyncNet and mel preprocessing.

| Video scored against new audio B | At unchanged timing | Best within ±10 frames | Best audio offset |
| --- | ---: | ---: | ---: |
| Original PRO video made for A — mismatch control | 0.116 | 0.200 | +3 frames |
| PRO base → MuseTalk B | 0.330 | 0.737 | +2 frames |
| Silent base → MuseTalk B | 0.343 | 0.820 | +2 frames |

The PRO→MuseTalk B output's best score against the **old A** is only **0.191**, versus **0.737** against B. Shuffling B audio windows drops its score to **0.230**. The positive control, original PRO video against its original A, reaches **0.686** (best offset −1).

There is a shared timing issue to investigate: both B-driven MuseTalk outputs prefer audio **two frames later** than the corresponding video window. At 25 FPS that is an estimated **80 ms visual lead** under this evaluator. The MuseTalk same-audio A control also prefers +2 frames. Thus the offset is not specific to using a previously speaking PRO base. No global runtime timing setting was modified, and the delivered files preserve unshifted audio timing. Evaluator preferences are not independent proof of an exact perceptual offset.

The silent control scores higher than the PRO base after alignment; these cosine scores are not percentages. This supports proceeding with a reviewable prototype, not claiming equivalent quality or production readiness.

## Exact construction

1. Decode the first frame of `closed_mouth_silence_20260917/silence-seed51.mp4`. Save it as [reference-from-silence-seed51.png](reference-from-silence-seed51.png). Apply SoulX's standard resize/center crop to the tested 320×576 profile; save [reference-320x576.png](reference-320x576.png).
2. Audio A is the first nine seconds of `benchmarks/comparison-10s.wav`. Audio B is the first nine seconds of MuseTalk's retained `kokoro_duration_matrix/multi_sentence.wav`. Each has 0.5 s silent handles and a 50 ms fade at the cut. They are different recordings, with hashes recorded in [inputs.json](inputs.json).
3. Generate PRO from that image using A, seed 51, four steps, default shift 5, two latent history frames (five RGB history frames for PRO's Wan VAE). Checkpoint/config hashes are in [pro/results.json](pro/results.json). The official tested PRO runner is reused, not the Lite serving engine.
4. Replace the first six and last six frames with the same reference image and use six cosine-blend frames immediately inward from each handle. Encode all-intra lossless H.264/yuv420p. All 12 decoded handle frames match exactly. This is explicit postprocessing, not a claimed native PRO endpoint constraint. [Anchor validation](anchor-validation.json).
5. Prepare the PRO and silent-control videos using native MuseTalk DWPose/S3FD bounding boxes, 10-pixel lower margin, VAE latents and BiSeNet jaw masks (cheek widths 90). Render A and B over the identical PRO cache, plus B over the silent cache. All 250 source frames were detected in each bank. The silent source was converted from 24 to 25 FPS and its final frame held after eight seconds before endpoint treatment. [MuseTalk results](musetalk/results.json).
6. Evaluate **raw** redubs with no final boundary correction. Separately package the finished PRO→B copy with the common reference handles/blends. Its first and last decoded RGB frames are identical and match the PRO base anchor. Merely giving MuseTalk identical input endpoints did **not** make its raw output endpoints identical.

No production avatar, serving process, pose manifest or model weights were replaced. This is an offline first-pass redub test; it does not test WebRTC playback, arbitrary speech start positions, or repeated ping-pong cycles.

## Validation and reproducibility

- 250 frames per clip, 25 FPS, complete decode of original PRO, both B outputs, the comparison and final package.
- All 750 comparison frames have MediaPipe landmarks. Sampled regular frames and the largest original-versus-redub mouth differences were visually inspected.
- 38 overlapping SyncNet windows per clip from 1.00 through 8.40 s, 16 video frames and 52 mel steps each; lag sweep −10 to +10 frames. Boundary blends excluded. Four video cases, both A and B, plus shuffled-audio controls.
- [Per-window SyncNet evidence](syncnet-results.json), [per-frame mouth/media checks](review-results.json), [generation and runtime provenance](pro/results.json).

Scripts: [input and endpoint preparation](prepare.py), [PRO generation](generate_pro.py), [MuseTalk rendering](redub.py), [SyncNet evaluation](evaluate.py), [review packaging](review.py). Run preparation/PRO/review in the SoulX `.venv` with `PYTHONPATH=.`; MuseTalk/render evaluation use `/workspace/.venvs/musetalk_trt_stagewise/bin/python` from the MuseTalk directory with `PYTHONPATH=.:scripts:musetalk/utils`. Scripts intentionally reject overwriting completed renders; use a fresh experiment directory for another run.

One initial MuseTalk attempt failed before rendering because the offline audio batches were on CPU; the diagnostic wrapper was corrected to transfer them to CUDA. A packaging retry corrected NumPy integer JSON serialization. Both original failure logs are retained; no failed attempt is included in quality scores.

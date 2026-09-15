# Male still-reference versus idle-motion experiment — September 13, 2026

> **GPU provenance — retrospective local-host attribution:** NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible). No independent per-run GPU snapshot was identified for this report. This applies to the local SoulX/MuseTalk inference and receiver tests described here; CPU-only checks do not establish GPU performance. Historical or upstream results on other GPUs retain their separate attribution. See [GPU run provenance](GPU_RUN_PROVENANCE.md) for dates, evidence, and attribution limits.

Using only the male avatar's neutral still portrait changes the generated
articulation and pose, but does not reliably reduce exposed teeth or fix their
smeared detail in this test. Some sampled still-start frames open the mouth
more than the idle-motion condition. The still portrait itself has a closed
mouth. It does not constrain subsequent generated speech to a closed mouth.

## Controlled conditions

Evidence: `/workspace/experiments/flashhead-still-reference-ri2mkn`.
Same male reference PNG and ten-second WAV as the previous
[male teeth experiment](MALE_TEETH_QUALITY_2026-09-13.md), at 320×576/four steps,
compiled Lite with INT8 storage/BF16 compute, compact memory, real RoPE,
optimized conditioning, lean output and fused QKV. Allocator cap: 3,584 MiB.
Two seeds, 50 and 51; two repeated renders per seed per condition (12 renders).
Six raw RGB arrays and ten-second movies retained; 250 useful frames per movie,
264 padded frames generated per render. Setup/warmup/encoding excluded from FPS.

All conditions use the **same still portrait as the identity reference**.
The variation is initial motion setup, not a different identity-conditioning API:

| Condition | Initial motion |
| --- | --- |
| `still` | Normal `prepare_call` portrait latent; no idle reconditioning |
| `still-nine` | Recondition on nine repeats of the same still portrait |
| `video-motion` | Recondition on the fixed nine decoded idle-video frames used previously |

`still-nine` and `video-motion` execute the same posterior-sampling path and
consume the same number of RNG draws. `still` omits that sampling call and uses
a one-frame portrait latent rather than the encoded nine-frame context. Its
differences therefore include context length and RNG progression as well as
motion. This is the actual still-start behavior, but not an isolated proof that
one motion variable alone caused every visible difference.

## Results

| Condition | Median useful FPS | Peak Torch MiB | Repeated raw frames identical |
| --- | ---: | ---: | --- |
| `still` | 53.73 | 3,328.07 | Yes, both seeds |
| `still-nine` | 53.48 | 3,328.07 | Yes, both seeds |
| `video-motion` | 53.51 | 3,328.07 | Yes, both seeds |

All retained raw clips have 250 valid RGB frames and no full-frame blackouts.
Visual review covers both seeds at frames 0, 5, 12, 25, 37, 75, 107, 125, 175
and 225. Teeth remain exposed during open-mouth speech in all modes, with the
same kind of broad/smeared tooth detail. This review found no convincing teeth
quality improvement from selecting a still start, and no consistent suppression
of mouth opening. It is a limited visual assessment, not an objective tooth
segmentation, blinded preference, or phoneme-level lip-sync test.

The fixed mouth ROI `[96:230, 236:328]` was compared over all 250 frames.
Still versus idle-motion mouth MAE was 15.88/255 and 14.41/255 for seeds 50/51;
repeated-still versus idle-motion was 5.80/255 and 5.63/255. These show changed
pose/articulation and appearance; they are not a naturalness or teeth-visibility
score. No actual WebRTC receiver capture was made in this experiment: movies
encode the generated raw frames using the same CRF18/veryfast playback format
as previous offline footage. Crossfades are absent from the raw comparisons.

## Inspect

- [Seed 50 comparison video](/workspace/experiments/flashhead-still-reference-ri2mkn/seed-50-comparison.mp4).
- [Seed 51 comparison video](/workspace/experiments/flashhead-still-reference-ri2mkn/seed-51-comparison.mp4).
- [Seed 50 mouth crops](/workspace/experiments/flashhead-still-reference-ri2mkn/seed-50-mouth.png).
- [Seed 51 mouth crops](/workspace/experiments/flashhead-still-reference-ri2mkn/seed-51-mouth.png).
- [Measurements](/workspace/experiments/flashhead-still-reference-ri2mkn/summary.json).

The live service was temporarily paused using the previously authorized
quality-test procedure and restored with original arguments/environment after
generation. The unrelated GPU service remained running. Private launch snapshots
remain outside git; lifecycle scripts must not be rerun to restore context.
Live rendering behavior was not changed to use a still avatar by this test.

This result weakens the hypothesis that a moving reference alone explains the
male-versus-female teeth difference. Portrait-specific model behavior and
generated mouth-detail limitations remain relevant. Choosing a still start can
be a useful motion preference, but it is not a validated teeth-quality fix.

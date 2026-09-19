# Distance / framing lip-sync experiments

September 17 shoulder-visible replication: [the exact earlier 1.25× portrait at six source resolutions](../portrait_source_detail_20260917/README.md). Native 320×576 is strongest; 256 remains close, 128 is softer and 64 degrades strongly. Upscaled 640/1280 files add no real detail and do not improve teeth.

September 17 causal follow-up: [isolated source-image detail at fixed 512×512/25 FPS](../source_detail_20260917/README.md). Reducing the exact same close face crop from 307/256 effective pixels to 128 and 64 progressively blurs teeth and the whole face. This clarifies the earlier distance finding: closer framing helps partly by allocating more real source and generated pixels to the face, while nominal file/output resolution alone is insufficient.

GPU generation for these September 16 experiments: NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible, driver 595.84; per-run `results.json` records runtime and pre-load allocation. Landmark/statistical analysis is separate from generation.

Controlled SoulX FlashHead Lite and Ditto experiments measuring how apparent subject scale changes generated mouth motion. The shared fixture uses ten-second audio and 320×576 individual outputs. Initial SoulX trials use four steps, seeds 50/51, and scales 1.00/0.72/0.50; tighter-framing follow-ups use seed 50 and scales 1.00/1.25/1.50. Ditto follows its own motion-generation configuration, not SoulX's four-step profile.

## Findings

- [Why 1.25× moves its head more, and control options](../../docs/research/HEAD_MOTION_CONTROL_2026-09-16.md): CPU reanalysis of six retained RTX 4070 SUPER clips confirms greater normalized head motion at 1.25× in this seed. No validated independent head-motion slider exists; the audio-strength hook is entangled.
- [SoulX concurrency potential versus MuseTalk](../../docs/research/CONCURRENCY_POTENTIAL_2026-09-16.md): historical multi-GPU audit, current-profile throughput, active-speaker versus connected-call limits, and a qualification plan. No new GPU load test.
- [Model preferences and why framing behaves differently](../../docs/research/MODEL_CHOICE_AND_FRAMING_2026-09-16.md): user prefers SoulX's value, MuseTalk's apparent lip-sync accuracy, and Ditto's lower memory footprint, but finds Ditto stiff. Separates user judgments, measured evidence, and source-backed causal hypotheses.
- [Ditto expression strength 0.7](DITTO_CLOSER_0P7_REPORT.md): mean mouth opening increases 37.4–39.5% versus 0.5; framing still barely changes the normalized trajectory (0.995/0.993 correlation). Includes a six-panel 0.5/0.7 comparison. Naturalness improvement remains unverified.
- [Ditto replication at expression strength 0.5](DITTO_CLOSER_0P5_REPORT.md): the same three Indian-male references preserve nearly identical mouth trajectories (0.991/0.988 correlation with 1.00×). Closer improves visibility; improved lip-sync accuracy is not established.
- [Three close framings at strength 0.5](INDIAN_MALE_CLOSER_STRENGTH_0P5_REPORT.md): 1.25× is the best combined candidate; it preserves its unmodified trajectory at 0.928 correlation while reducing p95 opening by 24.6%.
- [Indian male closer-framing follow-up](INDIAN_MALE_CLOSER_REPORT.md): the exact prior close clip compared with new 1.25× and 1.50× outputs; 1.50× has the strongest trajectory agreement, while 1.25× preserves more headroom.
- [Indian male replication](INDIAN_MALE_REPORT.md): close is clearly best; far framing collapses mean normalized mouth opening by 92.2% and mouth-trajectory agreement with close to 0.208.
- [Initial female portrait](REPORT.md): framing changes the mouth trajectory; medium remains aligned but changes opening magnitude, while far diverges more strongly.

These automated MediaPipe landmark measurements are output-motion diagnostics, not phoneme-level or human-rated audiovisual sync scores.

Earlier descriptions such as “best” refer to these limited examples or user preferences, not universal accuracy rankings. “Closer is always better” is not established. See the cross-model write-up for scope and confounds.

## Reproduction

GPU generation:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/distance_lipsync/run.py \
  --source SOURCE_IMAGE \
  --output benchmarks/distance_lipsync/NEW_EVIDENCE_DIRECTORY
```

CPU landmark analysis:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/distance_lipsync/analyze.py \
  benchmarks/distance_lipsync/NEW_EVIDENCE_DIRECTORY
```

The run refuses to overwrite an existing evidence directory. Exact input hashes, GPU provenance, profile, output hashes, and timings are saved in each `results.json`.

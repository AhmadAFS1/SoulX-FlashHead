# Comparison video catalog

The original 49-video rename was file organization only: those video bytes are unchanged. Subsequent recordings/comparisons are listed first below. Their fresh inference ran September 17–18 on **NVIDIA GeForce RTX 4070 SUPER**, physical 12 GB class / **12,282 MiB visible**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8, with OmniVoice and idle LTX resident. CPU packaging/media checks are separate from inference; exact profiles and measured resource use are in the linked reports. This catalog itself makes no performance measurements.

Names identify the model, comparison variable, framing/resolution and seed where applicable. `SR` means learned super-resolution; `history` is the number of conditioning latent frames; `shift` is the sampling-schedule setting. Frame sizes in names refer to generation panels, not the combined video canvas. Edge sharpness comparisons do not certify anatomically correct teeth.


## PRO quantization: Indian man at 1.50×

September 18 [implementation and measured results](pro_quantization_20260918/README.md): fresh **RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8, with two co-residents reporting 2,478 and 234 MiB. Left: stock PRO BF16. Right: FP8 FFNs plus compiled transformer/conditioning and BF16 decoder. Matched 320×576, four steps, shift 5, history 2. Approximately 7.1→9.4 useful FPS; encoded playback remains 25 FPS. Visible tooth detail remains in samples, with motion/articulation differences. CPU media packaging is separate from inference; this does not establish real-time operation.

- [Stock PRO versus optimized FP8 PRO — seed 50](pro_quantization_20260918/comparison-seed50/comparison.mp4)
- [Stock PRO versus optimized FP8 PRO — seed 51](pro_quantization_20260918/comparison-seed51/comparison.mp4)
- [Stock PRO versus optimized FP8 PRO — seed 0](pro_quantization_20260918/comparison-seed0/comparison.mp4)
- [Stock PRO versus optimized FP8 PRO — seed 1](pro_quantization_20260918/comparison-seed1/comparison.mp4)
- [Stock PRO versus optimized FP8 PRO — 60 seconds, seed 50](pro_quantization_20260918/comparison-60s-seed50/comparison.mp4)

## FaceLandmarker and native-2x TensorRT component tests

[Audit and report](ojin_components_20260917/README.md). MediaPipe 0.10.35 and TensorRT 10.16.1.11 cu13 match Ojin's listed versions. Columns compare bicubic enlargement, the earlier 4x SR resized to 2x, and a public native-2x SRVGG TensorRT model. Ojin's exact SR checkpoint/code and mouth refiner were unavailable; these are component tests, not an exact reproduction or proven teeth fix.

- [soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed50.mp4](ojin_components_20260917/matched-runtime/soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed50.mp4)
- [soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed51.mp4](ojin_components_20260917/matched-runtime/soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed51.mp4)
- [SoulX-LITE-default-closeup-character-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner.mp4](ojin_components_soulx_distant_20260917/SoulX-LITE-default-closeup-character-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner.mp4) — corrected test on the exact `bf16-batch5-c5-recorded-peer0.mp4` SoulX character and receiver output.
- [LTX23-distant-indian-man-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner.mp4](ojin_components_ltx_distant_20260917/LTX23-distant-indian-man-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner.mp4) — supplementary off-target LTX control retained after the original misunderstanding.

## Recorded hybrid and single-Lite ROI experiments

[Hybrid report](roi_hybrid_analysis_20260917/README.md) and [single-Lite report](single_lite_face_roi_20260917/README.md). The hybrid uses 1.50x framing; the single-Lite comparisons return generated face crops to the normal 1.00x portrait. `fixed` means square paste, while `aligned` anchors the generated face to a static body/background. Neither test establishes a teeth/lip-sync improvement.

- [soulx-LITE-vs-SoulX-plus-MuseTalk-indian-man-1.50x.mp4](roi_hybrid_analysis_20260917/recorded-comparison/soulx-LITE-vs-SoulX-plus-MuseTalk-indian-man-1.50x.mp4)
- [soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed50.mp4](single_lite_face_roi_20260917/soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed50.mp4)
- [soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed51.mp4](single_lite_face_roi_20260917/soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed51.mp4)
- [soulx-LITE-full-frame-vs-face-ROI256-384-512-fixed-normal-distance-seed50.mp4](single_lite_face_roi_20260917/soulx-LITE-full-frame-vs-face-ROI256-384-512-fixed-normal-distance-seed50.mp4)
- [soulx-LITE-full-frame-vs-face-ROI256-384-512-fixed-normal-distance-seed51.mp4](single_lite_face_roi_20260917/soulx-LITE-full-frame-vs-face-ROI256-384-512-fixed-normal-distance-seed51.mp4)

## distance lipsync

- [soulx-LITE-female-framing-1.00x-vs-0.72x-vs-0.50x-seed50.mp4](distance_lipsync/evidence-20260916/soulx-LITE-female-framing-1.00x-vs-0.72x-vs-0.50x-seed50.mp4)
- [soulx-LITE-female-framing-1.00x-vs-0.72x-vs-0.50x-seed51.mp4](distance_lipsync/evidence-20260916/soulx-LITE-female-framing-1.00x-vs-0.72x-vs-0.50x-seed51.mp4)
- [ditto-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-vad-alpha0p5.mp4](distance_lipsync/evidence-ditto-closer-0p5-20260916/ditto-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-vad-alpha0p5.mp4)
- [ditto-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-vad-alpha0p7.mp4](distance_lipsync/evidence-ditto-closer-0p7-20260916/ditto-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-vad-alpha0p7.mp4)
- [ditto-indian-man-vad-alpha0.5-vs-0.7-three-framings.mp4](distance_lipsync/evidence-ditto-closer-0p7-20260916/ditto-indian-man-vad-alpha0.5-vs-0.7-three-framings.mp4)
- [soulx-LITE-indian-man-framing-1.00x-vs-0.72x-vs-0.50x-seed50.mp4](distance_lipsync/evidence-indian-male-20260916/soulx-LITE-indian-man-framing-1.00x-vs-0.72x-vs-0.50x-seed50.mp4)
- [soulx-LITE-indian-man-framing-1.00x-vs-0.72x-vs-0.50x-seed51.mp4](distance_lipsync/evidence-indian-male-20260916/soulx-LITE-indian-man-framing-1.00x-vs-0.72x-vs-0.50x-seed51.mp4)
- [soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-labeled.mp4](distance_lipsync/evidence-indian-male-closer-20260916/soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-labeled.mp4)
- [soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-unlabeled.mp4](distance_lipsync/evidence-indian-male-closer-20260916/soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-unlabeled.mp4)
- [soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-audio-strength0.5.mp4](distance_lipsync/evidence-indian-male-closer-strength-0p5-20260916/soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-audio-strength0.5.mp4)

## mouth sr

- [soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-320-seed50.mp4](mouth_sr_20260917/320-seed50/soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-320-seed50.mp4)
- [soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-320-seed51.mp4](mouth_sr_20260917/320-seed51/soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-320-seed51.mp4)
- [soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-512-seed50.mp4](mouth_sr_20260917/512-seed50/soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-512-seed50.mp4)

## movement ranges

- [soulx-LITE-motion-history-1-2-3-4-seed50.mp4](movement_ranges_20260917/comparisons/soulx-LITE-motion-history-1-2-3-4-seed50.mp4)
- [soulx-LITE-motion-history-1-2-3-4-seed7.mp4](movement_ranges_20260917/comparisons/soulx-LITE-motion-history-1-2-3-4-seed7.mp4)
- [soulx-LITE-motion-history-1-2-3-4-seed99.mp4](movement_ranges_20260917/comparisons/soulx-LITE-motion-history-1-2-3-4-seed99.mp4)
- [soulx-LITE-sampling-shift-0.5-1-2-5-seed50.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-0.5-1-2-5-seed50.mp4)
- [soulx-LITE-sampling-shift-0.5-1-2-5-seed7.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-0.5-1-2-5-seed7.mp4)
- [soulx-LITE-sampling-shift-0.5-1-2-5-seed99.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-0.5-1-2-5-seed99.mp4)
- [soulx-LITE-sampling-shift-3-5-7-10-seed50.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-3-5-7-10-seed50.mp4)
- [soulx-LITE-sampling-shift-3-5-7-10-seed7.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-3-5-7-10-seed7.mp4)
- [soulx-LITE-sampling-shift-3-5-7-10-seed99.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-3-5-7-10-seed99.mp4)
- [soulx-LITE-sampling-shift-5-7-10-15-seed50.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-5-7-10-15-seed50.mp4)
- [soulx-LITE-sampling-shift-5-7-10-15-seed7.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-5-7-10-15-seed7.mp4)
- [soulx-LITE-sampling-shift-5-7-10-15-seed99.mp4](movement_ranges_20260917/comparisons/soulx-LITE-sampling-shift-5-7-10-15-seed99.mp4)
- [soulx-LITE-seeds-0-1-7-42-shift5-history2.mp4](movement_ranges_20260917/comparisons/soulx-LITE-seeds-0-1-7-42-shift5-history2.mp4)
- [soulx-LITE-seeds-50-51-99-1234-shift5-history2.mp4](movement_ranges_20260917/comparisons/soulx-LITE-seeds-50-51-99-1234-shift5-history2.mp4)
- [soulx-LITE-motion-history-1-2-3-4-seeds7-50-99-overview.mp4](movement_ranges_20260917/soulx-LITE-motion-history-1-2-3-4-seeds7-50-99-overview.mp4)
- [soulx-LITE-sampling-shift-0.5-1-2-5-seeds7-50-99-overview.mp4](movement_ranges_20260917/soulx-LITE-sampling-shift-0.5-1-2-5-seeds7-50-99-overview.mp4)
- [soulx-LITE-sampling-shift-3-5-7-10-seeds7-50-99-overview.mp4](movement_ranges_20260917/soulx-LITE-sampling-shift-3-5-7-10-seeds7-50-99-overview.mp4)
- [soulx-LITE-sampling-shift-5-7-10-15-seeds7-50-99-overview.mp4](movement_ranges_20260917/soulx-LITE-sampling-shift-5-7-10-15-seeds7-50-99-overview.mp4)
- [soulx-LITE-seeds-0-1-7-42-50-51-99-1234-overview.mp4](movement_ranges_20260917/soulx-LITE-seeds-0-1-7-42-50-51-99-1234-overview.mp4)

## portrait source detail

- [soulx-LITE-portrait-source-width-1280-vs-256-vs-native320.mp4](portrait_source_detail_20260917/soulx-LITE-portrait-source-width-1280-vs-256-vs-native320.mp4)
- [soulx-LITE-portrait-source-width-320-256-128-64-640-1280-output320x576.mp4](portrait_source_detail_20260917/soulx-LITE-portrait-source-width-320-256-128-64-640-1280-output320x576.mp4)

## pro lite 150x

- [soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50.mp4](pro_lite_150x_20260917/soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50.mp4)

## pro lite normal distance

- [soulx-LITE-vs-PRO-indian-man-normal-1.00x-320x576-seed50.mp4](pro_lite_normal_distance_20260917/soulx-LITE-vs-PRO-indian-man-normal-1.00x-320x576-seed50.mp4)

## pro lite teeth

- [soulx-LITE-vs-PRO-indian-man-square-closeup-512x512-seed42.mp4](pro_lite_teeth_20260917/soulx-LITE-vs-PRO-indian-man-square-closeup-512x512-seed42.mp4)

## refinement

- [soulx-LITE-4steps-vs-6steps-vs-low-noise-refinement-seed50.mp4](refinement_20260917/soulx-LITE-4steps-vs-6steps-vs-low-noise-refinement-seed50.mp4)
- [soulx-LITE-4steps-vs-6steps-vs-low-noise-refinement-seed51.mp4](refinement_20260917/soulx-LITE-4steps-vs-6steps-vs-low-noise-refinement-seed51.mp4)
- [soulx-LITE-4steps-vs-6steps-vs-strong-refinement-seed50.mp4](refinement_20260917/soulx-LITE-4steps-vs-6steps-vs-strong-refinement-seed50.mp4)
- [soulx-LITE-4steps-vs-6steps-vs-strong-refinement-seed51.mp4](refinement_20260917/soulx-LITE-4steps-vs-6steps-vs-strong-refinement-seed51.mp4)

## smile reference 320

- [soulx-LITE-neutral-vs-smiling-reference-indian-man-1.25x-seed50.mp4](smile_reference_320_20260917/soulx-LITE-neutral-vs-smiling-reference-indian-man-1.25x-seed50.mp4)

## source detail

- [soulx-LITE-source-detail-307-vs-256-vs-128-vs-64px-output512.mp4](source_detail_20260917/soulx-LITE-source-detail-307-vs-256-vs-128-vs-64px-output512.mp4)

## square teeth

- [soulx-LITE-output-resolution-512-vs-1024-and-15-vs-25fps.mp4](square_teeth_20260917/soulx-LITE-output-resolution-512-vs-1024-and-15-vs-25fps.mp4)

## strength recreated

- [soulx-LITE-audio-strength-1.0-0.9-0.75-0.5-indian-man-seed50.mp4](strength_recreated_20260917/soulx-LITE-audio-strength-1.0-0.9-0.75-0.5-indian-man-seed50.mp4)
- [soulx-LITE-audio-strength-1.0-0.9-0.75-0.5-indian-man-seed51.mp4](strength_recreated_20260917/soulx-LITE-audio-strength-1.0-0.9-0.75-0.5-indian-man-seed51.mp4)
- [soulx-LITE-motion-history-1-vs-2-vs-3-seed50-indian-man-1.25x.mp4](strength_recreated_20260917/soulx-LITE-motion-history-1-vs-2-vs-3-seed50-indian-man-1.25x.mp4)
- [soulx-LITE-sampling-shift-1-vs-3-vs-5-vs-7-seed50-indian-man-1.25x.mp4](strength_recreated_20260917/soulx-LITE-sampling-shift-1-vs-3-vs-5-vs-7-seed50-indian-man-1.25x.mp4)
- [soulx-LITE-seed-50-vs-51-vs-52-vs-53-indian-man-1.25x.mp4](strength_recreated_20260917/soulx-LITE-seed-50-vs-51-vs-52-vs-53-indian-man-1.25x.mp4)

The exact old-to-new paths for the original 49 videos are preserved in [the rename map](comparison-video-renames.json). Their [byte-preservation validation](comparison-video-rename-validation.json) records SHA-256 checks; later comparisons have media validation in their own experiment directories.

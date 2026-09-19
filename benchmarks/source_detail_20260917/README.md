# Isolated source-image detail experiment

Shoulder-visible replication: the [exact earlier 1.25× portrait](../portrait_source_detail_20260917/README.md) was tested at nominal widths 1280/640/320/256/128/64 with generation fixed at 320×576. It confirms that upscaling adds no dental detail, 256 remains close to native, 128 softens the face, and 64 causes obvious degradation.

September 17, 2026: fresh GPU inference on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. Each run records a direct `nvidia-smi` snapshot; whole-device use before every model load was 2,809 MiB because OmniVoice remained resident. All runs use an 8,704-MiB Torch allocator cap, which does not change physical VRAM. CPU MediaPipe/XNNPACK analysis and video packaging are separate from GPU generation.

## Result

**Base-image detail materially affects the teeth and overall face, but the evidence shows a threshold rather than “more pixels always wins.”** With output fixed at 512×512 and every generation setting held constant, effective source crops of 307 and 256 pixels look very similar. At 128 pixels, teeth, beard and skin texture visibly soften. At 64 pixels, the whole face becomes substantially blurred and its appearance/pose is less stable.

Watch the [four-way synchronized comparison](soulx-LITE-source-detail-307-vs-256-vs-128-vs-64px-output512.mp4), inspect the [full-frame lossless contact sheet](frames-comparison.png), or focus on the [mouth crops](mouth-comparison.png).

| Effective source crop | Video | Useful FPS | Median mouth-region edge energy | Opening trajectory vs. 307 px | Finding |
| ---: | --- | ---: | ---: | ---: | --- |
| 307×307 (native crop) | [video](detail-307/video.mp4) | 22.32 | 53.77 | 1.000 | Best retained source detail; teeth still synthetic/imperfect |
| 256×256 | [video](detail-256/video.mp4) | 22.13 | 56.16 | 0.940 | Visually close to 307; no clear loss in this sample |
| 128×128 | [video](detail-128/video.mp4) | 22.31 | 25.19 | 0.944 | Noticeably softer teeth and face texture |
| 64×64 | [video](detail-64/video.mp4) | 22.44 | 14.61 | 0.926 | Strong blur and visible facial degradation |

Edge energy is Laplacian variance over a native 96×52 mouth crop. It includes lips, beard, pose and H264 effects and is not a validated teeth-quality score. The roughly 53–56 to 25 to 15 progression supports the visible loss of detail, but it should not be read as a percentage dental-accuracy score. MediaPipe detected the face in all 1,000 decoded frames. All four videos decoded completely.

## What was isolated

The same source image and exact square crop `(38,64,345,371)` produce a native 307×307 crop. For each treatment, that crop is either retained or downsampled with Lanczos to 256, 128 or 64 pixels, then enlarged to the same 512×512 lossless PNG. Only this effective source-detail resolution changes.

Every run otherwise uses:

- 512×512 generation at 25 FPS for ten seconds;
- BF16, four denoising steps, seed 50 and stock audio conditioning;
- identical `benchmarks/comparison-10s.wav` audio;
- one state, eager execution and staged weight offloading;
- the same H264 CRF-18/preset-veryfast recording path;
- a fresh process, model load, warmup and reference preparation.

Timing begins after warmup/reference preparation and includes generation of full 24-frame chunks and discarded tail padding; encoding is outside timing. The nearly identical 22.1–22.4 useful FPS confirms that input-detail reduction does not reduce generation work after every reference is resized to the same model canvas.

## Interpretation across all retained evidence

This validates the user's source-resolution hypothesis in a qualified form:

1. **Effective face detail in the base image matters.** When the close face crop contains only 64–128 real pixels per side, SoulX cannot reconstruct stable fine dental detail merely because the input file or output canvas is enlarged.
2. **Framing still matters.** Moving/cropping closer makes the face occupy more source pixels and more generated pixels. That is why distance appeared important earlier: it changes the effective resolution allocated to the face and mouth.
3. **Output resolution has its own supported range.** The previous [square output test](../square_teeth_20260917/README.md) fed the same approximately 307-pixel source detail into 512 and 1024 output canvases. The 1024 outputs became severely blurred. Upscaling a source cannot create detail, and this model/runtime did not handle that larger canvas well.
4. **More source pixels above the current 307-pixel crop remain untested.** This experiment deliberately removes information to establish causality. It proves that insufficient source detail harms output, but it cannot prove that a truly native 1024-pixel face photograph would improve teeth beyond the 307/256 plateau.

The practical target from current evidence is therefore a sharp source with at least roughly **256 real pixels across the close face crop**, a close framing, and 512×512 generation. This is a one-avatar/one-audio/one-seed working threshold, not a universal model specification. Teeth remain generated content rather than faithful reconstruction, so even the best row contains fused or artificial-looking tooth boundaries in some frames.

## Reproduction and evidence

Run each detail treatment in a fresh process using [the fixed runner](../closer_capacity_20260916/run_source_detail.py):

```bash
PYTHONPATH=. .venv/bin/python benchmarks/closer_capacity_20260916/run_source_detail.py \
  --detail 307 --output NEW_DIRECTORY
```

Repeat for 256, 128 and 64. [Packaging script](package.py), [CPU analysis](analyze.py), [summary and decoded counts](summary.json), and [per-frame diagnostics](diagnostics.json). Each detail directory retains the exact reference PNG, result metadata, raw-frame hash, seven pre-encode PNGs, video and run log.

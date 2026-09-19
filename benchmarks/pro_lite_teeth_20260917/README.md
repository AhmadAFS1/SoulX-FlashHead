# SoulX FlashHead PRO versus LITE: teeth comparison

## Result

The PRO model is visibly sharper than LITE on the same 512x512 reference and
audio. Its median whole-mouth edge energy was **167.50 vs 51.31** (3.26x), and
central oral edge energy on open-mouth frames was **459.15 vs 125.33** (3.66x).
These are Laplacian-variance diagnostics, not validated dental-accuracy scores,
but the contact sheet and [synchronized comparison](soulx-LITE-vs-PRO-indian-man-square-closeup-512x512-seed42.mp4) show the
same direction: PRO preserves stronger tooth boundaries while LITE is usually
soft or gray. PRO can still collapse several teeth into a bright connected band,
so it improves the bottleneck without guaranteeing anatomically correct teeth.

The variants did not produce identical poses: opening trajectories correlated
at 0.644 over their shared 229 frames. PRO's median normalized opening was
0.0875 versus LITE's 0.0807, so the sharpness gain is not explained by LITE
simply opening much less. At some times the pose differs enough that a frame is
not a perfect pixel-level pair.

## Reproducible run

This was a fresh local run on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB
visible VRAM**, driver 595.84, Torch 2.7.1+cu128, CUDA 12.8. OmniVoice and an
idle LTX process remained resident. The PRO checkpoint was already installed;
the model and VAE hashes are in [model-sha256.txt](model-sha256.txt). The exact
reference and ten-second waveform are retained here. Both variants used BF16,
four denoising steps, seed 42, audio conditioning once, and no sharpening,
mouth-SR, refinement, or other post-processing.

The first optimized PRO startup failed during TorchInductor compiler workspace
autotuning with CUDA OOM. This was compiler overhead, not a bad or missing
checkpoint. The comparison was completed with `COMPILE_MODEL=False` and
`COMPILE_VAE=False` (eager execution), which fits the 12-GB card.

The released variants retain different numbers of frames for a ten-second
request because their recurrent chunk contracts differ: LITE retained 249
frames (9.96 s) and PRO retained 229 frames (9.16 s). The [media validation
record](media-validation.json) preserves those facts rather than pretending the
outputs are frame-identical.

## Why PRO helps

PRO is not just the LITE DiT with a setting changed. The local configs show:

| Variant | VAE stride | DiT patch | Latent channels | Chunk contract |
| --- | --- | --- | ---: | --- |
| LITE | 8x32x32 | 1x1x1 | 128 | 9 overlap / 24 new |
| PRO | 4x8x8 | 1x2x2 | 16 | 5 overlap / 28 new |

PRO keeps a much denser spatial representation and uses the Wan2.1 VAE. This
experiment proves that the released PRO path improves visible teeth detail, but
it cannot isolate whether the gain comes from the denser latent grid, the Wan
decoder, or their interaction. That is the architectural lead to carry into
SoulX: preserve more spatial information before the final face/teeth rendering
stage rather than only increasing denoising steps.

## Evidence

- [LITE video](lite/video.mp4), [PRO video](pro/video.mp4)
- [native-pixel mouth crops](mouth-comparison.png)
- [full-frame contact sheet](frames-comparison.png)
- [CPU diagnostics](diagnostics.json), [media validation](media-validation.json)
- [GPU runner](run_variant.py), [analysis](analyze.py), [packager](package.py)

# SoulX FlashHead PRO versus LITE at 1.50x framing

## Result

PRO remains sharper than LITE on the closest 1.50x Indian-male reference. Its
median whole-mouth edge energy is 110.34 versus 52.51 (2.10x), while central
oral edge energy on open-mouth frames is 380.68 versus 137.67 (2.77x).

The closer face makes the qualitative result easier to judge. PRO shows useful
individual-tooth divisions in several frames, especially around 1.5, 4.5 and
6.5 seconds. It still merges the teeth into a bright connected region around
3.5 and 8.5 seconds. LITE remains softer and grayer throughout. The closest
framing therefore helps expose real PRO detail, but it does not eliminate the
connected-tooth-band failure.

The generated mouth motion is different between variants. PRO's median
normalized opening is 0.0836 versus LITE's 0.0968, with p95 values of 0.1745
and 0.2550. Their opening trajectories correlate at 0.724. These are released
model outputs with different VAEs and latent geometry, not a decoder-only
ablation.

## Controlled generation

- Exact retained 1.50x reference: `reference-150x.png`, SHA256
  `a0a2317609afe70ed6c3fa3c620654dc56b123981429b2316dc6f7cc8222d676`.
- Exact shared ten-second waveform: `audio.wav`.
- 320x576, 25 FPS, 250 frames, four steps, seed 50, BF16 eager execution.
- Stock audio conditioning and color correction; no sharpening, mouth-SR,
  refinement or post-generation enhancement.
- NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM, driver 595.84,
  Torch 2.7.1+cu128 and CUDA runtime 12.8. Existing co-resident processes were
  left running.

## Resource telemetry during generation

The runner sampled every 0.5 seconds only while the generation loop was active.
GPU utilization and VRAM are whole-device readings and therefore include the
roughly 2.7 GiB co-resident baseline. Torch allocation/reservation and process
RSS isolate the SoulX process. System CPU is normalized over 32 logical CPUs;
process CPU uses the usual one-core-equals-100% convention. The initial idle
sample is retained in each average.

| Metric | LITE mean | LITE peak | PRO mean | PRO peak |
| --- | ---: | ---: | ---: | ---: |
| GPU utilization | 87.0% | 100.0% | 98.3% | 100.0% |
| Whole-device VRAM | 8,262 MiB | 8,264 MiB | 9,536 MiB | 9,582 MiB |
| Torch allocated | 4,134 MiB | 4,787 MiB* | 3,900 MiB | 5,397 MiB* |
| Torch reserved | 5,301 MiB | 5,302 MiB | 6,562 MiB | 6,620 MiB |
| System RAM used | 11,272 MiB | 11,306 MiB | 11,186 MiB | 11,386 MiB |
| SoulX process RSS | 4,607 MiB | 4,625 MiB | 4,516 MiB | 4,532 MiB |
| System CPU | 5.3% | 9.7% | 6.4% | 12.8% |
| SoulX process CPU | 88.7% | 100.5% | 98.6% | 101.9% |

`*` Torch allocated peak comes from PyTorch's exact peak counter; the other
means and peaks come from the half-second samples. The raw timeline is embedded
under `resource_sampling.samples` in each variant's `results.json`.

| Variant | Generation time | Useful FPS | Telemetry samples |
| --- | ---: | ---: | ---: |
| LITE | 4.45 s | 56.20 | 9 |
| PRO | 34.93 s | 7.16 | 65 |

PRO is 7.85x slower in this matched eager run. Its 9,582-MiB whole-device peak
fits the card with about 2.6 GiB of reported physical headroom, but 7.16 FPS is
well below the 25-FPS real-time target.

## Evidence

- [LITE video](lite/video.mp4) and [run record with telemetry](lite/results.json)
- [PRO video](pro/video.mp4) and [run record with telemetry](pro/results.json)
- [labeled comparison](soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50.mp4)
- [native-pixel mouth crops](mouth-comparison.png)
- [full-frame contact sheet](frames-comparison.png)
- [per-frame diagnostics](diagnostics.json) and [media validation](media-validation.json)


# PRO320 transformer partition experiments

**Apple M5 Pro GPU**, 24 GiB physical unified memory, macOS 27.0.1 build 26A434, November 2, 2026. The Mac native Core ML runner recorded the Metal device name, physical system memory, and 19,069,665,280-byte recommended Metal working set. The device shares system memory. The working set is a budget. The end-to-end comparisons requested CPU and GPU. Processor placement was unverified. The project serialized GPU benchmarks while other agents performed CPU checks. Other application load was unmeasured.

**Keep the default three-block export.** Combining attention projections preserved every compared pixel and gave no measurable end-to-end throughput gain on this fixture. Combining the projections and exporting six layers per partition gave a 1.65% pooled warm FPS gain on this Mac, with nearly twice the on-disk size per loaded partition. Physical iPhone latency and resident memory for the larger partition remain unmeasured.

The runner used the same compiled Swift binary, cached portrait features, active-speech audio, random seed, 320×176 profile, 30 layers and two denoising steps. Each comparison alternated six processes in A–B–B–A–A–B order. Each process generated four windows of 28 frames. Warm FPS pools windows 2–4 across three runs of each variant. Every comparison used its own contemporary baseline.

| Mac Core ML comparison on Apple M5 Pro | Baseline warm FPS | Candidate warm FPS | Change | Identical corresponding RGB frames |
|---|---:|---:|---:|---:|
| Three blocks, fused Q/K/V and audio K/V | 10.856 | 10.846 | −0.09% | 336 / 336 |
| Six blocks, fused Q/K/V and audio K/V | 10.719 | 10.896 | +1.65% | 336 / 336 |

Fusion joins checkpoint output rows before Core ML conversion. It retains every trained projection, INT8 per-output-channel weight compression, and the mixed precision policy. A three-block graph contains 21 learned linear operations after fusion, down from 30. Six-block packages reduce the number of transformer predictions per two-step window from 20 to 10. The largest package increases from 133.80 MiB to 267.07 MiB on disk. The six-block experiment changed both fusion and partition size.

All generated RGB pixels matched across each pair, including a fixed mouth rectangle at x=62…131, y=186…235 in the 176×320 frames. These partition tests used the historical Mac audio path. Subsequent diagnostics found that this path produced a constant projected context on this host. The pixel comparisons establish equivalence between partition layouts for that input. The runtime benchmark uses validated speech features for its character-quality comparison. [Raw timings, source payload hashes, and frame comparisons](dit_partition_trials_mac.json)

CPU-only numerical validation on the Apple M5 Pro Mac used PyTorch and Core ML CPU_ONLY, with no GPU memory used for the numerical check. Three-block fusion passed all 24 cases against the FP32 reference. Six-block grouping exceeded the 5% FP32 normalized-RMSE threshold in three of fourteen cases, at 5.30%, 5.74% and 6.61%. The validator also ran sequential pairs of the existing three-block packages on the same inputs. All 30,412,800 compared output elements matched the six-block graphs exactly, including at all three absolute-gate failures. The report preserves both measurements.

To reproduce the experimental DiT export from the repository root, run the following commands with the existing checkpoint and baseline pro320 packages. The second command writes the paired numerical report and returns a failing status for the three FP32 cases above.

```sh
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.export_dit \
  --height 320 --width 176 --blocks-per-chunk 6 --fuse-qkv \
  --output artifacts/coreml/pro320-fused6
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.validate_dit \
  --checkpoint models/SoulX-FlashHead-1_3B/Model_Pro \
  --models artifacts/coreml/pro320-fused6 --height 320 --width 176 \
  --blocks-per-chunk 6 --compare-blocks artifacts/coreml/pro320
```

A separate three-block probe replaced linear projections with channels-first 1×1 convolutions and kept the precision policy. Apple's transformer deployment research motivated that layout. [Apple Neural Engine transformer study](https://machinelearning.apple.com/research/neural-engine-transformers)

Under matched Core ML .all execution on the Apple M5 Pro Mac, MLComputePlan preferred the GPU for all 30 projections in both forms. The pooled median resident prediction was 47.06 ms for linear and 56.20 ms for convolution over 24 timed predictions per variant. The convolution probe stays outside the exporter. [Compute plans and latency samples](dit_conv1x1_probe_mac.json)

For a physical-device comparison, baseline the current three-block bundle and test the six-block bundle on the same portrait and audio with the same model residency and compute units. Measure first-window latency, later-window generation FPS, rendered frame agreement, peak process footprint and thermal state. The current three-block bundle retains the smaller partition footprint while this test is pending.

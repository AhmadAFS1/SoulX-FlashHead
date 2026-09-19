# Evidence synthesis: what limits SoulX tooth detail?

2026-09-17. **CPU-only documentation/source audit; no new GPU inference.**
The principal September 17 experiments reused below ran on **NVIDIA GeForce
RTX 4070 SUPER**, physical 12 GB class / 12,282 MiB visible (`nvidia-smi`),
driver 595.84, Torch 2.7.1+cu128 / CUDA runtime 12.8. Workload, allocator caps
and co-resident load differ and remain in each linked report. Most ran beside
OmniVoice; some also had an idle or active LTX process. No cross-report timing
comparison is made. September 13 evidence has retrospective **RTX 4070**
attribution, not a verified per-run SUPER snapshot. Historical LTX fork runs
record **RTX 5060 Ti**, approximately 16 GB class; their environment is distinct.

## Diagnosis

**The strongest working diagnosis is a limit in synthesizing and maintaining
fine dental structure in SoulX Lite's compressed generative video path.**
Insufficient real source detail and small mouth scale demonstrably aggravate
that limit. At the better tested source/framing settings, residual fused teeth
remain, and the tested extra denoising does not reliably restore them.

This localizes the problem to the neural image-generation path before delivery
encoding. It does **not** isolate a single failed component inside that path.
The DiT's learned predictions, VAE behavior on motion, reference encoding and
recurrent decode/re-encode feedback have not been independently separated.
Calling it exclusively a decoder failure, a distillation failure, a lack of
parameters or an insufficient number of steps would exceed the evidence.

### Subsequent learned-output enhancement test

The [mouth-SR follow-up](../mouth_sr_20260917/README.md) verifies the external
Lite/SR report and Ojin component manifest, then implements tracked mouth
super-resolution and feathered blending. Fresh RTX 4070 SUPER GPU runs
(12,282 MiB visible, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8; OmniVoice and
idle LTX resident) show stronger contours with a generic learned SR model,
but residual fused/artificial teeth remain. The native mouth-only integration
measured 24.25 FPS at 320×576 and preserved the underlying SoulX frames exactly.
This adds a concrete output-stage mitigation; it is distinct from earlier
interpolation-only reference enlargements and from extra latent sampling.
Ojin's dedicated refiner is proprietary and was not reproduced. Its manifest
does not establish how much quality improvement that refiner provides.

## Evidence by hypothesis

| Candidate bottleneck | What the retained evidence establishes | Assessment |
| --- | --- | --- |
| Real source detail | At fixed 512 output, reducing the same 307-pixel crop to 128/64 causes visible face/teeth blur; 307/256 look close. A separate 320×576 portrait test repeats the pattern. | Strong causal contributor when input is poor; no proof that current native source exhausts possible input improvements. |
| Mouth scale in the generated canvas | Closer crops improve the observed mouth; Lite generates the whole frame with a 32× spatially compressed representation. Crops also change conditioning and motion. | Strong practical factor; exact mechanism/threshold not isolated. |
| Lite learned detail generation | Sharp-source baseline, BF16, smiling references and extra sampling all retain fused/simplified teeth. | Leading explanation for the remaining quality ceiling, rather than a directly isolated DiT defect. |
| VAE compression | Upstream collaborator identifies a speed/detail tradeoff. Our VAE-only repeated-still reconstruction retains separated teeth. | Plausible contributor, especially in motion; not a universal inability to represent teeth. |
| Recurrent feedback | Last nine corrected frames are encoded into the next chunk. High-resolution and extreme-history configurations can progressively collapse. | Plausible amplifier; these interventions change other variables and do not prove ordinary recurrence is the primary cause. |
| Too few denoising steps | Four vs six continuous evaluations and four different 4+2 restarts, two seeds, show no consistent fix. | Tested additions fail; 10/16 steps have not been run. |
| INT8 storage | Matched BF16/INT8 raw clips both show the defect, across two seeds and two resolutions. | Not necessary for the core defect; can still introduce numerical changes. |
| H264/WebRTC | Defect exists in raw pre-encoding RGB; codec roundtrip adds some softening. | Secondary loss, not the originating cause in those fixtures. |
| Mouth amplitude / reference smile / seed | These change exposure, articulation and pose; open-mouth dental defects persist. | Appearance controls, not demonstrated detail reconstruction. |
| Nominal image/output resolution | Upscaling the original reference creates no captured detail; both 1024-square runs degraded severely compared with 512. | More nominal pixels are not a reliable remedy. |
| Low-VRAM offload/chunking | FFN chunking did not repair teeth; more memory is not itself new visual information. | No demonstrated dental mechanism. |

## Why the latent representation is relevant, without overstating it

The installed Lite config records spatial stride 32, temporal stride 8 and 128
latent channels. A 320×576 portrait becomes a 10×18 spatial latent grid; at
512×512 it is 16×16. A measured roughly 70-pixel mouth spans only about 2.2
latent spacings at the former profile. These are **not two RGB pixels**:
channels and decoder receptive fields can carry sub-cell detail. The compression
ratio alone cannot prove that individual teeth are impossible to reconstruct.

The portrait latent supplies appearance while audio supplies speech conditioning.
Neither is an exact per-frame dental target. The DiT predicts the moving latent
sequence, and the VAE turns it into frames. Generating stable tooth boundaries
through changing mouth openings is a harder requirement than copying one
encoded still. Repeating the same learned prediction process with extra noise
does not necessarily supply missing anatomical information.

This is an interpretation of the source and experiments, not a measurement of
the training loss or proof that a specific capacity limit was reached.

## Reconciling the VAE explanations

The April 7 upstream collaborator response says that Lite **may** lose detail
because of the LTX VAE's high compression and recommends Pro for greater detail.
The response was re-fetched through GitHub's public API during this audit:
[issue 21 comment](https://github.com/Soul-AILab/SoulX-FlashHead/issues/21#issuecomment-4196878092).
This is an upstream explanation, not a matched local Pro benchmark.

Our local diagnostic encodes nine copies of one clear-teeth LTX frame, then
decodes with SoulX's unchanged VAE at 480×832. Separated upper/lower teeth survive.
That weakens an absolute decoder-capacity explanation, but the fixture is a
different face/scale, has no changing speech, no DiT, no color correction and
no repeated chunk feedback. It cannot clear the VAE for the actual moving
320×576 avatar. Both observations can be true simultaneously.

## What the LTX fork does and does not establish

The later local graph audit supersedes the fork's early attribution:

- NAG is active and changes mouth/tooth appearance, but is not uniformly sharper.
- The clean-latent 6+2 restart changes some details; transferring a related
  refinement mechanism to SoulX did not produce a reliable teeth improvement.
- Prompt Relay temporal patches are overwritten by NAG in the pinned graph.
- The historical Q3/Q4 comparison also changes model revision; it is not a
  pure quantization experiment.
- The historical winning graph has the same locked video VAE as the older graph,
  no dedicated dental model and no spatial upscaler. Later accepted selfie
  variants additionally change prompts/audio-to-video scaling; they are separate
  recipes, not controlled proof of one transferable fix.
- Local LTX can produce separated teeth without NAG or a restart. Its different
  generator, conditioning and joint audio/video task prevent attributing that
  advantage to one node or memory optimization.

## Most discriminating next diagnostic

Use a genuinely sharp **moving** teeth clip at the target 320×576 geometry and
matched mouth scale. Compare the original with (1) one SoulX VAE roundtrip and
(2) controlled repeated encode/decode cycles using the actual motion-history
windowing, with RNG streams matched. Separately capture generated frames before
and after color correction and at recurrent chunk boundaries.

If moving reconstruction is already poor, the VAE/reference/feedback path deserves
priority. If it remains sharp while generated output is soft, the learned
DiT/conditioning path becomes the stronger target. If repeated cycles alone
degrade, feedback contributes. These tests still need multiple clips and careful
temporal boundaries; a simple repeatedly compressed full movie is not identical
to SoulX's streaming recurrence.

This is more diagnostic than another unstructured step-count sweep. Pro or a
trained detail-restoration approach may be worth evaluating afterward, but no
local result here proves either will solve this avatar's teeth.

## Source trail and review scope

All project Markdown files were inventoried and searched for quality evidence;
relevant quality/architecture reports were read in depth. Dependency/model/cache
Markdown is excluded. [Inventory and hashes](markdown-inventory.json) cover
61 SoulX files and the LTX fork documents as they stood before this synthesis.
Repeated index summaries are not counted as independent experiments. Missing
historical artifacts remain historical report evidence; they were not rerendered
or claimed as newly inspected media.

Primary local sources:

- [BF16/INT8 and raw/codec controls](../../docs/research/MALE_TEETH_QUALITY_2026-09-13.md)
- [Still versus idle motion](../../docs/research/STILL_REFERENCE_QUALITY_2026-09-13.md)
- [Source-detail isolation](../source_detail_20260917/README.md)
- [Shoulder-visible source-detail replication](../portrait_source_detail_20260917/README.md)
- [Smiling reference](../smile_reference_320_20260917/README.md)
- [512 versus 1024](../square_teeth_20260917/README.md)
- [LTX transfer and VAE roundtrip](../ltx_transfer_20260917/README.md)
- [Corrected LTX graph/component findings](../ltx_transfer_20260917/ROOT_COMPONENTS.md)
- [SoulX refinement controls](../refinement_20260917/README.md)
- [Movement sweeps](../movement_ranges_20260917/README.md)
- [Audio/DiT tensor contract](../../docs/architecture/02_AUDIO_AND_DIT.md)
- [VAE and recurrent continuity](../../docs/architecture/03_VAE_AND_CONTINUITY.md)

Broad mouth edge energy, landmark detections, delivery success and increased
tooth exposure are not dental quality scores. No percentages of root-cause
responsibility are supported by these reports.

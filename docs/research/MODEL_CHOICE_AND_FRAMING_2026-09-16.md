# Model preference, framing, and Ditto expression strength

Evidence host for the September 16 experiments: **NVIDIA GeForce RTX 4070 SUPER**, 12 GB class / 12,282 MiB visible, driver 595.84. Fresh Ditto 0.7 inference uses Torch 2.5.1+cu121 / CUDA 12.1 and TensorRT Ampere Plus; the prior SoulX framing runs used Torch 2.7.1+cu128 / CUDA 12.8. Current Ditto runs recorded 2,487 MiB allocated before loading, with workload identity unverified. [Run metadata](../../benchmarks/distance_lipsync/evidence-ditto-closer-0p7-20260916/results.json) is direct evidence. Resource comparisons below are explicitly reused historical measurements, not new measurements under that co-resident load. Source inspection and statistical analysis are not GPU performance tests.

## User review and working preference

Further investigation: [head-motion controls](HEAD_MOTION_CONTROL_2026-09-16.md) confirms that the 1.25× SoulX crop moves more in the retained clips; [concurrency potential](CONCURRENCY_POTENTIAL_2026-09-16.md) reconciles the current 35-FPS profile with historical SoulX and multi-GPU MuseTalk tests. These are CPU reanalysis/source audits, not additional throughput benchmarks or proof of a universal model ranking.

Recorded September 16 after reviewing the existing Indian-male comparisons, before reviewing the new 0.7 outputs:

| Model | User's qualitative assessment | Practical interpretation and qualification |
| --- | --- | --- |
| SoulX-FlashHead Lite | Best overall bang for buck; closer framing makes a major visual difference | Preferred direction for these conversational-avatar experiments. 1.25× is a strong composition; the user considers 1.50× ideal for the close-up look, accepting reduced headroom. This is a preference, not a cost-per-session benchmark or serving approval. |
| MuseTalk | Most accurate-looking lip sync, but needs base videos to feel suitable for real-time conversation | The tested workflow animates the mouth over prepared avatar frames. A repeated still can technically supply those frames; suitable idle/listening/head-motion footage is needed for the user's desired natural conversational presentation, not because inference universally requires pre-generated moving footage. |
| Ditto TalkingHead | Lowest hardware burden; clean output, but lip movements feel robotic/stiff compared with the others; closer framing looks essentially the same | Lowest measured peak GPU memory among the tested profiles. Framing consistency is corroborated by the motion proxies below. Clean appearance and stiffness are user judgments, not calibrated accuracy scores. |

No blinded study, phoneme-level scoring, or deployment cost study has established an absolute ranking. These findings do not supersede the historical production/transport acceptance gates in [the test catalog](TEST_CATALOG.md). In particular, user preference for SoulX does not imply that its earlier concurrency, streaming, or boundary-quality failures are resolved.

## Hardware evidence: lower memory, not proven cheapest operation

Reused September 16 same-avatar resource test on **RTX 4070 SUPER, 12,282 MiB**; 384×672, ten-second audio, one model at a time, device-wide `nvidia-smi` samples every 100 ms:

| Profile | Recorded peak device memory | Measurement window |
| --- | ---: | --- |
| MuseTalk: TRT UNet + mixed INT8 VAE, batch 8 | 10,037 MiB | 12.148-second WebRTC request, including negotiation/pacing |
| SoulX Lite: compiled, four steps, batch 1 | 6,165 MiB | 12.272-second WebRTC request, including negotiation/pacing |
| Ditto: TensorRT Ampere Plus | 2,581 MiB | 8.601-second offline inference, after loading |

The [preserved resource summary](../../benchmarks/distance_lipsync/evidence-ditto-closer-0p7-20260916/prior-resource-summary.json) is copied unchanged from `/workspace/benchmarks/same-avatar/resource-usage/summary.json`; its original README states models ran one at a time. The earlier same-avatar report records driver 595.84 and runtimes: MuseTalk/Ditto Torch 2.5.1+cu121 / CUDA 12.1, SoulX Torch 2.7.1+cu128 / CUDA 12.8; the resource summary itself does not independently repeat driver/runtime versions. No raw resource test was rerun for this write-up.

These are profile-specific device peaks, not dedicated model weights or minimum supported VRAM. Batch sizes, quantization, and transport differ. Ditto's lower memory supports the user's hardware impression, but power averages and latency are not directly comparable across paced WebRTC and offline windows. No smaller GPU, concurrent-user capacity, or monetary cost was tested. The resource-test audio/geometry also differ from the current 320×576 framing fixture: do not merge their timings into one controlled study.

## Why zoom changes SoulX more than Ditto

**Observed:** at Ditto 0.5, tighter framings retain 0.991/0.988 normalized mouth-trajectory correlation with 1.00×; at 0.7, they retain 0.995/0.993. At 0.7, mean normalized opening remains approximately 0.060–0.061 across the three framings. This agrees with the user's observation of little change in articulation, although displayed face size changes substantially. It is not literally identical pixels or zero numerical difference.

**Verified implementation difference:** Ditto detects landmarks and normalizes the face into a 512×512 crop at a fixed landmark-relative scale. Both source motion and appearance extraction receive a resized 256×256 crop. Its generated face is later transformed back using the inverse crop transform `M_c2o` and blended into the original frame. A 1.50× source zoom therefore mostly changes the size at which the same normalized facial animation is displayed; it does not allocate a larger internal face representation.

This SoulX engine instead sets `use_face_crop=False`, resizes/center-crops the whole reference to the output dimensions, and generates the full frame. The installed Lite configuration has spatial VAE stride 32 and unit latent patch sizes. At 320×576 the latent grid is 10×18. Moving the face closer within that fixed canvas allocates more output pixels and latent positions to facial details and less to background/torso.

**Inference, not an isolated causal proof:** the different normalization strategies are a strong explanation for the observed scale sensitivity. The earlier SoulX male far-shot experiment reduced detected face width to about 67 px versus 133 px at baseline, with a 92.2% reduction in mean normalized mouth opening. A roughly 200-px close-up gives SoulX substantially more spatial representation for the face. Dividing these widths by 32 gives approximately 2.1/4.2/6.2 latent spacings—an intuition, not a hard mouth-resolution threshold: latent channels and the decoder encode detail within each spatial position.

Evidence: [saved local source excerpts and file hashes](../../benchmarks/distance_lipsync/evidence-ditto-closer-0p7-20260916/source-audit.json), covering Ditto `source2info.py`, `putback.py`, and SoulX `engine.py`, `flash_head_pipeline.py`, and the actual Lite config. Upstream context confirms the different model families: [Ditto motion-space diffusion](https://github.com/antgroup/ditto-talkinghead) and [SoulX, with LTX-VAE in Lite](https://github.com/Soul-AILab/SoulX-FlashHead). The local source, not upstream performance advertising, governs this explanation.

Important boundaries:

- Ditto cannot recover detail absent from a truly tiny, blurry, occluded, or misdetected source face. The tested range is 1.00–1.50× close framing; no Ditto far-shot replication was performed.
- Source zoom here is a crop of the same portrait, not a new camera capture with genuinely increased source detail. Ditto's normalized crop largely cancels that crop/zoom; SoulX's fixed full-frame layout does not.
- SoulX framing also changes composition and learned pose/appearance conditioning. The earlier synthetic far-shot composition adds confounds. A crop-normalization ablation or matched face-pixel/resolution test is needed to isolate the mechanism.
- “Closer is always better” is not established. Tight cropping can remove hair/chin, cause border artifacts, or leave the model's comfortable composition range. The user's 1.25×/1.50× preference is supported for these examples, not all faces, models, or resolutions.
- Correlation with another generated clip is motion agreement, not proof of correct spoken phonemes; larger opening is not automatically better lip sync.

## Why Ditto may feel stiff, and the new 0.7 test

At 0.5, Ditto blends its generated expression halfway toward the source expression. Despite a lip-mask variable in `ctrl_vad`, the implemented assignment applies to the entire expression vector. The runtime also uses a three-frame moving-average motion smoother, which can attenuate rapid changes. These are plausible contributors to subdued articulation, alongside learned motion/renderer constraints; no smoothing ablation or independent stiffness test was run. Face normalization explains the framing invariance, not necessarily the subjective stiffness by itself.

The new controlled change retains 70% of generated expression (`vad_alpha=0.7`) instead of 50%. Mean measured mouth opening rises **37.4–39.5%**, and p95 rises **26.5–28.2%**, across the three framings. The pattern remains when excluding the final 15 fade frames. Thus alpha is a more effective amplitude control than zoom in these Ditto examples. It does not change the phoneme predictor or establish more natural motion; the six-panel video is for the user's direct judgment. No serving default was changed.

Full results, clips, controls, limitations, and reproduction: [Ditto 0.7 report](../../benchmarks/distance_lipsync/DITTO_CLOSER_0P7_REPORT.md). Earlier evidence: [Ditto 0.5](../../benchmarks/distance_lipsync/DITTO_CLOSER_0P5_REPORT.md), [SoulX male distances](../../benchmarks/distance_lipsync/INDIAN_MALE_REPORT.md), [SoulX tighter framing](../../benchmarks/distance_lipsync/INDIAN_MALE_CLOSER_REPORT.md), and [SoulX strength 0.5](../../benchmarks/distance_lipsync/INDIAN_MALE_CLOSER_STRENGTH_0P5_REPORT.md).

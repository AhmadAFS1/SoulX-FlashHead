# Complete tracked-source inventory

Audit baseline: `ac2c2bbb7308f87880c2ae4b01563d1e4d203c02`, 2026-09-07. [Guide index](README.md).

Selection: `git ls-files '*.py' '*.sh' '*.html' '*.yaml' '*requirements*.txt'`. This includes tracked vendored model files even where ordinary ignore-aware searches omit them. Weights, environments, caches, media and existing Markdown are outside this source count. Empty package initializers are explicitly listed.

**74 files; 14600 physical lines.** SHA-256 hashes cover exact file bytes. Symbols list module-level functions/classes and class methods (including async methods), with baseline source line numbers; nested local functions are covered by their enclosing source, not separately inventoried. Line numbers are a navigation aid, not stable identifiers after code edits.

The walkthrough explains active paths in detail and labels alternate/dormant libraries. A complete file inventory does not claim every optional branch was executed. No new GPU execution occurred in this audit.

## flash_head/audio_analysis/torch_utils.py

[Open source](../../flash_head/audio_analysis/torch_utils.py) · 20 physical lines · SHA-256 `5a523b78211a64d7efd838cc000c6251184de2a37c73614e2581faa5d09378cb`

Audio feature interpolation utilities; used to align convolutional features with video time.

`get_mask_from_lengths` (L5); `linear_interpolation` (L16).

## flash_head/audio_analysis/wav2vec2.py

[Open source](../../flash_head/audio_analysis/wav2vec2.py) · 125 physical lines · SHA-256 `c30cb3548889e52542d63e91b0301599f3dead5de97b7b69e144aa6232fa9476`

Custom Wav2Vec forward paths; interpolate features before transformer and expose hidden layers for audio conditioning.

`Wav2Vec2Model` (L9); `Wav2Vec2Model.__init__` (L10); `Wav2Vec2Model.forward` (L13); `Wav2Vec2Model.feature_extract` (L67); `Wav2Vec2Model.encode` (L78).

## flash_head/configs/infer_params.yaml

[Open source](../../flash_head/configs/infer_params.yaml) · 10 physical lines · SHA-256 `9c5111fa5ba2ebf20da09d3b6806e88ff952955fb82c6c3f74cefcc150bd7b71`

Upstream Lite/Pro/teacher presets: temporal windows, denoising steps, shift and guidance; RTC overrides some values.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/inference.py

[Open source](../../flash_head/inference.py) · 77 physical lines · SHA-256 `9c5b0be94ab38bb76dfce32e6c743165a9586a33d0f5144b3f63a10c251a819d`

Factory and helper façade: select variant/parallel layout, prepare reference/audio and invoke pipeline.

`get_pipeline` (L13); `get_base_data` (L39); `get_infer_params` (L52); `get_audio_embedding` (L56); `run_pipeline` (L72).

## flash_head/ltx_video/__init__.py

[Open source](../../flash_head/ltx_video/__init__.py) · 0 physical lines · SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Empty package marker; no runtime algorithm.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/ltx_video/ltx_vae.py

[Open source](../../flash_head/ltx_video/ltx_vae.py) · 42 physical lines · SHA-256 `b08681ad574077d7a5caade8a0bd0ade3097db5bbb9979fc0356bd80fb970d51`

Active Lite VAE wrapper: instantiate checkpoint-specific causal VAE; posterior sampling and per-channel latent normalization.

`LtxVAE` (L5); `LtxVAE.__init__` (L6); `LtxVAE.encode` (L16); `LtxVAE.decode` (L22); `LtxVAE.normalize_latents` (L31); `LtxVAE.un_normalize_latents` (L38).

## flash_head/ltx_video/models/__init__.py

[Open source](../../flash_head/ltx_video/models/__init__.py) · 0 physical lines · SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Empty package marker; no runtime algorithm.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/ltx_video/models/autoencoders/__init__.py

[Open source](../../flash_head/ltx_video/models/autoencoders/__init__.py) · 0 physical lines · SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Empty package marker; no runtime algorithm.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/ltx_video/models/autoencoders/causal_conv3d.py

[Open source](../../flash_head/ltx_video/models/autoencoders/causal_conv3d.py) · 63 physical lines · SHA-256 `26fcd3358ce706cc9afb9e20badaa307d700652fd403e977d1da40bc4f629654`

Active causal/noncausal temporal padding plus Conv3d wrapper; edge replication influences tiling/cache correctness.

`CausalConv3d` (L7); `CausalConv3d.__init__` (L8); `CausalConv3d.forward` (L44); `CausalConv3d.weight` (L62).

## flash_head/ltx_video/models/autoencoders/causal_video_autoencoder.py

[Open source](../../flash_head/ltx_video/models/autoencoders/causal_video_autoencoder.py) · 1412 physical lines · SHA-256 `719e51bdedcdceffd446d67675a5d04ff3cde628b262d41dfe04f9ec6b135839`

Active Lite encoder/decoder, residual blocks, all-axis compression/expansion, patchification, configuration loading; many optional blocks are inactive.

`CausalVideoAutoencoder` (L32); `CausalVideoAutoencoder.from_pretrained` (L34); `CausalVideoAutoencoder.from_config` (L121); `CausalVideoAutoencoder.config` (L178); `CausalVideoAutoencoder.is_video_supported` (L199); `CausalVideoAutoencoder.spatial_downscale_factor` (L206); `CausalVideoAutoencoder.temporal_downscale_factor` (L226); `CausalVideoAutoencoder.to_json_string` (L241); `CausalVideoAutoencoder.load_state_dict` (L246); `CausalVideoAutoencoder.last_layer` (L296); `CausalVideoAutoencoder.set_use_tpu_flash_attention` (L306); `Encoder` (L313); `Encoder.__init__` (L338); `Encoder.forward` (L506); `Decoder` (L556); `Decoder.__init__` (L581); `Decoder.forward` (L731); `UNetMidBlock3D` (L801); `UNetMidBlock3D.__init__` (L827); `UNetMidBlock3D.forward` (L893); `SpaceToDepthDownsample` (L972); `SpaceToDepthDownsample.__init__` (L973); `SpaceToDepthDownsample.forward` (L987); `DepthToSpaceUpsample` (L1019); `DepthToSpaceUpsample.__init__` (L1020); `DepthToSpaceUpsample.forward` (L1046); `LayerNorm` (L1075); `LayerNorm.__init__` (L1076); `LayerNorm.forward` (L1080); `ResnetBlock3D` (L1087); `ResnetBlock3D.__init__` (L1100); `ResnetBlock3D._feed_spatial_noise` (L1190); `ResnetBlock3D.forward` (L1204); `patchify` (L1268); `unpatchify` (L1289); `create_video_autoencoder_demo_config` (L1309); `test_vae_patchify_unpatchify` (L1348); `demo_video_autoencoder_forward_backward` (L1357).

## flash_head/ltx_video/models/autoencoders/conv_nd_factory.py

[Open source](../../flash_head/ltx_video/models/autoencoders/conv_nd_factory.py) · 90 physical lines · SHA-256 `641fae81b0e41615f8f32b100ec9efdd748d825ea825d166aa6414912626259d`

Convolution/pooling factory shared by model variants; dispatches dimensionality and causal options.

`make_conv_nd` (L9); `make_linear_nd` (L75).

## flash_head/ltx_video/models/autoencoders/dual_conv3d.py

[Open source](../../flash_head/ltx_video/models/autoencoders/dual_conv3d.py) · 217 physical lines · SHA-256 `2683701b800960aac74766f03cb4bc7a3cb10b80e881422e63cf5d6f5896b2c9`

Factorized spatial/temporal convolution for alternate architecture; not the installed Lite convolution path.

`DualConv3d` (L10); `DualConv3d.__init__` (L11); `DualConv3d.reset_parameters` (L86); `DualConv3d.forward` (L97); `DualConv3d.forward_with_3d` (L103); `DualConv3d.forward_with_2d` (L133); `DualConv3d.weight` (L185); `test_dual_conv3d_consistency` (L189).

## flash_head/ltx_video/models/autoencoders/pixel_norm.py

[Open source](../../flash_head/ltx_video/models/autoencoders/pixel_norm.py) · 12 physical lines · SHA-256 `37163441070d3ea85bcccf9a603bc1f19a4ee8e9da2afadfa8bba7a1db6e96da`

Channelwise RMS normalization helper; incoming dtype affects numerical range.

`PixelNorm` (L5); `PixelNorm.__init__` (L6); `PixelNorm.forward` (L11).

## flash_head/ltx_video/models/autoencoders/vae.py

[Open source](../../flash_head/ltx_video/models/autoencoders/vae.py) · 380 physical lines · SHA-256 `db6b9fa8a19ab528215b10c3a328d27ae69978fc5ff860ce6d0a796000e3cbb4`

Generic AutoencoderKL base, Gaussian posterior plumbing and spatial/temporal tiling; tiling assumptions do not match installed Lite.

`AutoencoderKLWrapper` (L16); `AutoencoderKLWrapper.__init__` (L31); `AutoencoderKLWrapper.set_tiling_params` (L79); `AutoencoderKLWrapper.enable_z_tiling` (L85); `AutoencoderKLWrapper.disable_z_tiling` (L98); `AutoencoderKLWrapper.enable_hw_tiling` (L105); `AutoencoderKLWrapper.disable_hw_tiling` (L111); `AutoencoderKLWrapper._hw_tiled_encode` (L117); `AutoencoderKLWrapper.blend_z` (L154); `AutoencoderKLWrapper.blend_v` (L164); `AutoencoderKLWrapper.blend_h` (L174); `AutoencoderKLWrapper._hw_tiled_decode` (L184); `AutoencoderKLWrapper.encode` (L226); `AutoencoderKLWrapper._normalize_latent_channels` (L261); `AutoencoderKLWrapper._unnormalize_latent_channels` (L275); `AutoencoderKLWrapper._encode` (L286); `AutoencoderKLWrapper._decode` (L292); `AutoencoderKLWrapper.decode` (L306); `AutoencoderKLWrapper.forward` (L352).

## flash_head/ltx_video/models/autoencoders/vae_encode.py

[Open source](../../flash_head/ltx_video/models/autoencoders/vae_encode.py) · 256 physical lines · SHA-256 `229c202d1a34aceceae4846d63dd28d8ed9ed48edcc03dc3d03d15702ff9423e`

Generic encode/decode/batch/scaling helpers; inspect selected wrapper before transferring generic behavior.

`vae_encode` (L22); `vae_decode` (L96); `_run_decoder` (L138); `get_vae_size_scale_factor` (L175); `latent_to_pixel_coords` (L198); `latent_to_pixel_coords_from_factors` (L224); `normalize_latents` (L237); `un_normalize_latents` (L248).

## flash_head/ltx_video/models/autoencoders/video_autoencoder.py

[Open source](../../flash_head/ltx_video/models/autoencoders/video_autoencoder.py) · 1045 physical lines · SHA-256 `5a9ab73be00bce34defb7742bb758312add2deb5e8fc94a90cd61dd5a51507bc`

Alternative video autoencoder using a different block/stride scheme; not the selected Lite causal checkpoint architecture.

`VideoAutoencoder` (L22); `VideoAutoencoder.from_pretrained` (L24); `VideoAutoencoder.from_config` (L61); `VideoAutoencoder.config` (L112); `VideoAutoencoder.is_video_supported` (L135); `VideoAutoencoder.downscale_factor` (L142); `VideoAutoencoder.to_json_string` (L145); `VideoAutoencoder.load_state_dict` (L150); `VideoAutoencoder.last_layer` (L174); `Encoder` (L185); `Encoder.__init__` (L208); `Encoder.downscale_factor` (L300); `Encoder.forward` (L313); `Decoder` (L378); `Decoder.__init__` (L399); `Decoder.forward` (L479); `DownEncoderBlock3D` (L517); `DownEncoderBlock3D.__init__` (L518); `DownEncoderBlock3D.forward` (L560); `UNetMidBlock3D` (L573); `UNetMidBlock3D.__init__` (L591); `UNetMidBlock3D.forward` (L621); `UpDecoderBlock3D` (L628); `UpDecoderBlock3D.__init__` (L629); `UpDecoderBlock3D.forward` (L671); `ResnetBlock3D` (L682); `ResnetBlock3D.__init__` (L695); `ResnetBlock3D.forward` (L746); `Downsample3D` (L773); `Downsample3D.__init__` (L774); `Downsample3D.forward` (L796); `Upsample3D` (L812); `Upsample3D.__init__` (L819); `Upsample3D.forward` (L828); `patchify` (L868); `unpatchify` (L906); `create_video_autoencoder_config` (L934); `create_video_autoencoder_pathify4x4x4_config` (L958); `create_video_autoencoder_pathify4x4_config` (L979); `test_vae_patchify_unpatchify` (L997); `demo_video_autoencoder_forward_backward` (L1006).

## flash_head/ltx_video/models/transformers/__init__.py

[Open source](../../flash_head/ltx_video/models/transformers/__init__.py) · 0 physical lines · SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Empty package marker; no runtime algorithm.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/ltx_video/models/transformers/attention.py

[Open source](../../flash_head/ltx_video/models/transformers/attention.py) · 1265 physical lines · SHA-256 `6a36cabbafb74fa3d83d46ae48584269c5f64471e29fd49adaea7387957ea6f3`

Vendored generic attention/feed-forward and processor implementations; supporting library, not the active SoulX DiT attention dispatch.

`BasicTransformerBlock` (L38); `BasicTransformerBlock.__init__` (L77); `BasicTransformerBlock.set_use_tpu_flash_attention` (L184); `BasicTransformerBlock.set_chunk_feed_forward` (L193); `BasicTransformerBlock.forward` (L198); `Attention` (L326); `Attention.__init__` (L379); `Attention.set_use_tpu_flash_attention` (L527); `Attention.set_processor` (L533); `Attention.get_processor` (L555); `Attention.forward` (L661); `Attention.batch_to_head_dim` (L721); `Attention.head_to_batch_dim` (L740); `Attention.get_attention_scores` (L772); `Attention.prepare_attention_mask` (L826); `Attention.norm_encoder_hidden_states` (L885); `Attention.apply_rotary_emb` (L919); `AttnProcessor2_0` (L936); `AttnProcessor2_0.__init__` (L941); `AttnProcessor2_0.__call__` (L944); `AttnProcessor` (L1118); `AttnProcessor.__call__` (L1123); `FeedForward` (L1205); `FeedForward.__init__` (L1219); `FeedForward.forward` (L1258).

## flash_head/ltx_video/models/transformers/embeddings.py

[Open source](../../flash_head/ltx_video/models/transformers/embeddings.py) · 129 physical lines · SHA-256 `30058c51f03d0631b3ce755762d6c3ea5180e37bd89458431c07822045917a46`

Vendored positional/timestep embedding helpers for generic LTX modules.

`get_timestep_embedding` (L10); `get_3d_sincos_pos_embed` (L53); `get_3d_sincos_pos_embed_from_grid` (L66); `get_1d_sincos_pos_embed_from_grid` (L79); `SinusoidalPositionalEmbedding` (L103); `SinusoidalPositionalEmbedding.__init__` (L115); `SinusoidalPositionalEmbedding.forward` (L126).

## flash_head/ltx_video/models/transformers/symmetric_patchifier.py

[Open source](../../flash_head/ltx_video/models/transformers/symmetric_patchifier.py) · 84 physical lines · SHA-256 `b373d066e41fd5700a0a076c1155211243201ee325bf3b6bcebb91d768831def`

Patchify/unpatchify and latent coordinate helpers used by LTX components; not a face ROI renderer.

`Patchifier` (L10); `Patchifier.__init__` (L11); `Patchifier.patchify` (L16); `Patchifier.unpatchify` (L20); `Patchifier.patch_size` (L30); `Patchifier.get_latent_coords` (L33); `SymmetricPatchifier` (L54); `SymmetricPatchifier.patchify` (L55); `SymmetricPatchifier.unpatchify` (L67).

## flash_head/ltx_video/models/transformers/transformer3d.py

[Open source](../../flash_head/ltx_video/models/transformers/transformer3d.py) · 507 physical lines · SHA-256 `6340881be1f8558211d3a9b10286fa7a4be01a3f14be20624403f133cde6cb64`

Generic LTX Transformer3DModel with caption/attention options; not WanModelAudioProject used by Lite.

`Transformer3DModelOutput` (L35); `Transformer3DModel` (L48); `Transformer3DModel.__init__` (L52); `Transformer3DModel.set_use_tpu_flash_attention` (L162); `Transformer3DModel.create_skip_layer_mask` (L173); `Transformer3DModel._set_gradient_checkpointing` (L190); `Transformer3DModel.get_fractional_positions` (L194); `Transformer3DModel.precompute_freqs_cis` (L204); `Transformer3DModel.load_state_dict` (L259); `Transformer3DModel.from_pretrained` (L274); `Transformer3DModel.forward` (L330).

## flash_head/ltx_video/utils/__init__.py

[Open source](../../flash_head/ltx_video/utils/__init__.py) · 0 physical lines · SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Empty package marker; no runtime algorithm.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/ltx_video/utils/diffusers_config_mapping.py

[Open source](../../flash_head/ltx_video/utils/diffusers_config_mapping.py) · 174 physical lines · SHA-256 `a773480b502f2709dff8e41636ddfeae1e856bf45b0f1c139adb56ba267a2931`

Maps recognized Diffusers checkpoint configurations to local architecture fields; relevant to actual VAE construction.

`make_hashable_key` (L1).

## flash_head/ltx_video/utils/prompt_enhance_utils.py

[Open source](../../flash_head/ltx_video/utils/prompt_enhance_utils.py) · 226 physical lines · SHA-256 `4e7bee1faee43cd2a9bca9f6992d48cf23e93b35cdd6308ef6ffb82bc9cb9603`

Optional caption/prompt enhancement utilities; no active RTC prompt-control feature.

`tensor_to_pil` (L47); `generate_cinematic_prompt` (L64); `_get_first_frames_from_conditioning_item` (L113); `_generate_t2v_prompt` (L121); `_generate_i2v_prompt` (L151); `_generate_image_captions` (L188); `_generate_and_decode_prompts` (L211).

## flash_head/ltx_video/utils/skip_layer_strategy.py

[Open source](../../flash_head/ltx_video/utils/skip_layer_strategy.py) · 8 physical lines · SHA-256 `1233195d09930941890e5114a9b2e2b25dc6b3be412aecd6137088ccfb54d62d`

Generic layer-skip strategy enumeration/helper for vendored transformer features; not a measured Lite optimization.

`SkipLayerStrategy` (L4).

## flash_head/ltx_video/utils/torch_utils.py

[Open source](../../flash_head/ltx_video/utils/torch_utils.py) · 25 physical lines · SHA-256 `2447fbcfcac10b2e7b60a21536816bdc666941d36c0ea404a811c8d10928d8fa`

Vendored Torch utilities used by LTX components; generic support rather than serving state ownership.

`append_dims` (L5); `Identity` (L17); `Identity.__init__` (L20); `Identity.forward` (L24).

## flash_head/src/distributed/usp_device.py

[Open source](../../flash_head/src/distributed/usp_device.py) · 35 physical lines · SHA-256 `8cc913ea900d31af9532b8b3ca79eeb14806dfbbb15993fa3f554a349a3f43ac`

Compute sequence/data-parallel device grouping; upstream multi-GPU setup, not per-peer admission.

`get_parallel_degree` (L7); `get_device` (L13).

## flash_head/src/modules/flash_head_model.py

[Open source](../../flash_head/src/modules/flash_head_model.py) · 621 physical lines · SHA-256 `aab3a774f1b09c17871ec06948582626dffa94a82a9df3a9bc430cb5f9bebae6`

Active Lite/Pro-family audio DiT: attention dispatch, rotary math, adaptive blocks, audio projection, cached conditioning and output head.

`flash_attention` (L36); `sinusoidal_embedding_1d` (L69); `precompute_freqs_cis_3d` (L76); `precompute_freqs_cis` (L84); `pad_freqs` (L92); `rope_apply` (L104); `prepare_rotary` (L143); `apply_prepared_rotary` (L156); `RMSNorm` (L166); `RMSNorm.__init__` (L167); `RMSNorm.norm` (L172); `RMSNorm.forward` (L175); `SelfAttention` (L179); `SelfAttention.__init__` (L180); `SelfAttention.forward` (L197); `CrossAttention` (L226); `CrossAttention.__init__` (L227); `CrossAttention.prepare_kv` (L245); `CrossAttention.forward` (L250); `DiTAudioBlock` (L270); `DiTAudioBlock.__init__` (L271); `DiTAudioBlock.forward` (L293); `MLP` (L320); `MLP.__init__` (L321); `MLP.forward` (L331); `Head` (L335); `Head.__init__` (L336); `Head.forward` (L344); `WanModelAudioProject` (L359); `WanModelAudioProject.__init__` (L362); `WanModelAudioProject.patchify` (L433); `WanModelAudioProject.unpatchify` (L439); `WanModelAudioProject.project_audio` (L446); `WanModelAudioProject.prepare_conditioning` (L460); `WanModelAudioProject.prepare_time` (L467); `WanModelAudioProject.forward` (L472); `WanModelAudioProject.forward_blocks` (L544); `AudioProjModel` (L558); `AudioProjModel.__init__` (L559); `AudioProjModel.forward` (L588).

## flash_head/src/pipeline/flash_head_pipeline.py

[Open source](../../flash_head/src/pipeline/flash_head_pipeline.py) · 316 physical lines · SHA-256 `093180cc7800bfb13c5d9b02da6499b9971a1420dc3dccaefb59f55920e06414`

Upstream orchestration: load variant, prepare reference/noise/timesteps, audio features, denoise, decode, color and recurrent feedback.

`get_cond_image_dict` (L24); `timestep_transform` (L43); `FlashHeadPipeline` (L55); `FlashHeadPipeline.__init__` (L56); `FlashHeadPipeline.prepare_params` (L144); `FlashHeadPipeline.reset_person_name` (L196); `FlashHeadPipeline.preprocess_audio` (L206); `FlashHeadPipeline.generate` (L229).

## flash_head/utils/cpu_face_handler.py

[Open source](../../flash_head/utils/cpu_face_handler.py) · 55 physical lines · SHA-256 `975885324af1df32c29ec1d659d2f26f2f4dd5de15c6a57ea31e585dc34b221b`

CPU face detection/landmarks/alignment support for optional cropping.

`CPUFaceHandler` (L6); `CPUFaceHandler.__init__` (L14); `CPUFaceHandler.detect` (L21); `CPUFaceHandler.__call__` (L46).

## flash_head/utils/facecrop.py

[Open source](../../flash_head/utils/facecrop.py) · 110 physical lines · SHA-256 `dc439a58b908a9693ab0192c0018efa0515ce6ff54322ba6db18d129d2a57a66`

Optional face crop/alignment path for reference preparation; not per-frame source-video compositing.

`get_scaled_bbox` (L12); `process_image` (L57).

## flash_head/utils/utils.py

[Open source](../../flash_head/utils/utils.py) · 233 physical lines · SHA-256 `89885620da4c35e757655991638824c7f5c11a8df9bb4eac270d033d5288062b`

Media/audio/device helpers and RGB/Lab color correction; optimized reference-statistic caching and color application.

`rgb_to_lab_torch` (L10); `lab_to_rgb_torch` (L57); `prepare_reference_color_stats` (L106); `match_and_blend_colors_torch` (L113); `resize_and_centercrop` (L195).

## flash_head/wan/modules/__init__.py

[Open source](../../flash_head/wan/modules/__init__.py) · 5 physical lines · SHA-256 `1c9b542fe510771db26071f5d9d78a97e5f4e9c46a7406d52b24147c1e156298`

Package exports/initialization; no independent generation or serving loop.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## flash_head/wan/modules/vae.py

[Open source](../../flash_head/wan/modules/vae.py) · 1598 physical lines · SHA-256 `7071d15350f30eed21a363c624e36c646ae8f7ec81303b83575dcf63bfbc49eb`

Pro-family Wan VAE, causal feature caches and parallel spatial variants; not the Lite LTX backend.

`CausalConv3d` (L17); `CausalConv3d.__init__` (L22); `CausalConv3d.forward` (L34); `RMS_norm` (L45); `RMS_norm.__init__` (L46); `RMS_norm.forward` (L56); `Upsample` (L65); `Upsample.forward` (L66); `Resample` (L73); `Resample.__init__` (L74); `Resample.forward` (L114); `Resample.init_weight` (L182); `Resample.init_weight2` (L194); `ResidualBlock` (L206); `ResidualBlock.__init__` (L207); `ResidualBlock.forward` (L226); `AttentionBlock` (L251); `AttentionBlock.__init__` (L256); `AttentionBlock.forward` (L268); `Encoder3d` (L296); `Encoder3d.__init__` (L297); `Encoder3d.forward` (L353); `Decoder3d` (L410); `Decoder3d.__init__` (L411); `Decoder3d.forward` (L470); `count_conv3d` (L528); `WanVAE_` (L536); `WanVAE_.__init__` (L537); `WanVAE_.forward` (L586); `WanVAE_.blend_v` (L592); `WanVAE_.blend_h` (L600); `WanVAE_.tiled_encode` (L608); `WanVAE_.tiled_decode` (L691); `WanVAE_.encode` (L770); `WanVAE_.decode` (L804); `WanVAE_.decode_stream` (L835); `WanVAE_.cached_decode` (L856); `WanVAE_.reparameterize` (L883); `WanVAE_.sample` (L888); `WanVAE_.clear_cache` (L895); `WanVAE_.encode_video` (L904); `WanVAE_.decode_video` (L912); `_video_vae` (L923); `WanVAE` (L954); `WanVAE.__init__` (L955); `WanVAE._calculate_2d_grid` (L1061); `WanVAE.current_device` (L1083); `WanVAE.encode_dist` (L1086); `WanVAE.encode_dist_2d` (L1170); `WanVAE.encode` (L1263); `WanVAE.decode_dist` (L1302); `WanVAE.decode_dist_2d` (L1369); `WanVAE.decode_dist_2d_stream` (L1454); `WanVAE.decode` (L1541); `WanVAE.decode_stream` (L1575); `WanVAE.encode_video` (L1594); `WanVAE.decode_video` (L1597).

## generate_video.py

[Open source](../../generate_video.py) · 218 physical lines · SHA-256 `5fa326061ecb899728d89b490fc0ff244ad35937aa90cbbed710f3db302762a3`

Offline CLI for once/rolling audio inference, accumulation and file muxing; terminal-window and argparse review findings documented.

`_validate_args` (L18); `_parse_args` (L28); `save_video` (L90); `generate` (L106).

## gradio_app.py

[Open source](../../gradio_app.py) · 427 physical lines · SHA-256 `055d8ae29e243864cccb98dc5914459971b7db68b59fb302db586ae290013e83`

Full-video demonstration UI with cached mutable pipeline, preparation and output saving; not multi-tenant RTC.

`run_multi_gpu_inference` (L22); `save_video_to_file` (L113); `run_inference` (L147).

## gradio_app_streaming.py

[Open source](../../gradio_app_streaming.py) · 339 physical lines · SHA-256 `66e00356d78b72150ffd25c3ddf575be7f644e93584cfb5bdf38a643766926d8`

Segmented MP4 demonstration with background generation queue and final accumulation; not persistent RTP.

`_write_frames_to_mp4` (L37); `save_video_with_audio` (L55); `_save_chunk_audio_to_wav` (L75); `run_inference_streaming` (L86).

## inference_script_multi_gpu_pro.sh

[Open source](../../inference_script_multi_gpu_pro.sh) · 11 physical lines · SHA-256 `93b6a5210765dddd55652a54adf932ae3b16b44710b6bc592ef1cbb647a7bec5`

Upstream distributed Pro invocation, not a session-sharding scheduler.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## inference_script_single_gpu_lite.sh

[Open source](../../inference_script_single_gpu_lite.sh) · 9 physical lines · SHA-256 `ef3febe6ffe445aa77fb5d2a8c59391cb641f48c468ed5db9eddb7d762dd09cf`

Upstream Lite CLI preset and example assets.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## inference_script_single_gpu_pro.sh

[Open source](../../inference_script_single_gpu_pro.sh) · 9 physical lines · SHA-256 `08f443c77db0047fc644bd57b7af18c2a4e58d17cf8a97eb7d5cbc158d04d493`

Upstream Pro CLI preset and example assets.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## requirements-trt-experiment.txt

[Open source](../../requirements-trt-experiment.txt) · 6 physical lines · SHA-256 `fd316243e426b330e48334f84dfc1d127ef3f0e69e9b8111450c0a4f7c0f455c`

Separate export/TensorRT experiment environment dependencies; not proof engines are installed.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## requirements-webrtc.txt

[Open source](../../requirements-webrtc.txt) · 5 physical lines · SHA-256 `6bedd67c5bd71067e8984662ed3ac93ae287c89a2a081c66fe9b24199c832c00`

RTC, media, server and test dependencies.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## requirements.txt

[Open source](../../requirements.txt) · 23 physical lines · SHA-256 `e63092c19936aaf43e457bc60978d8e25e8262432d129fcf077828ba8ea9f7a3`

General inference/demo dependency specifications; actual installed versions/dispatch require runtime verification.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## soulx_rtc/__init__.py

[Open source](../../soulx_rtc/__init__.py) · 1 physical lines · SHA-256 `3c6e544965f029d2903cd3c94fc37c82c6015a3d9902c64a620b78e542fe8f85`

Package exports/initialization; no independent generation or serving loop.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## soulx_rtc/analyze_recordings.py

[Open source](../../soulx_rtc/analyze_recordings.py) · 86 physical lines · SHA-256 `83f094e2fb41074152282f787196176fb41fe6c98cdbbde50f17757e233aee9b`

Decoded image differences and nominal chunk-boundary reports; persistent-stream boundary provenance remains a gap.

`inspect` (L19); `main` (L63).

## soulx_rtc/benchmark_calls.py

[Open source](../../soulx_rtc/benchmark_calls.py) · 292 physical lines · SHA-256 `3e5376f98c3f5f89e335cbf72efca22e2ad26685432d96ac8abed0c20cd4fc99`

Persistent peer/client benchmark: audio turns, retries, interruption, receiver stats, recordings, resource samples, cleanup; sequential turn gap behavior.

`run` (L23); `main` (L254).

## soulx_rtc/benchmark_engine.py

[Open source](../../soulx_rtc/benchmark_engine.py) · 70 physical lines · SHA-256 `0135c13371e644b55a450087b18493849b7fac405fce2c9607d45ae11edc14cf`

Earlier square engine useful-FPS/session sweep.

`main` (L13).

## soulx_rtc/benchmark_musetalk.py

[Open source](../../soulx_rtc/benchmark_musetalk.py) · 127 physical lines · SHA-256 `b81bb59d155573355a410ee77e6438ed00d5f0ea5f4205cef9edd3ff3a375301`

Local comparison wrapper for MuseTalk audio/UNet/VAE/composition; square-resized output limits native-portrait inference.

`main` (L25).

## soulx_rtc/benchmark_rtc.py

[Open source](../../soulx_rtc/benchmark_rtc.py) · 201 physical lines · SHA-256 `f3c821c1953007ddf35c155c58d621aaa4af9b35c3367321f6e17ac1cc8c3fe1`

Finite WebRTC benchmark/recorder; square assignment must be fixed for rectangular profiles.

`input_provenance` (L15); `run` (L23); `main` (L185).

## soulx_rtc/calls.py

[Open source](../../soulx_rtc/calls.py) · 592 physical lines · SHA-256 `17fc1b30688d57acb6383239333f25673c26f00bd8b95e10a76efdd3c6e3b270`

Persistent call data/state, queues, clocks, tracks, source/generated idle, turn admission, cancellation epochs, interruption and cleanup.

`Turn` (L32); `Turn.__post_init__` (L49); `Turn.release_audio` (L52); `Turn.summary` (L56); `IdleVideo` (L65); `IdleVideo.__init__` (L67); `IdleVideo.next` (L76); `IdleVideo._next` (L80); `IdleVideo.close` (L98); `Call` (L106); `Call.metrics` (L143); `Call.finish_if_drained` (L165); `CallVideoTrack` (L175); `CallVideoTrack.__init__` (L176); `CallVideoTrack.recv` (L180); `CallAudioTrack` (L247); `CallAudioTrack.__init__` (L248); `CallAudioTrack.recv` (L252); `CallService` (L286); `CallService.__init__` (L287); `CallService.routes` (L292); `CallService.get` (L297); `CallService.create` (L303); `CallService.offer` (L349); `CallService.append` (L383); `CallService.schedule_once` (L422); `CallService.interrupt` (L531); `CallService.stats` (L559); `CallService.close` (L562); `CallService.delete` (L585); `CallService.cleanup` (L591).

## soulx_rtc/check_recorded_audio.py

[Open source](../../soulx_rtc/check_recorded_audio.py) · 77 physical lines · SHA-256 `3eb4d3d239d9037e9d27674306928a8f95bc3f35dd23586061f388d5833cd8d0`

Waveform correlation and inserted-pause analysis; not visual lip-sync scoring.

`scores` (L12); `main` (L31).

## soulx_rtc/codec.py

[Open source](../../soulx_rtc/codec.py) · 67 physical lines · SHA-256 `184375b6c9a0a35166a562aa41014c15392edfb73dfd9a1a834b6090cdcef1b9`

Version-aware optional FastH264Encoder hook with preset/thread controls; fallback compatibility.

`FastH264Encoder` (L12); `FastH264Encoder.__init__` (L13); `FastH264Encoder._encode_frame` (L17); `install_encoder_factory` (L36); `sender_encoder_info` (L54).

## soulx_rtc/engine.py

[Open source](../../soulx_rtc/engine.py) · 360 physical lines · SHA-256 `1bae05e2c6a8ca33f29e0935c6e490894115ce6cf9a9cdac3688f3b1f6b79980`

Lite GPU owner: reference LRU, private session RNG/audio/motion, geometry, cached conditioning, memory modes, profiling and chunk generation.

`GenerationState` (L17); `validate_geometry` (L25); `Engine` (L36); `Engine.__init__` (L41); `Engine.move_dit` (L100); `Engine.constants` (L105); `Engine.prepare` (L116); `Engine._audio` (L144); `Engine.append` (L160); `Engine.prepare_call` (L175); `Engine.recondition` (L181); `Engine.generate` (L199); `Engine.warmup` (L339); `Engine.validate_isolation` (L346).

## soulx_rtc/experiment.py

[Open source](../../soulx_rtc/experiment.py) · 185 physical lines · SHA-256 `0befec2681a760e3abc3d3082fe751cfa064943b7f3d88d5ddaedc6f39e88683`

Repeated alternating useful-frame/profile benchmark, raw phase evidence and limited first-run recordings.

`record_failure` (L20); `record` (L29); `memory_snapshot` (L55); `main` (L70).

## soulx_rtc/gpu_lease.py

[Open source](../../soulx_rtc/gpu_lease.py) · 14 physical lines · SHA-256 `163f862da0ce5d1dbd5525cddc4a4b165d9ffc46dcc50e1630a219708c008f0b`

Advisory checkout file lock; not global device reservation.

`acquire_gpu_lease` (L6).

## soulx_rtc/index.html

[Open source](../../soulx_rtc/index.html) · 69 physical lines · SHA-256 `0e7035db2a841aefd7fb81cc1e2a5f944f5794e8cbdf98395b65588aca651aa6`

Browser call UI, one offer/ICE gathering, audio turns, idempotency IDs, interrupt and polling controls.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## soulx_rtc/metrics.py

[Open source](../../soulx_rtc/metrics.py) · 10 physical lines · SHA-256 `e93f9b987866fdeaf71ca720fb178e70ed88c19dd5b0d9b4120433ee95ab4167`

Chunk metric record schema; labels must be interpreted with mode/source revision.

`process_rss_mib` (L6).

## soulx_rtc/replay_lab.py

[Open source](../../soulx_rtc/replay_lab.py) · 290 physical lines · SHA-256 `10b6a4044e241403882194fe9a38ad01325d74d461735fac7f3e0630b40b4294`

Approved source-bank audit/manifest, path guards, ordered boundary checks and finite source-video relay control.

`source_paths` (L22); `frame_hash` (L42); `mae` (L46); `audit_sources` (L50); `MediaClock` (L97); `MediaClock.__init__` (L98); `MediaClock.pace` (L101); `decoded_frames` (L107); `BankVideoTrack` (L113); `BankVideoTrack.__init__` (L114); `BankVideoTrack.recv` (L120); `ClockedSilenceTrack` (L139); `ClockedSilenceTrack.__init__` (L140); `ClockedSilenceTrack.recv` (L145); `replay` (L157); `main` (L277).

## soulx_rtc/server.py

[Open source](../../soulx_rtc/server.py) · 518 physical lines · SHA-256 `c3798a8a1094076dc539e8f309124fa2edf54b84b0a502c9829c3a6454ab7c90`

HTTP validation/auth/readiness, finite tracks/scheduler, persistent call routes, shared worker and health/error propagation.

`Session` (L35); `Session.metrics` (L62); `VideoTrack` (L76); `VideoTrack.__init__` (L79); `VideoTrack.recv` (L85); `AudioTrack` (L134); `AudioTrack.__init__` (L137); `AudioTrack.recv` (L141); `decode_audio` (L179); `Service` (L193); `Service.__init__` (L194); `Service.gpu` (L215); `Service.mark_unhealthy` (L218); `Service.startup` (L224); `Service.close_session` (L239); `Service.cleanup` (L253); `Service.schedule` (L264); `Service.create` (L316); `Service.get` (L374); `Service.offer` (L380); `Service.stats` (L413); `Service.delete` (L416); `Service.health` (L420); `Service.config` (L441); `make_app` (L445); `main` (L466).

## soulx_rtc/trt_backend.py

[Open source](../../soulx_rtc/trt_backend.py) · 128 physical lines · SHA-256 `c8994d7017cfa74f9c132cfe06c15c36c151172cff0426ec433a23739de654a4`

Exact-shape engine validation, serial external-workspace contexts, Torch compiler boundary and optional FFN installation.

`file_hash` (L13); `TRTFeedForward` (L21); `TRTFeedForward.__init__` (L22); `TRTFeedForward.set_workspace` (L63); `TRTFeedForward.release_workspace` (L70); `TRTFeedForward.forward` (L80); `install_ffn_partitions` (L100).

## soulx_rtc/trt_experiment.py

[Open source](../../soulx_rtc/trt_experiment.py) · 179 physical lines · SHA-256 `c18448c3e43fa001eea1433f8aad07a8ecf92fc1ce7a5e12e5cd8f79787bf586`

Real-input FFN capture, ONNX/profile export, typed engine build, error/timing report and resumable artifact creation.

`capture` (L13); `benchmark` (L52); `build` (L68); `main` (L153).

## soulx_rtc/trt_vae_experiment.py

[Open source](../../soulx_rtc/trt_vae_experiment.py) · 137 physical lines · SHA-256 `42c44b18cb219f747d4f8c49b9fd0adde5cf7db4d123b1fcb74e71145fb70225`

Deterministic VAE decode export/build/compare; optional FP16 normalization experiment is not certified.

`VaeDecoder` (L13); `VaeDecoder.__init__` (L14); `VaeDecoder.forward` (L18); `FP32PixelNorm` (L24); `FP32PixelNorm.__init__` (L25); `FP32PixelNorm.forward` (L29); `main` (L34).

## soulx_rtc/worker.py

[Open source](../../soulx_rtc/worker.py) · 141 physical lines · SHA-256 `c4a0a916bd3412b122a9a04e19e70f8bee501d914c88001e48004509e2754df4`

Spawned process façade, module-global GPU engine/private state dictionary, warmup/isolation and opaque CPU-side remote state.

`RemoteState` (L20); `initialize` (L26); `prepare` (L57); `generate` (L64); `release` (L75); `prepare_call` (L79); `append` (L88); `recondition` (L92); `GPUProcess` (L96); `GPUProcess.__init__` (L97); `GPUProcess.call` (L102); `GPUProcess.start` (L105); `GPUProcess.prepare` (L113); `GPUProcess.prepare_call` (L118); `GPUProcess.append` (L123); `GPUProcess.recondition` (L126); `GPUProcess.generate` (L130); `GPUProcess.release` (L136); `GPUProcess.close` (L140).

## start_streaming_lite.sh

[Open source](../../start_streaming_lite.sh) · 24 physical lines · SHA-256 `bc7d1737124a1458b011063452ef933bd52ebb1f139df7cc9a1533840213baeb`

Environment/launch wrapper for the public-bind Gradio segmented demo; separate from RTC security/lease.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## start_webrtc.sh

[Open source](../../start_webrtc.sh) · 8 physical lines · SHA-256 `d399d0df5f6b9e6ffdaa1c30260be7e494a008888756ac504332d70cb99c0ddd`

RTC environment defaults, GPU visibility, allocator/compile cache setup and service invocation.

No Python function/class symbol entries (configuration, launcher, browser code or package marker).

## tests/test_calls.py

[Open source](../../tests/test_calls.py) · 322 physical lines · SHA-256 `4cad2ede76f1e7ad8cb13e26ceff2bac60b761cc41ee9075b17b93cc6ec0ebb4`

Controlled persistent-turn, idempotency, queue, cancellation, idle, clock/audio and cleanup regression fixtures.

`FakeEngine` (L19); `FakeEngine.prepare_call` (L23); `FakeEngine.append` (L26); `FakeEngine.recondition` (L29); `FakeEngine.generate` (L34); `wav` (L40); `test_persistent_real_peer_turns_retry_interrupt_cleanup` (L47); `test_no_idle_switch_before_audio_drain` (L141); `test_idle_clock_does_not_build_a_catchup_backlog` (L157); `test_audio_boundary_race_has_one_packet_allowance_but_stalls_are_counted` (L173); `test_generated_silence_keeps_state_without_filling_turn_history` (L193); `test_call_microbatch_respects_active_admission_and_private_outputs` (L222); `test_shared_worker_failure_marks_all_calls_unhealthy` (L252); `test_generated_idle_yields_admission_to_another_peers_speech` (L264); `test_generated_idle_renders_ahead_without_rearming_or_unbounded_queue` (L294).

## tests/test_codec.py

[Open source](../../tests/test_codec.py) · 26 physical lines · SHA-256 `046d8370006943789ec3d9ef941329e10f4f5fffd4c7befda18f2d6193b7ac13`

Encoder override/preset compatibility fixtures; parametrization contributes to test case count.

`test_fast_h264_packets_decode_and_recreate_on_bitrate_change` (L9).

## tests/test_gpu_lease.py

[Open source](../../tests/test_gpu_lease.py) · 12 physical lines · SHA-256 `dca44a0b6ffba0c70891a5607b55278a10c5e532ceed63ea56a302b729160ab1`

Same-checkout lock exclusion test.

`test_gpu_owner_is_exclusive_and_released` (L6).

## tests/test_optimizations.py

[Open source](../../tests/test_optimizations.py) · 103 physical lines · SHA-256 `a1ef259426cae6a390c1053948338f3789f1e7b6c804f0d9f18396addd344877`

Small/random-model optimized math, conditioning and private-state contracts, not full-checkpoint GPU certification.

`test_native_geometry` (L7); `test_prepared_rotary_preserves_complex_baseline` (L17); `test_cached_color_and_temporal_tiles` (L32); `test_cached_cross_attention_matches_batch_and_separate` (L46); `test_append_preserves_rolling_audio_and_chunk_alignment` (L66); `test_full_small_dit_cached_matches_uncached` (L78).

## tests/test_recorded_audio.py

[Open source](../../tests/test_recorded_audio.py) · 14 physical lines · SHA-256 `21653471bf567cf94912cdbcc1b70fa11b7396ae2b483d1017f50e5e883511c6`

Synthetic waveform alignment/pause analysis test.

`test_correspondence_rejects_silence_roundoff_and_finds_actual_waveform` (L6).

## tests/test_replay_lab.py

[Open source](../../tests/test_replay_lab.py) · 77 physical lines · SHA-256 `cc21cf96cefc209db6de246d77c95a102c8e7849efb3029450dfd3a3b0beda3e`

Source manifest/path and relay control tests.

`clip` (L16); `test_manifest_aliases_and_escape` (L30); `test_source_audit` (L48); `test_replay_real_persistent_peer` (L64).

## tests/test_rtc.py

[Open source](../../tests/test_rtc.py) · 145 physical lines · SHA-256 `1c9adbbc4d2da23f8f9824cdf72a9f7dc74ccbdfcec2e9dbce0bf4f80395d438`

Finite request validation, track and lifecycle fixtures with controlled generation.

`test_audio_decode_resamples_and_limits` (L12); `test_audio_rejects_empty` (L20); `test_tracks_monotonic_and_bounded` (L27); `test_cancel_unblocks_waiting_tracks` (L45); `test_rope_batch_equals_separate` (L58); `test_dit_audio_block_preserves_batch` (L69); `test_full_queue_does_not_block_other_sessions` (L84); `test_http_auth_and_capacity` (L111).

## tests/test_rtc_loopback.py

[Open source](../../tests/test_rtc_loopback.py) · 69 physical lines · SHA-256 `456bbddf8918da7659fb21081cf6ae94e9a1f20458a9c9f7631bd4672160b1a7`

Actual local peer transport loopback with H264/Opus, not WAN/TURN load qualification.

`test_real_media_loopback` (L13).

## tests/test_trt_contract.py

[Open source](../../tests/test_trt_contract.py) · 69 physical lines · SHA-256 `d4b992ecf7b153b890a96bbad5e75b309cceb28b5d53d93898362799551e6d30`

Mocked metadata/binding/profile contract validation, not real engine numerical benchmarking.

`test_trt_artifact_rejects_mismatches_before_execution` (L10); `test_trt_validates_actual_engine_binding_before_workspace` (L37); `test_transient_workspace_waits_before_release` (L57).

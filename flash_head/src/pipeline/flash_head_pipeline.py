# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
from PIL import Image
from loguru import logger
import time
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

from transformers import Wav2Vec2FeatureExtractor

from flash_head.src.modules.flash_head_model import WanModelAudioProject
from flash_head.audio_analysis.wav2vec2 import Wav2Vec2Model
from flash_head.utils.utils import match_and_blend_colors_torch, resize_and_centercrop
from flash_head.utils.facecrop import process_image
from flash_head.utils.latency import latency_scope
from flash_head.src.pipeline.schedules import raw_timestep_schedule

# compile models to speedup inference
COMPILE_MODEL = True
COMPILE_VAE = True
# use parallel vae to speedup decode/encode, only support WanVAE
USE_PARALLEL_VAE = True

def get_cond_image_dict(cond_image_path_or_dir, use_face_crop):
    def get_image(cond_image_path, use_face_crop):
        if use_face_crop:
            try:
                image = process_image(cond_image_path)
                return image
            except Exception as e:
                logger.error(f"Error processing {cond_image_path}: {e}")
        return Image.open(cond_image_path).convert("RGB")

    if os.path.isdir(cond_image_path_or_dir):
        import glob
        cond_image_list = glob.glob(os.path.join(cond_image_path_or_dir, "*.png"))
        cond_image_list.sort()
        cond_image_dict = {cond_image.split("/")[-1].split(".")[0]: get_image(cond_image, use_face_crop) for cond_image in cond_image_list}
    else:
        cond_image_dict = {cond_image_path_or_dir.split("/")[-1].split(".")[0]: get_image(cond_image_path_or_dir, use_face_crop)}
    return cond_image_dict

def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class FlashHeadPipeline:
    # Class-level defaults, NOT only instance attributes. generate() reads these
    # unconditionally, and callers legitimately build this object with
    # __new__ (tests/test_pipeline_latency.py does exactly that to exercise the
    # real generate() with small CPU substitutes). Assigning them only in
    # __init__ would raise AttributeError on any such instance.
    lean_delivery = False
    verbose_timing = False
    skip_zero_weighted_noise = False
    resolved_timesteps = ()
    timestep_variant = "shipped"

    def __init__(
        self,
        checkpoint_dir,
        model_type,
        wav2vec_dir,
        device="cuda",
        param_dtype=torch.bfloat16,
        use_usp=False,
        num_timesteps=1000,
        use_timestep_transform=True,
        model_transform=None,
    ):
        r"""
        Initializes the image-to-video generation model components.
        Args:
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            wav2vec_dir (`str`):
                Path to directory containing wav2vec checkpoints
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
        """
        self.param_dtype = param_dtype
        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Opt-in delivery/diagnostic switches. Both default to the historical
        # behaviour so every retained comparison stays reproducible.
        #
        # lean_delivery: trim the leading history frames and hand back device
        #   uint8 instead of host float32. Callers that already trim themselves
        #   (generate_video.py, the gradio apps) must leave this False.
        # verbose_timing: the per-window stdout timing lines. They have no
        #   consumer; latency_scope records the same boundaries properly.
        # skip_zero_weighted_noise: at the terminal level the randn term is
        #   multiplied by exactly zero, so the draw is pure waste. Skipping it
        #   advances the generator differently, so a run with this flag is NOT
        #   bitwise comparable to one without -- hence default off.
        self.lean_delivery = False
        self.verbose_timing = False
        self.skip_zero_weighted_noise = False
        # Populated by prepare_params; needed by the denoise loop to decide the
        # terminal step without a device comparison.
        self.resolved_timesteps = []
        self.use_usp = use_usp and dist.is_initialized()
        self.model_type = model_type
        self.use_ltx = model_type == "lite"

        if self.use_ltx:
            model_dir = os.path.join(checkpoint_dir, "Model_Lite")
            vae_dir = os.path.join(checkpoint_dir, "VAE_LTX")

            from flash_head.ltx_video.ltx_vae import LtxVAE
            with latency_scope("load.vae"):
                self.vae = LtxVAE(
                    pretrained_model_type_or_path=vae_dir,
                    dtype=self.param_dtype,
                    device=self.device,
                )
        else:
            vae_path = os.path.join(checkpoint_dir, "VAE_Wan/Wan2.1_VAE.pth")
            
            from flash_head.wan.modules import WanVAE
            with latency_scope("load.vae"):
                self.vae = WanVAE(
                    vae_path=vae_path,
                    dtype=self.param_dtype,
                    device=self.device,
                    parallel=(USE_PARALLEL_VAE and self.use_usp),
                )

            if self.model_type == "pretrained":
                self.audio_guide_scale = 3.0
                model_dir = os.path.join(checkpoint_dir, "teacher")
            elif self.model_type == "pro":
                model_dir = os.path.join(checkpoint_dir, "Model_Pro")
        
        with latency_scope("load.dit_weights", gpu=False):
            self.model = WanModelAudioProject.from_pretrained(model_dir)
        self.model.eval().requires_grad_(False)
        # Strict checkpoint loading stays unchanged. Optional inference-only
        # transformations run at target precision on CPU before GPU placement.
        if model_transform is not None:
            self.model.to(dtype=self.param_dtype)
            model_transform(self.model)
        with latency_scope("load.dit_placement"):
            self.model.to(device=self.device, dtype=self.param_dtype)

        self.config = self.model.config

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        if COMPILE_MODEL:
            self.model = torch.compile(self.model)
        if COMPILE_VAE:
            if self.use_ltx:
                self.vae.model.encode = torch.compile(self.vae.model.encode)
                self.vae.model.decode = torch.compile(self.vae.model.decode)
            else:
                self.vae.encode = torch.compile(self.vae.encode)
                self.vae.decode = torch.compile(self.vae.decode)

        with latency_scope("load.audio_encoder"):
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)

    @torch.no_grad()
    def prepare_params(self,
                        cond_image_path_or_dir,
                        target_size,
                        frame_num,
                        motion_frames_num,
                        sampling_steps,
                        seed=None,
                        shift=5.0,
                        color_correction_strength=0.0,
                        use_face_crop=False,
                        ):
        with latency_scope("prepare.reference_read", gpu=False):
            self.cond_image_dict = get_cond_image_dict(cond_image_path_or_dir, use_face_crop)

        self.frame_num = frame_num
        self.motion_frames_num = motion_frames_num
        self.color_correction_strength = color_correction_strength

        self.target_h, self.target_w = target_size
        self.lat_h, self.lat_w = self.target_h // self.config.vae_stride[1], self.target_w // self.config.vae_stride[2]

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # prepare timesteps. The default "shipped" variant reproduces the original
        # behaviour exactly, including the np.linspace fallback the CFG teacher
        # relies on at sample_steps=20. A None return means "untabulated under
        # shipped", so the original fallback runs verbatim.
        timesteps = raw_timestep_schedule(
            sampling_steps, num_timesteps=self.num_timesteps, variant=self.timestep_variant
        )
        if timesteps is None:
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
        self.timesteps = timesteps
        # Resolved post-shift levels, recorded so run reports can state the schedule.
        self.resolved_timesteps = [float(t.item()) for t in timesteps]

        self.cond_image_tensor_dict = {}
        self.ref_img_latent_dict = {}
        for i, (person_name, cond_image_pil) in enumerate(self.cond_image_dict.items()):
            with latency_scope("prepare.reference_resize_h2d"):
                cond_image_tensor = resize_and_centercrop(cond_image_pil, (self.target_h, self.target_w)).to(self.device, dtype=self.param_dtype) # 1 C 1 H W
                cond_image_tensor = (cond_image_tensor / 255 - 0.5) * 2

            self.cond_image_tensor_dict[person_name] = cond_image_tensor

            with latency_scope("prepare.reference_encode"):
                video_frames = cond_image_tensor.repeat(1, 1, self.frame_num, 1, 1)
                self.ref_img_latent_dict[person_name] = self.vae.encode(video_frames) # (16, 9, 64, 64) / (128, 5, 16, 16)
            if i == 0:
                self.reset_person_name(person_name)

        return
    
    @torch.no_grad()
    def reset_person_name(self, person_name=None):
        if person_name is None or person_name not in self.cond_image_dict:
            pass
        else:
            self.person_name = person_name
        self.original_color_reference = self.cond_image_tensor_dict[self.person_name]
        self.ref_img_latent = self.ref_img_latent_dict[self.person_name]
        self.latent_motion_frames = self.ref_img_latent[:, :1].clone()

    @torch.no_grad()
    def preprocess_audio(self, speech_array, sr=16000, fps=25, timing=None):
        video_length = len(speech_array) * fps / sr

        # wav2vec_feature_extractor
        with latency_scope("audio.normalize", gpu=False):
            audio_feature = np.squeeze(
                self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
            )
        with latency_scope("audio.h2d"):
            audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
            audio_feature = audio_feature.unsqueeze(0)
        if timing is not None:
            timing("audio_normalize_h2d")

        # audio encoder
        with torch.no_grad(), latency_scope("audio.wav2vec"):
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        if timing is not None:
            timing("wav2vec")

        if len(embeddings) == 0:
            logger.error("Fail to extract audio embedding")
            return None

        with latency_scope("audio.stack"):
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")
        if timing is not None:
            timing("audio_stack")
        return audio_emb

    @torch.no_grad()
    def generate(self, audio_embedding):
        # evaluation mode
        with torch.no_grad():

            # sample videos
            with latency_scope("denoise.noise_init"):
                noise = torch.randn(
                    self.config.out_dim,
                    (self.frame_num - 1) // self.config.vae_stride[0] + 1,
                    self.lat_h,
                    self.lat_w,
                    dtype=self.param_dtype,
                    device=self.device,
                    generator=self.generator)

            for i in range(len(self.timesteps)-1):
                torch.cuda.synchronize()
                start_time = time.time()

                with latency_scope("denoise.history_inject", step=i):
                    noise[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

                with latency_scope("denoise.model", step=i):
                    flow_pred = self.model(
                        x=noise.unsqueeze(0),
                        timestep=self.timesteps[i],
                        context=audio_embedding,
                        y=self.ref_img_latent.unsqueeze(0),
                    )[0]

                if self.model_type == "pretrained":
                    with latency_scope("denoise.teacher_unconditional", step=i):
                        flow_pred_drop_audio = self.model(
                            x=noise.unsqueeze(0),
                            timestep=self.timesteps[i],
                            context=torch.zeros_like(audio_embedding),
                            y=self.ref_img_latent.unsqueeze(0),
                        )[0]
                    with latency_scope("denoise.update", step=i):
                        flow_pred = flow_pred_drop_audio + self.audio_guide_scale * (flow_pred - flow_pred_drop_audio)
                        dt = self.timesteps[i] - self.timesteps[i + 1]
                        dt = (dt / self.num_timesteps).to(self.param_dtype)
                        noise = noise - flow_pred * dt[:, None, None, None]
                
                else:
                    # update latent
                    with latency_scope("denoise.update", step=i):
                        t_i = (self.timesteps[i][:, None, None, None] / self.num_timesteps).to(self.param_dtype)
                        t_i_1 = (self.timesteps[i+1][:, None, None, None] / self.num_timesteps).to(self.param_dtype)
                        x_0 = noise - flow_pred * t_i

                        # The terminal level is exactly 0, so the randn term is
                        # multiplied by zero and (1 - t_i_1) is 1. Decided from
                        # the Python-side schedule, never from the device tensor:
                        # comparing t_i_1 on device would cost a host sync far
                        # larger than the draw it saves.
                        terminal = (
                            self.skip_zero_weighted_noise
                            and i + 1 < len(self.resolved_timesteps)
                            and self.resolved_timesteps[i + 1] == 0.0
                        )
                        if terminal:
                            noise = x_0
                        else:
                            noise = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator)

                torch.cuda.synchronize()
                end_time = time.time()
                if self.rank == 0 and self.verbose_timing:
                    print(f'[generate] model denoise per step: {end_time - start_time}s')

            with latency_scope("denoise.final_history_inject"):
                noise[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

            torch.cuda.synchronize()
            start_decode_time = time.time()

            with latency_scope("decode.total"):
                videos = self.vae.decode(noise)

            torch.cuda.synchronize()
            end_decode_time = time.time()
            if self.rank == 0 and self.verbose_timing:
                print(f'[generate] decode video frames: {end_decode_time - start_decode_time}s')
        
        torch.cuda.synchronize()
        start_color_correction_time = time.time()
        if self.lean_delivery:
            # Trim the leading history frames BEFORE colour correction, so the
            # correction and every later pointwise pass run over 28 frames
            # instead of 33.
            #
            # Safe for two independent reasons:
            #  * Colour correction is per-frame independent --
            #    match_and_blend_colors_torch reduces over dim=[2,3] (H, W only)
            #    on a (B, T, H, W, C) tensor, keeping B and T -- so the
            #    surviving frames are bit-identical.
            #  * cond_frame takes the TRAILING motion_frames_num, and the
            #    trailing n of 33 are the trailing n of 28 once the leading n
            #    are removed. motion_frames_num equals the leading history count
            #    on both paths (PRO 5/5 via run.py, LITE 9/9 via engine.py).
            with latency_scope("postprocess.lean_trim"):
                videos = videos[:, :, self.motion_frames_num:]
        with latency_scope("postprocess.color_correction"):
            if self.color_correction_strength > 0.0:
                videos = match_and_blend_colors_torch(videos, self.original_color_reference, self.color_correction_strength)

        with latency_scope("motion.history_select"):
            cond_frame = videos[:, :, -self.motion_frames_num:].to(self.device)
        torch.cuda.synchronize()
        end_color_correction_time = time.time()
        if self.rank == 0 and self.verbose_timing:
            print(f'[generate] color correction: {end_color_correction_time - start_color_correction_time}s')

        torch.cuda.synchronize()
        start_encode_time = time.time()
        with latency_scope("motion.encode"):
            self.latent_motion_frames = self.vae.encode(cond_frame)
        torch.cuda.synchronize()
        end_encode_time = time.time()
        if self.rank == 0 and self.verbose_timing:
            print(f'[generate] encode motion frames: {end_encode_time - start_encode_time}s')

        if self.lean_delivery:
            # Already trimmed above, before colour correction.
            with latency_scope("postprocess.rgb_uint8_device"):
                frames = videos[0].to(torch.float32)
                # Identical op order to the historical host path, so
                # raw_rgb_sha256 stays comparable across the two deliveries.
                frames = (((frames + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255)
                if not torch.isfinite(frames).all():
                    # The host path checked this before astype(uint8); a
                    # non-finite value casts to an undefined byte, so the guard
                    # must move with the cast.
                    #
                    # Tradeoff to confirm on the 4070: this reduction forces a
                    # host sync the float32 path did not have, costing one
                    # ~62 MiB read (~0.13 ms at ~480 GB/s) plus a barrier,
                    # against ~10 ms saved on a D2H that drops from 59.06 MiB
                    # to 14.77 MiB. Expected net win, but it is an arithmetic
                    # expectation, not a measurement.
                    raise RuntimeError("Non-finite pixel in generated window")
                return frames.contiguous().to(torch.uint8)

        gen_video_samples = videos #[:, :, self.motion_frames_num:]

        with latency_scope("postprocess.output_fp32"):
            return gen_video_samples[0].to(torch.float32)

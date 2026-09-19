def generate(self, states):
    """Return one uint8 RGB chunk per session, without retaining history."""
    torch = self.torch
    from flash_head.utils.utils import match_and_blend_colors_torch

    if not states or any(s.cursor >= s.total_frames for s in states):
        raise ValueError("Expected unfinished sessions")
    started = time.perf_counter()
    stamps = []
    detail = []
    host_detail = []
    def phase(name):
        if self.profile:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            detail.append((name, event))
            host_detail.append((name, time.perf_counter()))
    def stamp():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        stamps.append(event)

    self.compute_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(self.compute_stream), torch.no_grad():
        stamp()
        phase("start")
        if self.memory_mode == "staged":
            # Keep DiT off-device during Wav2Vec too. After decode we leave
            # it on CPU until the next chunk's audio activations are gone.
            self.move_dit("cpu")
            phase("dit_offload_for_audio")
            self.pipeline.audio_encoder.to(self.pipeline.device)
            phase("audio_weights_reload")
        audio_contexts = []
        for index, state in enumerate(states):
            timing = None
            if self.profile:
                timing = phase if len(states) == 1 else lambda name, i=index: phase(f"{name}_{i}")
            audio_contexts.append(self._audio(state, timing))
        contexts = torch.cat(audio_contexts)
        del audio_contexts
        phase("audio")
        p = states[0].pipeline
        if self.memory_mode == "staged":
            self.pipeline.audio_encoder.to("cpu")
            torch.cuda.empty_cache()
            phase("audio_weights_offload")
            self.move_dit(p.device)
            phase("dit_reload_after_audio")
        shape = (p.config.out_dim, 5, p.lat_h, p.lat_w)
        def random_batch():
            return torch.stack([torch.randn(shape, device=p.device,
                dtype=p.param_dtype, generator=s.pipeline.generator) for s in states])
        noise = random_batch()
        reference = torch.stack([s.pipeline.ref_img_latent for s in states])
        kwargs = {}
        times = None
        if self.optimized:
            context, kv = self.conditioning(contexts)
            rotary, times = self.constants(p, len(states))
            kwargs = dict(prepared_context=context, cross_kv=kv, rotary=rotary)
        phase("conditioning_noise_reference")
        stamp()
        for i in range(len(p.timesteps) - 1):
            for j, state in enumerate(states):
                motion = state.pipeline.latent_motion_frames
                noise[j, :, :motion.shape[1]] = motion
            flow = self.denoise(p, noise, p.timesteps[i].expand(len(states)),
                           context=contexts, y=reference,
                           **kwargs, prepared_time=None if times is None else times[i])
            t = (p.timesteps[i] / p.num_timesteps).to(p.param_dtype)
            nxt = (p.timesteps[i + 1] / p.num_timesteps).to(p.param_dtype)
            noise = (1 - nxt) * (noise - flow * t) + nxt * random_batch()
            phase(f"dit_step_{i}")
        if self.refinement_timesteps:
            from .refinement import refine_clean_latent
            def refine_denoise(x, timestep):
                expanded = timestep.expand(len(states))
                return self.denoise(p, x, expanded, context=contexts, y=reference,
                    **kwargs, prepared_time=(self.raw_model.prepare_time(expanded)
                                            if self.optimized else None))
            noise = refine_clean_latent(noise, p.refinement_times, p.num_timesteps,
                [s.pipeline.latent_motion_frames for s in states],
                [s.pipeline.refinement_generator for s in states], refine_denoise)
            phase("dit_refinement")
        stamp()
        # Release conditioning tensors before VAE; weights remain shared.
        kwargs.clear()
        if self.optimized:
            del context, kv, rotary
        del contexts, reference, flow
        if self.trt_layers:
            # TensorRT weights/workspace live outside the caching allocator;
            # release unused DiT allocations before the high-water VAE stage.
            torch.cuda.empty_cache()
        self.move_dit("cpu")
        phase("dit_offload")
        outputs = []
        for j, state in enumerate(states):
            sp = state.pipeline
            motion = sp.latent_motion_frames
            noise[j, :, :motion.shape[1]] = motion
            videos = sp.vae.decode(noise[j])
            phase(f"decode_{j}")
            if self.trt_vae:
                sp.vae.decode.release_workspace()
                phase(f"decode_workspace_release_{j}")
            output_offset = self.motion_frames
            if False:  # Keep full context for histories longer than emitted chunks.
                # Color statistics are spatial/per-frame. These overlap
                # frames are neither sent nor used by the last-nine encoder.
                videos = videos[:, :, 9:]
                output_offset = 0
            if self.memory_mode in ("compact", "reference", "staged"):
                videos = torch.cat([self.color_match(videos[:, :, k:k+4],
                    sp.original_color_reference, sp.color_correction_strength,
                    reference_stats=sp.reference_color_stats if self.optimized else None)
                    for k in range(0, videos.shape[2], 4)], dim=2)
            else:
                videos = self.color_match(videos,
                    sp.original_color_reference, sp.color_correction_strength,
                    reference_stats=sp.reference_color_stats if self.optimized else None)
            phase(f"color_{j}")
            terminal = self.lean and state.terminal and state.cursor+self.chunk_frames >= state.total_frames
            if not terminal:
                posterior = sp.vae.model.encode(videos[:, :, -self.motion_frames:], return_dict=False)[0]
                sp.latent_motion_frames = sp.vae.normalize_latents(
                    posterior.sample(generator=sp.generator))[0]
                del posterior
            else:
                state.motion_valid = False
            phase(f"motion_encode_{j}")
            count = min(self.chunk_frames, state.total_frames - state.cursor)
            # Never transfer the nine overlap frames or float32 pixels to CPU.
            pixels = ((videos[0, :, output_offset:output_offset + count].float() + 1) * 127.5)
            pixels = pixels.clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
            phase(f"uint8_{j}")
            outputs.append(pixels.cpu().numpy())
            phase(f"transfer_{j}")
            state.cursor += count
            # Long-lived calls retain only rolling audio history + future input.
            trim_to = max(state.audio_offset, round(state.cursor * self.sample_rate / self.fps) - 8*self.sample_rate)
            trim = min(len(state.audio), trim_to - state.audio_offset)
            state.audio = state.audio[trim:].copy()
            state.audio_offset += trim
            del videos, pixels
        if self.memory_mode != "staged":
            self.move_dit(p.device)
            phase("dit_reload")
        stamp()
        phase("cleanup")
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    self.last_metrics = {
        "batch": len(states), "frames": sum(len(x) for x in outputs),
        "wall_s": elapsed,
        "audio_ms": stamps[0].elapsed_time(stamps[1]),
        "dit_ms": stamps[1].elapsed_time(stamps[2]),
        "vae_color_transfer_ms": stamps[2].elapsed_time(stamps[3]),
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "width": self.width, "height": self.height,
        "optimized": self.optimized, "real_rope": self.real_rope,
        "int8_weights": self.int8_weights, "weight_storage": self.weight_storage_report,
        "cuda_memory_mib": self.cuda_memory_mib,
        "lean": self.lean, "fused_qkv": self.fused_qkv, "dit_graph": self.dit_graph,
        "compile_color": self.compile_color, "compile_audio": self.compile_audio,
        "trt_ffn_layers": self.trt_layers,
        "trt_vae": self.trt_vae,
        "trt_vae_workspace": "transient" if self.trt_vae else None,
        "stages_ms": {name: a.elapsed_time(b) for (_, a), (name, b) in zip(detail, detail[1:])},
        "host_stages_ms": {name: (b-a)*1000 for (_, a), (name, b) in zip(host_detail, host_detail[1:])},
    }
    return outputs

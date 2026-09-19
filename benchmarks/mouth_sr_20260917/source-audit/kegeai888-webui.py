#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SoulX-FlashHead WebUI
作者：科哥 | 微信：312088415 公众号：科哥玩AI
承诺永远开源使用 但是需要保留本人版权信息！
"""

import os
import sys
import gradio as gr
import numpy as np
import librosa
import time
from datetime import datetime
from loguru import logger
import torch

# 导入项目模块
from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline
)
from generate_video import save_video

# 全局变量缓存pipeline
global_pipeline = None
global_model_type = None
global_sample_steps = None
global_performance_mode = None

# 自定义CSS样式
custom_css = """
/* 顶部标题区：紫蓝渐变、居中 */
.hero-header {
    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    margin-bottom: 18px;
}

.hero-title {
    margin: 0;
    color: #FFFFFF;
    font-size: 2.2em;
    font-weight: 800;
    line-height: 1.2;
}

.hero-subtitle {
    margin-top: 12px;
    color: #FFFFFF;
    font-size: 1.05em;
    line-height: 1.8;
}

.hero-subtitle p,
.hero-subtitle strong {
    color: #FFFFFF !important;
}

/* 页面背景 */
body {
    background-color: white !important;
}

.gradio-container {
    background-color: white !important;
}

/* 按钮样式 */
.primary-btn {
    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
    border: none !important;
    color: white !important;
}
"""

def load_pipeline_cached(ckpt_dir, wav2vec_dir, model_type, sample_steps=None, performance_mode="aggressive"):
    """加载或复用缓存的pipeline"""
    global global_pipeline, global_model_type, global_sample_steps, global_performance_mode

    # 检查是否需要重新加载（模型类型、采样步数或性能档位变化）
    need_reload = (
        global_pipeline is None or
        global_model_type != model_type or
        global_sample_steps != sample_steps or
        global_performance_mode != performance_mode
    )

    if need_reload:
        logger.info(f"加载模型: {model_type}, sample_steps={sample_steps}, performance_mode={performance_mode}")
        global_pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            model_type=model_type,
            sample_steps=sample_steps,
            performance_mode=performance_mode
        )
        global_model_type = model_type
        global_sample_steps = sample_steps
        global_performance_mode = performance_mode
        logger.info("模型加载完成")
    else:
        logger.info(f"复用缓存的pipeline: {model_type}, sample_steps={sample_steps}, performance_mode={performance_mode}")

    return global_pipeline


def apply_quality_preset(quality_preset):
    """根据质量预设返回推荐参数"""
    presets = {
        "quality": {
            "sample_steps": 8,
            "performance_mode": "stable"
        },
        "balanced": {
            "sample_steps": 4,
            "performance_mode": "aggressive"
        },
        "speed": {
            "sample_steps": 2,
            "performance_mode": "aggressive"
        }
    }

    params = presets.get(quality_preset, presets["balanced"])
    logger.info(f"[QUALITY_PRESET] {quality_preset} -> steps={params['sample_steps']}, mode={params['performance_mode']}")
    return params["sample_steps"], params["performance_mode"]


def apply_gpu_preset(gpu_preset, model_type):
    """根据GPU预设返回推荐参数"""
    # GPU预设矩阵（针对不同GPU和模型类型）
    presets = {
        "rtx3090": {
            "lite": {"sample_steps": 2, "performance_mode": "aggressive"},
            "pro": {"sample_steps": 4, "performance_mode": "stable"}  # 显存受限
        },
        "rtx4090": {
            "lite": {"sample_steps": 2, "performance_mode": "aggressive"},
            "pro": {"sample_steps": 2, "performance_mode": "aggressive"}
        },
        "rtx5090": {
            "lite": {"sample_steps": 2, "performance_mode": "extreme"},
            "pro": {"sample_steps": 2, "performance_mode": "extreme"}
        },
        "a6000": {
            "lite": {"sample_steps": 2, "performance_mode": "aggressive"},
            "pro": {"sample_steps": 2, "performance_mode": "aggressive"}
        },
        "auto": {
            "lite": {"sample_steps": 2, "performance_mode": "aggressive"},
            "pro": {"sample_steps": 2, "performance_mode": "aggressive"}
        }
    }

    preset = presets.get(gpu_preset, presets["auto"])
    params = preset.get(model_type, preset["lite"])

    logger.info(f"[GPU_PRESET] {gpu_preset} + {model_type} -> steps={params['sample_steps']}, mode={params['performance_mode']}")
    return params["sample_steps"], params["performance_mode"]


def generate_video_webui(
    ckpt_dir,
    wav2vec_dir,
    model_type,
    cond_image,
    audio_file,
    quality_preset,
    gpu_preset,
    use_face_crop,
    audio_encode_mode,
    sample_steps,
    performance_mode,
    base_seed,
    progress=gr.Progress()
):
    """WebUI视频生成函数"""
    try:
        # 参数验证
        if not ckpt_dir or not os.path.exists(ckpt_dir):
            return None, "❌ 错误：模型目录不存在"

        if not wav2vec_dir or not os.path.exists(wav2vec_dir):
            return None, "❌ 错误：Wav2Vec2目录不存在"

        if cond_image is None:
            return None, "❌ 错误：请上传条件图像"

        if audio_file is None:
            return None, "❌ 错误：请上传音频文件"

        # 质量预设优先（最高优先级）
        if quality_preset != "balanced":
            sample_steps, performance_mode = apply_quality_preset(quality_preset)
        # GPU预设次之
        elif gpu_preset != "auto":
            sample_steps, performance_mode = apply_gpu_preset(gpu_preset, model_type)
        # 否则使用手动参数

        # 创建输出目录
        os.makedirs("outputs", exist_ok=True)

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"outputs/outputs_{timestamp}.mp4"

        # 基线统计：重置显存计数器
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        total_start = time.time()

        progress(0.1, desc="加载模型...")
        logger.info("=" * 50)
        logger.info("开始生成视频")
        logger.info(f"质量预设: {quality_preset}")
        logger.info(f"GPU预设: {gpu_preset}")
        logger.info(f"模型类型: {model_type}")
        logger.info(f"采样步数: {sample_steps}")
        logger.info(f"性能档位: {performance_mode}")
        logger.info(f"条件图像: {cond_image}")
        logger.info(f"音频文件: {audio_file}")
        logger.info(f"输出路径: {output_path}")

        # 加载pipeline
        pipeline = load_pipeline_cached(ckpt_dir, wav2vec_dir, model_type, sample_steps, performance_mode)

        progress(0.2, desc="准备数据...")
        # 准备基础数据
        get_base_data(
            pipeline,
            cond_image_path_or_dir=cond_image,
            base_seed=base_seed,
            use_face_crop=use_face_crop
        )

        infer_params = get_infer_params()
        sample_rate = infer_params['sample_rate']
        tgt_fps = infer_params['tgt_fps']
        cached_audio_duration = infer_params['cached_audio_duration']
        frame_num = infer_params['frame_num']
        motion_frames_num = infer_params['motion_frames_num']
        slice_len = frame_num - motion_frames_num

        progress(0.3, desc="加载音频...")
        # 加载音频
        human_speech_array_all, _ = librosa.load(
            audio_file,
            sr=sample_rate,
            mono=True
        )

        logger.info("数据准备完成，开始生成视频...")
        logger.info(f"[BASELINE] Audio duration: {len(human_speech_array_all)/sample_rate:.2f}s")

        generated_list = []
        audio_embed_time = 0.0
        denoise_decode_time = 0.0

        if audio_encode_mode == 'once':
            # 一次性编码音频
            progress(0.4, desc="编码音频...")
            torch.cuda.synchronize()
            audio_start = time.time()
            audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
            torch.cuda.synchronize()
            audio_embed_time = time.time() - audio_start
            logger.info(f"[BASELINE] Audio embed (once): {audio_embed_time:.3f}s")

            audio_embedding_chunks_list = [
                audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous()
                for i in range((audio_embedding_all.shape[1] - frame_num) // slice_len)
            ]

            total_chunks = len(audio_embedding_chunks_list)
            for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
                progress(
                    0.4 + 0.5 * (chunk_idx / total_chunks),
                    desc=f"生成视频片段 {chunk_idx + 1}/{total_chunks}..."
                )

                torch.cuda.synchronize()
                start_time = time.time()

                video = run_pipeline(pipeline, audio_embedding_chunk)

                torch.cuda.synchronize()
                end_time = time.time()
                chunk_time = end_time - start_time
                denoise_decode_time += chunk_time
                logger.info(f"[BASELINE] Chunk-{chunk_idx} denoise+decode: {chunk_time:.3f}s")

                generated_list.append(video.cpu())

        else:  # stream模式
            from collections import deque

            cached_audio_length_sum = sample_rate * cached_audio_duration
            audio_end_idx = cached_audio_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num

            audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

            human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
            human_speech_array_slices = human_speech_array_all[
                :(len(human_speech_array_all) // human_speech_array_slice_len) * human_speech_array_slice_len
            ].reshape(-1, human_speech_array_slice_len)

            total_chunks = len(human_speech_array_slices)
            for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
                progress(
                    0.4 + 0.5 * (chunk_idx / total_chunks),
                    desc=f"生成视频片段 {chunk_idx + 1}/{total_chunks}..."
                )

                torch.cuda.synchronize()
                start_time = time.time()

                # 流式编码音频
                audio_dq.extend(human_speech_array.tolist())
                audio_array = np.array(audio_dq)
                torch.cuda.synchronize()
                audio_start = time.time()
                audio_embedding = get_audio_embedding(
                    pipeline,
                    audio_array,
                    audio_start_idx,
                    audio_end_idx
                )
                torch.cuda.synchronize()
                audio_embed_time += time.time() - audio_start

                denoise_start = time.time()
                video = run_pipeline(pipeline, audio_embedding)
                torch.cuda.synchronize()
                denoise_decode_time += time.time() - denoise_start

                torch.cuda.synchronize()
                end_time = time.time()
                logger.info(f"[BASELINE] Chunk-{chunk_idx} total: {(end_time - start_time):.3f}s (audio: {time.time() - audio_start:.3f}s, denoise+decode: {time.time() - denoise_start:.3f}s)")

                generated_list.append(video.cpu())

        progress(0.95, desc="保存视频...")
        # 保存视频
        torch.cuda.synchronize()
        save_start = time.time()
        save_video(generated_list, output_path, audio_file, fps=tgt_fps)
        torch.cuda.synchronize()
        save_time = time.time() - save_start

        total_time = time.time() - total_start

        logger.info(f"✅ 视频已保存到: {output_path}")
        logger.info("=" * 60)
        logger.info("[BASELINE] Performance Summary:")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Audio embed: {audio_embed_time:.3f}s ({audio_embed_time/total_time*100:.1f}%)")
        logger.info(f"  Denoise+Decode: {denoise_decode_time:.3f}s ({denoise_decode_time/total_time*100:.1f}%)")
        logger.info(f"  Save video: {save_time:.3f}s ({save_time/total_time*100:.1f}%)")
        if torch.cuda.is_available():
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
            logger.info(f"  GPU Memory: allocated={max_allocated:.2f}GB, reserved={max_reserved:.2f}GB")
        logger.info("=" * 60)

        progress(1.0, desc="完成！")

        summary = f"""✅ 生成成功！
输出路径: {output_path}

[性能统计]
总耗时: {total_time:.2f}s
- 音频编码: {audio_embed_time:.2f}s ({audio_embed_time/total_time*100:.1f}%)
- 推理解码: {denoise_decode_time:.2f}s ({denoise_decode_time/total_time*100:.1f}%)
- 保存视频: {save_time:.2f}s ({save_time/total_time*100:.1f}%)
"""
        if torch.cuda.is_available():
            summary += f"显存峰值: {max_allocated:.2f}GB (allocated) / {max_reserved:.2f}GB (reserved)"

        return output_path, summary

    except Exception as e:
        error_msg = f"❌ 生成失败: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, error_msg


def create_webui():
    """创建Gradio界面"""

    with gr.Blocks(css=custom_css, title="SoulX-FlashHead WebUI") as demo:
        gr.HTML(
            """
            <div class="hero-header">
                <h1 class="hero-title">SoulX-FlashHead 实时说话人头像生成</h1>
                <div class="hero-subtitle">
                    <p><strong>webUI二次开发 by 科哥 | 微信：312088415 公众号：科哥玩AI</strong></p>
                    <p><strong>承诺永远开源使用 但是需要保留本人版权信息！</strong></p>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 模型配置")

                ckpt_dir = gr.Textbox(
                    label="FlashHead模型目录",
                    value="models/SoulX-FlashHead-1_3B",
                    placeholder="models/SoulX-FlashHead-1_3B"
                )

                wav2vec_dir = gr.Textbox(
                    label="Wav2Vec2模型目录",
                    value="models/wav2vec2-base-960h",
                    placeholder="models/wav2vec2-base-960h"
                )

                model_type = gr.Radio(
                    choices=["lite", "pro"],
                    value="lite",
                    label="模型类型",
                    info="Lite: 96 FPS (单卡实时) | Pro: 10.8 FPS (高质量)"
                )

                gr.Markdown("### 🎨 输入数据")

                cond_image = gr.Image(
                    label="条件图像",
                    type="filepath",
                    height=300
                )

                audio_file = gr.Audio(
                    label="音频文件",
                    type="filepath"
                )

                gr.Markdown("### ⚙️ 生成参数")

                # 质量/速度预设
                quality_preset = gr.Radio(
                    choices=["quality", "balanced", "speed"],
                    value="balanced",
                    label="质量预设",
                    info="quality: 质量优先 | balanced: 平衡（推荐） | speed: 速度优先"
                )

                # GPU预设（自动配置最优参数）
                gpu_preset = gr.Radio(
                    choices=["auto", "rtx3090", "rtx4090", "rtx5090", "a6000"],
                    value="auto",
                    label="GPU预设",
                    info="auto: 自动检测 | 其他: 手动选择GPU型号（自动配置最优参数）"
                )

                use_face_crop = gr.Checkbox(
                    label="启用人脸裁剪",
                    value=False,
                    info="使用MediaPipe检测并裁剪人脸"
                )

                audio_encode_mode = gr.Radio(
                    choices=["stream", "once"],
                    value="stream",
                    label="音频编码模式",
                    info="stream: 流式编码（推荐） | once: 一次性编码"
                )

                sample_steps = gr.Number(
                    label="采样步数",
                    value=4,
                    minimum=1,
                    maximum=20,
                    step=1,
                    precision=0,
                    info="推荐：2-4步（步数越少越快，步数越多质量越高）"
                )

                performance_mode = gr.Radio(
                    choices=["stable", "aggressive", "extreme"],
                    value="aggressive",
                    label="性能档位",
                    info="stable: 稳健 | aggressive: 激进（推荐） | extreme: 狂暴（fp16+TF32）"
                )

                base_seed = gr.Number(
                    label="随机种子",
                    value=42,
                    precision=0
                )

                generate_btn = gr.Button(
                    "🚀 生成视频",
                    variant="primary",
                    elem_classes=["primary-btn"]
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🎬 生成结果")

                output_video = gr.Video(
                    label="生成的视频",
                    height=400
                )

                output_log = gr.Textbox(
                    label="生成日志",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )

        # 绑定生成按钮
        generate_btn.click(
            fn=generate_video_webui,
            inputs=[
                ckpt_dir,
                wav2vec_dir,
                model_type,
                cond_image,
                audio_file,
                quality_preset,
                gpu_preset,
                use_face_crop,
                audio_encode_mode,
                sample_steps,
                performance_mode,
                base_seed
            ],
            outputs=[output_video, output_log]
        )

        gr.Markdown(
            """
            ---
            ### 📖 使用说明
            1. **模型配置**: 确保模型已下载到指定目录
            2. **上传数据**: 上传人脸图像和音频文件
            3. **质量预设**: 选择质量优先/平衡/速度优先（推荐）
            4. **生成视频**: 点击"生成视频"按钮开始生成

            ### 💡 提示
            - **质量预设**: quality=8步+稳健（质量最高），balanced=4步+激进（推荐），speed=2步+激进（最快）
            - **GPU预设**: 可选，自动配置GPU专属参数（RTX3090/4090/5090/A6000）
            - **Lite模型**: 速度快，适合实时应用（96 FPS）
            - **Pro模型**: 质量高，适合高质量输出（10.8 FPS）
            - **采样步数**: 手动调节（质量预设会覆盖此值）
            - **性能档位**: stable=稳健（bf16），aggressive=激进（推荐），extreme=狂暴（fp16+TF32）

            ### 🎯 质量问题排查
            - **视频模糊**: 选择"quality"预设，或使用Pro模型
            - **速度太慢**: 选择"speed"预设，或使用Lite模型
            - **显存不足**: 选择"balanced"预设，降低采样步数

            ### 📧 联系方式
            - 微信：312088415
            - 公众号：科哥玩AI
            """
        )

    return demo


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 创建并启动WebUI
    demo = create_webui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

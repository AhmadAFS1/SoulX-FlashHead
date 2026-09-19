# SoulX-FlashHead 用户使用手册

**版本**: v2.0.0 激进优化版
**作者**: 科哥 | 微信：312088415 公众号：科哥玩AI
**仓库地址**: https://github.com/kegeai888/SoulX-FlashHead

---

## 📖 快速开始

### 启动WebUI

```bash
bash start_app.sh
```

访问地址：**http://localhost:7860**

---

## 🎯 WebUI使用指南

### 1. 基础操作流程

#### 步骤1：选择GPU预设（推荐）

在"GPU预设"下拉框中选择你的GPU型号：
- **auto**：自动检测（推荐新手）
- **rtx3090**：RTX 3090
- **rtx4090**：RTX 4090
- **rtx5090**：RTX 5090
- **a6000**：NVIDIA A6000

**为什么要选GPU预设？**
- 自动配置最优参数（采样步数、性能档位）
- 避免显存不足（OOM）
- 获得最佳性能

#### 步骤2：选择模型类型

- **lite**：速度快，适合实时应用（96 FPS）
- **pro**：质量高，适合高质量输出（10.8 FPS）

**如何选择？**
- 追求速度 → lite
- 追求质量 → pro
- 不确定 → 先试lite

#### 步骤3：上传输入数据

1. **条件图像**：点击"条件图像"区域，上传人脸图像
   - 支持格式：PNG、JPG、JPEG
   - 推荐尺寸：512x512（会自动resize）
   - 建议：正面人脸、光线均匀、背景简洁

2. **音频文件**：点击"音频文件"区域，上传语音音频
   - 支持格式：WAV、MP3、M4A等
   - 采样率：自动转换为16kHz
   - 时长：无限制（长音频自动分段处理）

#### 步骤4：调整生成参数（可选）

如果你选择了GPU预设，这些参数会自动配置，无需手动调整。

**高级用户可手动调整**：

- **启用人脸裁剪**：自动检测并裁剪人脸（推荐关闭，除非图像包含多人）
- **音频编码模式**：
  - **stream**（推荐）：流式编码，适合长音频，内存占用低
  - **once**：一次性编码，适合短音频（<30秒），速度稍快
- **采样步数**：
  - **2**：激进档，速度优先（推荐）
  - **4**：稳健档，质量优先
- **性能档位**：
  - **stable**：稳健，最安全
  - **aggressive**：激进，推荐（速度与稳定平衡）
  - **extreme**：狂暴，最快（可能OOM，自动回退）

#### 步骤5：生成视频

点击"🚀 生成视频"按钮，等待生成完成。

**生成过程**：
1. 加载模型（首次较慢，后续复用缓存）
2. 准备数据（图像编码、音频加载）
3. 生成视频片段（逐chunk生成，显示进度）
4. 保存视频（合并音频）

**生成时间**：
- Lite模型：10秒音频约5-10秒
- Pro模型：10秒音频约20-40秒
- 具体时间取决于GPU型号和档位

#### 步骤6：查看结果

生成完成后：
1. **视频预览**：在右侧"生成的视频"区域自动播放
2. **性能统计**：在"生成日志"区域查看详细统计
   - 总耗时
   - 各阶段耗时占比（音频编码、推理解码、保存视频）
   - 显存峰值
3. **文件保存**：自动保存到 `outputs/outputs_年月日时分秒.mp4`

---

## 🎨 GPU预设推荐配置

### 配置矩阵

| GPU型号 | Lite模型 | Pro模型 | 说明 |
|---------|---------|---------|------|
| **RTX 3090** | 2步+激进 | 4步+稳健 | 显存24GB，Pro模型建议稳健档 |
| **RTX 4090** | 2步+激进 | 2步+激进 | 显存24GB，性能强劲 |
| **RTX 5090** | 2步+狂暴 | 2步+狂暴 | 显存32GB，可用狂暴档 |
| **A6000** | 2步+激进 | 2步+激进 | 显存48GB，专业卡 |

### 如何选择？

**场景1：我有RTX 4090，想要最快速度**
- 选择：GPU预设=rtx4090，模型类型=lite
- 预期：10秒音频约5秒生成

**场景2：我有RTX 3090，想要高质量**
- 选择：GPU预设=rtx3090，模型类型=pro
- 预期：10秒音频约30秒生成（稳健档，避免OOM）

**场景3：我有RTX 5090，想要极致性能**
- 选择：GPU预设=rtx5090，模型类型=pro
- 预期：10秒音频约15秒生成（狂暴档，fp16+TF32）

**场景4：我不知道我的GPU型号**
- 选择：GPU预设=auto
- 系统会自动配置安全参数

---

## ⚙️ 参数详解

### 采样步数

**作用**：控制生成质量与速度的平衡

- **2步（激进档）**：
  - 速度：快（理论2倍于4步）
  - 质量：略低于4步，但肉眼难以区分
  - 推荐：追求速度、实时应用

- **4步（稳健档）**：
  - 速度：慢
  - 质量：更高
  - 推荐：追求质量、显存受限（如RTX 3090 Pro模型）

### 性能档位

**作用**：控制内核加速与精度

- **stable（稳健）**：
  - 精度：bf16
  - 加速：默认后端
  - 推荐：调试、首次使用

- **aggressive（激进）**：
  - 精度：bf16
  - 加速：cudnn.benchmark
  - 推荐：日常使用（速度与稳定平衡）

- **extreme（狂暴）**：
  - 精度：fp16
  - 加速：cudnn.benchmark + TF32
  - 推荐：追求极致性能（可能OOM，自动回退）

### 音频编码模式

**作用**：控制音频处理方式

- **stream（流式）**：
  - 内存占用：低
  - 速度：适中
  - 推荐：长音频（>30秒）、显存受限

- **once（一次性）**：
  - 内存占用：高
  - 速度：稍快
  - 推荐：短音频（<30秒）

---

## 📊 性能统计解读

生成完成后，"生成日志"区域会显示性能统计：

```
✅ 生成成功！
输出路径: outputs/outputs_20260303120000.mp4

[性能统计]
总耗时: 15.23s
- 音频编码: 1.23s (8.1%)
- 推理解码: 12.46s (81.8%)
- 保存视频: 1.54s (10.1%)
显存峰值: 8.45GB (allocated) / 9.12GB (reserved)
```

**如何解读？**

1. **总耗时**：端到端生成时间
   - 越短越好
   - 对比基线，评估优化效果

2. **音频编码**：音频预处理与特征提取时间
   - 占比通常5-10%
   - stream模式略高于once模式

3. **推理解码**：模型推理与VAE解码时间
   - 占比通常80-90%（主要瓶颈）
   - 采样步数减半，此项时间理论减半

4. **保存视频**：视频编码与音频合并时间
   - 占比通常5-10%
   - 与视频长度成正比

5. **显存峰值**：
   - allocated：实际使用的显存
   - reserved：PyTorch预留的显存
   - 如果接近GPU显存上限，建议降低档位

---

## ❓ 常见问题

### Q1: 生成速度慢怎么办？

**解决方案**：
1. **使用GPU预设**：选择你的GPU型号，自动配置最优参数
2. **使用Lite模型**：速度比Pro模型快5-10倍
3. **降低采样步数**：4步 → 2步
4. **提升性能档位**：stable → aggressive → extreme
5. **检查GPU利用率**：运行 `nvidia-smi`，确保GPU利用率>80%

### Q2: 遇到OOM（显存不足）怎么办？

**解决方案**：
1. **使用GPU预设**：自动配置安全参数（推荐）
2. **降低性能档位**：extreme → aggressive → stable
3. **增加采样步数**：2步 → 4步（降低显存占用）
4. **使用Lite模型**：显存占用比Pro模型小
5. **关闭其他程序**：释放显存
6. **等待自动回退**：extreme档OOM会自动回退到aggressive

### Q3: 视频质量不满意怎么办？

**解决方案**：
1. **使用Pro模型**：质量比Lite模型高
2. **增加采样步数**：2步 → 4步
3. **降低性能档位**：extreme → aggressive（fp16可能影响质量）
4. **使用高质量输入**：
   - 图像：正面人脸、光线均匀、高分辨率
   - 音频：清晰、无噪音、16kHz以上采样率
5. **启用颜色校正**（需手动修改配置文件）

### Q4: 视频保存在哪里？

**答案**：
- 所有生成的视频保存在 `outputs/` 目录
- 文件名格式：`outputs_年月日时分秒.mp4`
- 例如：`outputs/outputs_20260303120000.mp4`

### Q5: 如何批量生成视频？

**方案1：使用命令行**（推荐）
```bash
# 编写批处理脚本
for image in examples/*.png; do
    for audio in examples/*.wav; do
        python generate_video.py \
            --ckpt_dir models/SoulX-FlashHead-1_3B \
            --wav2vec_dir models/wav2vec2-base-960h \
            --model_type lite \
            --cond_image "$image" \
            --audio_path "$audio" \
            --sample_steps 2 \
            --performance_mode aggressive
    done
done
```

**方案2：使用WebUI**
- 逐个上传生成（暂不支持批量）

### Q6: 首次生成很慢，后续会快吗？

**答案**：会！

**原因**：
- 首次运行需要编译模型（`torch.compile`）
- 编译开销较大（可能需要1-2分钟）
- 后续运行会复用编译缓存，速度提升明显

**建议**：
- 首次生成时耐心等待
- 后续生成会快很多

### Q7: 如何查看当前使用的注意力后端？

**答案**：
查看启动日志，搜索 `[PERF] Attention backend`：
```
[PERF] Attention backend: SageAttention (fastest)
```

**后端优先级**：
1. SageAttention（最快）
2. Flash Attention 3
3. Flash Attention 2
4. PyTorch SDPA（兜底）

**如何安装SageAttention？**
```bash
pip install sageattention==2.2.0 --no-build-isolation
```

---

## 🔧 命令行使用（高级）

### 单卡推理

```bash
# Lite模型（激进档）
bash inference_script_single_gpu_lite.sh

# Pro模型（激进档）
bash inference_script_single_gpu_pro.sh

# 自定义参数
python generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite \
    --cond_image examples/girl.png \
    --audio_path examples/podcast.wav \
    --audio_encode_mode stream \
    --sample_steps 2 \
    --performance_mode aggressive
```

### 多卡推理

```bash
# 双卡Pro模式（推荐）
bash inference_script_multi_gpu_pro.sh

# 自定义GPU数量
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_NUM=4 torchrun --nproc_per_node=4 generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type pro \
    --cond_image examples/girl.png \
    --audio_path examples/podcast.wav \
    --audio_encode_mode stream \
    --sample_steps 2 \
    --performance_mode aggressive
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ckpt_dir` | FlashHead模型目录 | - |
| `--wav2vec_dir` | Wav2Vec2模型目录 | - |
| `--model_type` | 模型类型（lite/pro） | - |
| `--cond_image` | 条件图像路径 | - |
| `--audio_path` | 音频文件路径 | - |
| `--audio_encode_mode` | 音频编码模式（stream/once） | `stream` |
| `--sample_steps` | 采样步数（2/4） | `2` |
| `--performance_mode` | 性能档位（stable/aggressive/extreme） | `aggressive` |
| `--use_face_crop` | 启用人脸裁剪 | `False` |
| `--base_seed` | 随机种子 | `42` |

---

## 💡 使用技巧

### 技巧1：快速测试

**场景**：想快速测试效果，不在乎质量

**配置**：
- GPU预设：auto
- 模型类型：lite
- 采样步数：2
- 性能档位：aggressive

**预期**：10秒音频约5秒生成

### 技巧2：高质量输出

**场景**：需要高质量视频，不在乎时间

**配置**：
- GPU预设：根据你的GPU选择
- 模型类型：pro
- 采样步数：4
- 性能档位：stable

**预期**：10秒音频约30-60秒生成

### 技巧3：极致性能

**场景**：追求极致速度，有RTX 5090

**配置**：
- GPU预设：rtx5090
- 模型类型：lite
- 采样步数：2
- 性能档位：extreme

**预期**：10秒音频约3秒生成

### 技巧4：显存受限

**场景**：显存不足（如RTX 3090 Pro模型）

**配置**：
- GPU预设：rtx3090
- 模型类型：pro
- 采样步数：4
- 性能档位：stable

**预期**：10秒音频约40秒生成，不会OOM

### 技巧5：长音频处理

**场景**：处理长音频（>5分钟）

**配置**：
- 音频编码模式：stream（必选）
- 其他参数：根据GPU选择

**说明**：stream模式会自动分段处理，避免显存不足

---

## 📞 获取帮助

### 联系方式

- **微信**: 312088415
- **公众号**: 科哥玩AI

### 反馈问题

遇到问题时，请提供以下信息：
1. GPU型号
2. 模型类型（lite/pro）
3. 参数配置（采样步数、性能档位）
4. 错误日志（完整的终端输出）
5. 输入数据（图像、音频）

---

## 🎓 进阶阅读

想了解更多技术细节？查看以下文档：
- `WEBUI_README.md`：WebUI技术细节
- `todo.md`：二次开发记录
- `CLAUDE.md`：项目架构说明

---

**最后更新**: 2026-03-03
**版本**: v2.0.0 激进优化版

**感谢使用 SoulX-FlashHead！**

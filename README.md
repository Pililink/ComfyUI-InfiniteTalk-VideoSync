# ComfyUI-InfiniteTalk-VideoSync 🎤

一站式 InfiniteTalk 视频对口型节点 — **输入视频路径 + 音频 → 内部完成所有处理 → 输出视频路径**

## 为什么需要这个节点

现有方案的根本矛盾：

| 方案 | 动作一致 | 长视频 | 音画同步 | 段间连续 |
|------|---------|--------|---------|---------|
| **A (Kijai 内部循环)** | ✅ WanVideoEncode | ❌ 全量加载OOM | ✅ 内部对齐 | ✅ latent重叠 |
| **B (官方外部循环)** | ❌ 只有参考图 | ✅ 分段保存 | ✅ audio_offset | ✅ previous_frames |
| **C (A+B 强行合并)** | ✅ | ✅ | ❌ 累积漂移 | ❌ 像素拼接 |
| **本节点** | ✅ | ✅ | ✅ | ✅ |

**核心思路**: 不在 ComfyUI 工作流层面拼接，而是在 Python 节点内部完成分段 + 内部循环 + 拼接的全部逻辑。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│ InfiniteTalk Video Sync (Video Path) 🎤                        │
│                                                                 │
│  输入:                                                          │
│    video_path ─────────────────┐                                │
│    audio_path ─────────────┐   │                                │
│    参数(target_size, fps等)│   │                                │
│                            │   │                                │
│  内部处理:                  ▼   ▼                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Phase 0: ffprobe 读取视频元数据 (零内存)                 │   │
│  │ Phase 1: 加载所有模型 (一次性)                           │   │
│  │   Wan I2V + InfiniteTalk + LoRA + VAE + CLIP + Wav2Vec  │   │
│  │ Phase 2: 提取完整音频 wav2vec 特征 (一次性)              │   │
│  │ Phase 3: 分段处理                                        │   │
│  │   for each segment (30秒):                               │   │
│  │     ├─ ffmpeg 按需读取源视频帧 (仅当前段)                │   │
│  │     ├─ WanVideoEncode 编码源帧 → latent (动作引导)       │   │
│  │     ├─ 切取当前段音频特征                                │   │
│  │     ├─ WanVideoSampler + 内部 multitalk_loop:            │   │
│  │     │    81帧窗口 + motion_frame latent重叠               │   │
│  │     │    音频特征自动切片对齐                              │   │
│  │     │    运动帧噪声注入 → 完美连续                        │   │
│  │     ├─ WanVideoDecode → 分段保存 mp4                     │   │
│  │     └─ 释放显存                                          │   │
│  │ Phase 4: ffmpeg concat 所有分段 + 合并音频                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  输出:                                                          │
│    video_path → 最终 mp4 路径                                   │
│    filename → 文件名                                            │
│    audio → 音频                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 为什么能解决之前的问题

### 1. 内存问题 → ffmpeg 按需读取
- 不使用 VHS_LoadVideo 全量加载
- 每段只读 30 秒 × 25fps × 480×832 ≈ 3.6GB
- 处理完释放，下一段重新读取

### 2. 音画同步 → 一次性音频特征 + 偏移切片
- 完整音频的 wav2vec 特征一次提取
- 每段按全局帧偏移切片，无累积漂移
- 不再手动 Audio Cut

### 3. 段间连续 → 内部 multitalk_loop
- 每段内部走 Kijai 的 multitalk_loop
- 81 帧窗口 + 25 帧 latent 重叠 → 完美连续
- 不是像素级 CrossFade，是 latent 空间运动帧注入

### 4. 动作一致 → WanVideoEncode
- 每段编码源视频帧 → latent → 喂给 Sampler
- 生成的动作跟随源视频，不是自由发挥

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-InfiniteTalk-VideoSync.git
```

### 前置依赖
- 无需额外安装 `ComfyUI-WanVideoWrapper`，当前仓库已内置运行时
- ffmpeg (系统 PATH 中可用)
- 所有 InfiniteTalk 模型文件

## 使用方法

1. 添加 `InfiniteTalk Video Sync (Video Path) 🎤` 节点
2. 填入源视频路径和音频路径
3. 调整参数（大部分保持默认即可）
4. 运行

### 推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| target_width | 480 | 模型训练分辨率 |
| target_height | 832 | 模型训练分辨率 |
| target_fps | 25 | 标准帧率 |
| segment_seconds | 30 | 30秒/段, 平衡内存和接缝 |
| frame_window_size | 81 | 标准窗口大小 |
| motion_frame | 25 | 窗口间重叠, 越大越连续 |
| steps | 5 | 配合 LightX2V LoRA |
| block_swap | 30 | RTX 4090 推荐 |

### 长视频建议

- 5分钟视频: segment_seconds=30, 约10段, ~33分钟推理
- 10分钟视频: segment_seconds=30, 约20段, ~66分钟推理
- 段间边界每30秒出现一次，远好于外部循环的每3.2秒

## 项目状态

⚠️ **Beta** — 当前仓库已内置 WanVideo / InfiniteTalk 运行时，不再依赖外部安装 `ComfyUI-WanVideoWrapper`，但仍需要在你的 ComfyUI 运行环境里做一次实机验证。

### 需要完成的工作

1. **实机回归**: 在你的 ComfyUI 环境里跑一条真实视频，确认内置 runtime 与本机模型目录、模型文件名兼容
2. **参数收敛**: 根据素材类型微调 `segment_seconds / motion_frame / start_step`
3. **质量验证**: 对比旧工作流的动作一致性、口型同步和段间连续性

## 致谢

- [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) — 核心推理引擎
- [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) — InfiniteTalk 模型
- [Pililink/ComfyUI-Pililink-LatentSyncWrapper](https://github.com/Pililink/ComfyUI-Pililink-LatentSyncWrapper) — 节点设计模式参考

本仓库 `vendor/wanvideo_wrapper` 内嵌了 WanVideoWrapper 所需运行时代码，遵循其 Apache 2.0 许可证。

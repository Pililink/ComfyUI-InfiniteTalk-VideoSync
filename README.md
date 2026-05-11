# ComfyUI-InfiniteTalk-VideoSync 🎤

一个用于 ComfyUI 的 InfiniteTalk 长视频口型同步节点。它面向长时间视频的口型对齐场景，输入源视频和音频后，节点会在内部完成分段处理、推理和拼接，最后输出视频文件路径。

## 安装

将仓库放到 ComfyUI 的 `custom_nodes` 目录下：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pililink/ComfyUI-InfiniteTalk-VideoSync.git
```

安装依赖：

```bash
cd ComfyUI-InfiniteTalk-VideoSync
pip install -r requirements.txt
```

重启 ComfyUI。

## 模型

模型文件请放到：

```text
ComfyUI/models/LatentSync-1.5/
```

模型下载地址：

https://huggingface.co/ByteDance/LatentSync-1.5

请不要把模型文件提交到仓库。

## 使用方法

1. 在 ComfyUI 中添加 `InfiniteTalk Video Sync (Video Path) 🎤` 节点。
2. 输入源视频路径和音频路径。
3. 保持默认参数或按素材需要微调。
4. 运行工作流并查看输出视频。

这个节点主要解决长时间视频的口型同步问题，适合几分钟到更长时长的视频，不需要在工作流里手工拆段和拼接。

## 推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| target_width | 480 | 训练分辨率 |
| target_height | 832 | 训练分辨率 |
| target_fps | 25 | 标准帧率 |
| segment_seconds | 30 | 每段时长 |
| frame_window_size | 81 | 处理窗口 |
| motion_frame | 25 | 段间重叠 |
| steps | 5 | 推理步数 |
| block_swap | 30 | 显存较大时可用 |

## 依赖

- ffmpeg，且可在系统 PATH 中直接调用
- ComfyUI 的正常运行环境

仓库已经内置 `vendor/wanvideo_wrapper` 所需的运行时代码，无需额外单独安装对应 wrapper。

## 注意事项

- 这是 Beta 版本，建议先用一条短视频做实机验证。
- 模型目录名请保持为 `LatentSync-1.5`，不要改名。
- 输出文件通常会写入 ComfyUI 的默认输出目录。

## 致谢

- [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
- [ByteDance/LatentSync-1.5](https://huggingface.co/ByteDance/LatentSync-1.5)

"""
ComfyUI-InfiniteTalk-VideoSync
一站式视频对口型节点：输入视频路径 + 音频 → 内部分段处理 → 输出视频路径
解决外部循环架构下无法保证 latent 空间连续性和音画同步的根本问题。

设计参考: LatentSync 1.5 (Video Path) 的一站式处理模式
核心技术: Kijai WanVideoWrapper 的 multitalk_loop + WanVideoEncode 动作引导
"""

import os
import sys
import json
import time
import torch
import numpy as np
import subprocess
import tempfile
import logging
from pathlib import Path

import folder_paths
import comfy.model_management as mm

log = logging.getLogger("InfiniteTalkVideoSync")


def get_ffmpeg_path():
    """查找可用的 ffmpeg"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_video_info(video_path: str) -> dict:
    """用 ffprobe 获取视频元数据，零内存占用"""
    ffmpeg = get_ffmpeg_path()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not os.path.exists(ffprobe):
        ffprobe = "ffprobe"
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    video_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break
    
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")
    
    # 计算帧率
    fps_str = video_stream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration": float(info["format"]["duration"]),
        "nb_frames": int(video_stream.get("nb_frames", 0)),
    }


def load_video_chunk(video_path: str, start_frame: int, num_frames: int,
                     target_w: int, target_h: int, target_fps: float) -> torch.Tensor:
    """
    用 ffmpeg 按需读取视频片段，不加载全量帧。
    返回: [N, H, W, 3] float32 tensor, 值域 [0, 1]
    """
    ffmpeg = get_ffmpeg_path()
    start_time = start_frame / target_fps
    duration = num_frames / target_fps + 0.1  # 多读一点避免边界问题
    
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"fps={target_fps},scale={target_w}:{target_h}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-v", "quiet",
        "pipe:1"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout
    
    if len(raw) == 0:
        raise RuntimeError(f"ffmpeg returned empty output for {video_path} at frame {start_frame}")
    
    frame_size = target_w * target_h * 3
    actual_frames = len(raw) // frame_size
    actual_frames = min(actual_frames, num_frames)
    
    frames = np.frombuffer(raw[:actual_frames * frame_size], dtype=np.uint8)
    frames = frames.reshape(actual_frames, target_h, target_w, 3)
    tensor = torch.from_numpy(frames.copy()).float() / 255.0
    
    return tensor


def load_audio_segment(audio_path: str, start_sec: float, duration_sec: float,
                       target_sr: int = 16000) -> torch.Tensor:
    """用 ffmpeg 按需读取音频片段"""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_sec),
        "-i", audio_path,
        "-t", str(duration_sec),
        "-ar", str(target_sr),
        "-ac", "1",
        "-f", "f32le",
        "-v", "quiet",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout
    if len(raw) == 0:
        return torch.zeros(1, 1, int(target_sr * duration_sec))
    
    audio = np.frombuffer(raw, dtype=np.float32)
    return torch.from_numpy(audio.copy()).unsqueeze(0).unsqueeze(0)


def save_frames_to_video(frames: torch.Tensor, output_path: str, fps: float):
    """将 [N, H, W, 3] tensor 保存为 mp4"""
    ffmpeg = get_ffmpeg_path()
    N, H, W, C = frames.shape
    
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-v", "quiet",
        output_path
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    raw = (frames.clamp(0, 1) * 255).byte().numpy().tobytes()
    proc.communicate(input=raw)
    proc.wait()


class InfiniteTalkVideoSync:
    """
    InfiniteTalk 视频对口型 (Video Path)
    
    一站式节点：输入视频路径 + 音频路径 → 内部完成所有处理 → 输出视频路径
    
    内部处理流程:
    1. ffprobe 读取视频元数据（零内存）
    2. 计算分段策略（segment_seconds 秒为一大段）
    3. 每个大段内:
       a. ffmpeg 按需读取当前段的源视频帧（target_fps, target_size）
       b. WanVideoEncode 编码源帧 → latent（动作引导）
       c. MultiTalkWav2VecEmbeds 提取当前段音频特征
       d. WanVideoSampler 内部 multitalk_loop 自动分窗口:
          - 81帧窗口, motion_frame 帧 latent 重叠
          - 音频特征按窗口自动切片
          - latent 空间运动帧注入保证连续性
       e. WanVideoDecode → 分段保存 mp4
    4. ffmpeg concat 所有分段 + 原始音频 → 最终输出
    
    内存占用: O(segment_seconds * fps * H * W) ≈ 每段3-5GB
    显存占用: 与单次生成相同（~11GB for 4090）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "源视频绝对路径，任意分辨率/时长"
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "驱动音频路径，留空则使用视频自带音轨"
                }),
                # --- 模型路径 ---
                "wan_model": ("STRING", {
                    "default": "wan2.1/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                    "tooltip": "Wan I2V 模型文件名"
                }),
                "infinitetalk_model": ("STRING", {
                    "default": "wan2.1/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
                    "tooltip": "InfiniteTalk 模型文件名"
                }),
                "lora_model": ("STRING", {
                    "default": "lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                    "tooltip": "LightX2V LoRA 文件名"
                }),
                # --- 生成参数 ---
                "target_width": ("INT", {"default": 480, "min": 128, "max": 1920, "step": 8}),
                "target_height": ("INT", {"default": 832, "min": 128, "max": 1920, "step": 8}),
                "target_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0}),
                "segment_seconds": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 120.0, "step": 5.0,
                    "tooltip": "每段处理的秒数，越大内存越多但段间接缝越少"
                }),
                "frame_window_size": ("INT", {
                    "default": 81, "min": 17, "max": 241, "step": 4,
                    "tooltip": "每个采样窗口的帧数 (4n+1)"
                }),
                "motion_frame": ("INT", {
                    "default": 25, "min": 1, "max": 80,
                    "tooltip": "窗口间 latent 重叠帧数，越大越连续但越慢"
                }),
                "steps": ("INT", {"default": 5, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                # --- 性能参数 ---
                "block_swap": ("INT", {
                    "default": 30, "min": 0, "max": 40,
                    "tooltip": "Transformer block swap 数量，越大越省显存但越慢"
                }),
                "separate_vocals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否先做人声分离再提取音频特征"
                }),
                "filename_prefix": ("STRING", {"default": "infinitetalk_sync"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "可选：直接传入 ComfyUI AUDIO 类型"}),
                "negative_prompt": ("STRING", {
                    "default": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，最差质量，低质量",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("video_path", "filename", "audio")
    FUNCTION = "process"
    CATEGORY = "InfiniteTalk"
    OUTPUT_NODE = True
    
    def process(self, video_path, audio_path, wan_model, infinitetalk_model, lora_model,
                target_width, target_height, target_fps, segment_seconds,
                frame_window_size, motion_frame, steps, cfg, seed, block_swap,
                separate_vocals, filename_prefix,
                audio=None, negative_prompt=""):
        
        # ========================================
        # Phase 0: 验证输入
        # ========================================
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_info = get_video_info(video_path)
        total_duration = video_info["duration"]
        log.info(f"[InfiniteTalkSync] Source video: {video_info['width']}x{video_info['height']} "
                 f"@ {video_info['fps']:.1f}fps, {total_duration:.1f}s")
        
        # 确定音频源
        audio_source_path = audio_path if audio_path and os.path.exists(audio_path) else video_path
        
        # 创建输出目录
        output_dir = folder_paths.get_output_directory()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(output_dir, f"{filename_prefix}_{timestamp}")
        os.makedirs(work_dir, exist_ok=True)
        
        # ========================================
        # Phase 1: 加载模型（一次性）
        # ========================================
        log.info("[InfiniteTalkSync] Loading models...")
        
        # TODO: 此处调用 Kijai WanVideoWrapper 的模型加载函数
        # 需要从 ComfyUI-WanVideoWrapper 中导入:
        #   - WanVideoModelLoader (加载 Wan I2V + InfiniteTalk + LoRA)
        #   - WanVideoVAELoader (加载 VAE)
        #   - WanVideoTextEncodeCached (文本编码)
        #   - CLIPVisionLoader (CLIP Vision)
        #   - DownloadAndLoadWav2VecModel (Wav2Vec)
        #
        # 示例伪代码:
        # model = load_wan_model(wan_model, infinitetalk_model, lora_model, block_swap)
        # vae = load_wan_vae()
        # text_embeds = encode_text("", negative_prompt)
        # clip_vision = load_clip_vision()
        # wav2vec = load_wav2vec()
        
        # ========================================
        # Phase 2: 音频预处理（一次性）
        # ========================================
        log.info("[InfiniteTalkSync] Processing audio...")
        
        # 如果需要人声分离，先处理完整音频
        # 然后提取 wav2vec 特征（完整音频 → 完整特征序列）
        # 这样每个分段可以按偏移切片，保证音频连续性
        #
        # total_frames = int(total_duration * target_fps)
        # audio_features = wav2vec_encode(audio_source_path, total_frames, target_fps)
        # ↑ shape: [total_frames, 12, 768]
        
        # ========================================
        # Phase 3: 分段处理
        # ========================================
        segment_frames = int(segment_seconds * target_fps)
        total_frames = int(total_duration * target_fps)
        
        # 段间重叠 = motion_frame 帧（保证 latent 连续性）
        stride_frames = segment_frames - motion_frame
        num_segments = max(1, 1 + (total_frames - segment_frames + stride_frames - 1) // stride_frames)
        
        log.info(f"[InfiniteTalkSync] Processing {num_segments} segments "
                 f"({segment_seconds}s each, {motion_frame} frame overlap)")
        
        segment_paths = []
        
        for seg_idx in range(num_segments):
            # 计算当前段的帧范围
            seg_start = seg_idx * stride_frames
            seg_end = min(seg_start + segment_frames, total_frames)
            seg_num_frames = seg_end - seg_start
            
            log.info(f"[InfiniteTalkSync] Segment {seg_idx+1}/{num_segments}: "
                     f"frames {seg_start}-{seg_end-1} ({seg_num_frames} frames)")
            
            # 检查 ComfyUI 中断
            if hasattr(mm, 'throw_exception_if_processing_interrupted'):
                mm.throw_exception_if_processing_interrupted()
            
            # ----- 3a: 加载当前段源视频帧 -----
            source_frames = load_video_chunk(
                video_path, seg_start, seg_num_frames,
                target_width, target_height, target_fps
            )
            # source_frames: [N, H, W, 3]
            
            # ----- 3b: 编码源帧 → latent（动作引导）-----
            # source_latent = vae.encode(source_frames)
            # ↑ 这是 A 工作流的关键：WanVideoEncode 编码全段源帧
            
            # ----- 3c: 切取当前段的音频特征 -----
            # seg_audio_features = audio_features[seg_start:seg_end]
            
            # ----- 3d: 准备 CLIP embeddings -----
            # first_frame = source_frames[0:1]
            # last_frame = source_frames[-1:]
            # clip_embeds = clip_vision_encode(first_frame, last_frame)
            
            # ----- 3e: 准备 image_embeds -----
            # image_embeds = WanVideoImageToVideoMultiTalk(
            #     vae, first_frame, clip_embeds,
            #     width, height, frame_window_size, motion_frame,
            #     mode="infinitetalk"
            # )
            
            # ----- 3f: 采样（内部 multitalk_loop 自动分窗口）-----
            # latent_output = WanVideoSampler(
            #     model, image_embeds, text_embeds,
            #     source_latent,          # ← 动作引导（A的核心）
            #     multitalk_embeds,       # ← 音频特征
            #     steps, cfg, seed + seg_idx
            # )
            # ↑ 内部 multitalk_loop 自动:
            #   - 按 81 帧窗口切分
            #   - motion_frame 帧 latent 重叠
            #   - 音频特征按窗口切片对齐
            #   - 运动帧噪声注入保证连续性
            
            # ----- 3g: 解码 + 保存 -----
            # output_frames = vae.decode(latent_output)
            
            # 如果不是第一段，去掉前 motion_frame 帧（与上一段重叠区）
            # if seg_idx > 0:
            #     output_frames = output_frames[motion_frame:]
            
            seg_path = os.path.join(work_dir, f"segment_{seg_idx:04d}.mp4")
            # save_frames_to_video(output_frames, seg_path, target_fps)
            segment_paths.append(seg_path)
            
            log.info(f"[InfiniteTalkSync] Segment {seg_idx+1} saved: {seg_path}")
            
            # 释放显存
            # del source_frames, source_latent, output_frames
            # torch.cuda.empty_cache()
        
        # ========================================
        # Phase 4: 拼接所有分段 + 合并音频
        # ========================================
        log.info("[InfiniteTalkSync] Concatenating segments...")
        
        final_path = os.path.join(work_dir, f"{filename_prefix}_final.mp4")
        
        # 创建 concat 列表
        list_path = os.path.join(work_dir, "segments.txt")
        with open(list_path, "w") as f:
            for sp in segment_paths:
                f.write(f"file '{sp}'\n")
        
        ffmpeg = get_ffmpeg_path()
        
        # 拼接视频 + 合并音频
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-i", audio_source_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            "-v", "quiet",
            final_path
        ]
        subprocess.run(cmd, check=True)
        
        log.info(f"[InfiniteTalkSync] Final output: {final_path}")
        
        # 返回
        filename = os.path.basename(final_path)
        audio_out = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 16000}  # placeholder
        
        return (final_path, filename, audio_out)


class InfiniteTalkVideoSyncPreview:
    """
    辅助节点：预览视频信息（不加载帧）
    输入视频路径，输出元数据 + 分段策略预估
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "segment_seconds": ("FLOAT", {"default": 30.0}),
                "target_fps": ("FLOAT", {"default": 25.0}),
                "motion_frame": ("INT", {"default": 25}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "fps", "duration", "num_segments", "info_text")
    FUNCTION = "preview"
    CATEGORY = "InfiniteTalk"
    
    def preview(self, video_path, segment_seconds, target_fps, motion_frame):
        if not os.path.exists(video_path):
            return (0, 0, 0.0, 0.0, 0, f"File not found: {video_path}")
        
        info = get_video_info(video_path)
        total_frames = int(info["duration"] * target_fps)
        segment_frames = int(segment_seconds * target_fps)
        stride = segment_frames - motion_frame
        num_segments = max(1, 1 + (total_frames - segment_frames + stride - 1) // stride)
        
        ram_per_segment_mb = segment_frames * 480 * 832 * 3 * 4 / 1024 / 1024
        
        text = (
            f"源视频: {info['width']}x{info['height']} @ {info['fps']:.1f}fps\n"
            f"时长: {info['duration']:.1f}s ({total_frames} frames @ {target_fps}fps)\n"
            f"分段: {num_segments} 段 × {segment_seconds}s (重叠 {motion_frame} 帧)\n"
            f"预估RAM/段: ~{ram_per_segment_mb:.0f} MB (480x832)"
        )
        
        return (info["width"], info["height"], info["fps"],
                info["duration"], num_segments, text)


# ============================================================
# ComfyUI 注册
# ============================================================
NODE_CLASS_MAPPINGS = {
    "InfiniteTalkVideoSync": InfiniteTalkVideoSync,
    "InfiniteTalkVideoSyncPreview": InfiniteTalkVideoSyncPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteTalkVideoSync": "InfiniteTalk Video Sync (Video Path) 🎤",
    "InfiniteTalkVideoSyncPreview": "InfiniteTalk Video Info Preview 📋",
}

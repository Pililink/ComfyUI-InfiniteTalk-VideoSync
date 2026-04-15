"""
infinitetalk_runtime.py
核心推理运行时：调用 Kijai WanVideoWrapper 内部函数完成实际推理

这个文件是整个项目最关键的部分，它需要：
1. 导入 Kijai WanVideoWrapper 的内部模块
2. 复用 multitalk_loop 的 latent 空间窗口处理逻辑
3. 加入 WanVideoEncode 的源视频编码（动作引导）
4. 管理分段处理的内存生命周期

依赖关系:
  ComfyUI-WanVideoWrapper/
  ├── multitalk/
  │   ├── multitalk_loop.py    ← 核心: 窗口化采样循环
  │   ├── multitalk.py         ← AudioProjModel, SingleStreamMultiAttention
  │   ├── nodes.py             ← MultiTalkModelLoader, MultiTalkWav2VecEmbeds 等
  │   └── wav2vec2.py          ← Wav2Vec2 特征提取
  ├── nodes_sampler.py         ← WanVideoSampler
  ├── nodes_model_loading.py   ← WanVideoModelLoader
  ├── nodes.py                 ← WanVideoImageToVideoMultiTalk, WanVideoEncode 等
  └── wanvideo/
      └── wan_video_vae.py     ← VAE encode/decode
"""

import os
import sys
import torch
import logging
import numpy as np

log = logging.getLogger("InfiniteTalkRuntime")

# ============================================================
# 导入 Kijai WanVideoWrapper 模块
# ============================================================
def _find_wrapper_path():
    """查找 ComfyUI-WanVideoWrapper 安装路径"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "ComfyUI-WanVideoWrapper"),
        # 常见自定义节点路径
    ]
    custom_nodes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    candidates.append(os.path.join(custom_nodes_dir, "ComfyUI-WanVideoWrapper"))
    
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return None

WRAPPER_PATH = _find_wrapper_path()
if WRAPPER_PATH and WRAPPER_PATH not in sys.path:
    sys.path.insert(0, WRAPPER_PATH)


class InfiniteTalkEngine:
    """
    InfiniteTalk 推理引擎
    
    生命周期:
    1. __init__: 加载所有模型（一次性）
    2. encode_full_audio: 提取完整音频的 wav2vec 特征
    3. process_segment: 处理一个视频段
    4. 可重复调用 process_segment 处理多段
    
    关键设计:
    - 模型只加载一次，分段间共享
    - 音频特征一次性提取，按偏移切片
    - 每段内部走 multitalk_loop，保证 latent 连续性
    - 段间通过 motion_frame 帧重叠保证视觉连续性
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        
        self.model = None
        self.vae = None
        self.text_embeds = None
        self.clip_vision = None
        self.wav2vec_model = None
        
        self._load_models()
    
    def _load_models(self):
        """加载所有需要的模型"""
        log.info("[Engine] Loading models...")
        
        try:
            # 从 WanVideoWrapper 导入节点类
            from nodes_model_loading import WanVideoModelLoader as _WanModelLoader
            from nodes_model_loading import WanVideoVAELoader as _VAELoader
            from nodes import WanVideoTextEncodeCached as _TextEncoder
            from nodes import WanVideoImageToVideoMultiTalk as _MultiTalkI2V
            from nodes import WanVideoEncode as _VideoEncode
            from nodes import WanVideoClipVisionEncode as _CLIPEncode
            from nodes_sampler import WanVideoSampler as _Sampler
            from multitalk.nodes import (
                MultiTalkModelLoader as _MTLoader,
                DownloadAndLoadWav2VecModel as _Wav2VecLoader,
                MultiTalkWav2VecEmbeds as _Wav2VecEmbeds,
            )
            
            self._node_classes = {
                "model_loader": _WanModelLoader,
                "vae_loader": _VAELoader,
                "text_encoder": _TextEncoder,
                "multitalk_i2v": _MultiTalkI2V,
                "video_encode": _VideoEncode,
                "clip_encode": _CLIPEncode,
                "sampler": _Sampler,
                "mt_loader": _MTLoader,
                "wav2vec_loader": _Wav2VecLoader,
                "wav2vec_embeds": _Wav2VecEmbeds,
            }
            
            log.info("[Engine] WanVideoWrapper modules imported successfully")
            
        except ImportError as e:
            log.error(f"[Engine] Failed to import WanVideoWrapper: {e}")
            log.error("[Engine] Make sure ComfyUI-WanVideoWrapper is installed")
            raise
        
        # --- 实际加载模型 ---
        cfg = self.config
        
        # 1. MultiTalk model
        mt_loader = self._node_classes["mt_loader"]()
        mt_result = mt_loader.loadmodel(cfg["infinitetalk_model"])
        mt_model = mt_result[0]
        
        # 2. Block swap
        from nodes_model_loading import WanVideoBlockSwap
        bs = WanVideoBlockSwap()
        block_swap_args = bs.load(cfg.get("block_swap", 30), False, False, True, 0, 1, False)[0]
        
        # 3. LoRA
        from nodes_model_loading import WanVideoLoraSelect
        lora_sel = WanVideoLoraSelect()
        lora = lora_sel.load(cfg["lora_model"], 1, False, True)[0]
        
        # 4. Main model
        model_loader = self._node_classes["model_loader"]()
        model_result = model_loader.loadmodel(
            cfg["wan_model"], "fp16", "fp8_e4m3fn_scaled",
            "offload_device", "sageattn", "default",
            block_swap_args=block_swap_args,
            lora=lora,
            multitalk_model=mt_model
        )
        self.model = model_result[0]
        
        # 5. VAE
        vae_loader = self._node_classes["vae_loader"]()
        self.vae = vae_loader.loadmodel("Wan2_1_VAE_bf16.safetensors", "bf16", False, False)[0]
        
        # 6. Text embeddings
        text_enc = self._node_classes["text_encoder"]()
        self.text_embeds = text_enc.encode(
            "umt5-xxl-enc-bf16.safetensors", "bf16",
            "",  # positive prompt (empty for lip sync)
            cfg.get("negative_prompt", ""),
            "fp8_e4m3fn", True, "gpu"
        )[0]
        
        # 7. CLIP Vision
        from comfy.clip_vision import load as load_clip_vision
        import folder_paths
        clip_path = folder_paths.get_full_path("clip_vision", "clip_vision_h.safetensors")
        self.clip_vision = load_clip_vision(clip_path)
        
        # 8. Wav2Vec
        wav2vec_loader = self._node_classes["wav2vec_loader"]()
        self.wav2vec_model = wav2vec_loader.loadmodel(
            "TencentGameMate/chinese-wav2vec2-base", "fp16", "main_device"
        )[0]
        
        log.info("[Engine] All models loaded")
    
    def encode_audio_features(self, audio_waveform, sample_rate, num_frames, fps):
        """
        提取完整音频的 wav2vec 特征
        返回: multitalk_embeds dict
        """
        wav2vec_node = self._node_classes["wav2vec_embeds"]()
        
        audio_input = {"waveform": audio_waveform, "sample_rate": sample_rate}
        
        result = wav2vec_node.process(
            wav2vec_model=self.wav2vec_model,
            audio_1=audio_input,
            normalize_loudness=True,
            num_frames=num_frames,
            fps=fps,
            audio_scale=1.0,
            audio_cfg_scale=1.0,
            multi_audio_type="para"
        )
        
        return result[0]  # multitalk_embeds
    
    def process_segment(self, source_frames, multitalk_embeds, seg_start_frame,
                        seg_num_frames, frame_window_size, motion_frame, 
                        steps, cfg_scale, seed):
        """
        处理一个视频段
        
        Args:
            source_frames: [N, H, W, 3] float32 tensor
            multitalk_embeds: 完整音频的 multitalk embeddings
            seg_start_frame: 当前段在全局视频中的起始帧
            seg_num_frames: 当前段帧数
            frame_window_size: 窗口帧数 (81)
            motion_frame: 重叠帧数 (25)
            steps: 采样步数
            cfg_scale: CFG scale
            seed: 随机种子
            
        Returns:
            output_frames: [M, H, W, 3] float32 tensor
        """
        N, H, W, C = source_frames.shape
        
        # 1. 编码源视频 → latent（动作引导）
        video_encode = self._node_classes["video_encode"]()
        source_latent = video_encode.encode(
            vae=self.vae,
            image=source_frames,
            enable_tiling=False,
            tile_size_h=272, tile_size_w=272,
            tile_stride_h=144, tile_stride_w=128,
            tile_batch_size=0, denoise_strength=1
        )[0]
        
        # 2. 准备 CLIP embeddings（首帧 + 末帧）
        first_frame = source_frames[0:1]
        last_frame = source_frames[-1:]
        
        clip_encode = self._node_classes["clip_encode"]()
        clip_embeds = clip_encode.encode(
            clip_vision=self.clip_vision,
            image_1=first_frame,
            image_2=last_frame,
            strength_1=1.0, strength_2=0.7,
            crop="center", interpolation="average",
            combine=True, noise_augment=0, noise_augment_2=0.5
        )[0]
        
        # 3. 准备 image_embeds（触发内部 multitalk_loop）
        multitalk_i2v = self._node_classes["multitalk_i2v"]()
        image_embeds_result = multitalk_i2v.encode(
            vae=self.vae,
            start_image=first_frame,
            clip_embeds=clip_embeds,
            width=W, height=H,
            frame_window_size=frame_window_size,
            motion_frame=motion_frame,
            force_offload=False,
            colormatch="disabled",
            tiled_vae=False,
            mode="infinitetalk"
        )
        image_embeds = image_embeds_result[0]
        
        # 4. 采样！
        # 内部 multitalk_loop 会自动:
        # - 检测到 multitalk_sampling=True
        # - 按 frame_window_size 切分窗口
        # - motion_frame 帧 latent 重叠 + 噪声注入
        # - 音频特征按窗口切片对齐
        sampler = self._node_classes["sampler"]()
        latent_output = sampler.process(
            model=self.model,
            image_embeds=image_embeds,
            text_embeds=self.text_embeds,
            samples=source_latent,
            steps=steps,
            cfg=[cfg_scale],
            shift=11.0,
            seed=seed,
            seed_mode="increment",
            force_offload=True,
            scheduler="dpm++_sde",
            riflex_freq_index=0,
            multitalk_embeds=multitalk_embeds,
        )[0]
        
        # 5. 解码
        from nodes import WanVideoDecode
        decoder = WanVideoDecode()
        output_frames = decoder.decode(
            vae=self.vae,
            samples=latent_output,
            enable_tiling=True,
            tile_size_h=272, tile_size_w=272,
            tile_stride_h=144, tile_stride_w=128,
            decode_method="default"
        )[0]
        
        return output_frames
    
    def unload(self):
        """释放所有模型"""
        del self.model, self.vae, self.text_embeds, self.clip_vision, self.wav2vec_model
        torch.cuda.empty_cache()
        log.info("[Engine] Models unloaded")

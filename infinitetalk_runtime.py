import importlib
import inspect
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import torch

import folder_paths
import comfy.model_management as mm

log = logging.getLogger("InfiniteTalkRuntime")

_WRAPPER_NAMESPACE = "_infinitetalk_wanvideo_wrapper"


def _format_log_fields(**fields):
    parts = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def _timed_log(stage, **fields):
    details = _format_log_fields(**fields)
    suffix = f" | {details}" if details else ""
    log.info("[InfiniteTalk] %s | start%s", stage, suffix)
    started_at = time.perf_counter()
    success = False
    try:
        yield
        success = True
    finally:
        elapsed = time.perf_counter() - started_at
        status = "done" if success else "failed"
        log.info("[InfiniteTalk] %s | %s in %.2fs%s", stage, status, elapsed, suffix)


def _dedupe_paths(paths):
    seen = set()
    ordered = []
    for path in paths:
        if path is None:
            continue
        try:
            resolved = Path(path).resolve()
        except Exception:
            resolved = Path(path)
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)
    return ordered


def _candidate_custom_nodes_dirs():
    candidates = []

    for env_name in (
        "INFINITETALK_CUSTOM_NODES_DIR",
        "COMFYUI_CUSTOM_NODES_DIR",
        "CUSTOM_NODES_DIR",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))

    current_file = Path(__file__).resolve()
    for parent in [current_file.parent] + list(current_file.parents):
        candidates.append(parent)
        candidates.append(parent / "custom_nodes")
        candidates.append(parent / "ComfyUI" / "custom_nodes")

    try:
        models_dir = Path(folder_paths.models_dir).resolve()
        models_parent = models_dir.parent
        candidates.append(models_parent / "custom_nodes")
        for child in models_parent.iterdir():
            if child.is_dir() and "custom_nodes" in child.name.lower():
                candidates.append(child)
    except Exception:
        pass

    return _dedupe_paths(candidates)


def _find_custom_node_dir(node_dir_name):
    wanted = node_dir_name.lower()
    for base_dir in _candidate_custom_nodes_dirs():
        if base_dir.is_dir() and base_dir.name.lower() == wanted:
            return base_dir
        candidate = base_dir / node_dir_name
        if candidate.is_dir():
            return candidate
    return None


def _embedded_wrapper_root():
    return Path(__file__).resolve().parent / "vendor" / "wanvideo_wrapper"


def _ensure_namespace_package(package_name, package_root):
    package_root = Path(package_root).resolve()
    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_path = getattr(existing, "__path__", None)
        if existing_path is not None and str(package_root) not in existing_path:
            existing_path.append(str(package_root))
        return

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]
    package.__package__ = package_name
    package.__file__ = str(package_root / "__init__.py")
    sys.modules[package_name] = package


def _import_wrapper_module(submodule):
    wrapper_root = _embedded_wrapper_root()
    if not wrapper_root.is_dir():
        raise FileNotFoundError(
            f"Embedded WanVideo runtime not found: {wrapper_root}"
        )

    _ensure_namespace_package(_WRAPPER_NAMESPACE, wrapper_root)
    return importlib.import_module(f"{_WRAPPER_NAMESPACE}.{submodule}")


def _filter_call_kwargs(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def _invoke(instance, method_names, **kwargs):
    last_error = None
    for method_name in method_names:
        method = getattr(instance, method_name, None)
        if not callable(method):
            continue
        try:
            return method(**_filter_call_kwargs(method, kwargs))
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise AttributeError(
        f"{instance.__class__.__name__} does not provide any of {method_names!r}"
    )


def _module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def resolve_attention_mode(attention_mode, module_available=None):
    attention_mode = str(attention_mode or "comfy").strip() or "comfy"
    module_available = module_available or _module_available
    sage_requirements = {
        "sageattn": ("sageattention",),
        "sageattn_compiled": ("sageattention",),
        "sageattn_varlen": ("sageattention",),
        "sageattn_3": ("sageattn3", "sageattention"),
    }
    required_modules = sage_requirements.get(attention_mode)
    if required_modules and not any(module_available(name) for name in required_modules):
        log.warning(
            "[InfiniteTalk] attention_mode=%s requested but required package is not available; fallback to comfy",
            attention_mode,
        )
        return "comfy"
    return attention_mode


def _sequence_length(sequence):
    shape = getattr(sequence, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(sequence)


def _slice_sequence(sequence, start_frame, end_frame):
    sliced = sequence[start_frame:end_frame]
    if hasattr(sliced, "contiguous"):
        return sliced.contiguous()
    return sliced


def slice_multitalk_embeds(multitalk_embeds, start_frame, num_frames):
    start_frame = max(0, int(start_frame))
    requested_frames = max(0, int(num_frames))
    audio_features = list(multitalk_embeds.get("audio_features") or [])
    if not audio_features:
        raise RuntimeError("No global MultiTalk audio features are available")

    available_frames = [
        max(0, _sequence_length(feature) - start_frame)
        for feature in audio_features
    ]
    actual_frames = min([requested_frames] + available_frames)
    if actual_frames <= 0:
        raise RuntimeError(
            f"No MultiTalk audio features available at frame {start_frame}"
        )

    end_frame = start_frame + actual_frames
    sliced_embeds = dict(multitalk_embeds)
    sliced_embeds["audio_features"] = [
        _slice_sequence(feature, start_frame, end_frame)
        for feature in audio_features
    ]
    return sliced_embeds, actual_frames


def select_clip_reference_frames(source_frames, use_last_frame=False):
    first_frame = source_frames[0:1]
    second_frame = source_frames[-1:] if bool(use_last_frame) else None
    return first_frame, second_frame


def _get_clip_vision_path(model_name):
    clip_path = None
    if hasattr(folder_paths, "get_full_path"):
        clip_path = folder_paths.get_full_path("clip_vision", model_name)
    if not clip_path and hasattr(folder_paths, "get_full_path_or_raise"):
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", model_name)
    if not clip_path:
        raise FileNotFoundError(f"clip_vision model not found: {model_name}")
    return clip_path


class InfiniteTalkEngine:
    def __init__(self, config):
        self.config = dict(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.vae = None
        self.text_embeds = None
        self.clip_vision = None
        self.wav2vec_model = None

        self._node_instances = {}
        self._load_models()

    def _load_models(self):
        log.info("[InfiniteTalk] Loading embedded WanVideo runtime")

        with _timed_log("Import wrapper module", module="nodes"):
            wrapper_nodes = _import_wrapper_module("nodes")
        with _timed_log("Import wrapper module", module="nodes_sampler"):
            wrapper_sampler = _import_wrapper_module("nodes_sampler")
        with _timed_log("Import wrapper module", module="nodes_model_loading"):
            wrapper_loading = _import_wrapper_module("nodes_model_loading")
        with _timed_log("Import wrapper module", module="multitalk.nodes"):
            multitalk_nodes = _import_wrapper_module("multitalk.nodes")
        with _timed_log("Import wrapper module", module="fantasytalking.nodes"):
            fantasytalking_nodes = _import_wrapper_module("fantasytalking.nodes")

        from comfy.clip_vision import load as load_clip_vision

        node_classes = {
            "model_loader": wrapper_loading.WanVideoModelLoader,
            "vae_loader": wrapper_loading.WanVideoVAELoader,
            "block_swap": wrapper_loading.WanVideoBlockSwap,
            "lora_select": wrapper_loading.WanVideoLoraSelect,
            "text_encode_cached": wrapper_nodes.WanVideoTextEncodeCached,
            "video_encode": wrapper_nodes.WanVideoEncode,
            "clip_encode": wrapper_nodes.WanVideoClipVisionEncode,
            "decode": wrapper_nodes.WanVideoDecode,
            "sampler": wrapper_sampler.WanVideoSampler,
            "multitalk_model_loader": multitalk_nodes.MultiTalkModelLoader,
            "multitalk_wav2vec_embeds": multitalk_nodes.MultiTalkWav2VecEmbeds,
            "multitalk_i2v": multitalk_nodes.WanVideoImageToVideoMultiTalk,
            "wav2vec_loader": fantasytalking_nodes.DownloadAndLoadWav2VecModel,
        }

        with _timed_log("Instantiate wrapper nodes", count=len(node_classes)):
            self._node_instances = {
                key: cls() for key, cls in node_classes.items()
            }

        with _timed_log("Load MultiTalk model", model=self.config["infinitetalk_model"]):
            multitalk_model = _invoke(
                self._node_instances["multitalk_model_loader"],
                ("loadmodel",),
                model=self.config["infinitetalk_model"],
            )[0]

        with _timed_log(
            "Configure block swap",
            blocks_to_swap=int(self.config.get("block_swap", 30)),
        ):
            block_swap_args = _invoke(
                self._node_instances["block_swap"],
                ("setargs",),
                blocks_to_swap=int(self.config.get("block_swap", 30)),
                offload_img_emb=False,
                offload_txt_emb=False,
                use_non_blocking=bool(self.config.get("use_non_blocking", True)),
                vace_blocks_to_swap=0,
                prefetch_blocks=int(self.config.get("prefetch_blocks", 1)),
                block_swap_debug=False,
            )[0]

        lora = None
        lora_name = str(self.config.get("lora_model", "") or "").strip()
        if lora_name and lora_name.lower() != "none":
            with _timed_log(
                "Resolve LoRA",
                lora=lora_name,
                strength=float(self.config.get("lora_strength", 1.0)),
            ):
                lora = _invoke(
                    self._node_instances["lora_select"],
                    ("getlorapath",),
                    lora=lora_name,
                    strength=float(self.config.get("lora_strength", 1.0)),
                    unique_id=None,
                    blocks={},
                    prev_lora=None,
                    low_mem_load=False,
                    merge_loras=True,
                )[0]
        else:
            log.info("[InfiniteTalk] Resolve LoRA | skipped")

        attention_mode = resolve_attention_mode(self.config.get("attention_mode", "sageattn"))
        with _timed_log(
            "Load diffusion model",
            model=self.config["wan_model"],
            attention_mode=attention_mode,
        ):
            self.model = _invoke(
                self._node_instances["model_loader"],
                ("loadmodel",),
                model=self.config["wan_model"],
                base_precision=self.config.get("base_precision", "fp16"),
                quantization=self.config.get("quantization", "fp8_e4m3fn_scaled"),
                load_device="offload_device",
                attention_mode=attention_mode,
                block_swap_args=block_swap_args,
                lora=lora,
                multitalk_model=multitalk_model,
                rms_norm_function="default",
            )[0]

        with _timed_log("Load VAE", model=self.config["vae_model"]):
            self.vae = _invoke(
                self._node_instances["vae_loader"],
                ("loadmodel",),
                model_name=self.config["vae_model"],
                precision=self.config.get("vae_precision", "bf16"),
                compile_args=None,
                use_cpu_cache=False,
                verbose=False,
            )[0]

        with _timed_log("Encode text prompts", model=self.config["text_encoder_model"]):
            self.text_embeds = _invoke(
                self._node_instances["text_encode_cached"],
                ("process",),
                model_name=self.config["text_encoder_model"],
                precision=self.config.get("text_precision", "bf16"),
                positive_prompt=self.config.get("positive_prompt", ""),
                negative_prompt=self.config.get("negative_prompt", ""),
                quantization=self.config.get("text_quantization", "disabled"),
                use_disk_cache=True,
                device=self.config.get("text_device", "gpu"),
            )[0]

        clip_path = _get_clip_vision_path(self.config["clip_vision_model"])
        with _timed_log("Load CLIP vision", model=self.config["clip_vision_model"]):
            self.clip_vision = load_clip_vision(clip_path)

        with _timed_log("Load wav2vec", model=self.config.get("wav2vec_model", "TencentGameMate/chinese-wav2vec2-base")):
            self.wav2vec_model = _invoke(
                self._node_instances["wav2vec_loader"],
                ("loadmodel",),
                model=self.config.get("wav2vec_model", "TencentGameMate/chinese-wav2vec2-base"),
                base_precision=self.config.get("wav2vec_precision", "fp16"),
                load_device=self.config.get("wav2vec_load_device", "main_device"),
            )[0]

        log.info("[InfiniteTalk] Models loaded")

    def _build_multitalk_embeds(self, audio_input, num_frames, fps):
        common_kwargs = {
            "wav2vec_model": self.wav2vec_model,
            "audio_1": audio_input,
            "normalize_loudness": bool(self.config.get("normalize_loudness", True)),
            "num_frames": int(num_frames),
            "fps": float(fps),
            "audio_scale": float(self.config.get("audio_scale", 1.0)),
            "audio_cfg_scale": float(self.config.get("audio_cfg_scale", 1.0)),
            "multi_audio_type": "para",
            "add_noise_floor": False,
            "smooth_transients": False,
        }

        try:
            return _invoke(
                self._node_instances["multitalk_wav2vec_embeds"],
                ("process",),
                **common_kwargs,
            )
        except ImportError as exc:
            if "pyloudnorm" not in str(exc).lower():
                raise
            log.warning("[InfiniteTalk] pyloudnorm is missing, retrying without loudness normalization")
            common_kwargs["normalize_loudness"] = False
            return _invoke(
                self._node_instances["multitalk_wav2vec_embeds"],
                ("process",),
                **common_kwargs,
            )

    def encode_source_latent(self, source_frames, label=""):
        """VAE-encode a chunk of source pixel frames and return a CPU latent.

        The wrapper's `WanVideoEncode.encode` already moves the result to CPU,
        so this is just a thin typed wrapper that we can call repeatedly to
        build up a long video's full latent without holding all the pixel
        frames or the full latent on the GPU at once.
        """
        with _timed_log(
            "Source VAE encode",
            chunk=label,
            frames=int(source_frames.shape[0]),
            size=f"{int(source_frames.shape[2])}x{int(source_frames.shape[1])}",
        ):
            encode_result = _invoke(
                self._node_instances["video_encode"],
                ("encode",),
                vae=self.vae,
                image=source_frames,
                enable_vae_tiling=bool(self.config.get("encode_tiled_vae", False)),
                tile_x=int(self.config.get("tile_x", 272)),
                tile_y=int(self.config.get("tile_y", 272)),
                tile_stride_x=int(self.config.get("tile_stride_x", 144)),
                tile_stride_y=int(self.config.get("tile_stride_y", 128)),
                noise_aug_strength=0.0,
                latent_strength=1.0,
                mask=None,
            )[0]
        latents = encode_result["samples"]
        # Make sure we are not holding GPU memory between chunks.
        if hasattr(latents, "device") and latents.device.type != "cpu":
            latents = latents.cpu()
        return latents

    def encode_clip_vision_for_first_frame(self, source_frames):
        first_frame, second_frame = select_clip_reference_frames(
            source_frames,
            self.config.get("clip_use_last_frame", False),
        )
        clip_embeds = _invoke(
            self._node_instances["clip_encode"],
            ("process",),
            clip_vision=self.clip_vision,
            image_1=first_frame,
            image_2=second_frame,
            strength_1=float(self.config.get("clip_strength_1", 1.0)),
            strength_2=float(self.config.get("clip_strength_2", 1.0)),
            crop="center",
            combine_embeds="average",
            force_offload=True,
            tiles=0,
            ratio=0.5,
            negative_image=None,
        )[0]
        return clip_embeds, first_frame

    def render_full_video(
        self,
        first_frame,
        clip_embeds,
        source_latent,
        multitalk_embeds,
        actual_num_frames,
        fps,
        frame_window_size,
        motion_frame,
        steps,
        cfg_scale,
        seed,
        output_dir,
        start_step,
        target_width,
        target_height,
    ):
        """Run the multitalk_loop end-to-end against a full-length latent.

        The wrapper's multitalk_loop natively walks the full audio embedding
        with a sliding `frame_window_size` window, injecting the previous
        window's last `motion_frame` latent frames into the next iteration.
        Giving it `output_path` causes each window's decoded frames to land
        on disk as PNG, so we never hold the whole rendered video in VRAM.
        """
        log.info(
            "[InfiniteTalk] Full render start frames=%s fps=%.3f size=%sx%s output_dir=%s",
            int(actual_num_frames),
            float(fps),
            int(target_width),
            int(target_height),
            output_dir or "<memory>",
        )

        with _timed_log(
            "Build image embeds",
            frame_window_size=int(frame_window_size),
            motion_frame=int(motion_frame),
            output_dir=output_dir,
        ):
            image_embeds_result = _invoke(
                self._node_instances["multitalk_i2v"],
                ("process",),
                vae=self.vae,
                width=int(target_width),
                height=int(target_height),
                frame_window_size=int(frame_window_size),
                motion_frame=int(motion_frame),
                force_offload=bool(self.config.get("image_embeds_force_offload", False)),
                colormatch="disabled",
                start_image=first_frame,
                tiled_vae=bool(self.config.get("encode_tiled_vae", False)),
                clip_embeds=clip_embeds,
                mode="infinitetalk",
                output_path=str(output_dir or ""),
            )

        image_embeds = image_embeds_result[0]
        actual_output_dir = ""
        if len(image_embeds_result) > 1 and isinstance(image_embeds_result[1], str):
            actual_output_dir = image_embeds_result[1]
        if not actual_output_dir and isinstance(image_embeds, dict):
            actual_output_dir = str(image_embeds.get("output_path", "") or "")
        if not actual_output_dir:
            actual_output_dir = str(output_dir or "")

        samples_payload = {"samples": source_latent, "noise_mask": None}

        with _timed_log(
            "Full sampler",
            frames=int(actual_num_frames),
            steps=int(steps),
            cfg=float(cfg_scale),
            seed=int(seed),
            scheduler=self.config.get("scheduler", "dpm++_sde"),
            start_step=int(start_step),
        ):
            _invoke(
                self._node_instances["sampler"],
                ("process",),
                model=self.model,
                image_embeds=image_embeds,
                text_embeds=self.text_embeds,
                samples=samples_payload,
                steps=int(steps),
                cfg=float(cfg_scale),
                shift=float(self.config.get("shift", 11.0)),
                seed=int(seed),
                force_offload=bool(self.config.get("sampler_force_offload", True)),
                scheduler=self.config.get("scheduler", "dpm++_sde"),
                riflex_freq_index=int(self.config.get("riflex_freq_index", 0)),
                multitalk_embeds=multitalk_embeds,
                denoise_strength=float(self.config.get("denoise_strength", 1.0)),
                batched_cfg=bool(self.config.get("batched_cfg", False)),
                rope_function=self.config.get("rope_function", "comfy"),
                start_step=int(start_step),
                end_step=int(self.config.get("end_step", -1)),
                add_noise_to_samples=bool(self.config.get("add_noise_to_samples", True)),
            )

        log.info(
            "[InfiniteTalk] Full render done output_dir=%s",
            actual_output_dir,
        )
        return {
            "actual_num_frames": int(actual_num_frames),
            "output_path": actual_output_dir,
        }

    def render_segment(
        self,
        source_frames,
        audio_input,
        fps,
        frame_window_size,
        motion_frame,
        steps,
        cfg_scale,
        seed,
        segment_output_dir="",
        start_step=3,
        segment_label="",
        multitalk_embeds=None,
        actual_num_frames=None,
    ):
        frame_count = int(source_frames.shape[0])
        frame_size = f"{int(source_frames.shape[2])}x{int(source_frames.shape[1])}"
        log.info(
            "[InfiniteTalk] Segment %s runtime start input_frames=%s size=%s fps=%.3f output_dir=%s",
            segment_label or "?",
            frame_count,
            frame_size,
            float(fps),
            segment_output_dir or "<memory>",
        )

        if multitalk_embeds is None:
            with _timed_log(
                "Segment wav2vec",
                segment=segment_label,
                input_frames=frame_count,
                fps=f"{float(fps):.3f}",
            ):
                multitalk_embeds, _, actual_num_frames = self._build_multitalk_embeds(
                    audio_input=audio_input,
                    num_frames=frame_count,
                    fps=fps,
                )
        else:
            actual_num_frames = actual_num_frames or frame_count

        actual_num_frames = max(1, int(actual_num_frames))
        if actual_num_frames < source_frames.shape[0]:
            source_frames = source_frames[:actual_num_frames].contiguous()
        log.info(
            "[InfiniteTalk] Segment %s wav2vec ready actual_num_frames=%s",
            segment_label or "?",
            actual_num_frames,
        )

        with _timed_log(
            "Segment video encode",
            segment=segment_label,
            frames=int(source_frames.shape[0]),
            size=f"{int(source_frames.shape[2])}x{int(source_frames.shape[1])}",
        ):
            source_latent = _invoke(
                self._node_instances["video_encode"],
                ("encode",),
                vae=self.vae,
                image=source_frames,
                enable_vae_tiling=bool(self.config.get("encode_tiled_vae", False)),
                tile_x=int(self.config.get("tile_x", 272)),
                tile_y=int(self.config.get("tile_y", 272)),
                tile_stride_x=int(self.config.get("tile_stride_x", 144)),
                tile_stride_y=int(self.config.get("tile_stride_y", 128)),
                noise_aug_strength=0.0,
                latent_strength=1.0,
                mask=None,
            )[0]

        first_frame, second_frame = select_clip_reference_frames(
            source_frames,
            self.config.get("clip_use_last_frame", False),
        )

        with _timed_log(
            "Segment clip encode",
            segment=segment_label,
            use_last_frame=bool(self.config.get("clip_use_last_frame", False)),
        ):
            clip_embeds = _invoke(
                self._node_instances["clip_encode"],
                ("process",),
                clip_vision=self.clip_vision,
                image_1=first_frame,
                image_2=second_frame,
                strength_1=float(self.config.get("clip_strength_1", 1.0)),
                strength_2=float(self.config.get("clip_strength_2", 1.0)),
                crop="center",
                combine_embeds="average",
                force_offload=True,
                tiles=0,
                ratio=0.5,
                negative_image=None,
            )[0]

        with _timed_log(
            "Segment image embeds",
            segment=segment_label,
            frame_window_size=int(frame_window_size),
            motion_frame=int(motion_frame),
            output_dir=segment_output_dir,
        ):
            image_embeds_result = _invoke(
                self._node_instances["multitalk_i2v"],
                ("process",),
                vae=self.vae,
                width=int(source_frames.shape[2]),
                height=int(source_frames.shape[1]),
                frame_window_size=int(frame_window_size),
                motion_frame=int(motion_frame),
                force_offload=bool(self.config.get("image_embeds_force_offload", False)),
                colormatch="disabled",
                start_image=first_frame,
                tiled_vae=bool(self.config.get("encode_tiled_vae", False)),
                clip_embeds=clip_embeds,
                mode="infinitetalk",
                output_path=str(segment_output_dir or ""),
            )

        image_embeds = image_embeds_result[0]
        actual_output_dir = ""
        if len(image_embeds_result) > 1 and isinstance(image_embeds_result[1], str):
            actual_output_dir = image_embeds_result[1]
        if not actual_output_dir and isinstance(image_embeds, dict):
            actual_output_dir = str(image_embeds.get("output_path", "") or "")
        if not actual_output_dir:
            actual_output_dir = str(segment_output_dir or "")
        log.info(
            "[InfiniteTalk] Segment %s image embeds resolved output_dir=%s",
            segment_label or "?",
            actual_output_dir or "<memory>",
        )

        with _timed_log(
            "Segment sampler",
            segment=segment_label,
            steps=int(steps),
            cfg=float(cfg_scale),
            seed=int(seed),
            start_step=int(start_step),
            scheduler=self.config.get("scheduler", "flowmatch_distill"),
        ):
            latent_output = _invoke(
                self._node_instances["sampler"],
                ("process",),
                model=self.model,
                image_embeds=image_embeds,
                text_embeds=self.text_embeds,
                samples=source_latent,
                steps=int(steps),
                cfg=float(cfg_scale),
                shift=float(self.config.get("shift", 11.0)),
                seed=int(seed),
                force_offload=bool(self.config.get("sampler_force_offload", True)),
                scheduler=self.config.get("scheduler", "flowmatch_distill"),
                riflex_freq_index=int(self.config.get("riflex_freq_index", 0)),
                multitalk_embeds=multitalk_embeds,
                denoise_strength=float(self.config.get("denoise_strength", 1.0)),
                batched_cfg=bool(self.config.get("batched_cfg", False)),
                rope_function=self.config.get("rope_function", "comfy"),
                start_step=int(start_step),
                end_step=int(self.config.get("end_step", -1)),
                add_noise_to_samples=bool(self.config.get("add_noise_to_samples", True)),
            )[0]

        result = {
            "actual_num_frames": actual_num_frames,
            "output_path": actual_output_dir,
        }

        if segment_output_dir:
            log.info(
                "[InfiniteTalk] Segment %s runtime done actual_num_frames=%s output_dir=%s",
                segment_label or "?",
                actual_num_frames,
                actual_output_dir,
            )
            return result

        with _timed_log("Segment decode", segment=segment_label):
            output_frames = _invoke(
                self._node_instances["decode"],
                ("decode",),
                vae=self.vae,
                samples=latent_output,
                enable_vae_tiling=bool(self.config.get("decode_tiled_vae", False)),
                tile_x=int(self.config.get("tile_x", 272)),
                tile_y=int(self.config.get("tile_y", 272)),
                tile_stride_x=int(self.config.get("tile_stride_x", 144)),
                tile_stride_y=int(self.config.get("tile_stride_y", 128)),
                normalization="default",
            )[0]

        result["frames"] = output_frames
        log.info(
            "[InfiniteTalk] Segment %s runtime done actual_num_frames=%s decoded_shape=%s",
            segment_label or "?",
            actual_num_frames,
            tuple(output_frames.shape),
        )
        return result

    def unload(self):
        for attribute in ("model", "vae", "text_embeds", "clip_vision", "wav2vec_model"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)
        self._node_instances = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mm.soft_empty_cache()
        log.info("[InfiniteTalk] Runtime released")

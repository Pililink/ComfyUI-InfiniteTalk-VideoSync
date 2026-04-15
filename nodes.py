import importlib
import logging
import math
import os
import random
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import folder_paths
import comfy.model_management as mm

from .infinitetalk_runtime import (
    InfiniteTalkEngine,
    _candidate_custom_nodes_dirs,
    _ensure_namespace_package,
)

log = logging.getLogger("InfiniteTalkVideoSync")

_AUDIO_SEPARATION_NAMESPACE = "_infinitetalk_audio_separation"

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, walking backwards"
)


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


def _throw_if_interrupted():
    if hasattr(mm, "throw_exception_if_processing_interrupted"):
        mm.throw_exception_if_processing_interrupted()


def _run_command(command, error_message, text=False):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )

    result = {}
    failure = {}

    def _communicate_worker():
        try:
            stdout, stderr = process.communicate()
            result["stdout"] = stdout
            result["stderr"] = stderr
        except Exception as exc:
            failure["error"] = exc

    worker = threading.Thread(target=_communicate_worker, daemon=True)
    worker.start()

    try:
        while worker.is_alive():
            _throw_if_interrupted()
            worker.join(0.1)
    finally:
        if worker.is_alive():
            process.kill()
            worker.join()

    if "error" in failure:
        raise failure["error"]

    stdout = result.get("stdout", "" if text else b"")
    stderr = result.get("stderr", "" if text else b"")

    if process.returncode != 0:
        details = (stderr or stdout or "").strip()
        if isinstance(details, bytes):
            details = details.decode("utf-8", errors="ignore").strip()
        if details:
            raise RuntimeError(f"{error_message}: {details}")
        raise RuntimeError(f"{error_message} (exit code {process.returncode})")
    return stdout


def get_ffmpeg_path():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def get_ffprobe_path():
    ffmpeg = Path(get_ffmpeg_path())
    if ffmpeg.name.lower().startswith("ffmpeg"):
        candidate = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffprobe") or "ffprobe"


def resolve_user_path(path_value, input_name):
    cleaned = str(path_value or "").strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError(f"InfiniteTalk: {input_name} cannot be empty")
    resolved = os.path.abspath(os.path.expanduser(os.path.expandvars(cleaned)))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"InfiniteTalk: {input_name} not found: {resolved}")
    return resolved


def normalize_frame_window_size(value):
    value = max(5, int(value))
    return ((value - 1) // 4) * 4 + 1


def _safe_fps_value(fps_str):
    if not fps_str:
        return 25.0
    if "/" in fps_str:
        numerator, denominator = fps_str.split("/", 1)
        denominator = float(denominator or 1)
        if denominator == 0:
            return 25.0
        return float(numerator) / denominator
    return float(fps_str)


def _probe_media(media_path):
    ffprobe = get_ffprobe_path()
    output = _run_command(
        [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            media_path,
        ],
        f"Failed to probe media: {media_path}",
        text=True,
    )
    import json

    return json.loads(output)


def get_video_info(video_path):
    info = _probe_media(video_path)
    video_stream = next(
        (stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": _safe_fps_value(video_stream.get("r_frame_rate", "25/1")),
        "duration": float(info.get("format", {}).get("duration", 0.0)),
        "nb_frames": int(video_stream.get("nb_frames") or 0),
    }


def probe_media_duration(media_path, label):
    info = _probe_media(media_path)
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise RuntimeError(f"Unable to determine duration for {label}: {media_path}")
    return duration


def _align_down_to_multiple(value, multiple):
    value = int(value)
    multiple = max(1, int(multiple))
    if value <= 0:
        return multiple
    return max(multiple, (value // multiple) * multiple)


def _resolve_auto_target_size(source_width, source_height, max_width=832, max_height=480):
    source_width = int(source_width)
    source_height = int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source size: {source_width}x{source_height}")

    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    scale = min(1.0, max_width / source_width, max_height / source_height)

    target_width = _align_down_to_multiple(source_width * scale, 16)
    target_height = _align_down_to_multiple(source_height * scale, 16)
    return target_width, target_height


def _coerce_int_value(
    value,
    field_name,
    default,
    min_value=None,
    max_value=None,
    allow_randomize=False,
):
    if value is None:
        parsed = default
    elif isinstance(value, bool):
        parsed = int(value)
    else:
        text = str(value).strip()
        if text == "":
            parsed = default
        elif allow_randomize and text.lower() == "randomize":
            parsed = random.randint(0, 2**31 - 1)
        else:
            try:
                parsed = int(float(text))
            except (TypeError, ValueError):
                log.warning(
                    "[InfiniteTalk] Invalid integer input for %s: %r, fallback to %s",
                    field_name,
                    value,
                    default,
                )
                parsed = default

    if min_value is not None and parsed < min_value:
        log.warning(
            "[InfiniteTalk] %s=%s is below min=%s, clamped to min",
            field_name,
            parsed,
            min_value,
        )
        parsed = min_value
    if max_value is not None and parsed > max_value:
        log.warning(
            "[InfiniteTalk] %s=%s is above max=%s, clamped to max",
            field_name,
            parsed,
            max_value,
        )
        parsed = max_value
    return parsed


def _coerce_float_value(value, field_name, default, min_value=None, max_value=None):
    if value is None:
        parsed = default
    else:
        if isinstance(value, bool):
            parsed = float(value)
        else:
            text = str(value).strip()
            if text == "":
                parsed = default
            else:
                try:
                    parsed = float(text)
                except (TypeError, ValueError):
                    log.warning(
                        "[InfiniteTalk] Invalid float input for %s: %r, fallback to %s",
                        field_name,
                        value,
                        default,
                    )
                    parsed = default

    if min_value is not None and parsed < min_value:
        log.warning(
            "[InfiniteTalk] %s=%s is below min=%s, clamped to min",
            field_name,
            parsed,
            min_value,
        )
        parsed = min_value
    if max_value is not None and parsed > max_value:
        log.warning(
            "[InfiniteTalk] %s=%s is above max=%s, clamped to max",
            field_name,
            parsed,
            max_value,
        )
        parsed = max_value
    return parsed


def load_video_chunk(video_path, start_frame, num_frames, target_width, target_height, target_fps):
    ffmpeg = get_ffmpeg_path()
    start_time = max(0.0, float(start_frame) / float(target_fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        video_path,
        "-vf",
        f"fps={target_fps},scale={target_width}:{target_height}:flags=lanczos",
        "-frames:v",
        str(int(num_frames)),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    raw = _run_command(command, f"Failed to load video frames from {video_path}", text=False)
    if not raw:
        raise RuntimeError(f"ffmpeg returned no frames for {video_path} @ frame {start_frame}")

    frame_size = int(target_width) * int(target_height) * 3
    actual_frames = len(raw) // frame_size
    if actual_frames <= 0:
        raise RuntimeError(f"Decoded frame buffer is empty for {video_path}")

    frames = np.frombuffer(raw[: actual_frames * frame_size], dtype=np.uint8)
    frames = frames.reshape(actual_frames, int(target_height), int(target_width), 3)
    return torch.tensor(frames, dtype=torch.float32).div_(255.0)


def load_audio_segment(audio_path, start_sec=0.0, duration_sec=None, target_sr=16000):
    ffmpeg = get_ffmpeg_path()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start_sec and start_sec > 0:
        command.extend(["-ss", f"{float(start_sec):.6f}"])
    command.extend(["-i", audio_path])
    if duration_sec is not None:
        command.extend(["-t", f"{max(0.0, float(duration_sec)):.6f}"])
    command.extend(
        [
            "-vn",
            "-ar",
            str(int(target_sr)),
            "-ac",
            "1",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    raw = _run_command(command, f"Failed to load audio from {audio_path}", text=False)
    if not raw:
        if duration_sec is None:
            return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32), "sample_rate": int(target_sr)}
        sample_count = int(max(0.0, float(duration_sec)) * int(target_sr))
        return {
            "waveform": torch.zeros((1, 1, sample_count), dtype=torch.float32),
            "sample_rate": int(target_sr),
        }
    audio = np.frombuffer(raw, dtype=np.float32)
    return {
        "waveform": torch.from_numpy(audio.copy()).unsqueeze(0).unsqueeze(0),
        "sample_rate": int(target_sr),
    }


def save_audio_input(audio, target_audio_path):
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError("torchaudio is required to save AUDIO inputs for InfiniteTalk") from exc

    waveform = audio["waveform"].detach().clone().cpu()
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    with torch.inference_mode(False):
        torchaudio.save(target_audio_path, waveform, sample_rate)
    return target_audio_path


def load_audio_file(audio_path):
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError("torchaudio is required to load audio files for InfiniteTalk") from exc

    waveform, sample_rate = torchaudio.load(audio_path)
    return {
        "waveform": waveform.unsqueeze(0),
        "sample_rate": int(sample_rate),
    }


def extract_audio_to_wav(media_path, target_audio_path):
    ffmpeg = get_ffmpeg_path()
    _run_command(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            media_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            target_audio_path,
        ],
        f"Failed to extract audio from {media_path}",
        text=False,
    )
    return target_audio_path


def _find_audio_separation_root():
    for custom_nodes_dir in _candidate_custom_nodes_dirs():
        candidate = custom_nodes_dir / "audio-separation-nodes-comfyui"
        if candidate.is_dir():
            return candidate
    return None


def run_audio_separation(audio, fade_shape="linear", chunk_length=10.0, chunk_overlap=0.1):
    root = _find_audio_separation_root()
    if root is None:
        raise RuntimeError(
            "audio-separation-nodes-comfyui not found. Disable separate_vocals or install that node."
        )

    _ensure_namespace_package(_AUDIO_SEPARATION_NAMESPACE, root)
    module = importlib.import_module(f"{_AUDIO_SEPARATION_NAMESPACE}.src.separation")
    separator = module.AudioSeparation()
    return separator.main(
        audio=audio,
        chunk_fade_shape=fade_shape,
        chunk_length=float(chunk_length),
        chunk_overlap=float(chunk_overlap),
    )[3]


def get_available_ffmpeg_video_encoders():
    ffmpeg = get_ffmpeg_path()
    try:
        output = _run_command(
            [ffmpeg, "-hide_banner", "-encoders"],
            "Failed to query ffmpeg encoders",
            text=True,
        )
    except Exception:
        return set()

    encoders = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) >= 2 and "V" in parts[0]:
            encoders.add(parts[1])
    return encoders


def get_preferred_ffmpeg_video_codec():
    encoders = get_available_ffmpeg_video_encoders()
    for codec in ("h264_nvenc", "h264_qsv", "h264_amf", "libx264"):
        if codec == "libx264" or codec in encoders:
            return codec
    return "libx264"


def get_ffmpeg_video_encode_args(codec=None):
    codec = codec or get_preferred_ffmpeg_video_codec()
    if codec == "h264_nvenc":
        return ["-c:v", codec, "-preset", "p4", "-cq", "19", "-pix_fmt", "yuv420p"]
    if codec == "h264_qsv":
        return ["-c:v", codec, "-global_quality", "21", "-pix_fmt", "yuv420p"]
    if codec == "h264_amf":
        return ["-c:v", codec, "-quality", "balanced", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


def _resolve_frame_sequence_dir(frames_dir):
    frames_dir = os.path.abspath(frames_dir)
    direct_match = os.path.join(frames_dir, "frame_00000.png")
    if os.path.exists(direct_match):
        return frames_dir

    child_candidates = []
    try:
        for entry in sorted(os.scandir(frames_dir), key=lambda item: item.name):
            if not entry.is_dir():
                continue
            child_match = os.path.join(entry.path, "frame_00000.png")
            if os.path.exists(child_match):
                child_candidates.append(entry.path)
    except FileNotFoundError:
        pass

    if len(child_candidates) == 1:
        resolved = child_candidates[0]
        log.info(
            "[InfiniteTalk] Resolved nested frame directory %s -> %s",
            frames_dir,
            resolved,
        )
        return resolved

    return frames_dir


def encode_png_sequence_to_video(frames_dir, output_path, fps, start_number=0, frame_count=None):
    frames_dir = _resolve_frame_sequence_dir(frames_dir)
    pattern = os.path.join(frames_dir, "frame_%05d.png")
    ffmpeg = get_ffmpeg_path()
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(float(fps)),
        "-start_number",
        str(int(start_number)),
        "-i",
        pattern,
    ]
    if frame_count is not None:
        command.extend(["-frames:v", str(int(frame_count))])
    command.extend(get_ffmpeg_video_encode_args())
    command.extend(["-movflags", "+faststart", output_path])
    _run_command(command, f"Failed to encode png sequence from {frames_dir}", text=False)


def concat_segments_with_audio(segment_paths, audio_path, output_path, duration):
    ffmpeg = get_ffmpeg_path()
    list_path = f"{output_path}.segments.txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for segment_path in segment_paths:
            safe_path = str(Path(segment_path).resolve()).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")

    copy_command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{float(duration):.6f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        _run_command(copy_command, "Failed to mux segments with copied video stream", text=False)
    except Exception:
        reencode_command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{float(duration):.6f}",
        ]
        reencode_command.extend(get_ffmpeg_video_encode_args())
        reencode_command.extend(["-c:a", "aac", "-movflags", "+faststart", output_path])
        _run_command(reencode_command, "Failed to mux InfiniteTalk output", text=False)
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def build_output_file_ui_entry(file_path, media_format="video/mp4"):
    output_root = Path(folder_paths.get_output_directory()).resolve()
    absolute_path = Path(file_path).resolve()
    try:
        relative_path = absolute_path.relative_to(output_root)
    except Exception:
        relative_path = absolute_path.name
    return {
        "filename": str(relative_path).replace("\\", "/"),
        "subfolder": "",
        "type": "output",
        "format": media_format,
        "fullpath": str(absolute_path),
    }


def get_output_video_path(filename_prefix, output_path=""):
    output_root = os.path.abspath(folder_paths.get_output_directory())
    output_path = str(output_path or "").strip().strip('"').strip("'")

    if output_path:
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(output_path)))
        treat_as_directory = output_path.endswith(("/", "\\")) or (
            os.path.isdir(expanded) and not os.path.isfile(expanded)
        )
        if treat_as_directory:
            target_dir = expanded
            target_prefix = filename_prefix
        else:
            target_dir = os.path.dirname(expanded) or output_root
            target_prefix = os.path.splitext(os.path.basename(expanded))[0] or filename_prefix
        try:
            full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
                target_prefix,
                target_dir,
            )
        except TypeError:
            full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
                target_prefix,
                target_dir,
                0,
                0,
            )
        os.makedirs(full_output_folder, exist_ok=True)
        output_filename = f"{filename}_{counter:05d}.mp4"
        return os.path.join(full_output_folder, output_filename), output_filename

    try:
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_root,
        )
    except TypeError:
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_root,
            0,
            0,
        )
    os.makedirs(full_output_folder, exist_ok=True)
    output_filename = f"{filename}_{counter:05d}.mp4"
    return os.path.join(full_output_folder, output_filename), output_filename


def _folder_choices(category, preferred_substrings, allow_none=False):
    try:
        values = list(folder_paths.get_filename_list(category))
    except Exception:
        values = []

    if allow_none and "none" not in values:
        values = ["none"] + values

    lowered = [(value, value.lower()) for value in values]
    default = values[0] if values else ("none" if allow_none else "")
    for substring in preferred_substrings:
        lowered_substring = substring.lower()
        for value, lowered_value in lowered:
            if lowered_substring in lowered_value:
                default = value
                break
        else:
            continue
        break

    if not values:
        values = [default]
    return values, default


class InfiniteTalkVideoPathNode:
    @classmethod
    def INPUT_TYPES(cls):
        wan_models, wan_default = _folder_choices(
            "diffusion_models",
            ("Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ", "i2v-14b-480p"),
        )
        multitalk_models, multitalk_default = _folder_choices(
            "diffusion_models",
            ("InfiniTetalk", "InfiniteTalk"),
        )
        loras, lora_default = _folder_choices(
            "loras",
            ("lightx2v_I2V_14B_480p_cfg_step_distill_rank128", "lightx2v"),
            allow_none=True,
        )
        vaes, vae_default = _folder_choices(
            "vae",
            ("Wan2_1_VAE_bf16", "wan_2.1_vae"),
        )
        text_encoders, text_default = _folder_choices(
            "text_encoders",
            ("umt5-xxl-enc-bf16", "umt5"),
        )
        clip_models, clip_default = _folder_choices(
            "clip_vision",
            ("clip_vision_h",),
        )

        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "D:/videos/input.mp4",
                    },
                ),
                "wan_model": (wan_models, {"default": wan_default}),
                "infinitetalk_model": (multitalk_models, {"default": multitalk_default}),
                "lora_model": (loras, {"default": lora_default}),
                "vae_model": (vaes, {"default": vae_default}),
                "text_encoder_model": (text_encoders, {"default": text_default}),
                "clip_vision_model": (clip_models, {"default": clip_default}),
                "wav2vec_model": (
                    [
                        "TencentGameMate/chinese-wav2vec2-base",
                        "facebook/wav2vec2-base-960h",
                    ],
                    {"default": "TencentGameMate/chinese-wav2vec2-base"},
                ),
                "target_fps": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.1,
                    },
                ),
                "segment_seconds": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 4.0,
                        "max": 3600.0,
                        "step": 1.0,
                        "tooltip": "Outer segment length. Larger values reduce splice count but use more RAM.",
                    },
                ),
                "frame_window_size": (
                    "INT",
                    {
                        "default": 81,
                        "min": 5,
                        "max": 1024,
                        "step": 4,
                        "tooltip": "Internal InfiniteTalk window size. Must be 4n+1.",
                    },
                ),
                "motion_frame": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Overlap frames for both internal windows and outer segments.",
                    },
                ),
                "steps": ("INT", {"default": 5, "min": 1, "max": 1024, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
                "block_swap": ("INT", {"default": 30, "min": 0, "max": 1024, "step": 1}),
                "filename_prefix": ("STRING", {"default": "InfiniteTalk"}),
                "audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Leave empty to use the source video's audio",
                    },
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
                "output_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional output file or directory"}),
                "max_width": ("STRING", {"default": "832"}),
                "max_height": ("STRING", {"default": "480"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_step": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "separate_vocals": ("BOOLEAN", {"default": False}),
                "keep_intermediates": ("BOOLEAN", {"default": False}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE_PROMPT, "multiline": True}),
                "target_width": ("STRING", {"default": "0"}),
                "target_height": ("STRING", {"default": "0"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("video_path", "filename", "audio")
    FUNCTION = "process"
    CATEGORY = "InfiniteTalk"
    OUTPUT_NODE = True

    def process(
        self,
        video_path,
        wan_model,
        infinitetalk_model,
        lora_model,
        vae_model,
        text_encoder_model,
        clip_vision_model,
        wav2vec_model,
        target_fps,
        segment_seconds,
        frame_window_size,
        motion_frame,
        steps,
        cfg,
        seed,
        block_swap,
        filename_prefix,
        audio_path="",
        audio=None,
        output_path="",
        max_width="832",
        max_height="480",
        lora_strength=1.0,
        start_step=3,
        separate_vocals=False,
        keep_intermediates=False,
        positive_prompt="",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        target_width="0",
        target_height="0",
    ):
        legacy_target_width = target_width
        legacy_target_height = target_height

        target_fps = _coerce_float_value(target_fps, "target_fps", 25.0, min_value=1.0, max_value=60.0)
        segment_seconds = _coerce_float_value(
            segment_seconds,
            "segment_seconds",
            20.0,
            min_value=4.0,
            max_value=120.0,
        )
        frame_window_size = _coerce_int_value(
            frame_window_size,
            "frame_window_size",
            81,
            min_value=5,
            max_value=241,
        )
        motion_frame = _coerce_int_value(
            motion_frame,
            "motion_frame",
            9,
            min_value=1,
            max_value=80,
        )
        steps = _coerce_int_value(steps, "steps", 5, min_value=1, max_value=30)
        cfg = _coerce_float_value(cfg, "cfg", 1.0, min_value=0.0, max_value=10.0)
        lora_strength = _coerce_float_value(
            lora_strength,
            "lora_strength",
            1.0,
            min_value=0.0,
            max_value=4.0,
        )
        start_step = _coerce_int_value(start_step, "start_step", 3, min_value=0, max_value=20)
        seed = _coerce_int_value(seed, "seed", 42, min_value=0, max_value=2**31 - 1, allow_randomize=True)
        block_swap = _coerce_int_value(block_swap, "block_swap", 30, min_value=0, max_value=40)
        audio_path = str(audio_path or "").strip()
        filename_prefix = str(filename_prefix or "InfiniteTalk").strip()
        if filename_prefix.lower() == "randomize" or filename_prefix.isdigit():
            log.warning(
                "[InfiniteTalk] Suspicious filename_prefix=%r, fallback to default",
                filename_prefix,
            )
            filename_prefix = "InfiniteTalk"
        if not audio_path:
            audio_path = ""
        elif audio_path.lower() == "randomize" or audio_path.isdigit():
            log.warning(
                "[InfiniteTalk] Suspicious audio_path=%r, fallback to source video audio",
                audio_path,
            )
            audio_path = ""
        output_path = str(output_path or "")

        resolved_video_path = resolve_user_path(video_path, "video_path")
        frame_window_size = normalize_frame_window_size(frame_window_size)
        motion_frame = int(motion_frame)

        if motion_frame >= frame_window_size:
            raise ValueError("motion_frame must be smaller than frame_window_size")

        final_output_path, output_filename = get_output_video_path(filename_prefix, output_path)
        final_output_path = os.path.abspath(final_output_path)
        work_dir = os.path.join(
            os.path.dirname(final_output_path),
            f".{Path(final_output_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}_infinitetalk",
        )
        os.makedirs(work_dir, exist_ok=True)

        engine = None
        success = False

        try:
            video_info = get_video_info(resolved_video_path)
            log.info(
                "[InfiniteTalk] Source video %s x %s @ %.3ffps, %.3fs",
                video_info["width"],
                video_info["height"],
                video_info["fps"],
                video_info["duration"],
            )
            max_width = _coerce_int_value(max_width, "max_width", 832, min_value=16, max_value=4096)
            max_height = _coerce_int_value(max_height, "max_height", 480, min_value=16, max_value=4096)
            target_width = _coerce_int_value(
                legacy_target_width,
                "target_width",
                0,
                min_value=0,
                max_value=8192,
            )
            target_height = _coerce_int_value(
                legacy_target_height,
                "target_height",
                0,
                min_value=0,
                max_value=8192,
            )
            if target_width > 0:
                log.info("[InfiniteTalk] Using legacy target_width=%s for auto resolution clamp", target_width)
                max_width = target_width
            if target_height > 0:
                log.info("[InfiniteTalk] Using legacy target_height=%s for auto resolution clamp", target_height)
                max_height = target_height
            target_width, target_height = _resolve_auto_target_size(
                video_info["width"],
                video_info["height"],
                max_width=max_width,
                max_height=max_height,
            )
            log.info(
                "[InfiniteTalk] Auto target size %sx%s (max %sx%s, aligned to 16)",
                target_width,
                target_height,
                max_width,
                max_height,
            )
            log.info("[InfiniteTalk] Work directory: %s", work_dir)

            prepared_audio_path = os.path.join(work_dir, "input_audio.wav")
            if str(audio_path or "").strip():
                resolved_audio_path = resolve_user_path(audio_path, "audio_path")
                with _timed_log(
                    "Prepare audio",
                    source="audio_path",
                    path=resolved_audio_path,
                ):
                    extract_audio_to_wav(resolved_audio_path, prepared_audio_path)
            elif audio is not None:
                with _timed_log("Prepare audio", source="AUDIO input"):
                    save_audio_input(audio, prepared_audio_path)
            else:
                with _timed_log(
                    "Prepare audio",
                    source="video audio track",
                    path=resolved_video_path,
                ):
                    extract_audio_to_wav(resolved_video_path, prepared_audio_path)

            if separate_vocals:
                log.info("[InfiniteTalk] Separating vocals from source audio")
                with _timed_log("Load prepared audio", path=prepared_audio_path):
                    separation_audio = load_audio_file(prepared_audio_path)
                with _timed_log("Separate vocals", path=prepared_audio_path):
                    vocals_audio = run_audio_separation(separation_audio)
                vocals_path = os.path.join(work_dir, "vocals.wav")
                with _timed_log("Save separated vocals", path=vocals_path):
                    save_audio_input(vocals_audio, vocals_path)
                prepared_audio_path = vocals_path

            audio_duration = probe_media_duration(prepared_audio_path, "audio")
            output_duration = min(float(video_info["duration"]), float(audio_duration))
            if output_duration <= 0:
                raise RuntimeError("Resolved output duration is 0. Check the input video and audio.")

            total_frames = max(1, int(output_duration * float(target_fps)))
            segment_frames = max(int(float(segment_seconds) * float(target_fps)), int(frame_window_size))
            stride_frames = segment_frames - motion_frame
            if stride_frames <= 0:
                raise ValueError("segment_seconds is too small for the chosen motion_frame")

            num_segments = max(
                1,
                int(math.ceil(max(total_frames - segment_frames, 0) / float(stride_frames))) + 1,
            )
            log.info(
                "[InfiniteTalk] Rendering %s frames in %s segments (segment=%s, overlap=%s)",
                total_frames,
                num_segments,
                segment_frames,
                motion_frame,
            )

            with _timed_log(
                "Initialize runtime",
                wan_model=wan_model,
                infinitetalk_model=infinitetalk_model,
                lora_model=lora_model,
            ):
                engine = InfiniteTalkEngine(
                    {
                        "wan_model": wan_model,
                        "infinitetalk_model": infinitetalk_model,
                        "lora_model": lora_model,
                        "lora_strength": lora_strength,
                        "vae_model": vae_model,
                        "text_encoder_model": text_encoder_model,
                        "clip_vision_model": clip_vision_model,
                        "wav2vec_model": wav2vec_model,
                        "block_swap": block_swap,
                        "use_non_blocking": True,
                        "prefetch_blocks": 1,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "base_precision": "fp16",
                        "quantization": "fp8_e4m3fn_scaled",
                        "attention_mode": "sageattn",
                        "vae_precision": "bf16",
                        "text_precision": "bf16",
                        "text_quantization": "disabled",
                        "wav2vec_precision": "fp16",
                        "wav2vec_load_device": "main_device",
                        "audio_scale": 1.0,
                        "audio_cfg_scale": 1.0,
                        "clip_strength_1": 1.0,
                        "clip_strength_2": 0.7,
                        "scheduler": "dpm++_sde",
                        "shift": 11.0,
                        "tile_x": 272,
                        "tile_y": 272,
                        "tile_stride_x": 144,
                        "tile_stride_y": 128,
                        "encode_tiled_vae": False,
                        "decode_tiled_vae": False,
                        "normalize_loudness": True,
                    }
                )

            segment_paths = []
            for segment_index in range(num_segments):
                _throw_if_interrupted()
                segment_start_frame = segment_index * stride_frames
                requested_frames = min(segment_frames, total_frames - segment_start_frame)
                if requested_frames <= 0:
                    break

                segment_start_sec = float(segment_start_frame) / float(target_fps)
                segment_duration_sec = float(requested_frames) / float(target_fps)
                segment_label = f"{segment_index + 1}/{num_segments}"
                log.info(
                    "[InfiniteTalk] Segment %s start_frame=%s requested_frames=%s start_sec=%.3f duration_sec=%.3f",
                    segment_label,
                    segment_start_frame,
                    requested_frames,
                    segment_start_sec,
                    segment_duration_sec,
                )

                with _timed_log(
                    "Segment video load",
                    segment=segment_label,
                    start_frame=segment_start_frame,
                    frames=requested_frames,
                    width=int(target_width),
                    height=int(target_height),
                    fps=float(target_fps),
                ):
                    source_frames = load_video_chunk(
                        resolved_video_path,
                        segment_start_frame,
                        requested_frames,
                        int(target_width),
                        int(target_height),
                        float(target_fps),
                    )
                log.info(
                    "[InfiniteTalk] Segment %s video ready actual_frames=%s tensor_shape=%s",
                    segment_label,
                    int(source_frames.shape[0]),
                    tuple(source_frames.shape),
                )

                with _timed_log(
                    "Segment audio load",
                    segment=segment_label,
                    start_sec=f"{segment_start_sec:.3f}",
                    duration_sec=f"{segment_duration_sec:.3f}",
                    sample_rate=16000,
                ):
                    segment_audio = load_audio_segment(
                        prepared_audio_path,
                        start_sec=segment_start_sec,
                        duration_sec=segment_duration_sec,
                        target_sr=16000,
                    )
                log.info(
                    "[InfiniteTalk] Segment %s audio ready samples=%s waveform_shape=%s",
                    segment_label,
                    int(segment_audio["waveform"].shape[-1]),
                    tuple(segment_audio["waveform"].shape),
                )

                segment_frames_dir = os.path.join(work_dir, f"segment_{segment_index:04d}_frames")
                os.makedirs(segment_frames_dir, exist_ok=True)

                with _timed_log(
                    "Segment render",
                    segment=segment_label,
                    input_frames=int(source_frames.shape[0]),
                    frame_window_size=int(frame_window_size),
                    motion_frame=motion_frame,
                    steps=int(steps),
                    cfg=float(cfg),
                    seed=int(seed),
                ):
                    render_result = engine.render_segment(
                        source_frames=source_frames,
                        audio_input=segment_audio,
                        fps=float(target_fps),
                        frame_window_size=int(frame_window_size),
                        motion_frame=motion_frame,
                        steps=int(steps),
                        cfg_scale=float(cfg),
                        seed=int(seed),
                        segment_output_dir=segment_frames_dir,
                        start_step=int(start_step),
                        segment_label=segment_label,
                    )

                actual_num_frames = min(int(render_result["actual_num_frames"]), int(requested_frames))
                rendered_frames_dir = str(render_result.get("output_path", "") or segment_frames_dir)
                skip_start = 0 if segment_index == 0 else min(motion_frame, actual_num_frames)
                visible_frames = max(0, actual_num_frames - skip_start)
                log.info(
                    "[InfiniteTalk] Segment %s render result actual_num_frames=%s skip_start=%s visible_frames=%s output_dir=%s",
                    segment_label,
                    actual_num_frames,
                    skip_start,
                    visible_frames,
                    rendered_frames_dir,
                )
                if visible_frames <= 0:
                    log.warning(
                        "[InfiniteTalk] Segment %s produced no visible frames after overlap trim",
                        segment_label,
                    )
                    continue

                segment_video_path = os.path.join(work_dir, f"segment_{segment_index:04d}.mp4")
                with _timed_log(
                    "Segment encode video",
                    segment=segment_label,
                    start_number=skip_start,
                    frame_count=visible_frames,
                    frames_dir=rendered_frames_dir,
                    output=segment_video_path,
                ):
                    encode_png_sequence_to_video(
                        rendered_frames_dir,
                        segment_video_path,
                        float(target_fps),
                        start_number=skip_start,
                        frame_count=visible_frames,
                    )
                log.info(
                    "[InfiniteTalk] Segment %s saved video=%s",
                    segment_label,
                    segment_video_path,
                )
                segment_paths.append(segment_video_path)

            if not segment_paths:
                raise RuntimeError("InfiniteTalk did not produce any segment output")

            with _timed_log(
                "Final concat",
                segments=len(segment_paths),
                duration_sec=f"{output_duration:.3f}",
                output=final_output_path,
            ):
                concat_segments_with_audio(segment_paths, prepared_audio_path, final_output_path, output_duration)
            with _timed_log(
                "Final audio load",
                duration_sec=f"{output_duration:.3f}",
                sample_rate=16000,
            ):
                output_audio = load_audio_segment(prepared_audio_path, 0.0, output_duration, target_sr=16000)

            preview_entry = build_output_file_ui_entry(final_output_path)
            log.info("[InfiniteTalk] Final output ready: %s", final_output_path)
            success = True
            return {
                "ui": {
                    "text": [final_output_path],
                    "gifs": [preview_entry],
                },
                "result": (
                    final_output_path,
                    output_filename,
                    output_audio,
                ),
            }
        finally:
            if engine is not None:
                engine.unload()
            if success and not keep_intermediates:
                shutil.rmtree(work_dir, ignore_errors=True)


class InfiniteTalkVideoSyncPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "segment_seconds": ("FLOAT", {"default": 20.0}),
                "target_fps": ("FLOAT", {"default": 25.0}),
                "motion_frame": ("INT", {"default": 9}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "fps", "duration", "num_segments", "info_text")
    FUNCTION = "preview"
    CATEGORY = "InfiniteTalk"

    def preview(self, video_path, segment_seconds, target_fps, motion_frame):
        try:
            resolved_video_path = resolve_user_path(video_path, "video_path")
        except Exception as exc:
            return (0, 0, 0.0, 0.0, 0, str(exc))

        info = get_video_info(resolved_video_path)
        total_frames = max(1, int(info["duration"] * float(target_fps)))
        segment_frames = max(int(float(segment_seconds) * float(target_fps)), 81)
        stride = segment_frames - int(motion_frame)
        num_segments = max(1, int(math.ceil(max(total_frames - segment_frames, 0) / float(stride))) + 1)

        ram_per_segment_mb = (
            segment_frames * info["width"] * info["height"] * 3 * 4 / 1024 / 1024
        )

        text = (
            f"source: {info['width']}x{info['height']} @ {info['fps']:.3f}fps\n"
            f"duration: {info['duration']:.3f}s ({total_frames} frames @ {target_fps}fps)\n"
            f"segments: {num_segments} x {segment_seconds:.1f}s with {motion_frame} overlap frames\n"
            f"estimated frame RAM/segment: {ram_per_segment_mb:.0f} MB"
        )

        return (
            info["width"],
            info["height"],
            info["fps"],
            info["duration"],
            num_segments,
            text,
        )


NODE_CLASS_MAPPINGS = {
    "InfiniteTalkVideoPathNode": InfiniteTalkVideoPathNode,
    "InfiniteTalkVideoSync": InfiniteTalkVideoPathNode,
    "InfiniteTalkVideoSyncPreview": InfiniteTalkVideoSyncPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteTalkVideoPathNode": "InfiniteTalk Lip Sync (Video Path)",
    "InfiniteTalkVideoSync": "InfiniteTalk Lip Sync (Video Path)",
    "InfiniteTalkVideoSyncPreview": "InfiniteTalk Video Info Preview",
}

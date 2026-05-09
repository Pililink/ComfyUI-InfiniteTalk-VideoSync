import importlib
import logging
import math
import os
import queue
import random
import re
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
    slice_multitalk_embeds,
)

log = logging.getLogger("InfiniteTalkVideoSync")

_AUDIO_SEPARATION_NAMESPACE = "_infinitetalk_audio_separation"

DEFAULT_POSITIVE_PROMPT = ""

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

RESIZE_MODES = ("crop_to_size", "fit_inside")
RESIZE_FILTERS = ("nearest-exact", "lanczos", "bicubic", "bilinear", "area")
VIDEO_CODECS = ("libx264", "h264_nvenc", "h264_qsv", "h264_amf", "auto")
OUTPUT_MODES = ("ffmpeg_pipe", "png_sequence")
REFERENCE_WORKFLOW_MODEL_HINTS = {
    "lora_model": ("lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16",),
    "vae_model": ("Wan2_1_VAE_bf16",),
    "text_encoder_model": ("umt5-xxl-enc-bf16",),
}
FFMPEG_SCALE_FLAGS = {
    "nearest-exact": "neighbor",
    "lanczos": "lanczos",
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "area": "area",
}


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


def _should_cleanup_work_dir(keep_intermediates):
    return not bool(keep_intermediates)


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


def _normalize_resize_mode(resize_mode):
    resize_mode = str(resize_mode or "crop_to_size").strip()
    if resize_mode in RESIZE_MODES:
        return resize_mode
    log.warning(
        "[InfiniteTalk] Unknown resize_mode=%r, fallback to crop_to_size",
        resize_mode,
    )
    return "crop_to_size"


def _normalize_resize_filter(resize_filter):
    resize_filter = str(resize_filter or "nearest-exact").strip()
    if resize_filter in RESIZE_FILTERS:
        return resize_filter
    log.warning(
        "[InfiniteTalk] Unknown resize_filter=%r, fallback to nearest-exact",
        resize_filter,
    )
    return "nearest-exact"


def _normalize_video_codec(video_codec):
    video_codec = str(video_codec or "libx264").strip()
    if video_codec in VIDEO_CODECS:
        return video_codec
    log.warning(
        "[InfiniteTalk] Unknown video_codec=%r, fallback to libx264",
        video_codec,
    )
    return "libx264"


def _normalize_output_mode(output_mode):
    output_mode = str(output_mode or "ffmpeg_pipe").strip()
    if output_mode in OUTPUT_MODES:
        return output_mode
    log.warning(
        "[InfiniteTalk] Unknown output_mode=%r, fallback to ffmpeg_pipe",
        output_mode,
    )
    return "ffmpeg_pipe"


def resolve_target_size(source_width, source_height, max_width=480, max_height=832, resize_mode="crop_to_size"):
    resize_mode = _normalize_resize_mode(resize_mode)
    if resize_mode == "fit_inside":
        return _resolve_auto_target_size(
            source_width,
            source_height,
            max_width=max_width,
            max_height=max_height,
        )

    source_width = int(source_width)
    source_height = int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source size: {source_width}x{source_height}")

    target_width = _align_down_to_multiple(max_width, 16)
    target_height = _align_down_to_multiple(max_height, 16)
    return target_width, target_height


def build_video_filter(
    target_width,
    target_height,
    target_fps,
    resize_mode="crop_to_size",
    resize_filter="nearest-exact",
):
    resize_mode = _normalize_resize_mode(resize_mode)
    resize_filter = _normalize_resize_filter(resize_filter)
    scale_flags = FFMPEG_SCALE_FLAGS[resize_filter]
    target_width = int(target_width)
    target_height = int(target_height)
    fps_filter = f"fps={target_fps}"
    if resize_mode == "fit_inside":
        return f"{fps_filter},scale={target_width}:{target_height}:flags={scale_flags}"
    return (
        f"{fps_filter},"
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase:flags={scale_flags},"
        f"crop={target_width}:{target_height}"
    )


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


def _build_full_pipe_command(
    video_path,
    target_width,
    target_height,
    target_fps,
    resize_mode,
    resize_filter,
    max_total_frames,
):
    ffmpeg = get_ffmpeg_path()
    base_filter = build_video_filter(target_width, target_height, target_fps, resize_mode, resize_filter)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        base_filter,
        "-vsync",
        "cfr",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
    ]
    if max_total_frames is not None and max_total_frames > 0:
        command.extend(["-frames:v", str(int(max_total_frames))])
    command.append("pipe:1")
    return command


def stream_video_frame_chunks(
    video_path,
    target_width,
    target_height,
    target_fps,
    resize_mode="crop_to_size",
    resize_filter="lanczos",
    max_total_frames=None,
    chunk_frames=81,
):
    """Yield successive chunks of decoded frames as `[T, H, W, 3]` float32
    tensors.

    A *single* ffmpeg subprocess decodes the file linearly with the requested
    fps + scale filter. Chunks are sliced from the rawvideo stdout stream by
    byte count, so the frame indices observed by the consumer match exactly
    the indices used to slice the global wav2vec embeddings - no `-ss` /
    `select` games, no per-chunk redecode, no GOP-boundary drift. When the
    source video ends before `max_total_frames` the generator simply stops.
    """
    target_width = int(target_width)
    target_height = int(target_height)
    chunk_frames = max(1, int(chunk_frames))
    frame_size = target_width * target_height * 3

    command = _build_full_pipe_command(
        video_path,
        target_width,
        target_height,
        target_fps,
        resize_mode,
        resize_filter,
        max_total_frames,
    )
    log.info("[InfiniteTalk] Streaming source video via single ffmpeg pipe (max_frames=%s)", max_total_frames)

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    delivered_frames = 0
    try:
        while True:
            _throw_if_interrupted()
            if max_total_frames is not None and delivered_frames >= max_total_frames:
                break
            want_frames = chunk_frames
            if max_total_frames is not None:
                want_frames = min(want_frames, max_total_frames - delivered_frames)
            want_bytes = frame_size * want_frames
            buffer = bytearray()
            while len(buffer) < want_bytes:
                read_size = want_bytes - len(buffer)
                more = proc.stdout.read(read_size)
                if not more:
                    break
                buffer.extend(more)
            actual_frames = len(buffer) // frame_size
            if actual_frames == 0:
                break
            valid = bytes(buffer[: actual_frames * frame_size])
            del buffer
            arr = np.frombuffer(valid, dtype=np.uint8).reshape(
                actual_frames, target_height, target_width, 3
            )
            tensor = torch.tensor(arr, dtype=torch.float32).div_(255.0)
            del arr, valid
            delivered_frames += actual_frames
            yield tensor
            if actual_frames < want_frames:
                # ffmpeg returned a short read; the source is exhausted.
                break
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        except Exception:
            stderr_bytes = b""
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()
        if proc.returncode not in (0, None) and delivered_frames == 0:
            details = stderr_bytes.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"ffmpeg failed for {video_path}: {details or 'no output'}"
            )


def load_video_chunk(
    video_path,
    start_frame,
    num_frames,
    target_width,
    target_height,
    target_fps,
    resize_mode="crop_to_size",
    resize_filter="lanczos",
):
    """Decode `num_frames` frames starting at `start_frame` (in the resampled
    target_fps timeline) and return them as a `[T, H, W, 3]` float32 tensor.

    Implemented on top of `stream_video_frame_chunks` so that there is one
    code path for "give me frames N..M" regardless of whether the caller
    needs the whole video (streaming long encode) or just a slice (preview).
    The decoder runs a single ffmpeg pass and discards the leading
    `start_frame` frames in Python land - this is O(start_frame) decode but
    perfectly accurate at any GOP structure, unlike `-ss` + `select`.
    """
    start_frame = max(0, int(start_frame))
    num_frames = max(1, int(num_frames))
    needed_total = start_frame + num_frames
    chunk_frames = min(num_frames, 81)
    collected = []
    consumed = 0
    for chunk in stream_video_frame_chunks(
        video_path,
        target_width=target_width,
        target_height=target_height,
        target_fps=target_fps,
        resize_mode=resize_mode,
        resize_filter=resize_filter,
        max_total_frames=needed_total,
        chunk_frames=chunk_frames,
    ):
        chunk_size = int(chunk.shape[0])
        chunk_start = consumed
        chunk_end = consumed + chunk_size
        consumed = chunk_end
        if chunk_end <= start_frame:
            continue
        slice_start = max(0, start_frame - chunk_start)
        slice_end = min(chunk_size, needed_total - chunk_start)
        if slice_end <= slice_start:
            continue
        collected.append(chunk[slice_start:slice_end].clone())
        if consumed >= needed_total:
            break
    if not collected:
        raise RuntimeError(f"ffmpeg returned no frames for {video_path} @ frame {start_frame}")
    return torch.cat(collected, dim=0)


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


def normalize_source_video_to_cfr(
    src_path,
    dst_path,
    target_fps,
    video_codec="libx264",
    video_crf=18,
    pad_to_seconds=None,
):
    """Re-encode `src_path` to a CFR file at exactly `target_fps` and
    optionally extend it with cloned last frame to `pad_to_seconds`.

    Why this is the default in process():

    - Many web/mobile videos lie in their container metadata. ffprobe
      reports e.g. `r_frame_rate=25, duration=33.28s` but the file is
      actually VFR or only contains ~31.92s of real frames. This pass
      runs `-r target_fps -vsync cfr` so the output's frame count is
      exactly `round(duration * target_fps)` regardless of the source's
      header.
    - When the audio is longer than the resulting video, we want the
      tail of the rendered output to hold on the last source frame
      rather than fail the run. That's what `tpad=stop_mode=clone` does:
      it clones the last frame indefinitely; combined with `-t` the
      muxer trims to exactly `pad_to_seconds`. The user picks this
      behavior; v2v fidelity is preserved up to the source's natural
      end and the rest is a static still.
    """
    ffmpeg = get_ffmpeg_path()
    encode_args = get_ffmpeg_video_encode_args(video_codec, video_crf)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
    ]
    if pad_to_seconds is not None and float(pad_to_seconds) > 0:
        # tpad's stop_duration is *additional* time appended after EOF; we
        # set it generously and rely on `-t` to clip the output to exactly
        # the requested length, so we don't need to know the source's true
        # duration up front.
        command.extend(
            [
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={float(pad_to_seconds):.6f}",
                "-t",
                f"{float(pad_to_seconds):.6f}",
            ]
        )
    command.extend(
        [
            "-r",
            str(float(target_fps)),
            "-vsync",
            "cfr",
            "-an",
        ]
    )
    command.extend(encode_args)
    command.extend(["-movflags", "+faststart", dst_path])
    _run_command(
        command,
        f"Failed to normalize {src_path} to CFR {target_fps}fps",
        text=False,
    )
    return dst_path


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


def get_ffmpeg_video_encode_args(codec="libx264", crf=19):
    codec = _normalize_video_codec(codec)
    if codec == "auto":
        codec = get_preferred_ffmpeg_video_codec()
    crf = max(0, min(51, int(crf)))
    if codec == "h264_nvenc":
        return ["-c:v", codec, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if codec == "h264_qsv":
        return ["-c:v", codec, "-global_quality", str(crf), "-pix_fmt", "yuv420p"]
    if codec == "h264_amf":
        return ["-c:v", codec, "-quality", "quality", "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]


def build_ffmpeg_pipe_command(
    audio_path,
    output_path,
    target_width,
    target_height,
    target_fps,
    duration,
    video_codec="libx264",
    video_crf=19,
):
    command = [
        get_ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(target_width)}x{int(target_height)}",
        "-r",
        str(float(target_fps)),
        "-i",
        "pipe:0",
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{float(duration):.6f}",
    ]
    command.extend(get_ffmpeg_video_encode_args(video_codec, video_crf))
    command.extend(["-c:a", "aac", "-movflags", "+faststart", output_path])
    return command


class FfmpegPipeSink:
    def __init__(
        self,
        audio_path,
        output_path,
        target_width,
        target_height,
        target_fps,
        duration,
        video_codec="libx264",
        video_crf=19,
        max_queue_windows=1,
    ):
        self.output_path = output_path
        self.command = build_ffmpeg_pipe_command(
            audio_path=audio_path,
            output_path=output_path,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            duration=duration,
            video_codec=video_codec,
            video_crf=video_crf,
        )
        self._queue = queue.Queue(maxsize=max(1, int(max_queue_windows)))
        self._closed = False
        self._worker_error = None
        self._expected_frames = max(1, int(round(float(duration) * float(target_fps))))
        self._written_frames = 0
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._worker = threading.Thread(
            target=self._write_worker,
            name="InfiniteTalkFfmpegPipeSink",
            daemon=True,
        )
        self._worker.start()

    def write_window(self, videos):
        if self._closed:
            raise RuntimeError("Cannot write to a closed ffmpeg pipe")
        self._raise_worker_error()
        frame_count = int(videos.shape[1])
        remaining_frames = self._expected_frames - self._written_frames
        if remaining_frames <= 0 or frame_count <= 0:
            return 0
        if frame_count > remaining_frames:
            videos = videos[:, :remaining_frames]
            frame_count = int(videos.shape[1])
        while True:
            _throw_if_interrupted()
            self._raise_worker_error()
            try:
                self._queue.put(videos, timeout=0.1)
                self._written_frames += frame_count
                return frame_count
            except queue.Full:
                continue

    def close(self):
        if not self._closed:
            self._closed = True
            if self._worker.is_alive():
                self._put_sentinel()
                self._queue.join()
            else:
                self._drain_queue()
            self._worker.join()
        self._raise_worker_error()

    def abort(self):
        self._closed = True
        self._drain_queue()
        try:
            if self._process.poll() is None:
                self._process.kill()
        finally:
            if self._worker.is_alive():
                self._worker.join(timeout=2.0)

    def _put_sentinel(self):
        while True:
            _throw_if_interrupted()
            self._raise_worker_error()
            try:
                self._queue.put(None, timeout=0.1)
                return
            except queue.Full:
                continue

    def _write_worker(self):
        try:
            while True:
                videos = self._queue.get()
                try:
                    if videos is None:
                        break
                    raw_bytes = self._videos_to_raw_rgb_bytes(videos)
                    if raw_bytes:
                        self._process.stdin.write(raw_bytes)
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._worker_error = exc
            self._drain_queue()
            try:
                if self._process.poll() is None:
                    self._process.kill()
            except Exception:
                pass
        finally:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except Exception:
                pass

            return_code = self._process.wait()
            stderr = b""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read() or b""
            if return_code != 0 and self._worker_error is None:
                details = stderr.decode("utf-8", errors="ignore").strip()
                if details:
                    self._worker_error = RuntimeError(
                        f"Failed to encode final InfiniteTalk video to {self.output_path}: {details}"
                    )
                else:
                    self._worker_error = RuntimeError(
                        f"Failed to encode final InfiniteTalk video to {self.output_path} (exit code {return_code})"
                    )

    def _videos_to_raw_rgb_bytes(self, videos):
        video_np = (
            videos.clamp(-1.0, 1.0)
            .add(1.0)
            .div(2.0)
            .mul(255)
            .cpu()
            .float()
            .numpy()
            .transpose(1, 2, 3, 0)
            .astype("uint8")
        )
        return np.ascontiguousarray(video_np).tobytes()

    def _drain_queue(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def _raise_worker_error(self):
        if self._worker_error is not None:
            raise self._worker_error


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


def encode_png_sequence_to_video(
    frames_dir,
    output_path,
    fps,
    start_number=0,
    frame_count=None,
    video_codec="libx264",
    video_crf=19,
):
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
    command.extend(get_ffmpeg_video_encode_args(video_codec, video_crf))
    command.extend(["-movflags", "+faststart", output_path])
    _run_command(command, f"Failed to encode png sequence from {frames_dir}", text=False)


def _stream_encode_full_source_latent(
    engine,
    video_path,
    total_frames,
    target_width,
    target_height,
    target_fps,
    resize_mode,
    resize_filter,
    chunk_frames,
):
    """Decode + VAE-encode the source video in chunks and return a CPU latent
    that covers `total_frames` frames at `target_fps`.

    InfiniteTalk's multitalk_loop slices `samples["samples"]` along the time
    axis per inner window, so the latent must span the entire output length.
    Encoding chunk by chunk keeps the GPU pixel buffer bounded while the
    latent (which is ~150x smaller per byte) accumulates on CPU.
    """
    chunk_frames = max(1, int(chunk_frames))
    chunk_frames = ((chunk_frames - 1) // 4) * 4 + 1  # latent stride is 4
    chunks = []
    first_frame_pixels = None
    last_frame_pixels = None
    delivered = 0
    chunk_index = 0
    for source_chunk in stream_video_frame_chunks(
        video_path,
        target_width=int(target_width),
        target_height=int(target_height),
        target_fps=float(target_fps),
        resize_mode=resize_mode,
        resize_filter=resize_filter,
        max_total_frames=int(total_frames),
        chunk_frames=int(chunk_frames),
    ):
        actual_chunk_frames = int(source_chunk.shape[0])
        log.info(
            "[InfiniteTalk] Source video chunk %s ready start_frame=%s frames=%s",
            chunk_index + 1,
            delivered,
            actual_chunk_frames,
        )
        if first_frame_pixels is None:
            first_frame_pixels = source_chunk[0:1].clone()
        last_frame_pixels = source_chunk[-1:].clone()

        latent_chunk = engine.encode_source_latent(
            source_chunk,
            label=f"chunk_{chunk_index + 1}",
        )
        chunks.append(latent_chunk)
        delivered += actual_chunk_frames
        chunk_index += 1
        del source_chunk

    if not chunks:
        raise RuntimeError("Source video produced no frames")

    if delivered < total_frames:
        # The duration check at the top of process() lets a few frames of
        # rounding slack through, but actual decoded frames must cover the
        # audio one-for-one. If ffmpeg's fps filter dropped frames (common
        # with mis-tagged VFR sources reporting r_frame_rate=25 but actually
        # decoding at ~24fps) we surface that here with actionable advice
        # instead of silently padding.
        raise RuntimeError(
            f"Source video produced only {delivered} frames at {target_fps}fps "
            f"but {total_frames} are required to match the audio. "
            "Enable auto_normalize_fps (default ON) to let the node re-encode "
            "the source to constant frame rate, or re-encode it manually with "
            f"`ffmpeg -i in.mp4 -r {target_fps} -vsync cfr out.mp4`."
        )

    full_latent = torch.cat(chunks, dim=2) if chunks[0].dim() == 5 else torch.cat(chunks, dim=1)
    expected_latent_t = (total_frames - 1) // 4 + 1
    actual_latent_t = full_latent.shape[2] if full_latent.dim() == 5 else full_latent.shape[1]
    if actual_latent_t > expected_latent_t:
        if full_latent.dim() == 5:
            full_latent = full_latent[:, :, :expected_latent_t].contiguous()
        else:
            full_latent = full_latent[:, :expected_latent_t].contiguous()
    return full_latent, first_frame_pixels, last_frame_pixels


def _encode_full_video_with_audio(
    frames_dir,
    audio_path,
    output_path,
    target_fps,
    duration,
    video_codec="libx264",
    video_crf=19,
):
    """Encode the rendered PNG sequence into a single mp4 with the audio
    track muxed in, in one ffmpeg invocation. Faster and more accurate than
    the old per-segment encode + concat-copy pipeline.
    """
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
        str(float(target_fps)),
        "-start_number",
        "0",
        "-i",
        pattern,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{float(duration):.6f}",
    ]
    command.extend(get_ffmpeg_video_encode_args(video_codec, video_crf))
    command.extend(["-c:a", "aac", "-movflags", "+faststart", output_path])
    _run_command(command, f"Failed to encode final InfiniteTalk video to {output_path}", text=False)


def concat_segments_with_audio(
    segment_paths,
    audio_path,
    output_path,
    duration,
    video_codec="libx264",
    video_crf=19,
):
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
        reencode_command.extend(get_ffmpeg_video_encode_args(video_codec, video_crf))
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


def _strip_counter_suffix(filename_prefix):
    return re.sub(r"_(\d{5,})$", "", str(filename_prefix or "").strip())


def _next_output_counter(output_folder, filename_prefix):
    filename_prefix = str(filename_prefix or "InfiniteTalk")
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)(?:[_.]|$)", re.IGNORECASE)
    max_counter = 0
    try:
        entries = os.scandir(output_folder)
    except FileNotFoundError:
        return 1

    with entries:
        for entry in entries:
            match = pattern.match(entry.name)
            if not match:
                continue
            try:
                max_counter = max(max_counter, int(match.group(1)))
            except ValueError:
                continue
    return max_counter + 1


def _numbered_mp4_path(output_folder, filename_prefix):
    filename_prefix = _strip_counter_suffix(filename_prefix) or "InfiniteTalk"
    os.makedirs(output_folder, exist_ok=True)
    counter = _next_output_counter(output_folder, filename_prefix)
    output_filename = f"{filename_prefix}_{counter:05d}.mp4"
    return os.path.join(output_folder, output_filename), output_filename


def _comfy_save_folder_and_prefix(filename_prefix, output_root):
    try:
        full_output_folder, filename, _counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_root,
        )
    except TypeError:
        full_output_folder, filename, _counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_root,
            0,
            0,
        )
    return full_output_folder, _strip_counter_suffix(filename)


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
        return _numbered_mp4_path(target_dir, target_prefix)

    full_output_folder, filename = _comfy_save_folder_and_prefix(filename_prefix, output_root)
    return _numbered_mp4_path(full_output_folder, filename)


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


def _warn_reference_model_mismatch(field_name, value):
    hints = REFERENCE_WORKFLOW_MODEL_HINTS.get(field_name)
    if not hints:
        return
    lowered = str(value or "").lower()
    if any(hint.lower() in lowered for hint in hints):
        return
    log.warning(
        "[InfiniteTalk] %s=%s differs from the reference workflow preferred model (%s); output quality may differ",
        field_name,
        value or "<empty>",
        " or ".join(hints),
    )


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
            (
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16",
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank128",
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank64",
                "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64",
                "lightx2v",
            ),
            allow_none=True,
        )
        vaes, vae_default = _folder_choices(
            "vae",
            ("Wan2_1_VAE_bf16", "wan_2.1_vae"),
        )
        text_encoders, text_default = _folder_choices(
            "text_encoders",
            ("umt5-xxl-enc-bf16", "umt5-xxl-enc-fp8_e4m3fn", "umt5"),
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
                "audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "D:/audio/input.wav",
                    },
                ),
            },
            "optional": {
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
                "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "fp16"}),
                "quantization": (
                    ["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e4m3fn_scaled"],
                    {"default": "fp8_e4m3fn_scaled"},
                ),
                "attention_mode": (
                    [
                        "sageattn",
                        "comfy",
                        "sdpa",
                        "flash_attn_2",
                        "flash_attn_3",
                        "sageattn_3",
                        "sageattn_compiled",
                    ],
                    {"default": "sageattn"},
                ),
                "vae_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "text_precision": (["fp32", "bf16"], {"default": "bf16"}),
                "text_quantization": (
                    ["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast"],
                    {"default": "disabled"},
                ),
                "wav2vec_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "wav2vec_load_device": (["main_device", "offload_device"], {"default": "main_device"}),
                "target_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "segment_seconds": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 4.0,
                        "max": 120.0,
                        "step": 1.0,
                        "tooltip": "Outer segment length. Larger values reduce splice count but use more RAM.",
                    },
                ),
                "frame_window_size": (
                    "INT",
                    {
                        "default": 81,
                        "min": 5,
                        "max": 241,
                        "step": 4,
                        "tooltip": "Internal InfiniteTalk window size. Must be 4n+1.",
                    },
                ),
                "motion_frame": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 80,
                        "step": 1,
                        "tooltip": "Overlap frames for both internal windows and outer segments.",
                    },
                ),
                "steps": ("INT", {"default": 5, "min": 1, "max": 30, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "scheduler": (
                    ["dpm++_sde", "flowmatch_distill", "unipc", "euler", "euler_ancestral"],
                    {"default": "dpm++_sde"},
                ),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**31 - 1}),
                "start_step": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batched_cfg": ("BOOLEAN", {"default": False}),
                "rope_function": (["default", "comfy", "comfy_chunked"], {"default": "comfy"}),
                "add_noise_to_samples": ("BOOLEAN", {"default": True}),
                "sampler_force_offload": ("BOOLEAN", {"default": True}),
                "block_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1}),
                "use_non_blocking": ("BOOLEAN", {"default": True}),
                "prefetch_blocks": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "normalize_loudness": ("BOOLEAN", {"default": True}),
                "clip_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "clip_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "clip_use_last_frame": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Match the reference workflow by default: only image_1 is used for CLIP vision.",
                    },
                ),
                "encode_tiled_vae": ("BOOLEAN", {"default": False}),
                "decode_tiled_vae": ("BOOLEAN", {"default": False}),
                "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 16}),
                "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 16}),
                "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 16}),
                "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 16}),
                "filename_prefix": ("STRING", {"default": "InfiniteTalk"}),
                "output_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional output file or directory"}),
                "output_mode": (list(OUTPUT_MODES), {"default": "ffmpeg_pipe"}),
                "video_codec": (list(VIDEO_CODECS), {"default": "libx264"}),
                "video_crf": (
                    "INT",
                    {
                        "default": 19,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "Lower is higher quality. Default matches the reference VideoCombine CRF.",
                    },
                ),
                "resize_mode": (
                    list(RESIZE_MODES),
                    {
                        "default": "crop_to_size",
                        "tooltip": "crop_to_size matches the reference workflow. fit_inside keeps the whole frame with lower memory use.",
                    },
                ),
                "resize_filter": (
                    list(RESIZE_FILTERS),
                    {
                        "default": "lanczos",
                        "tooltip": "lanczos matches the reference workflow ImageResizeKJv2 upscale_method.",
                    },
                ),
                "max_width": ("STRING", {"default": "480"}),
                "max_height": ("STRING", {"default": "832"}),
                "target_width": ("STRING", {"default": "0"}),
                "target_height": ("STRING", {"default": "0"}),
                "separate_vocals": ("BOOLEAN", {"default": True}),
                "auto_normalize_fps": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Re-encode the source to constant target_fps and clone the last frame to match the audio length. Recommended ON: handles VFR / mis-tagged r_frame_rate sources, and lets a slightly-too-short video still render (the tail holds on the last frame). Turn OFF only when the source is already exact-length CFR and you want to skip the transcode.",
                    },
                ),
                "keep_intermediates": ("BOOLEAN", {"default": False}),
                "positive_prompt": ("STRING", {"default": DEFAULT_POSITIVE_PROMPT, "multiline": True}),
                "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE_PROMPT, "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "process"
    CATEGORY = "InfiniteTalk"
    OUTPUT_NODE = True

    def process(
        self,
        video_path,
        audio_path="",
        wan_model="",
        infinitetalk_model="",
        lora_model="none",
        vae_model="",
        text_encoder_model="",
        clip_vision_model="",
        wav2vec_model="TencentGameMate/chinese-wav2vec2-base",
        base_precision="fp16",
        quantization="fp8_e4m3fn_scaled",
        attention_mode="sageattn",
        vae_precision="bf16",
        text_precision="bf16",
        text_quantization="disabled",
        wav2vec_precision="fp16",
        wav2vec_load_device="main_device",
        target_fps=25.0,
        segment_seconds=20.0,
        frame_window_size=81,
        motion_frame=9,
        steps=5,
        cfg=1.0,
        shift=11.0,
        scheduler="dpm++_sde",
        seed=1,
        start_step=3,
        end_step=-1,
        denoise_strength=1.0,
        batched_cfg=False,
        rope_function="comfy",
        add_noise_to_samples=True,
        sampler_force_offload=True,
        block_swap=20,
        use_non_blocking=True,
        prefetch_blocks=1,
        lora_strength=1.0,
        audio_scale=1.0,
        audio_cfg_scale=1.0,
        normalize_loudness=True,
        clip_strength_1=1.0,
        clip_strength_2=1.0,
        clip_use_last_frame=False,
        encode_tiled_vae=False,
        decode_tiled_vae=False,
        tile_x=272,
        tile_y=272,
        tile_stride_x=144,
        tile_stride_y=128,
        filename_prefix="InfiniteTalk",
        output_path="",
        output_mode="ffmpeg_pipe",
        video_codec="libx264",
        video_crf=19,
        resize_mode="crop_to_size",
        resize_filter="lanczos",
        max_width="480",
        max_height="832",
        target_width="0",
        target_height="0",
        separate_vocals=True,
        auto_normalize_fps=True,
        keep_intermediates=False,
        positive_prompt=DEFAULT_POSITIVE_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    ):
        legacy_target_width = target_width
        legacy_target_height = target_height
        resize_mode = _normalize_resize_mode(resize_mode)
        resize_filter = _normalize_resize_filter(resize_filter)

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
        shift = _coerce_float_value(shift, "shift", 11.0, min_value=0.0, max_value=100.0)
        lora_strength = _coerce_float_value(
            lora_strength,
            "lora_strength",
            1.0,
            min_value=0.0,
            max_value=4.0,
        )
        start_step = _coerce_int_value(start_step, "start_step", 3, min_value=0, max_value=20)
        end_step = _coerce_int_value(end_step, "end_step", -1, min_value=-1, max_value=1000)
        denoise_strength = _coerce_float_value(
            denoise_strength,
            "denoise_strength",
            1.0,
            min_value=0.0,
            max_value=1.0,
        )
        seed = _coerce_int_value(seed, "seed", 1, min_value=0, max_value=2**31 - 1, allow_randomize=True)
        block_swap = _coerce_int_value(block_swap, "block_swap", 20, min_value=0, max_value=40)
        prefetch_blocks = _coerce_int_value(prefetch_blocks, "prefetch_blocks", 1, min_value=0, max_value=16)
        tile_x = _coerce_int_value(tile_x, "tile_x", 272, min_value=64, max_value=2048)
        tile_y = _coerce_int_value(tile_y, "tile_y", 272, min_value=64, max_value=2048)
        tile_stride_x = _coerce_int_value(tile_stride_x, "tile_stride_x", 144, min_value=32, max_value=2048)
        tile_stride_y = _coerce_int_value(tile_stride_y, "tile_stride_y", 128, min_value=32, max_value=2048)
        audio_scale = _coerce_float_value(audio_scale, "audio_scale", 1.0, min_value=0.0, max_value=10.0)
        audio_cfg_scale = _coerce_float_value(audio_cfg_scale, "audio_cfg_scale", 1.0, min_value=0.0, max_value=10.0)
        clip_strength_1 = _coerce_float_value(clip_strength_1, "clip_strength_1", 1.0, min_value=0.0, max_value=4.0)
        clip_strength_2 = _coerce_float_value(clip_strength_2, "clip_strength_2", 1.0, min_value=0.0, max_value=4.0)
        video_codec = _normalize_video_codec(video_codec)
        output_mode = _normalize_output_mode(output_mode)
        video_crf = _coerce_int_value(video_crf, "video_crf", 19, min_value=0, max_value=51)
        clip_use_last_frame = bool(clip_use_last_frame)
        audio_path = str(audio_path or "").strip()
        filename_prefix = str(filename_prefix or "InfiniteTalk").strip()
        if filename_prefix.lower() == "randomize" or filename_prefix.isdigit():
            log.warning(
                "[InfiniteTalk] Suspicious filename_prefix=%r, fallback to default",
                filename_prefix,
            )
            filename_prefix = "InfiniteTalk"
        if audio_path.lower() == "randomize" or audio_path.isdigit():
            log.warning(
                "[InfiniteTalk] Suspicious audio_path=%r, rejecting input",
                audio_path,
            )
            audio_path = ""
        output_path = str(output_path or "")

        resolved_video_path = resolve_user_path(video_path, "video_path")
        resolved_audio_path = resolve_user_path(audio_path, "audio_path")
        frame_window_size = normalize_frame_window_size(frame_window_size)
        motion_frame = int(motion_frame)

        if motion_frame >= frame_window_size:
            raise ValueError("motion_frame must be smaller than frame_window_size")

        final_output_path, _output_filename = get_output_video_path(filename_prefix, output_path)
        final_output_path = os.path.abspath(final_output_path)
        work_dir = os.path.join(
            os.path.dirname(final_output_path),
            f".{Path(final_output_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}_infinitetalk",
        )
        os.makedirs(work_dir, exist_ok=True)

        engine = None
        output_sink = None
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
            max_width = _coerce_int_value(max_width, "max_width", 480, min_value=16, max_value=4096)
            max_height = _coerce_int_value(max_height, "max_height", 832, min_value=16, max_value=4096)
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
            target_width, target_height = resolve_target_size(
                video_info["width"],
                video_info["height"],
                max_width=max_width,
                max_height=max_height,
                resize_mode=resize_mode,
            )
            log.info(
                "[InfiniteTalk] Auto target size %sx%s (max %sx%s, resize_mode=%s, aligned to 16)",
                target_width,
                target_height,
                max_width,
                max_height,
                resize_mode,
            )
            log.info("[InfiniteTalk] Work directory: %s", work_dir)
            log.info(
                "[InfiniteTalk] Inputs video=%s audio=%s output=%s keep_intermediates=%s",
                resolved_video_path,
                resolved_audio_path,
                final_output_path,
                bool(keep_intermediates),
            )
            log.info(
                "[InfiniteTalk] Models wan=%s infinitetalk=%s lora=%s vae=%s text=%s clip=%s wav2vec=%s",
                wan_model,
                infinitetalk_model,
                lora_model,
                vae_model,
                text_encoder_model,
                clip_vision_model,
                wav2vec_model,
            )
            _warn_reference_model_mismatch("lora_model", lora_model)
            _warn_reference_model_mismatch("vae_model", vae_model)
            _warn_reference_model_mismatch("text_encoder_model", text_encoder_model)
            log.info(
                "[InfiniteTalk] Runtime options attention=%s quantization=%s precision=%s vae_precision=%s text_precision=%s scheduler=%s steps=%s cfg=%.3f shift=%.3f seed=%s start_step=%s end_step=%s",
                attention_mode,
                quantization,
                base_precision,
                vae_precision,
                text_precision,
                scheduler,
                steps,
                cfg,
                shift,
                seed,
                start_step,
                end_step,
            )
            log.info(
                "[InfiniteTalk] Segment options target=%sx%s fps=%.3f resize_mode=%s resize_filter=%s segment_seconds=%.3f frame_window=%s motion_frame=%s block_swap=%s tiled_encode=%s tiled_decode=%s output_mode=%s video_codec=%s video_crf=%s",
                target_width,
                target_height,
                target_fps,
                resize_mode,
                resize_filter,
                segment_seconds,
                frame_window_size,
                motion_frame,
                block_swap,
                bool(encode_tiled_vae),
                bool(decode_tiled_vae),
                output_mode,
                video_codec,
                video_crf,
            )

            prepared_audio_path = os.path.join(work_dir, "input_audio.wav")
            with _timed_log(
                "Prepare audio",
                source="audio_path",
                path=resolved_audio_path,
            ):
                extract_audio_to_wav(resolved_audio_path, prepared_audio_path)

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
            video_duration = float(video_info["duration"])
            if audio_duration <= 0 or video_duration <= 0:
                raise RuntimeError(
                    "Resolved input duration is 0. Check the input video and audio."
                )

            output_duration = float(audio_duration)
            total_frames = max(1, int(round(output_duration * float(target_fps))))

            # Always normalize the source to CFR + pad to audio length when
            # auto_normalize_fps is on. This handles two failure modes in one
            # pass: (1) phone/web videos that lie about r_frame_rate and would
            # otherwise decode to fewer frames than total_frames; (2) sources
            # whose true content is shorter than the audio - the tail is
            # extended by cloning the last frame so v2v animation falls back
            # to a still on the held frame instead of erroring out.
            if bool(auto_normalize_fps):
                normalized_path = os.path.join(work_dir, "normalized_source.mp4")
                if audio_duration > video_duration + 0.05:
                    log.warning(
                        "[InfiniteTalk] Source video (%.3fs) is shorter than the audio (%.3fs). "
                        "Last frame will be cloned for the trailing %.3fs.",
                        video_duration,
                        audio_duration,
                        audio_duration - video_duration,
                    )
                with _timed_log(
                    "Normalize source to CFR",
                    src=resolved_video_path,
                    dst=normalized_path,
                    target_fps=float(target_fps),
                    pad_to_seconds=f"{output_duration:.3f}",
                ):
                    normalize_source_video_to_cfr(
                        resolved_video_path,
                        normalized_path,
                        target_fps=float(target_fps),
                        video_codec=video_codec,
                        video_crf=18,
                        pad_to_seconds=output_duration,
                    )
                resolved_video_path = normalized_path
                normalized_info = get_video_info(resolved_video_path)
                normalized_nb_frames = int(normalized_info.get("nb_frames") or 0)
                log.info(
                    "[InfiniteTalk] Normalized video: %.3fs, nb_frames=%s (target_fps=%s, total_frames=%s)",
                    normalized_info["duration"],
                    normalized_nb_frames,
                    float(target_fps),
                    total_frames,
                )
                if normalized_nb_frames and normalized_nb_frames < total_frames:
                    raise RuntimeError(
                        f"Normalized video produced only {normalized_nb_frames} frames "
                        f"at {target_fps}fps but {total_frames} are required. "
                        "tpad clone failed; please file a bug with the source video."
                    )
            else:
                # No auto_normalize_fps: enforce the strict contract so the
                # streaming encoder doesn't silently underflow.
                if audio_duration > video_duration + 0.05:
                    raise RuntimeError(
                        "Source video is shorter than the audio: "
                        f"video={video_duration:.3f}s, audio={audio_duration:.3f}s. "
                        "Enable auto_normalize_fps to auto-pad with the last frame, "
                        "trim the audio, or extend the video."
                    )
            # `segment_seconds` is now only used to size the streaming source
            # latent encode chunks. We keep the historical name so existing
            # workflows do not break.
            chunk_frames = normalize_frame_window_size(
                max(int(float(segment_seconds) * float(target_fps)), int(frame_window_size))
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
                        "use_non_blocking": bool(use_non_blocking),
                        "prefetch_blocks": prefetch_blocks,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "base_precision": base_precision,
                        "quantization": quantization,
                        "attention_mode": attention_mode,
                        "vae_precision": vae_precision,
                        "text_precision": text_precision,
                        "text_quantization": text_quantization,
                        "wav2vec_precision": wav2vec_precision,
                        "wav2vec_load_device": wav2vec_load_device,
                        "audio_scale": audio_scale,
                        "audio_cfg_scale": audio_cfg_scale,
                        "clip_strength_1": clip_strength_1,
                        "clip_strength_2": clip_strength_2,
                        "clip_use_last_frame": clip_use_last_frame,
                        "scheduler": scheduler,
                        "shift": shift,
                        "end_step": end_step,
                        "denoise_strength": denoise_strength,
                        "batched_cfg": bool(batched_cfg),
                        "rope_function": rope_function,
                        "add_noise_to_samples": bool(add_noise_to_samples),
                        "sampler_force_offload": bool(sampler_force_offload),
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tile_stride_x": tile_stride_x,
                        "tile_stride_y": tile_stride_y,
                        "encode_tiled_vae": bool(encode_tiled_vae),
                        "decode_tiled_vae": bool(decode_tiled_vae),
                        "normalize_loudness": bool(normalize_loudness),
                    }
                )

            with _timed_log(
                "Global audio load",
                duration_sec=f"{output_duration:.3f}",
                sample_rate=16000,
            ):
                global_audio = load_audio_segment(
                    prepared_audio_path,
                    start_sec=0.0,
                    duration_sec=output_duration,
                    target_sr=16000,
                )
            with _timed_log(
                "Global wav2vec",
                frames=total_frames,
                fps=f"{float(target_fps):.3f}",
            ):
                global_multitalk_embeds, _, global_actual_frames = engine._build_multitalk_embeds(
                    audio_input=global_audio,
                    num_frames=total_frames,
                    fps=float(target_fps),
                )
            global_actual_frames = max(1, min(total_frames, int(global_actual_frames)))
            if global_actual_frames != total_frames:
                log.info(
                    "[InfiniteTalk] Global wav2vec limited total frames %s -> %s",
                    total_frames,
                    global_actual_frames,
                )
                total_frames = global_actual_frames

            log.info(
                "[InfiniteTalk] Streaming source latent encode in chunks of %s frames (total=%s)",
                chunk_frames,
                total_frames,
            )

            full_source_latent, first_frame_pixels, last_frame_pixels = _stream_encode_full_source_latent(
                engine=engine,
                video_path=resolved_video_path,
                total_frames=total_frames,
                target_width=int(target_width),
                target_height=int(target_height),
                target_fps=float(target_fps),
                resize_mode=resize_mode,
                resize_filter=resize_filter,
                chunk_frames=int(chunk_frames),
            )
            log.info(
                "[InfiniteTalk] Full source latent ready shape=%s frames=%s",
                tuple(full_source_latent.shape),
                total_frames,
            )

            with _timed_log(
                "CLIP vision encode (reference frames)",
                use_last_frame=bool(clip_use_last_frame),
            ):
                # When clip_use_last_frame is True we feed image_1=first,
                # image_2=last so CLIP captures the source video's overall
                # appearance. Otherwise image_2 stays None to match the
                # reference workflow (single-frame CLIP context).
                if clip_use_last_frame and last_frame_pixels is not None:
                    clip_input_frames = torch.cat([first_frame_pixels, last_frame_pixels], dim=0)
                else:
                    clip_input_frames = first_frame_pixels
                clip_embeds, clip_first_frame = engine.encode_clip_vision_for_first_frame(
                    clip_input_frames
                )

            frames_dir = os.path.join(work_dir, "rendered_frames")
            render_output_dir = frames_dir if output_mode == "png_sequence" else ""
            if output_mode == "png_sequence":
                os.makedirs(frames_dir, exist_ok=True)
            else:
                with _timed_log(
                    "Open ffmpeg pipe output",
                    output=final_output_path,
                    duration_sec=f"{output_duration:.3f}",
                    video_codec=video_codec,
                ):
                    output_sink = FfmpegPipeSink(
                        audio_path=prepared_audio_path,
                        output_path=final_output_path,
                        target_width=int(target_width),
                        target_height=int(target_height),
                        target_fps=float(target_fps),
                        duration=output_duration,
                        video_codec=video_codec,
                        video_crf=video_crf,
                    )

            with _timed_log(
                "Full render (single sampler, sliding window inside)",
                input_frames=total_frames,
                frame_window_size=int(frame_window_size),
                motion_frame=motion_frame,
                steps=int(steps),
                cfg=float(cfg),
                seed=int(seed),
                output_mode=output_mode,
            ):
                render_result = engine.render_full_video(
                    first_frame=clip_first_frame,
                    clip_embeds=clip_embeds,
                    source_latent=full_source_latent,
                    multitalk_embeds=global_multitalk_embeds,
                    actual_num_frames=total_frames,
                    fps=float(target_fps),
                    frame_window_size=int(frame_window_size),
                    motion_frame=int(motion_frame),
                    steps=int(steps),
                    cfg_scale=float(cfg),
                    seed=int(seed),
                    output_dir=render_output_dir,
                    output_sink=output_sink,
                    start_step=int(start_step),
                    target_width=int(target_width),
                    target_height=int(target_height),
                )

            if output_mode == "ffmpeg_pipe":
                with _timed_log(
                    "Final encode (ffmpeg pipe finalize + audio)",
                    output=final_output_path,
                    duration_sec=f"{output_duration:.3f}",
                ):
                    output_sink.close()
                    output_sink = None
            else:
                rendered_frames_dir = str(render_result.get("output_path", "") or frames_dir)
                with _timed_log(
                    "Final encode (PNG sequence -> mp4 + audio)",
                    frames_dir=rendered_frames_dir,
                    output=final_output_path,
                    duration_sec=f"{output_duration:.3f}",
                ):
                    _encode_full_video_with_audio(
                        rendered_frames_dir,
                        prepared_audio_path,
                        final_output_path,
                        target_fps=float(target_fps),
                        duration=output_duration,
                        video_codec=video_codec,
                        video_crf=video_crf,
                    )
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
                ),
            }
        finally:
            if output_sink is not None:
                output_sink.abort()
            if engine is not None:
                engine.unload()
            if _should_cleanup_work_dir(keep_intermediates):
                shutil.rmtree(work_dir, ignore_errors=True)


class InfiniteTalkVideoSyncPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "segment_seconds": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "tooltip": "Hint for inner sampler-window framing only. The pipeline now uses one continuous multitalk_loop instead of outer segment splicing.",
                    },
                ),
                "target_fps": ("FLOAT", {"default": 25.0}),
                "motion_frame": ("INT", {"default": 9}),
            },
            "optional": {
                "frame_window_size": ("INT", {"default": 81, "min": 5, "max": 241, "step": 4}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "fps", "duration", "num_segments", "info_text")
    FUNCTION = "preview"
    CATEGORY = "InfiniteTalk"

    def preview(self, video_path, segment_seconds, target_fps, motion_frame, frame_window_size=81):
        try:
            resolved_video_path = resolve_user_path(video_path, "video_path")
        except Exception as exc:
            return (0, 0, 0.0, 0.0, 0, str(exc))

        target_fps = _coerce_float_value(target_fps, "target_fps", 25.0, min_value=1.0, max_value=60.0)
        segment_seconds = _coerce_float_value(
            segment_seconds,
            "segment_seconds",
            20.0,
            min_value=4.0,
            max_value=120.0,
        )
        motion_frame = _coerce_int_value(
            motion_frame,
            "motion_frame",
            9,
            min_value=1,
            max_value=240,
        )
        frame_window_size = normalize_frame_window_size(
            _coerce_int_value(
                frame_window_size,
                "frame_window_size",
                81,
                min_value=5,
                max_value=241,
            )
        )

        info = get_video_info(resolved_video_path)
        total_frames = max(1, int(info["duration"] * float(target_fps)))
        segment_frames = max(int(float(segment_seconds) * float(target_fps)), int(frame_window_size))
        if motion_frame >= segment_frames:
            clamped_motion = max(1, segment_frames - 1)
            log.warning(
                "[InfiniteTalk] motion_frame=%s >= segment_frames=%s, clamped to %s for preview",
                motion_frame,
                segment_frames,
                clamped_motion,
            )
            motion_frame = clamped_motion
        stride = max(1, frame_window_size - int(motion_frame))
        num_segments = max(1, int(math.ceil(max(total_frames - frame_window_size, 0) / float(stride))) + 1)

        ram_window_mb = (
            frame_window_size * info["width"] * info["height"] * 3 * 4 / 1024 / 1024
        )

        text = (
            f"source: {info['width']}x{info['height']} @ {info['fps']:.3f}fps\n"
            f"duration: {info['duration']:.3f}s ({total_frames} frames @ {target_fps}fps)\n"
            f"segments: {num_segments} inner sampler windows ({frame_window_size} frames each, {motion_frame} overlap)\n"
            f"estimated frame RAM/window: {ram_window_mb:.0f} MB"
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
    "InfiniteTalkVideoSyncPreview": InfiniteTalkVideoSyncPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteTalkVideoPathNode": "InfiniteTalk Lip Sync (Video Path)",
    "InfiniteTalkVideoSyncPreview": "InfiniteTalk Video Info Preview",
}

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "_infinitetalk_video_sync_under_test"


def install_comfy_stubs():
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.device = lambda value: value
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    sys.modules["torch"] = torch

    numpy = types.ModuleType("numpy")
    sys.modules["numpy"] = numpy

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(ROOT / "models")
    folder_paths.get_output_directory = lambda: str(ROOT / "output")
    folder_paths.get_filename_list = lambda category: {
        "diffusion_models": [
            "wan2.1/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
            "wan2.1/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
        ],
        "loras": ["lightx2v/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"],
        "vae": ["wan_2.1_vae.safetensors"],
        "text_encoders": ["umt5-xxl-enc-fp8_e4m3fn.safetensors"],
        "clip_vision": ["clip_vision_h.safetensors"],
    }.get(category, [])
    folder_paths.get_save_image_path = lambda prefix, output_root, *args: (
        str(ROOT / "output"),
        prefix,
        1,
        "",
        prefix,
    )
    folder_paths.get_full_path = lambda category, name: str(ROOT / "models" / name)
    folder_paths.get_full_path_or_raise = folder_paths.get_full_path
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.throw_exception_if_processing_interrupted = lambda: None
    model_management.soft_empty_cache = lambda: None
    comfy.model_management = model_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management


def import_under_test(module_name):
    install_comfy_stubs()
    for name in list(sys.modules):
        if name == PACKAGE_NAME or name.startswith(PACKAGE_NAME + "."):
            sys.modules.pop(name)
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{module_name}")


class NodeBehaviorTests(unittest.TestCase):
    def test_input_defaults_follow_portrait_workflow_size(self):
        nodes = import_under_test("nodes")

        input_types = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()

        self.assertEqual("480", input_types["optional"]["max_width"][1]["default"])
        self.assertEqual("832", input_types["optional"]["max_height"][1]["default"])
        self.assertEqual("crop_to_size", input_types["optional"]["resize_mode"][1]["default"])
        self.assertEqual("lanczos", input_types["optional"]["resize_filter"][1]["default"])
        self.assertEqual("", input_types["optional"]["positive_prompt"][1]["default"])
        self.assertIn("细节模糊不清", input_types["optional"]["negative_prompt"][1]["default"])
        self.assertFalse(input_types["optional"]["clip_use_last_frame"][1]["default"])

    def test_default_resize_matches_reference_workflow_crop(self):
        nodes = import_under_test("nodes")

        self.assertEqual(
            (480, 832),
            nodes.resolve_target_size(1920, 1080, 480, 832, "crop_to_size"),
        )
        video_filter = nodes.build_video_filter(480, 832, 25.0, "crop_to_size", "lanczos")

        self.assertIn("fps=25.0", video_filter)
        self.assertIn("flags=lanczos", video_filter)
        self.assertIn("force_original_aspect_ratio=increase", video_filter)
        self.assertIn("crop=480:832", video_filter)

    def test_fit_inside_resize_keeps_old_memory_saving_behavior(self):
        nodes = import_under_test("nodes")

        self.assertEqual(
            (480, 256),
            nodes.resolve_target_size(1920, 1080, 480, 832, "fit_inside"),
        )
        video_filter = nodes.build_video_filter(480, 256, 25.0, "fit_inside", "lanczos")

        self.assertEqual("fps=25.0,scale=480:256:flags=lanczos", video_filter)

    def test_node_contract_uses_path_inputs_and_single_video_path_output(self):
        nodes = import_under_test("nodes")

        input_types = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()

        self.assertEqual({"video_path", "audio_path"}, set(input_types["required"]))
        self.assertEqual(("STRING",), nodes.InfiniteTalkVideoPathNode.RETURN_TYPES)
        self.assertEqual(("video_path",), nodes.InfiniteTalkVideoPathNode.RETURN_NAMES)

    def test_advanced_workflow_parameters_are_exposed(self):
        nodes = import_under_test("nodes")

        optional = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()["optional"]

        expected_keys = {
            "wan_model",
            "infinitetalk_model",
            "lora_model",
            "vae_model",
            "text_encoder_model",
            "clip_vision_model",
            "wav2vec_model",
            "base_precision",
            "quantization",
            "attention_mode",
            "vae_precision",
            "text_precision",
            "text_quantization",
            "wav2vec_precision",
            "wav2vec_load_device",
            "target_fps",
            "segment_seconds",
            "frame_window_size",
            "motion_frame",
            "steps",
            "cfg",
            "shift",
            "scheduler",
            "start_step",
            "end_step",
            "denoise_strength",
            "batched_cfg",
            "rope_function",
            "add_noise_to_samples",
            "sampler_force_offload",
            "block_swap",
            "use_non_blocking",
            "prefetch_blocks",
            "lora_strength",
            "audio_scale",
            "audio_cfg_scale",
            "normalize_loudness",
            "clip_strength_1",
            "clip_strength_2",
            "clip_use_last_frame",
            "encode_tiled_vae",
            "decode_tiled_vae",
            "tile_x",
            "tile_y",
            "tile_stride_x",
            "tile_stride_y",
            "separate_vocals",
            "keep_intermediates",
            "positive_prompt",
            "negative_prompt",
            "filename_prefix",
            "output_path",
            "video_codec",
            "video_crf",
            "resize_mode",
            "resize_filter",
        }
        self.assertFalse(expected_keys - set(optional))

    def test_default_video_encoding_matches_reference_workflow(self):
        nodes = import_under_test("nodes")
        optional = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()["optional"]

        self.assertEqual("libx264", optional["video_codec"][1]["default"])
        self.assertEqual(19, optional["video_crf"][1]["default"])
        self.assertEqual(
            ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"],
            nodes.get_ffmpeg_video_encode_args("libx264", 19),
        )

    def test_default_models_prefer_reference_workflow_files(self):
        nodes = import_under_test("nodes")
        original_get_filename_list = nodes.folder_paths.get_filename_list
        nodes.folder_paths.get_filename_list = lambda category: {
            "diffusion_models": [
                "wan2.1/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                "wan2.1/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
            ],
            "loras": [
                "lightx2v/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                "lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            ],
            "vae": [
                "wan_2.1_vae.safetensors",
                "Wan2_1_VAE_bf16.safetensors",
            ],
            "text_encoders": [
                "umt5-xxl-enc-fp8_e4m3fn.safetensors",
                "umt5-xxl-enc-bf16.safetensors",
            ],
            "clip_vision": ["clip_vision_h.safetensors"],
        }.get(category, [])
        try:
            optional = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()["optional"]
        finally:
            nodes.folder_paths.get_filename_list = original_get_filename_list

        self.assertEqual(
            "lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            optional["lora_model"][1]["default"],
        )
        self.assertEqual("Wan2_1_VAE_bf16.safetensors", optional["vae_model"][1]["default"])
        self.assertEqual("umt5-xxl-enc-bf16.safetensors", optional["text_encoder_model"][1]["default"])

    def test_default_lora_prefers_rank128_reference_over_rank64(self):
        nodes = import_under_test("nodes")
        original_get_filename_list = nodes.folder_paths.get_filename_list
        nodes.folder_paths.get_filename_list = lambda category: {
            "diffusion_models": [
                "wan2.1/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                "wan2.1/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
            ],
            "loras": [
                "lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                "lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            ],
            "vae": ["wan_2.1_vae.safetensors"],
            "text_encoders": ["umt5-xxl-enc-fp8_e4m3fn.safetensors"],
            "clip_vision": ["clip_vision_h.safetensors"],
        }.get(category, [])
        try:
            optional = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()["optional"]
        finally:
            nodes.folder_paths.get_filename_list = original_get_filename_list

        self.assertEqual(
            "lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            optional["lora_model"][1]["default"],
        )

    def test_sampler_defaults_match_reference_workflow(self):
        nodes = import_under_test("nodes")
        optional = nodes.InfiniteTalkVideoPathNode.INPUT_TYPES()["optional"]

        self.assertEqual(5, optional["steps"][1]["default"])
        self.assertEqual("dpm++_sde", optional["scheduler"][1]["default"])
        self.assertEqual(3, optional["start_step"][1]["default"])
        self.assertEqual(20, optional["block_swap"][1]["default"])
        self.assertEqual("disabled", optional["text_quantization"][1]["default"])
        self.assertFalse(optional["decode_tiled_vae"][1]["default"])
        self.assertTrue(optional["separate_vocals"][1]["default"])

    def test_output_path_uses_comfy_style_mp4_counter(self):
        nodes = import_under_test("nodes")

        with tempfile.TemporaryDirectory() as output_dir:
            Path(output_dir, "InfiniteTalk_00001.mp4").touch()
            Path(output_dir, "InfiniteTalk_00002.png").touch()
            original_output_directory = nodes.folder_paths.get_output_directory
            original_save_image_path = nodes.folder_paths.get_save_image_path
            nodes.folder_paths.get_output_directory = lambda: output_dir
            nodes.folder_paths.get_save_image_path = lambda prefix, root, *args: (
                root,
                prefix,
                1,
                "",
                prefix,
            )
            try:
                path, filename = nodes.get_output_video_path("InfiniteTalk", "")
            finally:
                nodes.folder_paths.get_output_directory = original_output_directory
                nodes.folder_paths.get_save_image_path = original_save_image_path

        self.assertEqual("InfiniteTalk_00003.mp4", filename)
        self.assertEqual(str(Path(output_dir, "InfiniteTalk_00003.mp4")), path)

    def test_explicit_output_file_strips_existing_counter_before_incrementing(self):
        nodes = import_under_test("nodes")

        with tempfile.TemporaryDirectory() as output_dir:
            Path(output_dir, "InfiniteTalk_00001.mp4").touch()
            path, filename = nodes.get_output_video_path(
                "Ignored",
                str(Path(output_dir, "InfiniteTalk_00001.mp4")),
            )

        self.assertEqual("InfiniteTalk_00002.mp4", filename)
        self.assertEqual(str(Path(output_dir, "InfiniteTalk_00002.mp4")), path)

    def test_preview_handles_motion_frame_equal_to_segment_frames(self):
        nodes = import_under_test("nodes")
        original_resolve_user_path = nodes.resolve_user_path
        original_get_video_info = nodes.get_video_info
        nodes.resolve_user_path = lambda path, input_name: "video.mp4"
        nodes.get_video_info = lambda path: {
            "width": 480,
            "height": 832,
            "fps": 25.0,
            "duration": 10.0,
            "nb_frames": 250,
        }
        try:
            result = nodes.InfiniteTalkVideoSyncPreview().preview(
                "video.mp4",
                segment_seconds=4.0,
                target_fps=25.0,
                motion_frame=100,
            )
        finally:
            nodes.resolve_user_path = original_resolve_user_path
            nodes.get_video_info = original_get_video_info

        self.assertGreater(result[4], 0)
        self.assertIn("segments:", result[5])

    def test_cleanup_policy_removes_work_dir_unless_user_keeps_intermediates(self):
        nodes = import_under_test("nodes")

        self.assertTrue(nodes._should_cleanup_work_dir(False))
        self.assertFalse(nodes._should_cleanup_work_dir(True))

    def test_video_pipe_command_is_seek_free_and_cfr(self):
        nodes = import_under_test("nodes")

        original_get_ffmpeg = nodes.get_ffmpeg_path
        nodes.get_ffmpeg_path = lambda: "ffmpeg"
        try:
            command = nodes._build_full_pipe_command(
                "video.mp4",
                target_width=16,
                target_height=16,
                target_fps=25.0,
                resize_mode="crop_to_size",
                resize_filter="lanczos",
                max_total_frames=831,
            )
        finally:
            nodes.get_ffmpeg_path = original_get_ffmpeg

        joined = " ".join(command)
        # Streaming pipeline: no -ss anywhere, no select filter, single -i,
        # CFR locked, frames capped to max_total_frames.
        self.assertNotIn("-ss", joined)
        self.assertNotIn("select=", joined)
        self.assertIn("-vsync cfr", joined)
        self.assertIn("-frames:v 831", joined)
        self.assertIn("-pix_fmt rgb24", joined)
        self.assertIn("-f rawvideo", joined)
        self.assertEqual("pipe:1", command[-1])

    def test_chunk_frames_is_normalized_to_4n_plus_1(self):
        nodes = import_under_test("nodes")

        # 41 is already 4*10+1
        self.assertEqual(41, nodes.normalize_frame_window_size(41))
        # 42 should round down to 41
        self.assertEqual(41, nodes.normalize_frame_window_size(42))
        # Below the floor of 5 still normalizes to 5
        self.assertEqual(5, nodes.normalize_frame_window_size(3))

    def test_preview_clamps_motion_frame_when_it_meets_or_exceeds_window(self):
        nodes = import_under_test("nodes")
        original_resolve_user_path = nodes.resolve_user_path
        original_get_video_info = nodes.get_video_info
        nodes.resolve_user_path = lambda path, input_name: "video.mp4"
        nodes.get_video_info = lambda path: {
            "width": 480,
            "height": 832,
            "fps": 25.0,
            "duration": 60.0,
            "nb_frames": 1500,
        }
        try:
            result = nodes.InfiniteTalkVideoSyncPreview().preview(
                "video.mp4",
                segment_seconds=20.0,
                target_fps=25.0,
                motion_frame=200,  # >= frame_window_size (default 81)
                frame_window_size=81,
            )
        finally:
            nodes.resolve_user_path = original_resolve_user_path
            nodes.get_video_info = original_get_video_info

        # Should not error; should report > 0 inner windows.
        self.assertGreater(result[4], 0)

    def test_stream_encode_rejects_video_shorter_than_audio(self):
        """The hard contract: if the source decodes fewer frames than the
        audio asks for, the encoder must raise with actionable advice rather
        than silently padding (which would break v2v motion fidelity)."""
        nodes = import_under_test("nodes")

        class _FakeChunk:
            def __init__(self, frames):
                self._frames = frames
                self.shape = (frames, 8, 8, 3)

            def __getitem__(self, key):
                start, stop, _ = (key.start or 0, key.stop or self._frames, 1)
                length = max(0, stop - start)
                if start < 0:
                    length = abs(start)
                return _FakeChunk(length)

            def clone(self):
                return _FakeChunk(self._frames)

        class _FakeLatent:
            def __init__(self, t_latent):
                self.shape = (1, 16, t_latent, 4, 4)

            def dim(self):
                return 5

            def contiguous(self):
                return self

        class _FakeEngine:
            def encode_source_latent(self, source_chunk, label=""):
                t_latent = (int(source_chunk.shape[0]) - 1) // 4 + 1
                return _FakeLatent(t_latent)

        # Stream yields 798 frames total but caller asked for 831 (mirrors the
        # mis-tagged 25fps video bug we hit in production).
        def fake_stream(*args, **kwargs):
            yield _FakeChunk(497)
            yield _FakeChunk(301)

        original_stream = nodes.stream_video_frame_chunks
        original_cat = getattr(nodes.torch, "cat", None)
        nodes.stream_video_frame_chunks = fake_stream
        nodes.torch.cat = lambda chunks, dim=0: chunks[0]
        try:
            with self.assertRaises(RuntimeError) as cm:
                nodes._stream_encode_full_source_latent(
                    engine=_FakeEngine(),
                    video_path="video.mp4",
                    total_frames=831,
                    target_width=8,
                    target_height=8,
                    target_fps=25.0,
                    resize_mode="crop_to_size",
                    resize_filter="lanczos",
                    chunk_frames=497,
                )
        finally:
            nodes.stream_video_frame_chunks = original_stream
            if original_cat is None:
                del nodes.torch.cat
            else:
                nodes.torch.cat = original_cat

        msg = str(cm.exception)
        self.assertIn("798", msg)
        self.assertIn("831", msg)
        self.assertIn("25", msg)

    def test_pipeline_no_longer_relies_on_outer_segment_concat(self):
        """Refactor invariant: process() must not call concat_segments_with_audio."""
        nodes = import_under_test("nodes")
        import inspect

        source = inspect.getsource(nodes.InfiniteTalkVideoPathNode.process)
        self.assertNotIn("concat_segments_with_audio", source)
        self.assertIn("_encode_full_video_with_audio", source)
        self.assertIn("_stream_encode_full_source_latent", source)
        self.assertIn("render_full_video", source)


class RuntimeBehaviorTests(unittest.TestCase):
    def test_sageattention_falls_back_to_comfy_when_unavailable(self):
        runtime = import_under_test("infinitetalk_runtime")

        self.assertEqual(
            "comfy",
            runtime.resolve_attention_mode("sageattn", module_available=lambda name: False),
        )
        self.assertEqual(
            "sageattn",
            runtime.resolve_attention_mode(
                "sageattn",
                module_available=lambda name: name == "sageattention",
            ),
        )

    def test_slice_multitalk_embeds_uses_global_frame_offsets(self):
        runtime = import_under_test("infinitetalk_runtime")
        audio_features = list(range(10))
        embeds = {
            "audio_features": [audio_features],
            "audio_scale": 1.0,
            "audio_cfg_scale": 1.0,
            "ref_target_masks": None,
        }

        sliced, actual_frames = runtime.slice_multitalk_embeds(embeds, 3, 4)

        self.assertEqual(4, actual_frames)
        self.assertEqual([3, 4, 5, 6], sliced["audio_features"][0])
        self.assertEqual(10, len(embeds["audio_features"][0]))

    def test_clip_reference_frames_match_single_image_workflow_by_default(self):
        runtime = import_under_test("infinitetalk_runtime")
        frames = ["frame0", "frame1", "frame2"]

        first_frame, second_frame = runtime.select_clip_reference_frames(frames, False)

        self.assertEqual(["frame0"], first_frame)
        self.assertIsNone(second_frame)

    def test_clip_reference_frames_can_use_last_frame_when_requested(self):
        runtime = import_under_test("infinitetalk_runtime")
        frames = ["frame0", "frame1", "frame2"]

        first_frame, second_frame = runtime.select_clip_reference_frames(frames, True)

        self.assertEqual(["frame0"], first_frame)
        self.assertEqual(["frame2"], second_frame)

    def test_engine_exposes_streaming_encode_and_full_render_apis(self):
        runtime = import_under_test("infinitetalk_runtime")

        self.assertTrue(hasattr(runtime.InfiniteTalkEngine, "encode_source_latent"))
        self.assertTrue(hasattr(runtime.InfiniteTalkEngine, "render_full_video"))
        self.assertTrue(hasattr(runtime.InfiniteTalkEngine, "encode_clip_vision_for_first_frame"))


class VendorPatchTests(unittest.TestCase):
    """The vendored multitalk_loop must keep the long-video latent on CPU
    until the active window is sliced out, otherwise streamed source latents
    would be reuploaded to the GPU each iteration.
    """

    def test_multitalk_loop_keeps_full_latent_on_cpu(self):
        loop_path = ROOT / "vendor" / "wanvideo_wrapper" / "multitalk" / "multitalk_loop.py"
        text = loop_path.read_text(encoding="utf-8")
        # The patched form: never call .to(noise) on the *full* tensor.
        self.assertIn("input_samples_cpu", text)
        self.assertNotIn(
            "input_samples = input_samples.squeeze(0).to(noise)",
            text,
        )


if __name__ == "__main__":
    unittest.main()

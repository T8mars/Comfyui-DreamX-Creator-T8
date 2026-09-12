import pytest
import torch

from comfy_nodes.nodes import (
    DreamXBundleLoader,
    DreamXFirstFrameAVLatent,
    DreamXRefinerLoader,
    DreamXSplitAVLatent,
)
from comfy_nodes.utils import snap_video_frames


class FakeWanVAE:
    def encode(self, images):
        return torch.zeros(
            (1, 48, 1, images.shape[1] // 16, images.shape[2] // 16),
            dtype=torch.float32,
        )


def test_first_frame_node_builds_native_nested_latent_and_metadata():
    image = torch.zeros((1, 720, 1280, 3), dtype=torch.float32)
    conditioning = [[torch.zeros(1, 512, 4096), {}]]
    output = DreamXFirstFrameAVLatent.execute(
        conditioning, conditioning, FakeWanVAE(), image,
        duration=5.0, fps=24.0, target_spatial_tokens=880, batch_size=1,
    )
    positive, negative, latent, width, height, frames, fps = output.result
    assert (width, height, frames) == (1248, 704, 117)
    assert fps == 24.0
    assert positive[0][1]["dreamx_fps"] == negative[0][1]["dreamx_fps"] == 24.0
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 48, 30, 44, 78)
    assert audio.shape == (1, 128, 244)
    assert latent["dreamx_audio_samples"] == 234000
    video_out, audio_out = DreamXSplitAVLatent.execute(latent).result
    assert video_out["samples"].shape == video.shape
    assert audio_out["samples"].shape == audio.shape


def test_video_frames_floor_to_released_pipeline_length():
    assert snap_video_frames(5.0, 24.0) == 117
    assert snap_video_frames(2.0, 8.0) == 13
    assert snap_video_frames(0.25, 1.0) == 1


def test_first_frame_node_rejects_unsafe_multi_video_batch():
    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    conditioning = [[torch.zeros(1, 8, 16), {}]]
    with pytest.raises(ValueError, match="queue batching"):
        DreamXFirstFrameAVLatent.execute(
            conditioning, conditioning, FakeWanVAE(), image,
            duration=1.0, fps=4.0, target_spatial_tokens=64, batch_size=2,
        )


def test_refiner_loader_passes_explicit_memory_controls(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("comfy_nodes.nodes.resolve_and_validate", lambda _root: tmp_path)
    monkeypatch.setattr(
        "comfy_nodes.nodes.load_refiner",
        lambda *args: calls.append(args) or "refiner",
    )
    output = DreamXRefinerLoader.execute("auto", "bfloat16", 1, 3)
    assert output.result == ("refiner",)
    assert calls == [(tmp_path, torch.bfloat16, 1, 3)]


def test_bundle_keeps_released_audio_vae_in_float32(monkeypatch, tmp_path):
    audio_calls = []
    monkeypatch.setattr("comfy_nodes.nodes.resolve_and_validate", lambda _root: tmp_path)
    monkeypatch.setattr("comfy_nodes.nodes.load_model_patcher", lambda *_args: "model")
    monkeypatch.setattr("comfy_nodes.nodes.load_text_encoder", lambda *_args: "clip")
    monkeypatch.setattr("comfy_nodes.nodes.load_wan_vae", lambda *_args: "vae")
    monkeypatch.setattr(
        "comfy_nodes.nodes.load_audio_vae",
        lambda _root, dtype: audio_calls.append(dtype) or "audio_vae",
    )

    output = DreamXBundleLoader.execute("auto", "bfloat16")

    assert output.result == ("model", "clip", "vae", "audio_vae")
    assert audio_calls == [torch.float32]

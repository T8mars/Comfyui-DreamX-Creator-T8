import pytest
import torch

from comfy_nodes.nodes import (
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
    assert (width, height, frames) == (1248, 704, 121)
    assert fps == 24.0
    assert positive[0][1]["dreamx_fps"] == negative[0][1]["dreamx_fps"] == 24.0
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 48, 31, 44, 78)
    assert audio.shape == (1, 128, 253)
    assert latent["dreamx_audio_samples"] == 242000
    video_out, audio_out = DreamXSplitAVLatent.execute(latent).result
    assert video_out["samples"].shape == video.shape
    assert audio_out["samples"].shape == audio.shape


def test_video_frames_snap_to_nearest_vae_length():
    assert snap_video_frames(5.0, 24.0) == 121
    assert snap_video_frames(2.0, 8.0) == 17
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

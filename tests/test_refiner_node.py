from contextlib import nullcontext
from types import SimpleNamespace

import torch

import comfy.model_management

from comfy_nodes.refiner import DreamXRefinerHandle, run_refiner
from comfy_nodes.nodes import DreamXCausalRefine


class FakeWanVAE:
    def __init__(self):
        self.encoded_frames = None
        self.encode_tiles = None
        self.decode_tiles = None
        self.encode_inference = False
        self.decode_inference = False

    def encode_tiled(self, images, **kwargs):
        self.encoded_frames = images.shape[0]
        self.encode_tiles = kwargs
        self.encode_inference = torch.is_inference_mode_enabled()
        return torch.zeros(
            1, 48, ((images.shape[0] - 1) // 4) + 1,
            images.shape[1] // 16, images.shape[2] // 16,
        )

    def spacial_compression_decode(self):
        return 16

    def temporal_compression_decode(self):
        return 4

    def decode_tiled(self, latent, **kwargs):
        self.decode_tiles = kwargs
        self.decode_inference = torch.is_inference_mode_enabled()
        frames = (latent.shape[2] - 1) * 4 + 1
        return torch.zeros(frames, latent.shape[3] * 16, latent.shape[4] * 16, 3)


class FakeWindowAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_window_attn = True
        self.window_chunk = 4


class FakeRefinerPipeline(torch.nn.Module):
    def __init__(self, fail_once=False):
        super().__init__()
        model = torch.nn.Module()
        model.window_chunk = 4
        model.attn = FakeWindowAttention()
        self.generator = SimpleNamespace(model=model)
        self.num_frame_per_block = 3
        self.kv_caches = None
        self.calls = 0
        self.fail_once = fail_once

    def inference_sr(self, lr_latent, noise, **kwargs):
        self.calls += 1
        self.kv_caches = [torch.ones(1)]
        if self.fail_once and self.calls == 1:
            raise torch.OutOfMemoryError("synthetic OOM")
        return noise


def test_refiner_pads_before_vae_crops_source_frames_and_retries_oom(monkeypatch):
    load_calls = []
    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda models, **kwargs: load_calls.append((models, kwargs)),
    )
    monkeypatch.setattr(comfy.model_management, "cuda_device_context", lambda *_: nullcontext())
    monkeypatch.setattr(comfy.model_management, "soft_empty_cache", lambda *_, **__: None)
    pipeline = FakeRefinerPipeline(fail_once=True)
    refiner = DreamXRefinerHandle(
        pipeline=pipeline,
        patcher=SimpleNamespace(load_device=torch.device("cpu")),
        dtype=torch.float32,
        window_chunk=4,
    )
    vae = FakeWanVAE()
    images = torch.zeros(10, 64, 96, 3)
    positive = [[torch.zeros(1, 512, 4096), {}]]
    output_images, output_latent = run_refiner(
        refiner, vae, positive, images, 2, 7, 0.6251, True, 1.0, 0, -1,
    )
    assert vae.encoded_frames == 13
    assert vae.encode_tiles == {
        "tile_x": 512, "tile_y": 512, "overlap": 64,
        "tile_t": 64, "overlap_t": 8,
    }
    assert vae.decode_tiles == {
        "tile_x": 32, "tile_y": 32, "overlap": 4,
        "tile_t": 16, "overlap_t": 2,
    }
    assert vae.encode_inference is True
    assert vae.decode_inference is True
    assert output_images.shape == (10, 128, 192, 3)
    assert output_latent["samples"].shape == (1, 48, 4, 8, 12)
    assert output_latent["dreamx_num_frames"] == 10
    assert pipeline.calls == 2
    assert pipeline.kv_caches is None
    assert refiner.window_chunk == pipeline.generator.model.attn.window_chunk == 1
    assert load_calls == [([refiner.patcher], {"force_full_load": True})]


def test_public_refiner_node_returns_only_exact_length_images(monkeypatch):
    images = torch.zeros(10, 64, 96, 3)
    padded_latent = {"samples": torch.zeros(1, 48, 4, 8, 12)}
    monkeypatch.setattr(
        "comfy_nodes.nodes.run_refiner", lambda *args: (images, padded_latent)
    )
    output = DreamXCausalRefine.execute(
        object(), object(), [[torch.zeros(1), {}]], images,
        2, 7, 0.6251, True, 1.0, 0, -1,
    )
    assert len(output.result) == 1
    assert output.result[0] is images

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import comfy.model_management

from comfy_nodes.refiner import DreamXRefinerHandle, run_refiner


class FakeWanVAE:
    def __init__(self):
        self.encoded_frames = None

    def encode(self, images):
        self.encoded_frames = images.shape[0]
        return torch.zeros(
            1, 48, ((images.shape[0] - 1) // 4) + 1,
            images.shape[1] // 16, images.shape[2] // 16,
        )

    def decode(self, latent):
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
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda *_: None)
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
    assert output_images.shape == (10, 128, 192, 3)
    assert output_latent["samples"].shape == (1, 48, 4, 8, 12)
    assert output_latent["dreamx_num_frames"] == 10
    assert pipeline.calls == 2
    assert pipeline.kv_caches is None
    assert refiner.window_chunk == pipeline.generator.model.attn.window_chunk == 1

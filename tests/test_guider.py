import torch

import comfy.model_management
import comfy.samplers
from comfy.nested_tensor import NestedTensor

from comfy_nodes.guider import DreamXMultimodalGuider, apply_multimodal_guidance


def test_three_branch_guidance_has_independent_stream_scales():
    base = torch.zeros(1, 1, 6)
    bridge = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    text = bridge + 10.0
    result = apply_multimodal_guidance(
        base, bridge, text, video_numel=4,
        text_cfg=2.0, video_bridge=3.0, audio_bridge=5.0,
    )
    expected = torch.tensor([[[23.0, 26.0, 29.0, 32.0, 45.0, 50.0]]])
    torch.testing.assert_close(result, expected)


def test_zero_bridge_scales_reduce_to_text_cfg_branch():
    base = torch.randn(2, 1, 7)
    bridge = torch.randn(2, 1, 7)
    text = torch.randn(2, 1, 7)
    result = apply_multimodal_guidance(base, bridge, text, 4, 4.5, 0.0, 0.0)
    torch.testing.assert_close(result, base + 4.5 * (text - bridge))


def test_guider_preserves_comfy_sampler_hook_chain():
    calls = []
    guider = object.__new__(DreamXMultimodalGuider)
    guider.inner_model = object()
    guider.conds = {"base": "base", "bridge": "bridge", "text": "text"}
    guider.text_cfg = 2.0
    guider.video_bridge = 0.0
    guider.audio_bridge = 0.0
    guider.enable_a2v = True
    guider.enable_v2a = True
    guider.video_numel = 2
    x = torch.zeros(1, 1, 4)
    branches = [torch.zeros_like(x), torch.ones_like(x), torch.full_like(x, 3.0)]

    def calc(args):
        calls.append("calc")
        return branches

    def pre(args):
        calls.append("pre")
        return args["conds_out"]

    def cfg(args):
        calls.append("cfg")
        denoised = args["uncond_denoised"] + 2.0 * (
            args["cond_denoised"] - args["uncond_denoised"]
        )
        return args["input"] - denoised

    def post(args):
        calls.append("post")
        return args["denoised"] + 10.0

    result = guider.predict_noise(x, torch.ones(1), model_options={
        "sampler_calc_cond_batch_function": calc,
        "sampler_pre_cfg_function": [pre],
        "sampler_cfg_function": cfg,
        "sampler_post_cfg_function": [post],
    })
    assert calls == ["calc", "pre", "cfg", "post"]
    torch.testing.assert_close(result, torch.full_like(x, 14.0))


def test_guider_forces_coherent_full_model_load(monkeypatch):
    calls = []
    guider = object.__new__(DreamXMultimodalGuider)
    guider.model_patcher = object()
    latent = NestedTensor([
        torch.zeros(1, 48, 2, 4, 4),
        torch.zeros(1, 128, 2),
    ])
    monkeypatch.setattr(
        comfy.model_management,
        "load_models_gpu",
        lambda models, **kwargs: calls.append((models, kwargs)),
    )
    monkeypatch.setattr(
        comfy.samplers.CFGGuider,
        "sample",
        lambda self, noise, latent_image, *args, **kwargs: latent_image,
    )
    assert guider.sample(latent, latent) is latent
    assert calls == [([guider.model_patcher], {"force_full_load": True})]

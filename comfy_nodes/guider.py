"""DreamX three-branch multimodal guidance on top of ComfyUI's sampler API."""

from __future__ import annotations

import math

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.samplers
import node_helpers


def apply_multimodal_guidance(
    no_text_no_bridge: torch.Tensor,
    no_text_bridge: torch.Tensor,
    text_bridge: torch.Tensor,
    video_numel: int,
    text_cfg: float,
    video_bridge: float,
    audio_bridge: float,
) -> torch.Tensor:
    """Combine the three model branches with independent video/audio bridge scales."""
    text_guided = no_text_bridge + text_cfg * (text_bridge - no_text_bridge)
    return apply_bridge_guidance(
        no_text_no_bridge, no_text_bridge, text_guided, video_numel,
        video_bridge, audio_bridge,
    )


def apply_bridge_guidance(
    no_text_no_bridge: torch.Tensor,
    no_text_bridge: torch.Tensor,
    text_guided: torch.Tensor,
    video_numel: int,
    video_bridge: float,
    audio_bridge: float,
) -> torch.Tensor:
    """Add modality-specific bridge deltas to an already CFG-guided text branch."""
    guided = no_text_no_bridge + (text_guided - no_text_bridge)
    bridge_delta = no_text_bridge - no_text_no_bridge
    guided = guided.clone()
    guided[..., :video_numel] += video_bridge * bridge_delta[..., :video_numel]
    guided[..., video_numel:] += audio_bridge * bridge_delta[..., video_numel:]
    return guided


class DreamXMultimodalGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive, negative):
        base = node_helpers.conditioning_set_values(
            negative, {"dreamx_bridge": torch.tensor([[0.0]]), "prompt_type": "negative"}
        )
        bridge = node_helpers.conditioning_set_values(
            negative, {"dreamx_bridge": torch.tensor([[1.0]]), "prompt_type": "negative"}
        )
        text = node_helpers.conditioning_set_values(
            positive, {"dreamx_bridge": torch.tensor([[1.0]]), "prompt_type": "positive"}
        )
        self.inner_set_conds({"base": base, "bridge": bridge, "text": text})

    def set_scales(self, text_cfg, video_bridge, audio_bridge, enable_a2v, enable_v2a):
        self.text_cfg = float(text_cfg)
        self.video_bridge = float(video_bridge)
        self.audio_bridge = float(audio_bridge)
        self.enable_a2v = bool(enable_a2v)
        self.enable_v2a = bool(enable_v2a)

    def sample(self, noise, latent_image, *args, **kwargs):
        if not getattr(latent_image, "is_nested", False):
            raise ValueError("DreamX guider requires a DreamX audio/video NestedTensor latent")
        self.video_numel = math.prod(latent_image.unbind()[0].shape[1:])
        # DreamX vendors regular torch Linear/Conv modules rather than Comfy's
        # manual-cast ops. Comfy's generic partial-weight offload would therefore
        # leave CPU weights behind while activations are on CUDA. The released
        # bf16 model fits fully on the supported 24 GB target once CLIP/VAE are
        # unloaded, so require a coherent full load before sampling.
        comfy.model_management.load_models_gpu(
            [self.model_patcher], force_full_load=True
        )
        return super().sample(noise, latent_image, *args, **kwargs)

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        model_options = comfy.model_patcher.create_model_options_clone(model_options or {})
        transformer_options = model_options.setdefault("transformer_options", {})
        transformer_options["dreamx_enable_a2v"] = self.enable_a2v
        transformer_options["dreamx_enable_v2a"] = self.enable_v2a
        conds = [self.conds.get("base"), self.conds.get("bridge"), self.conds.get("text")]
        if "sampler_calc_cond_batch_function" in model_options:
            branches = model_options["sampler_calc_cond_batch_function"]({
                "conds": conds, "input": x, "sigma": timestep,
                "model": self.inner_model, "model_options": model_options,
            })
        else:
            branches = comfy.samplers.calc_cond_batch(
                self.inner_model, conds, x, timestep, model_options,
            )
        for fn in model_options.get("sampler_pre_cfg_function", []):
            branches = fn({
                "conds": conds, "conds_out": branches, "cond_scale": self.text_cfg,
                "timestep": timestep, "input": x, "sigma": timestep,
                "model": self.inner_model, "model_options": model_options,
            })

        # Let standard sampler CFG hooks transform the text-vs-bridge pair. Post-CFG
        # hooks run below on the fully combined multimodal prediction.
        cfg_options = comfy.model_patcher.create_model_options_clone(model_options)
        cfg_options.pop("sampler_post_cfg_function", None)
        text_guided = comfy.samplers.cfg_function(
            self.inner_model, branches[2], branches[1], self.text_cfg, x, timestep,
            model_options=cfg_options, cond=conds[2], uncond=conds[1],
        )
        guided = apply_bridge_guidance(
            branches[0], branches[1], text_guided, self.video_numel,
            self.video_bridge, self.audio_bridge,
        )
        for fn in model_options.get("sampler_post_cfg_function", []):
            guided = fn({
                "denoised": guided, "cond": conds[2], "uncond": conds[1],
                "cond_scale": self.text_cfg, "model": self.inner_model,
                "uncond_denoised": branches[1], "cond_denoised": branches[2],
                "sigma": timestep, "model_options": model_options, "input": x,
            })
        return guided

"""Model loaders and lightweight ComfyUI wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sd
import comfy.utils

from .creator_model import load_creator_model
from .utils import (
    dtype_from_name,
    resolve_model_root,
    validate_model_files,
    verify_trusted_model_file,
)


def force_full_creator_load(executor, model, noise_shape, conds, *args, **kwargs):
    """Prevent generic partial-weight offload for vendored DreamX torch layers."""
    kwargs["force_full_load"] = True
    return executor(model, noise_shape, conds, *args, **kwargs)


def load_model_patcher(model_root, dtype: torch.dtype):
    base_model = load_creator_model(model_root, dtype)
    patcher = comfy.model_patcher.ModelPatcher(
        base_model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )

    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        "dreamx_force_full_creator_load",
        force_full_creator_load,
    )
    return patcher


class ManagedTextEncoder(torch.nn.Module):
    """Give ComfyUI a mutable device attribute around Diffusers' read-only one."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = torch.device("cpu")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class DreamXTextEncoder:
    """Small CLIP-compatible facade around the released UMT5 encoder."""

    def __init__(self, model, tokenizer):
        self.cond_stage_model = model
        self.tokenizer_backend = tokenizer
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.text_encoder_device(),
            offload_device=comfy.model_management.text_encoder_offload_device(),
        )
        self.patcher.is_clip = True

    def clone(self, disable_dynamic=False):
        clone = object.__new__(DreamXTextEncoder)
        clone.cond_stage_model = self.cond_stage_model
        clone.tokenizer_backend = self.tokenizer_backend
        clone.patcher = self.patcher.clone(disable_dynamic=disable_dynamic)
        clone.patcher.is_clip = True
        return clone

    def tokenize(self, text, return_word_ids=False, **kwargs):
        encoded = self.tokenizer_backend(
            [text],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {"input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask}

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        cond = self.encode_from_tokens(tokens)
        return [[cond, dict(add_dict or {})]]

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        comfy.model_management.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        dtype = next(self.cond_stage_model.parameters()).dtype
        input_ids = tokens["input_ids"].to(device=device)
        attention_mask = tokens["attention_mask"].to(device=device)
        with comfy.model_management.cuda_device_context(device), torch.inference_mode():
            encoded = self.cond_stage_model(input_ids, attention_mask=attention_mask)[0]
            encoded = encoded.to(dtype=dtype)
            encoded = encoded * attention_mask[..., None].to(encoded.dtype)
        encoded = encoded.to(comfy.model_management.intermediate_device())
        if return_dict:
            return {"cond": encoded, "pooled_output": None}
        if return_pooled:
            return encoded, None
        return encoded

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))

    def get_models(self):
        return [self.patcher]

    def is_dynamic(self):
        return self.patcher.is_dynamic()


def load_text_encoder(model_root, dtype: torch.dtype):
    from transformers import AutoTokenizer
    if "." in (__package__ or ""):
        from ..audio_video_generation.videox_fun.models.wan_text_encoder import WanT5EncoderModel
    else:
        from audio_video_generation.videox_fun.models.wan_text_encoder import WanT5EncoderModel

    t5_path = verify_trusted_model_file(
        model_root, "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_path = model_root / "wan2.2_ti2v_5b" / "google" / "umt5-xxl"
    config = {
        "vocab": 256384,
        "dim": 4096,
        "dim_attn": 4096,
        "dim_ffn": 10240,
        "num_heads": 64,
        "num_layers": 24,
        "num_buckets": 32,
        "shared_pos": False,
        "dropout": 0.0,
    }
    model = WanT5EncoderModel.from_pretrained(
        str(t5_path), additional_kwargs=config, low_cpu_mem_usage=True, torch_dtype=dtype
    )
    model.eval().requires_grad_(False)
    model = ManagedTextEncoder(model)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    return DreamXTextEncoder(model, tokenizer)


@dataclass
class DreamXAudioVAEHandle:
    model: torch.nn.Module
    patcher: comfy.model_patcher.CoreModelPatcher

    @property
    def sample_rate(self):
        return int(self.model.sample_rate)

    @property
    def hop_length(self):
        return int(self.model.hop_length)

    @property
    def latent_dim(self):
        return int(self.model.latent_dim)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        comfy.model_management.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        dtype = next(self.model.parameters()).dtype
        with comfy.model_management.cuda_device_context(device), torch.inference_mode():
            device_type = device.type
            enabled = device_type in {"cuda", "xpu"} and dtype != torch.float32
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
                waveform = self.model.decode(latent.to(device=device, dtype=dtype))
        return waveform.float().cpu()


def load_audio_vae(model_root, dtype: torch.dtype):
    if "." in (__package__ or ""):
        from ..audio_video_generation.videox_fun.models.creator_dac_vae import CreatorDACVAE
    else:
        from audio_video_generation.videox_fun.models.creator_dac_vae import CreatorDACVAE

    model = CreatorDACVAE.from_pretrained(model_root / "audio_vae")
    model.to(dtype=dtype).eval().requires_grad_(False)
    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.vae_offload_device(),
    )
    return DreamXAudioVAEHandle(model, patcher)


def load_wan_vae(model_root):
    path = verify_trusted_model_file(model_root, "wan2.2_ti2v_5b/Wan2.2_VAE.pth")
    state_dict, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    vae = comfy.sd.VAE(sd=state_dict, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def resolve_and_validate(model_root: str):
    root = resolve_model_root(model_root)
    validate_model_files(root)
    return root

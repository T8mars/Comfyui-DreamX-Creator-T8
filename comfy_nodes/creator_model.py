"""Adapter that makes the released DreamX Creator DiTs a native ComfyUI MODEL."""

from __future__ import annotations

import math

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base
import comfy.utils
from comfy.nested_tensor import NestedTensor


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    """Map a sigma from one shifted flow schedule onto another."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


class DreamXModelConfig(comfy.supported_models_base.BASE):
    unet_config = {"disable_unet_model_creation": True}
    latent_format = comfy.latent_formats.Wan22
    sampling_settings = {"shift": 5.0, "audio_shift": 5.0}
    memory_usage_factor = 1.0

    def __init__(self, dtype: torch.dtype):
        super().__init__(self.unet_config)
        self.set_inference_dtype(dtype, None)


class DreamXCreatorDiffusion(torch.nn.Module):
    """Translate ComfyUI's unpacked multimodal call to the released joint model."""

    def __init__(self, joint_model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.joint_model = joint_model
        self.compute_dtype = dtype

    @property
    def dtype(self):
        return self.compute_dtype

    def forward(
        self,
        x,
        timestep,
        context=None,
        transformer_options=None,
        dreamx_bridge=None,
        dreamx_fps=24.0,
        dreamx_audio_scale=1.0,
        dreamx_video_shift=5.0,
        dreamx_audio_shift=5.0,
        **kwargs,
    ):
        if not isinstance(x, (tuple, list)) or len(x) < 2:
            raise ValueError("DreamX Creator expects unpacked [video, audio] latents")
        video_x, audio_x = x[0], x[1]
        audio_source = audio_x
        batch = video_x.shape[0]
        patch_t, patch_h, patch_w = self.joint_model.video_patch_size
        video_seq_len = (
            video_x.shape[2] * video_x.shape[3] * video_x.shape[4]
        ) // (patch_t * patch_h * patch_w)
        audio_seq_len = audio_x.shape[-1] // self.joint_model.audio_patch_size[0]

        timestep = timestep.reshape(-1).to(device=video_x.device, dtype=torch.float32)
        if timestep.numel() == 1 and batch > 1:
            timestep = timestep.expand(batch)
        video_t = timestep[:, None].expand(batch, video_seq_len).clone()
        first_frame_tokens = (video_x.shape[3] * video_x.shape[4]) // (patch_h * patch_w)
        video_t[:, :first_frame_tokens] = 0.0

        audio_scale = float(dreamx_audio_scale)
        audio_sigma = (timestep / 1000.0).clamp(min=1.0e-6)
        if not math.isclose(audio_scale, 1.0):
            audio_sigma = time_shift_sigma(
                audio_sigma, float(dreamx_video_shift), float(dreamx_audio_shift)
            )
            carry = (audio_sigma / (timestep / 1000.0).clamp(min=1.0e-6)).to(audio_x.dtype)
            audio_x = audio_x * carry.view(batch, *([1] * (audio_x.ndim - 1)))
        audio_timestep = audio_sigma * 1000.0

        if dreamx_bridge is None:
            bridge = torch.ones(batch, device=video_x.device, dtype=torch.bool)
        else:
            bridge = dreamx_bridge.reshape(-1).to(device=video_x.device).bool()
            if bridge.numel() == 1 and batch > 1:
                bridge = bridge.expand(batch)

        transformer_options = transformer_options or {}
        enable_a2v = bridge & bool(transformer_options.get("dreamx_enable_a2v", True))
        enable_v2a = bridge & bool(transformer_options.get("dreamx_enable_v2a", True))
        fps = float(dreamx_fps.item()) if torch.is_tensor(dreamx_fps) else float(dreamx_fps)

        video_inputs = {
            "x": [sample for sample in video_x],
            "t": video_t,
            "context": context,
            "y": None,
            "seq_len": video_seq_len,
            "video_fps": fps,
        }
        audio_inputs = {
            "x": [sample for sample in audio_x],
            "t": audio_timestep,
            "context": context,
            "y": None,
            "seq_len": audio_seq_len,
        }
        device_type = video_x.device.type
        autocast_enabled = device_type in {"cuda", "xpu"} and self.compute_dtype != torch.float32
        with torch.autocast(device_type=device_type, dtype=self.compute_dtype, enabled=autocast_enabled):
            output = self.joint_model(
                video=video_inputs,
                audio=audio_inputs,
                dtype=self.compute_dtype,
                enable_a2v=enable_a2v,
                enable_v2a=enable_v2a,
            )
        audio_output = output["audio"]
        if not math.isclose(audio_scale, 1.0):
            carry = (audio_sigma / (timestep / 1000.0).clamp(min=1.0e-6)).to(audio_source.dtype)
            carried_audio = audio_source * carry.view(batch, *([1] * (audio_source.ndim - 1)))
            velocity_scale = 1.0 + (audio_scale - 1.0) * audio_sigma
            audio_output = (
                (1.0 - audio_scale) * carried_audio
                + velocity_scale.to(audio_output.dtype).view(
                    batch, *([1] * (audio_output.ndim - 1))
                ) * audio_output
            )
        return output["video"], audio_output


class DreamXCreatorBaseModel(comfy.model_base.BaseModel):
    """ComfyUI BaseModel with stream-aware Wan latent normalization."""

    def __init__(self, joint_model: torch.nn.Module, dtype: torch.dtype):
        config = DreamXModelConfig(dtype)
        super().__init__(config, model_type=comfy.model_base.ModelType.FLOW_AV, device=torch.device("cpu"))
        self.diffusion_model = DreamXCreatorDiffusion(joint_model, dtype)

    def audio_scale(self):
        if self.latent_shapes is None or len(self.latent_shapes) < 2:
            return 1.0
        return float(self.model_sampling.audio_scale)

    def _map_streams(self, latent, video_fn, audio_scale):
        if getattr(latent, "is_nested", False):
            streams = list(latent.unbind())
            streams[0] = video_fn(streams[0])
            if len(streams) > 1 and audio_scale != 1.0:
                streams[1] = streams[1] * audio_scale
            return NestedTensor(streams)
        if self.latent_shapes is None or len(self.latent_shapes) < 2:
            return video_fn(latent)
        streams = comfy.utils.unpack_latents(latent, self.latent_shapes)
        streams[0] = video_fn(streams[0])
        if audio_scale != 1.0:
            streams[1] = streams[1] * audio_scale
        packed, _ = comfy.utils.pack_latents(streams)
        return packed

    def process_latent_in(self, latent):
        return self._map_streams(latent, self.latent_format.process_in, self.audio_scale())

    def process_latent_out(self, latent):
        return self._map_streams(
            latent, self.latent_format.process_out, 1.0 / self.audio_scale()
        )

    def scale_latent_inpaint(self, sigma, noise, latent_image, x=None, denoise_mask=None, **kwargs):
        return latent_image

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is not None:
            out["latent_shapes"] = comfy.conds.CONDConstant(latent_shapes)
        bridge = kwargs.get("dreamx_bridge")
        if bridge is not None:
            if not torch.is_tensor(bridge):
                bridge = torch.tensor([[float(bridge)]], dtype=torch.float32)
            if bridge.ndim == 0:
                bridge = bridge.reshape(1, 1)
            elif bridge.ndim == 1:
                bridge = bridge[:, None]
            out["dreamx_bridge"] = comfy.conds.CONDRegular(bridge)
        out["dreamx_fps"] = comfy.conds.CONDConstant(float(kwargs.get("dreamx_fps", 24.0)))
        out["dreamx_audio_scale"] = comfy.conds.CONDConstant(self.audio_scale())
        out["dreamx_video_shift"] = comfy.conds.CONDConstant(float(self.model_sampling.shift))
        audio_shift = self.model_sampling.audio_shift
        out["dreamx_audio_shift"] = comfy.conds.CONDConstant(
            float(self.model_sampling.shift if audio_shift is None else audio_shift)
        )
        return out


def load_creator_model(model_root, dtype: torch.dtype):
    """Load the official released Creator model without changing its architecture."""
    if "." in (__package__ or ""):
        from ..audio_video_generation.videox_fun.models.creator_gating import WanCreatorGatingAVModel
    else:
        from audio_video_generation.videox_fun.models.creator_gating import WanCreatorGatingAVModel

    creator_root = str(model_root / "creator")
    model = WanCreatorGatingAVModel.from_pretrained(
        pretrained_model_path=creator_root,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        use_temporal_rope=True,
        audio_fps=50.0,
        vae_temporal_stride=4,
        a2v_cross_attn_layers=list(range(15, 30)),
        v2a_cross_attn_layers=list(range(15, 30)),
        use_gating=True,
        zero_init_cross_attn=False,
        zero_init_gating=False,
        gate_init_value=0.5,
    )
    model.eval().requires_grad_(False)
    return DreamXCreatorBaseModel(model, dtype)

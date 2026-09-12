"""ComfyUI V3 nodes for native DreamX-Creator generation and decoding."""

from __future__ import annotations

import math

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.model_sampling
import comfy.utils
import node_helpers
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

from .guider import DreamXMultimodalGuider
from .loaders import (
    load_audio_vae,
    load_model_patcher,
    load_text_encoder,
    load_wan_vae,
    resolve_and_validate,
)
from .types import AudioVAE
from .types import Refiner
from .refiner import load_refiner, run_refiner
from .utils import compute_dynamic_resolution, dtype_from_name, snap_video_frames, tensor_streams


ROOT_INPUT = lambda: io.String.Input(
    "model_root",
    default="auto",
    tooltip="'auto' uses this repository's checkpoints/ or ComfyUI/models/dreamx_creator.",
)
DTYPE_INPUT = lambda: io.Combo.Input(
    "dtype", options=["bfloat16", "float16", "float32"], default="bfloat16"
)


class DreamXCreatorModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXCreatorModelLoader",
            display_name="DreamX Creator Model Loader",
            category="DreamX-Creator/loaders",
            inputs=[ROOT_INPUT(), DTYPE_INPUT()],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model_root, dtype):
        root = resolve_and_validate(model_root)
        return io.NodeOutput(load_model_patcher(root, dtype_from_name(dtype)))


class DreamXUMT5Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXUMT5Loader",
            display_name="DreamX UMT5-XXL Loader",
            category="DreamX-Creator/loaders",
            inputs=[ROOT_INPUT(), DTYPE_INPUT()],
            outputs=[io.Clip.Output("clip")],
        )

    @classmethod
    def execute(cls, model_root, dtype):
        root = resolve_and_validate(model_root)
        return io.NodeOutput(load_text_encoder(root, dtype_from_name(dtype)))


class DreamXWanVaeLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXWanVaeLoader",
            display_name="DreamX Wan2.2 VAE Loader",
            category="DreamX-Creator/loaders",
            inputs=[ROOT_INPUT()],
            outputs=[io.Vae.Output("vae")],
        )

    @classmethod
    def execute(cls, model_root):
        return io.NodeOutput(load_wan_vae(resolve_and_validate(model_root)))


class DreamXAudioVaeLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXAudioVaeLoader",
            display_name="DreamX Audio VAE Loader",
            category="DreamX-Creator/loaders",
            inputs=[ROOT_INPUT(), DTYPE_INPUT()],
            outputs=[AudioVAE.Output("audio_vae")],
        )

    @classmethod
    def execute(cls, model_root, dtype):
        root = resolve_and_validate(model_root)
        return io.NodeOutput(load_audio_vae(root, dtype_from_name(dtype)))


class DreamXBundleLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXBundleLoader",
            display_name="DreamX Creator Complete Loader",
            category="DreamX-Creator/loaders",
            inputs=[ROOT_INPUT(), DTYPE_INPUT()],
            outputs=[
                io.Model.Output("model"), io.Clip.Output("clip"), io.Vae.Output("vae"),
                AudioVAE.Output("audio_vae"),
            ],
        )

    @classmethod
    def execute(cls, model_root, dtype):
        root = resolve_and_validate(model_root)
        torch_dtype = dtype_from_name(dtype)
        return io.NodeOutput(
            load_model_patcher(root, torch_dtype),
            load_text_encoder(root, torch_dtype),
            load_wan_vae(root),
            load_audio_vae(root, torch_dtype),
        )


class DreamXFirstFrameAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXFirstFrameAVLatent",
            display_name="DreamX First Frame AV Latent",
            category="DreamX-Creator/conditioning",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("start_image"),
                io.Float.Input("duration", default=5.0, min=0.25, max=60.0, step=0.25),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=1.0),
                io.Int.Input("target_spatial_tokens", default=880, min=64, max=4096, step=1),
                io.Int.Input("batch_size", default=1, min=1, max=16),
            ],
            outputs=[
                io.Conditioning.Output("positive"), io.Conditioning.Output("negative"),
                io.Latent.Output("latent"), io.Int.Output("width"), io.Int.Output("height"),
                io.Int.Output("frames"), io.Float.Output("fps"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, start_image, duration, fps, target_spatial_tokens, batch_size):
        source_h, source_w = int(start_image.shape[1]), int(start_image.shape[2])
        height, width, _ = compute_dynamic_resolution(
            source_h, source_w, int(target_spatial_tokens)
        )
        frames = snap_video_frames(float(duration), float(fps))
        latent_frames = ((frames - 1) // 4) + 1
        intermediate = comfy.model_management.intermediate_device()

        resized = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bicubic", "disabled"
        ).movedim(1, -1)
        first_latent = vae.encode(resized)
        expected_first_shape = (1, 48, 1, height // 16, width // 16)
        if tuple(first_latent.shape) != expected_first_shape:
            raise ValueError(
                "DreamX requires the Wan2.2 48-channel video VAE. "
                f"Expected {expected_first_shape}, got {tuple(first_latent.shape)}"
            )
        video_shape = (1, 48, latent_frames, height // 16, width // 16)
        video_mask = torch.ones((1, 1, latent_frames, height // 16, width // 16), device=intermediate)
        video_mask[:, :, : first_latent.shape[2]] = 0.0
        normalized_zero = torch.zeros(video_shape, device=intermediate, dtype=first_latent.dtype)
        video_latent = comfy.latent_formats.Wan22().process_out(normalized_zero)
        video_latent[:, :, : first_latent.shape[2]] = first_latent.to(intermediate)

        seconds = frames / float(fps)
        audio_len = math.ceil(seconds * 48000 / 960)
        audio_latent = torch.zeros((1, 128, audio_len), device=intermediate, dtype=first_latent.dtype)
        audio_mask = torch.ones((1, 1, audio_len), device=intermediate, dtype=first_latent.dtype)

        repeat_video = (batch_size, 1, 1, 1, 1)
        repeat_audio = (batch_size, 1, 1)
        latent = {
            "samples": NestedTensor([
                video_latent.repeat(repeat_video), audio_latent.repeat(repeat_audio)
            ]),
            "noise_mask": NestedTensor([
                video_mask.repeat(repeat_video), audio_mask.repeat(repeat_audio)
            ]),
            "dreamx_num_frames": frames,
            "dreamx_fps": float(fps),
            "dreamx_audio_samples": round(seconds * 48000),
        }
        values = {"dreamx_fps": float(fps)}
        return io.NodeOutput(
            node_helpers.conditioning_set_values(positive, values),
            node_helpers.conditioning_set_values(negative, values),
            latent, width, height, frames, float(fps),
        )


class DreamXAVSamplingPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXAVSamplingPatch",
            display_name="DreamX AV Flow Shifts",
            category="DreamX-Creator/sampling",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("video_shift", default=5.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("audio_shift", default=5.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, video_shift, audio_shift):
        patched = model.clone()

        class ModelSamplingDreamX(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass

        sampling = ModelSamplingDreamX(model.model.model_config)
        sampling.set_parameters(shift=float(video_shift), audio_shift=float(audio_shift))
        patched.add_object_patch("model_sampling", sampling)
        return io.NodeOutput(patched)


class DreamXGuiderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXMultimodalGuider",
            display_name="DreamX Multimodal Guider",
            category="DreamX-Creator/sampling",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("text_cfg", default=5.0, min=0.0, max=30.0, step=0.1),
                io.Float.Input("video_bridge", default=3.5, min=0.0, max=30.0, step=0.1),
                io.Float.Input("audio_bridge", default=3.5, min=0.0, max=30.0, step=0.1),
                io.Boolean.Input("enable_a2v", default=True),
                io.Boolean.Input("enable_v2a", default=True),
            ],
            outputs=[io.Guider.Output("guider")],
        )

    @classmethod
    def execute(cls, model, positive, negative, text_cfg, video_bridge, audio_bridge, enable_a2v, enable_v2a):
        guider = DreamXMultimodalGuider(model)
        guider.set_conds(positive, negative)
        guider.set_scales(text_cfg, video_bridge, audio_bridge, enable_a2v, enable_v2a)
        return io.NodeOutput(guider)


class DreamXSplitAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXSplitAVLatent",
            display_name="DreamX Split AV Latent",
            category="DreamX-Creator/latent",
            inputs=[io.Latent.Input("latent")],
            outputs=[io.Latent.Output("video_latent"), io.Latent.Output("audio_latent")],
        )

    @classmethod
    def execute(cls, latent):
        video, audio = tensor_streams(latent["samples"])[:2]
        metadata = {k: v for k, v in latent.items() if k not in {"samples", "noise_mask"}}
        return io.NodeOutput({"samples": video, **metadata}, {"samples": audio, **metadata})


class DreamXAudioVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXAudioVAEDecode",
            display_name="DreamX Audio VAE Decode",
            category="DreamX-Creator/audio",
            inputs=[AudioVAE.Input("audio_vae"), io.Latent.Input("audio_latent")],
            outputs=[io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, audio_vae, audio_latent):
        waveform = audio_vae.decode(audio_latent["samples"])
        trim = audio_latent.get("dreamx_audio_samples")
        if trim is not None:
            waveform = waveform[..., : int(trim)]
        return io.NodeOutput({"waveform": waveform, "sample_rate": audio_vae.sample_rate})


class DreamXRefinerLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXRefinerLoader",
            display_name="DreamX Causal Refiner Loader",
            category="DreamX-Creator/refiner",
            inputs=[
                ROOT_INPUT(), DTYPE_INPUT(),
                io.Int.Input(
                    "window_chunk", default=4, min=1, max=64,
                    tooltip="Windows-safe attention window batch; lower uses less VRAM.",
                ),
            ],
            outputs=[Refiner.Output("refiner")],
        )

    @classmethod
    def execute(cls, model_root, dtype, window_chunk):
        root = resolve_and_validate(model_root)
        return io.NodeOutput(load_refiner(root, dtype_from_name(dtype), window_chunk))


class DreamXCausalRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamXCausalRefine",
            display_name="DreamX Causal Refine Video",
            category="DreamX-Creator/refiner",
            inputs=[
                Refiner.Input("refiner"), io.Vae.Input("vae"),
                io.Conditioning.Input("positive"), io.Image.Input("images"),
                io.Int.Input("scale", default=2, min=2, max=2),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Float.Input("sigma_start", default=0.6251, min=0.0, max=1.0, step=0.0001),
                io.Boolean.Input("use_lq_anchor", default=True),
                io.Float.Input("lq_guidance_scale", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Int.Input("anchor_context_noise", default=0, min=0, max=1000),
                io.Int.Input(
                    "anchor_keep_prefix", default=-1, min=-1, max=30,
                    tooltip="-1 applies the LQ anchor to every trained layer.",
                ),
            ],
            outputs=[io.Image.Output("images"), io.Latent.Output("video_latent")],
        )

    @classmethod
    def execute(
        cls, refiner, vae, positive, images, scale, seed, sigma_start,
        use_lq_anchor, lq_guidance_scale, anchor_context_noise, anchor_keep_prefix,
    ):
        return io.NodeOutput(*run_refiner(
            refiner, vae, positive, images, scale, seed, sigma_start,
            use_lq_anchor, lq_guidance_scale, anchor_context_noise, anchor_keep_prefix,
        ))


NODE_LIST = [
    DreamXCreatorModelLoader,
    DreamXUMT5Loader,
    DreamXWanVaeLoader,
    DreamXAudioVaeLoader,
    DreamXBundleLoader,
    DreamXFirstFrameAVLatent,
    DreamXAVSamplingPatch,
    DreamXGuiderNode,
    DreamXSplitAVLatent,
    DreamXAudioVAEDecode,
    DreamXRefinerLoader,
    DreamXCausalRefine,
]

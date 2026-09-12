"""ComfyUI integration for the released DreamX 5B causal video refiner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .utils import REPO_ROOT, verify_trusted_model_file


MAX_REFINER_OUTPUT_PIXELS = 8 * 1024 * 1024
VAE_TILE_SIZE = 512
VAE_TILE_OVERLAP = 64
VAE_TEMPORAL_SIZE = 64
VAE_TEMPORAL_OVERLAP = 8


@dataclass
class DreamXRefinerHandle:
    pipeline: torch.nn.Module
    patcher: comfy.model_patcher.CoreModelPatcher
    dtype: torch.dtype
    window_chunk: int


def _refiner_imports():
    if "." in (__package__ or ""):
        from ..video_refiner.pipeline_sr.causal_inference import CausalInferencePipeline
        from ..video_refiner.wan.modules.latent_upsampler.flash_latent_up import FlashLatentUpsampler
    else:
        from video_refiner.pipeline_sr.causal_inference import CausalInferencePipeline
        from video_refiner.wan.modules.latent_upsampler.flash_latent_up import FlashLatentUpsampler
    return CausalInferencePipeline, FlashLatentUpsampler


def _set_window_chunk(pipeline, window_chunk: int) -> None:
    value = max(1, int(window_chunk))
    model = pipeline.generator.model
    if hasattr(model, "window_chunk"):
        model.window_chunk = value
    for module in model.modules():
        if getattr(module, "use_window_attn", False):
            module.window_chunk = value


def load_refiner(
    model_root: Path,
    dtype: torch.dtype,
    window_chunk: int = 1,
    kv_history_frames: int = 3,
) -> DreamXRefinerHandle:
    from omegaconf import OmegaConf

    CausalInferencePipeline, FlashLatentUpsampler = _refiner_imports()
    config = OmegaConf.load(str(REPO_ROOT / "video_refiner" / "configs" / "sr_dit_5b.yaml"))
    upsampler_config = OmegaConf.load(
        str(REPO_ROOT / "video_refiner" / "configs" / "latent_upsampler_flash.yaml")
    )
    config.wan_model_dir = str(model_root / "wan2.2_ti2v_5b")
    config.sr_dit_checkpoint = str(
        verify_trusted_model_file(model_root, "refiner/sr_dit_5b.pt")
    )
    config.timestep_shift = float(config.model_kwargs.get("timestep_shift", 5.0))
    config.independent_first_frame = False
    config.num_frame_per_block = int(config.get("num_frame_per_block", 3))
    kv_history_frames = max(1, int(kv_history_frames))
    config.stream_kv_len = kv_history_frames
    config.dit_arch_config.stream_kv_len = kv_history_frames
    config.dit_arch_config.window_causal_kv_len = kv_history_frames
    config.dit_arch_config.window_chunk = max(1, int(window_chunk))
    config.latent_upsample_mode = upsampler_config.latent_upsample_mode
    config.latent_upsampler_precision = (
        "bf16" if dtype == torch.bfloat16 else "fp16" if dtype == torch.float16 else "fp32"
    )
    config.latent_upsampler_arch_config = upsampler_config.latent_upsampler_arch_config
    config.latent_upsampler_arch_config.target = (
        f"{FlashLatentUpsampler.__module__}.FlashLatentUpsampler"
    )
    config.latent_upsampler_ckpt = str(
        verify_trusted_model_file(model_root, "refiner/latent_upsampler_flash.pt")
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        pipeline = CausalInferencePipeline(
            config,
            device=comfy.model_management.get_torch_device(),
            text_encoder=torch.nn.Identity(),
            vae=None,
            need_vae=False,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    pipeline.eval().requires_grad_(False)
    _set_window_chunk(pipeline, window_chunk)
    patcher = comfy.model_patcher.CoreModelPatcher(
        pipeline,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    return DreamXRefinerHandle(pipeline, patcher, dtype, max(1, int(window_chunk)))


def run_refiner(
    refiner: DreamXRefinerHandle,
    vae,
    positive,
    images: torch.Tensor,
    scale: int,
    seed: int,
    sigma_start: float,
    use_lq_anchor: bool,
    lq_guidance_scale: float,
    anchor_context_noise: int,
    anchor_keep_prefix: int,
):
    if not positive:
        raise ValueError("Refiner positive conditioning is empty")
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Refiner images must be a ComfyUI IMAGE sequence [T,H,W,C]")
    if images.shape[0] < 1:
        raise ValueError("DreamX Refiner requires at least one video frame")
    if images.shape[1] % 16 or images.shape[2] % 16:
        raise ValueError(
            "DreamX Refiner requires image height and width divisible by 16; resize the input first"
        )
    source_frames = int(images.shape[0])
    vae_frames = ((source_frames - 1 + 3) // 4) * 4 + 1
    if vae_frames != source_frames:
        images = torch.cat(
            [images, images[-1:].expand(vae_frames - source_frames, -1, -1, -1)], dim=0
        )

    output_pixels = int(images.shape[1] * scale) * int(images.shape[2] * scale)
    if output_pixels > MAX_REFINER_OUTPUT_PIXELS:
        raise ValueError(
            f"DreamX Refiner output would be {output_pixels / 1_000_000:.2f} MP; "
            f"the guarded limit is {MAX_REFINER_OUTPUT_PIXELS / 1_000_000:.2f} MP. "
            "Resize the input before 2x refinement."
        )

    # Explicit tiling is required for long/high-resolution video. Comfy's
    # regular VAE methods only auto-fallback on errors classified as OOM, while
    # cuDNN can report the same memory pressure as CUDNN_STATUS_EXECUTION_FAILED.
    with torch.inference_mode():
        lr_raw = vae.encode_tiled(
            images[..., :3],
            tile_x=VAE_TILE_SIZE,
            tile_y=VAE_TILE_SIZE,
            overlap=VAE_TILE_OVERLAP,
            tile_t=VAE_TEMPORAL_SIZE,
            overlap_t=VAE_TEMPORAL_OVERLAP,
        )
    expected_shape = (48, ((vae_frames - 1) // 4) + 1, images.shape[1] // 16, images.shape[2] // 16)
    if lr_raw.ndim != 5 or tuple(lr_raw.shape[1:]) != expected_shape:
        raise ValueError(
            "DreamX requires the Wan2.2 48-channel video VAE. "
            f"Expected latent [B,{expected_shape[0]},{expected_shape[1]},"
            f"{expected_shape[2]},{expected_shape[3]}], got {tuple(lr_raw.shape)}"
        )
    latent_format = comfy.latent_formats.Wan22()
    lr_latent = latent_format.process_in(lr_raw).permute(0, 2, 1, 3, 4)
    original_t = lr_latent.shape[1]
    chunk = int(refiner.pipeline.num_frame_per_block)
    if original_t % chunk:
        pad = chunk - original_t % chunk
        lr_latent = torch.cat(
            [lr_latent, lr_latent[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1
        )

    # The released SR-DiT also vendors regular torch modules, so generic partial
    # weight offload cannot safely leave individual layers on CPU while its
    # activations run on CUDA. The 5B bf16 refiner fits fully on the supported
    # 24 GB target after Comfy unloads the VAE.
    comfy.model_management.load_models_gpu(
        [refiner.patcher], force_full_load=True
    )
    device = refiner.patcher.load_device
    lr_latent = lr_latent.to(device=device, dtype=refiner.dtype)
    target_h = lr_latent.shape[-2] * int(scale)
    target_w = lr_latent.shape[-1] * int(scale)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        (lr_latent.shape[0], lr_latent.shape[1], lr_latent.shape[2], target_h, target_w),
        generator=generator,
        device=device,
        dtype=refiner.dtype,
    )
    prompt_embeds = positive[0][0].to(device=device, dtype=refiner.dtype)
    anchor_layers = None
    if int(anchor_keep_prefix) >= 0:
        anchor_layers = list(range(min(int(anchor_keep_prefix), 30)))

    def infer_once():
        pbar = comfy.utils.ProgressBar(lr_latent.shape[1] // chunk)

        def progress(done, total):
            comfy.model_management.throw_exception_if_processing_interrupted()
            pbar.update_absolute(done, total)

        try:
            with comfy.model_management.cuda_device_context(device), torch.inference_mode():
                return refiner.pipeline.inference_sr(
                    lr_latent=lr_latent,
                    noise=noise,
                    text_prompts=[""],
                    sigma_start=float(sigma_start),
                    return_latents=False,
                    return_video=False,
                    conditional_dict={"prompt_embeds": prompt_embeds},
                    use_lq_anchor=bool(use_lq_anchor),
                    lq_guidance_scale=(float(lq_guidance_scale),),
                    anchor_context_noise=int(anchor_context_noise),
                    anchor_layers=anchor_layers,
                    anchor_align="frame",
                    anchor_window_scope="window",
                    native_lq_anchor=False,
                    native_anchor_scheme="repeat",
                    progress_callback=progress,
                )
        finally:
            # The pipeline only clears this cache in its own pixel-decode path. We
            # return latents and use Comfy's VAE, so always release it here—even on
            # cancellation or an exception—before loading the VAE.
            refiner.pipeline.kv_caches = None
            comfy.model_management.soft_empty_cache()

    try:
        output = infer_once()
    except torch.OutOfMemoryError:
        if refiner.window_chunk <= 1:
            raise
        refiner.window_chunk = 1
        _set_window_chunk(refiner.pipeline, 1)
        comfy.model_management.soft_empty_cache(force=True)
        output = infer_once()
    output = output[:, :original_t].permute(0, 2, 1, 3, 4)
    raw_output = latent_format.process_out(output.float()).to(
        comfy.model_management.intermediate_device()
    )
    spatial = int(vae.spacial_compression_decode())
    temporal = vae.temporal_compression_decode()
    tile_t = None
    overlap_t = None
    if temporal is not None:
        temporal = int(temporal)
        tile_t = max(2, VAE_TEMPORAL_SIZE // temporal)
        overlap_t = max(1, min(tile_t // 2, VAE_TEMPORAL_OVERLAP // temporal))
    with torch.inference_mode():
        images_out = vae.decode_tiled(
            raw_output,
            tile_x=VAE_TILE_SIZE // spatial,
            tile_y=VAE_TILE_SIZE // spatial,
            overlap=VAE_TILE_OVERLAP // spatial,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )
    if images_out.ndim == 5:
        images_out = images_out.reshape(
            -1, images_out.shape[-3], images_out.shape[-2], images_out.shape[-1]
        )
    images_out = images_out[:source_frames]
    return images_out, {"samples": raw_output, "dreamx_num_frames": source_frames}

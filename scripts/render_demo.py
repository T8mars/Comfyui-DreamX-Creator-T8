"""Render a small real-model Creator + Refiner demo for release verification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_PATH", REPO_ROOT.parent / "ComfyUI"))
if not COMFY_ROOT.is_dir():
    raise RuntimeError("Set COMFYUI_PATH to a ComfyUI checkout before running this script")
sys.path[:0] = [str(COMFY_ROOT), str(REPO_ROOT)]

import comfy.model_management
import comfy.sample
import comfy.samplers
from comfy.nested_tensor import NestedTensor

from comfy_nodes.guider import DreamXMultimodalGuider
from comfy_nodes.loaders import (
    load_audio_vae,
    load_model_patcher,
    load_text_encoder,
    load_wan_vae,
)
from comfy_nodes.nodes import DreamXAVSamplingPatch, DreamXFirstFrameAVLatent
from comfy_nodes.refiner import load_refiner, run_refiner


CHECKPOINT_VERSION = 1
CONDITIONING_CACHE_VERSION = 1
VAE_TILE_SIZE = 512
VAE_TILE_OVERLAP = 64
VAE_TEMPORAL_SIZE = 64
VAE_TEMPORAL_OVERLAP = 8
DEFAULT_PROMPT = (
    "A cheerful hand-drawn yellow-eared cartoon character gently waves both arms, "
    "soft breeze moving the grass, small birds chirping, colorful clean animation"
)
DEFAULT_NEGATIVE_PROMPT = (
    "static, blurry, distorted, extra limbs, subtitles, watermark, low quality, silent"
)


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image).copy()).float().div_(255.0).unsqueeze(0)


def load_frame_sequence(directory: Path) -> torch.Tensor:
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No rendered frames found in {directory}")
    frames = [load_image(path) for path in paths]
    shape = tuple(frames[0].shape)
    if any(tuple(frame.shape) != shape for frame in frames[1:]):
        raise RuntimeError(f"Rendered frame dimensions are inconsistent in {directory}")
    return torch.cat(frames, dim=0)


def save_conditioning_cache(path: Path, prompt: str, positive) -> None:
    if not positive or not torch.is_tensor(positive[0][0]):
        raise ValueError("Positive conditioning is empty or invalid")
    payload = {
        "version": CONDITIONING_CACHE_VERSION,
        "prompt": prompt,
        "embedding": positive[0][0].detach().cpu(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_conditioning_cache(path: Path, prompt: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"Refiner conditioning cache is missing: {path}. "
            "Run once with --conditioning-only first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != CONDITIONING_CACHE_VERSION:
        raise RuntimeError(f"Unsupported conditioning cache version in {path}")
    if payload.get("prompt") != prompt:
        raise RuntimeError(f"Conditioning cache prompt does not match this run: {path}")
    embedding = payload.get("embedding")
    if not torch.is_tensor(embedding):
        raise RuntimeError(f"Invalid conditioning embedding in {path}")
    return [[embedding, {}]]


def save_frames(images: torch.Tensor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    images = images.detach().float().cpu().clamp(0.0, 1.0)
    for index, frame in enumerate(images):
        pixels = (frame.numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(pixels, "RGB").save(directory / f"frame_{index:05d}.png")


def save_wav(waveform: torch.Tensor, path: Path, sample_rate: int) -> None:
    samples = waveform.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0)
    pcm = (samples.numpy() * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def mux_video(frame_dir: Path, wav_path: Path, fps: float, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-i", str(frame_dir / "frame_%05d.png"),
            "-i", str(wav_path), "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-b:a", "192k", "-shortest", str(output),
        ],
        check=True,
    )


def release_models() -> None:
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(force=True)


def decode_video_tiled(vae, latent: torch.Tensor) -> torch.Tensor:
    """Mirror ComfyUI's native VAEDecodeTiled defaults for a video VAE."""
    spatial = int(vae.spacial_compression_decode())
    temporal = vae.temporal_compression_decode()
    tile_t = None
    overlap_t = None
    if temporal is not None:
        temporal = int(temporal)
        tile_t = max(2, VAE_TEMPORAL_SIZE // temporal)
        overlap_t = max(1, min(tile_t // 2, VAE_TEMPORAL_OVERLAP // temporal))
    with torch.inference_mode():
        return vae.decode_tiled(
            latent,
            tile_x=VAE_TILE_SIZE // spatial,
            tile_y=VAE_TILE_SIZE // spatial,
            overlap=VAE_TILE_OVERLAP // spatial,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )


def checkpoint_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def euler_step_state(x, denoised, sigma: float, sigma_next: float):
    """Return Euler's post-step state from ComfyUI's pre-step callback tensors."""
    if sigma == 0.0:
        return x
    return x + (x - denoised) * ((sigma_next - sigma) / sigma)


def save_creator_checkpoint(path: Path, fingerprint: str, completed_steps: int, state) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "completed_steps": int(completed_steps),
        "state": [stream.detach().cpu() for stream in state.unbind()],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_creator_checkpoint(path: Path, fingerprint: str, total_steps: int):
    if not path.is_file():
        return 0, None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported Creator checkpoint version in {path}")
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Creator checkpoint parameters do not match this run: {path}. "
            "Remove it or choose a different output directory."
        )
    completed_steps = int(payload.get("completed_steps", -1))
    if not 0 <= completed_steps <= total_steps:
        raise RuntimeError(f"Invalid completed step count in {path}: {completed_steps}")
    state = payload.get("state")
    if not isinstance(state, list) or len(state) < 2 or not all(torch.is_tensor(x) for x in state):
        raise RuntimeError(f"Invalid Creator latent state in {path}")
    return completed_steps, state


def creator_resume_noise(checkpoint_state, latent_image, model_sampling, base_model, sigma: float):
    """Invert CONST.noise_scaling so KSAMPLER reconstructs a saved internal state."""
    if sigma <= 0.0:
        raise ValueError("A zero-sigma Creator checkpoint is already complete")
    shapes = [tuple(stream.shape) for stream in latent_image.unbind()]
    base_model.latent_shapes = shapes
    latent_internal = base_model.process_latent_in(latent_image)
    scale = float(getattr(model_sampling, "noise_scale", 1.0))
    resumed = []
    for current, initial in zip(checkpoint_state, latent_internal.unbind(), strict=True):
        current = current.to(device=initial.device, dtype=torch.float32)
        initial = initial.to(dtype=torch.float32)
        resumed.append((current - (1.0 - sigma) * initial) / (sigma * scale))
    return NestedTensor(resumed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=COMFY_ROOT / "input" / "example.png")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "demo")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--resume", action="store_true",
        help="Atomically checkpoint each completed Creator Euler step and resume it after interruption.",
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--conditioning-only", action="store_true",
        help="Encode and cache the Refiner prompt in a clean process, then exit.",
    )
    stages.add_argument(
        "--refiner-only", action="store_true",
        help="Refine existing base frames using the cached prompt without loading Creator or T5.",
    )
    stages.add_argument(
        "--creator-only", action="store_true",
        help="Render and mux the Creator result, then exit before loading the Refiner.",
    )
    args = parser.parse_args()

    root = REPO_ROOT / "checkpoints"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    started = time.time()
    audio_path = output_dir / "dreamx_demo.wav"
    base_frames = output_dir / "base_frames"
    base_video = output_dir / "dreamx_creator_base.mp4"
    conditioning_path = output_dir / "refiner_conditioning.pt"

    if args.refiner_only:
        positive = load_conditioning_cache(conditioning_path, prompt)
        base_images = load_frame_sequence(base_frames)
        if not audio_path.is_file() or not base_video.is_file():
            raise FileNotFoundError(
                f"Refiner-only mode requires {audio_path.name} and {base_video.name} in {output_dir}"
            )
        vae = load_wan_vae(root)
        print(
            f"Refiner-only input: {len(base_images)} frames, "
            f"{base_images.shape[2]}x{base_images.shape[1]}, {args.fps:g} fps",
            flush=True,
        )
    else:
        clip = load_text_encoder(root, torch.bfloat16)
        positive = [[clip.encode(prompt), {}]]
        negative = [[clip.encode(negative_prompt), {}]]
        save_conditioning_cache(conditioning_path, prompt, positive)
        del clip
        release_models()
        print(f"Refiner conditioning cache: {conditioning_path}", flush=True)
        if args.conditioning_only:
            print(f"Completed in {time.time() - started:.1f}s", flush=True)
            return

        vae = load_wan_vae(root)
        image = load_image(args.input)
        conditioned = DreamXFirstFrameAVLatent.execute(
            positive, negative, vae, image, args.duration, args.fps, args.tokens, 1
        ).result
        positive, negative, latent, width, height, frames, _ = conditioned
        print(f"Creator target: {frames} frames, {width}x{height}, {args.fps:g} fps", flush=True)

        model = load_model_patcher(root, torch.bfloat16)
        model = DreamXAVSamplingPatch.execute(model, 5.0, 5.0).result[0]
        guider = DreamXMultimodalGuider(model)
        guider.set_conds(positive, negative)
        guider.set_scales(5.0, 3.5, 3.5, True, True)
        noise = comfy.sample.prepare_noise(latent["samples"], args.seed)
        # ComfyUI's ``normal`` schedule is the native equivalent of the released
        # Diffusers FlowMatchEulerDiscreteScheduler.set_timesteps() sequence.
        # ``simple`` subsamples the 1,000-point table and diverges sharply near
        # sigma=0, so it must not be used for DreamX inference.
        sigmas = comfy.samplers.normal_scheduler(
            model.get_model_object("model_sampling"), args.steps
        )
        checkpoint_path = output_dir / "creator_sampling_checkpoint.pt"
        fingerprint = checkpoint_fingerprint({
            "steps": args.steps,
            "seed": args.seed,
            "duration": args.duration,
            "fps": args.fps,
            "tokens": args.tokens,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "latent_shapes": [list(stream.shape) for stream in latent["samples"].unbind()],
            "sigmas": [float(sigma) for sigma in sigmas],
        })
        completed_steps, checkpoint_state = (
            load_creator_checkpoint(checkpoint_path, fingerprint, args.steps)
            if args.resume else (0, None)
        )
        if completed_steps:
            print(
                f"Resuming Creator after step {completed_steps}/{args.steps}: {checkpoint_path}",
                flush=True,
            )

        if completed_steps == args.steps:
            base_model = model.model
            base_model.latent_shapes = [tuple(stream.shape) for stream in latent["samples"].unbind()]
            internal = NestedTensor(checkpoint_state)
            sampled = base_model.process_latent_out(internal).to(torch.float32)
        else:
            active_sigmas = sigmas[completed_steps:]
            if checkpoint_state is not None:
                noise = creator_resume_noise(
                    checkpoint_state,
                    latent["samples"],
                    model.get_model_object("model_sampling"),
                    model.model,
                    float(active_sigmas[0]),
                )

            def checkpoint_callback(local_step, denoised, current, _total_steps):
                global_step = completed_steps + int(local_step)
                next_state = NestedTensor([
                    euler_step_state(
                        stream, clean, float(sigmas[global_step]), float(sigmas[global_step + 1])
                    )
                    for stream, clean in zip(current.unbind(), denoised.unbind(), strict=True)
                ])
                save_creator_checkpoint(
                    checkpoint_path, fingerprint, global_step + 1, next_state
                )
                print(
                    f"Creator checkpoint: {global_step + 1}/{args.steps} -> {checkpoint_path}",
                    flush=True,
                )

            sampled = guider.sample(
                noise,
                latent["samples"],
                comfy.samplers.sampler_object("euler"),
                active_sigmas,
                denoise_mask=latent["noise_mask"],
                callback=checkpoint_callback if args.resume else None,
                disable_pbar=False,
                seed=args.seed,
            )
        video_latent, audio_latent = sampled.unbind()[:2]
        del guider, model, sampled, noise
        release_models()

        base_images = decode_video_tiled(vae, video_latent)
        if base_images.ndim == 5:
            base_images = base_images.reshape(-1, *base_images.shape[-3:])
        base_images = base_images[:frames]
        audio_vae = load_audio_vae(root, torch.float32)
        waveform = audio_vae.decode(audio_latent)[..., : latent["dreamx_audio_samples"]]
        save_wav(waveform, audio_path, audio_vae.sample_rate)
        del audio_vae, waveform
        release_models()

        save_frames(base_images, base_frames)
        mux_video(base_frames, audio_path, args.fps, base_video)
        print(f"Base video: {base_video}", flush=True)

    if args.creator_only:
        print(f"Completed in {time.time() - started:.1f}s", flush=True)
        return

    refiner = load_refiner(
        root, torch.bfloat16, window_chunk=1, kv_history_frames=3
    )
    refined_images, _ = run_refiner(
        refiner, vae, positive, base_images, 2, args.seed, 0.6251,
        True, 1.0, 0, -1,
    )
    refined_frames = output_dir / "refined_frames"
    save_frames(refined_images, refined_frames)
    refined_video = output_dir / "dreamx_creator_refined_2x.mp4"
    mux_video(refined_frames, audio_path, args.fps, refined_video)
    print(f"Refined video: {refined_video}", flush=True)
    print(f"Completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

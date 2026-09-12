"""Render a small real-model Creator + Refiner demo for release verification."""

from __future__ import annotations

import argparse
import gc
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

from comfy_nodes.guider import DreamXMultimodalGuider
from comfy_nodes.loaders import (
    load_audio_vae,
    load_model_patcher,
    load_text_encoder,
    load_wan_vae,
)
from comfy_nodes.nodes import DreamXAVSamplingPatch, DreamXFirstFrameAVLatent
from comfy_nodes.refiner import load_refiner, run_refiner


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image).copy()).float().div_(255.0).unsqueeze(0)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=COMFY_ROOT / "input" / "example.png")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "demo")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()

    root = REPO_ROOT / "checkpoints"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "A cheerful hand-drawn yellow-eared cartoon character gently waves both arms, "
        "soft breeze moving the grass, small birds chirping, colorful clean animation"
    )
    negative_prompt = (
        "static, blurry, distorted, extra limbs, subtitles, watermark, low quality, silent"
    )
    started = time.time()

    clip = load_text_encoder(root, torch.bfloat16)
    positive = [[clip.encode(prompt), {}]]
    negative = [[clip.encode(negative_prompt), {}]]
    del clip
    release_models()

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
    sigmas = comfy.samplers.simple_scheduler(model.get_model_object("model_sampling"), args.steps)
    sampled = guider.sample(
        noise,
        latent["samples"],
        comfy.samplers.sampler_object("euler"),
        sigmas,
        denoise_mask=latent["noise_mask"],
        disable_pbar=False,
        seed=args.seed,
    )
    video_latent, audio_latent = sampled.unbind()[:2]
    del guider, model, sampled, noise
    release_models()

    base_images = vae.decode(video_latent)
    if base_images.ndim == 5:
        base_images = base_images.reshape(-1, *base_images.shape[-3:])
    base_images = base_images[:frames]
    audio_vae = load_audio_vae(root, torch.bfloat16)
    waveform = audio_vae.decode(audio_latent)[..., : latent["dreamx_audio_samples"]]
    audio_path = output_dir / "dreamx_demo.wav"
    save_wav(waveform, audio_path, audio_vae.sample_rate)
    del audio_vae, waveform
    release_models()

    base_frames = output_dir / "base_frames"
    save_frames(base_images, base_frames)
    base_video = output_dir / "dreamx_creator_base.mp4"
    mux_video(base_frames, audio_path, args.fps, base_video)
    print(f"Base video: {base_video}", flush=True)

    refiner = load_refiner(root, torch.bfloat16, window_chunk=4)
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

# DreamX-Creator native ComfyUI nodes

This checkout is a ComfyUI V3 custom-node package. It exposes the released DreamX
joint audio/video generator and causal 2x refiner through ComfyUI's native `MODEL`,
`CLIP`, `VAE`, `LATENT`, `AUDIO`, `GUIDER`, and `VIDEO` pipeline types.

## Install

Place or link this repository directly below `ComfyUI/custom_nodes/`, then install
the node-only dependencies with the same Python that starts ComfyUI:

```powershell
python -m pip install -r requirements.txt
```

The requirements intentionally do not install or replace PyTorch. Restart ComfyUI
after installation. The loader's `model_root=auto` searches, in order, the
repository's `checkpoints/` and `ComfyUI/models/dreamx_creator/`.

Validate a downloaded model tree before loading 50+ GB of weights:

```powershell
python scripts\verify_models.py
```

## Native generation graph

Use these nodes in order:

1. `DreamX Creator Complete Loader` (or the four individual loaders).
2. Two native `CLIP Text Encode` nodes for positive and negative prompts.
3. `DreamX First Frame AV Latent` with a required start image.
4. `DreamX AV Flow Shifts` (the released model uses 5.0 / 5.0).
5. `DreamX Multimodal Guider`.
6. Native `RandomNoise`, `BasicScheduler`, `KSamplerSelect`, and
   `SamplerCustomAdvanced`.
7. `DreamX Split AV Latent`, native `VAE Decode`, and
   `DreamX Audio VAE Decode`.
8. Native `Create Video` and `Save Video`.

The guider evaluates three branches: no-text/no-bridge, no-text/bridge, and
text/bridge. `text_cfg`, `video_bridge`, and `audio_bridge` are therefore genuinely
independent. `enable_a2v` controls audio-to-video cross attention and `enable_v2a`
controls video-to-audio cross attention. Independent video/audio flow shifts are
mapped onto ComfyUI's single packed-latent schedule; the released preset is 5.0/5.0.

Drag `examples/dreamx_creator_ui.json` directly onto the ComfyUI canvas (or use
**Workflows → Open**) and select a start image in `Load Image`.

## Causal 2x refiner graph

`DreamX Causal Refiner Loader` loads the released SR-DiT 5B and Flash latent
upsampler. Feed a frame sequence, a native Wan2.2 VAE, and UMT5 conditioning into
`DreamX Causal Refine Video`. The node performs 2x spatial refinement, exposes the
released sigma/anchor controls, reports block progress, and honors ComfyUI's cancel
signal between causal blocks. Recombine the returned frames with the original audio
through native `Create Video`. Inputs that are not `4N+1` frames are padded before
Wan VAE encoding and cropped after decoding, so the returned video keeps the exact
source frame count and remains aligned to the original audio.

Drag `examples/dreamx_refiner_ui.json` directly onto the ComfyUI canvas. This is a
separate second-phase workflow so the 7B Creator and 5B Refiner do not stay resident
at the same time. Select the Creator MP4 in `Load Video`; its audio and FPS are
preserved automatically.

## Memory notes

The released generator is about 7B parameters and the refiner is about 5B. The
loaders create ComfyUI model patchers on CPU and let ComfyUI move/offload weights.
Use bfloat16 on modern NVIDIA GPUs. Generator and refiner are intended to run in
separate graph phases; ComfyUI can unload one before loading the next. At the
released 5-second / ~880-token preset, activation memory can still be substantial.
The refiner defaults to `window_chunk=4` for the Windows SDPA fallback, retries once
with `window_chunk=1` after a CUDA OOM, clears streaming KV caches before VAE decode,
and rejects outputs above 8 MiPixels. Lower the loader's window chunk to reduce peak
VRAM. Pickle-backed official checkpoints are SHA-256 verified against
`model_manifest.json` before safe tensor-only loading.

## Tests

```powershell
$env:COMFYUI_PATH = "C:\path\to\ComfyUI"
python -m pytest -q tests
```

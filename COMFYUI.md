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

Download the complete ComfyUI-ready model bundle from
[t8star/DreamX-Creator-Comfy](https://huggingface.co/t8star/DreamX-Creator-Comfy)
directly into the recommended shared directory:

```powershell
hf download t8star/DreamX-Creator-Comfy --local-dir ComfyUI/models/dreamx_creator
```

The bundle is approximately 54.25 GB (50.53 GiB). Its root must directly contain
`creator/`, `audio_vae/`, `refiner/`, and `wan2.2_ti2v_5b/`; do not add another
nested `DreamX-Creator-Comfy/` directory between the selected model root and these
four folders.

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
   `SamplerCustomAdvanced`. Keep `BasicScheduler` on **normal**, 20 steps,
   and `KSamplerSelect` on **euler** for the released FlowMatch pipeline.
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

The shipped quick-start profile renders directly from Creator at 256 spatial
tokens; a square input therefore produces 512×512 output. It uses 2 seconds,
24 FPS, 20 steps, and a fixed verified seed, producing 45 frames (1.875 s) after
the native `4N+1` frame snap. Inspect the decoded Creator video before optionally
sending it to the separate Refiner workflow. Do not rely on
2× refinement to repair an already overexposed, structurally collapsed, or
flickering 256×256 base sample—increase Creator spatial tokens or change the seed
first.

The included prompt asks the native joint model to say “你在干嘛”, but native
speech generation is stochastic and does not guarantee an exact transcript.
The separately verified exact-word sample used operating-system TTS post-dubbing;
that post-processing is intentionally not represented as native DreamX audio in
this UI workflow.

DreamX emits one synchronized audio/video pair per execution, so `batch_size` is
intentionally fixed at 1. Use ComfyUI's queue batch count to generate multiple
seed variants. Requested duration is truncated down to a Wan-compatible `4N+1`
frame count; for example, 5 seconds at 24 FPS becomes 117 frames (4.875 s),
matching the released pipeline's floor-to-`4n+1` rule. The Audio VAE defaults to
float32, as in the released decoder.
The graph deliberately uses native `VAEDecodeTiled` with 512 px / 64-frame tiles:
on long video, cuDNN can report memory pressure as an execution failure rather
than a standard OOM, which bypasses the ordinary VAE node's automatic fallback.

## Causal 2x refiner graph

`DreamX Causal Refiner Loader` loads the released SR-DiT 5B and Flash latent
upsampler. Feed a frame sequence, a native Wan2.2 VAE, and UMT5 conditioning into
`DreamX Causal Refine Video`. The node performs 2x spatial refinement, exposes the
released sigma/anchor controls, reports block progress, and honors ComfyUI's cancel
signal between causal blocks. Recombine the returned frames with the original audio
through native `Create Video`. Inputs that are not `4N+1` frames are padded before
Wan VAE encoding and cropped after decoding, so the returned video keeps the exact
source frame count and remains aligned to the original audio. The public node returns
only the exact-length IMAGE sequence; its internally padded latent is not exposed as
a generic LATENT that native `VAE Decode` could accidentally decode to extra frames.
Wan VAE encoding and final decoding are tiled internally with the same conservative
defaults used by ComfyUI's native tiled VAE nodes.

Drag `examples/dreamx_refiner_ui.json` directly onto the ComfyUI canvas. This is a
separate second-phase workflow so the 7B Creator and 5B Refiner do not stay resident
at the same time. Select the Creator MP4 in `Load Video`; its audio and FPS are
preserved automatically.

## Memory notes

The released generator is about 7B parameters and the refiner is about 5B. The
loaders create ComfyUI model patchers on CPU; before Creator or Refiner sampling,
each model's regular torch layers are loaded together on the GPU because generic
partial-weight offload can otherwise mix CPU weights with CUDA activations. Use
bfloat16 on modern NVIDIA GPUs. Generator and refiner are intended to run in
separate graph phases; ComfyUI can unload one before loading the next. At the
released 5-second / ~880-token preset, activation memory can still be substantial
and a 24 GB GPU runs near its VRAM limit.
The refiner defaults to `window_chunk=1` and `kv_history_frames=3` for 24 GB Windows
systems, where WDDM can spill CUDA allocations into shared RAM instead of raising a
catchable OOM. The model's higher-context preset is 9 latent history frames; select it
only with measured VRAM and system commit headroom. The node clears unused CUDA
workspaces during causal sampling, clears streaming KV caches before VAE decode, and
rejects outputs above 8 MiPixels. Pickle-backed official checkpoints are SHA-256
verified against
`model_manifest.json` before safe tensor-only loading.

## Tests

```powershell
$env:COMFYUI_PATH = "C:\path\to\ComfyUI"
python -m pytest -q tests
```

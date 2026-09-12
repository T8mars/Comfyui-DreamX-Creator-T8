---
license: apache-2.0
pipeline_tag: image-to-video
tags:
  - comfyui
  - video
  - audio
  - image-to-video
  - video-upscaling
  - dreamx-creator
---

<div align="center">
  <h1>DreamX Creator Comfy</h1>
  <p>ComfyUI-ready DreamX-Creator generator, Audio VAE, Wan dependencies, and causal 2× refiner weights.</p>
</div>

## Links

- **ComfyUI nodes and frontend workflows:** [T8mars/Comfyui-DreamX-Creator-T8](https://github.com/T8mars/Comfyui-DreamX-Creator-T8)
- **ComfyUI Registry:** [dreamx-creator-t8](https://registry.comfy.org/nodes/dreamx-creator-t8)
- **Original project and model source:** [AMAP-ML/DreamX-Creator](https://github.com/AMAP-ML/DreamX-Creator) · [GD-ML/DreamX-Creator](https://huggingface.co/GD-ML/DreamX-Creator)

## Install for ComfyUI

Install the `hf` command if needed:

```bash
python -m pip install -U huggingface_hub
```

Download this repository directly into ComfyUI's shared model directory:

```bash
hf download t8star/DreamX-Creator-Comfy --local-dir ComfyUI/models/dreamx_creator
```

The resulting model root must be:

```text
ComfyUI/
└── models/
    └── dreamx_creator/          # select model_root=auto in the loader
        ├── creator/
        │   ├── cross_attn_weights.safetensors
        │   ├── merged_lora_info.json
        │   ├── audio_model/
        │   │   ├── config.json
        │   │   └── diffusion_pytorch_model.safetensors
        │   └── video_model/
        │       ├── config.json
        │       ├── diffusion_pytorch_model-00001-of-00002.safetensors
        │       ├── diffusion_pytorch_model-00002-of-00002.safetensors
        │       └── diffusion_pytorch_model.safetensors.index.json
        ├── audio_vae/
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        ├── refiner/
        │   ├── sr_dit_5b.pt
        │   ├── latent_upsampler_flash.pt
        │   ├── latent_upsampler_2d_causal.pt
        │   └── lightvae_nu_scheme3.pt
        └── wan2.2_ti2v_5b/
            ├── Wan2.2_VAE.pth
            ├── models_t5_umt5-xxl-enc-bf16.pth
            └── google/
                └── umt5-xxl/
                    ├── special_tokens_map.json
                    ├── spiece.model
                    ├── tokenizer_config.json
                    └── tokenizer.json
```

`model_root=auto` searches the custom-node repository's `checkpoints/` first,
then `ComfyUI/models/dreamx_creator/`. Therefore this node-local layout also works:

```bash
hf download t8star/DreamX-Creator-Comfy --local-dir ComfyUI/custom_nodes/Comfyui-DreamX-Creator-T8/checkpoints
```

The complete bundle contains 20 model/config/tokenizer files and is approximately
54.25 GB (50.53 GiB). Validate the download before loading the models:

```bash
python ComfyUI/custom_nodes/Comfyui-DreamX-Creator-T8/scripts/verify_models.py ComfyUI/models/dreamx_creator
```

## Components

| Directory | Used by | Contents |
| --- | --- | --- |
| `creator/` | Creator workflow | 7B joint video/audio DiTs and cross-modal attention |
| `audio_vae/` | Creator workflow | CreatorDACVAE audio decoder |
| `refiner/` | Refiner workflow | SR-DiT 5B and latent upsamplers |
| `wan2.2_ti2v_5b/` | Both workflows | Wan VAE, UMT5-XXL encoder, and tokenizer |

The Creator and Refiner are intended to run as separate ComfyUI workflow phases
so both large models do not remain resident on the GPU at the same time.

## T8star social links

| Resource | Link |
| --- | --- |
| Bilibili | [T8star on Bilibili](https://space.bilibili.com/385085361) |
| YouTube | [@T8star-Aix](https://www.youtube.com/@T8star-Aix/) |
| Seedance API | [API signup](https://api.seedance.nz/sign-up?aff=5f4w) |
| Online AI Apps | [RunningHub profile](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) |
| ComfyUI Package | [Quark download](https://pan.quark.cn/s/264edb7e36bd) |
| Hugging Face profile | [huggingface.co/t8star](https://huggingface.co/t8star) |

## Attribution and license

These files are organized for the native ComfyUI nodes from the weights released
by the DreamX Team at [GD-ML/DreamX-Creator](https://huggingface.co/GD-ML/DreamX-Creator).
The project is licensed under Apache License 2.0. See the `LICENSE` file in this
repository. Please retain the original attribution when redistributing the weights.

```bibtex
@misc{zhu2026dreamxcreatordemocratizingnativeaudiovideo,
  title={DreamX-Creator: Democratizing Native Audio-Video Generation at 2K Resolution},
  author={Jiashu Zhu and Yanhao Zheng and Ruitian Tian and Rujing Dang and Shen Zhang and Bingze Song and Jiachen Lei and Ruimin Lin and Jiahong Wu and Xiangxiang Chu},
  year={2026},
  eprint={2608.31106},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.31106}
}
```

"""Shared path, shape, and latent helpers for the ComfyUI integration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = REPO_ROOT / "checkpoints"
MODEL_MANIFEST = REPO_ROOT / "model_manifest.json"
_VERIFIED_MODEL_FILES: set[tuple[str, int, int]] = set()
_MANIFEST_ENTRIES: dict[str, dict] | None = None

REQUIRED_MODEL_FILES = (
    "creator/video_model/config.json",
    "creator/video_model/diffusion_pytorch_model.safetensors.index.json",
    "creator/video_model/diffusion_pytorch_model-00001-of-00002.safetensors",
    "creator/video_model/diffusion_pytorch_model-00002-of-00002.safetensors",
    "creator/audio_model/config.json",
    "creator/audio_model/diffusion_pytorch_model.safetensors",
    "creator/cross_attn_weights.safetensors",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "wan2.2_ti2v_5b/Wan2.2_VAE.pth",
    "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth",
    "wan2.2_ti2v_5b/google/umt5-xxl/tokenizer.json",
    "refiner/sr_dit_5b.pt",
    "refiner/latent_upsampler_flash.pt",
)


def resolve_model_root(model_root: str | Path | None = None) -> Path:
    """Resolve an explicit, repo-local, or ComfyUI models/dreamx_creator root."""
    candidates: list[Path] = []
    if model_root and str(model_root).strip().lower() not in {"auto", "default"}:
        candidates.append(Path(model_root).expanduser())
    candidates.append(DEFAULT_MODEL_ROOT)
    try:
        import folder_paths

        candidates.extend(Path(p) for p in folder_paths.get_folder_paths("dreamx_creator"))
        candidates.append(Path(folder_paths.models_dir) / "dreamx_creator")
    except Exception:
        pass

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "creator" / "video_model" / "config.json").is_file():
            return candidate
    checked = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "DreamX-Creator model root was not found. Expected a directory containing "
        f"creator/video_model/config.json. Checked:\n  - {checked}"
    )


def validate_model_files(root: Path, relative_paths: Iterable[str] = REQUIRED_MODEL_FILES) -> None:
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DreamX-Creator model root is incomplete: {root}\nMissing:\n  - "
            + "\n  - ".join(missing)
        )


def _manifest_entries() -> dict[str, dict]:
    global _MANIFEST_ENTRIES
    if _MANIFEST_ENTRIES is None:
        if not MODEL_MANIFEST.is_file():
            raise FileNotFoundError(f"DreamX model hash manifest is missing: {MODEL_MANIFEST}")
        data = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
        if data.get("algorithm") != "sha256":
            raise ValueError(f"Unsupported DreamX model manifest: {MODEL_MANIFEST}")
        _MANIFEST_ENTRIES = {
            entry["path"].replace("\\", "/"): entry for entry in data["files"]
        }
    return _MANIFEST_ENTRIES


def verify_trusted_model_file(root: Path, relative_path: str) -> Path:
    """Verify an official pickle checkpoint before any ``torch.load`` call."""
    root = root.resolve()
    relative_path = relative_path.replace("\\", "/")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Model path escapes the selected DreamX root: {path}") from exc

    entry = _manifest_entries().get(relative_path)
    if entry is None:
        raise ValueError(f"No trusted hash is recorded for DreamX checkpoint: {relative_path}")
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    if cache_key in _VERIFIED_MODEL_FILES:
        return path
    if stat.st_size != int(entry["size"]):
        raise ValueError(
            f"DreamX checkpoint size mismatch for {path}: expected {entry['size']}, got {stat.st_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(
            f"DreamX checkpoint SHA-256 mismatch for {path}. Refusing unsafe pickle load."
        )
    _VERIFIED_MODEL_FILES.add(cache_key)
    return path


def compute_dynamic_resolution(
    source_height: int,
    source_width: int,
    target_spatial_tokens: int = 880,
    min_token_ratio: float = 0.95,
    spatial_divisor: int = 32,
) -> tuple[int, int, int]:
    """Choose a multiple-of-32 resolution near the source aspect ratio."""
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Invalid source size: {(source_height, source_width)}")
    if target_spatial_tokens <= 0:
        raise ValueError("target_spatial_tokens must be positive")
    if not 0.0 < min_token_ratio <= 1.0:
        raise ValueError("min_token_ratio must be in (0, 1]")

    min_tokens = max(1, math.ceil(target_spatial_tokens * min_token_ratio))
    source_ratio = source_height / source_width
    best = None
    for token_h in range(1, target_spatial_tokens + 1):
        max_token_w = target_spatial_tokens // token_h
        ideal_w = source_width * token_h / source_height
        for token_w in {1, max_token_w, math.floor(ideal_w), math.ceil(ideal_w)}:
            if not 1 <= token_w <= max_token_w:
                continue
            used = token_h * token_w
            height, width = token_h * spatial_divisor, token_w * spatial_divisor
            score = (
                max(0, min_tokens - used),
                abs(math.log((height / width) / source_ratio)),
                target_spatial_tokens - used,
            )
            if best is None or score < best[0]:
                best = (score, height, width, used)
    if best is None:
        raise RuntimeError("Unable to resolve a dynamic DreamX resolution")
    return best[1], best[2], best[3]


def snap_video_frames(duration: float, fps: float, temporal_stride: int = 4) -> int:
    requested = max(1.0, float(duration) * float(fps))
    latent_intervals = math.floor((requested - 1.0) / temporal_stride + 0.5)
    return max(1, latent_intervals * temporal_stride + 1)


def tensor_streams(value):
    """Return the tensors from a NestedTensor or a two-item sequence."""
    if getattr(value, "is_nested", False):
        return list(value.unbind())
    if isinstance(value, (tuple, list)):
        return list(value)
    raise TypeError("Expected a packed DreamX audio/video NestedTensor")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16

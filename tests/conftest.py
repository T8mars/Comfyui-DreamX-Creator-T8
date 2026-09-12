"""Test bootstrap for running the custom node suite outside ComfyUI."""

import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
configured_comfy = os.environ.get("COMFYUI_PATH")
COMFY = Path(configured_comfy) if configured_comfy else REPO.parent / "ComfyUI"
if not COMFY.is_dir():
    raise RuntimeError(
        "Set COMFYUI_PATH to a ComfyUI checkout before running tests "
        f"(looked for {COMFY})"
    )
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(REPO))

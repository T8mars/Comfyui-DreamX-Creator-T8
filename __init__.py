"""ComfyUI entry point for the DreamX-Creator native node package."""

if __package__:
    from .comfy_nodes import comfy_entrypoint
else:  # Allows direct execution of a checkout whose folder contains '-'.
    from comfy_nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]

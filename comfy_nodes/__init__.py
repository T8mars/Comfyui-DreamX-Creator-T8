"""ComfyUI V3 extension registration."""

from pathlib import Path

from comfy_api.latest import ComfyExtension
from typing_extensions import override


class DreamXCreatorExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        import folder_paths

        repo_models = Path(__file__).resolve().parents[1] / "checkpoints"
        comfy_models = Path(folder_paths.models_dir) / "dreamx_creator"
        folder_paths.add_model_folder_path("dreamx_creator", str(repo_models), is_default=True)
        folder_paths.add_model_folder_path("dreamx_creator", str(comfy_models))

    @override
    async def get_node_list(self):
        from .nodes import NODE_LIST

        return NODE_LIST


async def comfy_entrypoint() -> DreamXCreatorExtension:
    return DreamXCreatorExtension()


__all__ = ["comfy_entrypoint", "DreamXCreatorExtension"]

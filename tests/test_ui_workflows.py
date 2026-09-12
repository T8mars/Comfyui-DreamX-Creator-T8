"""Static checks for the frontend-importable ComfyUI workflows."""

import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = [
    REPO / "examples" / "dreamx_creator_ui.json",
    REPO / "examples" / "dreamx_refiner_ui.json",
]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.stem)
def test_ui_workflow_link_integrity(path):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert workflow["version"] == pytest.approx(0.4)
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    assert len(links) == len(workflow["links"])

    for link_id, (_, source_id, source_slot, target_id, target_slot, link_type) in links.items():
        source = nodes[source_id]
        target = nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert source["outputs"][source_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] == link_type


def test_only_frontend_ui_workflows_are_shipped():
    shipped = {path.name for path in (REPO / "examples").glob("*.json")}
    assert shipped == {path.name for path in WORKFLOWS}


def test_creator_wires_runtime_fps_into_video_container():
    workflow = json.loads(WORKFLOWS[0].read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    first_frame = next(node for node in nodes.values() if node["type"] == "DreamXFirstFrameAVLatent")
    create_video = next(node for node in nodes.values() if node["type"] == "CreateVideo")
    fps_link = first_frame["outputs"][6]["links"][0]
    assert create_video["inputs"][2]["link"] == fps_link


def test_creator_uses_explicit_tiled_video_decode():
    workflow = json.loads(WORKFLOWS[0].read_text(encoding="utf-8"))
    decode = next(node for node in workflow["nodes"] if node["id"] == 13)
    assert decode["type"] == "VAEDecodeTiled"
    assert decode["widgets_values_named"] == {
        "tile_size": 512,
        "overlap": 64,
        "temporal_size": 64,
        "temporal_overlap": 8,
    }


def test_creator_workflow_matches_released_flow_schedule():
    workflow = json.loads(WORKFLOWS[0].read_text(encoding="utf-8"))
    scheduler = next(node for node in workflow["nodes"] if node["type"] == "BasicScheduler")
    sampler = next(node for node in workflow["nodes"] if node["type"] == "KSamplerSelect")

    assert scheduler["widgets_values"] == ["normal", 20, 1.0]
    assert scheduler["widgets_values_named"] == {
        "scheduler": "normal",
        "steps": 20,
        "denoise": 1.0,
    }
    assert sampler["widgets_values"] == ["euler"]


def test_creator_workflow_defaults_to_verified_direct_512_profile():
    workflow = json.loads(WORKFLOWS[0].read_text(encoding="utf-8"))
    positive = next(
        node
        for node in workflow["nodes"]
        if node["id"] == 3 and node["type"] == "CLIPTextEncode"
    )
    negative = next(
        node
        for node in workflow["nodes"]
        if node["id"] == 4 and node["type"] == "CLIPTextEncode"
    )
    first_frame = next(
        node for node in workflow["nodes"] if node["type"] == "DreamXFirstFrameAVLatent"
    )
    noise = next(node for node in workflow["nodes"] if node["type"] == "RandomNoise")
    save = next(node for node in workflow["nodes"] if node["type"] == "SaveVideo")

    assert first_frame["widgets_values"] == [2.0, 24.0, 256, 1]
    assert first_frame["widgets_values_named"] == {
        "duration": 2.0,
        "fps": 24.0,
        "target_spatial_tokens": 256,
        "batch_size": 1,
    }
    assert noise["widgets_values"] == [20260914, "fixed"]
    assert noise["widgets_values_named"] == {"noise_seed": 20260914}
    assert "你在干嘛" in positive["widgets_values"][0]
    assert "保持原画" in positive["widgets_values"][0]
    assert "过曝" in negative["widgets_values"][0]
    assert "颜色漂移" in negative["widgets_values"][0]
    assert save["widgets_values_named"]["filename_prefix"] == (
        "video/DreamX-Creator-Direct"
    )


def test_refiner_exposes_only_frame_exact_image_output():
    workflow = json.loads(WORKFLOWS[1].read_text(encoding="utf-8"))
    loader = next(node for node in workflow["nodes"] if node["type"] == "DreamXRefinerLoader")
    refiner = next(node for node in workflow["nodes"] if node["type"] == "DreamXCausalRefine")
    assert loader["widgets_values_named"]["window_chunk"] == 1
    assert loader["widgets_values_named"]["kv_history_frames"] == 3
    assert [(output["name"], output["type"]) for output in refiner["outputs"]] == [
        ("images", "IMAGE")
    ]

import importlib.util
from pathlib import Path

import torch
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_demo.py"
SPEC = importlib.util.spec_from_file_location("dreamx_render_demo", SCRIPT)
render_demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_demo)


def test_euler_checkpoint_is_the_post_step_state():
    current = torch.tensor([3.0, -2.0])
    denoised = torch.tensor([1.0, 2.0])
    result = render_demo.euler_step_state(current, denoised, 0.8, 0.5)
    expected = current + ((current - denoised) / 0.8) * (0.5 - 0.8)
    torch.testing.assert_close(result, expected)


def test_checkpoint_round_trip_and_parameter_guard(tmp_path):
    from comfy.nested_tensor import NestedTensor

    path = tmp_path / "creator.pt"
    state = NestedTensor([torch.randn(1, 2, 3), torch.randn(1, 4, 5)])
    render_demo.save_creator_checkpoint(path, "same", 7, state)
    completed, loaded = render_demo.load_creator_checkpoint(path, "same", 20)
    assert completed == 7
    for expected, actual in zip(state.unbind(), loaded, strict=True):
        torch.testing.assert_close(expected, actual)

    try:
        render_demo.load_creator_checkpoint(path, "different", 20)
    except RuntimeError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("mismatched resume parameters must be rejected")


def test_demo_decode_uses_native_tiled_video_defaults():
    class FakeVAE:
        def spacial_compression_decode(self):
            return 16

        def temporal_compression_decode(self):
            return 4

        def decode_tiled(self, latent, **kwargs):
            self.call = (latent, kwargs, torch.is_inference_mode_enabled())
            return "decoded"

    vae = FakeVAE()
    latent = torch.zeros(1, 48, 31, 58, 58)
    assert render_demo.decode_video_tiled(vae, latent) == "decoded"
    called_latent, kwargs, inference_enabled = vae.call
    assert called_latent is latent
    assert inference_enabled is True
    assert kwargs == {
        "tile_x": 32, "tile_y": 32, "overlap": 4,
        "tile_t": 16, "overlap_t": 2,
    }


def test_conditioning_cache_round_trip_and_prompt_guard(tmp_path):
    path = tmp_path / "conditioning.pt"
    expected = torch.randn(1, 8, 16)
    render_demo.save_conditioning_cache(path, "same", [[expected, {}]])
    loaded = render_demo.load_conditioning_cache(path, "same")
    torch.testing.assert_close(loaded[0][0], expected)

    try:
        render_demo.load_conditioning_cache(path, "different")
    except RuntimeError as error:
        assert "prompt does not match" in str(error)
    else:
        raise AssertionError("mismatched conditioning prompt must be rejected")


def test_load_frame_sequence_is_sorted_and_stacked(tmp_path):
    Image.new("RGB", (3, 2), (20, 40, 60)).save(tmp_path / "frame_00001.png")
    Image.new("RGB", (3, 2), (10, 30, 50)).save(tmp_path / "frame_00000.png")
    frames = render_demo.load_frame_sequence(tmp_path)
    assert tuple(frames.shape) == (2, 2, 3, 3)
    torch.testing.assert_close(
        frames[:, 0, 0],
        torch.tensor([[10, 30, 50], [20, 40, 60]], dtype=torch.float32) / 255.0,
    )
